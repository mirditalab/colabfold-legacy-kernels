// Gated dual projection for Volta (sm_70+), with nvcuda::wmma.
// It calculates mask * (x@wp + bp) * act(x@wg + bg) in one kernel.

#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <string>
#include "xla/ffi/api/ffi.h"

#include "volta_gdp.h"

namespace ffi = xla::ffi;
using namespace nvcuda;

#define FRAG 16

template <int BM, int BN, int BK, GdpAct A>
__global__ __launch_bounds__(BM / FRAG * WARP, 2) void volta_gdp_wmma_kernel(
    const __half* __restrict__ x, const __half* __restrict__ wp, const __half* __restrict__ bp,
    const __half* __restrict__ wg, const __half* __restrict__ bg, const __half* __restrict__ mask,
    __half* __restrict__ out, int M, int K, int N) {
    constexpr int NWARP = BM / FRAG;
    constexpr int NN = BN / FRAG;
    constexpr int TPB = NWARP * WARP;
    constexpr int XS_LD = BK + 8;
    constexpr int W_LD = BN + 8;

    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;

    extern __shared__ char smem[];
    __half* Xs = reinterpret_cast<__half*>(smem);         // [BM][XS_LD]
    __half* Wp = Xs + BM * XS_LD;                         // [BK][W_LD]
    __half* Wg = Wp + BK * W_LD;                          // [BK][W_LD]
    float* Ep = reinterpret_cast<float*>(Wg + BK * W_LD); // [NWARP][2][16][16]

    wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> accp[NN], accg[NN];
#pragma unroll
    for (int i = 0; i < NN; ++i) {
        wmma::fill_fragment(accp[i], 0.0f);
        wmma::fill_fragment(accg[i], 0.0f);
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        __syncthreads();
        for (int i = tid; i < BM * BK; i += TPB) {
            int r = i / BK, c = i - r * BK;
            int gm = m0 + r, gk = k0 + c;
            Xs[r * XS_LD + c] = (gm < M && gk < K) ? x[(long long)gm * K + gk] : __float2half(0.f);
        }
        for (int i = tid; i < BK * BN; i += TPB) {
            int r = i / BN, c = i - r * BN;
            int gk = k0 + r, gn = n0 + c;
            bool ok = (gk < K && gn < N);
            Wp[r * W_LD + c] = ok ? wp[(long long)gk * N + gn] : __float2half(0.f);
            Wg[r * W_LD + c] = ok ? wg[(long long)gk * N + gn] : __float2half(0.f);
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < BK / FRAG; ++ks) {
            wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, Xs + (warp * FRAG) * XS_LD + ks * FRAG, XS_LD);
#pragma unroll
            for (int nt = 0; nt < NN; ++nt) {
                wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, __half, wmma::row_major> fbp, fbg;
                wmma::load_matrix_sync(fbp, Wp + (ks * FRAG) * W_LD + nt * FRAG, W_LD);
                wmma::load_matrix_sync(fbg, Wg + (ks * FRAG) * W_LD + nt * FRAG, W_LD);
                wmma::mma_sync(accp[nt], fa, fbp, accp[nt]);
                wmma::mma_sync(accg[nt], fa, fbg, accg[nt]);
            }
        }
    }

    // fused epilogue, one 16x16 tile at a time
    float* ep = Ep + warp * 2 * FRAG * FRAG;
    float* eg = ep + FRAG * FRAG;
#pragma unroll
    for (int nt = 0; nt < NN; ++nt) {
        wmma::store_matrix_sync(ep, accp[nt], FRAG, wmma::mem_row_major);
        wmma::store_matrix_sync(eg, accg[nt], FRAG, wmma::mem_row_major);
        __syncwarp();
        // two adjacent columns per lane, stored as one half2
        for (int i = 2 * lane; i < FRAG * FRAG; i += 2 * WARP) {
            const int r = i / FRAG, c = i - r * FRAG;
            const int gm = m0 + warp * FRAG + r, gn = n0 + nt * FRAG + c;
            if (gm >= M || gn >= N) {
                continue;
            }
            const float mv = __half2float(mask[gm]);
            const bool pair = gn + 1 < N;
            const float o0 = mv * (ep[i] + __half2float(bp[gn])) * gdp_act<A>(eg[i] + __half2float(bg[gn]));
            const float o1 = pair ? mv * (ep[i + 1] + __half2float(bp[gn + 1])) *
                                        gdp_act<A>(eg[i + 1] + __half2float(bg[gn + 1]))
                                  : 0.f;
            __half* o = out + (long long)gm * N + gn;
            if (pair && (N & 1) == 0) {
                *reinterpret_cast<__half2*>(o) = __floats2half2_rn(o0, o1);
            } else {
                o[0] = __float2half(o0);
                if (pair) {
                    o[1] = __float2half(o1);
                }
            }
        }
        __syncwarp();
    }
}

ffi::Error GdpWmmaImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F16> x,
                       ffi::Buffer<ffi::DataType::F16> wp, ffi::Buffer<ffi::DataType::F16> bp,
                       ffi::Buffer<ffi::DataType::F16> wg, ffi::Buffer<ffi::DataType::F16> bg,
                       ffi::Buffer<ffi::DataType::F16> mask,
                       ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, int64_t activation) {
    auto dx = x.dimensions();
    if (dx.size() != 2) {
        return ffi::Error::InvalidArgument("x must be [M,K]");
    }
    const int M = (int)dx[0], K = (int)dx[1], N = (int)wp.dimensions()[1];
    constexpr int BM = 64, BN = 64, BK = 32;
    constexpr int XS_LD = BK + 8, W_LD = BN + 8, NWARP = BM / FRAG;
    const size_t smem = (size_t)(BM * XS_LD + 2 * BK * W_LD) * sizeof(__half) +
                        (size_t)(NWARP * 2 * FRAG * FRAG) * sizeof(float);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    GDP_ACT_DISPATCH(
        activation, volta_gdp_wmma_kernel<BM, BN, BK, kAct><<<grid, NWARP * WARP, smem, stream>>>(
                        reinterpret_cast<const __half*>(x.typed_data()),
                        reinterpret_cast<const __half*>(wp.typed_data()),
                        reinterpret_cast<const __half*>(bp.typed_data()),
                        reinterpret_cast<const __half*>(wg.typed_data()),
                        reinterpret_cast<const __half*>(bg.typed_data()),
                        reinterpret_cast<const __half*>(mask.typed_data()),
                        reinterpret_cast<__half*>(out->typed_data()), M, K, N));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_gdp_wmma launch: ") +
                                    cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaGdpWmma, GdpWmmaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Attr<int64_t>("activation"),
                              {ffi::Traits::kCmdBufferCompatible});

// LayerNorm-fused projections, the same kernels as volta_ops.cu. The fragment layout is not
// known, thus results go through shared memory.

using FragA = wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, __half, wmma::col_major>;
using Acc = wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float>;

// A warp's 16x16 fp32 scratch for reading back an accumulator.
constexpr int kScratch = FRAG * FRAG;

// acc[f] += S[r0, r0+16) . W for columns n0 + 16 f of the tile [o0, o0+kTileN) of wT [N][K],
// streamed through Ws [kTileN][kWsLd]. Every thread of the block must call it.
template <int K, int NF>
__device__ __forceinline__ void tile_gemm(Acc* acc, const __half* S, int s_ld, __half* Ws,
                                          const __half* __restrict__ wT, int o0, int r0, int n0) {
    for (int k0 = 0; k0 < K; k0 += kTileK) {
        __syncthreads();
        load_rows<kTileN, kTileK, kWsLd>(Ws, wT, K, o0, k0, INT_MAX);
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < kTileK; ks += FRAG) {
            FragA fa;
            wmma::load_matrix_sync(fa, S + r0 * s_ld + k0 + ks, s_ld);
#pragma unroll
            for (int f = 0; f < NF; ++f) {
                // col_major: B(k, n) at n * ld + k, which is the [n][k] tile
                FragB fb;
                wmma::load_matrix_sync(fb, Ws + (n0 + f * FRAG) * kWsLd + ks, kWsLd);
                wmma::mma_sync(acc[f], fa, fb, acc[f]);
            }
        }
    }
}

template <int NF> __device__ __forceinline__ void clear(Acc* acc) {
#pragma unroll
    for (int f = 0; f < NF; ++f) {
        wmma::fill_fragment(acc[f], 0.f);
    }
}

// Give this lane's 8 of the 256 accumulator values, as the layout stores them at lane + 32 t.
__device__ __forceinline__ void read_back(float* v, float* ep, const Acc& acc,
                                          wmma::layout_t layout) {
    const int lane = threadIdx.x & (WARP - 1);
    wmma::store_matrix_sync(ep, acc, FRAG, layout);
    __syncwarp();
#pragma unroll
    for (int t = 0; t < kScratch / WARP; ++t) {
        v[t] = ep[lane + t * WARP];
    }
    __syncwarp();
}

// wmma wants 256-bit aligned tile pointers, so every buffer starts on a 32-byte boundary.
__host__ __device__ constexpr size_t align32(size_t b) {
    return (b + 31) / 32 * 32;
}

template <int K> __host__ __device__ constexpr size_t ln_gdp_xs() {
    return align32(kRows * (K + 8) * sizeof(__half));
}
__host__ __device__ constexpr size_t ln_gdp_ws() {
    constexpr size_t ws = kTileN * kWsLd, ts = kTileN * (kRows + 8);
    return align32((ws > ts ? ws : ts) * sizeof(__half));
}
template <int K> __host__ __device__ constexpr size_t ln_gdp_smem() {
    return ln_gdp_xs<K>() + ln_gdp_ws() + kThreads / WARP * kScratch * sizeof(float);
}

// See volta_ops.cu: 64 rows per block, the output columns stream past in 64-wide tiles,
// each staged channel major so the store is coalesced along M.
template <int K> __global__ __launch_bounds__(kThreads) void ln_gdp_kernel(LnGdpArgs a) {
    constexpr int BM = kRows, NF = kTileN / FRAG, XS_LD = K + 8, TS_LD = BM + 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int m0 = blockIdx.x * BM;

    extern __shared__ __align__(128) char smem[];
    __half* Xs = reinterpret_cast<__half*>(smem);                  // [BM][XS_LD]
    __half* Ws = reinterpret_cast<__half*>(smem + ln_gdp_xs<K>()); // [kTileN][kWsLd]
    __half* Ts = Ws;                                               // [kTileN][TS_LD] after Ws
    float* ep = reinterpret_cast<float*>(smem + ln_gdp_xs<K>() + ln_gdp_ws()) + warp * kScratch;

    // the mask is the keep: below zero outside what the contraction reads, inside a computed tile 0
    const TileWork work = tile_work<BM>(a.mask, a.M, m0, 0.f);
    if (work != kCompute) {
        if (work == kPadding) {
            zero_columns<BM>(a.left, a.P / 2, a.M, m0);
            zero_columns<BM>(a.right, a.P / 2, a.M, m0);
        }
        return;
    }
    load_rows<BM, K, XS_LD>(Xs, a.x, K, m0, 0, a.M);
    __syncthreads();
    ln_rows<BM, K, XS_LD>(Xs, a.scale, a.offset, a.eps);

    const int ci = a.P / 2;
    for (int n0 = 0; n0 < a.P; n0 += kTileN) {
        Acc accp[NF], accg[NF];
        clear<NF>(accp);
        clear<NF>(accg);
        tile_gemm<K, NF>(accp, Xs, XS_LD, Ws, a.wpT, n0, warp * FRAG, 0);
        tile_gemm<K, NF>(accg, Xs, XS_LD, Ws, a.wgT, n0, warp * FRAG, 0);
        __syncthreads();
#pragma unroll
        for (int f = 0; f < NF; ++f) {
            // column major: value lane + 32 t is row i % 16, column i / 16
            float p[kScratch / WARP], g[kScratch / WARP];
            read_back(p, ep, accp[f], wmma::mem_col_major);
            read_back(g, ep, accg[f], wmma::mem_col_major);
#pragma unroll
            for (int t = 0; t < kScratch / WARP; ++t) {
                const int i = lane + t * WARP, r = warp * FRAG + i % FRAG, n = f * FRAG + i / FRAG;
                const float mv = m0 + r < a.M ? fmaxf(__half2float(a.mask[m0 + r]), 0.f) : 0.f;
                const float pv = p[t] + __half2float(a.bp[n0 + n]);
                const float gv = g[t] + __half2float(a.bg[n0 + n]);
                Ts[n * TS_LD + r] = __float2half(mv * pv * gdp_act<GdpAct::kSigmoid>(gv));
            }
        }
        __syncthreads();
        __half* out = n0 < ci ? a.left : a.right;
        const int c0 = n0 < ci ? n0 : n0 - ci;
        for (int i = threadIdx.x; i < kTileN * BM; i += kThreads) {
            const int n = i / BM, r = i - n * BM;
            if (m0 + r < a.M) {
                out[(long long)(c0 + n) * a.M + m0 + r] = Ts[n * TS_LD + r];
            }
        }
    }
}

template <int K> void launch_ln_gdp(cudaStream_t stream, const LnGdpArgs& a) {
    const dim3 grid((a.M + kRows - 1) / kRows);
    ln_gdp_kernel<K><<<grid, kThreads, ln_gdp_smem<K>(), stream>>>(a);
}

__host__ __device__ constexpr size_t ws_bytes() {
    return align32(kTileN * kWsLd * sizeof(__half));
}
template <int C> __host__ __device__ constexpr size_t epi_ys() {
    return align32(kEpiRows * (C + 8) * sizeof(__half));
}
__host__ __device__ constexpr size_t epi_ep() {
    return kThreads / WARP * kScratch * sizeof(float);
}
template <int C, int CA> __host__ __device__ constexpr size_t epilogue_smem() {
    return epi_ys<C>() + epi_ys<CA>() + ws_bytes() + epi_ep() + kThreads * sizeof(float);
}

// See volta_ops.cu: 32 rows per block with LN(x_cm.T) and LN_in(x) in shared memory, a warp
// taking 16 rows x 32 columns of each 64-wide output tile.
template <int C, int CA, bool ADD>
__global__ __launch_bounds__(kThreads) void epilogue_kernel(EpilogueArgs a) {
    constexpr int BM = kEpiRows, NF = kTileN / 2 / FRAG, YS_LD = C + 8, AS_LD = CA + 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int wm = warp & 1, wn = warp >> 1;
    const int m0 = blockIdx.x * BM;

    extern __shared__ __align__(128) char smem[];
    char* p = smem;
    __half* Ys = reinterpret_cast<__half*>(p); // [BM][YS_LD] LN(x_cm.T)
    p += epi_ys<C>();
    __half* As = reinterpret_cast<__half*>(p); // [BM][AS_LD] LN_in(x)
    p += epi_ys<CA>();
    __half* Ws = reinterpret_cast<__half*>(p); // [kTileN][kWsLd]
    p += ws_bytes();
    float* ep = reinterpret_cast<float*>(p) + warp * kScratch;
    float* red = reinterpret_cast<float*>(p + epi_ep());

    const TileWork work = tile_work<BM>(a.keep, a.M, m0, -1.f);
    if (work != kCompute) {
        if (work == kPadding && !ADD) { // with add_input the output aliases x, which it keeps
            zero_rows<BM>(a.out, a.O, a.M, m0);
        }
        return;
    }
    load_rows<BM, CA, AS_LD>(As, a.x, CA, m0, 0, a.M);
    ln_columns<BM, C, YS_LD>(Ys, red, a.x_cm, a.M, m0, a.ln_scale, a.ln_offset, a.eps);
    ln_rows<BM, CA, AS_LD>(As, a.in_scale, a.in_offset, a.eps);

    for (int o0 = 0; o0 < a.O; o0 += kTileN) {
        Acc acc[NF], gate[NF];
        clear<NF>(acc);
        clear<NF>(gate);
        tile_gemm<C, NF>(acc, Ys, YS_LD, Ws, a.woT, o0, wm * FRAG, wn * 32);
        tile_gemm<CA, NF>(gate, As, AS_LD, Ws, a.wgT, o0, wm * FRAG, wn * 32);
#pragma unroll
        for (int f = 0; f < NF; ++f) {
            // row major: value lane + 32 t is row i / 16, column i % 16
            float v[kScratch / WARP], g[kScratch / WARP];
            read_back(v, ep, acc[f], wmma::mem_row_major);
            read_back(g, ep, gate[f], wmma::mem_row_major);
#pragma unroll
            for (int t = 0; t < kScratch / WARP; ++t) {
                const int i = lane + t * WARP;
                const int gm = m0 + wm * FRAG + i / FRAG, o = o0 + wn * 32 + f * FRAG + i % FRAG;
                if (gm >= a.M) {
                    continue;
                }
                const float gv = g[t] + __half2float(a.bg[o]);
                float out = (v[t] + __half2float(a.bo[o])) * gdp_act<GdpAct::kSigmoid>(gv);
                if constexpr (ADD) {
                    out += __half2float(a.x[(long long)gm * CA + o]);
                }
                a.out[(long long)gm * a.O + o] = __float2half(out);
            }
        }
    }
}

template <int C, int CA, bool ADD>
void launch_epilogue(cudaStream_t stream, const EpilogueArgs& a) {
    const dim3 grid((a.M + kEpiRows - 1) / kEpiRows);
    epilogue_kernel<C, CA, ADD><<<grid, kThreads, epilogue_smem<C, CA>(), stream>>>(a);
}

template <int K> __host__ __device__ constexpr size_t proj_xs() {
    return align32(kRows * (K + 8) * sizeof(__half));
}
template <int K> __host__ __device__ constexpr size_t proj_smem() {
    return proj_xs<K>() + ws_bytes() + kThreads / WARP * kScratch * sizeof(float) +
           kMaxHeads * kRows * sizeof(__half);
}

// See volta_ops.cu: 64 rows of x, optionally layer-normed, the per-head bias by row
// reductions, then out = x@w (+ b) (+ residual) over 64-wide column tiles.
template <int K, bool LN> __global__ __launch_bounds__(kThreads) void proj_kernel(ProjArgs a) {
    constexpr int BM = kRows, NF = kTileN / FRAG, XS_LD = K + 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int m0 = blockIdx.x * BM;

    extern __shared__ __align__(128) char smem[];
    __half* Xs = reinterpret_cast<__half*>(smem);                // [BM][XS_LD]
    __half* Ws = reinterpret_cast<__half*>(smem + proj_xs<K>()); // [kTileN][kWsLd]
    float* scratch = reinterpret_cast<float*>(smem + proj_xs<K>() + ws_bytes());
    float* ep = scratch + warp * kScratch;
    __half* Hb = reinterpret_cast<__half*>(scratch + kThreads / WARP * kScratch); // [kMaxHeads][BM]

    const TileWork work = tile_work<BM>(a.keep, a.M, m0, -1.f);
    if (work != kCompute) {
        if (work == kPadding) { // with a residual the output aliases it, which it keeps
            if (a.N && !a.residual) {
                zero_rows<BM>(a.out, a.N, a.M, m0);
            }
            if (a.heads) {
                zero_columns<BM>(a.bias, a.heads, a.M, m0);
            }
        }
        return;
    }
    load_rows<BM, K, XS_LD>(Xs, a.x, K, m0, 0, a.M);
    __syncthreads();
    if constexpr (LN) {
        ln_rows<BM, K, XS_LD>(Xs, a.scale, a.offset, a.eps);
        __syncthreads();
    }
    if (a.heads) {
        head_bias<BM, K, XS_LD>(Xs, a.wbT, a.heads, Hb, a.bias, a.M, m0);
    }

    for (int n0 = 0; n0 < a.N; n0 += kTileN) {
        Acc acc[NF];
        clear<NF>(acc);
        tile_gemm<K, NF>(acc, Xs, XS_LD, Ws, a.wT, n0, warp * FRAG, 0);
#pragma unroll
        for (int f = 0; f < NF; ++f) {
            float v[kScratch / WARP];
            read_back(v, ep, acc[f], wmma::mem_row_major);
#pragma unroll
            for (int t = 0; t < kScratch / WARP; ++t) {
                const int i = lane + t * WARP;
                const int gm = m0 + warp * FRAG + i / FRAG, n = n0 + f * FRAG + i % FRAG;
                if (gm >= a.M) {
                    continue;
                }
                float out = v[t];
                if (a.b) {
                    out += __half2float(a.b[n]);
                }
                if (a.residual) {
                    out += __half2float(a.residual[(long long)gm * a.N + n]);
                }
                a.out[(long long)gm * a.N + n] = __float2half(out);
            }
        }
    }
}

template <int K, bool LN> void launch_proj(cudaStream_t stream, const ProjArgs& a) {
    const dim3 grid((a.M + kRows - 1) / kRows);
    proj_kernel<K, LN><<<grid, kThreads, proj_smem<K>(), stream>>>(a);
}

// See volta_ops.cu: 64 rows a block, 32 with a 2x2 warp grid at 256 channels.
template <int C> __host__ __device__ constexpr int transition_rows() {
    return C > 128 ? kEpiRows : kRows;
}
template <int C> __host__ __device__ constexpr size_t tr_xs() {
    return align32(transition_rows<C>() * (C + 8) * sizeof(__half));
}
template <int C> __host__ __device__ constexpr size_t tr_hs() {
    return align32(transition_rows<C>() * (kHidden + 8) * sizeof(__half));
}
template <int C> __host__ __device__ constexpr size_t transition_smem() {
    return tr_xs<C>() + tr_hs<C>() + ws_bytes() + kThreads / WARP * kScratch * sizeof(float);
}

// See volta_ops.cu: BM rows of LN(x) and the fp32 output per block, the hidden activation through
// shared memory one kHidden chunk at a time, WN warps side by side on each 16 rows.
template <int C, bool ADD>
__global__ __launch_bounds__(kThreads) void transition_kernel(TransitionArgs a) {
    constexpr int BM = transition_rows<C>(), WM = BM / FRAG, WN = kThreads / WARP / WM;
    constexpr int NH = kHidden / WN / FRAG, NO = kTileN / WN / FRAG, NC = C / kTileN * NO;
    constexpr int XS_LD = C + 8, HS_LD = kHidden + 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    // one warp column at 64 rows: wn a constant there, not an index the registers carry
    const int wm = WN == 1 ? warp : warp % WM, wn = WN == 1 ? 0 : warp / WM;
    const int m0 = blockIdx.x * BM;

    extern __shared__ __align__(128) char smem[];
    __half* Xs = reinterpret_cast<__half*>(smem);                            // [BM][XS_LD] LN(x)
    __half* Hs = reinterpret_cast<__half*>(smem + tr_xs<C>());               // [BM][HS_LD]
    __half* Ws = reinterpret_cast<__half*>(smem + tr_xs<C>() + tr_hs<C>()); // [kTileN][kWsLd]
    float* ep = reinterpret_cast<float*>(smem + tr_xs<C>() + tr_hs<C>() + ws_bytes()) +
                warp * kScratch;

    const TileWork work = tile_work<BM>(a.keep, a.M, m0, -1.f);
    if (work != kCompute) {
        if (work == kPadding && !ADD) { // with add_input the output aliases x, which it keeps
            zero_rows<BM>(a.out, C, a.M, m0);
        }
        return;
    }
    load_rows<BM, C, XS_LD>(Xs, a.x, C, m0, 0, a.M);
    __syncthreads();
    ln_rows<BM, C, XS_LD>(Xs, a.scale, a.offset, a.eps);

    Acc acc[NC];
    clear<NC>(acc);
    for (int h0 = 0; h0 < a.H; h0 += kHidden) {
        Acc hid[NH];
        clear<NH>(hid);
        // its first barrier also orders this chunk's Hs writes after the last chunk's reads
        tile_gemm<C, NH>(hid, Xs, XS_LD, Ws, a.w1T, h0, wm * FRAG, wn * NH * FRAG);
#pragma unroll
        for (int f = 0; f < NH; ++f) {
            float v[kScratch / WARP];
            read_back(v, ep, hid[f], wmma::mem_row_major);
#pragma unroll
            for (int t = 0; t < kScratch / WARP; ++t) {
                const int i = lane + t * WARP, n = (wn * NH + f) * FRAG + i % FRAG;
                Hs[(wm * FRAG + i / FRAG) * HS_LD + n] =
                    __float2half(fmaxf(v[t] + __half2float(a.b1[h0 + n]), 0.f));
            }
        }
        const __half* w2 = a.w2T + (long long)(h0 / kHidden) * C * kHidden;
#pragma unroll
        for (int o0 = 0; o0 < C; o0 += kTileN) {
            tile_gemm<kHidden, NO>(acc + o0 / kTileN * NO, Hs, HS_LD, Ws, w2, o0, wm * FRAG,
                                   wn * NO * FRAG);
        }
    }

#pragma unroll
    for (int f = 0; f < NC; ++f) {
        float v[kScratch / WARP];
        read_back(v, ep, acc[f], wmma::mem_row_major);
#pragma unroll
        for (int t = 0; t < kScratch / WARP; ++t) {
            const int i = lane + t * WARP;
            const int gm = m0 + wm * FRAG + i / FRAG;
            const int n = f / NO * kTileN + (wn * NO + f % NO) * FRAG + i % FRAG;
            if (gm >= a.M) {
                continue;
            }
            float out = v[t] + __half2float(a.b2[n]);
            if constexpr (ADD) {
                out += __half2float(a.x[(long long)gm * C + n]);
            }
            a.out[(long long)gm * C + n] = __float2half(out);
        }
    }
}

template <int C, bool ADD> void launch_transition(cudaStream_t stream, const TransitionArgs& a) {
    constexpr int BM = transition_rows<C>();
    const dim3 grid((a.M + BM - 1) / BM);
    transition_kernel<C, ADD><<<grid, kThreads, transition_smem<C>(), stream>>>(a);
}

#define FUSED_SYM(name) name##Wmma
#include "volta_fused_ffi.h"
