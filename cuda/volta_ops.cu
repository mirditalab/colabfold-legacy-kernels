// LayerNorm for sm_70+, the gated dual projection and the LayerNorm-fused projections for sm_75+.
// LayerNorm uses no tensor cores, thus Volta can use it.
// The projections need mma.sync.m16n8k8. Volta uses volta_gdp_wmma.cu instead.

#include <cuda_fp16.h>
#include <cstdint>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "xla/ffi/api/ffi.h"

#include "volta_gdp.h"

namespace ffi = xla::ffi;

// VOLTA_MMA_OK removes the projection below sm_75. LayerNorm stays.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750)
#define VOLTA_MMA_OK 1
#endif

// LayerNorm on the last axis. One warp does one row and adds in fp32.
__global__ void volta_layer_norm_kernel(const __half* __restrict__ x,
                                        const float* __restrict__ scale,
                                        const float* __restrict__ offset, __half* __restrict__ out,
                                        int M, int C, float eps) {
    const int lane = threadIdx.x & (WARP - 1);
    const int warp_in_block = threadIdx.x / WARP;
    const int warps_per_block = blockDim.x / WARP;
    for (int row = blockIdx.x * warps_per_block + warp_in_block; row < M;
         row += gridDim.x * warps_per_block) {
        const __half* xr = x + (long long)row * C;
        __half* orow = out + (long long)row * C;

        float sum = 0.f, sqsum = 0.f;
        for (int c = lane; c < C; c += WARP) {
            float v = __half2float(xr[c]);
            sum += v;
            sqsum += v * v;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
            sqsum += __shfl_xor_sync(0xffffffffu, sqsum, off);
        }
        const float inv = 1.0f / (float)C;
        const float mean = sum * inv;
        const float var = fmaxf(sqsum * inv - mean * mean, 0.f);
        const float rstd = rsqrtf(var + eps);
        for (int c = lane; c < C; c += WARP) {
            float v = (__half2float(xr[c]) - mean) * rstd;
            orow[c] = __float2half(v * scale[c] + offset[c]);
        }
    }
}

ffi::Error LayerNormImpl(cudaStream_t stream, ffi::Buffer<ffi::DataType::F16> x,
                         ffi::Buffer<ffi::DataType::F32> scale,
                         ffi::Buffer<ffi::DataType::F32> offset,
                         ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, float eps) {
    auto d = x.dimensions();
    if (d.size() != 2) {
        return ffi::Error::InvalidArgument("x must be [M,C]");
    }
    const int M = (int)d[0], C = (int)d[1];
    constexpr int TPB = 128;
    const int warps = TPB / WARP;
    int blocks = (M + warps - 1) / warps;
    if (blocks > 65535) {
        blocks = 65535;
    }
    volta_layer_norm_kernel<<<blocks, TPB, 0, stream>>>(
        reinterpret_cast<const __half*>(x.typed_data()), scale.typed_data(), offset.typed_data(),
        reinterpret_cast<__half*>(out->typed_data()), M, C, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_gdp launch: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaLayerNorm, LayerNormImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Attr<float>("eps"),
                              {ffi::Traits::kCmdBufferCompatible});

// Gated dual projection: out = mask * (x@wp + bp) * act(x@wg + bg).
// The two GEMMs share one tile of x and keep the result in registers.
#ifdef VOLTA_MMA_OK
using MmaOp =
    cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 8>, WARP, cutlass::half_t,
                       cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
                       float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;
using FragA = cutlass::Array<cutlass::half_t, 4>;
using FragB = cutlass::Array<cutlass::half_t, 2>;
using FragC = cutlass::Array<float, 4>;

// BN must be a multiple of 8, BK a multiple of 8, BM a multiple of 16.
template <int BM, int BN, int BK, GdpAct A>
__global__ __launch_bounds__(BM / 16 *
                             WARP) void volta_gdp_kernel(const __half* __restrict__ x, // [M, K]
                                                         const __half* __restrict__ wp,
                                                         const __half* __restrict__ bp, // [K,N],[N]
                                                         const __half* __restrict__ wg,
                                                         const __half* __restrict__ bg,
                                                         const __half* __restrict__ mask, // [M]
                                                         __half* __restrict__ out,        // [M, N]
                                                         int M, int K, int N) {
    constexpr int NWARP = BM / 16;
    constexpr int NN = BN / 8;
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int tid = threadIdx.x;
    const int nthreads = NWARP * WARP;
    const int lr = lane >> 2, lc = (lane & 3) * 2;

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;

    extern __shared__ char smem[];
    __half* Xs = reinterpret_cast<__half*>(smem); // [BM][BK]
    __half* Wp = Xs + BM * BK;                    // [BK][BN]
    __half* Wg = Wp + BK * BN;                    // [BK][BN]

    FragC accp[NN], accg[NN];
#pragma unroll
    for (int i = 0; i < NN; ++i) {
        accp[i].clear();
        accg[i].clear();
    }

    MmaOp mma_op;
    for (int k0 = 0; k0 < K; k0 += BK) {
        __syncthreads();
        for (int i = tid; i < BM * BK; i += nthreads) {
            int r = i / BK, c = i - r * BK;
            int gm = m0 + r, gk = k0 + c;
            Xs[i] = (gm < M && gk < K) ? x[(long long)gm * K + gk] : __float2half(0.f);
        }
        for (int i = tid; i < BK * BN; i += nthreads) {
            int r = i / BN, c = i - r * BN;
            int gk = k0 + r, gn = n0 + c;
            bool ok = (gk < K && gn < N);
            Wp[i] = ok ? wp[(long long)gk * N + gn] : __float2half(0.f);
            Wg[i] = ok ? wg[(long long)gk * N + gn] : __float2half(0.f);
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < BK / 8; ++ks) {
            FragA fa;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                fa[0 + j] =
                    reinterpret_cast<cutlass::half_t&>(Xs[(warp * 16 + lr) * BK + ks * 8 + lc + j]);
                fa[2 + j] = reinterpret_cast<cutlass::half_t&>(
                    Xs[(warp * 16 + lr + 8) * BK + ks * 8 + lc + j]);
            }
#pragma unroll
            for (int nt = 0; nt < NN; ++nt) {
                // B is column-major (k x n): lane holds k=(t%4)*2+{0,1}, n=t/4
                FragB fbp, fbg;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int idx = (ks * 8 + lc + j) * BN + nt * 8 + lr;
                    fbp[j] = reinterpret_cast<cutlass::half_t&>(Wp[idx]);
                    fbg[j] = reinterpret_cast<cutlass::half_t&>(Wg[idx]);
                }
                mma_op(accp[nt], fa, fbp, accp[nt]);
                mma_op(accg[nt], fa, fbg, accg[nt]);
            }
        }
    }

// fused epilogue: + bias, gate, row mask
#pragma unroll
    for (int nt = 0; nt < NN; ++nt) {
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int gm = m0 + warp * 16 + lr + half_i * 8;
            if (gm >= M) {
                continue;
            }
            const float mv = __half2float(mask[gm]);
            // the lane's two adjacent columns as one half2: N % 8 == 0, so gn + 1 < N
            const int gn = n0 + nt * 8 + lc;
            if (gn >= N) {
                continue;
            }
            float o[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float p = accp[nt][half_i * 2 + j] + __half2float(bp[gn + j]);
                const float g = accg[nt][half_i * 2 + j] + __half2float(bg[gn + j]);
                o[j] = mv * p * gdp_act<A>(g);
            }
            *reinterpret_cast<__half2*>(out + (long long)gm * N + gn) = __floats2half2_rn(o[0], o[1]);
        }
    }
}

ffi::Error GdpImpl(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::DataType::F16> x,
                   ffi::Buffer<ffi::DataType::F16> wp, ffi::Buffer<ffi::DataType::F16> bp,
                   ffi::Buffer<ffi::DataType::F16> wg, ffi::Buffer<ffi::DataType::F16> bg,
                   ffi::Buffer<ffi::DataType::F16> mask,
                   ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, int64_t activation) {
    auto dx = x.dimensions();
    if (dx.size() != 2) {
        return ffi::Error::InvalidArgument("x must be [M,K]");
    }
    const int M = (int)dx[0], K = (int)dx[1];
    const int N = (int)wp.dimensions()[1];
    // The kernel has no code below sm_75. Give an error before you start it.
    {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        const int cc = major * 10 + minor;
        if (cc < 75) {
            return ffi::Error::InvalidArgument(
                "VoltaGdp needs sm_75+ (mma.sync.m16n8k8); this device is sm_" +
                std::to_string(cc) + ". Volta needs an m8n8k4 port.");
        }
    }
    constexpr int BM = 64, BN = 64, BK = 32;
    if (N % 8) {
        return ffi::Error::InvalidArgument("gdp: N must be a multiple of 8");
    }
    const size_t smem = (size_t)(BM * BK + 2 * BK * BN) * sizeof(__half);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    GDP_ACT_DISPATCH(
        activation, volta_gdp_kernel<BM, BN, BK, kAct><<<grid, BM / 16 * WARP, smem, stream>>>(
                        reinterpret_cast<const __half*>(x.typed_data()),
                        reinterpret_cast<const __half*>(wp.typed_data()),
                        reinterpret_cast<const __half*>(bp.typed_data()),
                        reinterpret_cast<const __half*>(wg.typed_data()),
                        reinterpret_cast<const __half*>(bg.typed_data()),
                        reinterpret_cast<const __half*>(mask.typed_data()),
                        reinterpret_cast<__half*>(out->typed_data()), M, K, N));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_gdp launch: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaGdp, GdpImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Attr<int64_t>("activation"),
                              {ffi::Traits::kCmdBufferCompatible});

// LayerNorm-fused projections: the tri-mul gated dual projection and epilogue, the attention
// input and output projections, and the transition.

// Rows [r0, r0+16) x [k, k+8) of a row-major tile: lane holds rows lr and lr+8, columns lc, lc+1.
__device__ __forceinline__ FragA frag_a(const __half* s, int ld, int r0, int k) {
    const int lane = threadIdx.x & (WARP - 1), lr = lane >> 2, lc = (lane & 3) * 2;
    alignas(8) FragA a;
    uint32_t* p = reinterpret_cast<uint32_t*>(&a);
    p[0] = *reinterpret_cast<const uint32_t*>(s + (r0 + lr) * ld + k + lc);
    p[1] = *reinterpret_cast<const uint32_t*>(s + (r0 + lr + 8) * ld + k + lc);
    return a;
}

// Columns [n0, n0+8) x [k, k+8) of B from a tile stored [n][k]: lane holds n=lr, k=lc, lc+1.
__device__ __forceinline__ FragB frag_b(const __half* s, int ld, int n0, int k) {
    const int lane = threadIdx.x & (WARP - 1), lr = lane >> 2, lc = (lane & 3) * 2;
    alignas(4) FragB b;
    *reinterpret_cast<uint32_t*>(&b) =
        *reinterpret_cast<const uint32_t*>(s + (n0 + lr) * ld + k + lc);
    return b;
}

// The same fragment from a tile stored [k][n].
__device__ __forceinline__ FragB frag_b_kn(const __half* s, int ld, int n0, int k) {
    const int lane = threadIdx.x & (WARP - 1), lr = lane >> 2, lc = (lane & 3) * 2;
    FragB b;
    b[0] = reinterpret_cast<const cutlass::half_t&>(s[(k + lc) * ld + n0 + lr]);
    b[1] = reinterpret_cast<const cutlass::half_t&>(s[(k + lc + 1) * ld + n0 + lr]);
    return b;
}

// acc[nt] += S[r0, r0+16) . W for columns n0 + 8 nt of the tile [o0, o0+kTileN) of wT [N][K],
// streamed through Ws [kTileN][kWsLd]. Every thread of the block must call it.
template <int K, int NN>
__device__ __forceinline__ void tile_gemm(FragC* acc, const __half* S, int s_ld, __half* Ws,
                                          const __half* __restrict__ wT, int o0, int r0, int n0) {
    MmaOp mma;
    for (int k0 = 0; k0 < K; k0 += kTileK) {
        __syncthreads();
        load_rows<kTileN, kTileK, kWsLd>(Ws, wT, K, o0, k0, INT_MAX);
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < kTileK; ks += 8) {
            const FragA fa = frag_a(S, s_ld, r0, k0 + ks);
#pragma unroll
            for (int nt = 0; nt < NN; ++nt) {
                mma(acc[nt], fa, frag_b(Ws, kWsLd, n0 + nt * 8, ks), acc[nt]);
            }
        }
    }
}

template <int NN> __device__ __forceinline__ void clear(FragC* acc) {
#pragma unroll
    for (int i = 0; i < NN; ++i) {
        acc[i].clear();
    }
}

template <int K> constexpr size_t ln_gdp_smem() {
    constexpr size_t ws = kTileN * kWsLd, ts = kTileN * (kRows + 8);
    return (kRows * (K + 8) + (ws > ts ? ws : ts)) * sizeof(__half);
}

// A block owns 64 rows: their LN(x) stays in shared memory while the 2*ci output columns
// stream past in 64-wide tiles, each staged channel major so the store is coalesced along M.
template <int K> __global__ __launch_bounds__(kThreads) void ln_gdp_kernel(LnGdpArgs a) {
    constexpr int BM = kRows, NN = kTileN / 8, XS_LD = K + 8, TS_LD = BM + 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int lr = lane >> 2, lc = (lane & 3) * 2;
    const int m0 = blockIdx.x * BM;

    extern __shared__ __align__(16) char smem[];
    __half* Xs = reinterpret_cast<__half*>(smem); // [BM][XS_LD]
    __half* Ws = Xs + BM * XS_LD;                 // [kTileN][kWsLd] weight tile
    __half* Ts = Ws;                              // [kTileN][TS_LD] output tile, once Ws is done

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
        FragC accp[NN], accg[NN];
        clear<NN>(accp);
        clear<NN>(accg);
        tile_gemm<K, NN>(accp, Xs, XS_LD, Ws, a.wpT, n0, warp * 16, 0);
        tile_gemm<K, NN>(accg, Xs, XS_LD, Ws, a.wgT, n0, warp * 16, 0);
        __syncthreads();
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            const int r = warp * 16 + lr + hi * 8;
            const float mv = m0 + r < a.M ? fmaxf(__half2float(a.mask[m0 + r]), 0.f) : 0.f;
#pragma unroll
            for (int nt = 0; nt < NN; ++nt) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int n = nt * 8 + lc + j;
                    const float p = accp[nt][hi * 2 + j] + __half2float(a.bp[n0 + n]);
                    const float g = accg[nt][hi * 2 + j] + __half2float(a.bg[n0 + n]);
                    Ts[n * TS_LD + r] = __float2half(mv * p * gdp_act<GdpAct::kSigmoid>(g));
                }
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

template <int C, int CA> constexpr size_t epilogue_smem() {
    return (kEpiRows * (C + 8) + kEpiRows * (CA + 8) + kTileN * kWsLd) * sizeof(__half) +
           kThreads * sizeof(float);
}

// A block owns 32 rows: LN(x_cm.T) and LN_in(x) both stay in shared memory while the output
// columns stream past in 64-wide tiles, a warp taking 16 rows x 32 columns of each.
template <int C, int CA, bool ADD>
__global__ __launch_bounds__(kThreads) void epilogue_kernel(EpilogueArgs a) {
    constexpr int BM = kEpiRows, NN = kTileN / 2 / 8, YS_LD = C + 8, AS_LD = CA + 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int lr = lane >> 2, lc = (lane & 3) * 2;
    const int wm = warp & 1, wn = warp >> 1;
    const int m0 = blockIdx.x * BM;

    extern __shared__ __align__(16) char smem[];
    __half* Ys = reinterpret_cast<__half*>(smem); // [BM][YS_LD] LN(x_cm.T)
    __half* As = Ys + BM * YS_LD;                 // [BM][AS_LD] LN_in(x)
    __half* Ws = As + BM * AS_LD;                 // [kTileN][kWsLd]
    float* red = reinterpret_cast<float*>(Ws + kTileN * kWsLd);

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
        FragC acc[NN], gate[NN];
        clear<NN>(acc);
        clear<NN>(gate);
        tile_gemm<C, NN>(acc, Ys, YS_LD, Ws, a.woT, o0, wm * 16, wn * 32);
        tile_gemm<CA, NN>(gate, As, AS_LD, Ws, a.wgT, o0, wm * 16, wn * 32);
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            const int gm = m0 + wm * 16 + lr + hi * 8;
            if (gm >= a.M) {
                continue;
            }
#pragma unroll
            for (int nt = 0; nt < NN; ++nt) {
                const int o = o0 + wn * 32 + nt * 8 + lc;
                float v[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float g = gate[nt][hi * 2 + j] + __half2float(a.bg[o + j]);
                    v[j] = (acc[nt][hi * 2 + j] + __half2float(a.bo[o + j])) *
                           gdp_act<GdpAct::kSigmoid>(g);
                }
                if constexpr (ADD) {
                    const float2 x = __half22float2(
                        *reinterpret_cast<const __half2*>(a.x + (long long)gm * CA + o));
                    v[0] += x.x;
                    v[1] += x.y;
                }
                *reinterpret_cast<__half2*>(a.out + (long long)gm * a.O + o) =
                    __floats2half2_rn(v[0], v[1]);
            }
        }
    }
}

template <int C, int CA, bool ADD>
void launch_epilogue(cudaStream_t stream, const EpilogueArgs& a) {
    const dim3 grid((a.M + kEpiRows - 1) / kEpiRows);
    epilogue_kernel<C, CA, ADD><<<grid, kThreads, epilogue_smem<C, CA>(), stream>>>(a);
}

template <int K> constexpr size_t proj_smem() {
    return (kRows * (K + 8) + kTileN * kWsLd + kMaxHeads * kRows) * sizeof(__half);
}

// A block owns 64 rows of x, optionally layer-normed: the per-head bias by row reductions, then
// out = x@w (+ b) (+ residual) over 64-wide column tiles.
template <int K, bool LN> __global__ __launch_bounds__(kThreads) void proj_kernel(ProjArgs a) {
    constexpr int BM = kRows, NN = kTileN / 8, XS_LD = K + 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int lr = lane >> 2, lc = (lane & 3) * 2;
    const int m0 = blockIdx.x * BM;

    extern __shared__ __align__(16) char smem[];
    __half* Xs = reinterpret_cast<__half*>(smem); // [BM][XS_LD]
    __half* Ws = Xs + BM * XS_LD;                 // [kTileN][kWsLd]
    __half* Hb = Ws + kTileN * kWsLd;             // [kMaxHeads][BM]

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
        FragC acc[NN];
        clear<NN>(acc);
        tile_gemm<K, NN>(acc, Xs, XS_LD, Ws, a.wT, n0, warp * 16, 0);
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            const int gm = m0 + warp * 16 + lr + hi * 8;
            if (gm >= a.M) {
                continue;
            }
#pragma unroll
            for (int nt = 0; nt < NN; ++nt) {
                const int n = n0 + nt * 8 + lc;
                float v0 = acc[nt][hi * 2], v1 = acc[nt][hi * 2 + 1];
                if (a.b) {
                    v0 += __half2float(a.b[n]);
                    v1 += __half2float(a.b[n + 1]);
                }
                if (a.residual) {
                    const float2 r = __half22float2(
                        *reinterpret_cast<const __half2*>(a.residual + (long long)gm * a.N + n));
                    v0 += r.x;
                    v1 += r.y;
                }
                *reinterpret_cast<__half2*>(a.out + (long long)gm * a.N + n) =
                    __floats2half2_rn(v0, v1);
            }
        }
    }
}

template <int K, bool LN> void launch_proj(cudaStream_t stream, const ProjArgs& a) {
    const dim3 grid((a.M + kRows - 1) / kRows);
    proj_kernel<K, LN><<<grid, kThreads, proj_smem<K>(), stream>>>(a);
}

// Rows per block: 64 with a warp per 16 rows, 32 with a 2x2 warp grid at 256 channels, where a
// 64-row fp32 output would not fit the registers.
template <int C> __host__ __device__ constexpr int transition_rows() {
    return C > 128 ? kEpiRows : kRows;
}

template <int C> constexpr size_t transition_smem() {
    constexpr int BM = transition_rows<C>();
    return (BM * (C + 8) + BM * (kHidden + 8) + kTileN * kWsLd) * sizeof(__half);
}

// A block owns BM rows of LN(x) and the fp32 [BM, C] output: each kHidden-wide chunk of the hidden
// activation goes through shared memory, WN warps side by side on each 16 rows.
template <int C, bool ADD>
__global__ __launch_bounds__(kThreads) void transition_kernel(TransitionArgs a) {
    constexpr int BM = transition_rows<C>(), WM = BM / 16, WN = kThreads / WARP / WM;
    constexpr int NH = kHidden / WN / 8, NO = kTileN / WN / 8, NC = C / kTileN * NO;
    constexpr int XS_LD = C + 8, HS_LD = kHidden + 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int lr = lane >> 2, lc = (lane & 3) * 2;
    // one warp column at 64 rows: wn a constant there, not an index the registers carry
    const int wm = WN == 1 ? warp : warp % WM, wn = WN == 1 ? 0 : warp / WM;
    const int m0 = blockIdx.x * BM;

    extern __shared__ __align__(16) char smem[];
    __half* Xs = reinterpret_cast<__half*>(smem); // [BM][XS_LD] LN(x)
    __half* Hs = Xs + BM * XS_LD;                 // [BM][HS_LD] one hidden chunk
    __half* Ws = Hs + BM * HS_LD;                 // [kTileN][kWsLd]

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

    FragC acc[NC];
    clear<NC>(acc);
    for (int h0 = 0; h0 < a.H; h0 += kHidden) {
        FragC hid[NH];
        clear<NH>(hid);
        // its first barrier also orders this chunk's Hs writes after the last chunk's reads
        tile_gemm<C, NH>(hid, Xs, XS_LD, Ws, a.w1T, h0, wm * 16, wn * NH * 8);
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            const int r = wm * 16 + lr + hi * 8;
#pragma unroll
            for (int nt = 0; nt < NH; ++nt) {
                const int n = wn * NH * 8 + nt * 8 + lc;
                *reinterpret_cast<__half2*>(Hs + r * HS_LD + n) = __floats2half2_rn(
                    fmaxf(hid[nt][hi * 2] + __half2float(a.b1[h0 + n]), 0.f),
                    fmaxf(hid[nt][hi * 2 + 1] + __half2float(a.b1[h0 + n + 1]), 0.f));
            }
        }
        const __half* w2 = a.w2T + (long long)(h0 / kHidden) * C * kHidden;
#pragma unroll
        for (int o0 = 0; o0 < C; o0 += kTileN) {
            tile_gemm<kHidden, NO>(acc + o0 / kTileN * NO, Hs, HS_LD, Ws, w2, o0, wm * 16,
                                   wn * NO * 8);
        }
    }

#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
        const int gm = m0 + wm * 16 + lr + hi * 8;
        if (gm >= a.M) {
            continue;
        }
#pragma unroll
        for (int t = 0; t < NC; ++t) {
            const int n = t / NO * kTileN + wn * NO * 8 + t % NO * 8 + lc;
            float v0 = acc[t][hi * 2] + __half2float(a.b2[n]);
            float v1 = acc[t][hi * 2 + 1] + __half2float(a.b2[n + 1]);
            if constexpr (ADD) {
                const float2 x =
                    __half22float2(*reinterpret_cast<const __half2*>(a.x + (long long)gm * C + n));
                v0 += x.x;
                v1 += x.y;
            }
            *reinterpret_cast<__half2*>(a.out + (long long)gm * C + n) = __floats2half2_rn(v0, v1);
        }
    }
}

template <int C, bool ADD> void launch_transition(cudaStream_t stream, const TransitionArgs& a) {
    constexpr int BM = transition_rows<C>();
    const dim3 grid((a.M + BM - 1) / BM);
    transition_kernel<C, ADD><<<grid, kThreads, transition_smem<C>(), stream>>>(a);
}

template <int F> constexpr size_t opm_update_smem() {
    return (kRows * kWsLd + kTileK * (F + 8)) * sizeof(__half);
}

// A block owns 64 rows b of one column d: lhs and t[:, d] stream past in kTileK-deep tiles, then
// the bias, the mask norm and the residual on the fp32 [64, F] product.
template <int F> __global__ __launch_bounds__(kThreads) void opm_update_kernel(OpmUpdateArgs a) {
    constexpr int BM = kRows, NN = F / 8, TS_LD = F + 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int lr = lane >> 2, lc = (lane & 3) * 2;
    const int m0 = blockIdx.x * BM, d = blockIdx.y;

    extern __shared__ __align__(16) char smem[];
    __half* As = reinterpret_cast<__half*>(smem); // [BM][kWsLd] lhs
    __half* Ts = As + BM * kWsLd;                 // [kTileK][TS_LD] t[:, d]

    FragC acc[NN];
    clear<NN>(acc);
    MmaOp mma;
    for (int k0 = 0; k0 < a.K; k0 += kTileK) {
        __syncthreads();
        load_rows<BM, kTileK, kWsLd>(As, a.lhs, a.K, m0, k0, a.B);
        load_rows<kTileK, F, TS_LD>(Ts, a.t, a.D * F, k0, d * F, INT_MAX);
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < kTileK; ks += 8) {
            const FragA fa = frag_a(As, kWsLd, warp * 16, ks);
#pragma unroll
            for (int nt = 0; nt < NN; ++nt) {
                mma(acc[nt], fa, frag_b_kn(Ts, TS_LD, nt * 8, ks), acc[nt]);
            }
        }
    }

#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
        const int b = m0 + warp * 16 + lr + hi * 8;
        if (b >= a.B) {
            continue;
        }
        const long long row = (long long)b * a.D + d;
        const float den = a.eps + a.norm[row];
#pragma unroll
        for (int nt = 0; nt < NN; ++nt) {
            const int n = nt * 8 + lc;
            float v0 = (acc[nt][hi * 2] + a.bias[n]) / den;
            float v1 = (acc[nt][hi * 2 + 1] + a.bias[n + 1]) / den;
            if (a.residual) {
                const float2 r =
                    __half22float2(*reinterpret_cast<const __half2*>(a.residual + row * F + n));
                v0 += r.x;
                v1 += r.y;
            }
            *reinterpret_cast<__half2*>(a.out + row * F + n) = __floats2half2_rn(v0, v1);
        }
    }
}

template <int F> void launch_opm_update(cudaStream_t stream, const OpmUpdateArgs& a) {
    const dim3 grid((a.B + kRows - 1) / kRows, a.D);
    opm_update_kernel<F><<<grid, kThreads, opm_update_smem<F>(), stream>>>(a);
}

template <int F> constexpr size_t opm_outer_smem() {
    return (kOpmChunk * kOpmB * kLtLd + F * kWsLd) * sizeof(__half);
}

// A block owns kOpmB rows b and kOpmD columns d, warp w the rows (b, d0 + w). Per outer channel c,
// g[b, e] = sum_a left[a, b, c] right[a, d, e] over the 16 MSA rows, rounded to f16 in the
// registers (m16n8k8's accumulator layout is its A layout), times w[c*E + e, :] into the fp32
// [16, F] output; the [B, C*E, D] product is never written.
template <int F> __global__ __launch_bounds__(kThreads) void opm_outer_kernel(OpmOuterArgs a) {
    constexpr int NN = F / 8, NE = kOpmE / 8;
    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP;
    const int lr = lane >> 2, lc = (lane & 3) * 2;
    const int b0 = blockIdx.x * kOpmB, d = blockIdx.y * kOpmD + warp;

    extern __shared__ __align__(16) char smem[];
    __half* Lt = reinterpret_cast<__half*>(smem); // left[a, b, c] at [c][b][a], kOpmChunk channels
    __half* Ws = Lt + kOpmChunk * kOpmB * kLtLd;  // [F][kWsLd] w's rows for one c, transposed

    // right[:, d] as B fragments, k = a and n = e, the same for every c
    FragB rf[kOpmRows / 8][NE];
#pragma unroll
    for (int ks = 0; ks < kOpmRows / 8; ++ks) {
#pragma unroll
        for (int nt = 0; nt < NE; ++nt) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                rf[ks][nt][j] = reinterpret_cast<const cutlass::half_t&>(
                    a.right[((long long)(ks * 8 + lc + j) * a.D + d) * kOpmE + nt * 8 + lr]);
            }
        }
    }

    FragC acc[NN];
    clear<NN>(acc);
    MmaOp mma;
    for (int c = 0; c < a.C; ++c) {
        const int cl = c % kOpmChunk;
        __syncthreads();
        if (cl == 0) {
            stage_left(Lt, a.left, a.B, a.C, b0, c);
        }
        load_rows<F, kOpmE, kWsLd>(Ws, a.wT, a.C * kOpmE, 0, c * kOpmE, INT_MAX);
        __syncthreads();

        FragC g[NE];
        clear<NE>(g);
#pragma unroll
        for (int ks = 0; ks < kOpmRows / 8; ++ks) {
            const FragA fl = frag_a(Lt + cl * kOpmB * kLtLd, kLtLd, 0, ks * 8);
#pragma unroll
            for (int nt = 0; nt < NE; ++nt) {
                mma(g[nt], fl, rf[ks][nt], g[nt]);
            }
        }
#pragma unroll
        for (int ke = 0; ke < NE; ++ke) {
            FragA fg;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                fg[i] = cutlass::half_t(g[ke][i]);
            }
#pragma unroll
            for (int nt = 0; nt < NN; ++nt) {
                mma(acc[nt], fg, frag_b(Ws, kWsLd, nt * 8, ke * 8), acc[nt]);
            }
        }
    }

#pragma unroll
    for (int hi = 0; hi < 2; ++hi) {
        const int b = b0 + lr + hi * 8;
        if (b >= a.B) {
            continue;
        }
        __half* o = a.out + ((long long)b * a.D + d) * F;
#pragma unroll
        for (int nt = 0; nt < NN; ++nt) {
            *reinterpret_cast<__half2*>(o + nt * 8 + lc) =
                __floats2half2_rn(acc[nt][hi * 2], acc[nt][hi * 2 + 1]);
        }
    }
}

template <int F> void launch_opm_outer(cudaStream_t stream, const OpmOuterArgs& a) {
    const dim3 grid((a.B + kOpmB - 1) / kOpmB, a.D / kOpmD);
    opm_outer_kernel<F><<<grid, kThreads, opm_outer_smem<F>(), stream>>>(a);
}


#define FUSED_SYM(name) name
#include "volta_fused_ffi.h"
#endif // VOLTA_MMA_OK
