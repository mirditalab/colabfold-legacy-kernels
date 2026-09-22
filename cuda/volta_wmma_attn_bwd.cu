// Backward pass for volta_wmma_attn.cu (sm_70+), with nvcuda::wmma.
//
// This is the sm_70 twin of volta_mma_attn_bwd.cu, and the difference between
// them is the one the forward's header already names: with mma.sync.m16n8k8 the
// accumulator and the operand A share a fragment layout, so a computed P or dS
// tile feeds the next GEMM with only a cast. VOLTA CANNOT DO THIS -- wmma's
// fragment layout is opaque -- so every intermediate tile round-trips through
// shared memory: S and dP are stored as f32, the softmax and the dS product are
// scalar work on those, and dS goes back as f16 for the next GEMM to load.
//
// Inputs   q,k,v   [N,H,Sq|Sk,DR] f16
//          bias    [H,Sq,Sk] f16        (shared across the batch)
//          kmask   [N,Sk] u8
//          dout    [N,H,Sq,DR] f16
//          lse     [N,H,Sq] f32         from VoltaWmmaFwd (log2 domain)
//          delta   [N,H,Sq] f32         rowsum(dout * out); one line of XLA
// Outputs  dq,dk,dv [same as q,k,v] f16
//          dbias   [H,Sq,Sk] f32        summed over the batch with atomics
//
// dBIAS IS THE REASON THIS EXISTS. ColabFold's AF2 path never needs it -- its
// pair bias is not on the gradient path -- but AlphaFold 3 reaches the pair
// representation THROUGH the attention bias, so dQ/dK/dV alone would be useless
// to it.
//
// Two kernels, split so that only dbias needs atomics:
//   dq kernel:   one block owns 16 query rows per warp and loops over the keys.
//   dkdv kernel: one block owns 16 key rows per warp and loops over the
//                queries, working on the transposed problem.
//
// As in the forward, DR is the real head dim in global memory and D the
// shared-memory tile width, which wmma needs to be a multiple of 16.

#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
#include <string>

#include "volta_attn.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace nvcuda;

#define WARP 32
#define FRAG 16

#define MAX_DEVICES 16

static int bwd_shared_limit(int device) {
    static int cache[MAX_DEVICES] = {};
    if (device < 0 || device >= MAX_DEVICES) {
        return 0;
    }
    if (cache[device] == 0) {
        cudaDeviceGetAttribute(&cache[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    }
    return cache[device];
}

// Stage a [rows, D] tile of a [N,H,S,DR] tensor: zero past the end of the
// sequence, and zero in the columns that pad DR out to the tile width.
__device__ inline void load_tile(__half* dst, const __half* src, long long base, int row0, int rows,
                                 int S, int D, int DR, int tid, int nthreads) {
    for (int i = tid; i < rows * D; i += nthreads) {
        int r = i / D, c = i - r * D;
        int g = row0 + r;
        dst[i] = (g < S && c < DR) ? src[base + (long long)g * DR + c] : __float2half(0.f);
    }
}

// -----------------------------------------------------------------------------
// dQ (and dBias): 16 query rows per warp, looping over every key.
// -----------------------------------------------------------------------------
template <int D, int BK, int BQ, bool WANT_DBIAS>
__global__ __launch_bounds__(BK / FRAG * WARP) void volta_wmma_bwd_fused_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ bias, const uint8_t* __restrict__ kmask,
    const __half* __restrict__ dout, const float* __restrict__ lse,
    const float* __restrict__ delta, float* __restrict__ dq_accum, __half* __restrict__ dk,
    __half* __restrict__ dv, float* __restrict__ dbias, int N, int H, int Sq, int Sk, int DR,
    float sm_scale) {
    constexpr int NWARP = BK / FRAG;
    constexpr int SS_LD = (BQ > D ? BQ : D) + 4;
    constexpr int PS_LD = BQ + 8;
    constexpr int TPB = NWARP * WARP;

    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int tid = threadIdx.x;

    const int ktile = blockIdx.x, h = blockIdx.y, n = blockIdx.z;
    const int k0 = ktile * BK;
    if (k0 >= Sk) {
        return;
    }

    extern __shared__ char smem[];
    float* Ss = reinterpret_cast<float*>(smem);               // [BK][SS_LD]  S^T, then dK
    float* Ts = Ss + BK * SS_LD;                              // [BK][SS_LD]  dP^T, then dV
    __half* Ps = reinterpret_cast<__half*>(Ts + BK * SS_LD);  // [BK][PS_LD]  P^T
    __half* DSs = Ps + BK * PS_LD;                            // [BK][PS_LD]  dS^T
    __half* Ks = DSs + BK * PS_LD;                            // [BK][D]
    __half* Vs = Ks + BK * D;                                 // [BK][D]
    __half* Qs = Vs + BK * D;                                 // [BQ][D]
    __half* Os = Qs + BQ * D;                                 // [BQ][D]  dout

    const long long qkv = (long long)(n * H + h) * Sq * DR;
    const long long kv = (long long)(n * H + h) * Sk * DR;
    const long long bh = (long long)h * Sq * Sk;
    const long long rowbase = (long long)(n * H + h) * Sq;

    load_tile(Ks, k, kv, k0, BK, Sk, D, DR, tid, TPB);
    load_tile(Vs, v, kv, k0, BK, Sk, D, DR, tid, TPB);
    __syncthreads();

    wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::row_major> fk[D / FRAG];
    wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::row_major> fv[D / FRAG];
#pragma unroll
    for (int ks = 0; ks < D / FRAG; ++ks) {
        wmma::load_matrix_sync(fk[ks], Ks + (warp * FRAG) * D + ks * FRAG, D);
        wmma::load_matrix_sync(fv[ks], Vs + (warp * FRAG) * D + ks * FRAG, D);
    }

    wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> acc_dk[D / FRAG], acc_dv[D / FRAG];
#pragma unroll
    for (int d = 0; d < D / FRAG; ++d) {
        wmma::fill_fragment(acc_dk[d], 0.0f);
        wmma::fill_fragment(acc_dv[d], 0.0f);
    }

    const float qk_scale = sm_scale * LOG2E;

    for (int qq = 0; qq < Sq; qq += BQ) {
        __syncthreads();
        load_tile(Qs, q, qkv, qq, BQ, Sq, D, DR, tid, TPB);
        load_tile(Os, dout, qkv, qq, BQ, Sq, D, DR, tid, TPB);
        __syncthreads();

        // S^T = K @ Q^T and dP^T = V @ dO^T
#pragma unroll
        for (int nt = 0; nt < BQ / FRAG; ++nt) {
            wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> as, ap;
            wmma::fill_fragment(as, 0.0f);
            wmma::fill_fragment(ap, 0.0f);
#pragma unroll
            for (int ks = 0; ks < D / FRAG; ++ks) {
                wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, __half, wmma::col_major> fqb, fob;
                wmma::load_matrix_sync(fqb, Qs + (nt * FRAG) * D + ks * FRAG, D);
                wmma::load_matrix_sync(fob, Os + (nt * FRAG) * D + ks * FRAG, D);
                wmma::mma_sync(as, fk[ks], fqb, as);
                wmma::mma_sync(ap, fv[ks], fob, ap);
            }
            wmma::store_matrix_sync(Ss + (warp * FRAG) * SS_LD + nt * FRAG, as, SS_LD,
                                    wmma::mem_row_major);
            wmma::store_matrix_sync(Ts + (warp * FRAG) * SS_LD + nt * FRAG, ap, SS_LD,
                                    wmma::mem_row_major);
        }
        __syncwarp();

        // A key row the mask kills contributes nothing to dK or dV, and neither
        // does a query past the end: both fall out as P^T = 0.
        // key index runs with the lane, so a warp reads one bias row coalesced
        for (int i = lane; i < FRAG * BQ; i += WARP) {
            const int c = i / FRAG, r = i - c * FRAG;
            const int k_loc = warp * FRAG + r;
            const int gk = k0 + k_loc, gq = qq + c;
            const int gqc = gq < Sq ? gq : Sq - 1, gkc = gk < Sk ? gk : Sk - 1;
            const bool live = gq < Sq && gk < Sk && kmask[(long long)n * Sk + gk] != 0;
            const float l2 =
                live ? (Ss[k_loc * SS_LD + c] * qk_scale +
                        __half2float(bias[bh + (long long)gqc * Sk + gkc]) * LOG2E)
                     : (MASKED_LOGIT * LOG2E);
            // P carries dV even where the mask clamped the logit, but dS does
            // not: a constant logit has no gradient
            const float p = exp2f(l2 - lse[rowbase + gqc]);
            const float ds = live ? p * (Ts[k_loc * SS_LD + c] - delta[rowbase + gqc]) : 0.f;
            Ps[k_loc * PS_LD + c] = __float2half(p);
            DSs[k_loc * PS_LD + c] = __float2half(ds);
            // dBias sees the gradient of the pre-softmax logit itself, unscaled:
            // the bias is added AFTER the q.k scaling.
            if constexpr (WANT_DBIAS) {
                if (ds != 0.f) {
                    atomicAdd(&dbias[bh + (long long)gqc * Sk + gkc], ds);
                }
            }
        }
        __syncwarp();

        // dV += P^T @ dO ; dK += dS^T @ Q
#pragma unroll
        for (int d = 0; d < D / FRAG; ++d) {
#pragma unroll
            for (int ks = 0; ks < BQ / FRAG; ++ks) {
                wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::row_major> fp, fds;
                wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, __half, wmma::row_major> fob, fqb;
                wmma::load_matrix_sync(fp, Ps + (warp * FRAG) * PS_LD + ks * FRAG, PS_LD);
                wmma::load_matrix_sync(fds, DSs + (warp * FRAG) * PS_LD + ks * FRAG, PS_LD);
                wmma::load_matrix_sync(fob, Os + (ks * FRAG) * D + d * FRAG, D);
                wmma::load_matrix_sync(fqb, Qs + (ks * FRAG) * D + d * FRAG, D);
                wmma::mma_sync(acc_dv[d], fp, fob, acc_dv[d]);
                wmma::mma_sync(acc_dk[d], fds, fqb, acc_dk[d]);
            }
        }

        // dQ wants dS, and DSs holds dS^T; a col_major fragment is that
        // transpose for free. Every key tile adds to the same rows, thus float32
        // and atomics.
        __syncthreads();
        for (int qb = warp; qb < BQ / FRAG; qb += NWARP) {
            wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> acc_dq[D / FRAG];
#pragma unroll
            for (int d = 0; d < D / FRAG; ++d) {
                wmma::fill_fragment(acc_dq[d], 0.0f);
            }
#pragma unroll
            for (int ks = 0; ks < BK / FRAG; ++ks) {
                wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::col_major> fds2;
                wmma::load_matrix_sync(fds2, DSs + (ks * FRAG) * PS_LD + qb * FRAG, PS_LD);
#pragma unroll
                for (int d = 0; d < D / FRAG; ++d) {
                    wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, __half, wmma::row_major> fkb;
                    wmma::load_matrix_sync(fkb, Ks + (ks * FRAG) * D + d * FRAG, D);
                    wmma::mma_sync(acc_dq[d], fds2, fkb, acc_dq[d]);
                }
            }
#pragma unroll
            for (int d = 0; d < D / FRAG; ++d) {
                wmma::store_matrix_sync(Ss + (warp * FRAG) * SS_LD + d * FRAG, acc_dq[d], SS_LD,
                                        wmma::mem_row_major);
            }
            __syncwarp();
            for (int i = lane; i < FRAG * DR; i += WARP) {
                const int r = i / DR, c = i - r * DR;
                const int gq = qq + qb * FRAG + r;
                if (gq < Sq) {
                    const float g = Ss[(warp * FRAG + r) * SS_LD + c] * sm_scale;
                    if (g != 0.f) {
                        atomicAdd(&dq_accum[qkv + (long long)gq * DR + c], g);
                    }
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }

    __syncwarp();
#pragma unroll
    for (int d = 0; d < D / FRAG; ++d) {
        wmma::store_matrix_sync(Ss + (warp * FRAG) * SS_LD + d * FRAG, acc_dk[d], SS_LD,
                                wmma::mem_row_major);
        wmma::store_matrix_sync(Ts + (warp * FRAG) * SS_LD + d * FRAG, acc_dv[d], SS_LD,
                                wmma::mem_row_major);
    }
    __syncwarp();
    for (int i = lane; i < FRAG * DR; i += WARP) {
        const int r = i / DR, c = i - r * DR;
        const int gk = k0 + warp * FRAG + r;
        if (gk < Sk) {
            dk[kv + (long long)gk * DR + c] =
                __float2half(Ss[(warp * FRAG + r) * SS_LD + c] * sm_scale);
            dv[kv + (long long)gk * DR + c] = __float2half(Ts[(warp * FRAG + r) * SS_LD + c]);
        }
    }
}

static __global__ void to_half(const float* __restrict__ src, __half* __restrict__ dst,
                               long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        dst[i] = __float2half(src[i]);
    }
}

static __global__ void zero_f32(float* p, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        p[i] = 0.f;
    }
}

template <typename A, typename B>
static bool same_shape(const A& a, const B& b) {
    return a.dimensions().size() == b.dimensions().size() &&
           std::equal(a.dimensions().begin(), a.dimensions().end(), b.dimensions().begin());
}

template <typename A>
static bool rows_are(const A& a, int N, int H, int S) {
    auto d = a.dimensions();
    return d.size() == 3 && d[0] == N && d[1] == H && d[2] == S;
}

template <typename A>
static bool bias_is(const A& a, int H, int Sq, int Sk) {
    auto d = a.dimensions();
    return d.size() == 3 && d[0] == H && d[1] == Sq && d[2] == Sk;
}

template <int D, int BQ, int BK, bool WANT_DBIAS>
static ffi::Error launch_bwd(cudaStream_t stream, int device, ffi::ScratchAllocator& scratch,
                             const __half* q, const __half* k, const __half* v,
                             const __half* bias, const uint8_t* kmask, const __half* dout,
                             const float* lse, const float* delta, __half* dq, __half* dk,
                             __half* dv, float* dbias, int N, int H, int Sq, int Sk, int DR,
                             float scale) {
    constexpr int SS_LD = (BQ > D ? BQ : D) + 4;
    constexpr size_t smem =
        (size_t)(2 * BK * SS_LD) * sizeof(float) +
        (size_t)(2 * BK * (BQ + 8) + 2 * BK * D + 2 * BQ * D) * sizeof(__half);
    const int max_smem = bwd_shared_limit(device);
    if ((int)smem > max_smem) {
        return ffi::Error::InvalidArgument("volta_wmma_bwd: needs " + std::to_string(smem / 1024) +
                                           " KB shared, device allows " +
                                           std::to_string(max_smem / 1024) + " KB");
    }

    // dQ lands from every key tile, thus it accumulates in float32 and is cast
    // once at the end. The scratch is XLA's, and it goes away with the call.
    const long long nq = (long long)N * H * Sq * DR;
    auto mem = scratch.Allocate(nq * sizeof(float), alignof(float));
    if (!mem.has_value()) {
        return ffi::Error::Internal("volta_wmma_bwd: no scratch for the dQ accumulator");
    }
    float* dq_accum = reinterpret_cast<float*>(*mem);
    zero_f32<<<256, 256, 0, stream>>>(dq_accum, nq);
    zero_f32<<<256, 256, 0, stream>>>(dbias, (long long)H * Sq * Sk);

    auto kern = volta_wmma_bwd_fused_kernel<D, BK, BQ, WANT_DBIAS>;
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    dim3 grid((Sk + BK - 1) / BK, H, N);
    kern<<<grid, (BK / FRAG) * WARP, smem, stream>>>(q, k, v, bias, kmask, dout, lse, delta,
                                                     dq_accum, dk, dv, dbias, N, H, Sq, Sk, DR,
                                                     scale);
    to_half<<<256, 256, 0, stream>>>(dq_accum, dq, nq);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_wmma_bwd launch: ") +
                                    cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

ffi::Error VoltaWmmaBwdImpl(cudaStream_t stream, int32_t device, ffi::ScratchAllocator scratch,
                            ffi::Buffer<ffi::DataType::F16> q,
                            ffi::Buffer<ffi::DataType::F16> k, ffi::Buffer<ffi::DataType::F16> v,
                            ffi::Buffer<ffi::DataType::F16> bias,
                            ffi::Buffer<ffi::DataType::U8> kmask,
                            ffi::Buffer<ffi::DataType::F16> dout,
                            ffi::Buffer<ffi::DataType::F32> lse,
                            ffi::Buffer<ffi::DataType::F32> delta,
                            ffi::Result<ffi::Buffer<ffi::DataType::F16>> dq,
                            ffi::Result<ffi::Buffer<ffi::DataType::F16>> dk,
                            ffi::Result<ffi::Buffer<ffi::DataType::F16>> dv,
                            ffi::Result<ffi::Buffer<ffi::DataType::F32>> dbias, float scale,
                            bool want_dbias) {
    auto d = q.dimensions();
    if (d.size() != 4) {
        return ffi::Error::InvalidArgument("q must be [N,H,S,D]");
    }
    const int N = (int)d[0], H = (int)d[1], Sq = (int)d[2], D = (int)d[3];
    const int Sk = (int)k.dimensions()[2];
    if (!same_shape(v, k) || !same_shape(dout, q)) {
        return ffi::Error::InvalidArgument("volta_wmma_bwd: v must match k, dout must match q");
    }
    if (!rows_are(lse, N, H, Sq) || !rows_are(delta, N, H, Sq)) {
        return ffi::Error::InvalidArgument("volta_wmma_bwd: lse and delta must be [N,H,Sq]");
    }
    if (!bias_is(bias, H, Sq, Sk) || !bias_is(*dbias, H, Sq, Sk)) {
        return ffi::Error::InvalidArgument("volta_wmma_bwd: bias and dbias must be [H,Sq,Sk]");
    }

    const __half* qp = reinterpret_cast<const __half*>(q.typed_data());
    const __half* kp = reinterpret_cast<const __half*>(k.typed_data());
    const __half* vp = reinterpret_cast<const __half*>(v.typed_data());
    const __half* bp = reinterpret_cast<const __half*>(bias.typed_data());
    const uint8_t* mp = kmask.typed_data();
    const __half* dop = reinterpret_cast<const __half*>(dout.typed_data());
    const float* lp = lse.typed_data();
    const float* dlp = delta.typed_data();
    __half* dqp = reinterpret_cast<__half*>(dq->typed_data());
    __half* dkp = reinterpret_cast<__half*>(dk->typed_data());
    __half* dvp = reinterpret_cast<__half*>(dv->typed_data());
    float* dbp = reinterpret_cast<float*>(dbias->typed_data());

    // The backward holds four tiles where the forward holds two, thus a tiling
    // of its own rather than the forward's: 32x32 measures best, except at head
    // 64 where a 64-row query tile is a third faster. 63 KB at most.
#define DISPATCH_BWD_T(TILE, DD, BQ)                                                               \
    if (D == (DD))                                                                                 \
        return want_dbias                                                                          \
                   ? launch_bwd<TILE, BQ, 32, true>(stream, device, scratch, qp, kp, vp, bp, mp,   \
                                                    dop, lp, dlp, dqp, dkp, dvp, dbp, N, H, Sq,   \
                                                    Sk, (DD), scale)                               \
                   : launch_bwd<TILE, BQ, 32, false>(stream, device, scratch, qp, kp, vp, bp, mp, \
                                                     dop, lp, dlp, dqp, dkp, dvp, dbp, N, H, Sq,  \
                                                     Sk, (DD), scale);
    // head 8 rides in a 16-wide tile, zero-padded, as it does in the forward.
    DISPATCH_BWD_T(16, 8, 32)
    DISPATCH_BWD_T(16, 16, 32)
    DISPATCH_BWD_T(32, 32, 32)
    DISPATCH_BWD_T(64, 64, 64)
#undef DISPATCH_BWD_T
    return ffi::Error::InvalidArgument("volta_wmma_bwd: unsupported head dim");
}

// NOT kCmdBufferCompatible: this handler starts three kernels, and the first
// one zeroes the dbias accumulator.
XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaWmmaBwd, VoltaWmmaBwdImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U8>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()
                                  .Attr<float>("scale")
                                  .Attr<bool>("want_dbias"));
