// Backward pass for volta_mma_attn.cu (sm_75+), with mma.sync.m16n8k8.
//
// The forward is flash attention: it never materialises the [Sq, Sk] softmax,
// so the backward cannot read it back and has to recompute it. What it does
// NOT recompute is the row statistic -- VoltaMmaFwd returns `lse` (log2 domain,
// m + log2(l)) alongside the output, which is enough to rebuild P in one pass
// instead of two.
//
// Inputs   q,k,v   [N,H,Sq|Sk,D] f16
//          bias    [H,Sq,Sk] f16        (shared across the batch)
//          kmask   [N,Sk] u8
//          dout    [N,H,Sq,D] f16
//          lse     [N,H,Sq] f32         from VoltaMmaFwd
//          delta   [N,H,Sq] f32         rowsum(dout * out); one line of XLA,
//                                       so it is not worth a kernel here
// Outputs  dq,dk,dv [same as q,k,v] f16
//          dbias   [H,Sq,Sk] f32        summed over the batch with atomics
//
// dBIAS IS THE REASON THIS EXISTS. ColabFold's AF2 path never needs it -- its
// pair bias is not on the gradient path -- but AlphaFold 3 reaches the pair
// representation THROUGH the attention bias, so an attention backward without
// dBias is useless to it.
//
// Two kernels, split the standard way so that only dbias needs atomics:
//   dq kernel:  one block owns 16 query rows per warp and loops over all keys.
//   dkdv kernel: one block owns 16 key rows per warp and loops over all
//                queries, working on the transposed problem.
//
// Fragment layout for m16n8k8 (CUTLASS, and the same in the forward):
//   A[m][k]: lane holds m = lr, lr+8   and k = lc, lc+1
//   B[k][n]: lane holds k = lc, lc+1   and n = lr
//   C[m][n]: lane holds m = lr, lr+8   and n = lc, lc+1
// where lr = lane>>2 and lc = (lane&3)*2. A and C share a layout, so a
// computed tile becomes the next GEMM's A operand with only a cast.

#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "volta_attn.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#define WARP 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 8

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
#error "volta_mma_attn_bwd.cu requires sm_75+."
#endif

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

using MmaOp =
    cutlass::arch::Mma<cutlass::gemm::GemmShape<MMA_M, MMA_N, MMA_K>, WARP, cutlass::half_t,
                       cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
                       float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;

using FragA = cutlass::Array<cutlass::half_t, 4>;
using FragB = cutlass::Array<cutlass::half_t, 2>;
using FragC = cutlass::Array<float, 4>;

// Stage a [rows, D] tile of a [N,H,S,D] tensor, zero-padded past the end.
__device__ inline void load_tile(__half* dst, const __half* src, long long base, int row0, int rows,
                                 int S, int D, int tid, int nthreads) {
    for (int i = tid; i < rows * D; i += nthreads) {
        int r = i / D, c = i - r * D;
        int g = row0 + r;
        dst[i] = (g < S) ? src[base + (long long)g * D + c] : __float2half(0.f);
    }
}

// -----------------------------------------------------------------------------
// dQ (and dBias): 16 query rows per warp, looping over every key.
// -----------------------------------------------------------------------------
template <int D, int BK, int BQ, bool WANT_DBIAS>
__global__ __launch_bounds__(BK / MMA_M * WARP) void volta_mma_bwd_fused_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ bias, const uint8_t* __restrict__ kmask,
    const __half* __restrict__ dout, const float* __restrict__ lse,
    const float* __restrict__ delta, float* __restrict__ dq_accum, __half* __restrict__ dk,
    __half* __restrict__ dv, float* __restrict__ dbias, int N, int H, int Sq, int Sk,
    float sm_scale, bool n_fastest) {
    constexpr int NWARP = BK / MMA_M;
    constexpr int NQ = BQ / MMA_N;
    constexpr int ND = D / MMA_N;
    constexpr int KSTEP = D / MMA_K;
    constexpr int QB = BQ / MMA_M;      // query row blocks, for the dQ gemm
    constexpr int DS_LD = BQ + 8;       // dS^T stride, padded off the banks

    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int tid = threadIdx.x;
    const int nthreads = NWARP * WARP;

    // The bias tile is shared across the batch and is the largest thing read, so
    // past a certain N the batch runs fastest to hold it in L2. Below that the
    // key tile does, which suits the dQ atomics.
    const int h = blockIdx.y;
    const int n = n_fastest ? blockIdx.x : blockIdx.z;
    const int ktile = n_fastest ? blockIdx.z : blockIdx.x;
    const int k0 = ktile * BK;
    if (k0 >= Sk) {
        return;
    }
    const int lr = lane >> 2;
    const int lc = (lane & 3) * 2;

    extern __shared__ char smem[];
    __half* Ks = reinterpret_cast<__half*>(smem);
    __half* Vs = Ks + BK * D;
    __half* Qs = Vs + BK * D;
    __half* Os = Qs + BQ * D;      // dout tile
    __half* Bs = Os + BQ * D;      // [BK][BQ], transposed
    __half* DSs = Bs + BK * BQ;    // [BK][DS_LD], dS^T for the dQ gemm
    float* Ls = reinterpret_cast<float*>(DSs + BK * DS_LD);   // lse   [BQ]
    float* Ds = Ls + BQ;                                  // delta [BQ]

    const long long qkv = (long long)(n * H + h) * Sq * D;
    const long long kv = (long long)(n * H + h) * Sk * D;
    const long long bh = (long long)h * Sq * Sk;
    const long long rowbase = (long long)(n * H + h) * Sq;

    load_tile(Ks, k, kv, k0, BK, Sk, D, tid, nthreads);
    load_tile(Vs, v, kv, k0, BK, Sk, D, tid, nthreads);
    __syncthreads();

    FragA fk[KSTEP], fv[KSTEP];
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks) {
        const int krow = warp * MMA_M + lr;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            fk[ks][0 + j] = reinterpret_cast<cutlass::half_t&>(Ks[krow * D + ks * MMA_K + lc + j]);
            fk[ks][2 + j] =
                reinterpret_cast<cutlass::half_t&>(Ks[(krow + 8) * D + ks * MMA_K + lc + j]);
            fv[ks][0 + j] = reinterpret_cast<cutlass::half_t&>(Vs[krow * D + ks * MMA_K + lc + j]);
            fv[ks][2 + j] =
                reinterpret_cast<cutlass::half_t&>(Vs[(krow + 8) * D + ks * MMA_K + lc + j]);
        }
    }

    // A key row that the mask kills contributes nothing to dK or dV.
    bool live[2];
#pragma unroll
    for (int half_i = 0; half_i < 2; ++half_i) {
        const int gk = k0 + warp * MMA_M + lr + half_i * 8;
        live[half_i] = (gk < Sk) && (kmask[(long long)n * Sk + gk] != 0);
    }

    FragC acc_dk[ND], acc_dv[ND];
#pragma unroll
    for (int d = 0; d < ND; ++d) {
        acc_dk[d].clear();
        acc_dv[d].clear();
    }

    MmaOp mma_op;
    const float qk_scale = sm_scale * LOG2E;

    for (int qq = 0; qq < Sq; qq += BQ) {
        __syncthreads();
        load_tile(Qs, q, qkv, qq, BQ, Sq, D, tid, nthreads);
        load_tile(Os, dout, qkv, qq, BQ, Sq, D, tid, nthreads);
        // bias tile, read along its rows so the loads coalesce, transposed into shared
        for (int i = tid; i < BQ * BK; i += nthreads) {
            int r = i / BK, c = i - r * BK;
            int gq = qq + r, gk = k0 + c;
            Bs[c * BQ + r] =
                (gq < Sq && gk < Sk) ? bias[bh + (long long)gq * Sk + gk] : __float2half(0.f);
        }
        for (int i = tid; i < BQ; i += nthreads) {
            int gq = qq + i;
            Ls[i] = (gq < Sq) ? lse[rowbase + gq] : 0.f;
            Ds[i] = (gq < Sq) ? delta[rowbase + gq] : 0.f;
        }
        __syncthreads();

        // S^T = K @ Q^T and dP^T = V @ dO^T
        FragC st[NQ], dpt[NQ];
#pragma unroll
        for (int nt = 0; nt < NQ; ++nt) {
            st[nt].clear();
            dpt[nt].clear();
#pragma unroll
            for (int ks = 0; ks < KSTEP; ++ks) {
                FragB fqb, fob;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fqb[j] = reinterpret_cast<cutlass::half_t&>(
                        Qs[(nt * MMA_N + lr) * D + ks * MMA_K + lc + j]);
                    fob[j] = reinterpret_cast<cutlass::half_t&>(
                        Os[(nt * MMA_N + lr) * D + ks * MMA_K + lc + j]);
                }
                mma_op(st[nt], fk[ks], fqb, st[nt]);
                mma_op(dpt[nt], fv[ks], fob, dpt[nt]);
            }
        }

        FragA pt[NQ], dst[NQ];
#pragma unroll
        for (int nt = 0; nt < NQ; ++nt) {
#pragma unroll
            for (int half_i = 0; half_i < 2; ++half_i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int idx = half_i * 2 + j;
                    const int k_loc = warp * MMA_M + lr + half_i * 8;
                    const int q_loc = nt * MMA_N + lc + j;
                    const int gq = qq + q_loc;
                    const bool alive = live[half_i] && gq < Sq;
                    const float l2 = alive ? (st[nt][idx] * qk_scale +
                                              __half2float(Bs[k_loc * BQ + q_loc]) * LOG2E)
                                           : (MASKED_LOGIT * LOG2E);
                    // P carries dV even where the mask clamped the logit, but dS
                    // does not: a constant logit has no gradient
                    const float p = exp2f(l2 - Ls[q_loc]);
                    const float g = alive ? p * (dpt[nt][idx] - Ds[q_loc]) : 0.f;
                    pt[nt][idx] = cutlass::half_t(p);
                    dst[nt][idx] = cutlass::half_t(g);
                    DSs[k_loc * DS_LD + q_loc] = __float2half(g);
                    // dBias sees the gradient of the pre-softmax logit itself,
                    // unscaled: the bias is added AFTER the q.k scaling.
                    if constexpr (WANT_DBIAS) {
                        if (g != 0.f) {
                            atomicAdd(&dbias[bh + (long long)gq * Sk + k0 + k_loc], g);
                        }
                    }
                }
            }
        }

        // dV += P^T @ dO ; dK += dS^T @ Q
#pragma unroll
        for (int d = 0; d < ND; ++d) {
#pragma unroll
            for (int nt = 0; nt < NQ; ++nt) {
                FragB fo, fq2;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    fo[j] = reinterpret_cast<cutlass::half_t&>(
                        Os[(nt * MMA_N + lc + j) * D + d * MMA_N + lr]);
                    fq2[j] = reinterpret_cast<cutlass::half_t&>(
                        Qs[(nt * MMA_N + lc + j) * D + d * MMA_N + lr]);
                }
                mma_op(acc_dv[d], pt[nt], fo, acc_dv[d]);
                mma_op(acc_dk[d], dst[nt], fq2, acc_dk[d]);
            }
        }

        // dQ needs dS, not dS^T, and the two layouts differ by a transpose that
        // only shared memory can do. DSs took the tile as it was computed.
        __syncthreads();

        for (int qb = warp; qb < QB; qb += NWARP) {
            FragC acc_dq[ND];
#pragma unroll
            for (int d = 0; d < ND; ++d) {
                acc_dq[d].clear();
            }
#pragma unroll
            for (int kstep = 0; kstep < BK / MMA_K; ++kstep) {
                FragA fds;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int krow = kstep * MMA_K + lc + j;
                    fds[0 + j] = reinterpret_cast<cutlass::half_t&>(
                        DSs[krow * DS_LD + qb * MMA_M + lr]);
                    fds[2 + j] = reinterpret_cast<cutlass::half_t&>(
                        DSs[krow * DS_LD + qb * MMA_M + lr + 8]);
                }
#pragma unroll
                for (int d = 0; d < ND; ++d) {
                    FragB fkb;
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        fkb[j] = reinterpret_cast<cutlass::half_t&>(
                            Ks[(kstep * MMA_K + lc + j) * D + d * MMA_N + lr]);
                    }
                    mma_op(acc_dq[d], fds, fkb, acc_dq[d]);
                }
            }
#pragma unroll
            for (int d = 0; d < ND; ++d) {
#pragma unroll
                for (int half_i = 0; half_i < 2; ++half_i) {
                    const int gq = qq + qb * MMA_M + lr + half_i * 8;
                    if (gq >= Sq) {
                        continue;
                    }
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const float g = acc_dq[d][half_i * 2 + j] * sm_scale;
                        if (g != 0.f) {
                            atomicAdd(&dq_accum[qkv + (long long)gq * D + d * MMA_N + lc + j], g);
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int d = 0; d < ND; ++d) {
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int gk = k0 + warp * MMA_M + lr + half_i * 8;
            if (gk >= Sk) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = d * MMA_N + lc + j;
                const int idx = half_i * 2 + j;
                dk[kv + (long long)gk * D + col] = __float2half(acc_dk[d][idx] * sm_scale);
                dv[kv + (long long)gk * D + col] = __float2half(acc_dv[d][idx]);
            }
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
                             __half* dv, float* dbias, int N, int H, int Sq, int Sk, float scale) {
    constexpr size_t smem = (size_t)(2 * BK * D + 2 * BQ * D + BK * BQ + BK * (BQ + 8)) *
                                sizeof(__half) +
                            2 * BQ * sizeof(float);
    const int max_smem = bwd_shared_limit(device);
    if ((int)smem > max_smem) {
        return ffi::Error::InvalidArgument("volta_mma_bwd: needs " + std::to_string(smem / 1024) +
                                           " KB shared, device allows " +
                                           std::to_string(max_smem / 1024) + " KB");
    }

    // dQ lands from every key tile, thus it accumulates in float32 and is cast
    // once at the end. The scratch is XLA's, and it goes away with the call.
    const long long nq = (long long)N * H * Sq * D;
    auto mem = scratch.Allocate(nq * sizeof(float), alignof(float));
    if (!mem.has_value()) {
        return ffi::Error::Internal("volta_mma_bwd: no scratch for the dQ accumulator");
    }
    float* dq_accum = reinterpret_cast<float*>(*mem);
    zero_f32<<<256, 256, 0, stream>>>(dq_accum, nq);
    zero_f32<<<256, 256, 0, stream>>>(dbias, (long long)H * Sq * Sk);

    auto kern = volta_mma_bwd_fused_kernel<D, BK, BQ, WANT_DBIAS>;
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    const int nkt = (Sk + BK - 1) / BK;
    const bool n_fastest = N >= 64;
    dim3 grid = n_fastest ? dim3(N, H, nkt) : dim3(nkt, H, N);
    kern<<<grid, (BK / MMA_M) * WARP, smem, stream>>>(q, k, v, bias, kmask, dout, lse, delta,
                                                      dq_accum, dk, dv, dbias, N, H, Sq, Sk,
                                                      scale, n_fastest);
    to_half<<<256, 256, 0, stream>>>(dq_accum, dq, nq);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_mma_bwd launch: ") +
                                    cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

ffi::Error VoltaMmaBwdImpl(cudaStream_t stream, int32_t device, ffi::ScratchAllocator scratch,
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
        return ffi::Error::InvalidArgument("volta_mma_bwd: v must match k, dout must match q");
    }
    if (!rows_are(lse, N, H, Sq) || !rows_are(delta, N, H, Sq)) {
        return ffi::Error::InvalidArgument("volta_mma_bwd: lse and delta must be [N,H,Sq]");
    }
    if (!bias_is(bias, H, Sq, Sk) || !bias_is(*dbias, H, Sq, Sk)) {
        return ffi::Error::InvalidArgument("volta_mma_bwd: bias and dbias must be [H,Sq,Sk]");
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
    // of its own rather than the forward's: 32x32 measures best at every head
    // dim, and needs 18 KB at most, which every card it targets can give.
#define DISPATCH_BWD(DD)                                                                           \
    if (D == (DD))                                                                                 \
        return want_dbias                                                                          \
                   ? launch_bwd<DD, 32, 32, true>(stream, device, scratch, qp, kp, vp, bp, mp,    \
                                                  dop, lp, dlp, dqp, dkp, dvp, dbp, N, H, Sq, Sk, \
                                                  scale)                                           \
                   : launch_bwd<DD, 32, 32, false>(stream, device, scratch, qp, kp, vp, bp, mp,   \
                                                   dop, lp, dlp, dqp, dkp, dvp, dbp, N, H, Sq,   \
                                                   Sk, scale);
    DISPATCH_BWD(8)
    DISPATCH_BWD(16)
    DISPATCH_BWD(32)
    DISPATCH_BWD(64)
#undef DISPATCH_BWD
    return ffi::Error::InvalidArgument("volta_mma_bwd: unsupported head dim");
}

// NOT kCmdBufferCompatible: this handler starts three kernels, and the first
// one zeroes the dbias accumulator.
XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaMmaBwd, VoltaMmaBwdImpl,
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
