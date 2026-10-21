// Flash attention for Turing and later (sm_75+), with CUTLASS mma.sync.m16n8k8.
// The accumulator and the operand A have the same fragment layout.
// Thus the softmax stays in registers. Volta cannot do this.
// Inputs: q,k,v [N,H,S,D] f16 (or strided, see AttnLayout), bias [H,Sq,Sk] f16, kmask [N,Sk] u8.

#include <cuda_fp16.h>
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

#define MMA_M 16
#define MMA_N 8
#define MMA_K 8

// CUTLASS gives no error for sm_70, but it builds a kernel with no tensor cores.
// Thus stop the build here.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
#error "volta_mma_attn.cu requires sm_75+. Volta (sm_70) needs an m8n8k4 port."
#endif

#define MAX_DEVICES 16
static int device_shared_limit(int device) {
    static int cache[MAX_DEVICES] = {};
    if (device < 0 || device >= MAX_DEVICES) {
        return 0;
    }
    if (cache[device] == 0) {
        cudaDeviceGetAttribute(&cache[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    }
    return cache[device];
}

// Tell if the kernel must get the opt-in shared memory attribute on this device.
static bool needs_smem_optin(int device, const void* fn) {
    static const void* cache[MAX_DEVICES] = {};
    if (device < 0 || device >= MAX_DEVICES) {
        return true;
    }
    if (cache[device] == fn) {
        return false;
    }
    cache[device] = fn;
    return true;
}

using MmaOp =
    cutlass::arch::Mma<cutlass::gemm::GemmShape<MMA_M, MMA_N, MMA_K>, WARP, cutlass::half_t,
                       cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
                       float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>;

using FragA = cutlass::Array<cutlass::half_t, 4>;
using FragB = cutlass::Array<cutlass::half_t, 2>;
using FragC = cutlass::Array<float, 4>;

// 16 bytes from global memory, prefetching the 128-byte L2 sector (sm_75's ld.global .L2::128B)
__device__ __forceinline__ uint4 ldg16(const __half* p) {
    uint4 r;
    asm volatile("ld.global.nc.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p));
    return r;
}

// ldmatrix: N 8x8 b16 matrices from shared memory, lanes 8m..8m+7 giving matrix m's row addresses; a lane gets row
// lane/4, columns 2*(lane%4)+{0,1} of each (TRANS: of its transpose) -- an m16n8k8 B fragment, or a C fragment's half
template <int N, bool TRANS>
__device__ __forceinline__ void ldsm(uint32_t* r, const __half* p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    if constexpr (N == 4 && TRANS) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    } else if constexpr (N == 4) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    } else if constexpr (N == 2 && TRANS) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
    } else if constexpr (N == 2) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
    } else if constexpr (TRANS) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(r[0]) : "r"(a));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r[0]) : "r"(a));
    }
}

// the CNT 8x8 blocks side by side in the 8 rows at base (row stride in halves): CNT fragments, 4 per ldmatrix
template <int CNT, bool TRANS>
__device__ __forceinline__ void ldsm_blocks(uint32_t* r, const __half* base, int stride, int lane) {
    const __half* row = base + (lane & 7) * stride;
#pragma unroll
    for (int g = 0; g < CNT / 4; ++g) {
        ldsm<4, TRANS>(r + 4 * g, row + (4 * g + lane / 8) * 8);
    }
    constexpr int G = CNT / 4 * 4;
    if constexpr (CNT % 4 >= 2) {
        ldsm<2, TRANS>(r + G, row + (G + (lane / 8) % 2) * 8);
    }
    if constexpr (CNT % 2 == 1) {
        ldsm<1, TRANS>(r + CNT - 1, row + (CNT - 1) * 8);
    }
}

// BQ must be a multiple of 16 (one warp owns MMA_M=16 query rows).
// BK and D must be multiples of 8.
template <int D, int BQ, int BK, bool WANT_LSE>
__global__ __launch_bounds__(BQ / MMA_M * WARP) void volta_mma_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ bias, const uint8_t* __restrict__ kmask, __half* __restrict__ out,
    float* __restrict__ lse, int N, int H, int Sq, int Sk, float sm_scale, AttnLayout L) {
    constexpr int NWARP = BQ / MMA_M;
    constexpr int NK = BK / MMA_N;   // key sub-tiles per BK tile
    constexpr int ND = D / MMA_N;    // output dim sub-tiles
    constexpr int KSTEP = D / MMA_K; // contraction steps for Q@K^T
    // shared rows padded by 16 bytes: a fragment's 8 rows fall in different banks (a 64-byte stride put them in 2)
    constexpr int DS = D + 8, BS = BK + 8;

    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int tid = threadIdx.x;
    const int nthreads = NWARP * WARP;

    const int qtile = blockIdx.x, h = blockIdx.y, n = blockIdx.z;
    const int q0 = qtile * BQ;
    if (q0 >= Sq) {
        return;
    }

    // lane -> (row pair, column pair) for every fragment we touch
    const int lr = lane >> 2;                 // 0..7   : row within the 16-row tile
    const int lc = (lane & 3) * 2;            // 0,2,4,6: first of the lane's column pair
    const int row_a = q0 + warp * MMA_M + lr; // this lane's first query row
    const int row_b = row_a + 8;              // ... and its second

    extern __shared__ char smem[];
    __half* Qs = reinterpret_cast<__half*>(smem); // [BQ][D]
    __half* Ks = Qs + BQ * D;                     // [BK][DS]
    __half* Vs = Ks + BK * DS;                    // [BK][DS]
    __half* Bs = Vs + BK * DS;                    // [BQ][BS] bias tile

    const long long qb = n * L.qn + h * L.qh, kb = n * L.kn + h * L.kh, ob = n * L.on + h * L.oh;
    const long long bh = (long long)h * Sq * Sk;

    int k_end = Sk;
    if (L.lims != nullptr) {
        k_end = L.lims[n];
        if (k_end == 0 || q0 >= L.lims[N]) {
            for (int i = tid; i < BQ * D; i += nthreads) {
                const int gq = q0 + i / D;
                if (gq < Sq) {
                    out[ob + gq * L.os + i % D] = __float2half(0.f);
                }
            }
            return;
        }
    }

    for (int i = tid; i < BQ * D; i += nthreads) {
        int r = i / D, c = i - r * D;
        int gq = q0 + r;
        Qs[i] = (gq < Sq) ? q[qb + gq * L.qs + c] : __float2half(0.f);
    }
    __syncthreads();

    // Q operand fragments: fixed for this warp, so hoist them out of the k-loop.
    FragA fq[KSTEP];
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks) {
        const int qrow = warp * MMA_M + lr;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            fq[ks][0 + j] = reinterpret_cast<cutlass::half_t&>(Qs[(qrow)*D + ks * MMA_K + lc + j]);
            fq[ks][2 + j] =
                reinterpret_cast<cutlass::half_t&>(Qs[(qrow + 8) * D + ks * MMA_K + lc + j]);
        }
    }

    // Running online-softmax state and output accumulator -- all in registers.
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.f, 0.f};
    FragC acc[ND];
#pragma unroll
    for (int d = 0; d < ND; ++d) {
        acc[d].clear();
    }

    MmaOp mma_op;
    const float qk_scale = sm_scale * LOG2E; // fold log2(e) so we can use exp2
    // 16-byte staging where rows are 16-byte aligned (the packed layout's, and the bias at a length padded to 8)
    const auto aligned = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    const bool vec_kv = L.ks % 8 == 0 && kb % 8 == 0 && aligned(k) && aligned(v);
    const bool vec_b = Sk % 8 == 0 && aligned(bias);

    // both 16-byte paths: the next tile's K, V and bias are loaded into registers while this one computes, so the
    // loads' latency overlaps the MMAs instead of stalling every thread between the two barriers
    constexpr int NT = NWARP * WARP, NKV = (BK * D / 8 + NT - 1) / NT, NB = (BQ * BK / 8 + NT - 1) / NT;
    const bool pipe = vec_kv && vec_b;
    uint4 kreg[NKV], vreg[NKV], breg[NB];
    const auto fetch = [&](int k0) {
#pragma unroll
        for (int i = 0; i < NKV; ++i) {
            const int x = tid + i * NT, r = x / (D / 8), c = (x - r * (D / 8)) * 8, gk = k0 + r;
            const bool ok = x < BK * D / 8 && gk < Sk;
            kreg[i] = ok ? ldg16(k + kb + gk * L.ks + c) : make_uint4(0, 0, 0, 0);
            vreg[i] = ok ? ldg16(v + kb + gk * L.ks + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int x = tid + i * NT, r = x / (BK / 8), c = (x - r * (BK / 8)) * 8, gq = q0 + r, gk = k0 + c;
            const bool ok = x < BQ * BK / 8 && gq < Sq && gk < Sk;
            breg[i] = ok ? ldg16(bias + bh + (long long)gq * Sk + gk) : make_uint4(0, 0, 0, 0);
        }
    };
    const auto store = [&]() {
#pragma unroll
        for (int i = 0; i < NKV; ++i) {
            const int x = tid + i * NT, r = x / (D / 8), c = (x - r * (D / 8)) * 8;
            if (x < BK * D / 8) {
                *reinterpret_cast<uint4*>(Ks + r * DS + c) = kreg[i];
                *reinterpret_cast<uint4*>(Vs + r * DS + c) = vreg[i];
            }
        }
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            const int x = tid + i * NT, r = x / (BK / 8), c = (x - r * (BK / 8)) * 8;
            if (x < BQ * BK / 8) {
                *reinterpret_cast<uint4*>(Bs + r * BS + c) = breg[i];
            }
        }
    };
    if (pipe) {
        fetch(0);
    }

    for (int k0 = 0; k0 < k_end; k0 += BK) {
        __syncthreads();
        if (pipe) {
            store();
        } else {
            if (vec_kv) {
                for (int i = tid; i < BK * D / 8; i += nthreads) {
                    const int r = i / (D / 8), c = (i - r * (D / 8)) * 8, gk = k0 + r;
                    uint4 kx = make_uint4(0, 0, 0, 0), vx = kx;
                    if (gk < Sk) {
                        kx = ldg16(k + kb + gk * L.ks + c);
                        vx = ldg16(v + kb + gk * L.ks + c);
                    }
                    *reinterpret_cast<uint4*>(Ks + r * DS + c) = kx;
                    *reinterpret_cast<uint4*>(Vs + r * DS + c) = vx;
                }
            } else {
                for (int i = tid; i < BK * D; i += nthreads) {
                    int r = i / D, c = i - r * D;
                    int gk = k0 + r;
                    bool ok = (gk < Sk);
                    Ks[r * DS + c] = ok ? k[kb + gk * L.ks + c] : __float2half(0.f);
                    Vs[r * DS + c] = ok ? v[kb + gk * L.ks + c] : __float2half(0.f);
                }
            }
            if (vec_b) {
                for (int i = tid; i < BQ * BK / 8; i += nthreads) {
                    const int r = i / (BK / 8), c = (i - r * (BK / 8)) * 8, gq = q0 + r, gk = k0 + c;
                    const uint4 bx = (gq < Sq && gk < Sk) ? ldg16(bias + bh + (long long)gq * Sk + gk)
                                                          : make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(Bs + r * BS + c) = bx;
                }
            } else {
                for (int i = tid; i < BQ * BK; i += nthreads) { // coalesced bias staging
                    int r = i / BK, c = i - r * BK;
                    int gq = q0 + r, gk = k0 + c;
                    Bs[r * BS + c] = (gq < Sq && gk < Sk) ? bias[bh + (long long)gq * Sk + gk] : __float2half(0.f);
                }
            }
        }
        __syncthreads();
        if (pipe && k0 + BK < k_end) {
            fetch(k0 + BK);
        }

        // S = Q @ K^T, then softmax, entirely in registers
        FragC s[NK];
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
            s[nt].clear();
            // B[k][n] = K[key][dim]: the 8 keys' rows, one 8x8 block per contraction step
            uint32_t kf[KSTEP];
            ldsm_blocks<KSTEP, false>(kf, Ks + nt * MMA_N * DS, DS, lane);
#pragma unroll
            for (int ks = 0; ks < KSTEP; ++ks) {
                mma_op(s[nt], fq[ks], reinterpret_cast<FragB&>(kf[ks]), s[nt]);
            }
        }

        // scale + bias + mask without branches: a masked key takes MASKED_LOGIT, a key past Sk -inf
        float m_tile[2] = {-INFINITY, -INFINITY};
        // the bias of this warp's 16 rows, 2 key sub-tiles per ldmatrix: bf[2 nt + half_i] holds columns lc, lc+1
        static_assert(NK % 2 == 0, "bias fragments load 2 key sub-tiles at a time");
        uint32_t bf[2 * NK];
#pragma unroll
        for (int nt = 0; nt < NK; nt += 2) {
            const int m = lane / 8; // matrix m: key sub-tile nt + m/2, rows half m%2
            ldsm<4, false>(bf + 2 * nt, Bs + (warp * MMA_M + (m % 2) * 8 + (lane & 7)) * BS + (nt + m / 2) * MMA_N);
        }
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
            const int col0 = k0 + nt * MMA_N + lc;
            // the two keys' mask bytes in one load where they are 2-byte aligned
            uint32_t mk = 0;
            if (Sk % 2 == 0 && col0 + 1 < Sk) {
                mk = *reinterpret_cast<const uint16_t*>(kmask + (long long)n * Sk + col0);
            } else {
                mk = (col0 < Sk ? kmask[(long long)n * Sk + col0] : 0) |
                     (col0 + 1 < Sk ? kmask[(long long)n * Sk + col0 + 1] : 0) << 8;
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = col0 + j;
                const bool in = col < Sk;
                const bool keep = in && ((mk >> (8 * j)) & 0xff) != 0;
                const float fill = in ? MASKED_LOGIT * LOG2E : -INFINITY;
#pragma unroll
                for (int half_i = 0; half_i < 2; ++half_i) {
                    const __half b = reinterpret_cast<const __half*>(&bf[2 * nt + half_i])[j];
                    const float val = s[nt][half_i * 2 + j] * qk_scale + __half2float(b) * LOG2E;
                    s[nt][half_i * 2 + j] = keep ? val : fill;
                    m_tile[half_i] = fmaxf(m_tile[half_i], s[nt][half_i * 2 + j]);
                }
            }
        }
// the 8 columns of a row live in lanes {4r..4r+3}
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            m_tile[half_i] = fmaxf(m_tile[half_i], __shfl_xor_sync(0xffffffffu, m_tile[half_i], 1));
            m_tile[half_i] = fmaxf(m_tile[half_i], __shfl_xor_sync(0xffffffffu, m_tile[half_i], 2));
        }

        // the running max moves only when a tile raises it past 2^6 (FlashMLA's rule, decided per warp): otherwise O and
        // l keep their scale and this tile's exp2 stays below 2^6, well inside fp16 P and the fp32 sums
        const bool raise = __any_sync(0xffffffffu, m_tile[0] > m_run[0] + 6.f || m_tile[1] > m_run[1] + 6.f);
        float m_new[2], alpha[2], l_tile[2] = {0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            m_new[i] = raise ? fmaxf(m_run[i], m_tile[i]) : m_run[i];
            alpha[i] = raise ? ex2_ftz(m_run[i] - m_new[i]) : 1.f; // 0 on the first tile
        }

        // P = exp2(S - m_new), converted in place to the A operand of P@V
        FragA p[NK];
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
#pragma unroll
            for (int half_i = 0; half_i < 2; ++half_i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float e = ex2_ftz(s[nt][half_i * 2 + j] - m_new[half_i]);
                    l_tile[half_i] += e;
                    p[nt][half_i * 2 + j] = cutlass::half_t(e);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            l_tile[i] += __shfl_xor_sync(0xffffffffu, l_tile[i], 1);
            l_tile[i] += __shfl_xor_sync(0xffffffffu, l_tile[i], 2);
            l_run[i] = l_run[i] * alpha[i] + l_tile[i];
            m_run[i] = m_new[i];
        }

// O = O*alpha + P @ V
        if (raise) {
#pragma unroll
            for (int d = 0; d < ND; ++d)
#pragma unroll
                for (int half_i = 0; half_i < 2; ++half_i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        acc[d][half_i * 2 + j] *= alpha[half_i];
                    }
        }
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
            // B[k][n] = V[key][dim]: the transpose of the 8 keys' rows, one 8x8 block per output sub-tile
            uint32_t vf[ND];
            ldsm_blocks<ND, true>(vf, Vs + nt * MMA_N * DS, DS, lane);
#pragma unroll
            for (int d = 0; d < ND; ++d) {
                mma_op(acc[d], p[nt], reinterpret_cast<FragB&>(vf[d]), acc[d]);
            }
        }
    }

    // softmax statistic, for the backward pass. Every lane of a row group
    // holds the same reduced l_run/m_run, so one lane per group writes.
    if constexpr (WANT_LSE) {
        if ((lane & 3) == 0) {
#pragma unroll
            for (int half_i = 0; half_i < 2; ++half_i) {
                const int gq = (half_i == 0) ? row_a : row_b;
                if (gq < Sq) {
                    const float l = l_run[half_i];
                    lse[(long long)(n * H + h) * Sq + gq] =
                        (l > 0.f) ? (m_run[half_i] + log2f(l)) : -INFINITY;
                }
            }
        }
    }

// normalise and store
#pragma unroll
    for (int d = 0; d < ND; ++d) {
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const int gq = (half_i == 0) ? row_a : row_b;
            if (gq >= Sq) {
                continue;
            }
            const float inv = 1.0f / fmaxf(l_run[half_i], 1e-30f);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = d * MMA_N + lc + j;
                out[ob + gq * L.os + col] = __float2half(
                    L.apply_gate(acc[d][half_i * 2 + j] * inv, qb + gq * L.qs + col, h, col, D));
            }
        }
    }
}

//
template <int D, int BQ, int BK, bool WANT_LSE>
static ffi::Error launch(cudaStream_t stream, int device, const __half* q, const __half* k,
                         const __half* v, const __half* bias, const uint8_t* kmask, __half* out,
                         float* lse, int N, int H, int Sq, int Sk, float scale,
                         const AttnLayout& L) {
    constexpr int NWARP = BQ / MMA_M;
    constexpr size_t smem = (size_t)(BQ * D + 2 * BK * (D + 8) + BQ * (BK + 8)) * sizeof(__half);
    auto kern = volta_mma_kernel<D, BQ, BK, WANT_LSE>;

    // Read the device limit one time only.
    // The handler must do no host work, or XLA cannot put it in a CUDA graph.
    const int max_smem = device_shared_limit(device);
    if ((int)smem > max_smem) {
        return ffi::Error::InvalidArgument("volta_mma: needs " + std::to_string(smem / 1024) +
                                           " KB shared, device allows " +
                                           std::to_string(max_smem / 1024) + " KB");
    }
    if (smem > 48 * 1024 && needs_smem_optin(device, (const void*)kern)) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    dim3 grid((Sq + BQ - 1) / BQ, H, N);
    kern<<<grid, NWARP * WARP, smem, stream>>>(q, k, v, bias, kmask, out, lse, N, H, Sq, Sk,
                                               scale, L);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_mma launch: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

template <bool WANT_LSE>
static ffi::Error volta_mma_common(cudaStream_t stream, int32_t device, const __half* qp,
                                   const __half* kp, const __half* vp, const __half* bp,
                                   const uint8_t* mp, __half* op, float* lse, int N, int H, int Sq,
                                   int Sk, int D, const AttnLayout& L, float scale,
                                   int64_t block_q, int64_t block_k) {
#define DISPATCH(DD, BQ, BK)                                                                       \
    if (D == (DD) && block_q == (BQ) && block_k == (BK))                                           \
        return launch<DD, BQ, BK, WANT_LSE>(stream, device, qp, kp, vp, bp, mp, op, lse, N, H,   \
                                            Sq, Sk, scale, L);
    DISPATCH(8, 64, 64)
    DISPATCH(8, 64, 32)
    DISPATCH(8, 32, 32)
    DISPATCH(8, 128, 64) DISPATCH(8, 32, 64) DISPATCH(32, 64, 64) DISPATCH(32, 64, 32)
        DISPATCH(32, 32, 64) DISPATCH(32, 32, 32) DISPATCH(32, 128, 64) DISPATCH(32, 16, 64)
            DISPATCH(16, 64, 64) DISPATCH(16, 32, 32) DISPATCH(16, 64, 32) DISPATCH(64, 64, 64)
                DISPATCH(64, 32, 32)
    // af3 head dims: single 384/16, diffusion token 768/16, protenix2's 96
    DISPATCH(24, 64, 32) DISPATCH(24, 32, 32) DISPATCH(48, 64, 32) DISPATCH(48, 32, 32)
        DISPATCH(96, 64, 32) DISPATCH(96, 32, 32)
#undef DISPATCH
                    return ffi::Error::InvalidArgument("volta_mma: unsupported (D, bq, bk)");
}

// lse is the softmax statistic the backward needs. With want_lse false the
// store compiles out and the buffer goes untouched, so one element is enough.
ffi::Error VoltaMmaImpl(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::DataType::F16> q,
                        ffi::Buffer<ffi::DataType::F16> k, ffi::Buffer<ffi::DataType::F16> v,
                        ffi::Buffer<ffi::DataType::F16> bias, ffi::Buffer<ffi::DataType::U8> kmask,
                        ffi::Result<ffi::Buffer<ffi::DataType::F16>> out,
                        ffi::Result<ffi::Buffer<ffi::DataType::F32>> lse, float scale,
                        int64_t block_q, int64_t block_k, bool want_lse) {
    auto d = q.dimensions();
    if (d.size() != 4) {
        return ffi::Error::InvalidArgument("q must be [N,H,S,D]");
    }
    const int N = (int)d[0], H = (int)d[1], Sq = (int)d[2], D = (int)d[3];
    const int Sk = (int)k.dimensions()[2];
    const auto L = AttnLayout::contiguous(H, Sq, Sk, D);
    auto f16 = [](auto& b) { return reinterpret_cast<const __half*>(b.typed_data()); };
    __half* op = reinterpret_cast<__half*>(out->typed_data());
    return (want_lse ? volta_mma_common<true> : volta_mma_common<false>)(
        stream, device, f16(q), f16(k), f16(v), f16(bias), kmask.typed_data(), op,
        want_lse ? lse->typed_data() : nullptr, N, H, Sq, Sk, D, L, scale, block_q, block_k);
}

// Forward only: q/k/v (and the gate when G == 4) read in place from p [X, Y, G, H, D], out
// [X, Y, H, D]. Rows attend along Y; with column set, columns attend along X.
static ffi::Error VoltaMmaPackedImpl(const int* lims, cudaStream_t stream, int32_t device,
                              ffi::Buffer<ffi::DataType::F16> p,
                              ffi::Buffer<ffi::DataType::F16> bias,
                              ffi::Buffer<ffi::DataType::U8> kmask,
                              ffi::Buffer<ffi::DataType::F16> gate_bias,
                              ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, float scale,
                              int64_t block_q, int64_t block_k, bool column) {
    auto d = p.dimensions();
    if (d.size() != 5 || (d[2] != 3 && d[2] != 4)) {
        return ffi::Error::InvalidArgument("p must be [X,Y,3 or 4,H,D]");
    }
    const int X = (int)d[0], Y = (int)d[1], G = (int)d[2], H = (int)d[3], D = (int)d[4];
    const __half* pp = reinterpret_cast<const __half*>(p.typed_data());
    auto L = AttnLayout::packed(Y, G, H, D, column);
    if (G == 4) {
        L.gate = pp + 3 * H * D;
        L.gate_bias = reinterpret_cast<const __half*>(gate_bias.typed_data());
    }
    L.lims = lims;
    const int N = column ? Y : X, S = column ? X : Y;
    if (bias.element_count() != (size_t)H * S * S || kmask.element_count() != (size_t)N * S ||
        (G == 4 && gate_bias.element_count() != (size_t)H * D)) {
        return ffi::Error::InvalidArgument("bias must be [H,S,S], kmask [N,S], gate_bias [H,D]");
    }
    return volta_mma_common<false>(stream, device, pp, pp + H * D, pp + 2 * H * D,
                                   reinterpret_cast<const __half*>(bias.typed_data()),
                                   kmask.typed_data(),
                                   reinterpret_cast<__half*>(out->typed_data()), nullptr, N, H, S,
                                   S, D, L, scale, block_q, block_k);
}

static ffi::Error VoltaMmaPackedNoTrim(cudaStream_t stream, int32_t device,
                    ffi::Buffer<ffi::DataType::F16> p, ffi::Buffer<ffi::DataType::F16> bias,
                    ffi::Buffer<ffi::DataType::U8> kmask, ffi::Buffer<ffi::DataType::F16> gate_bias,
                    ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, float scale, int64_t block_q,
                    int64_t block_k, bool column) {
    return VoltaMmaPackedImpl(nullptr, stream, device, p, bias, kmask, gate_bias, out, scale, block_q, block_k, column);
}

// VoltaMmaPacked with trim: lims [N + 1] from packed_attn.trim_limits(kmask, 1)
static ffi::Error VoltaMmaPackedTrimImpl(cudaStream_t stream, int32_t device,
                    ffi::Buffer<ffi::DataType::F16> p, ffi::Buffer<ffi::DataType::F16> bias,
                    ffi::Buffer<ffi::DataType::U8> kmask, ffi::Buffer<ffi::DataType::F16> gate_bias, ffi::Buffer<ffi::DataType::S32> lims,
                    ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, float scale, int64_t block_q,
                    int64_t block_k, bool column) {
    return VoltaMmaPackedImpl(lims.typed_data(), stream, device, p, bias, kmask, gate_bias, out, scale, block_q, block_k, column);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaMmaPacked, VoltaMmaPackedNoTrim,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U8>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Attr<float>("scale")
                                  .Attr<int64_t>("block_q")
                                  .Attr<int64_t>("block_k")
                                  .Attr<bool>("column"),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaMmaPackedTrim, VoltaMmaPackedTrimImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U8>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::S32>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Attr<float>("scale")
                                  .Attr<int64_t>("block_q")
                                  .Attr<int64_t>("block_k")
                                  .Attr<bool>("column"),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaMma, VoltaMmaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::F16>>()
                                  .Arg<ffi::Buffer<ffi::DataType::U8>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F16>>()
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>()
                                  .Attr<float>("scale")
                                  .Attr<int64_t>("block_q")
                                  .Attr<int64_t>("block_k")
                                  .Attr<bool>("want_lse"),
                              {ffi::Traits::kCmdBufferCompatible});
