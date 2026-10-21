// Flash attention for Volta (sm_70+), with nvcuda::wmma.
// On Volta wmma gives the same HMMA.884 instructions as mma.sync.m8n8k4.
// The fragment layout is not known, thus the softmax uses shared memory.
// Inputs: q,k,v [N,H,S,D] f16, bias [H,Sq,Sk] f16, kmask [N,Sk] u8.

#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include "volta_attn.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace nvcuda;

#define FRAG 16

#define MAX_DEVICES 16

// Give the dynamic shared memory limit of one device. Read it one time only.
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

template <int D, int BQ, int BK, bool WANT_LSE>
__global__ __launch_bounds__(BQ / FRAG * WARP, 2) void volta_wmma_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ bias, const uint8_t* __restrict__ kmask, __half* __restrict__ out,
    float* __restrict__ lse, int N, int H, int Sq, int Sk, int DR, float sm_scale, AttnLayout L) {
    // DR is the real head dim in global memory; D is the shared-memory tile width, which
    // wmma needs to be a multiple of FRAG. For DR < D the staging zero-fills the rest, so
    // head 8 needs no padded copy of q/k/v (that copy costs ~2 GB on the extra MSA).
    constexpr int NWARP = BQ / FRAG;
    // The O tile comes back through this buffer too, thus D columns, not BK.
    constexpr int SS_LD = (BK > D ? BK : D) + 4; // f32 ldm: multiple of 4 for wmma stores
    constexpr int PS_LD = BK + 8; // f16 ldm: multiple of 8 for wmma loads
    // K and V rows padded by 16 bytes (a multiple of 8 halves for wmma, rows still 32-byte aligned): at a 64-byte stride
    // a fragment's rows fell in two bank groups
    constexpr int KS_LD = D + 8;
    constexpr int BS_LD = BK + 8; // the bias tile, staged with K and V
    constexpr int TPB = NWARP * WARP;
    constexpr int KV_PER_THREAD = (BK * D + TPB - 1) / TPB;

    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int tid = threadIdx.x;

    const int qtile = blockIdx.x, h = blockIdx.y, n = blockIdx.z;
    const int q0 = qtile * BQ;
    if (q0 >= Sq) {
        return;
    }

    extern __shared__ char smem[];
    float* Ss = reinterpret_cast<float*>(smem);              // [BQ][SS_LD]
    __half* Ps = reinterpret_cast<__half*>(Ss + BQ * SS_LD); // [BQ][PS_LD]
    __half* Ks = Ps + BQ * PS_LD;                            // [BK][KS_LD]
    __half* Vs = Ks + BK * KS_LD;                            // [BK][KS_LD]
    __half* Bs = Vs + BK * KS_LD;                            // [BQ][BS_LD]
    float* ms = reinterpret_cast<float*>(Bs + BQ * BS_LD);
    float* ls = ms + BQ;

    const long long qb = n * L.qn + h * L.qh, kb = n * L.kn + h * L.kh, ob = n * L.on + h * L.oh;
    const long long bh = (long long)h * Sq * Sk;

    int k_end = Sk;
    if (L.lims != nullptr) {
        k_end = L.lims[n];
        if (k_end == 0 || q0 >= L.lims[N]) {
            for (int i = tid; i < BQ * DR; i += TPB) {
                const int gq = q0 + i / DR;
                if (gq < Sq) {
                    out[ob + gq * L.os + i % DR] = __float2half(0.f);
                }
            }
            return;
        }
    }

    // stage Q in the (still unused) S buffer and hoist its fragments
    __half* Qs = reinterpret_cast<__half*>(Ss);
    for (int i = tid; i < BQ * D; i += TPB) {
        int r = i / D, c = i - r * D;
        int gq = q0 + r;
        Qs[i] = (gq < Sq && c < DR) ? q[qb + gq * L.qs + c] : __float2half(0.f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::row_major> fq[D / FRAG];
#pragma unroll
    for (int ks = 0; ks < D / FRAG; ++ks) {
        wmma::load_matrix_sync(fq[ks], Qs + (warp * FRAG) * D + ks * FRAG, D);
    }
    __syncthreads(); // Q fragments are in registers; S buffer is free again

    // O stays in accumulator fragments across key tiles; a rescale goes through the S buffer
    wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> acc_o[D / FRAG];
#pragma unroll
    for (int dt = 0; dt < D / FRAG; ++dt) {
        wmma::fill_fragment(acc_o[dt], 0.0f);
    }
    for (int i = tid; i < BQ; i += TPB) {
        ms[i] = -INFINITY;
        ls[i] = 0.f;
    }

    const int r_local = lane >> 1;
    const int half_id = lane & 1;
    const int r_blk = warp * FRAG + r_local;
    const int gq = q0 + r_blk;
    const float qk_scale = sm_scale * LOG2E;

    // prime the software pipeline: tile 0 -> registers -> shared
    __half kreg[KV_PER_THREAD], vreg[KV_PER_THREAD];
    // 16 bytes a load where the layout allows (the packed one does): K, V and the bias tile, which the softmax then
    // reads from shared memory instead of 4 bytes per row from global memory
    const auto aligned = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    // K and V a 16-byte load per 8 head columns (a head of 8 in a 16-wide tile: one load, the rest zero); the bias
    // tile wherever its rows are 16-byte aligned, whatever the head
    const bool vec = DR % 8 == 0 && L.ks % 8 == 0 && kb % 8 == 0 && aligned(k) && aligned(v);
    const bool vec_b = Sk % 8 == 0 && aligned(bias);
    constexpr int NKV = (BK * D / 8 + TPB - 1) / TPB, NB = (BQ * BK / 8 + TPB - 1) / TPB;
    uint4 kvec[NKV], vvec[NKV], bvec[NB];
    auto issue_loads = [&](int k0) {
        if (vec_b) {
#pragma unroll
            for (int t = 0; t < NB; ++t) {
                const int i = tid + t * TPB, r = i / (BK / 8), c = (i - r * (BK / 8)) * 8, gr = q0 + r, gk = k0 + c;
                const bool ok = i < BQ * BK / 8 && gr < Sq && gk < Sk;
                bvec[t] = ok ? *reinterpret_cast<const uint4*>(bias + bh + (long long)gr * Sk + gk) : make_uint4(0, 0, 0, 0);
            }
        }
        if (vec) {
#pragma unroll
            for (int t = 0; t < NKV; ++t) {
                const int i = tid + t * TPB, r = i / (D / 8), c = (i - r * (D / 8)) * 8, gk = k0 + r;
                const bool ok = i < BK * D / 8 && gk < Sk && c < DR;
                kvec[t] = ok ? *reinterpret_cast<const uint4*>(k + kb + gk * L.ks + c) : make_uint4(0, 0, 0, 0);
                vvec[t] = ok ? *reinterpret_cast<const uint4*>(v + kb + gk * L.ks + c) : make_uint4(0, 0, 0, 0);
            }
            return;
        }
#pragma unroll
        for (int t = 0; t < KV_PER_THREAD; ++t) {
            int i = tid + t * TPB;
            if (i < BK * D) {
                int r = i / D, c = i - r * D;
                int gk = k0 + r;
                bool ok = (gk < Sk) && (c < DR);
                kreg[t] = ok ? k[kb + gk * L.ks + c] : __float2half(0.f);
                vreg[t] = ok ? v[kb + gk * L.ks + c] : __float2half(0.f);
            }
        }
    };
    auto commit_loads = [&]() {
        if (vec_b) {
#pragma unroll
            for (int t = 0; t < NB; ++t) {
                const int i = tid + t * TPB, r = i / (BK / 8), c = (i - r * (BK / 8)) * 8;
                if (i < BQ * BK / 8) {
                    *reinterpret_cast<uint4*>(Bs + r * BS_LD + c) = bvec[t];
                }
            }
        }
        if (vec) {
#pragma unroll
            for (int t = 0; t < NKV; ++t) {
                const int i = tid + t * TPB, r = i / (D / 8), c = (i - r * (D / 8)) * 8;
                if (i < BK * D / 8) {
                    *reinterpret_cast<uint4*>(Ks + r * KS_LD + c) = kvec[t];
                    *reinterpret_cast<uint4*>(Vs + r * KS_LD + c) = vvec[t];
                }
            }
            return;
        }
#pragma unroll
        for (int t = 0; t < KV_PER_THREAD; ++t) {
            int i = tid + t * TPB;
            if (i < BK * D) {
                const int r = i / D, c = i - r * D;
                Ks[r * KS_LD + c] = kreg[t];
                Vs[r * KS_LD + c] = vreg[t];
            }
        }
    };

    issue_loads(0);
    __syncthreads();
    commit_loads();
    __syncthreads();

    for (int k0 = 0; k0 < k_end; k0 += BK) {
        // Start the loads for the next tile now. The GEMMs below hide the delay.
        const int k_next = k0 + BK;
        if (k_next < k_end) {
            issue_loads(k_next);
        }

// S = Q @ K^T
#pragma unroll
        for (int nt = 0; nt < BK / FRAG; ++nt) {
            wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int ks = 0; ks < D / FRAG; ++ks) {
                wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, Ks + (nt * FRAG) * KS_LD + ks * FRAG, KS_LD);
                wmma::mma_sync(acc, fq[ks], fb, acc);
            }
            wmma::store_matrix_sync(Ss + (warp * FRAG) * SS_LD + nt * FRAG, acc, SS_LD,
                                    wmma::mem_row_major);
        }
        __syncwarp();

        // online softmax (scalar; Volta's fragment layout is opaque)
        const float m_prev = ms[r_blk];
        float m_tile = -INFINITY;
        const bool row_ok = (gq < Sq);
        for (int c = half_id; c < BK; c += 2) {
            const int gk = k0 + c;
            float s;
            if (!row_ok || gk >= Sk || kmask[(long long)n * Sk + gk] == 0) {
                // keys past Sk get no weight, so a fully masked row averages Sk keys
                s = (gk >= Sk && row_ok) ? -INFINITY : MASKED_LOGIT * LOG2E;
            } else {
                const __half b = vec_b ? Bs[r_blk * BS_LD + c] : bias[bh + (long long)gq * Sk + gk];
                s = Ss[r_blk * SS_LD + c] * qk_scale + __half2float(b) * LOG2E;
            }
            Ss[r_blk * SS_LD + c] = s;
            m_tile = fmaxf(m_tile, s);
        }
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, 1));
        // the row's max moves only when a tile raises it past 2^6 (FlashMLA's rule): otherwise O and l keep their
        // scale and this tile's exp2 stays below 2^6, well inside fp16 P and the fp32 sums
        const bool raise = m_tile > m_prev + 6.f;
        const float m_cur = raise ? fmaxf(m_prev, m_tile) : m_prev;

        float l_part = 0.f;
        for (int c = half_id; c < BK; c += 2) {
            float p = ex2_ftz(Ss[r_blk * SS_LD + c] - m_cur);
            Ps[r_blk * PS_LD + c] = __float2half(p);
            l_part += p;
        }
        l_part += __shfl_xor_sync(0xffffffffu, l_part, 1);
        const float alpha = raise ? ex2_ftz(m_prev - m_cur) : 1.f;
        if (half_id == 0) {
            ms[r_blk] = m_cur;
            ls[r_blk] = ls[r_blk] * alpha + l_part;
        }
        __syncwarp();

// O = O*alpha + P @ V. A fragment's rows are opaque, so a per-row rescale goes through shared memory: only when a row
        // of this warp raised its max, and never on the first tile (O is still zero)
        if (k0 > 0 && __any_sync(0xffffffffu, raise)) {
#pragma unroll
            for (int dt = 0; dt < D / FRAG; ++dt) {
                wmma::store_matrix_sync(Ss + (warp * FRAG) * SS_LD + dt * FRAG, acc_o[dt], SS_LD, wmma::mem_row_major);
            }
            __syncwarp();
            for (int c = half_id; c < D; c += 2) {
                Ss[r_blk * SS_LD + c] *= alpha;
            }
            __syncwarp();
#pragma unroll
            for (int dt = 0; dt < D / FRAG; ++dt) {
                wmma::load_matrix_sync(acc_o[dt], Ss + (warp * FRAG) * SS_LD + dt * FRAG, SS_LD, wmma::mem_row_major);
            }
        }
#pragma unroll
        for (int dt = 0; dt < D / FRAG; ++dt) {
#pragma unroll
            for (int ks = 0; ks < BK / FRAG; ++ks) {
                wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, Ps + (warp * FRAG) * PS_LD + ks * FRAG, PS_LD);
                wmma::load_matrix_sync(fb, Vs + (ks * FRAG) * KS_LD + dt * FRAG, KS_LD);
                wmma::mma_sync(acc_o[dt], fa, fb, acc_o[dt]);
            }
        }

        // rotate the pipeline
        if (k_next < Sk) {
            __syncthreads(); // everyone is done reading Ks/Vs
            commit_loads();
            __syncthreads();
        }
    }

    // O's rows, for the normalisation and the gate
#pragma unroll
    for (int dt = 0; dt < D / FRAG; ++dt) {
        wmma::store_matrix_sync(Ss + (warp * FRAG) * SS_LD + dt * FRAG, acc_o[dt], SS_LD, wmma::mem_row_major);
    }
    __syncwarp();
    if (gq < Sq) {
        // softmax statistic, for the backward pass. Both lanes of a row hold
        // the same reduced m/l, so only one of them writes.
        if constexpr (WANT_LSE) {
            if (half_id == 0) {
                const float l = ls[r_blk];
                lse[(long long)(n * H + h) * Sq + gq] =
                    (l > 0.f) ? (ms[r_blk] + log2f(l)) : -INFINITY;
            }
        }
        const float inv = 1.0f / fmaxf(ls[r_blk], 1e-30f);
        for (int c = half_id; c < DR; c += 2) {
            out[ob + gq * L.os + c] =
                __float2half(L.apply_gate(Ss[r_blk * SS_LD + c] * inv, qb + gq * L.qs + c, h, c, DR));
        }
    }
}

template <int D, int BQ, int BK, bool WANT_LSE>
static ffi::Error launch(cudaStream_t stream, int device, const __half* q, const __half* k,
                         const __half* v, const __half* bias, const uint8_t* kmask, __half* out,
                         float* lse, int N, int H, int Sq, int Sk, int DR, float scale,
                         const AttnLayout& L) {
    constexpr int SS_LD = (BK > D ? BK : D) + 4, PS_LD = BK + 8;
    constexpr size_t smem = (size_t)(BQ * SS_LD) * sizeof(float) +
                            (size_t)(BQ * PS_LD + 2 * BK * (D + 8) + BQ * (BK + 8)) * sizeof(__half) +
                            (size_t)(2 * BQ) * sizeof(float);
    auto kern = volta_wmma_kernel<D, BQ, BK, WANT_LSE>;
    const int max_smem = device_shared_limit(device);
    if ((int)smem > max_smem) {
        return ffi::Error::InvalidArgument("volta_wmma: needs " + std::to_string(smem / 1024) +
                                           " KB shared, device allows " +
                                           std::to_string(max_smem / 1024) + " KB");
    }
    if (smem > 48 * 1024 && needs_smem_optin(device, (const void*)kern)) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    dim3 grid((Sq + BQ - 1) / BQ, H, N);
    kern<<<grid, BQ / FRAG * WARP, smem, stream>>>(q, k, v, bias, kmask, out, lse, N, H, Sq, Sk,
                                                  DR, scale, L);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_wmma launch: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

template <bool WANT_LSE>
static ffi::Error volta_wmma_common(cudaStream_t stream, int32_t device, const __half* qp,
                                    const __half* kp, const __half* vp, const __half* bp,
                                    const uint8_t* mp, __half* op, float* lse, int N, int H,
                                    int Sq, int Sk, int D, const AttnLayout& L, float scale,
                                    int64_t block_q, int64_t block_k) {
#define DISPATCH_T(TILE, DD, BQ, BK)                                                               \
    if (D == (DD) && block_q == (BQ) && block_k == (BK))                                           \
        return launch<TILE, BQ, BK, WANT_LSE>(stream, device, qp, kp, vp, bp, mp, op, lse, N, H, \
                                              Sq, Sk, (DD), scale, L);
#define DISPATCH(DD, BQ, BK) DISPATCH_T(DD, DD, BQ, BK)
    DISPATCH(32, 64, 64)
    DISPATCH(32, 64, 32)
    DISPATCH(32, 32, 64)
    DISPATCH(32, 32, 32) DISPATCH(32, 128, 64) DISPATCH(32, 128, 32) DISPATCH(16, 64, 64)
        DISPATCH(16, 32, 32) DISPATCH(16, 64, 32) DISPATCH(64, 64, 64) DISPATCH(64, 32, 32)
            DISPATCH(64, 64, 32)
    DISPATCH_T(16, 8, 64, 64) DISPATCH_T(16, 8, 64, 32) DISPATCH_T(16, 8, 32, 32)
    // af3 head dims: diffusion token 768/16, protenix2's 96. 24 is not a
    // multiple of FRAG, so it keeps padding to 32.
    DISPATCH(48, 64, 32) DISPATCH(48, 32, 32) DISPATCH(96, 64, 32) DISPATCH(96, 32, 32)
#undef DISPATCH
#undef DISPATCH_T
                return ffi::Error::InvalidArgument("volta_wmma: unsupported (D, bq, bk)");
}

// lse is the softmax statistic the backward needs. With want_lse false the
// store compiles out and the buffer goes untouched, so one element is enough.
ffi::Error VoltaWmmaImpl(cudaStream_t stream, int32_t device, ffi::Buffer<ffi::DataType::F16> q,
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
    return (want_lse ? volta_wmma_common<true> : volta_wmma_common<false>)(
        stream, device, f16(q), f16(k), f16(v), f16(bias), kmask.typed_data(), op,
        want_lse ? lse->typed_data() : nullptr, N, H, Sq, Sk, D, L, scale, block_q, block_k);
}


// m8n8k4, sm_70's native HMMA: one warp instruction is four 8x8x4 products, one per quad-pair qp = (lane >> 2) & 3 (lanes
// 4qp..4qp+3 and 16+4qp..16+4qp+3). Unlike wmma's, its fragment layouts are documented (and checked by a probe), so the
// packed attention below keeps S, P and O in registers: no shared-memory round trip per key tile but K, V and bias.
//   A (.row): a lane holds row (lane & 3) + 4 * (lane >> 4), k 0..3; B (.col): column (lane & 3) + 4 * (lane >> 4)
//   accumulator: element i at row (lane & 1) + (i & 2) + 4 * (lane >> 4), column (i & 4) + (lane & 2) + (i & 1)
__device__ __forceinline__ void mma884(float (&d)[8], uint32_t a01, uint32_t a23, uint32_t b01, uint32_t b23) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a01), "r"(a23), "r"(b01), "r"(b23));
}

// The packed forward (no LSE) where its 16-byte paths apply: D a multiple of 8, Sk of 8, rows 16-byte aligned. A warp
// owns 32 query rows, 8 per quad-pair; a lane holds its query row's A fragments, two rows of S and O.
template <int D, int BQ, int BK>
__global__ __launch_bounds__(BQ / 32 * WARP) void volta_mma884_packed_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
    const __half* __restrict__ bias, const uint8_t* __restrict__ kmask, __half* __restrict__ out, int N, int H,
    int Sq, int Sk, float sm_scale, AttnLayout L) {
    constexpr int NWARP = BQ / 32, TPB = NWARP * WARP;
    constexpr int NK = BK / 8, ND = D / 8, KST = D / 4; // key sub-tiles, output sub-tiles, Q K^T steps
    // rows padded by 16 bytes; V staged transposed so a B fragment (4 keys of one column) is one 8-byte load
    constexpr int QS_LD = D + 8, KS_LD = D + 8, VT_LD = BK + 8, BS_LD = BK + 8;
    constexpr int NKV = (BK * D / 8 + TPB - 1) / TPB, NB = (BQ * BK / 8 + TPB - 1) / TPB;

    const int lane = threadIdx.x & (WARP - 1), warp = threadIdx.x / WARP, tid = threadIdx.x;
    const int qp = (lane >> 2) & 3, hi = lane >> 4;
    const int qtile = blockIdx.x, h = blockIdx.y, n = blockIdx.z, q0 = qtile * BQ;
    if (q0 >= Sq) {
        return;
    }
    extern __shared__ char smem[];
    __half* Qs = reinterpret_cast<__half*>(smem); // [BQ][QS_LD]
    __half* Ks = Qs + BQ * QS_LD;                 // [BK][KS_LD]
    __half* Vt = Ks + BK * KS_LD;                 // [D][VT_LD]
    __half* Bs = Vt + D * VT_LD;                  // [BQ][BS_LD]

    const long long qb = n * L.qn + h * L.qh, kb = n * L.kn + h * L.kh, ob = n * L.on + h * L.oh;
    const long long bh = (long long)h * Sq * Sk;
    int k_end = Sk;
    if (L.lims != nullptr) {
        k_end = L.lims[n];
        if (k_end == 0 || q0 >= L.lims[N]) {
            for (int i = tid; i < BQ * D; i += TPB) {
                const int gq = q0 + i / D;
                if (gq < Sq) {
                    out[ob + gq * L.os + i % D] = __float2half(0.f);
                }
            }
            return;
        }
    }

    for (int i = tid; i < BQ * D / 8; i += TPB) {
        const int r = i / (D / 8), c = (i - r * (D / 8)) * 8, gq = q0 + r;
        *reinterpret_cast<uint4*>(Qs + r * QS_LD + c) =
            gq < Sq ? *reinterpret_cast<const uint4*>(q + qb + gq * L.qs + c) : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    const int wrow = warp * 32 + qp * 8;   // this quad-pair's 8 query rows within the block
    const int ra = (lane & 3) + 4 * hi;    // A row / B column within 8
    const int rx = (lane & 1) + 4 * hi;    // accumulator rows rx, rx + 2
    const int cx = lane & 2;               // accumulator columns cx + {0, 1, 4, 5}
    uint2 qa[KST];
#pragma unroll
    for (int ks = 0; ks < KST; ++ks) {
        qa[ks] = *reinterpret_cast<const uint2*>(Qs + (wrow + ra) * QS_LD + ks * 4);
    }

    float o[ND][8];
#pragma unroll
    for (int d = 0; d < ND; ++d)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[d][i] = 0.f;
        }
    float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
    const float qk_scale = sm_scale * LOG2E;

    uint4 kvec[NKV], vvec[NKV], bvec[NB];
    const auto fetch = [&](int k0) {
#pragma unroll
        for (int t = 0; t < NKV; ++t) {
            const int i = tid + t * TPB, r = i / (D / 8), c = (i - r * (D / 8)) * 8, gk = k0 + r;
            const bool ok = i < BK * D / 8 && gk < Sk;
            kvec[t] = ok ? *reinterpret_cast<const uint4*>(k + kb + gk * L.ks + c) : make_uint4(0, 0, 0, 0);
            vvec[t] = ok ? *reinterpret_cast<const uint4*>(v + kb + gk * L.ks + c) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int t = 0; t < NB; ++t) {
            const int i = tid + t * TPB, r = i / (BK / 8), c = (i - r * (BK / 8)) * 8, gr = q0 + r, gk = k0 + c;
            const bool ok = i < BQ * BK / 8 && gr < Sq && gk < Sk;
            bvec[t] = ok ? *reinterpret_cast<const uint4*>(bias + bh + (long long)gr * Sk + gk) : make_uint4(0, 0, 0, 0);
        }
    };
    const auto store = [&]() {
#pragma unroll
        for (int t = 0; t < NKV; ++t) {
            const int i = tid + t * TPB, r = i / (D / 8), c = (i - r * (D / 8)) * 8;
            if (i < BK * D / 8) {
                *reinterpret_cast<uint4*>(Ks + r * KS_LD + c) = kvec[t];
                const __half* vh = reinterpret_cast<const __half*>(&vvec[t]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    Vt[(c + j) * VT_LD + r] = vh[j];
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NB; ++t) {
            const int i = tid + t * TPB, r = i / (BK / 8), c = (i - r * (BK / 8)) * 8;
            if (i < BQ * BK / 8) {
                *reinterpret_cast<uint4*>(Bs + r * BS_LD + c) = bvec[t];
            }
        }
    };

    const int t4 = lane & 3, src0 = (lane & ~3) | (t4 & 1), src1 = src0 | 2, tslot = t4 >> 1;
    fetch(0);
    for (int k0 = 0; k0 < k_end; k0 += BK) {
        __syncthreads();
        store();
        __syncthreads();
        if (k0 + BK < k_end) {
            fetch(k0 + BK);
        }

        // S = Q K^T: B fragment = 4 head columns of key ra
        float s[NK][8];
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s[nt][i] = 0.f;
            }
#pragma unroll
            for (int ks = 0; ks < KST; ++ks) {
                const uint2 b = *reinterpret_cast<const uint2*>(Ks + (nt * 8 + ra) * KS_LD + ks * 4);
                mma884(s[nt], qa[ks].x, qa[ks].y, b.x, b.y);
            }
        }

        // scale, bias, mask (a masked key MASKED_LOGIT, one past Sk -inf, a row past Sq all masked); row max over the
        // two lanes holding a row (lane ^ 2)
        float m_tile[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nt = 0; nt < NK; ++nt) {
            const int kc = k0 + nt * 8;
            const bool in = kc < Sk; // Sk % 8 == 0: a sub-tile is all in or all out
            const uint32_t mk0 = in ? *reinterpret_cast<const uint16_t*>(kmask + (long long)n * Sk + kc + cx) : 0u;
            const uint32_t mk1 = in ? *reinterpret_cast<const uint16_t*>(kmask + (long long)n * Sk + kc + cx + 4) : 0u;
#pragma unroll
            for (int sl = 0; sl < 2; ++sl) {
                const int row = wrow + rx + 2 * sl;
                const bool row_ok = q0 + row < Sq;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const __half2 b2 = *reinterpret_cast<const __half2*>(Bs + row * BS_LD + nt * 8 + hh * 4 + cx);
                    const uint32_t mk = hh ? mk1 : mk0;
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int i = (hh << 2) | (sl << 1) | j;
                        const bool keep = row_ok && in && ((mk >> (8 * j)) & 0xff) != 0;
                        const float b = __half2float(j ? __high2half(b2) : __low2half(b2));
                        s[nt][i] = keep ? s[nt][i] * qk_scale + b * LOG2E
                                        : ((!in && row_ok) ? -INFINITY : MASKED_LOGIT * LOG2E);
                        m_tile[sl] = fmaxf(m_tile[sl], s[nt][i]);
                    }
                }
            }
        }
        float m_new[2], alpha[2];
        bool raise_any = false;
#pragma unroll
        for (int sl = 0; sl < 2; ++sl) {
            m_tile[sl] = fmaxf(m_tile[sl], __shfl_xor_sync(0xffffffffu, m_tile[sl], 2));
            // lazy rescale (FlashMLA's rule): the max moves only when a tile raises it past 2^6
            const bool raise = m_tile[sl] > m_run[sl] + 6.f;
            m_new[sl] = raise ? fmaxf(m_run[sl], m_tile[sl]) : m_run[sl];
            alpha[sl] = raise ? ex2_ftz(m_run[sl] - m_new[sl]) : 1.f;
            raise_any |= raise;
        }

        // P = exp2(S - m) as half pairs pp[nt][slot][half] (columns half * 4 + cx + {0, 1}), and the row sums
        uint32_t pp[NK][2][2];
        float l_tile[2] = {0.f, 0.f};
#pragma unroll
        for (int nt = 0; nt < NK; ++nt)
#pragma unroll
            for (int sl = 0; sl < 2; ++sl)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int i = (hh << 2) | (sl << 1);
                    const float e0 = ex2_ftz(s[nt][i] - m_new[sl]), e1 = ex2_ftz(s[nt][i + 1] - m_new[sl]);
                    l_tile[sl] += e0 + e1;
                    const __half2 e = __floats2half2_rn(e0, e1);
                    pp[nt][sl][hh] = reinterpret_cast<const uint32_t&>(e);
                }
#pragma unroll
        for (int sl = 0; sl < 2; ++sl) {
            l_tile[sl] += __shfl_xor_sync(0xffffffffu, l_tile[sl], 2);
            l_run[sl] = l_run[sl] * alpha[sl] + l_tile[sl];
            m_run[sl] = m_new[sl];
        }
        if (raise_any) {
#pragma unroll
            for (int d = 0; d < ND; ++d)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o[d][i] *= alpha[(i >> 1) & 1];
                }
        }

        // O += P V. P's A fragment (row t4 + 4 hi, 4 keys) from the accumulator layout: its keys 0-1 sit in lane
        // src0, 2-3 in src1, in row slot tslot, so 4 shuffles a step within the quad
#pragma unroll
        for (int nt = 0; nt < NK; ++nt)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const uint32_t x0 = __shfl_sync(0xffffffffu, pp[nt][0][hh], src0);
                const uint32_t x1 = __shfl_sync(0xffffffffu, pp[nt][1][hh], src0);
                const uint32_t y0 = __shfl_sync(0xffffffffu, pp[nt][0][hh], src1);
                const uint32_t y1 = __shfl_sync(0xffffffffu, pp[nt][1][hh], src1);
                const uint32_t a01 = tslot ? x1 : x0, a23 = tslot ? y1 : y0;
#pragma unroll
                for (int d = 0; d < ND; ++d) {
                    const uint2 b = *reinterpret_cast<const uint2*>(Vt + (d * 8 + ra) * VT_LD + nt * 8 + hh * 4);
                    mma884(o[d], a01, a23, b.x, b.y);
                }
            }
    }

#pragma unroll
    for (int sl = 0; sl < 2; ++sl) {
        const int gq = q0 + wrow + rx + 2 * sl;
        if (gq >= Sq) {
            continue;
        }
        const float inv = 1.0f / fmaxf(l_run[sl], 1e-30f);
#pragma unroll
        for (int d = 0; d < ND; ++d)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int col = d * 8 + hh * 4 + cx + j;
                    out[ob + gq * L.os + col] = __float2half(
                        L.apply_gate(o[d][(hh << 2) | (sl << 1) | j] * inv, qb + gq * L.qs + col, h, col, D));
                }
    }
}

template <int D, int BQ, int BK>
static ffi::Error launch_mma884(cudaStream_t stream, int device, const __half* q, const __half* k, const __half* v,
                                const __half* bias, const uint8_t* kmask, __half* out, int N, int H, int Sq, int Sk,
                                float scale, const AttnLayout& L) {
    constexpr size_t smem = (size_t)(BQ * (D + 8) + BK * (D + 8) + D * (BK + 8) + BQ * (BK + 8)) * sizeof(__half);
    auto kern = volta_mma884_packed_kernel<D, BQ, BK>;
    if (smem > 48 * 1024 && needs_smem_optin(device, (const void*)kern)) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }
    dim3 grid((Sq + BQ - 1) / BQ, H, N);
    kern<<<grid, BQ / 32 * WARP, smem, stream>>>(q, k, v, bias, kmask, out, N, H, Sq, Sk, scale, L);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_mma884 launch: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

// the m8n8k4 kernel where it applies (VOLTA_ATTN_WMMA=1 keeps the wmma one, for comparisons); false: use wmma
static bool mma884_packed(cudaStream_t stream, int device, const __half* q, const __half* k, const __half* v,
                          const __half* bias, const uint8_t* kmask, __half* out, int N, int H, int S, int D,
                          const AttnLayout& L, float scale, int64_t block_q, int64_t block_k, ffi::Error* err) {
    static const bool off = [] { const char* e = getenv("VOLTA_ATTN_WMMA"); return e && e[0] == '1'; }();
    const auto a16 = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; };
    if (off || S % 8 != 0 || L.qs % 8 != 0 || L.ks % 8 != 0 || L.qh % 8 != 0 || L.kh % 8 != 0 || !a16(q) ||
        !a16(k) || !a16(v) || !a16(bias)) {
        return false;
    }
#define DISPATCH884(DD, BQ, BK)                                                                                      \
    if (D == (DD) && block_q == (BQ) && block_k == (BK)) {                                                          \
        *err = launch_mma884<DD, BQ, BK>(stream, device, q, k, v, bias, kmask, out, N, H, S, S, scale, L);          \
        return true;                                                                                                \
    }
    DISPATCH884(32, 64, 32) DISPATCH884(32, 64, 64) DISPATCH884(32, 128, 32) DISPATCH884(32, 128, 64)
    DISPATCH884(8, 64, 32) DISPATCH884(8, 64, 64) DISPATCH884(8, 128, 64)
    DISPATCH884(16, 64, 32) DISPATCH884(16, 64, 64) DISPATCH884(64, 64, 32) // (64, 64, 64) spills: wmma
#undef DISPATCH884
    return false;
}

// Forward only, as VoltaMmaPacked: q/k/v (and the gate when G == 4) read in place from p.
static ffi::Error VoltaWmmaPackedImpl(const int* lims, cudaStream_t stream, int32_t device,
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
    ffi::Error err = ffi::Error::Success();
    if (mma884_packed(stream, device, pp, pp + H * D, pp + 2 * H * D, reinterpret_cast<const __half*>(bias.typed_data()),
                      kmask.typed_data(), reinterpret_cast<__half*>(out->typed_data()), N, H, S, D, L, scale, block_q,
                      block_k, &err)) {
        return err;
    }
    return volta_wmma_common<false>(stream, device, pp, pp + H * D, pp + 2 * H * D,
                                    reinterpret_cast<const __half*>(bias.typed_data()),
                                    kmask.typed_data(),
                                    reinterpret_cast<__half*>(out->typed_data()), nullptr, N, H,
                                    S, S, D, L, scale, block_q, block_k);
}

static ffi::Error VoltaWmmaPackedNoTrim(cudaStream_t stream, int32_t device,
                    ffi::Buffer<ffi::DataType::F16> p, ffi::Buffer<ffi::DataType::F16> bias,
                    ffi::Buffer<ffi::DataType::U8> kmask, ffi::Buffer<ffi::DataType::F16> gate_bias,
                    ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, float scale, int64_t block_q,
                    int64_t block_k, bool column) {
    return VoltaWmmaPackedImpl(nullptr, stream, device, p, bias, kmask, gate_bias, out, scale, block_q, block_k, column);
}

// VoltaWmmaPacked with trim: lims [N + 1] from packed_attn.trim_limits(kmask, 1)
static ffi::Error VoltaWmmaPackedTrimImpl(cudaStream_t stream, int32_t device,
                    ffi::Buffer<ffi::DataType::F16> p, ffi::Buffer<ffi::DataType::F16> bias,
                    ffi::Buffer<ffi::DataType::U8> kmask, ffi::Buffer<ffi::DataType::F16> gate_bias, ffi::Buffer<ffi::DataType::S32> lims,
                    ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, float scale, int64_t block_q,
                    int64_t block_k, bool column) {
    return VoltaWmmaPackedImpl(lims.typed_data(), stream, device, p, bias, kmask, gate_bias, out, scale, block_q, block_k, column);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaWmmaPacked, VoltaWmmaPackedNoTrim,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaWmmaPackedTrim, VoltaWmmaPackedTrimImpl,
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(VoltaWmma, VoltaWmmaImpl,
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
