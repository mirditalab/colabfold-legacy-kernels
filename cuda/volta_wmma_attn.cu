// Flash attention for Volta (sm_70+), with nvcuda::wmma.
// On Volta wmma gives the same HMMA.884 instructions as mma.sync.m8n8k4.
// The fragment layout is not known, thus the softmax uses shared memory.
// Inputs: q,k,v [N,H,S,D] f16, bias [H,Sq,Sk] f16, kmask [N,Sk] u8.

#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <string>
#include "volta_attn.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using namespace nvcuda;

#define WARP 32
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
    float* __restrict__ lse, int N, int H, int Sq, int Sk, int DR, float sm_scale) {
    // DR is the real head dim in global memory; D is the shared-memory tile width, which
    // wmma needs to be a multiple of FRAG. For DR < D the staging zero-fills the rest, so
    // head 8 needs no padded copy of q/k/v (that copy costs ~2 GB on the extra MSA).
    constexpr int NWARP = BQ / FRAG;
    // The O tile comes back through this buffer too, thus D columns, not BK.
    constexpr int SS_LD = (BK > D ? BK : D) + 4; // f32 ldm: multiple of 4 for wmma stores
    constexpr int PS_LD = BK + 8; // f16 ldm: multiple of 8 for wmma loads
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
    __half* Ks = Ps + BQ * PS_LD;                            // [BK][D]
    __half* Vs = Ks + BK * D;                                // [BK][D]
    float* Os = reinterpret_cast<float*>(Vs + BK * D);       // [BQ][D]
    float* ms = Os + BQ * D;
    float* ls = ms + BQ;

    const long long qkv = (long long)(n * H + h) * Sq * DR;
    const long long kv = (long long)(n * H + h) * Sk * DR;
    const long long bh = (long long)h * Sq * Sk;

    // stage Q in the (still unused) S buffer and hoist its fragments
    __half* Qs = reinterpret_cast<__half*>(Ss);
    for (int i = tid; i < BQ * D; i += TPB) {
        int r = i / D, c = i - r * D;
        int gq = q0 + r;
        Qs[i] = (gq < Sq && c < DR) ? q[qkv + (long long)gq * DR + c] : __float2half(0.f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::row_major> fq[D / FRAG];
#pragma unroll
    for (int ks = 0; ks < D / FRAG; ++ks) {
        wmma::load_matrix_sync(fq[ks], Qs + (warp * FRAG) * D + ks * FRAG, D);
    }
    __syncthreads(); // Q fragments are in registers; S buffer is free again

    for (int i = tid; i < BQ * D; i += TPB) {
        Os[i] = 0.f;
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
    auto issue_loads = [&](int k0) {
#pragma unroll
        for (int t = 0; t < KV_PER_THREAD; ++t) {
            int i = tid + t * TPB;
            if (i < BK * D) {
                int r = i / D, c = i - r * D;
                int gk = k0 + r;
                bool ok = (gk < Sk) && (c < DR);
                kreg[t] = ok ? k[kv + (long long)gk * DR + c] : __float2half(0.f);
                vreg[t] = ok ? v[kv + (long long)gk * DR + c] : __float2half(0.f);
            }
        }
    };
    auto commit_loads = [&]() {
#pragma unroll
        for (int t = 0; t < KV_PER_THREAD; ++t) {
            int i = tid + t * TPB;
            if (i < BK * D) {
                Ks[i] = kreg[t];
                Vs[i] = vreg[t];
            }
        }
    };

    issue_loads(0);
    __syncthreads();
    commit_loads();
    __syncthreads();

    for (int k0 = 0; k0 < Sk; k0 += BK) {
        // Start the loads for the next tile now. The GEMMs below hide the delay.
        const int k_next = k0 + BK;
        if (k_next < Sk) {
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
                wmma::load_matrix_sync(fb, Ks + (nt * FRAG) * D + ks * FRAG, D);
                wmma::mma_sync(acc, fq[ks], fb, acc);
            }
            wmma::store_matrix_sync(Ss + (warp * FRAG) * SS_LD + nt * FRAG, acc, SS_LD,
                                    wmma::mem_row_major);
        }
        __syncwarp();

        // online softmax (scalar; Volta's fragment layout is opaque)
        float m_prev = ms[r_blk], m_cur = m_prev;
        const bool row_ok = (gq < Sq);
        for (int c = half_id; c < BK; c += 2) {
            const int gk = k0 + c;
            float s;
            if (!row_ok || gk >= Sk || kmask[(long long)n * Sk + gk] == 0) {
                s = MASKED_LOGIT * LOG2E;
            } else {
                s = Ss[r_blk * SS_LD + c] * qk_scale +
                    __half2float(bias[bh + (long long)gq * Sk + gk]) * LOG2E;
            }
            Ss[r_blk * SS_LD + c] = s;
            m_cur = fmaxf(m_cur, s);
        }
        m_cur = fmaxf(m_cur, __shfl_xor_sync(0xffffffffu, m_cur, 1));

        float l_part = 0.f;
        for (int c = half_id; c < BK; c += 2) {
            float p = exp2f(Ss[r_blk * SS_LD + c] - m_cur);
            Ps[r_blk * PS_LD + c] = __float2half(p);
            l_part += p;
        }
        l_part += __shfl_xor_sync(0xffffffffu, l_part, 1);
        const float alpha = exp2f(m_prev - m_cur);
        if (half_id == 0) {
            ms[r_blk] = m_cur;
            ls[r_blk] = ls[r_blk] * alpha + l_part;
        }
        __syncwarp();

// O = O*alpha + P @ V
#pragma unroll
        for (int dt = 0; dt < D / FRAG; ++dt) {
            wmma::fragment<wmma::accumulator, FRAG, FRAG, FRAG, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int ks = 0; ks < BK / FRAG; ++ks) {
                wmma::fragment<wmma::matrix_a, FRAG, FRAG, FRAG, __half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, FRAG, FRAG, FRAG, __half, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, Ps + (warp * FRAG) * PS_LD + ks * FRAG, PS_LD);
                wmma::load_matrix_sync(fb, Vs + (ks * FRAG) * D + dt * FRAG, D);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(Ss + (warp * FRAG) * SS_LD + dt * FRAG, acc, SS_LD,
                                    wmma::mem_row_major);
        }
        __syncwarp();
        for (int c = half_id; c < D; c += 2) {
            const int dt = c / FRAG, off = c % FRAG;
            Os[r_blk * D + c] = Os[r_blk * D + c] * alpha + Ss[r_blk * SS_LD + dt * FRAG + off];
        }

        // rotate the pipeline
        if (k_next < Sk) {
            __syncthreads(); // everyone is done reading Ks/Vs
            commit_loads();
            __syncthreads();
        }
    }

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
            out[qkv + (long long)gq * DR + c] = __float2half(Os[r_blk * D + c] * inv);
        }
    }
}

template <int D, int BQ, int BK, bool WANT_LSE>
static ffi::Error launch(cudaStream_t stream, int device, const __half* q, const __half* k,
                         const __half* v, const __half* bias, const uint8_t* kmask, __half* out,
                         float* lse, int N, int H, int Sq, int Sk, int DR, float scale) {
    constexpr int SS_LD = (BK > D ? BK : D) + 4, PS_LD = BK + 8;
    const size_t smem = (size_t)(BQ * SS_LD) * sizeof(float) +
                        (size_t)(BQ * PS_LD + 2 * BK * D) * sizeof(__half) +
                        (size_t)(BQ * D + 2 * BQ) * sizeof(float);
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
                                                  DR, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("volta_wmma launch: ") + cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

template <bool WANT_LSE>
static ffi::Error volta_wmma_common(cudaStream_t stream, int32_t device,
                                    ffi::Buffer<ffi::DataType::F16> q,
                                    ffi::Buffer<ffi::DataType::F16> k,
                                    ffi::Buffer<ffi::DataType::F16> v,
                                    ffi::Buffer<ffi::DataType::F16> bias,
                                    ffi::Buffer<ffi::DataType::U8> kmask,
                                    ffi::Result<ffi::Buffer<ffi::DataType::F16>> out, float* lse,
                                    float scale, int64_t block_q, int64_t block_k) {
    auto d = q.dimensions();
    if (d.size() != 4) {
        return ffi::Error::InvalidArgument("q must be [N,H,S,D]");
    }
    const int N = (int)d[0], H = (int)d[1], Sq = (int)d[2], D = (int)d[3];
    const int Sk = (int)k.dimensions()[2];
    const __half* qp = reinterpret_cast<const __half*>(q.typed_data());
    const __half* kp = reinterpret_cast<const __half*>(k.typed_data());
    const __half* vp = reinterpret_cast<const __half*>(v.typed_data());
    const __half* bp = reinterpret_cast<const __half*>(bias.typed_data());
    const uint8_t* mp = kmask.typed_data();
    __half* op = reinterpret_cast<__half*>(out->typed_data());
#define DISPATCH_T(TILE, DD, BQ, BK)                                                               \
    if (D == (DD) && block_q == (BQ) && block_k == (BK))                                           \
        return launch<TILE, BQ, BK, WANT_LSE>(stream, device, qp, kp, vp, bp, mp, op, lse, N, H, \
                                              Sq, Sk, (DD), scale);
#define DISPATCH(DD, BQ, BK) DISPATCH_T(DD, DD, BQ, BK)
    DISPATCH(32, 64, 64)
    DISPATCH(32, 64, 32)
    DISPATCH(32, 32, 64)
    DISPATCH(32, 32, 32) DISPATCH(32, 128, 64) DISPATCH(32, 128, 32) DISPATCH(16, 64, 64)
        DISPATCH(16, 32, 32) DISPATCH(16, 64, 32) DISPATCH(64, 64, 64) DISPATCH(64, 32, 32)
            DISPATCH(64, 64, 32)
    DISPATCH_T(16, 8, 64, 64) DISPATCH_T(16, 8, 64, 32) DISPATCH_T(16, 8, 32, 32)
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
    if (want_lse) {
        return volta_wmma_common<true>(stream, device, q, k, v, bias, kmask, out,
                                       lse->typed_data(), scale, block_q, block_k);
    }
    return volta_wmma_common<false>(stream, device, q, k, v, bias, kmask, out, nullptr, scale,
                                    block_q, block_k);
}

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
