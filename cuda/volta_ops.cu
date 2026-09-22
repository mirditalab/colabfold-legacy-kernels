// LayerNorm for sm_70+ and the gated dual projection for sm_75+.
// LayerNorm uses no tensor cores, thus Volta can use it.
// The projection needs mma.sync.m16n8k8. Volta uses volta_gdp_wmma.cu instead.

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

#define WARP 32

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
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int gn = n0 + nt * 8 + lc + j;
                if (gn >= N) {
                    continue;
                }
                const float p = accp[nt][half_i * 2 + j] + __half2float(bp[gn]);
                const float g = accg[nt][half_i * 2 + j] + __half2float(bg[gn]);
                const float s = gdp_act<A>(g);
                out[(long long)gm * N + gn] = __float2half(mv * p * s);
            }
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
#endif // VOLTA_MMA_OK
