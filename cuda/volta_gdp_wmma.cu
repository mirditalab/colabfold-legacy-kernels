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

#define WARP 32
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
        for (int i = lane; i < FRAG * FRAG; i += WARP) {
            const int r = i / FRAG, c = i - r * FRAG;
            const int gm = m0 + warp * FRAG + r, gn = n0 + nt * FRAG + c;
            if (gm >= M || gn >= N) {
                continue;
            }
            const float p = ep[i] + __half2float(bp[gn]);
            const float g = eg[i] + __half2float(bg[gn]);
            const float s = gdp_act<A>(g);
            out[(long long)gm * N + gn] = __float2half(__half2float(mask[gm]) * p * s);
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
