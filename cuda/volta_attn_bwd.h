// Helpers both backward units need. Static, so each unit still keeps its own
// copy and its own device cache, exactly as when they were duplicated.
#pragma once

#include <cuda_fp16.h>

#include "volta_attn.h"

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

// delta_i = sum_c out_ic * dout_ic, the row statistic the backward subtracts.
// One warp per row, so the reads coalesce.
static __global__ void row_delta(const __half* __restrict__ out,
                                 const __half* __restrict__ dout, float* __restrict__ delta,
                                 long long rows, int DR) {
    const long long row = (long long)blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= rows) {
        return;
    }
    const int lane = threadIdx.x & (WARP - 1);
    float s = 0.f;
    for (int c = lane; c < DR; c += WARP) {
        s += __half2float(out[row * DR + c]) * __half2float(dout[row * DR + c]);
    }
#pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
    }
    if (lane == 0) {
        delta[row] = s;
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
