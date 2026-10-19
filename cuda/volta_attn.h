// Constants the attention units share; forward and backward must agree on them.
#pragma once

#include <cuda_fp16.h>

// Threads per warp.
#define WARP 32

// A masked logit, matching AlphaFold's float16 mask bias. -inf would leave
// inf - inf on an all-masked row.
#define MASKED_LOGIT (-1.0e4f)

// The softmax runs in the log2 domain, where exp2 is one instruction.
#define LOG2E 1.4426950408889634f

// exp2 as one MUFU.EX2: exp2f adds a denormal fixup the softmax never needs
__device__ __forceinline__ float ex2_ftz(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


// Element strides of q (and the gate), k/v and out, as [n][h][s][c] with c contiguous, and an
// optional gate: out = attention * sigmoid(gate + gate_bias[h]). Contiguous [N,H,S,D] by default;
// packed() reads q/k/v/gate in place from one [X, Y, G, H, D] projection, out [X, Y, H, D].
struct AttnLayout {
    long long qn, qh, qs, kn, kh, ks, on, oh, os;
    const __half* gate = nullptr;
    const __half* gate_bias = nullptr;
    // trim (packed_attn.trim_limits): lims[n] ends batch row n's live keys, lims[N] every row's; a query tile past that,
    // or a row with no live key, is padding and gets zeros. Null keeps every key and query.
    const int* lims = nullptr;

    static AttnLayout contiguous(int H, int Sq, int Sk, int D) {
        return {(long long)H * Sq * D, (long long)Sq * D, D, (long long)H * Sk * D,
                (long long)Sk * D, D, (long long)H * Sq * D, (long long)Sq * D, D};
    }
    // rows attend along Y (batch X); a column attends along X (batch Y)
    static AttnLayout packed(int Y, int G, int H, int D, bool column) {
        const long long pn = (long long)G * H * D, pr = (long long)Y * pn;
        const long long on = (long long)H * D, orow = (long long)Y * on;
        return column ? AttnLayout{pn, D, pr, pn, D, pr, on, D, orow}
                      : AttnLayout{pr, D, pn, pr, D, pn, orow, D, on};
    }
    __device__ float apply_gate(float o, long long qi, int h, int c, int d) const {
        if (gate == nullptr) {
            return o;
        }
        const float g = __half2float(gate[qi]) + __half2float(gate_bias[h * d + c]);
        return __fdividef(o, 1.f + ex2_ftz(-LOG2E * g));
    }
};
