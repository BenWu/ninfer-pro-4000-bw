
#pragma once

// DEBUG-ONLY scratch for the KV-port NaN/garbage timeline: a single-block BF16
// NaN/Inf/max-abs scanner launched from the append and attention launchers. One
// host call per tensor yields one `DBGSCAN <kind>#<seq>` line so a single run
// produces a per-layer timeline. Remove this file and its include/launch sites
// when the diagnosis is complete.

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>
#include <cstdio>

namespace ninfer::ops::detail {

// Anchor kinds (device-side tag table).
inline constexpr int kE8DbgAppendInKFull  = 0;
inline constexpr int kE8DbgAppendInVFull  = 1;
inline constexpr int kE8DbgAppendInKSmall = 2;
inline constexpr int kE8DbgAppendInVSmall = 3;
inline constexpr int kE8DbgAttnOutPrompt  = 4;
inline constexpr int kE8DbgAttnOutSmall   = 5;
inline constexpr int kE8DbgPlaneKScale    = 6;
inline constexpr int kE8DbgPlaneVScale    = 7;
inline constexpr int kE8DbgPlaneKCode     = 8;
inline constexpr int kE8DbgPlaneVCode     = 9;

// Row-major [head_dim, heads, tokens...] BF16 tensor: element i decodes to
// dim = i % head_dim, head = (i / head_dim) % heads, token = (i / head_dim) / heads.
// static: each including TU gets its own instance, so the RDC device linker sees no
// duplicate symbol (same rule as the templated inverse-rotation kernel).
static __global__ void e8_debug_scan_bf16_kernel(const __nv_bfloat16* __restrict__ x, std::int64_t n,
                                           int heads, int head_dim, int kind, int seq) {
    const int tid = static_cast<int>(threadIdx.x);
    const int nth = static_cast<int>(blockDim.x);
    __shared__ int nan_total;
    __shared__ int inf_total;
    __shared__ int nan_hits[8];
    __shared__ int hit_slots;
    __shared__ unsigned int maxbits;
    __shared__ unsigned long long maxloc;
    if (tid == 0) {
        nan_total = 0;
        inf_total = 0;
        hit_slots = 0;
        maxbits   = 0u;
        maxloc    = static_cast<unsigned long long>(-1);
    }
    __syncthreads();
    int local_nan = 0;
    int local_inf = 0;
    for (std::int64_t i = tid; i < n; i += nth) {
        const float f = __bfloat162float(x[i]);
        if (f != f) {
            ++local_nan;
            const int old = atomicAdd(&hit_slots, 1);
            if (old < 8) {
                const int rest  = static_cast<int>(i / head_dim);
                const int head  = rest % heads;
                const int token = rest / heads;
                nan_hits[old]   = head | (token << 5) | (static_cast<int>(i % head_dim) << 22);
            }
        } else if (f == CUDART_INF_F || f == -CUDART_INF_F) {
            ++local_inf;
        } else {
            // Finite values: track the max |value| (float bits order by magnitude for x >= 0).
            const unsigned int bits = __float_as_uint(fabsf(f));
            unsigned int cur = maxbits;
            while (bits > cur) {
                if (atomicCAS(&maxbits, cur, bits) == cur) {
                    atomicExch(&maxloc, i);
                    break;
                }
                cur = maxbits;
            }
        }
    }
    if (local_nan != 0) { atomicAdd(&nan_total, local_nan); }
    if (local_inf != 0) { atomicAdd(&inf_total, local_inf); }
    __syncthreads();
    if (tid == 0) {
        const char* name = "???";
        switch (kind) {
        case kE8DbgAppendInKFull:  name = "APK-full";  break;
        case kE8DbgAppendInVFull:  name = "APV-full";  break;
        case kE8DbgAppendInKSmall: name = "APK-small"; break;
        case kE8DbgAppendInVSmall: name = "APV-small"; break;
        case kE8DbgAttnOutPrompt:  name = "OUT-prompt"; break;
        case kE8DbgAttnOutSmall:   name = "OUT-small"; break;
        }
        const int reported = (hit_slots < 8) ? hit_slots : 8;
        printf("DBGSCAN %s#%d n=%lld nan=%d inf=%d maxabs=%f", name, seq, static_cast<long long>(n),
               nan_total, inf_total, maxbits != 0u ? __uint_as_float(maxbits) : -1.0f);
        if (maxloc != static_cast<unsigned long long>(-1)) {
            const unsigned long long rest = maxloc / head_dim;
            printf(" max@h=%lld t=%lld d=%lld", static_cast<long long>(rest % heads),
                   static_cast<long long>(rest / heads), static_cast<long long>(maxloc % head_dim));
        }
        for (int h = 0; h < reported; ++h) {
            const int pack = nan_hits[h];
            printf(" | h=%d t=%d d=%d", pack & 0x1f, (pack >> 5) & 0x3fff, pack >> 22);
        }
        printf("\n");
    }
}

inline void e8_debug_scan_bf16(const void* data, std::int64_t n, int heads, int head_dim,
                               int kind, int seq, cudaStream_t stream) {
    if (data == nullptr || n <= 0) { return; }
    e8_debug_scan_bf16_kernel<<<1, 1024, 0, stream>>>(static_cast<const __nv_bfloat16*>(data), n,
                                                      heads, head_dim, kind, seq);
}

} // namespace ninfer::ops::detail
