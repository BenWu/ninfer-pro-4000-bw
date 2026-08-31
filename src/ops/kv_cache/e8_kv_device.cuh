#pragma once

// Shared device encoder for an E8 (K, V) 64-dimension group: H64-rotate both planes, then encode
// K as a packed 4-bit lattice projection (RK4V4E8) or a 2-byte E8 root-cylinder code per 8 rotated
// dimensions (RK2V4E8), and V as packed 4-bit codes, with per-group FP16 scales. Reused by the
// standalone append kernels and by the fused small-T append path.

#include "ops/kv_cache/e8_lattice.cuh"
#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

namespace ninfer::ops {

template <typename Geometry, bool E8Lattice, bool E8Root>
__device__ __forceinline__ void kv_cache_append_e8_group(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v, int kv_head,
    int group, int token, int page, int page_off, std::uint8_t* __restrict__ cache_k,
    std::uint8_t* __restrict__ cache_v, __half* __restrict__ scale_k,
    __half* __restrict__ scale_v, int lane) {
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(E8Lattice != E8Root, "E8 modes are mutually exclusive");
    const int d0              = group * kKVCacheInt8Group + lane;
    const int d1              = d0 + 32;
    const std::int64_t src0   = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
    const std::int64_t src1   = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
    float k0                  = __bfloat162float(k[src0]);
    float k1                  = __bfloat162float(k[src1]);
    float v0                  = __bfloat162float(v[src0]);
    float v1                  = __bfloat162float(v[src1]);
    kv_cache_hadamard64(k0, k1, FullMask);
    kv_cache_hadamard64(v0, v1, FullMask);

    const float k_abs = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
    const float v_abs = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
    const __half ksh  = __float2half_rn(k_abs > 0.0f ? k_abs / 7.0f : 0.0f);
    const __half vsh  = __float2half_rn(v_abs > 0.0f ? v_abs / 7.0f : 0.0f);
    const float ks    = __half2float(ksh);
    const float vs    = __half2float(vsh);
    const float kinv  = ks > 0.0f ? 1.0f / ks : 0.0f;
    const float vinv  = vs > 0.0f ? 1.0f / vs : 0.0f;

    if constexpr (E8Root) {
        uint8_t c1_0, c2_0, c1_1, c2_1;
        e8_encode_cylinder_8d_warp(k0, ks, c1_0, c2_0, lane);
        e8_encode_cylinder_8d_warp(k1, ks, c1_1, c2_1, lane);
        if ((lane & 7) == 0) {
            const int k_group_d0 = group * (kKVCacheInt8Group / 4);
            std::uint8_t* k_root = reinterpret_cast<std::uint8_t*>(
                cache_k + kv_cache_int8_quant_e8_code_index<Geometry>(
                              page, kv_head, k_group_d0 + (lane / 8) * 2, page_off));
            k_root[0] = c1_0;
            k_root[1] = c2_0;
            std::uint8_t* k_root_hi = reinterpret_cast<std::uint8_t*>(
                cache_k + kv_cache_int8_quant_e8_code_index<Geometry>(
                             page, kv_head, k_group_d0 + (4 + lane / 8) * 2, page_off));
            k_root_hi[0] = c1_1;
            k_root_hi[1] = c2_1;
        }
        const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
        const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
        if ((lane & 1) == 0) {
            cache_v[kv_cache_int8_quant_i4_code_index<Geometry>(page, kv_head, d0 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_int4_quant_code(v0, vinv),
                                 kv_cache_int4_quant_code(v0_hi, vinv));
            cache_v[kv_cache_int8_quant_i4_code_index<Geometry>(page, kv_head, d1 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_int4_quant_code(v1, vinv),
                                 kv_cache_int4_quant_code(v1_hi, vinv));
        }
    } else {
        std::int8_t c0 = 0, c1 = 0;
        if constexpr (E8Lattice) {
            float k0_scaled = k0 * kinv;
            float k1_scaled = k1 * kinv;
            e8_project_8d_warp(k0_scaled, k1_scaled, lane);
            // Deliberate half-coset approximation, identical to the decode side: the
            // projected E8 point may lie in the half-integral D8+0.5 coset, which the
            // 4-bit code cannot hold; the half is collapsed by the round below and no
            // coset bit is stored or reconstructed.
            c0 = static_cast<std::int8_t>(max(-8, min(7, static_cast<int>(rintf(k0_scaled)))));
            c1 = static_cast<std::int8_t>(max(-8, min(7, static_cast<int>(rintf(k1_scaled)))));
        } else {
            c0 = kv_cache_int4_quant_code(k0, kinv);
            c1 = kv_cache_int4_quant_code(k1, kinv);
        }
        const std::int8_t c0_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c0), 1));
        const std::int8_t c1_hi =
            static_cast<std::int8_t>(__shfl_down_sync(FullMask, static_cast<int>(c1), 1));
        const float v0_hi = __shfl_down_sync(FullMask, v0, 1);
        const float v1_hi = __shfl_down_sync(FullMask, v1, 1);
        if ((lane & 1) == 0) {
            cache_k[kv_cache_int8_quant_i4_code_index<Geometry>(page, kv_head, d0 / 2, page_off)] =
                kv_cache_pack_i4(c0, c0_hi);
            cache_k[kv_cache_int8_quant_i4_code_index<Geometry>(page, kv_head, d1 / 2, page_off)] =
                kv_cache_pack_i4(c1, c1_hi);
            cache_v[kv_cache_int8_quant_i4_code_index<Geometry>(page, kv_head, d0 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_int4_quant_code(v0, vinv),
                                 kv_cache_int4_quant_code(v0_hi, vinv));
            cache_v[kv_cache_int8_quant_i4_code_index<Geometry>(page, kv_head, d1 / 2, page_off)] =
                kv_cache_pack_i4(kv_cache_int4_quant_code(v1, vinv),
                                 kv_cache_int4_quant_code(v1_hi, vinv));
        }
    }
    if (lane == 0) {
        const std::int64_t scale_off =
            kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, group, page_off);
        scale_k[scale_off] = ksh;
        scale_v[scale_off] = vsh;
    }
}

} // namespace ninfer::ops
