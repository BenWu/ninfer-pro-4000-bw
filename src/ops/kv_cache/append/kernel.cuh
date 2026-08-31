#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry>
__device__ __forceinline__ void
kv_cache_append_full_fp8_row(const __nv_bfloat16* __restrict__ k,
                             const __nv_bfloat16* __restrict__ v,
                             std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                             __half* __restrict__ scale_k, __half* __restrict__ scale_v, int token,
                             int kv_head, int physical_page, int page_off, int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    float values[8];
    float local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(k[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (float value : values) { local_absmax = fmaxf(local_absmax, fabsf(value)); }
    const auto k_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_k[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_off)] =
            kv_cache_fp8_quant_code(values[r], k_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_k[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_off)] =
            k_quant.scale;
    }

    local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d  = lane + 32 * r;
        values[r]    = __bfloat162float(v[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
        local_absmax = fmaxf(local_absmax, fabsf(values[r]));
    }
    const auto v_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_v[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_off)] =
            kv_cache_fp8_quant_code(values[r], v_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_v[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_off)] =
            v_quant.scale;
    }
}

template <typename Geometry, typename Metadata>
__global__ void kv_cache_append_full_bf16_kernel(const __nv_bfloat16* __restrict__ k,
                                                 const __nv_bfloat16* __restrict__ v,
                                                 const std::int32_t* __restrict__ positions,
                                                 Metadata metadata,
                                                 __nv_bfloat16* __restrict__ cache_k,
                                                 __half* __restrict__ cache_v, std::int32_t width) {
    constexpr int VecElems = 8;
    const int tokens       = metadata.valid_tokens(width);
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                           (kKVCacheAppendFullHeadDim / VecElems);
    if (idx >= n) return;

    const int vec      = static_cast<int>(idx % (kKVCacheAppendFullHeadDim / VecElems));
    const int tmp      = static_cast<int>(idx / (kKVCacheAppendFullHeadDim / VecElems));
    const int kv_head  = tmp % Geometry::KVHeads;
    const int token    = tmp / Geometry::KVHeads;
    const int d        = vec * VecElems;
    const int position = positions[0] + token;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const std::int64_t src_off =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendFullHeadDim) *
                                           (kv_head + Geometry::KVHeads * token);
    const int4 k_value = load_vec<int4>(&k[src_off]);
    const int4 v_value = bf16x8_bits_to_f16x8_bits(load_vec<int4>(&v[src_off]));
    physical_page      = __shfl_sync(0xffffffffu, physical_page, 0);
    const std::int64_t cache_off =
        paged_kv_element_offset<kKVCacheAppendFullHeadDim, Geometry::KVHeads>(
            physical_page, kv_head, position & kPagedKVPageMask, d);
    store_vec(&cache_k[cache_off], k_value);
    store_vec(&cache_v[cache_off], v_value);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_kernel(const __nv_bfloat16* __restrict__ k,
                                         const __nv_bfloat16* __restrict__ v,
                                         const std::int32_t* __restrict__ positions,
                                         Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                         std::uint8_t* __restrict__ cache_v,
                                         __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                         std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                           physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_page_kernel(const __nv_bfloat16* __restrict__ k,
                                              const __nv_bfloat16* __restrict__ v,
                                              const std::int32_t* __restrict__ positions,
                                              Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                              std::uint8_t* __restrict__ cache_v,
                                              __half* __restrict__ scale_k,
                                              __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    const int position              = base_position + token;
    kv_cache_append_full_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                           physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i8_kernel(const __nv_bfloat16* __restrict__ k,
                                        const __nv_bfloat16* __restrict__ v,
                                        const std::int32_t* __restrict__ positions,
                                        Metadata metadata, std::int8_t* __restrict__ cache_k,
                                        std::int8_t* __restrict__ cache_v,
                                        __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                        std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page                        = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const int page_off              = position & kPagedKVPageMask;
    page                            = __shfl_sync(FullMask, page, 0);

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] =
            __bfloat162float(k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        const int d0                 = group * kKVCacheInt8Group + lane;
        const int d1                 = d0 + 32;
        const float k0               = k_values[2 * group];
        const float k1               = k_values[2 * group + 1];
        const std::int64_t src0      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
        const float v0               = __bfloat162float(v[src0]);
        const float v1               = __bfloat162float(v[src1]);
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs            = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const auto v_quant           = kv_cache_int8_quant_params(v_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_off =
                kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, group, page_off);
            scale_k[scale_off] = k_quant.scale;
            scale_v[scale_off] = v_quant.scale;
        }
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i8_page_kernel(const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ v,
                                             const std::int32_t* __restrict__ positions,
                                             Metadata metadata, std::int8_t* __restrict__ cache_k,
                                             std::int8_t* __restrict__ cache_v,
                                             __half* __restrict__ scale_k,
                                             __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);

    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] =
            __bfloat162float(k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        const int d0                 = group * kKVCacheInt8Group + lane;
        const int d1                 = d0 + 32;
        const float k0               = k_values[2 * group];
        const float k1               = k_values[2 * group + 1];
        const std::int64_t src0      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
        const float v0               = __bfloat162float(v[src0]);
        const float v1               = __bfloat162float(v[src1]);
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs            = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const auto v_quant           = kv_cache_int8_quant_params(v_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            physical_page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_offset =
                kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, group, page_off);
            scale_k[scale_offset] = k_quant.scale;
            scale_v[scale_offset] = v_quant.scale;
        }
    }
}

// ---------------------------------------------------------------------------
// E8 append family (RK4V4E8, RK2V4E8).
//
// Both K and V are rotated per 64-dimension group (H64) before quantization.
// V is always stored as packed 4-bit codes. The key plane is the packed 4-bit
// lattice-projected code (RK4V4E8) or the 2-byte E8 root cylinder code per
// 8 rotated dimensions (RK2V4E8). The per-group FP16 scale uses the /7
// boundary for both planes; the RK2V4E8 encoder additionally consumes the
// scale as its log-radius normalization reference.
// ---------------------------------------------------------------------------

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

template <typename Geometry, bool E8Lattice, bool E8Root, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_e8_kernel(const __nv_bfloat16* __restrict__ k,
                                        const __nv_bfloat16* __restrict__ v,
                                        const std::int32_t* __restrict__ positions,
                                        Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                        std::uint8_t* __restrict__ cache_v,
                                        __half* __restrict__ scale_k,
                                        __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps = 8;
    const int tokens    = metadata.valid_tokens(width);
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int unit      = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units     = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page                        = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const int page_off              = position & kPagedKVPageMask;
    page                            = __shfl_sync(0xffffffffu, page, 0);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        kv_cache_append_e8_group<Geometry, E8Lattice, E8Root>(
            k, v, kv_head, group, token, page, page_off, cache_k, cache_v, scale_k, scale_v, lane);
    }
}

template <typename Geometry, bool E8Lattice, bool E8Root, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_e8_page_kernel(const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ v,
                                             const std::int32_t* __restrict__ positions,
                                             Metadata metadata,
                                             std::uint8_t* __restrict__ cache_k,
                                             std::uint8_t* __restrict__ cache_v,
                                             __half* __restrict__ scale_k,
                                             __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(0xffffffffu, physical_page, 0);
    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        kv_cache_append_e8_group<Geometry, E8Lattice, E8Root>(
            k, v, kv_head, group, token, physical_page, page_off, cache_k, cache_v, scale_k,
            scale_v, lane);
    }
}

inline constexpr int kKVCacheAppendPrefixHeadDim = 128;
inline constexpr int kKVCacheAppendPrefixHeads   = 8;
inline constexpr int kKVCacheAppendPrefixWindow  = 4096;
inline constexpr int kKVCacheAppendPrefixPage    = 64;

__device__ __forceinline__ void kv_cache_append_prefix_copy_cyclic_unit(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, __half* __restrict__ cache_v, int token, int unit_in_token,
    int slot, int padded_capacity) {
    constexpr int ElementsPerUnit = 16;
    constexpr int UnitsPerHead    = kKVCacheAppendPrefixHeadDim / ElementsPerUnit;
    const int kv_head             = unit_in_token / UnitsPerHead;
    const int d                   = (unit_in_token - kv_head * UnitsPerHead) * ElementsPerUnit;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst = static_cast<std::int64_t>(d) +
                             static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                 (slot + static_cast<std::int64_t>(padded_capacity) * kv_head);

    const int4 k0 = *reinterpret_cast<const int4*>(&k[src]);
    const int4 v0 = bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src]));
    *reinterpret_cast<int4*>(&cache_k[dst]) = k0;
    *reinterpret_cast<int4*>(&cache_v[dst]) = v0;
    const int4 k1                           = *reinterpret_cast<const int4*>(&k[src + 8]);
    const int4 v1 = bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src + 8]));
    *reinterpret_cast<int4*>(&cache_k[dst + 8]) = k1;
    *reinterpret_cast<int4*>(&cache_v[dst + 8]) = v1;
}

__device__ __forceinline__ void kv_cache_append_prefix_copy_paged_unit(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, __half* __restrict__ cache_v, int token, int unit_in_token,
    int page_offset, int physical_page, int physical_pages) {
    constexpr int ElementsPerUnit = 16;
    constexpr int UnitsPerHead    = kKVCacheAppendPrefixHeadDim / ElementsPerUnit;
    const int kv_head             = unit_in_token / UnitsPerHead;
    const int d                   = (unit_in_token - kv_head * UnitsPerHead) * ElementsPerUnit;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
            (page_offset + kKVCacheAppendPrefixPage * (physical_page + physical_pages * kv_head));

    const int4 k0 = *reinterpret_cast<const int4*>(&k[src]);
    const int4 v0 = bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src]));
    *reinterpret_cast<int4*>(&cache_k[dst]) = k0;
    *reinterpret_cast<int4*>(&cache_v[dst]) = v0;
    const int4 k1                           = *reinterpret_cast<const int4*>(&k[src + 8]);
    const int4 v1 = bf16x8_bits_to_f16x8_bits(*reinterpret_cast<const int4*>(&v[src + 8]));
    *reinterpret_cast<int4*>(&cache_k[dst + 8]) = k1;
    *reinterpret_cast<int4*>(&cache_v[dst + 8]) = v1;
}

__global__ void kv_cache_append_prefix_cyclic_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ lanes, __nv_bfloat16* __restrict__ cache_k,
    __half* __restrict__ cache_v, int min_count, int max_count, int width, int padded_capacity) {
    constexpr int UnitsPerToken  = kKVCacheAppendPrefixHeads * 8;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    static_assert(TokensPerBlock * UnitsPerToken == 256);
    const int batch = static_cast<int>(blockIdx.y);
    const int count = counts[batch];
    if (count < min_count || count > max_count) return;

    constexpr std::int64_t ElementsPerToken =
        kKVCacheAppendPrefixHeadDim * kKVCacheAppendPrefixHeads;
    const std::int64_t input_offset = ElementsPerToken * width * batch;
    const std::int64_t cache_offset =
        ElementsPerToken * static_cast<std::int64_t>(padded_capacity) * lanes[batch];
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    cache_k += cache_offset;
    cache_v += cache_offset;

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    const int token         = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    if (token >= count) return;
    const int position = positions[token];
    const int slot     = position & (kKVCacheAppendPrefixWindow - 1);
    kv_cache_append_prefix_copy_cyclic_unit(k, v, cache_k, cache_v, token, unit_in_token, slot,
                                            padded_capacity);
}

__global__ void kv_cache_append_prefix_paged_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ table_rows, __nv_bfloat16* __restrict__ cache_k,
    __half* __restrict__ cache_v, const std::int32_t* __restrict__ block_tables, int physical_pages,
    int logical_pages, int min_count, int max_count, int width) {
    constexpr int UnitsPerToken  = kKVCacheAppendPrefixHeads * 8;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    static_assert(TokensPerBlock * UnitsPerToken == 256);
    const int batch = static_cast<int>(blockIdx.y);
    const int count = counts[batch];
    if (count < min_count || count > max_count) return;

    constexpr std::int64_t ElementsPerToken =
        kKVCacheAppendPrefixHeadDim * kKVCacheAppendPrefixHeads;
    const std::int64_t input_offset = ElementsPerToken * width * batch;
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(logical_pages) * table_rows[batch];

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    const int lane          = local & 31;
    const int token         = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    int position            = 0;
    int physical_page       = 0;
    if (lane == 0 && token < count) {
        position      = positions[token];
        physical_page = block_table[position >> 6];
    }
    position      = __shfl_sync(0xffffffffu, position, 0);
    physical_page = __shfl_sync(0xffffffffu, physical_page, 0);
    if (token < count) {
        kv_cache_append_prefix_copy_paged_unit(k, v, cache_k, cache_v, token, unit_in_token,
                                               position & (kKVCacheAppendPrefixPage - 1),
                                               physical_page, physical_pages);
    }
}

} // namespace ninfer::ops
