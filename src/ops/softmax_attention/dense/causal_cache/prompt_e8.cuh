#pragma once

// E8-storage causal prompt kernel for the registered head geometries (RK4V4E8, RK2V4E8).
// Q is rotated per 64-dimension group (H64) before its G64 encoder, on top of the shared D256
// prepass the INT8 kernel applies. Cached K is stored H64-rotated as packed 4-bit
// lattice-projected codes (RK4V4E8) or 2-byte E8 root cylinder codes per 8 dimensions
// (RK2V4E8); cached V is always packed 4-bit. Producer warps decode the raw key codes into the
// INT8 QK arena, QK stays INT8 through m16n8k32.s8 Tensor Cores, and V is dequantized from the
// packed codes with packed FP16 arithmetic. Sixteen warps split each 16-row FP16 PV output
// across four 64-dimension slices. The PV output is H64-rotated per group; the launcher applies
// the inverse rotation to the attention result.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kv_cache/e8_lattice.cuh"
#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"
// Shared prompt-I8 layout constants and swizzle helpers (store_swz, p_swz, dequant_f16x8) live in
// prompt_i8.cuh; this E8 kernel reuses them and adds only the E8 smem sizing and decode helpers.
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"

#include <cstdint>

namespace ninfer::ops {

template <bool E8Root>
inline constexpr int kCausalPromptE8KeyRawBytes =
    kCausalPromptI8Bc * (E8Root ? kCausalPromptHeadDim / 4 : kCausalPromptHeadDim / 2);
inline constexpr int kCausalPromptE8ValueRawBytes =
    kCausalPromptI8Bc * (kCausalPromptHeadDim / 2);
// The packed-code staging replaces the INT8 V code arena, so the RK4V4E8 variant keeps the exact
// INT8 budget; the RK2V4E8 key arena is half-sized and the kernel uses 4 KiB less.
template <bool E8Root>
inline constexpr int kCausalPromptE8SmemBytes = kCausalPromptI8QBytes +
    kCausalPromptI8QScaleBytes + kCausalPromptI8KBytes + kCausalPromptE8KeyRawBytes<E8Root> +
    kCausalPromptE8ValueRawBytes + kCausalPromptI8VStageBytes + kCausalPromptI8PBytes +
    kCausalPromptI8ScaleBytes + kCausalPromptI8StatsBytes;

static_assert(kCausalPromptI8Groups == 4);
static_assert(kCausalPromptI8DConsumers == 4);
static_assert(kCausalPromptE8SmemBytes<false> == 92672);
static_assert(kCausalPromptE8SmemBytes<true> == 88576);

// Decode one 8-byte packed-4-bit key chunk (16 rotated dimensions, inside one G64 group) into
// the swizzled INT8 QK arena.
__device__ __forceinline__ void causal_prompt_e8_decode_packed_k16(const std::uint8_t* raw8,
                                                                  std::int8_t* k_tile, int row,
                                                                  int d) {
    const std::uint64_t raw = load_vec<std::uint64_t>(raw8);
    const auto* codes = reinterpret_cast<const std::uint8_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        causal_prompt_store_byte_swizzled<std::int8_t>(k_tile, row, d + 2 * i, kv_cache_unpack_i4(codes[i], 0));
        causal_prompt_store_byte_swizzled<std::int8_t>(k_tile, row, d + 2 * i + 1, kv_cache_unpack_i4(codes[i], 1));
    }
}

// Decode one 16-byte E8 root key chunk (one 64-dimension group: eight 8-dimension blocks) into
// the swizzled INT8 QK arena. The packed variant decodes four 8-byte chunks instead (producer
// handles it directly). Each 8-dimension block stores a consecutive (root, rad) byte pair, matching
// kv_cache_append_e8_group.
template <bool E8Root>
__device__ __forceinline__ void causal_prompt_e8_decode_key_group(const std::uint8_t* raw16,
                                                                 std::int8_t* k_tile, int row,
                                                                 int d) {
    if constexpr (E8Root) {
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            std::int8_t dec[8];
            e8_root_decode_8d_fast(raw16[2 * b], raw16[2 * b + 1], dec);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                causal_prompt_store_byte_swizzled<std::int8_t>(k_tile, row, d + 8 * b + i, dec[i]);
            }
        }
    }
}

template <typename Geometry, bool E8Root, typename Metadata>
__global__ __maxnreg__(120) void causal_attention_prompt_e8_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Br            = kCausalPromptI8Br;
    constexpr int Bc            = kCausalPromptI8Bc;
    constexpr int DB16          = kCausalPromptI8DB16;
    constexpr int Groups        = kCausalPromptI8Groups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / (kCausalPromptI8DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = kCausalPromptI8RowTiles;
    constexpr int VWorkerWarps  = kCausalPromptI8Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 2);
    static_assert(PVNtPerWarp == 8);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(smem_raw);
    float* q_scale    = reinterpret_cast<float*>(q_i8 + kCausalPromptI8QBytes);
    std::int8_t* k_i8 = reinterpret_cast<std::int8_t*>(reinterpret_cast<unsigned char*>(q_scale) +
                                                       kCausalPromptI8QScaleBytes);
    std::uint8_t* k_raw =
        reinterpret_cast<std::uint8_t*>(reinterpret_cast<unsigned char*>(k_i8) +
                                       kCausalPromptI8KBytes);
    std::uint8_t* v_raw   = k_raw + kCausalPromptE8KeyRawBytes<E8Root>;
    __half* v_f16         = reinterpret_cast<__half*>(v_raw + kCausalPromptE8ValueRawBytes);
    __half* p_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) + kCausalPromptI8VStageBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(p_s) + kCausalPromptI8PBytes);
    __half* v_scale_s    = k_scale_s + Bc * Groups;
    float* alpha_s       = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
    float* final_l_s     = alpha_s + Br;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    __nv_bfloat16* k_b16 = reinterpret_cast<__nv_bfloat16*>(k_i8);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 kCausalPromptI8Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    // Quantize Q cooperatively. One full warp rotates and encodes one D256 row at a time.
    for (int row = warp; row < Br; row += kCausalPromptI8Warps) {
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = 0.0f;
            if (row < tile_rows) {
                q_values[r] =
                    __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d, q0 + row)]);
            }
        }
        // E8 Q is H64-rotated per 64-dimension group only (matching the H64-rotated E8 K cache),
        // not the D256 prepass the INT8 path applies.
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            kv_cache_hadamard64(q_values[2 * grp], q_values[2 * grp + 1], FullMask);
        }

#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float absmax    = fmaxf(fabsf(x0), fabsf(x1));
            absmax          = warp_max(absmax, FullMask);
            const float qs  = absmax > 0.0f ? absmax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_prompt_store_byte_swizzled<std::int8_t>(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_prompt_store_byte_swizzled<std::int8_t>(q_i8, row, d1, kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale[row * Groups + grp] = qs; }
        }
    }
    __syncthreads();

    auto issue_kv_tile = [&](int tile_k0) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int key_l = tid; key_l < Bc; key_l += kCausalPromptI8Threads) {
            const int key = tile_k0 + key_l;
            __half* kd    = &k_scale_s[key_l * Groups];
            __half* vd    = &v_scale_s[key_l * Groups];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<8>(kd, &cache_k_scale[off]);
                ninfer::ops::cp_async<8>(vd, &cache_v_scale[off]);
            } else {
                store_vec(kd, make_int2(0, 0));
                store_vec(vd, make_int2(0, 0));
            }
        }
        if constexpr (E8Root) {
            // One 16-byte code chunk per (key row, 64-dimension group).
#pragma unroll 1
            for (int chunk = tid; chunk < Bc * Groups; chunk += kCausalPromptI8Threads) {
                const int key_l  = chunk / Groups;
                const int grp    = chunk - key_l * Groups;
                const int key    = tile_k0 + key_l;
                std::uint8_t* kd = &k_raw[key_l * (D / 4) + grp * (D / 4) / Groups];
                if (key <= max_query_abs) {
                    const std::int64_t off = kv_cache_int8_quant_e8_code_index<Geometry>(
                        physical_page, kv_head, grp * (D / 4) / Groups, key_l);
                    ninfer::ops::cp_async<16>(kd, &cache_k[off]);
                } else {
                    store_vec(kd, make_int4(0, 0, 0, 0));
                }
            }
        } else {
            // One 8-byte code chunk per (key row, 16 rotated dimensions).
#pragma unroll 1
            for (int chunk = tid; chunk < Bc * (D / 16); chunk += kCausalPromptI8Threads) {
                const int key_l = chunk / (D / 16);
                const int dc    = chunk - key_l * (D / 16);
                const int d     = dc * 16;
                const int key   = tile_k0 + key_l;
                std::uint8_t* kd = &k_raw[key_l * (D / 2) + d / 2];
                if (key <= max_query_abs) {
                    const std::int64_t off = kv_cache_int8_quant_i4_code_index<Geometry>(
                        physical_page, kv_head, d / 2, key_l);
                    ninfer::ops::cp_async<8>(kd, &cache_k[off]);
                } else {
                    store_vec(kd, make_int2(0, 0));
                }
            }
        }
        // Packed value codes: one 8-byte chunk per (key row, 16 rotated dimensions).
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += kCausalPromptI8Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = tile_k0 + key_l;
            std::uint8_t* vd = &v_raw[key_l * (D / 2) + d / 2];
            if (key <= max_query_abs) {
                const std::int64_t off = kv_cache_int8_quant_i4_code_index<Geometry>(
                    physical_page, kv_head, d / 2, key_l);
                ninfer::ops::cp_async<8>(vd, &cache_v[off]);
            } else {
                store_vec(vd, make_int2(0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    issue_kv_tile(0);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // Keeping exactly two group scales live is the spill-free 120-register point on SM120.
    // Groups 2/3 reload per key tile; retaining all four creates an 8-byte stack frame.
    float q_scale_r0[Groups - 2];
    float q_scale_r1[Groups - 2];
    if (warp < ProducerWarps) {
        const int scale_row0 = warp * 16 + gid;
        const int scale_row1 = scale_row0 + 8;
#pragma unroll
        for (int grp = 0; grp < Groups - 2; ++grp) {
            float qs0       = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
            float qs1       = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
            q_scale_r0[grp] = __shfl_sync(FullMask, qs0, gid * 4);
            q_scale_r1[grp] = __shfl_sync(FullMask, qs1, gid * 4);
        }
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = kb * Bc;
        // Decode the whole staged key tile into the INT8 QK arena. Every producer warp reads all
        // Bc rows through ldmatrix, so the decode must cover the tile cooperatively across the
        // CTA and be published by a barrier before QK runs (the INT8 kernel gets the same
        // ordering for free because its key codes land in k_i8 directly via cp.async).
        if constexpr (E8Root) {
            // One 16-byte chunk per (key row, 64-dimension group).
#pragma unroll 1
            for (int chunk = tid; chunk < Bc * Groups; chunk += kCausalPromptI8Threads) {
                const int key_l = chunk / Groups;
                const int grp   = chunk - key_l * Groups;
                const int key   = k0 + key_l;
                if (key <= max_query_abs) {
                    causal_prompt_e8_decode_key_group<E8Root>(
                        &k_raw[key_l * (D / 4) + grp * (D / 4) / Groups], k_i8, key_l,
                        grp * (D / Groups));
                } else {
#pragma unroll
                    for (int d = grp * (D / Groups); d < grp * (D / Groups) + D / Groups; ++d) {
                        causal_prompt_store_byte_swizzled<std::int8_t>(k_i8, key_l, d, 0);
                    }
                }
            }
        } else {
            // One 8-byte chunk per (key row, 16 dimensions).
#pragma unroll 1
            for (int chunk = tid; chunk < Bc * (D / 16); chunk += kCausalPromptI8Threads) {
                const int key_l = chunk / (D / 16);
                const int dc    = chunk - key_l * (D / 16);
                const int d     = dc * 16;
                const int key   = k0 + key_l;
                if (key <= max_query_abs) {
                    causal_prompt_e8_decode_packed_k16(&k_raw[key_l * (D / 2) + d / 2], k_i8,
                                                       key_l, d);
                } else {
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        causal_prompt_store_byte_swizzled<std::int8_t>(k_i8, key_l, d + i, 0);
                    }
                }
            }
        }
        __syncthreads();

        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                float qs0;
                float qs1;
                if (grp < Groups - 2) {
                    qs0 = q_scale_r0[grp];
                    qs1 = q_scale_r1[grp];
                } else {
                    const int scale_row0 = row_base + gid;
                    const int scale_row1 = scale_row0 + 8;
                    qs0                  = lid == 0 ? q_scale[scale_row0 * Groups + grp] : 0.0f;
                    qs1                  = lid == 0 ? q_scale[scale_row1 * Groups + grp] : 0.0f;
                    qs0                  = __shfl_sync(FullMask, qs0, gid * 4);
                    qs1                  = __shfl_sync(FullMask, qs1, gid * 4);
                }

                unsigned af[GroupKc][4];
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = grp * GroupKc + kk;
                    const int acol = k * 16 + a_coloff;
                    ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                                smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                                 causal_prompt_swz(row_base + a_rowoff, acol)]));
                }

#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = grp * GroupKc + kk;
                        const int brow = nt * 8 + b_rin;
                        const int bcol = k * 16 + b_koff;
                        unsigned bf[2];
                        ldmatrix_x2(bf[0], bf[1],
                                    smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                        mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                               bf[1]);
                    }
                    const int keya = nt * 8 + 2 * lid;
                    const int keyb = keya + 1;
                    float ks0      = 0.0f;
                    float ks1      = 0.0f;
                    if (gid == 0) {
                        ks0 = __half2float(k_scale_s[keya * Groups + grp]);
                        ks1 = __half2float(k_scale_s[keyb * Groups + grp]);
                    }
                    ks0          = __shfl_sync(FullMask, ks0, lid);
                    ks1          = __shfl_sync(FullMask, ks1, lid);
                    score[nt][0] = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                    score[nt][1] = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                    score[nt][2] = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                    score[nt][3] = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
                }
            }

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + causal_prompt_p_swz<kCausalPromptI8Bc>(row0, col0)] = __float2half_rn(p00);
                p_s[row0 * Bc + causal_prompt_p_swz<kCausalPromptI8Bc>(row0, col1)] = __float2half_rn(p01);
                p_s[row1 * Bc + causal_prompt_p_swz<kCausalPromptI8Bc>(row1, col0)] = __float2half_rn(p10);
                p_s[row1 * Bc + causal_prompt_p_swz<kCausalPromptI8Bc>(row1, col1)] = __float2half_rn(p11);
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 16); chunk += WorkerThreads) {
                const int key_l = chunk / (D / 16);
                const int dc    = chunk - key_l * (D / 16);
                const int d     = dc * 16;
                const int key   = k0 + key_l;
                const int grp   = d >> 6;
                // Sixteen chunks per key row against 32 lanes puts TWO key rows in one warp:
                // lanes 0-15 cover the even row, lanes 16-31 the odd one. The causal
                // `key <= max_query_abs` test below is therefore NOT warp-uniform whenever a
                // tile's last visible key lands on an even row, so the full-mask scale
                // broadcast has to happen above that branch - a __shfl_sync under a divergent
                // branch hangs the kernel, and did (roughly half of all prompt lengths).
                // A (row, group) scale is loaded by the first lane of its 4-lane quadrant and
                // broadcast within it, so the shuffle source stays in the same half-warp (same
                // key row); an out-of-range row still indexes inside v_scale_s.
                __half vs = __float2half_rn(0.0f);
                if ((lane & 3) == 0) { vs = v_scale_s[key_l * Groups + grp]; }
                vs = __shfl_sync(FullMask, vs, lane & ~3);
                // The 16 codes span two swizzle blocks (swz XOR-permutes 8-half groups). Store
                // each 8-half block at its own swizzled base; a single linear 16-half run would
                // skip group 0 for odd (row & 7) rows and overflow the row.
                __half* d0 = &v_f16[key_l * D + causal_prompt_swz(key_l, d)];
                __half* d1 = &v_f16[key_l * D + causal_prompt_swz(key_l, d + 8)];
                if (key <= max_query_abs) {
                    const __half2 s2  = __halves2half2(vs, vs);
                    const auto* src   = &v_raw[key_l * (D / 2) + d / 2];
                    std::int8_t codes[16];
                    kv_cache_unpack_i4x16(src, codes);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const __half2 value2 = __hmul2(
                            __floats2half2_rn(static_cast<float>(codes[2 * i]),
                                              static_cast<float>(codes[2 * i + 1])),
                            s2);
                        *reinterpret_cast<unsigned*>(d0 + 2 * i) =
                            *reinterpret_cast<const unsigned*>(&value2);
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const __half2 value2 = __hmul2(
                            __floats2half2_rn(static_cast<float>(codes[8 + 2 * i]),
                                              static_cast<float>(codes[8 + 2 * i + 1])),
                            s2);
                        *reinterpret_cast<unsigned*>(d1 + 2 * i) =
                            *reinterpret_cast<const unsigned*>(&value2);
                    }
                } else {
                    store_vec(d0, make_int4(0, 0, 0, 0));
                    store_vec(d1, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) { issue_kv_tile((kb + 1) * Bc); }

        const int row_tile = warp % kCausalPromptI8RowTiles;
        const int d_slice  = warp / kCausalPromptI8RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_rowoff) * Bc +
                                       causal_prompt_p_swz<kCausalPromptI8Bc>(row_base + a_rowoff, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                        vf[0], vf[1]);
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % kCausalPromptI8RowTiles;
    const int d_slice  = warp / kCausalPromptI8RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[causal_prompt_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                             kCausalPromptI8Threads);
}

} // namespace ninfer::ops
