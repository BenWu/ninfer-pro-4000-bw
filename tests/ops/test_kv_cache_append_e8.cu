// Append-side coverage for the E8 KV cache storage modes (RK4V4E8, RK2V4E8).
//
// test_kv_cache_append.cpp covers the bf16, int8-g64 and fp8-row256 planes but has no E8 case,
// so before this file the E8 encoder and its plane layout were only exercised indirectly: the
// attention test in ops/softmax_attention/causal_cache_e8.cu writes planes with this encoder and
// then decodes them again, which passes whenever encode and decode are wrong in mirrored ways.
// Everything checked here is compared against a host oracle instead.
//
// Both modes are verified byte for byte, but the key oracle differs. RK4V4E8 key codes are a
// deterministic sequence of rint/compare/select steps, so they are recomputed on the host.
// RK2V4E8 key codes come from the root cylinder encoder, which uses logf and sqrtf whose device
// results are not guaranteed to match host libm to the last ulp; a host reimplementation would
// test its own tolerance rather than the encoder. Those bytes are instead produced by running the
// warp encoder on device over the host-rotated key and host-derived scale, which still tests
// what this file is for, namely that append rotates, scales, and places correctly. The encoder
// itself is pinned by ninfer_e8_root_codec_test.
//
// Every plane is compared whole, so a write outside the appended positions fails too.

#include "ninfer/ops/kv_cache_append.h"
#include "ops/op_tester.h"

#include "ops/kv_cache/e8_root_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim = 256;
constexpr int kGroup   = 64;
constexpr int kGroups  = kHeadDim / kGroup;
constexpr int kPage    = 64;
constexpr int kPackedExtent = kHeadDim / 2;
constexpr int kRootExtent   = kHeadDim / 4;

std::size_t input_index(int d, int head, int token, int kv_heads) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(int leading_extent, int leading, int head, int position,
                        int physical_page, int kv_heads) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPage) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(physical_page)));
}

std::vector<std::uint8_t> patterned_bytes(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint8_t> out(count);
    std::uint32_t state = seed | 1u;
    for (std::size_t i = 0; i < count; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out[i] = static_cast<std::uint8_t>(state >> 11);
    }
    return out;
}

std::vector<std::uint16_t> patterned_halves(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> out(count);
    std::uint32_t state = seed | 1u;
    for (std::size_t i = 0; i < count; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Keep the pattern in the finite FP16 range so a stray compare reports a number.
        out[i] = static_cast<std::uint16_t>((state >> 11) & 0x3bffu);
    }
    return out;
}

// Host port of kv_cache_hadamard64 for one 64-dimension group, laid out as
// values[half * 32 + lane] so it matches the device's (d0 = group*64 + lane, d1 = d0 + 32) split.
// The butterfly order and the trailing 0.125 scale are reproduced exactly because the group scale
// derives from the absmax of the result.
void hadamard64_host(std::array<float, kGroup>& values) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        std::array<float, kGroup> next{};
        for (int lane = 0; lane < 32; ++lane) {
            const int partner = lane ^ offset;
            const bool hi     = (lane & offset) != 0;
            for (int half = 0; half < 2; ++half) {
                const float self  = values[static_cast<std::size_t>(half * 32 + lane)];
                const float other = values[static_cast<std::size_t>(half * 32 + partner)];
                next[static_cast<std::size_t>(half * 32 + lane)] = hi ? other - self : self + other;
            }
        }
        values = next;
    }
    for (int lane = 0; lane < 32; ++lane) {
        const float a = values[static_cast<std::size_t>(lane)];
        const float b = values[static_cast<std::size_t>(32 + lane)];
        values[static_cast<std::size_t>(lane)]      = (a + b) * 0.125f;
        values[static_cast<std::size_t>(32 + lane)] = (a - b) * 0.125f;
    }
}

// Sum eight values in the same pairing order as three XOR butterfly shuffles, so the floating
// point association matches the device reduction that decides which E8 coset wins.
float butterfly_sum8(const std::array<float, 8>& v) {
    std::array<float, 8> a{};
    std::array<float, 8> b{};
    for (int i = 0; i < 8; ++i) a[static_cast<std::size_t>(i)] = v[static_cast<std::size_t>(i)] +
                                                                 v[static_cast<std::size_t>(i ^ 1)];
    for (int i = 0; i < 8; ++i) b[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)] +
                                                                 a[static_cast<std::size_t>(i ^ 2)];
    return b[0] + b[4];
}

// Host port of e8_project_8d_warp_single over one 8-lane subgroup. The device version spreads
// eight consecutive dimensions across eight lanes, so a subgroup here is eight consecutive dims.
void e8_project_8d_host(std::array<float, 8>& x) {
    std::array<float, 8> f{};
    int sum_f = 0;
    for (int i = 0; i < 8; ++i) {
        f[static_cast<std::size_t>(i)] = std::rint(x[static_cast<std::size_t>(i)]);
        sum_f += static_cast<int>(f[static_cast<std::size_t>(i)]);
    }
    // The device tie-break keeps the lowest lane index on equal error; a strict > scan does the
    // same because it never displaces an earlier equal maximum.
    int worst       = 0;
    float max_error = std::fabs(x[0] - f[0]);
    for (int i = 1; i < 8; ++i) {
        const float error = std::fabs(x[static_cast<std::size_t>(i)] - f[static_cast<std::size_t>(i)]);
        if (error > max_error) {
            max_error = error;
            worst     = i;
        }
    }
    std::array<float, 8> d8 = f;
    if ((sum_f & 1) != 0) {
        d8[static_cast<std::size_t>(worst)] +=
            (x[static_cast<std::size_t>(worst)] >= f[static_cast<std::size_t>(worst)]) ? 1.0f : -1.0f;
    }

    std::array<float, 8> xs{};
    std::array<float, 8> fs{};
    int sum_fs = 0;
    for (int i = 0; i < 8; ++i) {
        xs[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] - 0.5f;
        fs[static_cast<std::size_t>(i)] = std::rint(xs[static_cast<std::size_t>(i)]);
        sum_fs += static_cast<int>(fs[static_cast<std::size_t>(i)]);
    }
    int worst_s       = 0;
    float max_error_s = std::fabs(xs[0] - fs[0]);
    for (int i = 1; i < 8; ++i) {
        const float error =
            std::fabs(xs[static_cast<std::size_t>(i)] - fs[static_cast<std::size_t>(i)]);
        if (error > max_error_s) {
            max_error_s = error;
            worst_s     = i;
        }
    }
    std::array<float, 8> coset1{};
    for (int i = 0; i < 8; ++i) coset1[static_cast<std::size_t>(i)] = fs[static_cast<std::size_t>(i)] + 0.5f;
    if ((sum_fs & 1) != 0) {
        coset1[static_cast<std::size_t>(worst_s)] +=
            (xs[static_cast<std::size_t>(worst_s)] >= fs[static_cast<std::size_t>(worst_s)]) ? 1.0f
                                                                                             : -1.0f;
    }

    std::array<float, 8> sq0{};
    std::array<float, 8> sq1{};
    for (int i = 0; i < 8; ++i) {
        const float e0 = x[static_cast<std::size_t>(i)] - d8[static_cast<std::size_t>(i)];
        const float e1 = x[static_cast<std::size_t>(i)] - coset1[static_cast<std::size_t>(i)];
        sq0[static_cast<std::size_t>(i)] = e0 * e0;
        sq1[static_cast<std::size_t>(i)] = e1 * e1;
    }
    const bool keep_d8 = butterfly_sum8(sq0) <= butterfly_sum8(sq1);
    for (int i = 0; i < 8; ++i) {
        x[static_cast<std::size_t>(i)] =
            keep_d8 ? d8[static_cast<std::size_t>(i)] : coset1[static_cast<std::size_t>(i)];
    }
}

int quant_i4_host(float value, float inverse_scale, int lo, int hi) {
    if (inverse_scale == 0.0f) { return 0; }
    const int code = static_cast<int>(std::nearbyint(value * inverse_scale));
    return std::clamp(code, lo, hi);
}

std::uint8_t pack_i4_host(int lo, int hi) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(lo) & 0x0fu) |
                                     ((static_cast<unsigned>(hi) & 0x0fu) << 4));
}

std::uint16_t half_bits(__half value) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Oracle for the RK2V4E8 key bytes. The root cylinder encoder uses logf and sqrtf, so a host
// reimplementation would not be reliably bit-exact; instead the production encoder is run here
// over the host-rotated key and the host-derived group scale, and its output is compared against
// what append stored. That keeps the check exact while still testing what this file is for:
// whether append rotates, scales, and places correctly. The encoder itself is pinned by
// ninfer_e8_root_codec_test.
//
// It has to be the warp encoder, not the scalar one. The two are not bit-identical: the scalar
// version accumulates the squared norm sequentially while the warp version sums it with a
// butterfly, and the different floating point association moves a few blocks across a radius or
// top-2 boundary. Append calls the warp version, so the oracle does too, with the same lane
// mapping: eight consecutive dimensions across eight consecutive lanes, four blocks per warp.
__global__ void encode_root_pairs_kernel(const float* __restrict__ rotated,
                                         const float* __restrict__ scales, int blocks,
                                         std::uint8_t* __restrict__ out) {
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = (static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                      static_cast<int>(threadIdx.x)) /
                     32;
    const int base_block = warp * 4;
    // Block count is always a multiple of four, so a warp is entirely in range or entirely out
    // and this guard never splits the warp across the shuffles inside the encoder.
    if (base_block >= blocks) { return; }
    const int block = base_block + (lane >> 3);
    const float value = rotated[static_cast<std::size_t>(warp) * 32 + lane];
    std::uint8_t root = 0;
    std::uint8_t rad_axis = 0;
    ninfer::ops::e8_encode_cylinder_8d_warp(value, scales[block], root, rad_axis, lane);
    if ((lane & 7) == 0) {
        out[2 * block]     = root;
        out[2 * block + 1] = rad_axis;
    }
}

std::vector<std::uint8_t> encode_root_pairs(const std::vector<float>& rotated,
                                            const std::vector<float>& scales) {
    const int blocks = static_cast<int>(scales.size());
    DeviceBuffer d_rotated = to_device(rotated);
    DeviceBuffer d_scales  = to_device(scales);
    DeviceBuffer d_out(static_cast<std::size_t>(blocks) * 2);
    const int warps_per_cta = 4;
    const int warps         = (blocks + 3) / 4;
    const int ctas          = (warps + warps_per_cta - 1) / warps_per_cta;
    encode_root_pairs_kernel<<<ctas, warps_per_cta * 32>>>(
        static_cast<const float*>(d_rotated.p), static_cast<const float*>(d_scales.p), blocks,
        static_cast<std::uint8_t*>(d_out.p));
    cuda_check_last_launch("encode_root_pairs_kernel");
    cuda_synchronize();
    return from_device<std::uint8_t>(d_out, static_cast<std::size_t>(blocks) * 2);
}

int e8_append_case(int kv_heads, bool root_mode, int tokens = 3) {
    const char* mode_name    = root_mode ? "rk2v4-e8" : "rk4v4-e8";
    const int k_extent       = root_mode ? kRootExtent : kPackedExtent;
    const int first_position = tokens >= 128 ? 61 : 63;
    const int logical_pages  = (first_position + tokens + kPage - 1) / kPage;
    const int physical_pages = 2 * logical_pages + 1;

    std::vector<std::int32_t> mapping(static_cast<std::size_t>(logical_pages));
    for (int page = 0; page < logical_pages; ++page) {
        mapping[static_cast<std::size_t>(page)] = 2 * page + 1;
    }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = first_position + token;
    }

    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kv_heads * tokens;
    const std::size_t k_count =
        static_cast<std::size_t>(k_extent) * kPage * kv_heads * physical_pages;
    const std::size_t v_count =
        static_cast<std::size_t>(kPackedExtent) * kPage * kv_heads * physical_pages;
    const std::size_t scale_count =
        static_cast<std::size_t>(kGroups) * kPage * kv_heads * physical_pages;

    std::vector<float> host_k(input_count);
    std::vector<float> host_v(input_count);
    fill_uniform(host_k, 0x51e8u + static_cast<std::uint32_t>(kv_heads), -0.75f, 0.75f);
    fill_uniform(host_v, 0x9c40u + static_cast<std::uint32_t>(kv_heads), -1.25f, 1.25f);
    round_to_bf16(host_k);
    round_to_bf16(host_v);
    // Zero one whole group of K and a different group of V on token 0, head 0. A zeroed group
    // has absmax 0, which must store a zero scale and all-zero codes rather than divide by zero.
    // For RK2V4E8 this is also the rad_idx == 0 path, which the encoder must not let leak into
    // the other 8-lane subgroups of the same warp.
    for (int i = 0; i < kGroup; ++i) {
        host_k[input_index(i, 0, 0, kv_heads)]           = 0.0f;
        host_v[input_index(kGroup + i, 0, 0, kv_heads)] = 0.0f;
    }

    std::vector<std::uint16_t> input_k(input_count);
    std::vector<std::uint16_t> input_v(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        input_k[i] = f32_to_bf16(host_k[i]);
        input_v[i] = f32_to_bf16(host_v[i]);
    }

    DeviceBuffer d_k         = to_device(input_k);
    DeviceBuffer d_v         = to_device(input_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_mapping   = to_device(mapping);
    Tensor k(d_k.p, DType::BF16, {kHeadDim, kv_heads, tokens});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kv_heads, tokens});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens});

    GuardedDeviceBuffer cache_k(k_count);
    GuardedDeviceBuffer cache_v(v_count);
    GuardedDeviceBuffer scale_k(scale_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer scale_v(scale_count * sizeof(std::uint16_t));

    // Seed every plane with a pattern; the expected buffers start from the same pattern so any
    // byte written outside an appended position shows up as a mismatch.
    auto expected_k       = patterned_bytes(k_count, 0x5ab3c1d0u);
    auto expected_v       = patterned_bytes(v_count, 0x0d1c3ba5u);
    auto expected_scale_k = patterned_halves(scale_count, 0x0f1e2d3cu);
    auto expected_scale_v = patterned_halves(scale_count, 0xc3d2e1f0u);
    cache_k.copy_from_host(expected_k.data(), expected_k.size());
    cache_v.copy_from_host(expected_v.data(), expected_v.size());
    scale_k.copy_from_host(expected_scale_k.data(), expected_scale_k.size() * sizeof(std::uint16_t));
    scale_v.copy_from_host(expected_scale_v.data(), expected_scale_v.size() * sizeof(std::uint16_t));

    PagedKVLayerView cache{
        .k_pages = Tensor(cache_k.data(), DType::U8, {k_extent, kPage, kv_heads, physical_pages}),
        .v_pages =
            Tensor(cache_v.data(), DType::U8, {kPackedExtent, kPage, kv_heads, physical_pages}),
        .k_scale_pages =
            Tensor(scale_k.data(), DType::FP16, {kGroups, kPage, kv_heads, physical_pages}),
        .v_scale_pages =
            Tensor(scale_v.data(), DType::FP16, {kGroups, kPage, kv_heads, physical_pages}),
        .block_table  = Tensor(d_mapping.p, DType::I32, {logical_pages}),
        .head_dim     = kHeadDim,
        .num_kv_heads = kv_heads,
        .storage      = root_mode ? KvCacheStorage::RK2V4E8 : KvCacheStorage::RK4V4E8,
    };

    // RK2V4E8 bookkeeping: the rotated key of every 8-dimension block, the group scale that goes
    // with it, and where its two bytes belong in the key plane.
    std::vector<float> root_values;
    std::vector<float> root_scales;
    std::vector<std::size_t> root_offsets;

    for (int token = 0; token < tokens; ++token) {
        const int position = positions[static_cast<std::size_t>(token)];
        const int page     = mapping[static_cast<std::size_t>(position / kPage)];
        for (int head = 0; head < kv_heads; ++head) {
            for (int group = 0; group < kGroups; ++group) {
                std::array<float, kGroup> k_row{};
                std::array<float, kGroup> v_row{};
                for (int i = 0; i < kGroup; ++i) {
                    const auto source = input_index(group * kGroup + i, head, token, kv_heads);
                    k_row[static_cast<std::size_t>(i)] = host_k[source];
                    v_row[static_cast<std::size_t>(i)] = host_v[source];
                }
                hadamard64_host(k_row);
                hadamard64_host(v_row);

                float k_absmax = 0.0f;
                float v_absmax = 0.0f;
                for (int i = 0; i < kGroup; ++i) {
                    k_absmax = std::max(k_absmax, std::fabs(k_row[static_cast<std::size_t>(i)]));
                    v_absmax = std::max(v_absmax, std::fabs(v_row[static_cast<std::size_t>(i)]));
                }
                const __half k_scale_half = __float2half_rn(k_absmax > 0.0f ? k_absmax / 7.0f : 0.0f);
                const __half v_scale_half = __float2half_rn(v_absmax > 0.0f ? v_absmax / 7.0f : 0.0f);
                const float k_scale       = __half2float(k_scale_half);
                const float v_scale       = __half2float(v_scale_half);
                const float k_inverse     = k_scale > 0.0f ? 1.0f / k_scale : 0.0f;
                const float v_inverse     = v_scale > 0.0f ? 1.0f / v_scale : 0.0f;

                const auto scale_target =
                    cache_index(kGroups, group, head, position, page, kv_heads);
                expected_scale_k[scale_target] = half_bits(k_scale_half);
                expected_scale_v[scale_target] = half_bits(v_scale_half);

                // V is packed 4-bit in both modes: byte p holds dimensions 2p (low) and 2p+1.
                for (int i = 0; i < kGroup; i += 2) {
                    const int d   = group * kGroup + i;
                    const int lo  = quant_i4_host(v_row[static_cast<std::size_t>(i)], v_inverse, -7, 7);
                    const int hi  = quant_i4_host(v_row[static_cast<std::size_t>(i + 1)], v_inverse, -7, 7);
                    const auto at = cache_index(kPackedExtent, d / 2, head, position, page, kv_heads);
                    expected_v[at] = pack_i4_host(lo, hi);
                }

                if (!root_mode) {
                    // RK4V4E8: scale, project each consecutive 8-dimension block onto E8, then
                    // round into the 4-bit range. The clamp is [-8, 7], not the [-7, 7] the plain
                    // packed-int4 codec uses, because the projection can land on -8.
                    std::array<float, kGroup> scaled{};
                    for (int i = 0; i < kGroup; ++i) {
                        scaled[static_cast<std::size_t>(i)] =
                            k_row[static_cast<std::size_t>(i)] * k_inverse;
                    }
                    for (int block = 0; block < kGroup / 8; ++block) {
                        std::array<float, 8> chunk{};
                        for (int i = 0; i < 8; ++i) {
                            chunk[static_cast<std::size_t>(i)] =
                                scaled[static_cast<std::size_t>(block * 8 + i)];
                        }
                        e8_project_8d_host(chunk);
                        for (int i = 0; i < 8; ++i) {
                            scaled[static_cast<std::size_t>(block * 8 + i)] =
                                chunk[static_cast<std::size_t>(i)];
                        }
                    }
                    for (int i = 0; i < kGroup; i += 2) {
                        const int d = group * kGroup + i;
                        const int lo = std::clamp(
                            static_cast<int>(std::rint(scaled[static_cast<std::size_t>(i)])), -8, 7);
                        const int hi = std::clamp(
                            static_cast<int>(std::rint(scaled[static_cast<std::size_t>(i + 1)])), -8,
                            7);
                        const auto at =
                            cache_index(kPackedExtent, d / 2, head, position, page, kv_heads);
                        expected_k[at] = pack_i4_host(lo, hi);
                    }
                } else {
                    // RK2V4E8: two bytes per 8-dimension block, encoded below once every block has
                    // been collected.
                    for (int block = 0; block < kGroup / 8; ++block) {
                        const int packed_d = group * (kGroup / 4) + block * 2;
                        for (int i = 0; i < 8; ++i) {
                            root_values.push_back(k_row[static_cast<std::size_t>(block * 8 + i)]);
                        }
                        root_scales.push_back(k_scale);
                        root_offsets.push_back(
                            cache_index(kRootExtent, packed_d, head, position, page, kv_heads));
                    }
                }
            }
        }
    }

    if (root_mode) {
        const auto oracle = encode_root_pairs(root_values, root_scales);
        for (std::size_t block = 0; block < root_offsets.size(); ++block) {
            expected_k[root_offsets[block]]     = oracle[block * 2];
            expected_k[root_offsets[block] + 1] = oracle[block * 2 + 1];
        }
    }

    ops::kv_cache_append(k, v, position_tensor, cache, nullptr);
    cuda_synchronize();

    const std::string label = std::string("kv_cache_append ") + mode_name +
                              " Hkv=" + std::to_string(kv_heads) + " T=" + std::to_string(tokens) +
                              " P=" + std::to_string(first_position);

    int failures = 0;
    const auto got_k       = from_device<std::uint8_t>(cache_k.data(), k_count);
    const auto got_v       = from_device<std::uint8_t>(cache_v.data(), v_count);
    const auto got_scale_k = from_device<std::uint16_t>(scale_k.data(), scale_count);
    const auto got_scale_v = from_device<std::uint16_t>(scale_v.data(), scale_count);

    failures += verify_exact((label + " v codes").c_str(), got_v, expected_v);
    failures += verify_exact((label + " k scales").c_str(), got_scale_k, expected_scale_k);
    failures += verify_exact((label + " v scales").c_str(), got_scale_v, expected_scale_v);

    failures += verify_exact((label + " k codes").c_str(), got_k, expected_k);

    if (root_mode) {
        // The zeroed K group on token 0 / head 0 must store a zero radius index in every one of
        // its eight blocks. That is the rad_idx == 0 path, which the encoder must not let leak
        // into the other 8-lane subgroups sharing its warp.
        const int zero_position = positions[0];
        const int zero_page     = mapping[static_cast<std::size_t>(zero_position / kPage)];
        std::vector<int> got_radius;
        std::vector<int> want_radius;
        for (int block = 0; block < kGroup / 8; ++block) {
            const auto at =
                cache_index(kRootExtent, block * 2, 0, zero_position, zero_page, kv_heads);
            got_radius.push_back(static_cast<int>(got_k[at + 1] >> 4));
            want_radius.push_back(0);
        }
        failures += verify_exact((label + " k zero-group radius").c_str(), got_radius, want_radius);
    }

    failures += cache_k.verify_guards((label + " k guards").c_str());
    failures += cache_v.verify_guards((label + " v guards").c_str());
    failures += scale_k.verify_guards((label + " k scale guards").c_str());
    failures += scale_v.verify_guards((label + " v scale guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "kv_cache_append_e8: SKIP (CUDA unavailable)\n";
        return 77;
    }

    int failures = 0;
    for (const int kv_heads : {4, 2}) {
        failures += e8_append_case(kv_heads, /*root_mode=*/false);
        failures += e8_append_case(kv_heads, /*root_mode=*/true);
    }
    // Cross a page boundary and exercise the multi-page block table.
    failures += e8_append_case(2, /*root_mode=*/false, 129);
    failures += e8_append_case(2, /*root_mode=*/true, 129);

    if (failures != 0) {
        std::cerr << "kv_cache_append_e8 failures=" << failures << '\n';
        return 1;
    }
    std::cout << "kv_cache_append_e8: PASS\n";
    return 0;
}
