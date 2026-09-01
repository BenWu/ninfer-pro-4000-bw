// E8 KV-cache coverage for the causal-cache Softmax Attention entries (RK4V4E8, RK2V4E8).
//
// The ninfer-4090 fork this storage family came from has no op test at this level: its only E8
// coverage is the root-cylinder encoder (ported here as ninfer_e8_root_codec_test). Two defects
// in the E8 attention kernels reached the engine as a result - the prompt kernel decoded 16 of
// its 64 staged key rows, and the small-T V loader wrote FP16 bit patterns into a BF16 tile - so
// this suite exists to pin the contract those defects broke.
//
// Contract under test: the entry's output is softmax attention over the values the cache
// actually holds. The reference reads the code and scale planes back after the call and decodes
// them on the host (unpack -> per-group scale -> inverse H64), reproducing no tile, swizzle,
// MMA, split, or reduction tree from the kernels. Cache *encoding* is deliberately not under
// test here; that belongs to ninfer_kv_cache_append_test and ninfer_e8_root_codec_test.
//
// This is a .cu translation unit only so the RK2V4E8 root decode can be tabulated once from the
// production codec header; the tabulation is a black-box dump of e8_root_decode_8d_fast over all
// 65536 code pairs, and every other step of the reference is ordinary host code.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kGroup       = 64;
constexpr std::int32_t kGroups      = kHeadDim / kGroup;
constexpr std::int32_t kVExtent     = kHeadDim / 2; // packed 4-bit values, two per byte
constexpr float kAttentionScale     = 0.0625f;
constexpr std::uint16_t kOutputCanary = 0x7fc1u;

// The reference consumes the decoded cache, so quantization error is not part of the residual.
// What remains is Q's int8 per-group quantization, half-precision PV accumulation, and - unique
// to E8 - the inverse H64 rotation the launcher applies to the attention result. That rotation
// reads BF16, mixes 64 components, and writes BF16, so each output element inherits roughly
// sqrt(64) BF16 roundings of the rotated vector; the gross term is sized for it and is widest on
// a single-key tile, where the output is one decoded V row and nothing averages the error down.
//
// Measured over all 72 comparisons in this suite on an RTX PRO 4000 Blackwell (sm_120a):
// max relative_l2 3.65e-3, max absolute error 5.62e-3 against a max reference of 1.16, and max
// absolute 1.59e-3 against a max reference of 0.27. The bounds below leave ~1.5x on each term.
constexpr ReductionCriterion kAttentionE8Criterion{
    /*relative_l2*/ 5.0e-3,
    /*gross_absolute*/ 1.5e-3,
    /*gross_relative_to_max_reference*/ 6.0e-3,
};

struct Mode {
    const char* name;
    KvCacheStorage storage;
    std::int32_t k_extent;
};

constexpr Mode kModes[] = {
    {"rk4v4-e8", KvCacheStorage::RK4V4E8, kHeadDim / 2},
    {"rk2v4-e8", KvCacheStorage::RK2V4E8, kHeadDim / 4},
};

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;
};

constexpr Geometry kGeometries[] = {
    {"d256-h24-kv4", 24, 4},
    {"d256-h16-kv2", 16, 2},
};

ops::AttentionHeadGeometry op_geometry(const Geometry& geometry) {
    return {kHeadDim, geometry.q_heads, geometry.kv_heads};
}

enum class MappingPattern { Identity, Offset, Fragmented };

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return logical_pages;
    case MappingPattern::Offset:
        return logical_pages + 2;
    case MappingPattern::Fragmented:
        return 2 * logical_pages + 1;
    }
    return 0;
}

std::vector<std::int32_t> make_block_table(std::int32_t logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    for (std::int32_t page = 0; page < logical_pages; ++page) {
        switch (pattern) {
        case MappingPattern::Identity:
            table[static_cast<std::size_t>(page)] = page;
            break;
        case MappingPattern::Offset:
            table[static_cast<std::size_t>(page)] = page + 1;
            break;
        case MappingPattern::Fragmented:
            table[static_cast<std::size_t>(page)] = 2 * page + 1;
            break;
        }
    }
    return table;
}

std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

// --- host value helpers -----------------------------------------------------

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exponent  = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exponent == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exponent - 15);
    }
    return negative ? -magnitude : magnitude;
}

int unpack_i4(std::uint8_t packed, int high) {
    const unsigned nibble = high != 0 ? (packed >> 4) & 0x0fu : packed & 0x0fu;
    return (nibble & 0x8u) != 0 ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

// Normalized 64-dimension Sylvester-Hadamard rotation, the host image of kv_cache_hadamard64.
// The device primitive holds dimension `lane` and `lane + 32` of the group in one thread, runs a
// five-stage XOR butterfly over both halves, and folds them with the final (sum, difference)
// stage. H64 is symmetric and orthonormal, so the same routine is its own inverse.
void hadamard64(float* group) {
    std::array<float, 32> low{};
    std::array<float, 32> high{};
    for (int i = 0; i < 32; ++i) {
        low[static_cast<std::size_t>(i)]  = group[i];
        high[static_cast<std::size_t>(i)] = group[i + 32];
    }
    for (int offset = 1; offset < 32; offset <<= 1) {
        std::array<float, 32> next_low  = low;
        std::array<float, 32> next_high = high;
        for (int i = 0; i < 32; ++i) {
            const auto self    = static_cast<std::size_t>(i);
            const auto partner = static_cast<std::size_t>(i ^ offset);
            if ((i & offset) != 0) {
                next_low[self]  = low[partner] - low[self];
                next_high[self] = high[partner] - high[self];
            } else {
                next_low[self]  = low[self] + low[partner];
                next_high[self] = high[self] + high[partner];
            }
        }
        low  = next_low;
        high = next_high;
    }
    for (int i = 0; i < 32; ++i) {
        const auto index = static_cast<std::size_t>(i);
        group[i]         = (low[index] + high[index]) * 0.125f;
        group[i + 32]    = (low[index] - high[index]) * 0.125f;
    }
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return values;
}

// --- RK2V4E8 root decode, tabulated once from the production codec ----------

__global__ void e8_root_decode_table_kernel(std::int8_t* out) {
    const int pair = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                     static_cast<int>(threadIdx.x);
    if (pair >= 256 * 256) { return; }
    std::int8_t decoded[8];
    ops::e8_root_decode_8d_fast(static_cast<std::uint8_t>(pair >> 8),
                                static_cast<std::uint8_t>(pair & 0xff), decoded);
    for (int i = 0; i < 8; ++i) { out[static_cast<std::size_t>(pair) * 8 + i] = decoded[i]; }
}

const std::vector<std::int8_t>& e8_root_decode_table() {
    static const std::vector<std::int8_t> table = [] {
        constexpr int kPairs = 256 * 256;
        DeviceBuffer device(static_cast<std::size_t>(kPairs) * 8);
        constexpr int kBlock = 256;
        e8_root_decode_table_kernel<<<kPairs / kBlock, kBlock>>>(
            static_cast<std::int8_t*>(device.p));
        cuda_check_last_launch("e8_root_decode_table_kernel");
        cuda_synchronize();
        return from_device<std::int8_t>(device, static_cast<std::size_t>(kPairs) * 8);
    }();
    return table;
}

// --- paged plane addressing -------------------------------------------------

std::size_t plane_index(std::int32_t leading_extent, std::int32_t leading, std::int32_t head,
                        std::int32_t position, std::int32_t physical_page, std::int32_t kv_heads) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(kv_heads) *
                         static_cast<std::size_t>(physical_page)));
}

std::size_t logical_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                          std::int32_t position) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::size_t>(position));
}

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::size_t>(token));
}

// Original-coordinate K and V, indexed [d, kv_head, position].
struct LogicalCache {
    std::vector<float> k;
    std::vector<float> v;
};

// --- device cache -----------------------------------------------------------

class E8DeviceCache {
public:
    E8DeviceCache(const Geometry& geometry, const Mode& mode, std::int32_t logical_capacity,
                  MappingPattern mapping)
        : geometry_(geometry), mode_(mode), logical_capacity_(logical_capacity),
          logical_pages_(logical_capacity / kPagedKVPageSize),
          physical_pages_(physical_page_count(logical_pages_, mapping)),
          block_table_host_(make_block_table(logical_pages_, mapping)),
          page_slots_(static_cast<std::size_t>(kPagedKVPageSize) * geometry.kv_heads *
                      physical_pages_),
          k_(page_slots_ * static_cast<std::size_t>(mode.k_extent)),
          v_(page_slots_ * static_cast<std::size_t>(kVExtent)),
          k_scale_(page_slots_ * kGroups * sizeof(std::uint16_t)),
          v_scale_(page_slots_ * kGroups * sizeof(std::uint16_t)),
          block_table_(block_table_host_.size() * sizeof(std::int32_t)) {
        k_.fill(0);
        v_.fill(0);
        k_scale_.fill(0);
        v_scale_.fill(0);
        block_table_.copy_from_host(block_table_host_.data(),
                                    block_table_host_.size() * sizeof(std::int32_t));
    }

    PagedKVLayerView view() {
        PagedKVLayerView result;
        result.k_pages     = Tensor(k_.data(), DType::U8,
                                    {mode_.k_extent, kPagedKVPageSize, geometry_.kv_heads,
                                     physical_pages_});
        result.v_pages     = Tensor(v_.data(), DType::U8,
                                    {kVExtent, kPagedKVPageSize, geometry_.kv_heads,
                                     physical_pages_});
        result.k_scale_pages = Tensor(k_scale_.data(), DType::FP16,
                                      {kGroups, kPagedKVPageSize, geometry_.kv_heads,
                                       physical_pages_});
        result.v_scale_pages = Tensor(v_scale_.data(), DType::FP16,
                                      {kGroups, kPagedKVPageSize, geometry_.kv_heads,
                                       physical_pages_});
        result.block_table   = Tensor(block_table_.data(), DType::I32, {logical_pages_});
        result.head_dim      = kHeadDim;
        result.num_kv_heads  = geometry_.kv_heads;
        result.storage       = mode_.storage;
        return result;
    }

    PagedKVBatchLayerView batch_view() {
        const PagedKVLayerView direct = view();
        return {
            .k_pages       = direct.k_pages,
            .v_pages       = direct.v_pages,
            .k_scale_pages = direct.k_scale_pages,
            .v_scale_pages = direct.v_scale_pages,
            .block_tables  = direct.block_table.view({logical_pages_, 1}),
            .head_dim      = direct.head_dim,
            .num_kv_heads  = direct.num_kv_heads,
            .storage       = direct.storage,
        };
    }

    // Write `positions.size()` consecutive positions through the production append so the fixture
    // never invents E8 bytes of its own.
    void fill_history(const std::vector<float>& k, const std::vector<float>& v,
                      const std::vector<std::int32_t>& positions) {
        const auto tokens                      = static_cast<std::int32_t>(positions.size());
        const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
        const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
        DeviceBuffer dk        = to_device(k_bits);
        DeviceBuffer dv        = to_device(v_bits);
        DeviceBuffer dp        = to_device(positions);
        const Tensor tk(dk.p, DType::BF16, {kHeadDim, geometry_.kv_heads, tokens});
        const Tensor tv(dv.p, DType::BF16, {kHeadDim, geometry_.kv_heads, tokens});
        const Tensor tp(dp.p, DType::I32, {tokens});
        PagedKVLayerView cache = view();
        ops::kv_cache_append(tk, tv, tp, cache, nullptr);
        cuda_synchronize();
    }

    // Read the planes back and decode them into original-coordinate K and V. This is the value
    // the attention entries must reproduce.
    LogicalCache decode(std::int32_t keys) const {
        const auto k_codes  = copy_plane<std::uint8_t>(k_, page_slots_ * mode_.k_extent);
        const auto v_codes  = copy_plane<std::uint8_t>(v_, page_slots_ * kVExtent);
        const auto k_scales = copy_plane<std::uint16_t>(k_scale_, page_slots_ * kGroups);
        const auto v_scales = copy_plane<std::uint16_t>(v_scale_, page_slots_ * kGroups);
        const std::vector<std::int8_t>& roots = e8_root_decode_table();

        LogicalCache logical;
        const std::size_t elements = static_cast<std::size_t>(kHeadDim) * geometry_.kv_heads *
                                     static_cast<std::size_t>(keys);
        logical.k.assign(elements, 0.0f);
        logical.v.assign(elements, 0.0f);

        for (std::int32_t head = 0; head < geometry_.kv_heads; ++head) {
            for (std::int32_t position = 0; position < keys; ++position) {
                const std::int32_t page =
                    block_table_host_[static_cast<std::size_t>(position / kPagedKVPageSize)];
                for (std::int32_t group = 0; group < kGroups; ++group) {
                    const std::size_t scale_offset =
                        plane_index(kGroups, group, head, position, page, geometry_.kv_heads);
                    const float k_scale = f16_bits_to_f32(k_scales[scale_offset]);
                    const float v_scale = f16_bits_to_f32(v_scales[scale_offset]);

                    std::array<float, kGroup> k_rotated{};
                    if (mode_.storage == KvCacheStorage::RK2V4E8) {
                        // Eight consecutive (root, radius/axis) byte pairs per 64-dimension group.
                        for (int block = 0; block < 8; ++block) {
                            const std::int32_t base = group * (kGroup / 4) + 2 * block;
                            const std::uint8_t root = k_codes[plane_index(
                                mode_.k_extent, base, head, position, page, geometry_.kv_heads)];
                            const std::uint8_t radius_axis =
                                k_codes[plane_index(mode_.k_extent, base + 1, head, position, page,
                                                    geometry_.kv_heads)];
                            const std::int8_t* decoded =
                                &roots[(static_cast<std::size_t>(root) * 256u + radius_axis) * 8u];
                            for (int i = 0; i < 8; ++i) {
                                k_rotated[static_cast<std::size_t>(8 * block + i)] =
                                    static_cast<float>(decoded[i]) * k_scale;
                            }
                        }
                    } else {
                        for (std::int32_t d = 0; d < kGroup; ++d) {
                            const std::int32_t full = group * kGroup + d;
                            const std::uint8_t byte = k_codes[plane_index(
                                mode_.k_extent, full / 2, head, position, page,
                                geometry_.kv_heads)];
                            k_rotated[static_cast<std::size_t>(d)] =
                                static_cast<float>(unpack_i4(byte, full & 1)) * k_scale;
                        }
                    }

                    std::array<float, kGroup> v_rotated{};
                    for (std::int32_t d = 0; d < kGroup; ++d) {
                        const std::int32_t full = group * kGroup + d;
                        const std::uint8_t byte = v_codes[plane_index(
                            kVExtent, full / 2, head, position, page, geometry_.kv_heads)];
                        v_rotated[static_cast<std::size_t>(d)] =
                            static_cast<float>(unpack_i4(byte, full & 1)) * v_scale;
                    }

                    // K and V are stored H64-rotated per group; H64 is its own inverse.
                    hadamard64(k_rotated.data());
                    hadamard64(v_rotated.data());
                    for (std::int32_t d = 0; d < kGroup; ++d) {
                        const std::size_t target =
                            logical_index(geometry_, head, group * kGroup + d, position);
                        logical.k[target] = k_rotated[static_cast<std::size_t>(d)];
                        logical.v[target] = v_rotated[static_cast<std::size_t>(d)];
                    }
                }
            }
        }
        return logical;
    }

    std::vector<std::uint8_t> raw_planes() const {
        std::vector<std::uint8_t> bytes;
        const auto append = [&bytes](const GuardedDeviceBuffer& buffer) {
            std::vector<std::uint8_t> chunk(buffer.bytes());
            buffer.copy_to_host(chunk.data(), chunk.size());
            bytes.insert(bytes.end(), chunk.begin(), chunk.end());
        };
        append(k_);
        append(v_);
        append(k_scale_);
        append(v_scale_);
        return bytes;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
        failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        failures += block_table_.verify_guards((label + " block-table").c_str());
        std::vector<std::int32_t> table(block_table_host_.size());
        block_table_.copy_to_host(table.data(), table.size() * sizeof(std::int32_t));
        failures += verify_exact((label + " block-table unchanged").c_str(), table,
                                 block_table_host_);
        return failures;
    }

private:
    template <typename T>
    static std::vector<T> copy_plane(const GuardedDeviceBuffer& buffer, std::size_t count) {
        std::vector<T> values(count);
        buffer.copy_to_host(values.data(), count * sizeof(T));
        return values;
    }

    Geometry geometry_;
    Mode mode_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_table_host_;
    std::size_t page_slots_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_table_;
};

std::vector<double> ideal_attention(const Geometry& geometry, const std::vector<float>& q,
                                    const LogicalCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const auto tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));
    naive_dense_softmax_attention(
        op_geometry(geometry), tokens, positions.back() + 1, static_cast<double>(kAttentionScale),
        [&](int d, int head, int token) {
            return static_cast<double>(q[q_index(geometry, head, d, token)]);
        },
        [&](int d, int head, int position) {
            return static_cast<double>(cache.k[logical_index(geometry, head, d, position)]);
        },
        [&](int d, int head, int position) {
            return static_cast<double>(cache.v[logical_index(geometry, head, d, position)]);
        },
        [&](int token, int position) {
            return position <= positions[static_cast<std::size_t>(token)];
        },
        [&](int d, int head, int token, double value) {
            output[q_index(geometry, head, d, token)] = value;
        });
    return output;
}

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
};

std::string case_label(const char* entry, const Geometry& geometry, const Mode& mode,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + mode.name +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max);
}

std::vector<std::int32_t> iota_positions(std::int32_t begin, std::int32_t count) {
    std::vector<std::int32_t> positions(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) { positions[static_cast<std::size_t>(i)] = begin + i; }
    return positions;
}

// A1: the entry appends this round's K/V into the cache and then attends over the whole history.
int run_append_case(const Geometry& geometry, const Mode& mode, const AttentionCase& test_case,
                    MappingPattern mapping) {
    const std::int32_t total = test_case.base + test_case.tokens;
    const std::int32_t max_context =
        static_cast<std::int32_t>(std::max<std::uint32_t>(
            static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::int32_t capacity = align_up_page(max_context);
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads *
                                    static_cast<std::size_t>(test_case.tokens);

    E8DeviceCache cache(geometry, mode, capacity, mapping);
    // Every position the entry may read carries real codes, so an uninitialized page can never
    // pass by accident.
    cache.fill_history(
        make_bf16_values(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * capacity,
                         test_case.seed + 10u, -0.25f, 0.25f),
        make_bf16_values(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * capacity,
                         test_case.seed + 11u, -1.0f, 1.0f),
        iota_positions(0, capacity));

    const std::vector<float> q =
        make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    const std::vector<float> k =
        make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    const std::vector<float> v =
        make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    const std::vector<std::int32_t> positions = iota_positions(test_case.base, test_case.tokens);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    const std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

    const Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    const Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    const Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    const Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                         test_case.envelope_max};
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), cache.view().storage, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::causal_softmax_attention(tq, tk, tv, tp, Tensor{}, ttable_row, op_geometry(geometry),
                                  kAttentionScale, cache.batch_view(), envelope, workspace, tout,
                                  nullptr);
    cuda_synchronize();

    // The cache now holds this round's tokens too; decode it and require the output to be
    // attention over exactly those values.
    const LogicalCache logical          = cache.decode(total);
    const std::vector<double> reference = ideal_attention(geometry, q, logical, positions);
    const std::string label = case_label("causal_softmax_attention", geometry, mode, test_case,
                                         mapping);
    std::vector<std::uint16_t> output_bits(q_bits.size());
    dout.copy_to_host(output_bits.data(), output_bits.size() * sizeof(std::uint16_t));

    int failures = verify_reduction(label, bf16_bits_to_double(output_bits), reference,
                                    kAttentionE8Criterion);
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(dq.data(), q_bits.size()), q_bits);
    failures += verify_exact((label + " k unchanged").c_str(),
                             from_device<std::uint16_t>(dk.data(), k_bits.size()), k_bits);
    failures += verify_exact((label + " v unchanged").c_str(),
                             from_device<std::uint16_t>(dv.data(), v_bits.size()), v_bits);
    failures += verify_exact((label + " positions unchanged").c_str(),
                             from_device<std::int32_t>(dp.data(), positions.size()), positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    failures += cache.verify_guards(label);
    return failures;
}

// A3: attention over an untouched cache. The entry must not write the KV planes.
int run_cached_case(const Geometry& geometry, const Mode& mode, const AttentionCase& test_case,
                    MappingPattern mapping) {
    const std::int32_t total = test_case.base + test_case.tokens;
    const std::int32_t max_context =
        static_cast<std::int32_t>(std::max<std::uint32_t>(
            static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::int32_t capacity = align_up_page(max_context);
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads *
                                   static_cast<std::size_t>(test_case.tokens);

    E8DeviceCache cache(geometry, mode, capacity, mapping);
    cache.fill_history(
        make_bf16_values(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * capacity,
                         test_case.seed + 10u, -0.25f, 0.25f),
        make_bf16_values(static_cast<std::size_t>(kHeadDim) * geometry.kv_heads * capacity,
                         test_case.seed + 11u, -1.0f, 1.0f),
        iota_positions(0, capacity));

    const std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    const std::vector<std::int32_t> positions = iota_positions(test_case.base, test_case.tokens);
    const LogicalCache logical                = cache.decode(total);
    const std::vector<double> reference       = ideal_attention(geometry, q, logical, positions);
    const std::vector<std::uint8_t> planes_before = cache.raw_planes();

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::vector<std::uint16_t> canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

    const Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::CausalAttentionExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                                         test_case.envelope_max};
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        op_geometry(geometry), cache.view().storage, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::causal_softmax_attention_cached(tq, tp, op_geometry(geometry), kAttentionScale,
                                         cache.view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label =
        case_label("causal_softmax_attention_cached", geometry, mode, test_case, mapping);
    std::vector<std::uint16_t> output_bits(q_bits.size());
    dout.copy_to_host(output_bits.data(), output_bits.size() * sizeof(std::uint16_t));

    int failures = verify_reduction(label, bf16_bits_to_double(output_bits), reference,
                                    kAttentionE8Criterion);
    failures += verify_exact((label + " cache unchanged").c_str(), cache.raw_planes(),
                             planes_before);
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(dq.data(), q_bits.size()), q_bits);
    failures += verify_exact((label + " positions unchanged").c_str(),
                             from_device<std::int32_t>(dp.data(), positions.size()), positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    failures += cache.verify_guards(label);
    return failures;
}

int run_mode(const Geometry& geometry, const Mode& mode) {
    int failures = 0;

    // T <= 6 takes the fused small-T route (attention and the E8 append share one kernel);
    // larger T takes the prompt kernel. Both are covered, and the key windows deliberately
    // straddle one 64-key tile so a kernel that decodes only part of a staged tile fails.
    const AttentionCase append_cases[] = {
        {1, 0, 1, 1001u},   {6, 61, 67, 1002u},  {6, 129, 200, 1003u},
        {17, 31, 48, 1004u}, {66, 63, 129, 1005u},
    };
    const AttentionCase cached_cases[] = {
        {1, 31, 32, 1101u},  {1, 128, 129, 1102u}, {6, 17, 23, 1103u},
        {7, 17, 512, 1104u}, {17, 31, 48, 1105u},
        // Parity pair for the prompt route. The V decode puts two key rows in one warp, so a
        // tile whose last visible key lands on an even row splits the warp across the causal
        // branch; a full-mask shuffle inside that branch hangs. The first case ends at absolute
        // position 16 (even, the shape that hung), the second at 17 (odd) as the control.
        {17, 0, 17, 1106u},  {17, 1, 18, 1107u},
    };

    for (const MappingPattern mapping :
         {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
        failures += run_append_case(geometry, mode, {6, 61, 67, 1201u}, mapping);
        failures += run_cached_case(geometry, mode, {1, 128, 129, 1202u}, mapping);
    }
    for (const AttentionCase& test_case : append_cases) {
        failures += run_append_case(geometry, mode, test_case, MappingPattern::Identity);
    }
    for (const AttentionCase& test_case : cached_cases) {
        failures += run_cached_case(geometry, mode, test_case, MappingPattern::Identity);
    }
    // One fragmented long-window case per mode keeps the multi-page split path honest.
    failures += run_cached_case(geometry, mode, {4, 511, 640, 1301u}, MappingPattern::Fragmented);
    failures += run_append_case(geometry, mode, {70, 449, 640, 1302u}, MappingPattern::Fragmented);
    return failures;
}

} // namespace

int run_softmax_attention_causal_cache_e8_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    for (const Geometry& geometry : kGeometries) {
        for (const Mode& mode : kModes) { failures += run_mode(geometry, mode); }
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " causal_softmax_attention E8 KV-cache correctness\n";
    return failures == 0 ? 0 : 1;
}
