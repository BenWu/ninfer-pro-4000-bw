#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Bf16GdnGatingScheduleId {
    GemvPairedRows,
    SmallTSplit10,
    SimtWarpRowC4,
    SimtWarpRowC8,
    MmaCooperativeSplit32,
    MmaCooperativeSplit16,
    MmaCooperativeSplit8,
    MmaCooperativeSplit4,
    MmaCooperativeSplit2,
    MmaUnsplit,
};

struct Bf16GdnGatingProblem {
    std::int32_t heads;
    std::int32_t input_rows;
    std::int32_t cols;
};

struct Bf16GdnGatingPlan {
    Bf16GdnGatingScheduleId schedule;
    Bf16GdnGatingTokenVariant token_variant;
    std::size_t workspace_bytes;
};

enum class Bf16GdnNormGatingScheduleId {
    Composed,
    MmaCooperativeSplit32,
};

struct Bf16GdnNormGatingPlan {
    Bf16GdnNormGatingScheduleId schedule;
    Bf16GdnGatingPlan control;
    std::size_t workspace_bytes;
};

const char* bf16_gdn_gating_schedule_name(Bf16GdnGatingScheduleId schedule) noexcept;

// The planning functions take the device's SM count explicitly so the pure planner stays
// testable without a CUDA context. Cooperative-MMA schedules only launch when their whole
// grid fits the device's cooperative residency (per-SM admission times sm_count); routes whose
// tuned schedule no longer fits on a smaller device demote to the next coarser cooperative
// split and finally to the non-cooperative unsplit grid. The 5090 (170 SM) selection is
// unchanged by this: every tuned route already fits there.
bool bf16_gdn_gating_admits(const Bf16GdnGatingProblem& problem) noexcept;
Bf16GdnGatingPlan bf16_gdn_gating_resolve_plan(const Bf16GdnGatingProblem& problem,
                                               std::int32_t sm_count);
Bf16GdnGatingPlan bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId schedule,
                                                    const Bf16GdnGatingProblem& problem,
                                                    std::int32_t sm_count);

std::size_t bf16_gdn_gating_capacity_workspace_bytes(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t min_cols, std::int32_t max_cols,
                                                     std::int32_t sm_count);
const char* bf16_gdn_norm_gating_schedule_name(Bf16GdnNormGatingScheduleId schedule) noexcept;
Bf16GdnNormGatingPlan bf16_gdn_norm_gating_resolve_plan(const Bf16GdnGatingProblem& problem,
                                                        std::int32_t sm_count);
std::size_t bf16_gdn_norm_gating_capacity_workspace_bytes(std::int32_t heads,
                                                          std::int32_t input_rows,
                                                          std::int32_t min_cols,
                                                          std::int32_t max_cols,
                                                          std::int32_t sm_count);

void bf16_gdn_gating_execute_plan(const Bf16GdnGatingPlan& plan, const Tensor& x,
                                  const Weight& a_weight, const Weight& b_weight,
                                  const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                  Tensor& g, Tensor& beta, cudaStream_t stream,
                                  std::int32_t sm_count);
void bf16_gdn_gating_execute_candidate(Bf16GdnGatingScheduleId schedule, const Tensor& x,
                                       const Weight& a_weight, const Weight& b_weight,
                                       const Tensor& A_log, const Tensor& dt_bias,
                                       WorkspaceArena& ws, Tensor& g, Tensor& beta,
                                       cudaStream_t stream, std::int32_t sm_count);
void bf16_gdn_gating_dispatch(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                              Tensor& g, Tensor& beta, cudaStream_t stream,
                              std::int32_t sm_count);
void bf16_gdn_norm_gating_dispatch(const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h,
                                   const Weight& a_weight, const Weight& b_weight,
                                   const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                   Tensor& g, Tensor& beta, cudaStream_t stream,
                                   std::int32_t sm_count);

} // namespace ninfer::ops::detail
