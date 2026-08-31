#pragma once

// Cached device geometry for host-side launcher and tuning code. The engine binds one CUDA
// device to its threads, so launcher and policy code resolves device-dependent tuning against
// the current device. The first query per device records the SM count; later calls (including
// graph capture and replay) reuse it without touching the CUDA runtime.

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

inline int current_device_sm_count() {
    struct Entry {
        int device;
        int sm_count;
    };
    static std::mutex mutex;
    static Entry cache[64]{};
    static int entries = 0;

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        throw std::runtime_error("current_device_sm_count: cudaGetDevice failed");
    }
    {
        const std::lock_guard lock(mutex);
        for (int i = 0; i < entries; ++i) {
            if (cache[i].device == device) { return cache[i].sm_count; }
        }
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess ||
            prop.multiProcessorCount < 1) {
            throw std::runtime_error(
                std::string("current_device_sm_count: cudaGetDeviceProperties failed on device ") +
                std::to_string(device));
        }
        const int sm_count = prop.multiProcessorCount;
        cache[entries].device = device;
        cache[entries].sm_count = sm_count;
        ++entries;
        return sm_count;
    }
}

} // namespace ninfer::ops::detail