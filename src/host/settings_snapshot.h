#pragma once

#include "host/settings.h"

#include <atomic>
#include <cstdint>
#include <mutex>

// Publish a whole settings snapshot before exposing its generation. Readers
// keep a copy per thread, so unchanged settings require no shared mutex.
class HostSettingsSnapshot {
public:
    HostSettingsSnapshot() {
        settings_.serial = next_serial();
        serial_.store(settings_.serial, std::memory_order_relaxed);
    }

    std::uint64_t serial() const { return serial_.load(std::memory_order_acquire); }

    void publish(HostSettings settings) {
        std::lock_guard<std::mutex> lock(mu_);
        settings.serial = next_serial();
        settings_ = settings;
        serial_.store(settings.serial, std::memory_order_release);
    }

    HostSettings read() const {
        struct Cache {
            const HostSettingsSnapshot* owner = nullptr;
            HostSettings value;
        };
        thread_local Cache cache;
        if (cache.owner == this && cache.value.serial == serial()) return cache.value;
        std::lock_guard<std::mutex> lock(mu_);
        cache.value = settings_;
        cache.owner = this;
        return cache.value;
    }

private:
    static std::uint64_t next_serial() {
        // Distinct lifetimes also get distinct generations, even if a store
        // is constructed at the address of a previously cached store.
        static std::atomic<std::uint64_t> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
    }

    mutable std::mutex mu_;
    HostSettings settings_;
    std::atomic<std::uint64_t> serial_{0};
};
