#pragma once

// How the settings object (host/settings.h) is published and handed out.
//
// A change builds a new snapshot; once published it is never written again,
// and every snapshot is kept for the run (a few hundred bytes a change, and a
// change is a player's key press), so a reader takes the newest with one
// atomic load and no lock - and a reference it holds stays good. Consumers
// that act on a setting register a listener: it is told the settings at once
// and then after each change, on the thread that made the change, so nothing
// has to ask every frame whether anything moved.
//
// The rules for a listener: quick, and it only keeps what it needs (an atomic
// its hot path reads, a flag that has work done on the next view); it must
// not change a setting or register another listener (both take the store's
// lock, which a listener runs under).

#include "host/settings.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>

class HostSettingsStore {
public:
    static constexpr int kMaxListeners = 32;

    HostSettingsStore() {
        HostSettings first;
        first.serial = 1;
        current_.store(&kept_.emplace_back(first), std::memory_order_release);
    }
    HostSettingsStore(const HostSettingsStore&) = delete;
    HostSettingsStore& operator=(const HostSettingsStore&) = delete;

    // The newest snapshot.
    const HostSettings& now() const { return *current_.load(std::memory_order_acquire); }

    // Publishes `next`, numbered one past the current snapshot, and tells
    // every listener in the order they registered.
    void publish(HostSettings next) {
        std::lock_guard<std::mutex> lock(mu_);
        next.serial = now().serial + 1;
        const HostSettings* s = &kept_.emplace_back(next);
        current_.store(s, std::memory_order_release);
        for (int i = 0; i < n_; ++i) listeners_[i](*s);
    }

    // Registers `fn` and calls it with the current snapshot. False when the
    // table is full (the listener then never hears of a change: a bug).
    bool listen(HostSettingsListener fn) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!fn || n_ >= kMaxListeners) return false;
        listeners_[n_++] = fn;
        fn(now());
        return true;
    }

    std::size_t snapshots() const {
        std::lock_guard<std::mutex> lock(mu_);
        return kept_.size();
    }

private:
    mutable std::mutex mu_;
    std::deque<HostSettings> kept_;  // every snapshot published (a deque keeps their addresses)
    std::atomic<const HostSettings*> current_{nullptr};
    HostSettingsListener listeners_[kMaxListeners] = {};
    int n_ = 0;
};
