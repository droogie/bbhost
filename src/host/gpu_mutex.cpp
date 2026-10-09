#include "host/gpu_mutex.h"

#include "core/portable.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <pthread.h>
#include <unistd.h>
#endif

#if defined(__linux__)
extern "C" char __executable_start;  // the linker's: where this binary's image starts
#endif

namespace {

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Sites by address in a small open-addressed table; a site is a return
// address into the function that took the lock.
struct Site {
    std::atomic<void*> at{nullptr};
    std::atomic<std::uint64_t> holds{0}, hold_ns{0}, waits{0}, wait_ns{0};
    char thread[16] = {};
};
constexpr std::size_t kSites = 1024;
Site g_sites[kSites];

Site* site_of(void* at) {
    std::size_t i = (reinterpret_cast<std::uintptr_t>(at) >> 2) % kSites;
    for (std::size_t n = 0; n < kSites; ++n, i = (i + 1) % kSites) {
        void* cur = g_sites[i].at.load(std::memory_order_acquire);
        if (cur == at) return &g_sites[i];
        if (!cur) {
            void* expected = nullptr;
            if (g_sites[i].at.compare_exchange_strong(expected, at)) {
#if !defined(_WIN32)
                pthread_getname_np(pthread_self(), g_sites[i].thread, sizeof(g_sites[i].thread));
#endif
                return &g_sites[i];
            }
            if (expected == at) return &g_sites[i];
        }
    }
    return nullptr;
}

std::atomic<std::uint64_t> g_last_report{0};

// addr2line on our own binary for a handful of addresses.
std::vector<std::string> symbolize(const std::vector<void*>& addrs) {
    std::vector<std::string> out(addrs.size(), "?");
#if defined(__APPLE__)
    // atos wants the image's load address (its Mach-O header) with -l.
    const std::string exe = host_exe_path();
    if (exe.empty()) return out;
    char base[32];
    std::snprintf(base, sizeof(base), "0x%lx", static_cast<unsigned long>(host_image_base()));
    std::string cmd = std::string("atos -o '") + exe + "' -l " + base;
    for (void* a : addrs) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), " 0x%lx", static_cast<unsigned long>(reinterpret_cast<std::uintptr_t>(a) - 1));
        cmd += buf;
    }
    if (FILE* p = popen(cmd.c_str(), "r")) {
        char line[1024];
        for (std::size_t i = 0; i < addrs.size(); ++i) {
            if (!std::fgets(line, sizeof(line), p)) break;
            line[std::strcspn(line, "\n")] = 0;
            out[i] = line;
        }
        pclose(p);
    }
#elif !defined(_WIN32)
    // Our own path: in the child, /proc/self/exe would be addr2line.
    char exe[4096] = {};
    if (readlink("/proc/self/exe", exe, sizeof(exe) - 1) <= 0) return out;
    std::string cmd = std::string("addr2line -f -C -e '") + exe + "'";
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(&__executable_start);
    for (void* a : addrs) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), " %lx", static_cast<unsigned long>(reinterpret_cast<std::uintptr_t>(a) - base - 1));
        cmd += buf;
    }
    if (FILE* p = popen(cmd.c_str(), "r")) {
        char line[1024];
        for (std::size_t i = 0; i < addrs.size(); ++i) {
            if (!std::fgets(line, sizeof(line), p)) break;
            line[std::strcspn(line, "\n")] = 0;
            out[i] = line;
            if (!std::fgets(line, sizeof(line), p)) break;  // file:line
        }
        pclose(p);
    }
#endif
    return out;
}

void maybe_report() {
    const std::uint64_t t = now_ns(), last = g_last_report.load(std::memory_order_relaxed);
    if (last && t - last < 5000000000ull) return;
    std::uint64_t expected = last;
    if (!g_last_report.compare_exchange_strong(expected, t)) return;
    if (!last) return;  // the first call only starts the clock
    struct Row {
        void* at;
        std::uint64_t holds, hold_ns, waits, wait_ns;
        const char* thread;
    };
    std::vector<Row> rows;
    for (Site& s : g_sites) {
        void* at = s.at.load(std::memory_order_acquire);
        if (!at) continue;
        rows.push_back({at, s.holds.exchange(0), s.hold_ns.exchange(0), s.waits.exchange(0), s.wait_ns.exchange(0), s.thread});
    }
    const double window_ms = static_cast<double>(t - last) / 1e6;
    // Symbolizing runs addr2line: off this thread, which may be the renderer's.
    std::thread([rows = std::move(rows), window_ms]() mutable {
        auto report = [&](const char* what, auto key) {
            std::sort(rows.begin(), rows.end(), [&](const Row& a, const Row& b) { return key(a) > key(b); });
            std::vector<void*> addrs;
            for (std::size_t i = 0; i < rows.size() && i < 10 && key(rows[i]); ++i) addrs.push_back(rows[i].at);
            const std::vector<std::string> names = symbolize(addrs);
            for (std::size_t i = 0; i < addrs.size(); ++i) {
                const Row& r = rows[i];
                host_log("lockprof: %s %5.1f%% of %.0f ms: %s [%s] holds %llu (%.1f ms), waits %llu (%.1f ms)", what,
                         100.0 * static_cast<double>(key(r)) / 1e6 / window_ms, window_ms, names[i].c_str(), r.thread,
                         static_cast<unsigned long long>(r.holds), static_cast<double>(r.hold_ns) / 1e6,
                         static_cast<unsigned long long>(r.waits), static_cast<double>(r.wait_ns) / 1e6);
            }
        };
        report("held", [](const Row& r) { return r.hold_ns; });
        report("waited", [](const Row& r) { return r.wait_ns; });
    }).detach();
}

}  // namespace

std::string GpuMutex::name_of(void* site) {
    if (!site) return "nobody";
    const std::vector<std::string> names = symbolize({site});
    return names.empty() || names[0] == "?" ? "an unnamed site" : names[0];
}

bool GpuMutex::profiling() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_LOCK_PROFILE");
        return e && *e == '1';
    }();
    return on;
}

void GpuMutex::lock_profiled(void* site) {
    if (m_.try_lock()) {
        acquired(site, 0);
        return;
    }
    const std::uint64_t t0 = now_ns();
    m_.lock();
    acquired(site, now_ns() - t0);
}

void GpuMutex::acquired(void* site, std::uint64_t waited_ns) {
    site_ = site;
    since_ns_ = now_ns();
    if (waited_ns) {
        if (Site* s = site_of(site)) {
            s->waits.fetch_add(1, std::memory_order_relaxed);
            s->wait_ns.fetch_add(waited_ns, std::memory_order_relaxed);
        }
    }
}

void GpuMutex::released() {
    const std::uint64_t held = now_ns() - since_ns_;
    if (Site* s = site_of(site_)) {
        s->holds.fetch_add(1, std::memory_order_relaxed);
        s->hold_ns.fetch_add(held, std::memory_order_relaxed);
    }
}

void GpuMutex::report() { maybe_report(); }
