// The main loop's waits, for frame_stats (frame_stats.h). Apart from it so
// that the sync primitives that time them link without the rest.

#include "host/frame_stats.h"

#if !defined(_WIN32)
#include <pthread.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "log.h"

namespace {
const bool g_on = [] {
    const char* e = std::getenv("BBHOST_FRAME_STATS");
    return e && *e == '1';
}();
// BBHOST_FRAME_DETAIL=1: the per-second lines' detail (frame_stats.h). A
// switch of its own, so a harness asking for it still gets the `frames:`
// lines from a build that predates it.
const bool g_detail = [] {
    const char* e = std::getenv("BBHOST_FRAME_DETAIL");
    return g_on && e && *e == '1';
}();
thread_local bool t_main_loop = false;
// BBHOST_WAIT_LOG=<prefix>[,<prefix>...]: threads whose names start with one
// of these log every wait of BBHOST_WAIT_LOG_MS (20) or more, with its call
// chain under BBHOST_HLE_COUNT - for the threads the main loop waits on.
const std::vector<std::string> g_log_threads = [] {
    std::vector<std::string> v;
    const char* e = std::getenv("BBHOST_WAIT_LOG");
    for (std::string s = e ? e : ""; !s.empty();) {
        const std::size_t c = s.find(',');
        v.push_back(s.substr(0, c));
        s = c == std::string::npos ? "" : s.substr(c + 1);
    }
    return v;
}();
const std::int64_t g_log_wait_ns = [] {
    const char* e = std::getenv("BBHOST_WAIT_LOG_MS");
    return static_cast<std::int64_t>(e ? std::atof(e) * 1e6 : 20e6);
}();
thread_local int t_log_waits = -1;  // -1: not looked up yet
thread_local char t_thread_name[16];
bool log_waits() {
    if (t_log_waits < 0) {
        t_log_waits = 0;
#if !defined(_WIN32)
        pthread_getname_np(pthread_self(), t_thread_name, sizeof(t_thread_name));
        for (const std::string& p : g_log_threads) {
            if (!p.empty() && std::strncmp(t_thread_name, p.c_str(), p.size()) == 0) t_log_waits = 1;
        }
#endif
    }
    return t_log_waits == 1;
}
std::atomic<std::uint64_t> g_waited_ns[3];
// BBHOST_HLE_COUNT: the guest return address of the thread's current HLE
// call (the thunk notes it), and the main loop's waits summed by it.
thread_local std::uint64_t t_site = 0;  // a hash of t_chain
thread_local std::uint64_t t_chain[3] = {};
std::unordered_map<std::uint64_t, std::array<std::uint64_t, 3>> g_chains;  // under g_sites_mu
std::uint64_t g_guest_base = 0;
std::mutex g_sites_mu;
std::unordered_map<std::uint64_t, std::uint64_t> g_site_ns[3];  // under g_sites_mu, by kind
// The current frame's waits (the main thread's alone: its waits and the
// frame-time manager's hook both run there), for a frame that runs long.
std::uint64_t g_frame_wait_ns = 0;
std::vector<std::pair<std::uint64_t, std::uint64_t>> g_frame_sites;  // (site, ns), with the kind in bits 62-63
const char* const kKindTag[3] = {"", " (sleep)", " (file)"};
std::mutex g_busy_mu;
std::vector<std::uint32_t> g_busy_us;  // under g_busy_mu

std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

bool frame_stats_enabled() { return g_on; }
bool frame_stats_detail() { return g_detail; }

void frame_stats_mark_main_thread() { t_main_loop = true; }

std::uint64_t main_thread_waited_ns(int kind) { return g_waited_ns[kind].exchange(0); }

MainThreadWait::MainThreadWait(int kind) : kind_(kind) {
    if ((t_main_loop && g_on) || (!g_log_threads.empty() && log_waits())) t0_ = now_ns();
}

MainThreadWait::~MainThreadWait() {
    if (!t0_) return;
    const auto ns = static_cast<std::uint64_t>(now_ns() - t0_);
    if (!t_main_loop) {
        if (static_cast<std::int64_t>(ns) >= g_log_wait_ns) {
            std::string chain;
            for (std::uint64_t a : t_chain) {
                if (!a) break;
                char b[24];
                std::snprintf(b, sizeof(b), "%s0x%llx", chain.empty() ? " at " : "<", static_cast<unsigned long long>(a - g_guest_base + 0x400000ull));
                chain += b;
            }
            host_log("wait: %s waited %.1f ms%s%s", t_thread_name, static_cast<double>(ns) / 1e6, kKindTag[kind_], chain.c_str());
        }
        return;
    }
    g_waited_ns[kind_].fetch_add(ns, std::memory_order_relaxed);
    g_frame_wait_ns += ns;
    if (t_site && g_frame_sites.size() < 256) {
        const std::uint64_t key = t_site | (static_cast<std::uint64_t>(kind_) << 62);
        bool found = false;
        for (auto& [k, v] : g_frame_sites) {
            if (k == key) {
                v += ns;
                found = true;
                break;
            }
        }
        if (!found) g_frame_sites.push_back({key, ns});
    }
    if (t_site) {
        std::lock_guard<std::mutex> lk(g_sites_mu);
        g_site_ns[kind_][t_site] += ns;
        if (!g_chains.count(t_site)) g_chains[t_site] = {t_chain[0], t_chain[1], t_chain[2]};
    }
}

void main_wait_note_site(const std::uint64_t* chain) {
    std::memcpy(t_chain, chain, sizeof(t_chain));
    t_site = (chain[0] ^ (chain[1] * 0x9e3779b97f4a7c15ull) ^ (chain[2] * 0xc2b2ae3d27d4eb4full)) & ~(3ull << 62);
}

static std::string chain_text(std::uint64_t site) {
    const auto it = g_chains.find(site);
    if (it == g_chains.end()) return "?";
    std::string t;
    for (std::uint64_t a : it->second) {
        if (!a) break;
        char b[24];
        std::snprintf(b, sizeof(b), "%s0x%llx", t.empty() ? "" : "<", static_cast<unsigned long long>(a - g_guest_base + 0x400000ull));
        t += b;
    }
    return t;
}
void main_wait_guest_base(std::uint64_t slide) { g_guest_base = slide; }

void main_wait_sites_report() {
    std::vector<std::pair<std::uint64_t, std::pair<std::uint64_t, int>>> v;
    std::string top;
    {
        std::lock_guard<std::mutex> lk(g_sites_mu);
        for (int k = 0; k < 3; ++k) {
            for (const auto& [site, ns] : g_site_ns[k]) v.push_back({ns, {site, k}});
            g_site_ns[k].clear();
        }
        if (v.empty()) return;
        std::sort(v.rbegin(), v.rend());
        for (std::size_t i = 0; i < v.size() && i < 10; ++i) {
            char b[40];
            std::snprintf(b, sizeof(b), "%s %.0f ms;", kKindTag[v[i].second.second], static_cast<double>(v[i].first) / 1e6);
            top += " " + chain_text(v[i].second.first) + b;
        }
    }
    host_log("main loop waits by call site (return address, Binary Ninja):%s", top.c_str());
}

void frame_stats_main_frame(std::uint64_t busy_us, std::uint64_t flip) {
    if (!g_on) return;
    // A frame of two or more: what the main loop waited for in it (by call
    // site with BBHOST_HLE_COUNT) - streaming and compile hitches, not the
    // steady cost the per-second line shows.
    if (busy_us > 33000) {
        std::sort(g_frame_sites.begin(), g_frame_sites.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string top;
        {
            std::lock_guard<std::mutex> lk(g_sites_mu);
            for (std::size_t i = 0; i < g_frame_sites.size() && i < 4; ++i) {
                char b[40];
                std::snprintf(b, sizeof(b), "%s %.1f ms;", kKindTag[g_frame_sites[i].first >> 62],
                              static_cast<double>(g_frame_sites[i].second) / 1e6);
                top += " " + chain_text(g_frame_sites[i].first & ~(3ull << 62)) + b;
            }
        }
        host_log("main loop: a %.1f ms frame before flip %llu, %.1f ms of it waiting%s%s", static_cast<double>(busy_us) / 1e3,
                 static_cast<unsigned long long>(flip), static_cast<double>(g_frame_wait_ns) / 1e6, top.empty() ? "" : ":", top.c_str());
    }
    g_frame_wait_ns = 0;
    g_frame_sites.clear();
    std::lock_guard<std::mutex> lk(g_busy_mu);
    if (g_busy_us.size() < 4096) g_busy_us.push_back(static_cast<std::uint32_t>(std::min<std::uint64_t>(busy_us, 0xffffffffu)));
}

std::vector<std::uint32_t> main_frames_busy_us() {
    std::vector<std::uint32_t> v;
    std::lock_guard<std::mutex> lk(g_busy_mu);
    std::swap(v, g_busy_us);
    return v;
}
