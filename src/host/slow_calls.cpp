// The driver calls a frame can wait on, timed (see slow_calls.h).
#include "host/slow_calls.h"

#include "core/host_clock.h"
#include "core/portable.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace slow_calls {

namespace {

constexpr int kSlots = 128;  // threads with a record at once; past that a thread's calls are counted, not shown
constexpr std::uint64_t kMs = 1000000ull;
constexpr std::uint64_t kBuckets[3] = {10 * kMs, 50 * kMs, 200 * kMs};
constexpr std::uint64_t kLinesInFull = 50;

struct CallNames {
    const char* name;
};
constexpr CallNames kNames[kCallCount] = {
    {"vkQueueSubmit"},
    {"vkQueuePresentKHR"},
    {"vkAcquireNextImageKHR"},
    {"vkQueueBindSparse"},
    {"vkWaitForFences"},
    {"vkWaitSemaphores"},
    {"vkGetSemaphoreCounterValue"},
    {"vkDeviceWaitIdle"},
    {"vkQueueWaitIdle"},
    {"vkAllocateMemory"},
    {"vkMapMemory"},
    {"vkFreeMemory"},
    {"vkBeginCommandBuffer"},
    {"vkEndCommandBuffer"},
    {"queue-lock"},
    {"submitted-wait"},
};

struct Config {
    std::uint64_t threshold = 50 * kMs;  // 0: off
};

Config parse(const char* ms) {
    Config c;
    if (ms && *ms) c.threshold = std::strtoull(ms, nullptr, 10) * kMs;
    return c;
}

Config g_cfg = parse(std::getenv("BBHOST_SLOW_CALL_MS"));
std::uint64_t slow_entry_ns() { return std::min(kBuckets[0], g_cfg.threshold); }

}  // namespace

struct alignas(64) Slot {
    std::atomic<const Site*> site{nullptr};  // the call it is in (null: none)
    std::atomic<std::uint64_t> since{0};     // when it went in
    std::atomic<std::uint64_t> slack{0};     // that call's poll step
    std::atomic<const char*> name{nullptr};  // its thread's name (interned)
    std::atomic<std::uint32_t> tid{0};
    std::atomic<std::uint32_t> taken{0};
    bool cp_phase = false;  // the owner's alone: its calls are the command processor's driver phase too
    // The owner's counts (one writer: plain loads and stores), read by the reports.
    alignas(64) std::atomic<std::uint64_t> calls[kCallCount] = {};
    std::atomic<std::uint64_t> ns[kCallCount] = {};
};

namespace {

Slot g_slots[kSlots];
std::mutex g_mu;  // the reports, a slot given back, the names
// Calls of threads without a slot, and of slots given back.
std::atomic<std::uint64_t> g_out_calls[kCallCount] = {}, g_out_ns[kCallCount] = {};
// Calls 10, 50 and 200 ms or more past their slack, and the longest of them.
std::atomic<std::uint64_t> g_over[kCallCount][3] = {};
std::atomic<std::uint64_t> g_max_ns[kCallCount] = {}, g_win_max_ns[kCallCount] = {};
std::uint64_t g_over_last[kCallCount][3] = {};  // under g_mu: the last window report's
std::atomic<std::uint64_t> g_logged{0}, g_unlogged{0};
std::atomic<bool> g_said{false};
std::mutex g_last_mu;
char g_last_line[1000];  // under g_last_mu (test_last_line); no destructor, for a line made as the process ends

char g_names[kSlots * 2][32];
int g_names_n = 0;  // under g_mu

thread_local Slot* t_slot = nullptr;
thread_local bool t_tried = false;  // a slot was asked for (none free, or the thread is going)

inline void add_own(std::atomic<std::uint64_t>& v, std::uint64_t n) { v.store(v.load(std::memory_order_relaxed) + n, std::memory_order_relaxed); }

std::uint32_t thread_id() {
#if defined(_WIN32)
    return static_cast<std::uint32_t>(GetCurrentThreadId());
#else
    return static_cast<std::uint32_t>(syscall(SYS_gettid));
#endif
}

// Under g_mu.
const char* intern_locked(const char* name) {
    for (int i = 0; i < g_names_n; ++i) {
        if (std::strcmp(g_names[i], name) == 0) return g_names[i];
    }
    if (g_names_n >= static_cast<int>(sizeof(g_names) / sizeof(g_names[0]))) return "thread";
    std::snprintf(g_names[g_names_n], sizeof(g_names[0]), "%s", name);
    return g_names[g_names_n++];
}

const char* this_thread_name() {
    char name[32] = {};
    if (!host_thread_name(name, sizeof(name))) std::snprintf(name, sizeof(name), "thread");
    std::lock_guard<std::mutex> lk(g_mu);
    return intern_locked(name);
}

void say_config() {
    if (g_said.exchange(true)) return;
    host_log("slow calls: the driver calls a frame can wait on are timed on every thread; one of %llu ms or more is logged with the "
             "calls the other threads were in then, the first %llu in full (BBHOST_SLOW_CALL_MS; 0: off)",
             static_cast<unsigned long long>(g_cfg.threshold / kMs), static_cast<unsigned long long>(kLinesInFull));
}

// Gives the thread's slot back as it exits: its counts go to the totals.
struct Keeper {
    bool armed = false;
    ~Keeper() {
        Slot* s = t_slot;
        if (!s) return;
        t_slot = nullptr;
        std::lock_guard<std::mutex> lk(g_mu);
        s->site.store(nullptr, std::memory_order_release);
        s->cp_phase = false;
        for (int k = 0; k < kCallCount; ++k) {
            g_out_calls[k].fetch_add(s->calls[k].exchange(0, std::memory_order_relaxed), std::memory_order_relaxed);
            g_out_ns[k].fetch_add(s->ns[k].exchange(0, std::memory_order_relaxed), std::memory_order_relaxed);
        }
        s->taken.store(0, std::memory_order_release);
    }
};
thread_local Keeper t_keeper;

Slot* claim() {
    if (t_tried) return t_slot;
    t_tried = true;
    say_config();
    for (Slot& s : g_slots) {
        std::uint32_t free = 0;
        if (!s.taken.compare_exchange_strong(free, 1, std::memory_order_acq_rel)) continue;
        s.site.store(nullptr, std::memory_order_relaxed);
        s.cp_phase = false;
        s.tid.store(thread_id(), std::memory_order_relaxed);
        s.name.store(this_thread_name(), std::memory_order_release);
        t_slot = &s;
        t_keeper.armed = true;
        return &s;
    }
    return nullptr;
}

const char* base_name(const char* path) {
    const char* b = path;
    for (const char* p = path; *p; ++p) {
        if (*p == '/' || *p == '\\') b = p + 1;
    }
    return b;
}

// "(one job)", "(render.cpp:6142)", "(the label landing, up to 50 ms a step)".
void where_text(const Site& s, std::uint64_t slack, char* out, std::size_t n) {
    if (s.line) std::snprintf(out, n, "%s:%d", base_name(s.where), s.line);
    else std::snprintf(out, n, "%s", s.where);
    if (slack) {
        const std::size_t len = std::strlen(out);
        std::snprintf(out + len, n - len, ", up to %.0f ms a step", static_cast<double>(slack) / 1e6);
    }
}

struct Seen {
    const Slot* slot;
    const Site* site;
    std::uint64_t ns, slack;
};
// The threads in a timed call at `now` but `skip`, longest first.
int snapshot(Seen* out, int cap, std::uint64_t now, const Slot* skip) {
    int n = 0;
    for (const Slot& s : g_slots) {
        if (&s == skip || !s.taken.load(std::memory_order_acquire)) continue;
        const Site* site = s.site.load(std::memory_order_acquire);
        if (!site) continue;
        const std::uint64_t since = s.since.load(std::memory_order_relaxed);
        if (n < cap) out[n++] = Seen{&s, site, now > since ? now - since : 0, s.slack.load(std::memory_order_relaxed)};
    }
    std::sort(out, out + n, [](const Seen& a, const Seen& b) { return a.ns > b.ns; });
    return n;
}

// Appends the threads in `seen` to buf (" name:tid call ms (where)," each) while they fit.
void append_seen(char* buf, std::size_t cap, std::size_t& len, const Seen* seen, int n) {
    for (int i = 0; i < n; ++i) {
        char where[96];
        where_text(*seen[i].site, seen[i].slack, where, sizeof(where));
        const char* name = seen[i].slot->name.load(std::memory_order_acquire);
        char one[200];
        const int m = std::snprintf(one, sizeof(one), "%s %s:%u %s %.1f ms (%s)", i ? "," : "", name ? name : "thread",
                                    seen[i].slot->tid.load(std::memory_order_relaxed), kNames[seen[i].site->call].name,
                                    static_cast<double>(seen[i].ns) / 1e6, where);
        if (m <= 0) continue;
        if (len + static_cast<std::size_t>(m) + 24 >= cap) {
            len += static_cast<std::size_t>(std::snprintf(buf + len, cap - len, " and %d more", n - i));
            return;
        }
        std::memcpy(buf + len, one, static_cast<std::size_t>(m) + 1);
        len += static_cast<std::size_t>(m);
    }
}

void update_max(std::atomic<std::uint64_t>& m, std::uint64_t v) {
    std::uint64_t cur = m.load(std::memory_order_relaxed);
    while (v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
    }
}

// A call of 10 ms or more (or past a lower threshold): counted, and logged
// past the threshold.
__attribute__((noinline)) void slow(const Site& site, std::uint64_t d, std::uint64_t slack, std::uint64_t t1, Slot* own) {
    const std::uint64_t past = d > slack ? d - slack : 0;
    const Call c = site.call;
    if (past >= kBuckets[0]) {
        for (int i = 0; i < 3; ++i) {
            if (past >= kBuckets[i]) g_over[c][i].fetch_add(1, std::memory_order_relaxed);
        }
        update_max(g_max_ns[c], d);
        update_max(g_win_max_ns[c], d);
    }
    if (past < g_cfg.threshold) return;
    if (g_logged.fetch_add(1, std::memory_order_relaxed) >= kLinesInFull) {
        g_unlogged.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Seen seen[kSlots];
    const int n = snapshot(seen, kSlots, t1, own);
    char where[96];
    where_text(site, slack, where, sizeof(where));
    const char* name = own ? own->name.load(std::memory_order_acquire) : nullptr;
    char buf[1000];
    int m = std::snprintf(buf, sizeof(buf), "slow call: %s %.1f ms on %s:%u (%s), before flip #%llu; others in calls:", kNames[c].name,
                          static_cast<double>(d) / 1e6, name ? name : this_thread_name(), thread_id(), where,
                          static_cast<unsigned long long>(hle_gnm_flip_calls()));
    if (m < 0) return;
    std::size_t len = std::min(static_cast<std::size_t>(m), sizeof(buf) - 1);
    if (n) append_seen(buf, sizeof(buf), len, seen, n);
    else len += static_cast<std::size_t>(std::snprintf(buf + len, sizeof(buf) - len, " none"));
    len = std::min(len, sizeof(buf) - 1);
    buf[len] = 0;
    host_log("%s", buf);
    std::lock_guard<std::mutex> lk(g_last_mu);
    std::memcpy(g_last_line, buf, len + 1);
}

// Calls of kind c since the start (`time`: their nanoseconds), every thread's.
std::uint64_t total(Call c, bool time) {
    std::uint64_t v = (time ? g_out_ns : g_out_calls)[c].load(std::memory_order_relaxed);
    for (const Slot& s : g_slots) v += (time ? s.ns : s.calls)[c].load(std::memory_order_relaxed);
    return v;
}

}  // namespace

const char* call_name(Call c) { return c < kCallCount ? kNames[c].name : "?"; }
bool on() { return g_cfg.threshold != 0 && host_tooling(); }

void Scope::begin(bool measure) noexcept {
    if (!on()) {
        if (measure) t0_ = host_clock_monotonic_ns();
        return;
    }
    counted_ = true;
    t0_ = host_clock_monotonic_ns();
    Slot* s = t_slot ? t_slot : claim();
    if (s && !s->site.load(std::memory_order_relaxed)) {
        s->since.store(t0_, std::memory_order_relaxed);
        s->slack.store(slack_, std::memory_order_relaxed);
        s->site.store(site_, std::memory_order_release);
        slot_ = s;
    }
}

std::uint64_t Scope::end() noexcept {
    if (!t0_) return 0;
    return end_at(host_clock_monotonic_ns());
}

std::uint64_t Scope::end_at(std::uint64_t t1) noexcept {
    if (!t0_) return 0;
    const std::uint64_t t0 = t0_;
    t0_ = 0;
    const std::uint64_t d = t1 > t0 ? t1 - t0 : 0;
    if (!counted_) return d;
    const Call c = site_->call;
    Slot* own = t_slot;
    if (Slot* s = slot_) {
        slot_ = nullptr;
        s->site.store(nullptr, std::memory_order_release);
        if (s->cp_phase) host_gpu_phase_add(kPhaseDriver, d);
    }
    if (own) {
        add_own(own->calls[c], 1);
        add_own(own->ns[c], d);
    } else {
        g_out_calls[c].fetch_add(1, std::memory_order_relaxed);
        g_out_ns[c].fetch_add(d, std::memory_order_relaxed);
    }
    if (d >= slow_entry_ns()) slow(*site_, d, slack_, t1, own);
    return d;
}

void Scope::next(const Site& next) noexcept {
    if (!t0_) {
        site_ = &next;
        slack_ = 0;
        return;
    }
    const std::uint64_t t = host_clock_monotonic_ns();
    end_at(t);
    site_ = &next;
    slack_ = 0;
    t0_ = t;
    if (!counted_) return;
    Slot* s = t_slot;
    if (s && !s->site.load(std::memory_order_relaxed)) {
        s->since.store(t, std::memory_order_relaxed);
        s->slack.store(0, std::memory_order_relaxed);
        s->site.store(site_, std::memory_order_release);
        slot_ = s;
    }
}

CpPhase::CpPhase() noexcept {
    if (!on()) return;
    slot_ = t_slot ? t_slot : claim();
    if (!slot_) return;
    prev_ = slot_->cp_phase;
    slot_->cp_phase = true;
}

CpPhase::~CpPhase() {
    if (slot_ && slot_ == t_slot) slot_->cp_phase = prev_;
}

std::string window_report() {
    if (!on()) return {};
    std::lock_guard<std::mutex> lk(g_mu);
    std::string calls;
    std::uint64_t longest = 0;
    Call longest_call = kCallCount;
    for (int k = 0; k < kCallCount; ++k) {
        std::uint64_t d[3];
        for (int i = 0; i < 3; ++i) {
            const std::uint64_t now = g_over[k][i].load(std::memory_order_relaxed);
            d[i] = now - g_over_last[k][i];
            g_over_last[k][i] = now;
        }
        const std::uint64_t m = g_win_max_ns[k].exchange(0, std::memory_order_relaxed);
        if (!d[0]) continue;
        char b[96];
        std::snprintf(b, sizeof(b), "%s%s %llu/%llu/%llu", calls.empty() ? "" : ", ", kNames[k].name, static_cast<unsigned long long>(d[0]),
                      static_cast<unsigned long long>(d[1]), static_cast<unsigned long long>(d[2]));
        calls += b;
        if (m > longest) {
            longest = m;
            longest_call = static_cast<Call>(k);
        }
    }
    char tail[160];
    const std::uint64_t logged = std::min(g_logged.load(std::memory_order_relaxed), kLinesInFull);
    std::snprintf(tail, sizeof(tail), "lines %llu so far, %llu more counted only", static_cast<unsigned long long>(logged),
                  static_cast<unsigned long long>(g_unlogged.load(std::memory_order_relaxed)));
    if (calls.empty()) return std::string("slow calls: none of 10 ms or more in this window; ") + tail;
    char lead[160];
    std::snprintf(lead, sizeof(lead), "; the longest %s %.1f ms; ", longest_call < kCallCount ? kNames[longest_call].name : "?",
                  static_cast<double>(longest) / 1e6);
    return "slow calls: of 10/50/200 ms or more in this window: " + calls + lead + tail;
}

void final_report() {
    if (!on()) return;
    const std::uint64_t logged = std::min(g_logged.load(std::memory_order_relaxed), kLinesInFull);
    host_log("slow calls (BBHOST_SLOW_CALL_MS=%llu): %llu lines logged, %llu more counted only", static_cast<unsigned long long>(g_cfg.threshold / kMs),
             static_cast<unsigned long long>(logged), static_cast<unsigned long long>(g_unlogged.load(std::memory_order_relaxed)));
    std::lock_guard<std::mutex> lk(g_mu);
    for (int k = 0; k < kCallCount; ++k) {
        const Call c = static_cast<Call>(k);
        const std::uint64_t n = total(c, false), ns = total(c, true);
        if (!n) continue;
        const std::uint64_t m = g_max_ns[k].load(std::memory_order_relaxed);
        char longest[64];
        if (m) std::snprintf(longest, sizeof(longest), "the longest %.1f ms", static_cast<double>(m) / 1e6);
        else std::snprintf(longest, sizeof(longest), "none of 10 ms or more");
        host_log("slow calls:   %s: %llu calls, %.1f ms in all (%.1f us each), %s; of 10/50/200 ms or more %llu/%llu/%llu", kNames[k].name,
                 static_cast<unsigned long long>(n), static_cast<double>(ns) / 1e6, static_cast<double>(ns) / 1e3 / static_cast<double>(n),
                 longest, static_cast<unsigned long long>(g_over[k][0].load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(g_over[k][1].load(std::memory_order_relaxed)),
                 static_cast<unsigned long long>(g_over[k][2].load(std::memory_order_relaxed)));
    }
}

std::string in_calls_now() {
    if (!on()) return {};
    Seen seen[kSlots];
    const int n = snapshot(seen, kSlots, host_clock_monotonic_ns(), nullptr);
    if (!n) return {};
    char buf[1000];
    std::size_t len = 0;
    append_seen(buf, sizeof(buf), len, seen, n);
    len = std::min(len, sizeof(buf) - 1);
    buf[len] = 0;
    return buf[0] == ' ' ? std::string(buf + 1) : std::string(buf);
}

void test_configure(const char* slow_call_ms) {
    g_cfg = parse(slow_call_ms);
    std::lock_guard<std::mutex> lk(g_mu);
    for (int k = 0; k < kCallCount; ++k) {
        g_out_calls[k].store(0);
        g_out_ns[k].store(0);
        g_max_ns[k].store(0);
        g_win_max_ns[k].store(0);
        for (int i = 0; i < 3; ++i) {
            g_over[k][i].store(0);
            g_over_last[k][i] = 0;
        }
        for (Slot& s : g_slots) {
            s.calls[k].store(0);
            s.ns[k].store(0);
        }
    }
    g_logged.store(0);
    g_unlogged.store(0);
    g_said.store(false);
}

TestCounts test_counts(Call c) {
    std::lock_guard<std::mutex> lk(g_mu);
    TestCounts t;
    t.calls = total(c, false);
    t.ns = total(c, true);
    for (int i = 0; i < 3; ++i) t.over[i] = g_over[c][i].load();
    t.logged = std::min(g_logged.load(), kLinesInFull);
    t.unlogged = g_unlogged.load();
    return t;
}

std::string test_last_line() {
    std::lock_guard<std::mutex> lk(g_last_mu);
    return std::string(g_last_line);
}

}  // namespace slow_calls
