// The slow-call registry without the game (host/slow_calls.h): a call over the
// threshold is logged on one line with what the other threads were calling
// then; its counts by call; the first lines in full and the rest counted; a
// poll step's slack; the command processor's driver phase; off, nothing
// timed. Then the cost of a timed call against an untimed one.
#include "hle/modules.h"
#include "host/gpu.h"
#include "host/slow_calls.h"
#include "log.h"

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

// What slow_calls.cpp asks the rest of bbhost.
std::uint64_t hle_gnm_flip_calls() { return 4242; }
bool host_thread_name(char* out, std::size_t n) { return pthread_getname_np(pthread_self(), out, n) == 0 && out[0]; }
std::atomic<std::uint64_t> g_driver_phase_ns{0};
void host_gpu_phase_add(int phase, std::uint64_t ns) {
    if (phase == kPhaseDriver) g_driver_phase_ns.fetch_add(ns);
}

namespace {

int g_failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using slow_calls::Scope;
using slow_calls::Site;

constexpr Site kSubmit{slow_calls::kQueueSubmit, "one job"};
constexpr Site kPresent{slow_calls::kQueuePresent, "present"};
constexpr Site kFence{slow_calls::kWaitFences, "a submission's fence"};
constexpr Site kWait{slow_calls::kWaitSemaphores, "the timeline"};
constexpr Site kLock{slow_calls::kQueueLock, "a submission"};
constexpr Site kAlloc{slow_calls::kAllocateMemory, "/a/private/dir/src/host/render.cpp", 6142};

void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
void name_thread(const char* n) { pthread_setname_np(pthread_self(), n); }
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// A call of `ms` on a thread of its own named `name`.
std::thread call_on(const char* name, const Site& site, int ms, int delay_ms = 0) {
    return std::thread([name, &site, ms, delay_ms] {
        name_thread(name);
        if (delay_ms) sleep_ms(delay_ms);
        Scope timed(site);
        sleep_ms(ms);
    });
}

void test_line_names_the_others() {
    slow_calls::test_configure("50");
    // The submit goes in first and stays 260 ms; the present, from 20 ms to
    // 120 ms, is logged at its end with the submit still under way.
    std::thread a = call_on("t-submit", kSubmit, 260);
    std::thread b = call_on("t-present", kPresent, 100, 20);
    b.join();
    const std::string line = slow_calls::test_last_line();
    std::fprintf(stderr, "line: %s\n", line.c_str());
    CHECK(has(line, "slow call: vkQueuePresentKHR "));
    CHECK(has(line, " on t-present:"));
    CHECK(has(line, "(present), before flip #4242; others in calls:"));
    CHECK(has(line, " t-submit:"));
    CHECK(has(line, " vkQueueSubmit "));
    CHECK(has(line, " ms (one job)"));
    // How long the submit had been in its call: about 120 ms.
    const std::size_t at = line.find(" vkQueueSubmit ");
    const double in_it = at == std::string::npos ? 0.0 : std::atof(line.c_str() + at + 15);
    CHECK(in_it >= 100.0 && in_it < 250.0);
    a.join();
    const std::string last = slow_calls::test_last_line();
    std::fprintf(stderr, "line: %s\n", last.c_str());
    CHECK(has(last, "slow call: vkQueueSubmit "));
    CHECK(has(last, "others in calls: none"));
    const slow_calls::TestCounts s = slow_calls::test_counts(slow_calls::kQueueSubmit), p = slow_calls::test_counts(slow_calls::kQueuePresent);
    CHECK(s.calls == 1 && s.over[0] == 1 && s.over[1] == 1 && s.over[2] == 1);
    CHECK(p.calls == 1 && p.over[0] == 1 && p.over[1] == 1 && p.over[2] == 0);
    CHECK(s.logged == 2 && s.unlogged == 0);
    // A file's site names only its file and line, never the path it was built from.
    std::thread c = call_on("t-alloc", kAlloc, 60);
    c.join();
    const std::string alloc = slow_calls::test_last_line();
    CHECK(has(alloc, "vkAllocateMemory") && has(alloc, "(render.cpp:6142)") && !has(alloc, "private"));
}

void test_buckets_and_threshold() {
    slow_calls::test_configure("50");
    for (int ms : {1, 15, 30, 60}) {
        Scope timed(kFence);
        sleep_ms(ms);
    }
    const slow_calls::TestCounts f = slow_calls::test_counts(slow_calls::kWaitFences);
    CHECK(f.calls == 4);
    CHECK(f.over[0] == 3 && f.over[1] == 1 && f.over[2] == 0);
    CHECK(f.logged == 1);  // the 60 ms one alone
    CHECK(f.ns >= 106'000'000ull);
    const std::string w = slow_calls::window_report();
    std::fprintf(stderr, "window: %s\n", w.c_str());
    CHECK(has(w, "vkWaitForFences 3/1/0"));
    CHECK(has(w, "the longest vkWaitForFences 6"));
    // The next window starts from nothing.
    CHECK(has(slow_calls::window_report(), "none of 10 ms or more in this window"));
    // A lower threshold logs sooner, and still counts from 10 ms.
    slow_calls::test_configure("5");
    {
        Scope timed(kFence);
        sleep_ms(7);
    }
    const slow_calls::TestCounts low = slow_calls::test_counts(slow_calls::kWaitFences);
    CHECK(low.logged == 1 && low.over[0] == 0);
}

void test_first_lines_in_full() {
    slow_calls::test_configure("1");
    for (int i = 0; i < 55; ++i) {
        Scope timed(kSubmit);
        sleep_ms(2);
    }
    const slow_calls::TestCounts s = slow_calls::test_counts(slow_calls::kQueueSubmit);
    CHECK(s.logged == 50 && s.unlogged == 5);
    CHECK(has(slow_calls::window_report(), "lines 50 so far, 5 more counted only"));
}

void test_poll_slack() {
    slow_calls::test_configure("50");
    {
        Scope timed(kWait, 50'000'000ull);  // a 50 ms step that came back after it: nothing wrong
        sleep_ms(62);
    }
    slow_calls::TestCounts w = slow_calls::test_counts(slow_calls::kWaitSemaphores);
    CHECK(w.calls == 1 && w.logged == 0 && w.over[0] == 1 && w.over[1] == 0);
    {
        Scope timed(kWait, 20'000'000ull);  // 90 ms for a 20 ms step: 70 ms past it
        sleep_ms(90);
    }
    w = slow_calls::test_counts(slow_calls::kWaitSemaphores);
    CHECK(w.logged == 1 && w.over[1] == 1);
    CHECK(has(slow_calls::test_last_line(), "(the timeline, up to 20 ms a step)"));
}

void test_lock_and_next() {
    slow_calls::test_configure("50");
    // A lock then the call made under it: one clock read shared, each counted as itself.
    {
        Scope timed(kLock);
        sleep_ms(55);
        timed.next(kSubmit);
        sleep_ms(3);
    }
    const slow_calls::TestCounts l = slow_calls::test_counts(slow_calls::kQueueLock), s = slow_calls::test_counts(slow_calls::kQueueSubmit);
    CHECK(l.calls == 1 && l.over[1] == 1);
    CHECK(s.calls == 1 && s.over[0] == 0 && s.ns >= 3'000'000ull && s.ns < 40'000'000ull);
    CHECK(has(slow_calls::test_last_line(), "slow call: queue-lock "));
}

void test_cp_phase() {
    slow_calls::test_configure("50");
    g_driver_phase_ns.store(0);
    std::thread([] {
        {
            Scope outside(kFence);  // not in a command buffer's packets: no phase
            sleep_ms(40);
        }
        slow_calls::CpPhase phase;
        Scope inside(kFence);
        sleep_ms(5);
    }).join();
    const std::uint64_t ns = g_driver_phase_ns.load();
    CHECK(ns >= 5'000'000ull && ns < 35'000'000ull);
}

void test_off() {
    slow_calls::test_configure("0");
    {
        Scope timed(kPresent);
        CHECK(timed.start_ns() == 0);
        CHECK(timed.end() == 0);
    }
    {
        Scope measured(kPresent, 0, true);  // a caller's own figure: the clock read, nothing timed
        sleep_ms(2);
        CHECK(measured.end() >= 2'000'000ull);
    }
    const slow_calls::TestCounts p = slow_calls::test_counts(slow_calls::kQueuePresent);
    CHECK(p.calls == 0);
    CHECK(slow_calls::window_report().empty() && slow_calls::in_calls_now().empty());
}

void test_in_calls_now() {
    slow_calls::test_configure("50");
    std::thread a = call_on("t-hung", kSubmit, 80);
    sleep_ms(40);
    const std::string now = slow_calls::in_calls_now();
    std::fprintf(stderr, "now: %s\n", now.c_str());
    CHECK(has(now, "t-hung:") && has(now, "vkQueueSubmit") && has(now, "(one job)"));
    a.join();
    CHECK(slow_calls::in_calls_now().empty());
}

// What a timed call costs over an untimed one: the clock read twice and the
// thread's slot written. Printed, not judged: a loaded machine's numbers move.
void cost() {
    constexpr int kN = 2'000'000;
    double ns[2];
    for (int on = 0; on < 2; ++on) {
        slow_calls::test_configure(on ? "50" : "0");
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kN; ++i) {
            Scope timed(kSubmit);
            asm volatile("" ::: "memory");
        }
        ns[on] = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / kN;
    }
    std::fprintf(stderr, "cost: a timed call %.1f ns, untimed %.1f ns: %.1f ns a call; at 3,000 submits and 260 presents a second %.1f us a second\n",
                 ns[1], ns[0], ns[1] - ns[0], (ns[1] - ns[0]) * 3260.0 / 1000.0);
}

}  // namespace

int main() {
    host_log_decide(true);
    test_line_names_the_others();
    test_buckets_and_threshold();
    test_first_lines_in_full();
    test_poll_slack();
    test_lock_and_next();
    test_cp_phase();
    test_off();
    test_in_calls_now();
    cost();
    if (g_failures) {
        std::fprintf(stderr, "slow_calls_test: %d failed\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "slow_calls_test: ok\n");
    return 0;
}
