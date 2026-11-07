// The driver calls a frame can wait on, timed on every thread that makes
// them, so a freeze names the call behind it (BBHOST_SLOW_CALL_MS=<ms>, 50 by
// default; 0: off, nothing timed).
//
// Why. A freeze in the world - a flip a seventh of a second late, the GPU idle
// meanwhile - showed in the log only as its shape: one wait of 140 ms for the
// submission thread to hand a submission to the driver, or 186 ms of the
// command processor's time in its packets with no draw, label, lock or
// pipeline behind it. Either could be a driver call that did not come back,
// another thread's or the command processor's own: a present on NVIDIA's
// Windows driver has been seen holding the driver's own lock with the
// submission thread's call queued behind it (window.cpp, present_own_queue_on).
// Which call it was, and who waited behind it, the log could not say.
//
// What. Each timed call - the submits, the present and the acquire, the
// sparse binds, the waits for fences, for the timeline and for an idle device
// or queue, the memory the renderer allocates, maps and frees in play, the
// command buffers' begin and end - is a Scope around it. A thread in one has
// its call and since when in a slot of a fixed registry (one a thread,
// atomics, no lock), so a call that takes longer than the threshold is logged
// on one line with every other thread in a timed call at that moment, its
// call and how long it had been in it:
//
//   slow call: vkQueuePresentKHR 201.3 ms on bb-present:4410 (present), before flip #12034; others in calls:
//   bb-submit:4402 vkQueueSubmit 196.0 ms (one job), bb-cp0:4398 submitted-wait 150.2 ms (a submission)
//
// (one line; the flip is the `stall:` line's number). Two of ours are timed
// as well, though no driver call: a wait for the queue's lock (QueueGuard, the
// submission thread's own) and a wait for a submission the submission thread
// has still to hand to the driver - the threads that wait behind a driver call
// without making one. A wait for a fence or the timeline holds the GPU's own
// time too; one bounded by a short timeout (a poll step) counts by its time
// past it.
//
// The first 50 such lines are logged in full, then counted. The 300-flip
// report says how many calls of each kind took 10, 50 and 200 ms or more in
// its window, the exit report since the start with their counts and time, and
// a hang watchdog's report names the calls under way. The command processor's
// time in timed calls is a phase of its own ("driver" on the `stall:` and
// `cp-thread ms:` lines): inside its packets, where `exec` includes it as it
// includes the other phases, and after each job, where no phase did - the
// job's submission and the slot it retires.
//
// The cost: the clock read as a call starts and as it ends - two reads, ~20
// ns each - and a few stores into the thread's own slot; where a call was
// timed already (the memory sites, a command buffer's end and begin, the
// fence waits) its reads are shared, not added.
//
// BBHOST_TEST_SLOW_CALL=<call>:<ms>[:<every>[:lock]] (tests): <ms> more in
// every <every>th call of that kind (present, acquire, submit, bind-sparse,
// wait-fences, wait-semaphores, semaphore-value, device-idle, queue-idle,
// allocate, map, free, begin, end; 1 by default: each one), slept inside the
// call's scope as if the driver took that long. With `lock` the delayed call
// holds a lock that every other thread's timed driver call takes as it
// starts, as a driver's own lock would: what waits behind it then shows in
// the line.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace slow_calls {

enum Call : std::uint8_t {
    kQueueSubmit,     // vkQueueSubmit
    kQueuePresent,    // vkQueuePresentKHR
    kAcquireImage,    // vkAcquireNextImageKHR
    kBindSparse,      // vkQueueBindSparse
    kWaitFences,      // vkWaitForFences
    kWaitSemaphores,  // vkWaitSemaphores
    kSemaphoreValue,  // vkGetSemaphoreCounterValue
    kDeviceIdle,      // vkDeviceWaitIdle
    kQueueIdle,       // vkQueueWaitIdle
    kAllocateMemory,  // vkAllocateMemory
    kMapMemory,       // vkMapMemory
    kFreeMemory,      // vkFreeMemory
    kBeginCommands,   // vkBeginCommandBuffer
    kEndCommands,     // vkEndCommandBuffer
    // Ours, no driver call: waits for another thread's.
    kQueueLock,  // the queue's lock, and before a QueueGuard's the submission thread's queue emptied
    kSubmitted,  // a submission the submission thread has still to hand to the driver
    kCallCount
};
// The call's name in the log ("vkQueueSubmit", "queue-lock", "submitted-wait").
const char* call_name(Call c);
inline bool is_driver_call(Call c) { return c < kQueueLock; }

// One place a call is made: static, its address what a thread's slot holds.
// `where` is a few words ("one job", "present") or, with a line, a file.
struct Site {
    Call call;
    const char* where;
    int line = 0;
};

// The calling thread's record (slow_calls.cpp): taken at its first timed
// call, given back when it exits.
struct Slot;

// Whether calls are timed: a threshold, and the tooling on (log.h).
bool on();

// A timed call: from the constructor to end() or the destructor. `slack_ns`:
// a wait bounded by a timeout that short - a poll step, which comes back
// after it with nothing wrong - counts and is logged by its time past it.
// `measure`: the clock is read even when calls are not timed, for a caller
// that keeps its own figure from end()'s.
class Scope {
public:
    explicit Scope(const Site& site, std::uint64_t slack_ns = 0, bool measure = false) noexcept : site_(&site), slack_(slack_ns) {
        begin(measure);
    }
    ~Scope() {
        if (t0_) end();
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    // The call's time in nanoseconds; 0 when it was not timed (or ended).
    std::uint64_t end() noexcept;
    // The same with the time it ended read by the caller already.
    std::uint64_t end_at(std::uint64_t t1) noexcept;
    // Ends this call and starts `next` on the same clock read: a lock and
    // the call made under it, three reads for the two.
    void next(const Site& next) noexcept;
    // When it started (host_clock_monotonic_ns), 0 when not timed.
    std::uint64_t start_ns() const { return t0_; }

private:
    void begin(bool measure) noexcept;

    const Site* site_;
    std::uint64_t slack_;
    std::uint64_t t0_ = 0;
    Slot* slot_ = nullptr;  // where it is published (null: no slot, not timed, or inside another timed call)
    bool counted_ = false;  // timed (not only measured)
};

// While it lives, the calling thread's timed calls are the command
// processor's "driver" phase as well (gnm_exec.cpp's cp_thread).
class CpPhase {
public:
    CpPhase() noexcept;
    ~CpPhase();
    CpPhase(const CpPhase&) = delete;
    CpPhase& operator=(const CpPhase&) = delete;

private:
    Slot* slot_ = nullptr;
    bool prev_ = false;
};

// The 300-flip report's line: this window's calls of 10, 50 and 200 ms and
// more by call, and the lines so far ("" while nothing is timed).
std::string window_report();
// The exit report's lines, since the start (empty while nothing is timed).
void final_report();
// Every thread in a timed call now, longest first ("" when none): the hang
// watchdog's report.
std::string in_calls_now();
// The timed call one thread is in now - `tid` its host_thread_id() - once it
// has been in it for `min_ns` or more: the call, its time so far and where, in
// the slow-call lines' words. False while that thread is in none, or in one
// for less, and while nothing is timed. For a line about something else that
// says what that thread was waiting on as it happened: the window's focus
// lines ask it of the presenter (host/window.cpp).
struct ThreadCall {
    Call call = kCallCount;
    std::uint64_t ns = 0;
    char where[96] = {};
};
bool thread_in_call(std::uint32_t tid, std::uint64_t min_ns, ThreadCall* out);

// For tests (tests/slow_calls_test.cpp): the threshold and the test hook set
// again as from these values (null: unset), the counters zeroed.
void test_configure(const char* slow_call_ms, const char* test_slow_call = nullptr);
struct TestCounts {
    std::uint64_t calls = 0, ns = 0, over[3] = {}, logged = 0, unlogged = 0, delays = 0;
};
TestCounts test_counts(Call c);
// The last line logged (its text, without the log's prefix).
std::string test_last_line();

}  // namespace slow_calls
