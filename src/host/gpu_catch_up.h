// The GX layer's catch-up (host_gpu_catch_up, gpu.cpp): what host_gpu_flush
// gave the resource-table block reclaim and the discard rename
// (decomp/gx/block_reclaim.cpp, rename_wait.cpp) when every block of a pool
// still waits on its label - the recording under way submitted, every
// submission through it run and its labels in memory, its slot retired - with
// the renderer's lock no longer held through the GPU's time:
//  1. under the lock: the recording under way handed over (the flush's own
//     submit, the same call at the same point), and `need`, the submissions
//     made with it;
//  2. without the lock: until `need` submissions have run (the timeline) and
//     their labels are in memory - the command processor records on
//     meanwhile, where it waited for the lock before;
//  3. under the lock: the slots through submission need - 1 retired. Those
//     have run, so nothing waits, and what they held (staging, descriptor
//     pools, dead images and memory) is let go where the flush let it go.
// A wait that runs out (`timeout_ns`: the GPU slow, or stuck) leaves step 3 to
// wait for those same submissions under the lock, as the flush waited for
// them, with its stuck-GPU handling (fence_wait_locked) - and no second
// submit, so the submissions the GX layer forces stay one a catch-up. With no
// timeline to wait on, or BBHOST_GX_CATCH_UP=0, it is the flush.
//
// The flush held the lock through its wait: on an RTX 4070 with a 4 GB card's
// memory acted out (BBHOST_TEST_VRAM_MB=2950) at Old Yharnam the command
// processor waited 1.96 ms a flip for the lock, 1.66 ms of it the GX threads'
// fence waits inside their flushes.
//
// Pure below, the renderer's calls passed in: tests/gpu_catch_up_test.cpp
// drives it with a model of the queue and its labels, and its mistakes.
#pragma once

#include <cstdint>

#ifdef GPU_CATCH_UP_MUTANTS
extern int g_gpu_catch_up_mutant;
#define GPU_CATCH_UP_MUTANT(n) (g_gpu_catch_up_mutant == (n))
#else
#define GPU_CATCH_UP_MUTANT(n) false
#endif

namespace gpu_catch_up {

enum class Way : std::uint8_t {
    kNothing,       // no GPU (g.ok false): nothing done, as the flush did nothing
    kFlushed,       // the flush: no timeline, or BBHOST_GX_CATCH_UP=0
    kCaughtUp,      // waited for outside the lock, retired without a wait
    kWaitedLocked,  // the wait ran out: the same submissions waited for under the lock
};

struct Result {
    Way way = Way::kNothing;
    std::uint64_t need = 0;  // the submissions waited for (the forced one is need - 1)
};

// The renderer, as the catch-up uses it (gpu.cpp's CatchUpOps; the test's model):
//   lock(), unlock()                    the renderer's lock (g.mu)
//   ok()                                a GPU to run work (g.ok), under the lock
//   timeline()                          a timeline semaphore to wait on
//   flush_locked()                      the flush: submit, then wait for and retire everything submitted
//   submit_locked()                     the recording under way handed over (nothing when none)
//   submissions_locked()                submissions made so far (g.flushes)
//   wait(need, timeout_ns)              without the lock: true once `need` submissions have run and their labels
//                                       are in memory; false when the timeout came first
//   retire_through_locked(serial)       the slots through submission `serial` retired, each fence waited for first
template <class Ops>
struct Locked {
    Ops& o;
    explicit Locked(Ops& ops) : o(ops) { o.lock(); }
    ~Locked() { o.unlock(); }
    Locked(const Locked&) = delete;
    Locked& operator=(const Locked&) = delete;
};

template <class Ops>
Result run(Ops& o, bool on, std::uint64_t timeout_ns) {
    Result res;
    {
        Locked<Ops> l(o);
        if (!o.ok()) return res;
        if (!on || !o.timeline()) {
            o.flush_locked();
            res.way = Way::kFlushed;
            return res;
        }
        if (GPU_CATCH_UP_MUTANT(1)) res.need = o.submissions_locked();  // (taken before the submit: one short)
        if (!GPU_CATCH_UP_MUTANT(2)) o.submit_locked();                  // (the recording under way left recording)
        if (!GPU_CATCH_UP_MUTANT(1)) res.need = o.submissions_locked();
        if (GPU_CATCH_UP_MUTANT(3)) {  // (the wait under the lock: the flush's hold again)
            const bool ran = o.wait(res.need, timeout_ns);
            if (res.need) o.retire_through_locked(res.need - 1);
            res.way = ran ? Way::kCaughtUp : Way::kWaitedLocked;
            return res;
        }
    }
    res.way = Way::kCaughtUp;
    if (!res.need) return res;  // nothing ever submitted: nothing to wait for
    const std::uint64_t want = GPU_CATCH_UP_MUTANT(4) ? res.need - 1 : res.need;  // (one short of the forced one)
    const bool ran = o.wait(want, timeout_ns);
    if (!ran) res.way = Way::kWaitedLocked;
    if (GPU_CATCH_UP_MUTANT(5) && !ran) {  // (a timeout taken for the flush: a second submit)
        Locked<Ops> l(o);
        if (o.ok()) o.flush_locked();
        return res;
    }
    if (GPU_CATCH_UP_MUTANT(6) && !ran) return res;  // (a timeout taken for done: nothing waited for)
    if (GPU_CATCH_UP_MUTANT(7)) {                    // (retired without the lock)
        if (o.ok()) o.retire_through_locked(res.need - 1);
        return res;
    }
    Locked<Ops> l(o);
    if (!o.ok()) return res;  // lost meanwhile: nothing more is waited for (fence_wait_locked)
    if (GPU_CATCH_UP_MUTANT(8)) o.retire_through_locked(o.submissions_locked() - 1);  // (the command processor's since too)
    else if (GPU_CATCH_UP_MUTANT(9)) o.retire_through_locked(res.need >= 2 ? res.need - 2 : 0);  // (the forced one kept)
    else o.retire_through_locked(res.need - 1);
    return res;
}

// The timeline read until it reaches `need` or `timeout_ns` has passed on
// now_ns() - the thread asleep between reads, pause(k) after the k-th - for a
// driver that holds its own lock through a wait inside it (NVIDIA's on
// Windows: every other thread's call queued behind it, the one that submits
// the work waited for too). read() is ~0 once nothing more will run: past any
// need.
template <class Read, class Now, class Pause>
bool poll_until(std::uint64_t need, std::uint64_t timeout_ns, Read&& read, Now&& now_ns, Pause&& pause) {
    const std::uint64_t t0 = now_ns();
    for (int k = 0;; ++k) {
        const std::uint64_t v = read();
        if (GPU_CATCH_UP_MUTANT(10) ? v > need : v >= need) return true;  // (one more than asked)
        if (now_ns() - t0 >= timeout_ns) return GPU_CATCH_UP_MUTANT(11);   // (a timeout taken for done)
        pause(k);
    }
}

}  // namespace gpu_catch_up
