// The GX layer's catch-up (host/gpu_catch_up.h), without the game: a model of
// the renderer - its lock, the recording under way, a ring of slots retired in
// order, a GPU that runs each submission some time after it went in, labels
// that land once their submission has run - and a command processor that
// records and submits while a catch-up waits without the lock.
//
// After each catch-up it must leave what the flush left: the recording under
// way at the call submitted, every submission through it run, its labels in
// memory and its slot retired - and no label ever lands before its submission
// has run. It forces one submission, as the flush did, never a second. Its wait
// is never made holding the lock, and when it did not run out nothing waits
// under the lock afterwards: no fence there that has not signalled. The command
// processor's submissions made meanwhile are its own to retire. A wait that
// runs out waits for the same submissions under the lock. No timeline, or the
// switch off: the flush. No GPU: nothing.
//
// Then the timeline's poll (Windows): true at the read that reaches the need,
// false once the timeout has passed, ~0 past any need.
//
// Then eleven deliberate mistakes, each of which must be caught.
#define GPU_CATCH_UP_MUTANTS
#include "host/gpu_catch_up.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

int g_gpu_catch_up_mutant = 0;

namespace {

int g_failed = 0;
void check(bool ok, const char* what, int at = -1) {
    if (ok) return;
    ++g_failed;
    if (!g_gpu_catch_up_mutant && g_failed <= 24) std::printf("FAIL: %s (case %d)\n", what, at);
}

using u64 = std::uint64_t;
constexpr u64 kMs = 1'000'000ull;

// The renderer and its GPU, in virtual time.
struct Model {
    // The queue.
    bool locked = false;
    bool gpu_ok = true;
    bool has_timeline = true;
    u64 now = 0;               // virtual ns
    int slots = 4;             // the ring
    u64 flushes = 0;           // submissions made (the next one's serial)
    bool recording = false;    // a recording under way
    std::vector<int> rec_labels;                // the blocks the recording's labels free
    std::vector<u64> done_at;                   // per submission: when the GPU has run it
    std::vector<std::vector<int>> sub_labels;   // per submission: its labels' blocks
    std::vector<bool> in_flight, retired, landed;
    u64 next_cost = 2 * kMs;                    // GPU time of the next submission
    // The blocks: freed once their label has landed.
    std::vector<bool> block_free;
    // What a catch-up did.
    int submits = 0, flushes_called = 0, waits = 0, waits_locked = 0, retires_unlocked = 0;
    int fence_waits_locked = 0;                 // fences waited for under the lock that had not signalled (not the CP's own)
    bool cp_active = false;                     // the command processor's own submit (its ring's waits are its own)
    int early_landings = 0;                     // labels landed before their submission had run
    int cp_lock_blocked = 0;                    // the command processor found the lock held during a catch-up's wait
    // The command processor's work while a catch-up waits (without the lock).
    std::function<void(Model&)> meanwhile;
    bool in_wait = false;

    void lock() {
        check(!locked, "the lock taken twice");
        locked = true;
    }
    void unlock() {
        check(locked, "the lock given back unheld");
        locked = false;
    }
    bool ok() const { return gpu_ok; }
    bool timeline() const { return has_timeline; }
    bool ran(u64 s) const { return s < done_at.size() && done_at[s] <= now; }
    bool free_block(int b) const { return b >= static_cast<int>(block_free.size()) || block_free[b]; }  // (never recorded: free)
    void land(u64 s) {
        if (landed[s]) return;
        if (!ran(s)) ++early_landings;
        landed[s] = true;
        for (int b : sub_labels[s]) block_free[b] = true;
    }
    // A slot's retirement: its fence waited for (time goes by under the lock), what it held let go, its labels landed.
    void retire(u64 s) {
        if (!in_flight[s]) return;
        if (!ran(s)) {
            if (locked && !cp_active) ++fence_waits_locked;
            now = std::max(now, done_at[s]);
        }
        in_flight[s] = false;
        retired[s] = true;
        for (u64 k = 0; k <= s; ++k) land(k);  // fences on one queue complete in order: every earlier one's labels too
    }
    void submit_locked() {
        check(locked, "a submit without the lock");
        if (!cp_active) ++submits;  // (the catch-up's: the command processor's own submits are not counted)
        if (!recording) return;
        const u64 s = flushes++;
        const u64 start = std::max(now, s ? done_at[s - 1] : 0);
        done_at.push_back(start + next_cost);
        sub_labels.push_back(rec_labels);
        in_flight.push_back(true);
        retired.push_back(false);
        landed.push_back(false);
        rec_labels.clear();
        recording = false;
        // The slot it moves to is retired before it is recorded into (submit_locked's retire_slot_locked).
        if (flushes >= static_cast<u64>(slots)) retire(flushes - slots);
    }
    u64 submissions_locked() const { return flushes; }
    void retire_through_locked(u64 serial) {
        if (!locked) ++retires_unlocked;
        if (!flushes) return;
        serial = std::min(serial, flushes - 1);
        const u64 n = static_cast<u64>(slots);
        for (u64 s = flushes > n ? flushes - n : 0; s <= serial; ++s) retire(s);
    }
    void flush_locked() {
        ++flushes_called;
        submit_locked();
        retire_through_locked(flushes - 1);
    }
    bool wait(u64 need, u64 timeout_ns) {
        ++waits;
        if (locked) ++waits_locked;
        if (meanwhile && !in_wait) {
            in_wait = true;
            meanwhile(*this);
            in_wait = false;
        }
        if (!gpu_ok) return true;  // nothing more will run
        if (need == 0) return true;
        if (need - 1 < done_at.size() && done_at[need - 1] <= now + timeout_ns) {
            now = std::max(now, done_at[need - 1]);
            for (u64 s = 0; s < need; ++s) land(s);
            return true;
        }
        now += timeout_ns;
        return false;
    }

    // The command processor (or the game) records draws whose labels free `blocks`.
    void record(std::initializer_list<int> blocks) {
        recording = true;
        for (int b : blocks) {
            rec_labels.push_back(b);
            if (b >= static_cast<int>(block_free.size())) block_free.resize(b + 1, true);
            block_free[b] = false;
        }
    }
    // The command processor's own submit, after a job.
    void cp_submit() {
        if (locked) {
            ++cp_lock_blocked;
            return;
        }
        cp_active = true;
        lock();
        submit_locked();
        unlock();
        cp_active = false;
    }
};

struct Expect {
    gpu_catch_up::Way way;
    bool forced_done = true;  // the recording under way at the call submitted, run, its labels in, its slot retired
};

// One catch-up from the model's state; the checks every catch-up must pass.
gpu_catch_up::Result catch_up(Model& m, bool on, u64 timeout_ns, const Expect& want, int at) {
    const u64 flushes0 = m.flushes;
    const bool had_recording = m.recording;
    const std::vector<int> rec_blocks = m.rec_labels;
    const int submits0 = m.submits, flushes_called0 = m.flushes_called, fw0 = m.fence_waits_locked;
    const int waits_locked0 = m.waits_locked, unlocked0 = m.retires_unlocked, early0 = m.early_landings, blocked0 = m.cp_lock_blocked;
    // The command processor's submissions made during the wait: [cp_from, cp_to).
    u64 cp_from = 0, cp_to = 0;
    auto user = m.meanwhile;
    if (user) {
        m.meanwhile = [&cp_from, &cp_to, user](Model& mm) {
            cp_from = mm.flushes;
            user(mm);
            cp_to = mm.flushes;
        };
    }
    const gpu_catch_up::Result r = gpu_catch_up::run(m, on, timeout_ns);
    m.meanwhile = user;

    check(!m.locked, "the lock left held", at);
    check(r.way == want.way, "the way it took", at);
    check(m.early_landings == early0, "a label landed before its submission had run (a block freed early)", at);
    check(m.waits_locked == waits_locked0, "the wait made holding the renderer's lock", at);
    check(m.retires_unlocked == unlocked0, "slots retired without the renderer's lock", at);
    check(m.cp_lock_blocked == blocked0, "the command processor found the lock held while the catch-up waited", at);
    if (want.way == gpu_catch_up::Way::kNothing) {
        check(m.submits == submits0 && m.flushes == flushes0, "no GPU, yet something was submitted", at);
        return r;
    }
    // One forced submission, as the flush forced one - never a second.
    check(m.submits - submits0 == 1, "the submissions forced: one a catch-up", at);
    if (want.way == gpu_catch_up::Way::kFlushed) check(m.flushes_called - flushes_called0 == 1, "the flush: once", at);
    else check(m.flushes_called == flushes_called0, "the flush called on the catch-up's way", at);
    if (want.forced_done) {
        // The flush's outcome for everything submitted up to and with the forced submission.
        const u64 through = had_recording ? flushes0 + 1 : flushes0;
        if (want.way != gpu_catch_up::Way::kFlushed) check(r.need == through, "the submissions waited for", at);
        for (u64 s = 0; s < through; ++s) {
            if (s >= m.flushes) {
                check(false, "the forced submission never made", at);
                break;
            }
            check(m.ran(s), "a submission through the forced one not run", at);
            check(m.landed[s], "a submission through the forced one with its labels not in memory", at);
            check(!m.in_flight[s], "a slot through the forced one not retired (its memory not let go)", at);
        }
        for (int b : rec_blocks) check(m.free_block(b), "a block of the recording under way not free after the catch-up", at);
    }
    if (want.way == gpu_catch_up::Way::kCaughtUp) {
        check(m.fence_waits_locked == fw0, "a fence that had not signalled waited for under the lock after a wait that did not run out", at);
    }
    // The command processor's submissions made meanwhile: not the catch-up's to wait for or retire (the CP retires
    // them in its own course - its ring wraps, its own flushes).
    for (u64 s = cp_from; s < cp_to; ++s) {
        if (m.retired[s] && !m.ran(s)) check(false, "a later submission of the command processor retired before it ran", at);
    }
    return r;
}

void cases() {
    using gpu_catch_up::Way;
    constexpr u64 kTimeout = 200 * kMs;
    int at = 0;
    {
        // 1. The plain case: blocks wait on the recording under way; the GPU runs it in 2 ms.
        Model m;
        m.record({0, 1});
        m.cp_submit();
        m.record({2, 3});
        catch_up(m, true, kTimeout, {Way::kCaughtUp}, ++at);
    }
    {
        // 2. Earlier submissions still running too, several, slow ones: everything through the forced one.
        Model m;
        m.slots = 8;
        for (int i = 0; i < 5; ++i) {
            m.next_cost = (5 + i) * kMs;
            m.record({i});
            m.cp_submit();
        }
        m.record({10, 11});
        catch_up(m, true, kTimeout, {Way::kCaughtUp}, ++at);
    }
    {
        // 3. The command processor records and submits while the catch-up waits; its submissions run long after.
        Model m;
        m.slots = 16;
        m.record({0});
        m.cp_submit();
        m.record({1, 2});
        m.meanwhile = [](Model& mm) {
            mm.next_cost = 50 * kMs;
            for (int i = 0; i < 3; ++i) {
                mm.record({20 + i});
                mm.cp_submit();
            }
            mm.record({30});  // and a recording left under way
        };
        catch_up(m, true, kTimeout, {Way::kCaughtUp}, ++at);
        // Its recording is the command processor's: still recording, its blocks still waiting.
        check(m.recording && !m.free_block(30), "the command processor's recording made meanwhile submitted by the catch-up", at);
    }
    {
        // 4. The command processor fills the ring meanwhile: the forced submission's slot retired by the CP's wrap (its
        //    fence waited for in its own course); the catch-up's retirement finds it gone and waits for nothing more.
        Model m;
        m.slots = 4;
        m.record({0});
        m.meanwhile = [](Model& mm) {
            for (int i = 0; i < 6; ++i) {
                mm.next_cost = 1 * kMs;
                mm.record({40 + i});
                mm.cp_submit();
            }
        };
        catch_up(m, true, kTimeout, {Way::kCaughtUp}, ++at);
    }
    {
        // 5. No recording under way: nothing to submit, the submissions in flight waited for.
        Model m;
        m.record({0});
        m.cp_submit();
        m.record({1});
        m.cp_submit();
        catch_up(m, true, kTimeout, {Way::kCaughtUp}, ++at);
    }
    {
        // 6. Nothing ever submitted, nothing recording: nothing to wait for.
        Model m;
        const gpu_catch_up::Result r = catch_up(m, true, kTimeout, {Way::kCaughtUp}, ++at);
        check(r.need == 0 && m.waits == 0, "nothing submitted, yet a wait", at);
    }
    {
        // 7. The GPU slow (a stuck submission): the wait runs out, the same submissions waited for under the lock -
        //    one submit, not two, though the command processor recorded meanwhile.
        Model m;
        m.slots = 16;
        m.next_cost = 1500 * kMs;
        m.record({0, 1});
        m.meanwhile = [](Model& mm) { mm.record({50}); };
        const gpu_catch_up::Result r = catch_up(m, true, kTimeout, {Way::kWaitedLocked}, ++at);
        check(m.fence_waits_locked == 1, "the run-out wait's fence waited for under the lock", at);
        check(m.recording && !m.free_block(50), "the command processor's recording submitted by a run-out catch-up", at);
        (void)r;
    }
    {
        // 8. A run-out wait with the command processor's slow submissions behind it: only the forced one's waited for.
        Model m;
        m.slots = 16;
        m.next_cost = 400 * kMs;
        m.record({0});
        m.meanwhile = [](Model& mm) {
            mm.next_cost = 3000 * kMs;
            mm.record({60});
            mm.cp_submit();
        };
        catch_up(m, true, kTimeout, {Way::kWaitedLocked}, ++at);
        check(m.in_flight.size() > 1 && m.in_flight[1] && !m.ran(1), "the command processor's later submission waited for by a run-out catch-up", at);
    }
    {
        // 9. No timeline: the flush (everything recorded, run and retired, under the lock).
        Model m;
        m.has_timeline = false;
        m.record({0});
        m.cp_submit();
        m.record({1});
        catch_up(m, true, kTimeout, {Way::kFlushed}, ++at);
        check(m.waits == 0, "a wait on a timeline that is not there", at);
    }
    {
        // 10. BBHOST_GX_CATCH_UP=0: the flush.
        Model m;
        m.record({0});
        m.cp_submit();
        m.record({1});
        catch_up(m, false, kTimeout, {Way::kFlushed}, ++at);
        check(m.waits == 0, "the switch off, yet a wait without the lock", at);
    }
    {
        // 11. No GPU: nothing done.
        Model m;
        m.record({0});
        m.gpu_ok = false;
        catch_up(m, true, kTimeout, {Way::kNothing}, ++at);
    }
    {
        // 12. The GPU lost while the catch-up waits: nothing retired after (fence_wait_locked waits for nothing more).
        Model m;
        m.record({0});
        m.meanwhile = [](Model& mm) { mm.gpu_ok = false; };
        const int submits0 = m.submits;
        ++at;
        gpu_catch_up::run(m, true, kTimeout);
        check(!m.locked, "the lock left held after a loss", at);
        check(m.submits - submits0 == 1, "a loss: one submit", at);
    }
    {
        // 13. A run of catch-ups as the GX threads make them, each between the command processor's jobs, with the
        //     GPU's speed changing: each leaves the flush's outcome, the ring wraps through them.
        Model m;
        m.slots = 6;
        u64 cost = 1 * kMs;
        for (int i = 0; i < 40; ++i) {
            cost = (cost * 7 + 3 * kMs) % (90 * kMs) + kMs;
            m.next_cost = cost;
            m.record({100 + i});
            if (i % 3 == 0) m.cp_submit();
            m.record({200 + i});
            m.meanwhile = [i](Model& mm) {
                if (i % 2) {
                    mm.record({300 + i});
                    mm.cp_submit();
                }
            };
            catch_up(m, true, 1000 * kMs, {Way::kCaughtUp}, ++at);  // (the chain behind it can be ~270 ms)
            m.meanwhile = nullptr;
        }
    }
}

// The timeline's poll (Windows): reads of a value that moves on with each read, a clock that moves with each look.
void poll_cases() {
    int at = 100;
    {
        // Reaches the need exactly at the fifth read.
        u64 v = 0, t = 0;
        int pauses = 0;
        const bool ok = gpu_catch_up::poll_until(
            5, 100 * kMs, [&] { return v++; }, [&] { return t += kMs; }, [&](int k) { check(k == pauses++, "the pauses in order", at); });
        check(ok && v == 6 && pauses == 5, "the poll: true at the read that reaches the need", ++at);
    }
    {
        // Never reaches it: false once the timeout has passed, not before.
        u64 t = 0;
        int reads = 0;
        const bool ok = gpu_catch_up::poll_until(
            10, 20 * kMs, [&] { ++reads; return u64{9}; }, [&] { return t += kMs; }, [](int) {});
        check(!ok, "the poll: false once the timeout passed", ++at);
        check(reads >= 10, "the poll: gave up before its timeout", at);
    }
    {
        // ~0 (no GPU any more): past any need, at once.
        u64 t = 0;
        int reads = 0;
        const bool ok = gpu_catch_up::poll_until(
            ~0ull - 1, kMs, [&] { ++reads; return ~0ull; }, [&] { return t += kMs; }, [](int) {});
        check(ok && reads == 1, "the poll: ~0 past any need", ++at);
    }
    {
        // Already there: no pause.
        u64 t = 0;
        int pauses = 0;
        const bool ok = gpu_catch_up::poll_until(
            3, kMs, [] { return u64{3}; }, [&] { return t += kMs; }, [&](int) { ++pauses; });
        check(ok && pauses == 0, "the poll: already there", ++at);
    }
}

}  // namespace

int main() {
    cases();
    poll_cases();
    int failed = g_failed;
    static const char* const kMistakes[11] = {
        "`need` taken before the submit (one short of the forced submission)",
        "the recording under way not submitted",
        "the wait made under the renderer's lock (the flush's hold again)",
        "the wait one short of the forced submission",
        "a run-out wait taken for the flush (a second submit)",
        "a run-out wait taken for done (nothing waited for)",
        "the slots retired without the renderer's lock",
        "the command processor's submissions since retired too",
        "the forced submission's slot left unretired",
        "the poll asking one more than the need",
        "the poll's timeout taken for done",
    };
    for (int m = 1; m <= 11; ++m) {
        g_failed = 0;
        g_gpu_catch_up_mutant = m;
        cases();
        poll_cases();
        g_gpu_catch_up_mutant = 0;
        if (g_failed == 0) {
            std::printf("FAIL: mistake %d (%s) not caught\n", m, kMistakes[m - 1]);
            ++failed;
        } else {
            std::printf("mistake %d (%s) caught: %d checks failed\n", m, kMistakes[m - 1], g_failed);
        }
    }
    std::printf(failed ? "gpu_catch_up_test: %d failed\n" : "gpu_catch_up_test: ok\n", failed);
    return failed ? 1 : 0;
}
