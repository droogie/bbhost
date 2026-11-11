// The recorder's rest at a frame cap (host/recorder_rest.h), without the game:
// the windows that rest it and wake it, the command processor's room for a GPU
// rest, and the command processor's batched wakes of a sleeping, resting
// recorder.
#include "host/recorder_rest.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    ++g_failures;
    std::printf("FAIL: %s\n", what);
}

}  // namespace

int main() {
    using namespace recorder_rest;
    {
        // Two windows in a row with the limiter waiting 15% or more rest it; one does not.
        State s;
        check(!step(s, 0.30, -1.0, -1.0, false), "one window at the cap does not rest it");
        check(step(s, 0.30, -1.0, -1.0, false), "the second window in a row at the cap rests it");
        State t;
        check(!step(t, kRestSlack, -1.0, -1.0, false) && step(t, kRestSlack, -1.0, -1.0, false), "15% exactly counts");
        State u;
        check(!step(u, 0.149, -1.0, -1.0, false) && !step(u, 0.149, -1.0, -1.0, false) && !step(u, 0.149, -1.0, -1.0, false), "under 15% never rests it");
        // A window off the cap starts the count again.
        State v;
        step(v, 0.40, -1.0, -1.0, false);
        check(!step(v, 0.10, -1.0, -1.0, false), "a window under 15% between breaks the run");
        check(!step(v, 0.40, -1.0, -1.0, false), "... so the next at the cap is the first again");
        check(step(v, 0.40, -1.0, -1.0, false), "... and the one after it rests");
    }
    {
        // Resting: on at 8-15%, woken under 8% at once, and by a loading screen.
        State s;
        step(s, 0.5, -1.0, -1.0, false);
        step(s, 0.5, -1.0, -1.0, false);
        check(step(s, 0.10, -1.0, -1.0, false) && step(s, kWakeSlack, -1.0, -1.0, false), "resting goes on while the limiter waits 8% or more");
        check(!step(s, 0.079, -1.0, -1.0, false), "under 8% wakes it at once");
        check(!step(s, 0.5, -1.0, -1.0, false) && step(s, 0.5, -1.0, -1.0, false), "woken, two windows at the cap rest it again");
        check(!step(s, 0.5, -1.0, -1.0, true), "a loading screen wakes it");
        check(!step(s, 0.5, -1.0, -1.0, false), "... and the count starts over after it");
        check(step(s, 0.5, -1.0, -1.0, false), "... two windows");
        State w;
        check(!step(w, 0.5, -1.0, -1.0, true) && !step(w, 0.5, -1.0, -1.0, true) && !step(w, 0.5, -1.0, -1.0, true), "loading-screen windows never rest it");
    }
    {
        // The GPU: two windows busy 90% or more rest it; it ends once the median of six windows is under 80%.
        State g;
        check(!step(g, 0.0, 0.95, -1.0, false), "one GPU-bound window does not rest it");
        check(step(g, 0.0, 0.95, -1.0, false), "the second GPU-bound window in a row rests it");
        check(step(g, 0.0, 0.50, -1.0, false) && step(g, 0.0, 0.50, -1.0, false), "a GPU rest goes on while the six windows' median is 80% or more");
        check(step(g, 0.0, 0.50, -1.0, false), "... the median of 0.95 0.95 0.5 0.5 0.5 is still not under 80% with six not yet counted");
        check(!step(g, 0.0, 0.50, -1.0, false), "the median of six (0.95 0.95 0.5 0.5 0.5 0.5) under 80% ends it");
        State h;
        check(!step(h, 0.0, 0.95, -1.0, false) && !step(h, 0.0, 0.85, -1.0, false) && !step(h, 0.0, 0.95, -1.0, false), "a window under 90% breaks the GPU's run");
        check(step(h, 0.0, 0.95, -1.0, false), "... two in a row again");
        State k;
        check(!step(k, 0.0, -1.0, -1.0, false) && !step(k, 0.0, -1.0, -1.0, false), "the GPU not known: no GPU rest");
        step(k, 0.0, 0.95, -1.0, false);
        step(k, 0.0, 0.95, -1.0, false);
        check(!step(k, 0.0, -1.0, -1.0, false), "a GPU rest ends when its busy share is no longer known");
        State e;
        step(e, 0.0, 0.95, -1.0, false);
        step(e, 0.0, 0.95, -1.0, false);
        for (int k = 0; k < 4; ++k) step(e, 0.0, 0.80, -1.0, false);
        check(e.gpu, "a median of exactly 80% keeps the GPU's rest (under 80% ends it)");
        State m;
        step(m, 0.0, 0.95, -1.0, false);
        step(m, 0.0, 0.95, -1.0, false);
        check(!step(m, 0.0, 0.95, -1.0, true), "a loading screen ends a GPU rest");
        // A cap holding the frame with the GPU near full rests it for neither reason (the Steam Deck's 60 fps profile at
        // Central Yharnam: the GPU 88-91% busy at the cap).
        State b;
        for (int k = 0; k < 6; ++k) check(!step(b, 0.5, 0.95, -1.0, false), "a cap with the GPU 95% busy: no rest");
        State c2;
        for (int k = 0; k < 4; ++k) check(!step(c2, 0.30, 0.88, 0.70, false), "a cap with the GPU 88% busy, the CP 70%: no rest");
        // The GPU's rest only under the cap: the limiter waiting 15% does not count, under 15% does; reaching 15% ends it.
        State g2;
        check(!step(g2, kRestSlack, 0.95, -1.0, false) && !step(g2, kRestSlack, 0.95, -1.0, false), "GPU-bound at a cap: no GPU rest");
        State g3;
        check(!step(g3, 0.149, 0.95, -1.0, false) && step(g3, 0.149, 0.95, -1.0, false), "GPU-bound, the limiter under 15%: a GPU rest");
        check(step(g3, 0.0, 0.95, -1.0, false) && !step(g3, kRestSlack, 0.95, -1.0, false), "... ended once the limiter waits 15%");
    }
    {
        // A cap's rest needs the GPU's room: busy under 85% to begin, ended at 92%; the GPU not known, the cap alone.
        State r;
        check(!step(r, 0.30, 0.85, -1.0, false) && !step(r, 0.30, 0.85, -1.0, false), "a cap with the GPU 85% busy: no rest");
        check(!step(r, 0.30, 0.84, -1.0, false) && step(r, 0.30, 0.84, -1.0, false), "a cap with the GPU 84% busy: it rests");
        check(step(r, 0.30, 0.91, -1.0, false) && r.cap, "resting at the cap goes on with the GPU under 92%");
        check(!step(r, 0.30, 0.92, -1.0, false) && !r.cap, "the GPU busy 92% ends it");
        State n;
        check(!step(n, 0.30, -1.0, -1.0, false) && step(n, 0.30, -1.0, -1.0, false), "the GPU not known: the cap alone rests it");
        // As measured at caps: a true 60 on the laptop's RTX (the GPU ~40% busy) and the Deck's 30 fps default (41-53%) rest.
        State l;
        check(!step(l, 0.70, 0.40, 0.35, false) && step(l, 0.70, 0.40, 0.35, false), "a true 60 with the GPU 40% busy: it rests");
    }
    {
        // The command processor's room: a GPU rest begins only with it busy under 85% and ends at 92%; a cap's rest
        // does not ask (its own wake ends it once the frame misses the cap); not known (negative: bb-cp0's CPU-time
        // clock not had), the GPU's rule as before.
        State c;
        check(!step(c, 0.0, 0.95, 0.85, false) && !step(c, 0.0, 0.95, 0.85, false), "the CP busy 85%: no GPU rest");
        check(!step(c, 0.0, 0.95, 0.84, false) && step(c, 0.0, 0.95, 0.84, false), "under 85%: two GPU-bound windows rest it");
        check(step(c, 0.0, 0.95, 0.91, false) && c.gpu, "resting for the GPU goes on with the CP busy under 92%");
        check(!step(c, 0.0, 0.95, 0.92, false) && !c.gpu, "the CP busy 92% ends it");
        check(!step(c, 0.0, 0.95, 0.84, false) && step(c, 0.0, 0.95, 0.84, false), "... and two windows with room rest it again");
        State r;
        step(r, 0.0, 0.95, 0.84, false);
        check(!step(r, 0.0, 0.95, 0.86, false) && !step(r, 0.0, 0.95, 0.84, false), "a window without room breaks the GPU's run");
        State d;
        step(d, 0.5, 0.50, 0.99, false);
        check(step(d, 0.5, 0.50, 0.99, false) && d.cap && !d.gpu, "a cap with the GPU's room rests it whatever the CP's share");
        check(step(d, 0.5, 0.50, 0.99, false) && d.cap, "... and the cap's rest goes on with the CP busy");
        State u;
        check(!step(u, 0.0, 0.95, -1.0, false) && step(u, 0.0, 0.95, -1.0, false), "the CP's share not known: the GPU's rule as before");
        check(step(u, 0.0, 0.95, -1.0, false), "... and it goes on");
        // As measured (bb-cp0's CPU share): GPU-bound with the CP near its limit - the Steam Deck's 60 fps profile at
        // the Nightmare of Mensis (94-95%) and Old Yharnam (87-90%), uncapped Old Yharnam on a laptop's RTX 3070 Ti
        // (91-95%) - it never rests for the GPU; GPU-bound at 32-37 fps on a Radeon 680M (29-44%) it does.
        for (const double busy : {0.95, 0.94, 0.90, 0.87}) {
            State full;
            for (int k = 0; k < 8; ++k) check(!step(full, 0.0, 0.97, busy, false), "GPU-bound, the CP near its limit: no rest");
        }
        State idle;
        check(!step(idle, 0.0, 0.98, 0.44, false) && step(idle, 0.0, 0.98, 0.44, false), "GPU-bound, the CP busy 44%: it rests");
    }
    {
        // The command processor's wakes of a sleeping recorder: every block while it works; resting, every kRestBatch.
        std::uint32_t n = 0;
        for (int k = 0; k < 10; ++k) check(wake_now(false, n), "working: every block wakes it");
        check(n == 0, "working: nothing counted");
        std::uint32_t stale = 5;
        check(wake_now(false, stale), "working: a block wakes it whatever count a rest left");
        int woken = 0;
        for (std::uint32_t k = 0; k < 10 * kRestBatch; ++k) woken += wake_now(true, n) ? 1 : 0;
        check(woken == 10, "resting: one wake every kRestBatch blocks");
        check(n == 0, "resting: the count starts over at each wake");
        std::uint32_t m = kRestBatch - 2;
        check(!wake_now(true, m) && wake_now(true, m), "resting: the batch's last block wakes it");
    }
    static_assert(kWakeSlack < kRestSlack, "the wake line is under the rest line");
    static_assert(kCpRoom < kCpFull, "a GPU rest's room line is under its end line");
    static_assert(kGpuRoom < kGpuFull && kGpuRoom < kGpuBound, "a cap's GPU room line is under its end line and the GPU's own rest");
    static_assert(kRestBatch >= 2 && kRestBatch <= 1024, "a batch fits the ring of 2,048 blocks and the 1,024 packets");
    if (g_failures) {
        std::printf("recorder_rest_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("recorder_rest_test: ok\n");
    return 0;
}
