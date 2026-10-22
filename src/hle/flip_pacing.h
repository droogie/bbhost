#pragma once
// The flip keeper (hle/video.cpp; BBHOST_FLIP_KEEPER=0 turns it off): it puts
// the game back on the vblank tick when it has come back from a stall off it.
//
// Bloodborne submits a flip only once the one two back has completed, and
// video.cpp completes a flip on a tick, no sooner than two flip periods after
// the command processor queued it, so the game's wait meets a tick every
// frame and its frames go out one a refresh. That holds while the game's frame
// is ready before the tick its wait meets. After a stall - a load, a cut in a
// cutscene - the game can come back with its frames ready after the tick
// instead: the flip it waits for has then always just completed when it asks,
// its wait never blocks, and its frames follow its own timing until the next
// stall happens to put it back. Both states keep themselves. In test tours at
// 60 about one arrival in 40 came back this way (Fishing Hamlet, Rom's lake,
// Cathedral Ward and others), with 13-20 ms frames for its whole stay; in one
// shot of Cainhurst's first-arrival cutscene three recorded runs of four did,
// with frames of 16.0 and 17.3 ms in turn, and the fourth stayed on the tick.
//
// Holding every flip a tick longer does not help: a frame held to the tick is
// queued a period later, so the hold grows a period a frame and the game falls
// to 40. Holding a flip from the game's previous call instead of from its
// queuing changes nothing: that tick is the one the flip completes on already.
//
// So a kick, once: when the game, once on the tick, has asked for the flip
// status 0.3 ms or more after the tick without its wait blocking, three frames
// in a row while its frames are ready at 60, one flip waits one tick more. The
// game's next wait blocks until that tick, and from there the hold as it is
// keeps it on the tick - the state the tours' games returned to after their
// next stall, and stayed in. It costs one frame of two periods less those few
// ms, and the frames after it wait for the tick after they are ready, as they
// do whenever the game is on the tick (about 7 ms more than off it before they
// are queued to be shown, in the cutscene's replays). At most one kick in 300
// ticks.
//
// Only at 60 paced to a 60 Hz flip clock: the at-once paths (loading screens,
// 90, 120, uncapped) and 30's cap are not held this way. Pure; replayed
// against the cutscene's recorded flip timelines in tests/flip_pacing_test.cpp.
#include <cstdint>

#ifdef FLIP_PACING_MUTANTS
extern int g_flip_pacing_mutant;
#define FLIP_PACING_MUTANT(n) (g_flip_pacing_mutant == (n))
#else
#define FLIP_PACING_MUTANT(n) false
#endif

namespace flip_pacing {

// Whether the keeper runs: flips paced to the vblank at flip rate 0 under a
// 60 cap, on a flip clock within a few percent of 60 Hz.
inline bool keeper_paced(int flip_rate, int fps_cap, std::int64_t tick_ns) {
    return flip_rate == 0 && fps_cap == 60 && tick_ns >= 16000000 && tick_ns <= 17400000;
}

struct KeeperTuning {
    std::int64_t period_ns = 16666667;  // the flip clock's tick (60 Hz)
    std::int64_t late_ns = 300000;      // an ask this long after the tick or more, up to half a period, missed it
    int need = 3;                       // such asks in a row before a kick
    std::uint64_t held_first = 10;      // a kick only once the game has been held this many times (it was on the tick)
    std::int64_t slack_ns = 300000;     // its frames ready at 60: its last 8 asks' intervals average at most a period and this
    std::uint64_t cooldown = 300;       // ticks from one kick to the next
};

class TickKeeper {
public:
    explicit TickKeeper(KeeperTuning t = {}) : t_(t) {}

    // The game's first look at the flip status since its last flip call, at
    // `ns`, `phase_ns` after the last tick; `held` when its wait will block
    // (two or more of its flips in flight); `tick` the vblank count.
    void note_ask(std::int64_t ns, std::int64_t phase_ns, bool held, std::uint64_t tick) {
        ++asks_;
        if (ask_ns_ && ns > ask_ns_) {  // how often its frames are ready: the asks' intervals
            intervals_[next_interval_] = ns - ask_ns_;
            next_interval_ = (next_interval_ + 1) % kKept;
            if (intervals_n_ < kKept) ++intervals_n_;
        }
        ask_ns_ = ns;
        if (held && !FLIP_PACING_MUTANT(1)) {
            ++held_asks_;
            late_run_ = 0;
            return;
        }
        if (phase_ns < t_.late_ns || phase_ns >= t_.period_ns / 2) {
            late_run_ = 0;
            return;
        }
        ++late_asks_;
        if (++late_run_ < (FLIP_PACING_MUTANT(2) ? 1 : t_.need)) return;
        if (!FLIP_PACING_MUTANT(3) && !kept_pace()) return;  // slower than 60: a kick would only cost a frame
        if (held_asks_ < t_.held_first) return;              // never on the tick yet (the game starting)
        if (!FLIP_PACING_MUTANT(4) && kicks_ && tick < last_kick_tick_ + t_.cooldown) return;
        kick_ = true;
        late_run_ = 0;
    }

    // At a tick whose front flip may complete: true once after a kick was
    // armed, and that flip waits this tick.
    bool take_kick(std::uint64_t tick) {
        if (!kick_) return false;
        kick_ = false;
        ++kicks_;
        last_kick_tick_ = tick;
        return true;
    }

    std::uint64_t asks() const { return asks_; }
    std::uint64_t held_asks() const { return held_asks_; }
    std::uint64_t late_asks() const { return late_asks_; }
    std::uint64_t kicks() const { return kicks_; }

private:
    static constexpr int kKept = 8;
    // Its frames ready at 60: the last kKept asks' intervals average at most a
    // period and the slack.
    bool kept_pace() const {
        if (intervals_n_ < kKept) return false;
        std::int64_t sum = 0;
        for (int i = 0; i < kKept; ++i) sum += intervals_[i];
        return sum <= static_cast<std::int64_t>(kKept) * (t_.period_ns + t_.slack_ns);
    }

    KeeperTuning t_;
    std::int64_t ask_ns_ = 0;
    std::int64_t intervals_[kKept] = {};
    int intervals_n_ = 0, next_interval_ = 0;
    int late_run_ = 0;
    bool kick_ = false;
    std::uint64_t asks_ = 0, held_asks_ = 0, late_asks_ = 0, kicks_ = 0, last_kick_tick_ = 0;
};

}  // namespace flip_pacing
