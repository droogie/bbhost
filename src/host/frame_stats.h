#pragma once

// BBHOST_FRAME_STATS=1: once a second, how the game's frames went and which
// threads did the work - the flips the game submitted in that second, their
// average and worst interval, how many missed 16.7 and 33.3 ms, and the
// busiest threads by CPU (from /proc/self/task, so every guest and host
// thread is counted by name). The bottleneck at a given frame rate is the
// thread near 100%.
//
// BBHOST_FRAME_DETAIL=1 (with BBHOST_FRAME_STATS=1): also, after each
// `frames:` line, a `frame detail:` line with the second's draws by how their
// pixel shader ran and its draw failures by reason, and `frame intervals ms:`
// lines with every interval of that second - what a harness takes exact
// percentiles of over a window of its own (tools/area_check.py).

// Called on every flip the game submits.
void frame_stats_on_flip();

#include <cstdint>
#include <vector>

// Called on the thread the game's main loop runs on. With frame statistics
// on, its waits are timed and each second's line says how long it waited on
// other threads ("sync": condition variables, mutexes, semaphores, event
// flags) and slept (the frame limiter): a frame that runs long while the main
// loop barely waited was the game's own work.
void frame_stats_mark_main_thread();

// Times one wait (kind 0: on another thread, 1: a sleep, 2: file I/O) when it
// happens on the main loop's thread; otherwise costs a thread-local check.
class MainThreadWait {
public:
    explicit MainThreadWait(int kind);
    ~MainThreadWait();
    MainThreadWait(const MainThreadWait&) = delete;
    MainThreadWait& operator=(const MainThreadWait&) = delete;

private:
    int kind_;
    std::int64_t t0_ = 0;
};

// main_wait.cpp: BBHOST_FRAME_STATS=1, and the time waited of a kind since
// the last call (taken and reset).
bool frame_stats_enabled();
bool frame_stats_detail();  // BBHOST_FRAME_DETAIL=1 as well
std::uint64_t main_thread_waited_ns(int kind);

// The main loop's work in each frame: from the game's frame-time manager
// releasing one frame to its being called for the next (engine/frame_rate.cpp),
// so the limiter's wait is left out. At 60 a frame whose work passes 16.7 ms
// is one the game cannot hold, whatever the frame rate reads; the second's
// line gives the average, p95 and worst. main_frames_busy_us takes them.
void frame_stats_main_frame(std::uint64_t busy_us, std::uint64_t flip);  // flip: the game's flips so far
std::vector<std::uint32_t> main_frames_busy_us();

// BBHOST_HLE_COUNT: the thunk notes each HLE call's guest return address and
// the two above it, and the main loop's waits are summed by that chain; main_wait_sites_report logs the top
// sites (with the 300-flip HLE call counts) and starts over.
void main_wait_note_site(const std::uint64_t* chain);  // three return addresses, innermost first (0: none)
void main_wait_guest_base(std::uint64_t slide);
void main_wait_sites_report();
