// The command processor's labels written by the host once their submission
// has run (BBHOST_LABEL_LANDING=1, the default but on Windows), instead of as
// GPU fills into imported guest memory at the end of the submission's command
// buffer (BBHOST_LABEL_LANDING=0).
//
// A label - EVENT_WRITE_EOP, EVENT_WRITE_EOS, RELEASE_MEM: four or eight
// bytes that say "the work before this is done" - is read by the CPU (the
// game's waits, the command-arena acquire, GX's Map and its heap's release)
// and by the command processor's own WAIT_REG_MEM, which passes on the value
// recorded for it (gnm_exec.cpp's PendingLabels) without reading memory. No
// GPU work reads one. Written on the GPU, a label needed its guest memory
// imported and its fills ordered by hand (flush_deferred_writes_locked: a
// barrier between fills that overlap, the command-arena chunk's retirement
// after the labels written into the chunk before it). Written here, each
// submission's labels go into guest memory in the order the command processor
// made them, after the submission's work and before any later submission's
// labels: the order is the queue's, and no import is needed.
//
// What counts as done changes with it: a submission is "completed" for the
// rest of bbhost (host_gpu_submissions_completed, a slot's retirement, a wait
// for the GPU) once its labels are in memory, so whoever reads memory or
// writes it on the CPU after such a wait finds the labels there first, as
// after a GPU fill.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace label_landing {

struct Write {
    std::uint64_t va;
    std::uint32_t bytes;  // 4 or 8
    std::uint8_t data[8];
};

// What the landing asks the renderer. All may be called without its lock.
struct Hooks {
    std::uint64_t (*completed)() = nullptr;  // submissions the GPU has run (the timeline's value); ~0: none will run any more
    std::uint64_t (*submitted)() = nullptr;  // submissions handed to the queue (the last one's serial + 1)
    // Until completed() >= need or the timeout (the driver's wait); null, or
    // BBHOST_LABEL_WAIT=poll (the default on Windows, where a thread in the
    // driver's wait holds its lock against every other call): the thread polls.
    bool (*wait)(std::uint64_t need, std::uint64_t timeout_ns) = nullptr;
    bool (*mapped)(std::uint64_t va, std::size_t bytes) = nullptr;  // null: everything is
    void (*thread_start)() = nullptr;                               // names the thread, sets its class
};

bool on();                       // BBHOST_LABEL_LANDING (default 1, 0 on Windows)
void start(const Hooks& hooks);  // the landing thread (bb-labels), once
// At submit (under the renderer's lock): the writes submission `serial`
// carries, in the order they were made.
void add(std::uint64_t serial, const Write* writes, std::size_t n);
// While submission `serial` is recorded (before its labels are handed over):
// `bytes` from `src` - a host buffer its GPU work writes - copied to guest
// `va` once it has run, ahead of its labels and in the order added. `src`
// must stay valid until the submission has landed (a slot's retirement lands
// it first).
void add_copy(std::uint64_t serial, std::uint64_t va, const void* src, std::uint32_t bytes);
// Lands the writes of every submission before `completed` (submissions are
// counted from 0: `completed` of them have run); returns landed().
std::uint64_t land_through(std::uint64_t completed);
// Every submission before this count has run and has its writes in memory or
// on their way: raised before they are stored, where landed() is raised after.
// So a thread that has seen one of a submission's writes in memory - the game
// reading a label - finds this count past that submission (x86 keeps the
// landing thread's stores in order, and the reader's later load after its
// earlier one).
std::uint64_t landing();
// Every submission before this count has its writes in memory.
std::uint64_t landed();
// A diagnostic's look at memory just before the labels of every submission
// before `completed` land, while nothing has been told they have run (the
// renderer's race census, BBHOST_VB_RACE): called with the landing's lock
// held, from whichever thread lands them; it must take no lock but its own.
void set_before_land(void (*f)(std::uint64_t completed));
// Until landed() > seen or the timeout; true when it is.
bool wait_landed_past(std::uint64_t seen, std::chrono::microseconds timeout);
std::string report();

// The last kRecent landings, kept for a diagnostic that finds memory a
// packet was read from written over (hle/gnm_exec.cpp: a WAIT_REG_MEM no
// value can pass): each store's guest address, size, first dword, submission
// and when it landed.
constexpr std::uint32_t kRecent = 4096;
struct Landed {
    std::uint64_t va;
    std::uint32_t bytes, value;
    std::uint64_t serial, ns;
};
// Those inside [lo, hi), newest first, up to `cap`; the count found. Takes
// the landing's lock: not for a hot path.
int recent_into(std::uint64_t lo, std::uint64_t hi, Landed* out, int cap);
std::uint64_t now_ns();  // the clock `ns` is on

}  // namespace label_landing
