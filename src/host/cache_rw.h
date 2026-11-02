// GX on Vulkan, stage 4b.1a: the caches a draw reads, safe for readers beside
// the command processor (the pool that reads them: host/gxvk_lists.h).
//
// The command processor stays the caches' only writer (every write is under
// g.mu, as before). While a stage 4b pool reads them (BBHOST_GXVK_LISTS=1 or
// 2), a writer takes this lock exclusive around every change a reader could
// see - an insert into one of the maps, an entry made, rewritten or let go, an
// epoch moved - and a reader holds it shared for the lookups of a draw. The
// command processor's own finds take nothing: nobody else writes. A reader
// takes the lookups' reader paths: no memo of the command processor's is read
// or written, nothing is made or updated, and a miss is "not made yet", for
// the command processor to make in order.
//
// The lock prefers its writer: a reader that finds the writer's bit set waits
// for it, so the command processor waits at most for the draws' lookups under
// way (a few microseconds), never for a stream of them. Each write moves
// gen(), which a reader notes with its answers: the command processor's
// compare (BBHOST_GXVK_LISTS=2) tells an answer that changed since apart from
// a reader's wrong one.
//
// With the pool off (BBHOST_GXVK_LISTS unset, 0 or census) nothing is taken:
// the writer's guard is one predictable branch.
#pragma once

#include <atomic>
#include <cstdint>

namespace cache_rw {

// Whether any reader may run: the pool's switch, read before any of the
// game's code runs.
extern const bool g_readers;

// True on a reader's thread while it holds the lock shared.
extern thread_local constinit bool t_reader;
inline bool reader() { return t_reader; }

// The writer's nesting on this thread (a write inside a write takes nothing).
extern thread_local constinit int t_write_depth;

void lock_slow();
void unlock_slow();
void lock_shared_slow();
void unlock_shared_slow();

// The caches, as the writes name them (gxvk_lists::LookKind's order).
enum Cache : int { kPrograms, kFetch, kPaths, kPlans, kViews, kSamplers, kCaches };

// Each cache's writes so far (moved under the exclusive lock; a reader reads
// them under the shared one).
extern std::atomic<std::uint64_t> g_gen[kCaches];
inline std::uint64_t gen(int cache) { return g_gen[cache].load(std::memory_order_relaxed); }

// A writer around a change of `cache` a reader could see.
struct Write {
    bool held = false;
    explicit Write(int cache) {
        if (!g_readers || t_reader) return;
        if (t_write_depth++ == 0) lock_slow();
        g_gen[cache].fetch_add(1, std::memory_order_relaxed);
        held = true;
    }
    ~Write() {
        if (held && --t_write_depth == 0) unlock_slow();
    }
    Write(const Write&) = delete;
    Write& operator=(const Write&) = delete;
};

// A reader's lookups (a pool thread's).
struct Read {
    Read() {
        lock_shared_slow();
        t_reader = true;
    }
    ~Read() {
        t_reader = false;
        unlock_shared_slow();
    }
    Read(const Read&) = delete;
    Read& operator=(const Read&) = delete;
};

// Counts for the report: exclusive takes, those that waited for a reader, the
// reader takes that waited for the writer.
struct Counts {
    std::uint64_t writes = 0, writes_waited = 0, reads = 0, reads_waited = 0;
};
Counts counts();

}  // namespace cache_rw
