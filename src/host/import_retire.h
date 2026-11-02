#pragma once
// Window imports let go while idle (gpu.cpp retire_idle_imports_locked,
// BBHOST_IMPORT_RETIRE): the switch, and which imports a check lets go. Kept
// apart from Vulkan so a test can drive them.
//
// Every window import stays until bbhost ends: a window an area's arrival
// imported for its meshes, its constant buffers' copy-backs or a prefetch the
// draws never used is pinned memory - which Windows counts as the process's
// shared GPU memory - long after the game has gone elsewhere. A check after a
// submission, once a second, lets go of an import none of whose windows
// anything used - a binding, a copy, a device address, a walk - for the
// switch's seconds: its windows' page-table entries point at the sink at once
// (an idle window is read by no work in flight, and the sink is memory every
// submission has seen since the start, so no walk under way is handed an
// address its job cannot see yet - the reason a merge's switch waits for the
// queue, host/import_table.h), and the import is freed once the work recorded
// so far has run, as a merge's. The window's next use imports it again by the
// path every first use takes, with its fallbacks if that import fails: no
// draw is skipped for it.
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace import_retire {

enum Mode : int { kOff = 0, kOn = 1, kCount = 2 };  // kCount: what kOn would let go, counted; nothing let go

struct Switch {
    int mode = kOff;
    std::uint64_t after_ms = 60000;          // a window's time without a use before its import goes
    std::uint64_t per_check = 128ull << 20;  // at most this much let go a check (a second): the frees spread
};

// BBHOST_IMPORT_RETIRE (`mode`: 1 on, 2 counted, 0 off; unset or anything
// else: `dflt`), BBHOST_IMPORT_RETIRE_S (`secs`: 1 or more, else 60) and
// BBHOST_IMPORT_RETIRE_MB (`mb`: 16 or more, else 128).
inline Switch parse(const char* mode, const char* secs, const char* mb, int dflt = kOff) {
    Switch s;
    s.mode = dflt;
    if (mode && (mode[0] == '0' || mode[0] == '1' || mode[0] == '2') && mode[1] == 0) s.mode = mode[0] - '0';
    if (secs && *secs) {
        const double v = std::atof(secs);
        if (v >= 1) s.after_ms = static_cast<std::uint64_t>(v * 1000);
    }
    if (mb && *mb) {
        const unsigned long long v = std::strtoull(mb, nullptr, 10);
        if (v >= 16) s.per_check = static_cast<std::uint64_t>(v) << 20;
    }
    return s;
}

// Whether import `c`, over windows [a, b] of a chunk, is idle at `now`: both
// views of each window name it - `window`, the import each window's bindings
// take, and `table`, the one its page-table entries name (none yet is fine: a
// fresh window's entries) - and none of them was used within `after_ms`. A
// window not stamped yet (0: made before the clock's first tick) is stamped
// `now`, so it counts as just used.
template <class C>
inline bool idle(C* const* window, C* const* table, std::atomic<std::uint64_t>* used, std::uint32_t a, std::uint32_t b, const C* c,
                 std::uint64_t now, std::uint64_t after_ms) {
    bool ok = a <= b;
    for (std::uint32_t w = a; w <= b; ++w) {
        if (window[w] != c || (table[w] && table[w] != c)) ok = false;  // a merge's view still moving over it
        const std::uint64_t u = used[w].load(std::memory_order_relaxed);
        if (!u) used[w].store(now, std::memory_order_relaxed);
        if (!u || now < u || now - u < after_ms) ok = false;
    }
    return ok;
}

// Whether a check that has let go of `so_far` bytes lets go of another
// (the first always: `so_far` is 0).
inline bool fits(std::uint64_t so_far, std::uint64_t per_check) { return so_far < per_check; }

}  // namespace import_retire
