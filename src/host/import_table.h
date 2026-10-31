#pragma once
// Which import the guest page table's entries point into, window by window
// (window imports, gpu.cpp install_import_locked), kept apart from the import
// that bindings and copies take (DmemChunk::window), and apart from Vulkan so
// a test can drive it.
//
// Why two views. The page table is host memory the GPU reads while it runs:
// a shader that walks it (the tessellation LS pass, a dispatch's reads through
// V#s) takes whatever device address an entry holds at that moment. A new
// import's addresses are only safe for the submissions made after it: amdgpu
// maps a new buffer object's addresses with a page-table update of its own,
// on the SDMA engine, that the command submissions made later wait for and
// the ones already in the queue do not. Writing a new import's addresses into
// entries that already named an older import - a merge, which replaces the
// imports a wider range overlaps - can therefore hand a walk that is already
// running an address its job cannot see yet: a GPU read fault in an import
// made a moment before - the shape of the Steam Deck's two GPU read faults of
// 2026-10-31, each about a second after an area's first frame, when ~30
// bindings a second widen the windows (not proven to be this: neither run
// logged its imports' addresses; BBHOST_VA_TRACE=1 does).
//
// So a window whose entries name an older import keeps them until the queue
// sees the new one - every submission made before it was made has run - and
// only then points them at it; the older import (the same pages: a merge
// covers every import it replaces) lives until then. A window whose entries
// named nothing (the sink page) points at the new import at once: no walk in
// the queue was reading those pages, and the walk the import was made for -
// the LS pass imports its windows before its dispatch - needs them now.
//
// A window's pages all point into the import its table names, or at the sink
// where that import does not reach. So a window is kept on the older import
// only where that one covers all the new import holds of it; one that covers
// less (a span that grew since) has pages new to the GPU that a binding or a
// walk asked for now - there the caller waits until the queue sees the new
// import, which it then names at once (rare: span edges).
#include <cstdint>

namespace import_table {

// Every submission made before import `c` (its made_serial: the serial of the
// recording open when it was made) has run, by the count of submissions run
// from the start (`completed`: the serial after the last one run).
template <class C>
inline bool seen(const C& c, std::uint64_t completed) {
    return completed >= c.made_serial;
}

// What a window's entries do when import `c` comes to cover it: point at it
// now; keep naming `table` (the older import, which `covers` all c holds of
// the window) until the queue sees `c`; or, where `table` covers less, point
// at it once the caller has waited for the queue to see it.
enum class Entry { kWrite, kKeep, kWait };
template <class C>
inline Entry on_cover(const C* table, const C& c, std::uint64_t completed, bool defer, bool covers) {
    if (!table || table == &c) return Entry::kWrite;  // the sink, or already it
    if (!defer || seen(c, completed)) return Entry::kWrite;
    return covers ? Entry::kKeep : Entry::kWait;
}

// Whether either view of the `n` windows names import `c`.
template <class C>
inline bool named(C* const* window, C* const* table, std::uint32_t n, const C* c) {
    for (std::uint32_t w = 0; w < n; ++w) {
        if (window[w] == c || table[w] == c) return true;
    }
    return false;
}

// The windows whose entries still name an older import than their window's.
template <class C>
inline std::uint32_t lagging(C* const* window, C* const* table, std::uint32_t n) {
    std::uint32_t k = 0;
    for (std::uint32_t w = 0; w < n; ++w) k += table[w] && window[w] && table[w] != window[w];
    return k;
}

// What install and publish did, for the counts and the trace.
struct Done {
    std::uint32_t written = 0;  // windows whose entries now name the import
    std::uint32_t fresh = 0;    // ... of them, ones that named nothing before (the sink)
    std::uint32_t kept = 0;     // windows whose entries keep naming the older import
    std::uint32_t retired = 0;  // imports neither view names any more
    bool waited = false;        // the caller waited for the queue to see the import (an older one covering less)
};

// Import `c` covers windows [a, b] of `n`: both views of each, the entries
// written (write(w): all of c's pages in window w), and every import neither
// view names any more retired (retire(import)), each once. covers(w, t):
// whether the older import `t` covers all c holds of window w. wait(): the
// queue made to see c - every submission before it run - returning the count
// of submissions run then.
template <class C, class Write, class Covers, class Wait, class Retire>
inline Done install(C** window, C** table, std::uint32_t n, std::uint32_t a, std::uint32_t b, C* c, std::uint64_t completed,
                    bool defer, Write write, Covers covers, Wait wait, Retire retire) {
    Done done;
    if (a > b || b >= n) return done;
    C* gone[2 * 64];
    std::uint32_t ngone = 0;
    const auto note = [&](C* o) {
        if (!o || o == c) return;
        for (std::uint32_t k = 0; k < ngone; ++k)
            if (gone[k] == o) return;
        if (ngone < sizeof(gone) / sizeof(gone[0])) gone[ngone++] = o;
    };
    for (std::uint32_t w = a; w <= b; ++w) {
        note(window[w]);
        window[w] = c;
        C* const t = table[w];
        Entry e = on_cover(t, *c, completed, defer, !t || t == c || covers(w, t));
        if (e == Entry::kWait) {
            completed = wait();
            done.waited = true;
            e = Entry::kWrite;
        }
        if (e == Entry::kWrite) {
            if (!t) ++done.fresh;
            note(t);
            table[w] = c;
            write(w);
            ++done.written;
        } else {
            ++done.kept;
        }
    }
    for (std::uint32_t k = 0; k < ngone; ++k) {
        if (named(window, table, n, gone[k])) continue;
        retire(gone[k]);
        ++done.retired;
    }
    return done;
}

// Every window whose entries lag its window's import, where the queue sees
// that import now: its entries point at it (write(w, import)), and the older
// imports neither view names any more retire, each once.
template <class C, class Write, class Retire>
inline Done publish(C** window, C** table, std::uint32_t n, std::uint64_t completed, Write write, Retire retire) {
    Done done;
    C* gone[64];
    std::uint32_t ngone = 0;
    for (std::uint32_t w = 0; w < n; ++w) {
        C* const t = table[w];
        C* const c = window[w];
        if (!t || !c || t == c || !seen(*c, completed)) continue;
        table[w] = c;
        write(w, c);
        ++done.written;
        bool listed = false;
        for (std::uint32_t k = 0; k < ngone; ++k) listed |= gone[k] == t;
        if (!listed && ngone < sizeof(gone) / sizeof(gone[0])) gone[ngone++] = t;
    }
    for (std::uint32_t k = 0; k < ngone; ++k) {
        if (named(window, table, n, gone[k])) continue;
        retire(gone[k]);
        ++done.retired;
    }
    return done;
}

}  // namespace import_table
