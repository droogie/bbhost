// GX's upload heap moved into video memory the CPU writes, in pieces
// (BBHOST_GXVK_ARENA=2: host/gxvk.h, gpu.cpp arena_place_locked): the heap's
// layout as pieces, the room the move needs, and the move itself over the
// operations its caller gives it - Vulkan's and the kernel's in gpu.cpp, a
// model's in tests/arena_move_test.cpp.
//
// NVIDIA's Linux driver (610, on an RTX 4070 without resizable BAR) places a
// mapping of that memory up to 124 MiB at a time and fails from 128, and the
// heap is 160 MiB. So the heap is mapped as pieces - each a mapping of the
// kernel's (hle_kernel_map_host_pieces), each moved as an allocation of its
// own placed over its part of the heap, each its own import - and a range
// across two pieces lies in no one buffer: its bindings take today's path.
//
// The move, with GX's writes to the heap waiting in the fault handler while
// it runs (gxvk::arena_move_fault):
//   1. every piece's memory allocated (none: nothing changed);
//   2. the heap made read-only, then its bytes saved;
//   3. each piece filled from the saved bytes and placed over its part;
//   4. all placed: each piece adopted as its part's import.
// A step that fails undoes the ones before. A placed piece is writable, and
// GX may have written it since: each is made read-only again, its bytes read
// back into the copy, unmapped (its range kept reserved where the driver can)
// and its part mapped again as host memory holding them; the pieces not yet
// placed are made writable again, and every allocation is given back - the
// heap as it was, in host memory, with every write GX made.
#pragma once

#include "host/arena_memory.h"

#include <cstdint>
#include <cstdlib>

namespace arena_move {

// tests/arena_move_test.cpp builds this header with mutants: small wrong turns
// it switches on one at a time and must catch.
#ifdef ARENA_MOVE_MUTANTS
extern int g_mutant;
#define ARENA_MOVE_MUTANT(n) (::arena_move::g_mutant == (n))
#else
#define ARENA_MOVE_MUTANT(n) false
#endif

constexpr std::uint64_t kPieceDefault = 32ull << 20;
constexpr std::uint64_t kPieceMax = 124ull << 20;  // the most NVIDIA's driver placed at a time
constexpr std::uint64_t kPieceUnit = 64ull << 10;  // a GPU page: every piece starts on one
// What the heap leaves of the CPU-writable video memory for the rest of the
// system, by the driver's count of what is in use.
constexpr std::uint64_t kKeep = 16ull << 20;

// The heap [lo, lo + bytes) as pieces of `piece` bytes, the last one shorter
// when they do not divide it.
struct Pieces {
    std::uint64_t lo = 0, bytes = 0, piece = 0;
    std::uint64_t count() const { return piece ? (bytes + piece - 1) / piece : 0; }
    std::uint64_t start(std::uint64_t k) const { return lo + k * piece; }
    std::uint64_t len(std::uint64_t k) const {
        const std::uint64_t off = k * piece;
        return off >= bytes ? 0 : bytes - off < piece ? bytes - off : piece;
    }
    // Whether [va, va + n) lies in the heap; *k, the piece holding va.
    bool holds(std::uint64_t va, std::uint64_t* k) const {
        if (!piece || va < lo || va - lo >= bytes) return false;
        *k = (va - lo) / piece;
        return true;
    }
    // Whether [va, va + n) lies within one piece (n 0 counts as 1 byte); *k,
    // that piece. False across two pieces, or past the heap's end.
    bool within(std::uint64_t va, std::uint64_t n, std::uint64_t* k) const {
        std::uint64_t i = 0;
        if (!holds(va, &i)) return false;
        const std::uint64_t in = va - start(i), room = len(i) - in;
        if (ARENA_MOVE_MUTANT(6) ? n >= room : n > room) return false;
        *k = i;
        return true;
    }
};

// BBHOST_GXVK_ARENA_PIECE_MB: the pieces' size in MiB (null or empty: the
// default), from 1 to kPieceMax's; anything else, the default.
inline std::uint64_t piece_bytes(const char* env) {
    if (!env || !*env) return kPieceDefault;
    const unsigned long long mib = std::strtoull(env, nullptr, 10);
    if (mib < 1 || (mib << 20) > kPieceMax) return kPieceDefault;
    return mib << 20;
}

// Room for `need` bytes in video memory the CPU writes, by what the driver
// says is in use: the heap's size less its use must hold them and kKeep more.
// Not the budget less the use, the rings' rule (host/ring_budget.h): NVIDIA's
// budget for that window falls with each allocation - it reads as what is
// left - so by it the 246 MiB window of an RTX 4070 with 66 MiB in use
// (bbhost's params ring and the driver's own) had 113 MiB of room, where 160
// MiB more allocated and placed beside them. Without the budget extension,
// only a heap of 1 GiB or more, holding them four times over, as the rings.
inline bool bar_room(const arena_memory::Heap& h, std::uint64_t need) {
    if (!h.have_budget) return h.size >= (1ull << 30) && need <= h.size / 4;
    if (ARENA_MOVE_MUTANT(7)) return h.budget > h.used && h.budget - h.used >= need + kKeep;
    return h.used < h.size && h.size - h.used >= need + kKeep;
}

struct Result {
    bool ok = false;
    const char* why = "";
    std::uint64_t piece = ~0ull;  // the piece the failing step was at, when it was one
    std::uint64_t placed = 0;     // pieces placed when it ended (all of them, or 0 after an undo)
};

// The move of the heap p describes, over ops:
//   bool allocate(k, len)   piece k's memory (and its buffer)
//   void release(n)         give back pieces [0, n)'s memory
//   void moving(on)         GX's writes to the heap wait while on
//   bool protect(va, len, writable)
//   bool save(va, len)      keep a copy of the bytes
//   bool fill(k, off, len)  piece k's memory from the copy's [off, off + len)
//   bool place(k, va)       piece k's memory placed over [va, va + len)
//   void read_back(k, off, len)  the placed piece's bytes into the copy's [off, off + len)
//   void unplace(k, va)     undo place: the range reserved, or free
//   bool restore(va, len, off)  host memory again over the range, holding the copy's [off, off + len)
//   void adopt(k, va)       piece k as its part's import
//   void drop_saved()
template <class Ops>
Result move(Ops& ops, const Pieces& p) {
    Result r;
    const std::uint64_t n = p.count();
    if (!n) {
        r.why = "the heap has no pieces";
        return r;
    }
    const auto len_of = [&](std::uint64_t k) { return ARENA_MOVE_MUTANT(5) ? p.piece : p.len(k); };
    for (std::uint64_t k = 0; k < n; ++k) {
        if (!ops.allocate(k, len_of(k))) {
            ops.release(k);
            r.why = "a piece's memory could not be allocated";
            r.piece = k;
            return r;
        }
    }
    ops.moving(true);
    // The copy is taken once nothing can write the heap: a write before it
    // would be in the copy, and one after it would have waited.
    bool saved = false;
    if (ARENA_MOVE_MUTANT(1)) saved = ops.save(p.lo, p.bytes);
    if (!ops.protect(p.lo, p.bytes, false)) {
        if (saved) ops.drop_saved();
        ops.moving(false);
        ops.release(n);
        r.why = "the heap could not be made read-only";
        return r;
    }
    if (!saved && !ops.save(p.lo, p.bytes)) {
        ops.protect(p.lo, p.bytes, true);
        ops.moving(false);
        ops.release(n);
        r.why = "the heap's bytes could not be saved";
        return r;
    }
    std::uint64_t placed = 0;
    const char* why = nullptr;
    for (std::uint64_t k = 0; k < n && !why; ++k) {
        if (!ops.fill(k, p.start(k) - p.lo, len_of(k))) {
            why = "a piece could not be filled";
        } else if (!ops.place(k, p.start(k))) {
            why = "a piece could not be placed over the heap";
        } else {
            ++placed;
            if (ARENA_MOVE_MUTANT(4)) ops.adopt(k, p.start(k));
        }
        if (why) r.piece = k;
    }
    if (!why) {
        if (!ARENA_MOVE_MUTANT(4))
            for (std::uint64_t k = 0; k < n; ++k) ops.adopt(k, p.start(k));
        ops.drop_saved();
        ops.moving(false);
        r.ok = true;
        r.placed = n;
        return r;
    }
    // Undo: each placed piece read-only again (a write to it waits from here),
    // its bytes - with whatever GX wrote to it since it was placed - read back
    // into the copy, and its part host memory again holding them; the pieces
    // not placed writable again.
    for (std::uint64_t k = 0; k < placed; ++k) {
        ops.protect(p.start(k), p.len(k), false);
        if (!ARENA_MOVE_MUTANT(2)) ops.read_back(k, p.start(k) - p.lo, p.len(k));
        ops.unplace(k, p.start(k));
        if (!ops.restore(p.start(k), p.len(k), p.start(k) - p.lo)) why = "a placed piece could not be undone";
    }
    for (std::uint64_t k = placed; k < n; ++k)
        if (!ARENA_MOVE_MUTANT(3)) ops.protect(p.start(k), p.len(k), true);
    ops.drop_saved();
    ops.moving(false);
    ops.release(n);
    r.why = why;
    return r;
}

}  // namespace arena_move
