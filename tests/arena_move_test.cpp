// GX's upload heap moved in pieces (host/arena_move.h): the pieces' layout,
// the room rule, and the move over a model of the heap's memory - pages with
// their protection and what backs them (the original host memory, a placed
// piece, a range reserved, host memory mapped again) - with GX writing the
// heap all the while: a write lands at once where its page is writable, and
// waits, as in the fault handler, while the move runs and its page is not.
// Every step of the move fails in turn, at every piece. After a move the heap
// must read what GX wrote, writable, through its pieces each adopted as its
// part's import; after a failed one, through host memory again, with nothing
// placed, adopted, allocated or saved, and no write left waiting.
// Then seven deliberate mistakes, each of which must be caught: the copy
// taken before the heap is read-only, the undo restoring a placed part from
// the copy without reading the piece back (losing what GX wrote to it after it
// was placed), the undo leaving the unplaced pieces read-only, pieces
// adopted as they are placed, the last piece taken at the full size, a range
// ending at its piece's end taken as crossing, and the room read as the
// budget less the use.
#define ARENA_MOVE_MUTANTS
#include "host/arena_move.h"

#include <cstdio>
#include <cstring>
#include <optional>
#include <random>
#include <vector>

int arena_move::g_mutant = 0;

namespace {

constexpr std::uint64_t kPage = 4096;

enum Back : int { kHost, kPlaced, kReserved, kRestored };

struct Model {
    // The heap.
    arena_move::Pieces p;
    std::vector<std::uint8_t> truth, host;  // what GX wrote; host memory's bytes
    std::vector<int> back, owner;           // per page: what backs it, the placed piece
    std::vector<bool> writable;
    struct Piece {
        bool allocated = false, placed = false;
        std::uint64_t len = 0;
        std::vector<std::uint8_t> bytes;
    };
    std::vector<Piece> pieces;
    std::optional<std::vector<std::uint8_t>> saved;
    std::vector<bool> adopted;
    bool is_moving = false;
    struct Write {
        std::uint64_t off;
        std::uint8_t v;
    };
    std::vector<Write> waiting;
    std::mt19937_64 rng;
    // Failures to inject, by step: a piece index, or 0/1 for the heap-wide steps.
    std::uint64_t fail_alloc = ~0ull, fail_fill = ~0ull, fail_place = ~0ull;
    bool fail_protect = false, fail_save = false;
    int errors = 0;
    const char* first_error = nullptr;

    void error(const char* what) {
        if (!errors++) first_error = what;
    }
    std::uint64_t pages() const { return (p.bytes + kPage - 1) / kPage; }
    std::uint64_t piece_of(std::uint64_t off) const { return off / p.piece; }

    Model(std::uint64_t bytes, std::uint64_t piece, std::uint64_t seed) : rng(seed) {
        p = {0x7000000000ull, bytes, piece};
        truth.resize(bytes);
        for (auto& b : truth) b = static_cast<std::uint8_t>(rng());
        host = truth;
        back.assign(pages(), kHost);
        owner.assign(pages(), -1);
        writable.assign(pages(), true);
        pieces.resize(p.count());
        adopted.assign(p.count(), false);
    }
    // The byte at heap offset `off` as the CPU reads it now; false where nothing is mapped.
    bool read(std::uint64_t off, std::uint8_t* out) const {
        const std::uint64_t pg = off / kPage;
        if (back[pg] == kHost || back[pg] == kRestored) {
            *out = host[off];
            return true;
        }
        if (back[pg] == kPlaced) {
            const Piece& pc = pieces[owner[pg]];
            const std::uint64_t in = off - (p.start(owner[pg]) - p.lo);
            if (in >= pc.bytes.size()) return false;
            *out = pc.bytes[in];
            return true;
        }
        return false;
    }
    void land(std::uint64_t off, std::uint8_t v) {
        const std::uint64_t pg = off / kPage;
        truth[off] = v;
        if (back[pg] == kHost || back[pg] == kRestored) {
            host[off] = v;
        } else if (back[pg] == kPlaced) {
            Piece& pc = pieces[owner[pg]];
            const std::uint64_t in = off - (p.start(owner[pg]) - p.lo);
            if (in < pc.bytes.size()) pc.bytes[in] = v;
            else error("a write landed past its piece's memory");
        } else {
            error("a write landed on no memory");
        }
    }
    // GX writes a byte of the heap: at once where its page is writable, else
    // it waits while the move runs (and faults for good once it does not).
    void gx_writes(int n = 2) {
        for (int i = 0; i < n; ++i) {
            const std::uint64_t off = rng() % p.bytes;
            const std::uint8_t v = static_cast<std::uint8_t>(rng());
            const std::uint64_t pg = off / kPage;
            if (writable[pg] && back[pg] != kReserved) land(off, v);
            else if (is_moving) waiting.push_back({off, v});
            else error("a write faulted with no move running");
        }
    }
    void set_pages(std::uint64_t va, std::uint64_t len, int b, int own, bool w) {
        if (va < p.lo || va + len > p.lo + p.bytes) {
            error("a step reached past the heap");
            if (va < p.lo) return;
            len = va - p.lo >= p.bytes ? 0 : p.lo + p.bytes - va;
        }
        for (std::uint64_t pg = (va - p.lo) / kPage; pg < (va - p.lo + len + kPage - 1) / kPage; ++pg) {
            if (b >= 0) back[pg] = b;
            if (own >= -1) owner[pg] = own;
            writable[pg] = w;
        }
    }

    // The operations arena_move::move asks for.
    bool allocate(std::uint64_t k, std::uint64_t len) {
        gx_writes();
        if (k == fail_alloc) return false;
        if (p.start(k) + len > p.lo + p.bytes) error("a piece allocated past the heap");
        pieces[k].allocated = true;
        pieces[k].len = len;
        pieces[k].bytes.assign(len, 0);
        return true;
    }
    void release(std::uint64_t n) {
        gx_writes();
        for (std::uint64_t k = 0; k < n && k < pieces.size(); ++k) {
            if (pieces[k].placed) error("a placed piece given back");
            pieces[k].allocated = false;
            pieces[k].bytes.clear();
        }
    }
    void moving(bool on) {
        gx_writes();
        is_moving = on;
        if (on) return;
        std::vector<Write> w;
        w.swap(waiting);
        for (const Write& x : w) {
            const std::uint64_t pg = x.off / kPage;
            if (!writable[pg] || back[pg] == kReserved) error("a write waiting for the move faults again after it");
            else land(x.off, x.v);
        }
    }
    bool protect(std::uint64_t va, std::uint64_t len, bool w) {
        gx_writes();
        if (fail_protect && !w) return false;
        set_pages(va, len, -1, -2, w);
        gx_writes();
        return true;
    }
    bool save(std::uint64_t va, std::uint64_t len) {
        gx_writes();
        if (fail_save) return false;
        if (va != p.lo || len != p.bytes) error("the copy is not the whole heap");
        std::vector<std::uint8_t> c(p.bytes);
        for (std::uint64_t off = 0; off < p.bytes; ++off)
            if (!read(off, &c[off])) error("the copy read unmapped memory");
        saved = std::move(c);
        gx_writes();
        return true;
    }
    bool fill(std::uint64_t k, std::uint64_t off, std::uint64_t len) {
        gx_writes();
        if (k == fail_fill) return false;
        if (!saved) {
            error("a piece filled with no copy");
            return false;
        }
        if (off + len > saved->size()) {
            error("a piece filled past the copy");
            len = off >= saved->size() ? 0 : saved->size() - off;
        }
        Piece& pc = pieces[k];
        if (!pc.allocated) error("an unallocated piece filled");
        pc.bytes.assign(saved->begin() + static_cast<std::ptrdiff_t>(off), saved->begin() + static_cast<std::ptrdiff_t>(off + len));
        pc.bytes.resize(pc.len, 0);
        return true;
    }
    bool place(std::uint64_t k, std::uint64_t va) {
        gx_writes();
        if (k == fail_place) return false;
        if (!pieces[k].allocated) error("an unallocated piece placed");
        set_pages(va, pieces[k].len, kPlaced, static_cast<int>(k), true);
        pieces[k].placed = true;
        gx_writes();
        return true;
    }
    void read_back(std::uint64_t k, std::uint64_t off, std::uint64_t len) {
        gx_writes();
        if (!saved || !pieces[k].placed) {
            error("a read-back with no copy or no placed piece");
            return;
        }
        for (std::uint64_t i = 0; i < len && off + i < saved->size() && i < pieces[k].bytes.size(); ++i) (*saved)[off + i] = pieces[k].bytes[i];
    }
    void unplace(std::uint64_t k, std::uint64_t va) {
        gx_writes();
        if (!pieces[k].placed) error("a piece not placed unplaced");
        set_pages(va, p.len(k), kReserved, -1, false);
        pieces[k].placed = false;
    }
    bool restore(std::uint64_t va, std::uint64_t len, std::uint64_t off) {
        gx_writes();
        set_pages(va, len, kRestored, -1, true);
        const std::uint64_t at = va - p.lo;
        for (std::uint64_t i = 0; i < len; ++i)
            host[at + i] = saved && off != ~0ull && off + i < saved->size() ? (*saved)[off + i] : 0;
        return true;
    }
    void adopt(std::uint64_t k, std::uint64_t va) {
        if (va != p.start(k)) error("a piece adopted at another address");
        adopted[k] = true;
    }
    void drop_saved() { saved.reset(); }

    // What must hold once the move returned.
    void check(const arena_move::Result& r) {
        if (is_moving) error("the move left GX's writes waiting");
        if (!waiting.empty()) error("writes still waiting");
        if (saved) error("the copy kept");
        for (std::uint64_t off = 0; off < p.bytes; ++off) {
            std::uint8_t b = 0;
            if (!read(off, &b)) {
                error("part of the heap unmapped");
                break;
            }
            if (b != truth[off]) {
                error("the heap does not read what GX wrote");
                break;
            }
        }
        for (std::uint64_t pg = 0; pg < pages(); ++pg)
            if (!writable[pg]) {
                error("part of the heap left read-only");
                break;
            }
        for (std::uint64_t k = 0; k < pieces.size(); ++k) {
            if (r.ok) {
                if (!pieces[k].placed || !adopted[k]) error("a piece not placed or not adopted after the move");
                for (std::uint64_t pg = (p.start(k) - p.lo) / kPage; pg < (p.start(k) - p.lo + p.len(k)) / kPage; ++pg)
                    if (back[pg] != kPlaced || owner[pg] != static_cast<int>(k)) {
                        error("a page not backed by its piece");
                        break;
                    }
            } else {
                if (pieces[k].placed) error("a piece still placed after a failed move");
                if (pieces[k].allocated) error("a piece's memory kept after a failed move");
                if (adopted[k]) error("a piece adopted after a failed move");
            }
        }
        if (!r.ok)
            for (std::uint64_t pg = 0; pg < pages(); ++pg)
                if (back[pg] != kHost && back[pg] != kRestored) {
                    error("a failed move left the heap off host memory");
                    break;
                }
    }
};

struct Tally {
    int runs = 0, failed = 0;
    const char* first = nullptr;
};

// Every failure point for a heap of `bytes` in pieces of `piece`.
Tally run_moves(std::uint64_t bytes, std::uint64_t piece, std::uint64_t seed) {
    Tally t;
    const arena_move::Pieces layout{0, bytes, piece};
    const std::uint64_t n = layout.count();
    std::vector<int> plans;  // 0 none, 1 protect, 2 save, then alloc/fill/place at each piece
    const int kinds = 3 + 3 * static_cast<int>(n);
    for (int plan = 0; plan < kinds; ++plan) {
        for (int rep = 0; rep < 3; ++rep) {
            Model m(bytes, piece, seed + 977ull * plan + rep);
            if (plan == 1) m.fail_protect = true;
            if (plan == 2) m.fail_save = true;
            if (plan >= 3) {
                const std::uint64_t k = static_cast<std::uint64_t>(plan - 3) / 3;
                const int which = (plan - 3) % 3;
                (which == 0 ? m.fail_alloc : which == 1 ? m.fail_fill : m.fail_place) = k;
            }
            const arena_move::Result r = arena_move::move(m, m.p);
            if (r.ok != (plan == 0)) m.error(plan == 0 ? "the move failed with nothing failing" : "the move succeeded through a failure");
            m.check(r);
            ++t.runs;
            if (m.errors) {
                ++t.failed;
                if (!t.first) t.first = m.first_error;
            }
        }
    }
    return t;
}

int check_layout() {
    int bad = 0;
    const auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            if (!bad) std::printf("  layout: %s\n", what);
            ++bad;
        }
    };
    const std::uint64_t MiB = 1ull << 20;
    const arena_move::Pieces p{0x100000000ull, 160 * MiB, 32 * MiB};
    std::uint64_t k = 99;
    expect(p.count() == 5 && p.len(4) == 32 * MiB && p.len(5) == 0, "160 MiB in 32 MiB pieces");
    expect(p.within(p.lo, 32 * MiB, &k) && k == 0, "a whole piece is within it");
    expect(p.within(p.lo + 32 * MiB - 16, 16, &k) && k == 0, "a range ending at its piece's end is within it");
    expect(!p.within(p.lo + 32 * MiB - 16, 17, &k), "a range one byte into the next piece crosses");
    expect(p.within(p.lo + 32 * MiB, 1, &k) && k == 1, "a range starting at a piece's start is in that piece");
    expect(p.within(p.lo + 64 * MiB, 0, &k) && k == 2, "an empty range at a piece's start is in it");
    expect(!p.within(p.lo + 160 * MiB, 1, &k), "the heap's end is outside");
    expect(!p.within(p.lo - 1, 2, &k), "below the heap is outside");
    expect(!p.within(p.lo + 160 * MiB - 8, 16, &k), "a range past the heap's end is not within a piece");
    const arena_move::Pieces q{0x100000000ull, 150 * MiB, 40 * MiB};
    expect(q.count() == 4 && q.len(3) == 30 * MiB, "an uneven last piece");
    expect(q.within(q.lo + 120 * MiB, 30 * MiB, &k) && k == 3, "the last piece whole");
    expect(!q.within(q.lo + 120 * MiB, 30 * MiB + 1, &k), "past the last piece");
    expect(arena_move::piece_bytes(nullptr) == 32 * MiB && arena_move::piece_bytes("") == 32 * MiB, "the default piece");
    expect(arena_move::piece_bytes("40") == 40 * MiB && arena_move::piece_bytes("124") == 124 * MiB, "pieces asked for");
    expect(arena_move::piece_bytes("0") == 32 * MiB && arena_move::piece_bytes("125") == 32 * MiB && arena_move::piece_bytes("x") == 32 * MiB,
           "pieces out of range");
    return bad;
}

int check_room() {
    int bad = 0;
    const auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            if (!bad) std::printf("  room: %s\n", what);
            ++bad;
        }
    };
    const std::uint64_t MiB = 1ull << 20;
    arena_memory::Heap bar;
    bar.size = 246 * MiB;
    bar.have_budget = true;
    // The RTX 4070's window at the move: 66 MiB in use (the driver's 34, the params ring's 32), its budget read as what is left.
    bar.used = 66 * MiB;
    bar.budget = 179 * MiB;
    expect(arena_move::bar_room(bar, 160 * MiB), "the heap beside the params ring in a 246 MiB window");
    bar.used = 80 * MiB;
    bar.budget = 165 * MiB;
    expect(!arena_move::bar_room(bar, 160 * MiB), "too little left for the heap and the keep");
    bar.used = 300 * MiB;
    expect(!arena_move::bar_room(bar, 1 * MiB), "a heap used past its size");
    arena_memory::Heap nob;
    nob.size = 246 * MiB;
    expect(!arena_move::bar_room(nob, 160 * MiB), "no budget: a 246 MiB window");
    nob.size = 8192 * MiB;
    expect(arena_move::bar_room(nob, 160 * MiB), "no budget: resizable BAR");
    return bad;
}

}  // namespace

int main() {
    int failed = 0;
    const std::uint64_t MiB = 1ull << 20;
    // The model's heaps: five whole pieces, and four with a short last one.
    struct Shape {
        std::uint64_t bytes, piece;
    };
    const Shape shapes[] = {{5 * 64 * 1024, 64 * 1024}, {4 * 64 * 1024 + 24 * 1024, 64 * 1024}};
    for (const Shape& s : shapes) {
        const Tally t = run_moves(s.bytes, s.piece, 1);
        std::printf("moves of a %llu KiB heap in %llu KiB pieces, every step failing at every piece: %d runs, %d wrong%s%s\n",
                    static_cast<unsigned long long>(s.bytes >> 10), static_cast<unsigned long long>(s.piece >> 10), t.runs, t.failed,
                    t.first ? " - first: " : "", t.first ? t.first : "");
        if (t.failed) ++failed;
    }
    const int layout = check_layout(), room = check_room();
    std::printf("layout checks %s, room checks %s\n", layout ? "FAILED" : "ok", room ? "FAILED" : "ok");
    if (layout || room) ++failed;
    (void)MiB;
    // The mistakes: each must be caught somewhere above.
    static const char* const kMutant[] = {
        "", "the copy taken before the heap is read-only", "the undo restores a placed part without reading it back",
        "the undo leaves the unplaced pieces read-only", "pieces adopted as they are placed", "the last piece taken at the full size",
        "a range ending at its piece's end taken as crossing", "the room read as the budget less the use"};
    for (int mu = 1; mu <= 7; ++mu) {
        arena_move::g_mutant = mu;
        int caught = 0;
        for (const Shape& s : shapes) caught += run_moves(s.bytes, s.piece, 7).failed;
        caught += check_layout() + check_room();
        arena_move::g_mutant = 0;
        std::printf("mistake %d (%s): %s\n", mu, kMutant[mu], caught ? "caught" : "NOT CAUGHT");
        if (!caught) ++failed;
    }
    std::printf(failed ? "arena_move_test: %d failed\n" : "arena_move_test: ok\n", failed);
    return failed ? 1 : 0;
}
