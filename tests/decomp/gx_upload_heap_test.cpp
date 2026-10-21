// GX's upload heap - the block pool every renamed constant buffer, Map
// rename and draw table comes from - run as the game's own code against a
// model (tests/decomp/eboot_kit.h). It is what the GX arena (host/gxvk.h)
// rests on: the arena binds a renamed copy where GX wrote it, in place, so a
// range must never be handed out again while work that reads it may still
// run. GX's rule for that: a range goes back with a release node whose state
// an end-of-pipe label sets to 4 once the GPU has passed it (0x2aaa620), and
// only the reclaim (0x2aaa860) returns a released range to the free list,
// and only when its state is 4.
//
// The pool is the game's: its init (0x2aa9b50) over a region, its two
// hand-outs (0x2aa9d50, 16-byte aligned; 0x2aa9fa0, a power of two), its
// release and its reclaim, with the node allocator they call. The label
// writer (gnm_write_end_of_pipe_event) is a recorder; a label lands when the
// test writes the value it was given (4) where it was told, in the order the
// GPU would (release order) or out of it. Over generated sequences of
// hand-outs of every size and alignment, releases, landings and reclaims,
// the model checks that every range handed out lies in the region, at the
// alignment asked, holds the bytes asked, and overlaps no range handed out
// before that the reclaim has not freed - none released and not yet landed
// above all; that the reclaim frees exactly the landed releases (the pool's
// free bytes and count of ranges out against the model's); and that once
// everything has landed and been reclaimed the region is whole again (one
// hand-out of all of it, at its start). A control run lands one label the
// model is not told of, and must be caught. Skips without the 1.09 eboot.
#include "eboot_kit.h"

#include "decomp/guest.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <vector>

using namespace decomp;

namespace {

constexpr u64 kPoolInit = 0x2aa9b50;        // (pool, allocator, region, bytes)
constexpr u64 kHandOut = 0x2aa9d50;         // (pool, bytes): 16-byte aligned, under the pool's lock
constexpr u64 kHandOutAligned = 0x2aa9fa0;  // (pool, bytes, alignment): a power of two
constexpr u64 kRelease = 0x2aaa620;         // (pool, command buffer holder, node): queued, its label written
constexpr u64 kReclaim = 0x2aaa860;         // (pool, context holder): the landed releases freed
constexpr u64 kEndOfPipe = 0x14778f0;       // gnm_write_end_of_pipe_event

using InitFn = GUEST_ABI u64 (*)(u64, u64, u64, u64);
using HandOutFn = GUEST_ABI u64 (*)(u64, u64);
using HandOutAlignedFn = GUEST_ABI u64 (*)(u64, u64, u64);
using ReleaseFn = GUEST_ABI u64 (*)(u64, u64, u64);
using ReclaimFn = GUEST_ABI u64 (*)(u64, u64);

eboot_kit::Rng g_rng(0x51ab);
int g_failed = 0;
bool g_quiet = false;
void fail(long step, const char* why) {
    if (++g_failed <= 30 && !g_quiet) std::printf("FAIL: step %ld: %s\n", step, why);
}

// The label writer's calls: where to write and what.
struct Label {
    u64 at;
    u32 value;
};
std::vector<Label> g_labels;
GUEST_ABI u64 fake_end_of_pipe(u64 cb, u64 event, u64 a2, u64 at, u64 size, u64 value, u64 a6, u64 a7) {
    (void)cb, (void)event, (void)a2, (void)size, (void)a6, (void)a7;
    g_labels.push_back({at, static_cast<u32>(value)});
    return 0;
}

// The node allocator's backing: allocate (vtable +0x58) aligned, free (+0x70).
std::map<u64, std::size_t> g_blocks;
GUEST_ABI u64 fake_alloc(u64 self, u64 bytes, u64 align, u64 extra) {
    (void)self, (void)extra;
    const std::size_t a = align < 16 ? 16 : static_cast<std::size_t>(align);
    void* p = std::aligned_alloc(a, (static_cast<std::size_t>(bytes) + a - 1) / a * a);
    if (p) std::memset(p, 0xcd, static_cast<std::size_t>(bytes));
    g_blocks[reinterpret_cast<u64>(p)] = static_cast<std::size_t>(bytes);
    return reinterpret_cast<u64>(p);
}
GUEST_ABI void fake_free(u64 self, u64 p) {
    (void)self;
    if (g_blocks.erase(p)) std::free(reinterpret_cast<void*>(static_cast<std::uintptr_t>(p)));
}
// The pool's lock (+0xd8): lock (+0x18) and unlock (+0x28).
GUEST_ABI u64 fake_lock(u64, u64) { return 0; }
GUEST_ABI u64 fake_unlock(u64) { return 0; }

alignas(16) u64 g_alloc_vtable[32], g_lock_vtable[16];
alignas(16) u64 g_allocator[2];

// One hand-out the reclaim has not freed: its range, released or not, landed or not.
struct Out {
    u64 start, end;
    bool released = false, landed = false;
};

struct Pool {
    alignas(64) u8 bytes[0x100] = {};
    u64 at() { return reinterpret_cast<u64>(bytes); }
};

u64 free_bytes(Pool& p) { return load<u64>(p.at() + 0xd0); }
u64 ranges_out(Pool& p) { return load<u64>(p.at() + 0xc8); }

// A hand-out's range: its node's start up to the next node's (the address list).
void range_of(u64 node, u64& start, u64& end) {
    start = load<u64>(node + 8);
    end = load<u64>(load<u64>(node + 0x18) + 8);
}

// control: a label lands that the model is not told of (a reuse before the
// GPU is done, as the model sees it), once, which the checks must catch.
long run_sequence(u64 region, u64 region_bytes, long steps, bool in_order, bool control = false) {
    Pool pool;
    store<u64>(pool.at() + 0xd8, reinterpret_cast<u64>(g_lock_vtable));
    alignas(16) static u64 cb[2] = {0x7700, 0};
    alignas(64) static u8 ctx[0xb500];
    const u64 ctx_holder_value = reinterpret_cast<u64>(ctx);
    const u64 cb_holder = reinterpret_cast<u64>(cb), ctx_holder = reinterpret_cast<u64>(&ctx_holder_value);
    if (!eboot_kit::fn<InitFn>(kPoolInit)(pool.at(), reinterpret_cast<u64>(g_allocator), region, region_bytes)) {
        fail(-1, "the pool's init failed");
        return 0;
    }
    const u64 base = load<u64>(pool.at() + 0xb8), limit = load<u64>(pool.at() + 0xc0);
    if (base != ((region + 15) & ~15ull) || base < region || limit > region + region_bytes || free_bytes(pool) != limit - base)
        fail(-1, "the pool's bounds and free bytes after init");
    std::map<u64, Out> out;          // by node
    std::map<u64, u64> by_start;     // start -> node: what is out, for the overlap check
    std::deque<u64> pending;         // released, label not landed, in release order
    long handed = 0, refused = 0, reclaims = 0;
    const auto out_bytes = [&]() {
        u64 n = 0;
        for (const auto& [node, o] : out) n += o.end - o.start;
        return n;
    };
    for (long step = 0; step < steps; ++step) {
        const double r = g_rng.next() / 18446744073709551616.0;
        if (r < 0.42 && out.size() < 4000) {
            // A hand-out: mostly small, some large; 16-byte or a power of two.
            u64 bytes;
            const int k = g_rng.range(0, 99);
            if (k < 55) bytes = static_cast<u64>(g_rng.range(1, 256));
            else if (k < 90) bytes = static_cast<u64>(g_rng.range(257, 64 << 10));
            else bytes = static_cast<u64>(g_rng.range((64 << 10) + 1, 4 << 20));
            const bool aligned = g_rng.chance(0.35);
            const u64 align = aligned ? 1ull << g_rng.range(0, 16) : 16;
            const u64 node = aligned ? eboot_kit::fn<HandOutAlignedFn>(kHandOutAligned)(pool.at(), bytes, align)
                                     : eboot_kit::fn<HandOutFn>(kHandOut)(pool.at(), bytes);
            if (!node) {
                ++refused;
                continue;
            }
            ++handed;
            u64 start, end;
            range_of(node, start, end);
            const u64 want = (bytes + 15) & ~15ull, a = align < 16 ? 16 : align;
            if (start < base || end > limit || end < start) fail(step, "a range outside the region");
            if (start % a) fail(step, "a range not at the alignment asked");
            if (end - start < want) fail(step, "a range shorter than asked");
            if (load<u64>(node) != 2) fail(step, "a handed-out node not in state 2");
            // No range still out - in use, or released and its label not landed or not reclaimed - may overlap it.
            auto it = by_start.upper_bound(start);
            if (it != by_start.begin()) {
                const auto prev = std::prev(it);
                if (out[prev->second].end > start) fail(step, "a range handed out over one still out (before it)");
            }
            if (it != by_start.end() && it->first < end) fail(step, "a range handed out over one still out (after it)");
            if (out.count(node)) fail(step, "a node handed out twice");
            out[node] = {start, end};
            by_start[start] = node;
        } else if (r < 0.68) {
            // A release of a range out and not released yet.
            if (out.empty()) continue;
            auto it = out.begin();
            std::advance(it, g_rng.range(0, static_cast<int>(out.size()) - 1));
            if (it->second.released) continue;
            const std::size_t labels = g_labels.size();
            eboot_kit::fn<ReleaseFn>(kRelease)(pool.at(), cb_holder, it->first);
            if (g_labels.size() != labels + 1 || g_labels.back().at != it->first || g_labels.back().value != 4)
                fail(step, "a release without its label (4 at the node)");
            it->second.released = true;
            pending.push_back(it->first);
        } else if (r < 0.84) {
            // Labels land: the oldest few (the GPU's order), or any.
            const int n = g_rng.range(1, 8);
            for (int i = 0; i < n && !pending.empty(); ++i) {
                std::size_t k = 0;
                if (!in_order) k = static_cast<std::size_t>(g_rng.range(0, static_cast<int>(pending.size()) - 1));
                const u64 node = pending[k];
                pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(k));
                store<u32>(node, 4);  // the label's write
                out[node].landed = true;
            }
            if (control && !pending.empty()) {
                store<u32>(pending.front(), 4);  // landed behind the model's back
                control = false;
            }
        } else {
            // The reclaim: exactly the landed releases come back.
            eboot_kit::fn<ReclaimFn>(kReclaim)(pool.at(), ctx_holder);
            ++reclaims;
            for (auto it = out.begin(); it != out.end();) {
                if (it->second.landed) {
                    by_start.erase(it->second.start);
                    it = out.erase(it);
                } else {
                    ++it;
                }
            }
            if (ranges_out(pool) != out.size()) fail(step, "the reclaim's count of ranges out is not the model's");
            if (free_bytes(pool) != (limit - base) - out_bytes()) fail(step, "the reclaim's free bytes are not the model's");
        }
    }
    // Everything released, landed and reclaimed: the region is whole again.
    for (auto& [node, o] : out) {
        if (!o.released) {
            eboot_kit::fn<ReleaseFn>(kRelease)(pool.at(), cb_holder, node);
            o.released = true;
        }
        store<u32>(node, 4);
    }
    eboot_kit::fn<ReclaimFn>(kReclaim)(pool.at(), ctx_holder);
    if (ranges_out(pool) != 0 || free_bytes(pool) != limit - base) fail(steps, "the region not whole after every release landed");
    const u64 all = eboot_kit::fn<HandOutFn>(kHandOut)(pool.at(), limit - base);
    u64 s = 0, e = 0;
    if (all) range_of(all, s, e);
    if (!all || s != base || e != limit) fail(steps, "one hand-out of the whole region did not get it (its ranges not merged back)");
    if (!g_quiet)
        std::printf("  %s landings, %llu-byte region: %ld steps, %ld hand-outs (%ld refused: no room), %ld reclaims, %zu labels\n",
                    in_order ? "in-order" : "out-of-order", static_cast<ull>(region_bytes), steps, handed, refused, reclaims, g_labels.size());
    g_labels.clear();
    return handed;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_upload_heap_test");
    const long steps = argc > 1 ? std::atol(argv[1]) : 200000;
    eboot_kit::stub(kEndOfPipe, reinterpret_cast<const void*>(&fake_end_of_pipe));
    g_alloc_vtable[0x58 / 8] = reinterpret_cast<u64>(&fake_alloc);
    g_alloc_vtable[0x70 / 8] = reinterpret_cast<u64>(&fake_free);
    g_allocator[0] = reinterpret_cast<u64>(g_alloc_vtable);
    g_lock_vtable[0x18 / 8] = reinterpret_cast<u64>(&fake_lock);
    g_lock_vtable[0x28 / 8] = reinterpret_cast<u64>(&fake_unlock);
    // The region is only addresses to the pool (it never touches the bytes):
    // the upload heap's as the game makes it, and an unaligned one.
    long total = 0;
    total += run_sequence(0x143400040ull, 0xa000000ull, steps, true);
    total += run_sequence(0x143400040ull, 0xa000000ull, steps, false);
    total += run_sequence(0x200000007ull, 0x1000009ull, steps / 2, false);  // 16 MiB, its start and size not 16-byte multiples
    const int before = g_failed;
    g_quiet = true;
    run_sequence(0x143400040ull, 0xa000000ull, 20000, true, true);
    g_quiet = false;
    const bool caught = g_failed > before;
    g_failed = before;
    if (!caught) {
        ++g_failed;
        std::printf("FAIL: a label landed behind the model's back was not caught\n");
    }
    if (g_failed) {
        std::printf("gx_upload_heap_test: %d failures\n", g_failed);
        return 1;
    }
    std::printf("gx_upload_heap_test: ok - %ld hand-outs, none over a range still out; the control caught\n", total);
    return 0;
}
