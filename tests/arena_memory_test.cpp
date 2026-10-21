// The arena's memory choice (host/arena_memory.h) on the memory layouts of
// the machines it is meant for: an RTX 3070 Ti laptop with resizable BAR (its
// log's type 5, flags 0x7, in an 8017 MiB heap), a desktop NVIDIA card and an
// RX 580 without it (a 256 MiB window the rings already fill), an AMD card
// with it, a Steam Deck's shared memory, lavapipe's single heap, a driver
// without the budget extension, and the edge cases: no coherent memory, a
// forced kind the device lacks, memory types the buffer does not allow.
#include "host/arena_memory.h"

#include <cstdio>
#include <vector>

namespace {

using namespace arena_memory;

constexpr std::uint64_t MiB = 1ull << 20;
constexpr std::uint64_t kArena = 160 * MiB;

int g_failed = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        ++g_failed;
        std::printf("FAIL: %s\n", what);
    }
}

struct Device {
    std::vector<Type> types;
    std::vector<Heap> heaps;
    Choice pick(bool cpu_reads, Kind forced = Kind::kNone, std::uint32_t bits = ~0u, std::uint64_t bytes = kArena) const {
        return choose(types.data(), static_cast<std::uint32_t>(types.size()), heaps.data(), bits, bytes, cpu_reads, forced);
    }
};

Heap heap(std::uint64_t size, bool video, bool budget = false, std::uint64_t b = 0, std::uint64_t used = 0) {
    Heap h;
    h.size = size;
    h.flags = video ? kHeapDeviceLocal : 0;
    h.have_budget = budget;
    h.budget = b;
    h.used = used;
    return h;
}

// NVIDIA's usual order: host types, video memory, then the CPU-writable window.
Device nvidia(bool rebar, std::uint64_t window_used) {
    Device d;
    const std::uint64_t window = rebar ? 8017 * MiB : 256 * MiB;
    d.heaps = {heap(8017 * MiB, true, true, 7000 * MiB, 2000 * MiB), heap(16384 * MiB, false, true, 15000 * MiB, 3000 * MiB)};
    std::uint32_t window_heap = 0;
    if (!rebar) {
        d.heaps.push_back(heap(window, true, true, 218 * MiB, window_used));
        window_heap = 2;
    }
    d.types = {{0, 1},
               {kDeviceLocal, 0},
               {kHostVisible | kHostCoherent, 1},
               {kHostVisible | kHostCoherent | kHostCached, 1},
               {kDeviceLocal, 0},
               {kDeviceLocal | kHostVisible | kHostCoherent, window_heap}};
    return d;
}

}  // namespace

int main() {
    // The laptop: resizable BAR, room in the 8 GiB heap.
    {
        const Device d = nvidia(true, 0);
        const Choice in_place = d.pick(false), snapshots = d.pick(true);
        expect(in_place.kind == Kind::kBar && in_place.type == 5, "resizable BAR, nothing reads it on the CPU: video memory, type 5");
        expect(snapshots.kind == Kind::kCached && snapshots.type == 3, "resizable BAR, snapshots read it: cached host memory, type 3");
    }
    // A desktop card without resizable BAR: the 256 MiB window holds the rings' 160 MiB.
    {
        const Device d = nvidia(false, 187 * MiB);
        const Choice c = d.pick(false);
        expect(c.kind == Kind::kCached && c.type == 3, "no resizable BAR, the window full of rings: cached host memory");
        expect(!c.room.fits, "no resizable BAR: the budget says no room");
        // The rings are placed first; with them in host memory the window has the room.
        const Device empty = nvidia(false, 0);
        expect(empty.pick(false).kind == Kind::kBar, "no resizable BAR, an empty window: 160 MiB and its 32 MiB margin fit in 218");
        expect(d.pick(false, Kind::kBar).kind == Kind::kBar, "forced into the window: taken whatever the budget says");
    }
    // An RX 580 without Smart Access Memory (AMD's order: video, host, window, cached).
    {
        Device d;
        d.heaps = {heap(8192 * MiB, true, true, 7900 * MiB, 1000 * MiB), heap(16384 * MiB, false, true, 15000 * MiB, 3000 * MiB),
                   heap(256 * MiB, true, true, 224 * MiB, 160 * MiB)};
        d.types = {{kDeviceLocal, 0},
                   {kHostVisible | kHostCoherent, 1},
                   {kDeviceLocal | kHostVisible | kHostCoherent, 2},
                   {kHostVisible | kHostCoherent | kHostCached, 1}};
        const Choice c = d.pick(false);
        expect(c.kind == Kind::kCached && c.type == 3, "RX 580: cached host memory");
        expect(d.pick(false, Kind::kHost).type == 1, "RX 580, forced host: the write-combined host type, not the window");
    }
    // An AMD card with Smart Access Memory: the CPU-writable type is all of video memory.
    {
        Device d;
        d.heaps = {heap(16368 * MiB, true, true, 15500 * MiB, 3000 * MiB), heap(32768 * MiB, false, true, 30000 * MiB, 4000 * MiB)};
        d.types = {{kDeviceLocal, 0},
                   {kDeviceLocal | kHostVisible | kHostCoherent, 0},
                   {kHostVisible | kHostCoherent, 1},
                   {kHostVisible | kHostCoherent | kHostCached, 1}};
        expect(d.pick(false).kind == Kind::kBar && d.pick(false).type == 1, "AMD with SAM: video memory");
        expect(d.pick(true).kind == Kind::kCached, "AMD with SAM, snapshots: cached");
    }
    // A Steam Deck: a 1 GiB carve-out and the shared RAM, all of it the same memory.
    {
        Device d;
        d.heaps = {heap(1024 * MiB, true, true, 1000 * MiB, 700 * MiB), heap(11264 * MiB, false, true, 9000 * MiB, 5000 * MiB)};
        d.types = {{kDeviceLocal, 0},
                   {kDeviceLocal | kHostVisible | kHostCoherent, 0},
                   {kHostVisible | kHostCoherent, 1},
                   {kHostVisible | kHostCoherent | kHostCached, 1}};
        expect(d.pick(false).kind == Kind::kBar, "Deck, room in the carve-out: there");
        Device full = d;
        full.heaps[0].used = 950 * MiB;
        expect(full.pick(false).kind == Kind::kCached, "Deck, the carve-out full: cached shared RAM");
        expect(d.pick(true).kind == Kind::kCached, "Deck, snapshots: cached shared RAM");
    }
    // Lavapipe: one heap, one type with every flag.
    {
        Device d;
        d.heaps = {heap(64 * 1024 * MiB, true, true, 60 * 1024 * MiB, 20 * 1024 * MiB)};
        d.types = {{kDeviceLocal | kHostVisible | kHostCoherent | kHostCached, 0}};
        expect(d.pick(false).kind == Kind::kBar && d.pick(false).type == 0, "lavapipe, in place: its one type");
        expect(d.pick(true).kind == Kind::kCached && d.pick(true).type == 0, "lavapipe, snapshots: its one type, as cached");
    }
    // No budget extension: only a heap of 1 GiB or more takes it, and the arena no more than a quarter.
    {
        Device d = nvidia(true, 0);
        for (Heap& h : d.heaps) h.have_budget = false;
        expect(d.pick(false).kind == Kind::kBar, "no budget extension, resizable BAR: video memory");
        Device small = nvidia(false, 0);
        for (Heap& h : small.heaps) h.have_budget = false;
        expect(small.pick(false).kind == Kind::kCached, "no budget extension, a 256 MiB window: cached");
    }
    // No coherent host-visible memory at all: no arena.
    {
        Device d;
        d.heaps = {heap(8192 * MiB, true), heap(16384 * MiB, false)};
        d.types = {{kDeviceLocal, 0}, {kHostVisible, 1}, {kHostVisible | kHostCached, 1}};
        const Choice c = d.pick(false);
        expect(c.kind == Kind::kNone && c.type == UINT32_MAX, "no coherent type: none");
        expect(d.pick(false, Kind::kCached).kind == Kind::kNone, "no coherent type, forced cached: none");
    }
    // A forced kind the device lacks falls back down the order.
    {
        Device d = nvidia(false, 187 * MiB);
        d.types[3].flags = kHostVisible | kHostCoherent;  // no cached type
        const Choice c = d.pick(true, Kind::kCached);
        expect(c.kind == Kind::kHost && c.type == 2, "forced cached without one: host memory outside video memory");
    }
    // Memory types the buffer does not allow are passed over.
    {
        const Device d = nvidia(true, 0);
        expect(d.pick(false, Kind::kNone, ~(1u << 5)).kind == Kind::kCached, "the BAR type not allowed: cached");
        expect(d.pick(false, Kind::kNone, (1u << 5) | (1u << 2)).kind == Kind::kBar, "only BAR and host allowed: BAR");
        expect(d.pick(true, Kind::kNone, (1u << 5) | (1u << 2)).kind == Kind::kBar, "snapshots, no cached type allowed: BAR before plain host");
    }
    if (g_failed) {
        std::printf("arena_memory_test: %d failures\n", g_failed);
        return 1;
    }
    std::printf("arena_memory_test: ok\n");
    return 0;
}
