// Stage 4b.3c's index of live homes by address (host/gxvk_homes_index.h),
// without the game or a GPU: homes made, ended and made again over the same
// memory - small constant buffers packed in a few pages, as GX's heaps hold
// them, a page shared by several, ranges that run across pages - and every
// answer the command processor asks of it (meets, holding, straddles, and the
// filter's "no") checked against a plain list of the live ranges. The filter
// may say "maybe" where nothing is; it may never say "no" where a home is.
// The pool's threads' view of it (read_meets, the published lists) gives the
// same answers, single-threaded; beside a writer thread changing it in batches,
// readers find every home that stays, nothing where nothing ever was, one of
// the two homes a batch swaps (never neither: a batch's middle), and only homes
// the writer put where they are (no freed list read back); every list retired
// is freed once the readers are done (concurrent_test).
// The handle's packing is checked too (host/gxvk_homes.h), and the default's
// rule (default_off: on only with a card with room or 5,000 MiB of small video
// memory free and a frame rate above 60 - at any frame rate where the run
// commit is on by the platform's default; any number of threads;
// small_refused; homes_bound and run_commit_gives_way: Windows' run commit
// gives way wherever the homes are bound, by the rule or BBHOST_GXVK_HOMES=1).
#include "host/gxvk_homes.h"
#include "host/gxvk_homes_index.h"
#include "host/list_resolve.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

namespace {

struct Live {
    std::uint32_t index;
    std::uint64_t memory;
    std::uint32_t bytes;
};

std::atomic<int> g_failures{0};  // (the concurrent test's threads check too)
void check(bool ok, const char* what, std::uint64_t va, std::uint64_t bytes) {
    if (ok) return;
    if (++g_failures <= 12) std::printf("FAIL: %s at 0x%llx + %llu\n", what, static_cast<unsigned long long>(va), static_cast<unsigned long long>(bytes));
}

// The model: the live homes meeting [va, +bytes), and whether one holds it whole.
void model(const std::vector<Live>& live, std::uint64_t va, std::uint64_t bytes, bool* meets, bool* holding, bool* straddles,
           const Live** first) {
    *meets = *holding = *straddles = false;
    *first = nullptr;
    for (const Live& l : live) {
        if (l.memory < va + bytes && va < l.memory + l.bytes) {
            *meets = true;
            if (va >= l.memory && va + bytes <= l.memory + l.bytes) *holding = true;
            if (!*first || l.memory < (*first)->memory) *first = &l;
        }
    }
    *straddles = *meets && !*holding;
}

// The index a home gets in concurrent_test: from its memory and size, so a reader tells a real entry from bytes that
// are not one.
std::uint32_t index_of(std::uint64_t memory, std::uint32_t bytes) {
    return static_cast<std::uint32_t>(((memory >> 4) * 2654435761ull) ^ (bytes * 40503u)) & gxvk_homes::kIndexMask;
}

void concurrent_test() {
    using gxvk_homes::HomeIndex;
    HomeIndex ix;
    const std::uint64_t base = 0x300000000ull;
    // Homes that stay: 64 of 4 KiB, 64 KiB apart.
    constexpr int kStable = 64;
    const auto stable_at = [&](int k) { return base + static_cast<std::uint64_t>(k) * 0x10000; };
    for (int k = 0; k < kStable; ++k) ix.insert(index_of(stable_at(k), 4096), stable_at(k), 4096);
    // Pairs the writer swaps in one batch: A at the slot, 40 KiB (ten pages); B 256 bytes on, 40 KiB less 512. Both hold
    // the probe, a page and a half in.
    constexpr int kPairs = 32;
    const auto pair_at = [&](int k) { return base + 0x1000000 + static_cast<std::uint64_t>(k) * 0x10000; };
    const auto a_of = [&](int k) { return Live{index_of(pair_at(k), 40960), pair_at(k), 40960}; };
    const auto b_of = [&](int k) { return Live{index_of(pair_at(k) + 256, 40448), pair_at(k) + 256, 40448}; };
    std::vector<int> pair_b(kPairs, 0);
    for (int k = 0; k < kPairs; ++k) ix.insert(a_of(k).index, a_of(k).memory, a_of(k).bytes);
    // Homes made and ended at random in 16 MiB; 16 MiB nobody touches.
    const std::uint64_t churn = base + 0x2000000, empty = base + 0x3000000;
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> batches{0};
    std::thread writer([&] {
        std::mt19937_64 r(99);
        std::vector<Live> live;
        while (!stop.load(std::memory_order_relaxed)) {
            {
                const HomeIndex::Batch b(ix);
                for (int n = 0; n < 4; ++n) {
                    const int k = static_cast<int>(r() % kPairs);
                    const Live from = pair_b[k] ? b_of(k) : a_of(k), to = pair_b[k] ? a_of(k) : b_of(k);
                    if (!ix.erase(from.index, from.memory)) check(false, "the writer's erase of a pair's home", from.memory, from.bytes);
                    ix.insert(to.index, to.memory, to.bytes);
                    pair_b[k] ^= 1;
                }
                if (live.size() < 200 || r() % 2) {
                    const std::uint32_t bytes = static_cast<std::uint32_t>(16 * (1 + r() % 1024));
                    const std::uint64_t memory = churn + 16 * (r() % (0x1000000 / 16 - 1024));
                    bool over = false;
                    for (const Live& l : live) over |= l.memory < memory + bytes && memory < l.memory + l.bytes;
                    if (!over) {
                        ix.insert(index_of(memory, bytes), memory, bytes);
                        live.push_back({index_of(memory, bytes), memory, bytes});
                    }
                } else {
                    const std::size_t k = r() % live.size();
                    ix.erase(live[k].index, live[k].memory);
                    live[k] = live.back();
                    live.pop_back();
                }
            }
            batches.fetch_add(1, std::memory_order_relaxed);
            // (The command processor drains its events between draws; three batches in four come right after another.)
            if (r() % 4 == 0) std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    });
    std::atomic<std::uint64_t> asked{0}, unsure{0};
    const auto reader = [&](int seed) {
        std::mt19937_64 r(seed);
        std::uint64_t n = 0, u = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            HomeIndex::Hit h;
            const int what = static_cast<int>(r() % 4);
            if (what == 0) {
                const int k = static_cast<int>(r() % kStable);
                const std::uint64_t va = stable_at(k) + 16 * (r() % 255);
                const int a = ix.read_meets(va, 16, &h);
                if (a == HomeIndex::kReadUnsure) ++u;
                else check(a == HomeIndex::kReadMeets && h.memory == stable_at(k) && h.bytes == 4096 && h.index == index_of(h.memory, h.bytes),
                           "a home that stays, found", va, 16);
            } else if (what == 1) {
                const int k = static_cast<int>(r() % kPairs);
                const std::uint64_t va = pair_at(k) + 6144;
                const int a = ix.read_meets(va, 4, &h);
                if (a == HomeIndex::kReadUnsure) ++u;
                else check(a == HomeIndex::kReadMeets && ((h.memory == a_of(k).memory && h.bytes == a_of(k).bytes) ||
                                                          (h.memory == b_of(k).memory && h.bytes == b_of(k).bytes)) &&
                               h.index == index_of(h.memory, h.bytes),
                           "one of a swapped pair, never neither", va, 4);
            } else if (what == 2) {
                const std::uint64_t va = empty + 16 * (r() % (0x1000000 / 16 - 4096));
                const int a = ix.read_meets(va, 16 * (1 + r() % 1024), &h);
                if (a == HomeIndex::kReadUnsure) ++u;
                else check(a == HomeIndex::kReadNone, "nothing where nothing ever was", va, 16);
            } else {
                const std::uint64_t va = churn + 16 * (r() % (0x1000000 / 16 - 4096)), bytes = 16 * (1 + r() % 1024);
                const int a = ix.read_meets(va, bytes, &h);
                if (a == HomeIndex::kReadUnsure) ++u;
                else if (a == HomeIndex::kReadMeets)
                    check(h.index == index_of(h.memory, h.bytes) && h.memory < va + bytes && va < h.memory + h.bytes,
                          "a hit is a home the writer put there, meeting the range", va, bytes);
            }
            ++n;
        }
        asked.fetch_add(n);
        unsure.fetch_add(u);
    };
    std::vector<std::thread> readers;
    for (int t = 0; t < 3; ++t) readers.emplace_back(reader, 1000 + t);
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    stop.store(true);
    writer.join();
    for (std::thread& t : readers) t.join();
    // Done: the lists every reader has left freed, and each pair's home the one the writer left.
    ix.reclaim();
    check(ix.retired() == 0, "every retired list freed once the readers are done", ix.retired(), 0);
    for (int k = 0; k < kPairs; ++k) {
        HomeIndex::Hit h;
        const Live want = pair_b[k] ? b_of(k) : a_of(k);
        check(ix.read_meets(pair_at(k) + 6144, 4, &h) == HomeIndex::kReadMeets && h.memory == want.memory && h.bytes == want.bytes,
              "the pair's home the writer left", pair_at(k), 0);
    }
    // (Unsure: a batch under way through every try - a reader that is always unsure checks nothing.)
    check(asked.load() > 1000 && unsure.load() * 2 < asked.load(), "readers mostly sure beside the writer", unsure.load(), asked.load());
    std::printf("gxvk_homes_test: concurrent: %llu batches, %llu reads, %llu unsure\n", static_cast<unsigned long long>(batches.load()),
                static_cast<unsigned long long>(asked.load()), static_cast<unsigned long long>(unsure.load()));
}

}  // namespace

int main() {
    using gxvk_homes::HomeIndex;
    // The handle: tag, generation and index survive the packing; stage 1b's tag is not ours.
    for (std::uint32_t gen = 0; gen < 256; gen += 37) {
        for (std::uint32_t index : {0u, 1u, 4095u, 4096u, gxvk_homes::kIndexMask}) {
            const std::uint32_t h = gxvk_homes::handle_of(index, gen);
            check(gxvk_homes::is_handle(h) && gxvk_homes::handle_index(h) == index && gxvk_homes::handle_gen(h) == gen, "handle packing", index, gen);
        }
    }
    check(!gxvk_homes::is_handle(0xa0000123u) && !gxvk_homes::is_handle(0), "another tag is no handle", 0, 0);
    // The default (BBHOST_GXVK_HOMES unset): on only with a card with room and a frame rate above 60 - Display-Limited at
    // 144 Hz or uncapped (960) on, 60 and 30 off - each reason its own bit; where the run commit is on by the platform's
    // default, at any frame rate.
    {
        using namespace gxvk_homes;
        check(default_off(false, false, false, 61, false, false) == 0 && default_off(false, false, false, 960, false, false) == 0 &&
                  default_off(false, false, false, 144, false, false) == 0,
              "on with room, above 60", 0, 0);
        check(default_off(false, false, false, 60, false, false) == kDefaultOffRate && default_off(false, false, false, 30, false, false) == kDefaultOffRate,
              "off at 60 and below", 0, 60);
        // Where the run commit is on by the platform's default (Windows) the frame rate keeps nothing off - the homes take
        // its place at 60 and 30 as uncapped - and the other reasons still do.
        check(default_off(false, false, false, 60, false, true) == 0 && default_off(false, false, false, 30, false, true) == 0 &&
                  default_off(false, false, false, 960, false, true) == 0,
              "Windows' default off at 60 and below", 16, 60);
        check(default_off(true, false, false, 60, false, true) == kDefaultOffIntegrated && default_off(false, false, true, 30, false, true) == kDefaultOffSmall,
              "Windows' default at 60 on with an integrated GPU or small video memory refused", 8, 60);
        // The threads keep nothing off (since 2026-11-09): the pool takes the threads the process's CPUs give it - one
        // from 8, none under 8 (the homes alone) - host/gxvk_lists.h pool_threads_for.
        check(default_off(true, false, false, 960, false, false) == kDefaultOffIntegrated && default_off(false, true, false, 960, false, false) == kDefaultOffTight &&
                  default_off(false, false, true, 960, false, false) == kDefaultOffSmall,
              "off on an integrated GPU, with tight or small video memory", 0, 0);
        check(default_off(false, false, true, 60, false, false) == (kDefaultOffSmall | kDefaultOffRate), "every reason said", 8, 60);
        check(default_off(false, false, false, 960, true, false) == kDefaultOffRunCommit, "off with the run commit turned on by its setting", 80, 960);
        // Small video memory: refused under 5,000 MiB free at the start (a 6 GB card has more) unless a budget is given;
        // a card with room never.
        check(small_refused(true, 4000, false) && small_refused(true, 4999, false) && small_refused(true, 0, false) &&
                  small_refused(true, 2950, false),
              "small video memory under 5,000 MiB free refused (a 4 GB card's 2,950)", 4999, 0);
        check(!small_refused(true, 5000, false) && !small_refused(true, 5300, false) && !small_refused(true, 6143, false) &&
                  !small_refused(true, 7249, false),
              "small video memory from 5,000 MiB free taken (a 6 GB card), with the pool's budget", 5000, 0);
        check(!small_refused(true, 4000, true) && !small_refused(true, 0, true), "a budget given takes small video memory at any size", 4000, 1);
        check(!small_refused(false, 4000, false) && !small_refused(false, 0, false), "a card with room is never refused", 4000, 0);
        check(default_off(false, false, small_refused(true, 7000, false), 960, false, false) == 0 &&
                  default_off(false, false, small_refused(true, 4000, false), 960, false, false) == kDefaultOffSmall,
              "on with 7,000 MiB free, off with 4,000", 7000, 4000);
        // The homes end bound by the default where nothing keeps them off, and by BBHOST_GXVK_HOMES=1 wherever the device
        // takes them: the rule's frame rate and run-commit reasons do not stop 1; the device's (an integrated GPU,
        // tight memory, small video memory refused) do. 0, compare and twins bind none.
        check(homes_bound(Mode::kAuto, 0, false, false, false), "the default binds where the rule holds", 0, 0);
        for (unsigned bit : {kDefaultOffIntegrated, kDefaultOffTight, kDefaultOffSmall, kDefaultOffRate, kDefaultOffRunCommit})
            check(!homes_bound(Mode::kAuto, bit, false, false, false), "the default binds none where a reason keeps it off", bit, 0);
        check(homes_bound(Mode::kHandle, kDefaultOffRate, false, false, false) &&
                  homes_bound(Mode::kHandle, kDefaultOffRate | kDefaultOffRunCommit, false, false, false) &&
                  homes_bound(Mode::kHandle, kDefaultOffRunCommit, false, false, false),
              "BBHOST_GXVK_HOMES=1 binds at 60 fps and with the run commit set", 8, 60);
        check(!homes_bound(Mode::kHandle, kDefaultOffIntegrated, true, false, false) &&
                  !homes_bound(Mode::kHandle, kDefaultOffTight, false, true, false) &&
                  !homes_bound(Mode::kHandle, kDefaultOffSmall, false, false, true),
              "BBHOST_GXVK_HOMES=1 binds none on an integrated GPU, with tight memory or small video memory refused", 0, 1);
        check(!homes_bound(Mode::kOff, 0, false, false, false) && !homes_bound(Mode::kCompare, 0, false, false, false) &&
                  !homes_bound(Mode::kTwins, 0, false, false, false),
              "BBHOST_GXVK_HOMES=0, compare and twins bind none", 0, 2);
        // Windows' run commit (on by its default alone) gives way wherever the homes are bound - the default's rule or
        // BBHOST_GXVK_HOMES=1 at 60 fps alike - and is kept where they are not, or where BBHOST_RUN_COMMIT is set.
        check(run_commit_gives_way(homes_bound(Mode::kAuto, 0, false, false, false), true) &&
                  run_commit_gives_way(homes_bound(Mode::kHandle, 0, false, false, false), true),
              "the run commit gives way where the rule holds", 0, 1);
        check(run_commit_gives_way(homes_bound(Mode::kHandle, default_off(false, false, false, 60, false, false), false, false, false), true) &&
                  run_commit_gives_way(homes_bound(Mode::kHandle, default_off(false, false, false, 960, false, false), false, false, false), true),
              "the run commit gives way to BBHOST_GXVK_HOMES=1 at 60 fps", 8, 60);
        check(!run_commit_gives_way(true, false) && !run_commit_gives_way(false, false),
              "nothing to give way without the platform's default (Linux, or BBHOST_RUN_COMMIT set)", 0, 0);
        check(!run_commit_gives_way(homes_bound(Mode::kAuto, default_off(true, false, false, 960, false, true), true, false, false), true),
              "Windows' default keeps the run commit on an integrated GPU", 8, 60);
        check(run_commit_gives_way(homes_bound(Mode::kAuto, default_off(false, false, false, 60, false, true), false, false, false), true) &&
                  run_commit_gives_way(homes_bound(Mode::kAuto, default_off(false, false, false, 30, false, true), false, false, false), true),
              "Windows' default gives the run commit way at 60 and 30 fps", 16, 60);
        check(!run_commit_gives_way(homes_bound(Mode::kHandle, kDefaultOffIntegrated, true, false, false), true),
              "BBHOST_GXVK_HOMES=1 on an integrated GPU keeps the run commit", 0, 3);
        check(!run_commit_gives_way(homes_bound(Mode::kOff, 0, false, false, false), true) &&
                  !run_commit_gives_way(homes_bound(Mode::kCompare, 0, false, false, false), true) &&
                  !run_commit_gives_way(homes_bound(Mode::kTwins, 0, false, false, false), true),
              "BBHOST_GXVK_HOMES=0, compare and twins keep the run commit", 0, 2);
        // Where Windows' default keeps the run commit (the homes not bound), its duty decides when it forks
        // (host/list_resolve.h duty_wanted: on by the platform's default alone, still forking, BBHOST_RUN_COMMIT_REST not 0);
        // where it gave way, with BBHOST_RUN_COMMIT set (or Linux's default) and with BBHOST_RUN_COMMIT_REST=0 there is none.
        {
            const bool kept_igpu = !run_commit_gives_way(homes_bound(Mode::kAuto, default_off(true, false, false, 60, false, true), true, false, false), true);
            const bool kept_small = !run_commit_gives_way(homes_bound(Mode::kAuto, default_off(false, false, true, 60, false, true), false, false, true), true);
            check(list_resolve::duty_wanted(true, kept_igpu, 1) && list_resolve::duty_wanted(true, kept_small, 1) &&
                      list_resolve::duty_wanted(true, kept_small, 2),
                  "the duty decides Windows' run commit on an integrated GPU and with small video memory refused", 8, 60);
            const bool kept_room = !run_commit_gives_way(homes_bound(Mode::kAuto, default_off(false, false, false, 60, false, true), false, false, false), true);
            check(!list_resolve::duty_wanted(true, kept_room, 1), "no duty where the run commit gave way to the homes", 16, 60);
            check(!list_resolve::duty_wanted(false, true, 1) && !list_resolve::duty_wanted(false, true, 2),
                  "no duty with BBHOST_RUN_COMMIT set or Linux's default", 0, 0);
            check(!list_resolve::duty_wanted(true, true, 0), "BBHOST_RUN_COMMIT_REST=0: no duty", 0, 1);
        }
    }

    std::mt19937_64 rng(4321);
    HomeIndex ix;
    std::vector<Live> live;
    std::uint32_t next_index = 0;
    // Two "heaps" of 256 pages each, far apart: buffers of 16 B .. 64 KiB at 16-byte steps, never overlapping a live one.
    const std::uint64_t heaps[2] = {0x200000000ull, 0x7f0000000ull};
    const auto overlaps = [&](std::uint64_t m, std::uint64_t n) {
        for (const Live& l : live) {
            if (l.memory < m + n && m < l.memory + l.bytes) return true;
        }
        return false;
    };
    std::uint64_t queries = 0, maybes = 0;
    for (int step = 0; step < 60000; ++step) {
        const int op = static_cast<int>(rng() % 10);
        if (op < 4 || live.empty()) {
            const std::uint64_t base = heaps[rng() % 2];
            const std::uint32_t bytes = rng() % 8 ? static_cast<std::uint32_t>(16 * (1 + rng() % 64)) : static_cast<std::uint32_t>(16 * (1 + rng() % 4096));
            const std::uint64_t memory = base + 16 * (rng() % (256 * 4096 / 16));
            if (overlaps(memory, bytes)) continue;
            ix.insert(next_index, memory, bytes);
            live.push_back({next_index++, memory, bytes});
        } else if (op == 4 && rng() % 4 == 0) {
            // A new home at a live one's memory (no larger: nothing else is met) replaces it.
            const std::size_t k = rng() % live.size();
            const std::uint32_t bytes = 16 * (1 + static_cast<std::uint32_t>(rng() % (live[k].bytes / 16)));
            ix.insert(next_index, live[k].memory, bytes);
            live[k] = {next_index++, live[k].memory, bytes};
        } else if (op < 6) {
            const std::size_t k = rng() % live.size();
            check(ix.erase(live[k].index, live[k].memory), "erase of a live home", live[k].memory, live[k].bytes);
            check(!ix.erase(live[k].index, live[k].memory), "a second erase does nothing", live[k].memory, live[k].bytes);
            live[k] = live.back();
            live.pop_back();
        } else {
            // A query: near a live home (its start, inside it, across its edge, just past it) or anywhere in a heap.
            std::uint64_t va, bytes;
            if (!live.empty() && rng() % 4) {
                const Live& l = live[rng() % live.size()];
                const int where = static_cast<int>(rng() % 5);
                va = where == 0 ? l.memory : where == 1 ? l.memory + 4 * (rng() % (l.bytes / 4)) : where == 2 ? l.memory + l.bytes - 4 : where == 3 ? l.memory + l.bytes : l.memory - 16;
                bytes = 4 * (1 + rng() % 64);
            } else {
                va = heaps[rng() % 2] + 4 * (rng() % (256 * 4096 / 4));
                bytes = rng() % 16 ? 4 * (1 + rng() % 256) : 4096 * (1 + rng() % 200);  // some long ranges: the map's
            }
            bool m = false, h = false, s = false;
            const Live* first = nullptr;
            model(live, va, bytes, &m, &h, &s, &first);
            HomeIndex::Hit hit;
            const bool im = ix.meets(va, bytes, &hit);
            // The pool's view: the same home, or unsure only for a range of 64 pages or more.
            {
                HomeIndex::Hit rh;
                const int r = ix.read_meets(va, bytes, &rh);
                const bool long_range = ((va + bytes - 1) >> 12) - (va >> 12) >= HomeIndex::kReadMostPages;
                if (r == HomeIndex::kReadUnsure) {
                    check(long_range, "the published lists unsure of a short range", va, bytes);
                } else {
                    check(!long_range && (r == HomeIndex::kReadMeets) == m, "the published lists' meets", va, bytes);
                    if (r == HomeIndex::kReadMeets && im)
                        check(rh.index == hit.index && rh.memory == hit.memory && rh.bytes == hit.bytes, "the published lists' home", va, bytes);
                }
            }
            ++queries;
            if (ix.may_meet(va, bytes)) ++maybes;
            check(im == m, "meets", va, bytes);
            check(!m || ix.may_meet(va, bytes), "the filter said no where a home is", va, bytes);
            check(ix.holding(va, bytes, nullptr) == h, "holding", va, bytes);
            check(ix.straddles(va, bytes) == s, "straddles", va, bytes);
            // The hit is a live home that meets the range (the one at or below va, else the lowest above it).
            if (im) {
                bool found = false;
                for (const Live& l : live) found |= l.index == hit.index && l.memory == hit.memory && l.bytes == hit.bytes;
                check(found && hit.memory < va + bytes && va < hit.memory + hit.bytes, "the hit is a live home meeting the range", va, bytes);
                if (h) check(va >= hit.memory && va + bytes <= hit.memory + hit.bytes, "holding names the home that holds it", va, bytes);
            }
        }
        check(ix.size() == live.size(), "size", 0, live.size());
    }
    // Every home erased: nothing meets anything, and the filter is back to "no".
    for (const Live& l : live) check(ix.erase(l.index, l.memory), "final erase", l.memory, l.bytes);
    for (int q = 0; q < 2000; ++q) {
        const std::uint64_t va = heaps[q % 2] + 16 * (rng() % (256 * 4096 / 16));
        check(!ix.meets(va, 64, nullptr) && !ix.may_meet(va, 64) && ix.read_meets(va, 64, nullptr) == HomeIndex::kReadNone, "empty index", va, 64);
    }
    ix.reclaim();
    check(ix.retired() == 0, "no reader: every retired list freed", ix.retired(), 0);
    // A reader inside its section: what the writer retires meanwhile stays until it leaves (and only that long).
    {
        HomeIndex::Hold* hold = new HomeIndex::Hold(ix);
        ix.insert(next_index, heaps[0], 64);
        check(ix.erase(next_index, heaps[0]), "erase beside a held reader", heaps[0], 64);
        ix.reclaim();
        check(ix.retired() > 0, "a list retired while a reader is inside kept", ix.retired(), 0);
        delete hold;
        ix.reclaim();
        check(ix.retired() == 0, "freed once the reader left", ix.retired(), 0);
    }
    concurrent_test();
    std::printf("gxvk_homes_test: %llu queries, the filter said maybe to %llu; %d failures\n", static_cast<unsigned long long>(queries),
                static_cast<unsigned long long>(maybes), g_failures.load());
    return g_failures.load() ? 1 : 0;
}
