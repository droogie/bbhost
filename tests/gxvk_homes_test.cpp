// Stage 4b.3c's index of live homes by address (host/gxvk_homes_index.h),
// without the game or a GPU: homes made, ended and made again over the same
// memory - small constant buffers packed in a few pages, as GX's heaps hold
// them, a page shared by several, ranges that run across pages - and every
// answer the command processor asks of it (meets, holding, straddles, and the
// filter's "no") checked against a plain list of the live ranges. The filter
// may say "maybe" where nothing is; it may never say "no" where a home is.
// The handle's packing is checked too (host/gxvk_homes.h), and the default's
// rule (default_off: on only with a card with room or 5,000 MiB of small video
// memory free and a frame rate above 60 - at any frame rate where the run
// commit is on by the platform's default; any number of threads;
// small_refused; homes_bound and run_commit_gives_way: Windows' run commit
// gives way wherever the homes are bound, by the rule or BBHOST_GXVK_HOMES=1).
#include "host/gxvk_homes.h"
#include "host/gxvk_homes_index.h"
#include "host/list_resolve.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Live {
    std::uint32_t index;
    std::uint64_t memory;
    std::uint32_t bytes;
};

int g_failures = 0;
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
        check(!ix.meets(va, 64, nullptr) && !ix.may_meet(va, 64), "empty index", va, 64);
    }
    std::printf("gxvk_homes_test: %llu queries, the filter said maybe to %llu; %d failures\n", static_cast<unsigned long long>(queries),
                static_cast<unsigned long long>(maybes), g_failures);
    return g_failures ? 1 : 0;
}
