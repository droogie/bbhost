// The image heap's placement in its blocks (host/heap_place.h) and its
// compaction's policy (host/heap_compact.h): which blocks an image may go in
// and in what order, the first free range that takes it; when a compaction
// begins and stops, the block it chooses and why another is not, the room it
// keeps for the largest target, and the pace of its copies.
#include "host/heap_compact.h"

#include <cstdio>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

using namespace heap_compact;
using heap_place::kTargets;
using heap_place::kTextures;

constexpr std::uint64_t M = kMiB;

// A live block of class `cls` with free ranges {offset, size} (MiB), its use the rest of the block.
Block block(int cls, std::initializer_list<std::pair<std::uint64_t, std::uint64_t>> holes, std::uint32_t type = 0) {
    Block b;
    b.cls = cls;
    b.live = true;
    b.type = type;
    std::uint64_t free = 0;
    for (const auto& [o, s] : holes) {
        b.free[o * M] = s * M;
        free += s * M;
    }
    b.used = kBlock - free;
    return b;
}

void test_place() {
    // Which blocks an image may go in.
    CHECK(heap_place::class_fits(true, true, -1, kTargets));
    CHECK(heap_place::class_fits(true, true, kTargets, kTargets));
    CHECK(!heap_place::class_fits(true, true, kTextures, kTargets));  // a target never goes in the textures' blocks
    CHECK(heap_place::class_fits(true, true, kTargets, kTextures));   // a texture takes the targets' holes
    CHECK(!heap_place::class_fits(true, false, kTargets, kTextures));  // ... not with the classes strict
    CHECK(heap_place::class_fits(false, false, kTextures, kTargets));  // the classes off: anywhere
    // The first free range that takes it, at its alignment.
    std::map<std::uint64_t, std::uint64_t> free = {{0, 4 * M}, {10 * M, 16 * M}, {40 * M, 64 * M}};
    std::uint64_t at = 0;
    CHECK(heap_place::first_fit(free, 4 * M, 256, at) && at == 0);
    CHECK(heap_place::first_fit(free, 5 * M, 256, at) && at == 10 * M);
    CHECK(heap_place::first_fit(free, 20 * M, 256, at) && at == 40 * M);
    CHECK(!heap_place::first_fit(free, 65 * M, 256, at));
    std::map<std::uint64_t, std::uint64_t> odd = {{M + 4096, 8 * M}};
    CHECK(heap_place::first_fit(odd, 7 * M, 64 * 1024, at) && at == M + 64 * 1024);  // aligned up inside the range
    CHECK(!heap_place::first_fit(odd, 8 * M, 64 * 1024, at));  // the alignment costs it the room
    // Taken out of the middle of a range: the two ends stay free.
    heap_place::take(free, 12 * M, 2 * M);
    CHECK(free.size() == 4 && free[10 * M] == 2 * M && free[14 * M] == 12 * M);
    heap_place::take(free, 0, 4 * M);  // a whole range
    CHECK(free.count(0) == 0 && free.size() == 3);
    CHECK(heap_place::largest(free) == 64 * M);
    // The order: its own class's blocks fullest first, then another class's holes, then empty ones.
    std::vector<Block> bl = {block(kTextures, {{0, 200}}), block(-1, {{0, 256}}), block(kTargets, {{0, 100}}), block(kTargets, {{0, 10}}),
                             block(kTextures, {{0, 50}})};
    std::vector<std::size_t> idx = {0, 1, 2, 3, 4};
    heap_place::order(bl, idx, kTextures, true);
    CHECK((idx == std::vector<std::size_t>{4, 0, 3, 2, 1}));
    idx = {0, 1, 2, 3, 4};
    heap_place::order(bl, idx, kTargets, true);
    CHECK((idx == std::vector<std::size_t>{3, 2, 4, 0, 1}));
    idx = {0, 1, 2, 3, 4};
    heap_place::order(bl, idx, kTargets, false);  // block order
    CHECK((idx == std::vector<std::size_t>{0, 1, 2, 3, 4}));
}

void test_begin_and_stop() {
    // Begins only on, settled, with no load, none under way, no block lying empty and the quiet over.
    CHECK(may_begin(true, true, false, false, false, 50000, 40000));
    CHECK(!may_begin(false, true, false, false, false, 50000, 40000));
    CHECK(!may_begin(true, false, false, false, false, 50000, 40000));
    CHECK(!may_begin(true, true, true, false, false, 50000, 40000));
    CHECK(!may_begin(true, true, false, true, false, 50000, 40000));
    CHECK(!may_begin(true, true, false, false, true, 50000, 40000));
    CHECK(!may_begin(true, true, false, false, false, 39999, 40000));
    CHECK(may_begin(true, true, false, false, false, 40000, 40000));
    // Stops at a load, when the world is no longer settled, or kGiveUpMs on.
    CHECK(!must_stop(false, true, 10000 + kGiveUpMs - 1, 10000));
    CHECK(must_stop(false, true, 10000 + kGiveUpMs, 10000));
    CHECK(must_stop(true, true, 10001, 10000));
    CHECK(must_stop(false, false, 10001, 10000));
    // The world settled by flips as well: kSettleFlips since the latest load ended.
    CHECK(!flips_settled(1000 + kSettleFlips - 1, 1000));
    CHECK(flips_settled(1000 + kSettleFlips, 1000));
    CHECK(kSettleFlips == 900);
    // The avoided-block slot the targets' drain and the compaction share: either begins only while it is free, and
    // goes on only with the block it took.
    CHECK(slot_free(-1));
    CHECK(!slot_free(0));
    CHECK(!slot_free(3));
    CHECK(slot_mine(3, 3));
    CHECK(slot_mine(0, 0));
    CHECK(!slot_mine(3, 2));   // another's block
    CHECK(!slot_mine(3, -1));  // given back, or its slot reused by a new block
    CHECK(!slot_mine(-1, -1)); // nothing taken is nothing owned
    // A turn a second.
    CHECK(!turn_due(1999, 1000));
    CHECK(turn_due(2000, 1000));
}

void test_pace() {
    // A turn's credit is kTurnBytes, never saved beyond it.
    std::int64_t c = turn_credit(0);
    CHECK(c == static_cast<std::int64_t>(kTurnBytes));
    CHECK(turn_credit(c) == static_cast<std::int64_t>(kTurnBytes));
    // Copies take their size off while it is above 0: 16 + 16 MiB in a turn, then none.
    CHECK(may_copy(c, 0));
    c -= static_cast<std::int64_t>(16 * M);
    CHECK(may_copy(c, 0));
    c -= static_cast<std::int64_t>(16 * M);
    CHECK(!may_copy(c, 0));
    // A 48 MiB target moves at once and leaves a debt: the next turn has 16 MiB, the one after 32.
    c = turn_credit(0) - static_cast<std::int64_t>(48 * M);
    CHECK(!may_copy(c, 0));
    c = turn_credit(c);
    CHECK(c == static_cast<std::int64_t>(16 * M));
    c = turn_credit(c);
    CHECK(c == static_cast<std::int64_t>(32 * M));
    // Over ten turns no more than ten turns' bytes, whatever the sizes.
    c = 0;
    std::uint64_t moved = 0;
    for (int turn = 0; turn < 10; ++turn) {
        c = turn_credit(c);
        while (may_copy(c, 0)) {
            c -= static_cast<std::int64_t>(24 * M);
            moved += 24 * M;
        }
    }
    CHECK(moved <= 10 * kTurnBytes + 24 * M);
    // At most kTurnItems images a turn, whatever their size.
    c = turn_credit(0);
    std::size_t n = 0;
    while (may_copy(c, n)) {
        c -= 4096;
        ++n;
    }
    CHECK(n == kTurnItems && c > 0);
}

void test_room_and_place() {
    // The free room of the blocks holding anything; an empty one is noticed, not counted.
    std::vector<Block> bl = {block(kTargets, {{0, 100}}), block(kTextures, {{0, 50}}), block(-1, {{0, 256}}), block(kTargets, {{0, 10}}, 1)};
    bool empty = false;
    CHECK(room(bl, 0, &empty) == 150 * M && empty);
    CHECK(room(bl, 1, &empty) == 10 * M && !empty);
    // A trial placement on a copy: largest first, the targets only in the targets' blocks.
    std::vector<Block> two = {block(kTargets, {{0, 20}, {100, 8}}), block(kTextures, {{0, 30}}), block(kTargets, {{0, 64}})};
    std::uint64_t after = 0;
    CHECK(try_place(two, 2, 0, {{16 * M, kTargets}, {8 * M, kTargets}}, true, true, true, after));
    CHECK(after == 30 * M);  // the textures' block's range is the largest left
    CHECK(two[0].used == kBlock - 28 * M);  // the original is untouched
    CHECK(!try_place(two, 2, 0, {{24 * M, kTargets}}, true, true, true, after));  // a target cannot use the textures' block
    CHECK(try_place(two, 2, 0, {{24 * M, kTextures}}, true, true, true, after) && after == 20 * M);  // a texture can
    CHECK(!try_place(two, 2, 0, {{16 * M, kTargets}, {16 * M, kTargets}}, true, true, true, after));  // the second finds none
    CHECK(!try_place(two, 0, 1, {{4 * M, kTargets}}, true, true, true, after));  // another memory type has no blocks
}

void test_choose() {
    // Four blocks; only the targets' block holding 96 MiB is at most kMostUsed, the others hold 166-178 MiB.
    const std::vector<Block> bl = {block(kTargets, {{0, 50}, {100, 40}}), block(kTargets, {{0, 48}, {64, 30}}), block(kTextures, {{0, 90}}),
                                   block(kTargets, {{0, 160}})};
    std::vector<std::vector<Item>> items(4);
    items[3] = {{32 * M, kTargets}, {32 * M, kTargets}, {16 * M, kTextures}, {16 * M, kTextures}};
    std::vector<bool> pinned(4, false);
    Why why = kNone;
    CHECK(room(bl, 0, nullptr) == 418 * M);
    CHECK(choose(bl, 0, items, pinned, 16 * M, true, true, true, &why) == 3 && why == kChosen);
    // The room kept: its copies placed leave 58 MiB as the largest range (the textures' block's).
    CHECK(choose(bl, 0, items, pinned, 58 * M, true, true, true, &why) == 3);
    CHECK(choose(bl, 0, items, pinned, 59 * M, true, true, true, &why) == -1 && why == kKeep);
    // Holding what cannot move.
    pinned[3] = true;
    CHECK(choose(bl, 0, items, pinned, 0, true, true, true, &why) == -1 && why == kPinned);
    pinned[3] = false;
    // Too many images to copy within the turns before it would give up.
    std::vector<std::vector<Item>> many = items;
    many[3].assign(kMostCopies + 1, Item{64 * 1024, kTextures});
    CHECK(choose(bl, 0, many, pinned, 0, true, true, true, &why) == -1 && why == kMany);
    many[3].resize(kMostCopies);
    CHECK(choose(bl, 0, many, pinned, 0, true, true, true, &why) == 3);
    // The others' room short of its copies and kSpare (258 MiB against 252 + 32).
    std::vector<std::vector<Item>> big = items;
    big[3] = {{64 * M, kTargets}, {64 * M, kTargets}, {64 * M, kTextures}, {60 * M, kTextures}};
    CHECK(choose(bl, 0, big, pinned, 0, true, true, true, &why) == -1 && why == kSpareShort);
    // A copy that finds no hole: a 72 MiB target, where no targets' range is 72 MiB.
    std::vector<std::vector<Item>> wide = items;
    wide[3] = {{72 * M, kTargets}};
    CHECK(choose(bl, 0, wide, pinned, 0, true, true, true, &why) == -1 && why == kNoFit);
    // A texture goes in its own blocks first, then in the targets' holes - unless the classes are strict.
    wide[3] = {{60 * M, kTextures}, {45 * M, kTextures}};  // the second in a targets' 48 MiB hole
    CHECK(choose(bl, 0, wide, pinned, 0, true, true, true, &why) == 3);
    CHECK(choose(bl, 0, wide, pinned, 0, true, false, true, &why) == -1 && why == kNoFit);
    wide[3] = {{91 * M, kTextures}};  // no block has 91 MiB free
    CHECK(choose(bl, 0, wide, pinned, 0, true, true, true, &why) == -1 && why == kNoFit);
    // The heap's room under a block and kRoomOver: none; exactly that is enough.
    const std::vector<Block> under = {block(kTargets, {{0, 64}}), block(kTargets, {{0, 63}}), block(kTargets, {{0, 192}})};
    const std::vector<Block> edge = {block(kTargets, {{0, 64}}), block(kTargets, {{0, 64}}), block(kTargets, {{0, 192}})};
    std::vector<std::vector<Item>> e_items(3);
    std::vector<bool> e_pin(3, false);
    CHECK(choose(under, 0, e_items, e_pin, 0, true, true, true, &why) == -1 && why == kRoom);
    CHECK(room(edge, 0, nullptr) == kBlock + kRoomOver);
    CHECK(choose(edge, 0, e_items, e_pin, 0, true, true, true, &why) == 2 && why == kChosen);
    // The emptiest first: of two candidates the one holding less.
    const std::vector<Block> pair = {block(kTargets, {{0, 150}}), block(kTargets, {{0, 64}}), block(kTargets, {{0, 170}})};
    std::vector<std::vector<Item>> p_items(3);
    std::vector<bool> p_pin(3, false);
    CHECK(choose(pair, 0, p_items, p_pin, 0, true, true, true, &why) == 2);
    p_pin[2] = true;  // ... and the next emptiest when that one cannot
    CHECK(choose(pair, 0, p_items, p_pin, 0, true, true, true, &why) == 0);
    // More than kMostUsed is too full; exactly that is not.
    const std::vector<Block> full = {block(kTargets, {{0, 95}}), block(kTargets, {{0, 95}}), block(kTargets, {{0, 95}}), block(kTargets, {{0, 95}})};
    const std::vector<Block> at_most = {block(kTargets, {{0, 96}}), block(kTargets, {{0, 96}}), block(kTargets, {{0, 96}}),
                                        block(kTargets, {{0, 96}})};
    std::vector<std::vector<Item>> f_items(4);
    std::vector<bool> f_pin(4, false);
    CHECK(choose(full, 0, f_items, f_pin, 0, true, true, true, &why) == -1 && why == kFull);
    CHECK(choose(at_most, 0, f_items, f_pin, 0, true, true, true, &why) >= 0);
    // Never the last block of its memory type.
    const std::vector<Block> one = {block(kTargets, {{0, 200}}), block(kTargets, {{0, 200}}, 1)};
    std::vector<std::vector<Item>> o_items(2);
    std::vector<bool> o_pin(2, false);
    CHECK(choose(one, 0, o_items, o_pin, 0, true, true, true, &why) == -1 && why == kNone);
    // An empty block is the keeper's: not counted in the room, not a candidate.
    std::vector<Block> with_empty = bl;
    with_empty.push_back(block(-1, {{0, 256}}));
    items.push_back({});
    pinned.push_back(false);
    CHECK(room(with_empty, 0, nullptr) == 418 * M);
    CHECK(choose(with_empty, 0, items, pinned, 16 * M, true, true, true, &why) == 3);
}

}  // namespace

int main() {
    test_place();
    test_begin_and_stop();
    test_pace();
    test_room_and_place();
    test_choose();
    if (g_fail) {
        std::printf("heap_compact_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("heap_compact_test: ok\n");
    return 0;
}
