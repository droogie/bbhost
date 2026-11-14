// font_set_remove_handle (engine/font_handle_set.h) against models of the game's
// own code for Scaleform's font-handle set: its add (0x27d6250: a new entry at
// its chain's head, the old head moved to a free slot; a slot held by another
// chain's entry handed back to that chain), its lookup (0x27d6980: the first
// entry of the chain with the key) and its removal (RemoveAlt, 0x27d6a80: the
// first entry that is the handle OR has its key). Checks: with one entry a key
// - the stock game - the removal by pointer leaves exactly the table the
// game's removal leaves; with two entries of one key (a live Text scale
// change) the game's removal takes the wrong one and leaves a freed handle in
// the set, and ours never does; and a long run of live changes, texts made and
// dropped, keeps the set to the handles alive, never growing past them.
#include "engine/font_handle_set.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace {

int g_failed = 0;
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_failed;                                             \
        }                                                           \
    } while (0)

struct Entry {
    std::int64_t next;
    std::uint64_t natural;
    std::uint64_t handle;
    std::uint8_t byte;
    std::uint8_t pad[7];
};

struct Table {
    std::vector<std::uint8_t> bytes;
    explicit Table(std::uint64_t mask) : bytes(16 + 32 * (mask + 1), 0) {
        std::memcpy(bytes.data() + 8, &mask, 8);
        for (std::uint64_t i = 0; i <= mask; ++i) e(i).next = -2;
    }
    std::uint64_t count() const {
        std::uint64_t c;
        std::memcpy(&c, bytes.data(), 8);
        return c;
    }
    void set_count(std::uint64_t c) { std::memcpy(bytes.data(), &c, 8); }
    std::uint64_t mask() const {
        std::uint64_t m;
        std::memcpy(&m, bytes.data() + 8, 8);
        return m;
    }
    Entry& e(std::uint64_t i) { return *reinterpret_cast<Entry*>(bytes.data() + 16 + 32 * i); }
    std::uint8_t* raw() { return bytes.data(); }
};

// A handle's key: its name's hash slot and the name itself (flags folded in).
std::map<std::uint64_t, std::pair<std::uint64_t, int>> g_keys;  // handle -> {hash, name id}
std::uint64_t slot_of(Table& t, std::uint64_t h) { return g_keys[h].first & t.mask(); }
bool same_key(std::uint64_t a, std::uint64_t b) { return g_keys[a].second == g_keys[b].second; }

// 0x27d6250, the table never full (no rehash).
void game_add(Table& t, std::uint64_t handle) {
    const std::uint64_t mask = t.mask(), natural = slot_of(t, handle);
    t.set_count(t.count() + 1);
    Entry& n = t.e(natural);
    if (n.next == -2) {
        n = Entry{-1, natural, handle, 0, {}};
        return;
    }
    std::uint64_t blank = (natural + 1) & mask;
    while (t.e(blank).next != -2) blank = (blank + 1) & mask;
    if (n.natural != natural) {
        // Another chain's entry in the slot: moved to the free one, its chain relinked.
        std::uint64_t p = n.natural;
        while (static_cast<std::uint64_t>(t.e(p).next) != natural) p = static_cast<std::uint64_t>(t.e(p).next);
        t.e(blank) = n;
        t.e(p).next = static_cast<std::int64_t>(blank);
        n = Entry{-1, natural, handle, 0, {}};
    } else {
        // The chain's head moved to the free slot; the new entry is the head.
        t.e(blank) = n;
        n.handle = handle;
        n.byte = 0;
        n.next = static_cast<std::int64_t>(blank);
        n.natural = natural;
    }
}

// 0x27d6980: the first entry of the key's chain with the key, or none.
bool game_lookup(Table& t, std::uint64_t key, std::uint64_t* out) {
    const std::uint64_t b = slot_of(t, key);
    if (t.e(b).next == -2 || t.e(b).natural != b) return false;
    for (std::uint64_t i = b;;) {
        if (t.e(i).natural == b && same_key(t.e(i).handle, key)) {
            *out = t.e(i).handle;
            return true;
        }
        if (t.e(i).next == -1) return false;
        i = static_cast<std::uint64_t>(t.e(i).next);
    }
}

// 0x27d6a80: the first entry of the chain that is the handle or has its key.
bool game_remove(Table& t, std::uint64_t key) {
    const std::uint64_t b = slot_of(t, key);
    if (t.e(b).next == -2 || t.e(b).natural != b) return false;
    std::int64_t prev = -1;
    std::uint64_t i = b;
    for (;;) {
        const Entry& x = t.e(i);
        if (x.natural == b && (x.handle == key || same_key(x.handle, key))) break;
        if (x.next == -1) return false;
        prev = static_cast<std::int64_t>(i);
        i = static_cast<std::uint64_t>(x.next);
    }
    Entry& x = t.e(i);
    if (i == b) {
        if (x.next == -1) {
            x.next = -2;
        } else {
            Entry& n = t.e(static_cast<std::uint64_t>(x.next));
            x.next = n.next;
            x.natural = n.natural;
            x.handle = n.handle;
            x.byte = n.byte;
            n.next = -2;
        }
    } else {
        t.e(static_cast<std::uint64_t>(prev)).next = x.next;
        x.next = -2;
    }
    t.set_count(t.count() - 1);
    return true;
}

std::set<std::uint64_t> handles_in(Table& t) {
    std::set<std::uint64_t> out;
    for (std::uint64_t i = 0; i <= t.mask(); ++i)
        if (t.e(i).next != -2) out.insert(t.e(i).handle);
    return out;
}

std::uint64_t g_next_handle = 0x1000;
std::uint64_t new_handle(std::uint64_t hash, int name) {
    const std::uint64_t h = g_next_handle;
    g_next_handle += 0x40;
    g_keys[h] = {hash, name};
    return h;
}

// One key an entry (the stock game): every handle removed by pointer leaves
// the table the game's removal leaves, in any order, chains crossing slots.
void test_stock_identical() {
    std::mt19937 rng(7);
    for (int round = 0; round < 200; ++round) {
        Table t(15);
        std::vector<std::uint64_t> hs;
        const int n = 1 + static_cast<int>(rng() % 12);
        for (int k = 0; k < n; ++k) {
            const std::uint64_t h = new_handle(rng() % 6, 1000 * round + k);  // few slots: chains and handed-back slots
            hs.push_back(h);
            game_add(t, h);
        }
        std::shuffle(hs.begin(), hs.end(), rng);
        for (const std::uint64_t h : hs) {
            Table a = t, b = t;
            const bool ra = game_remove(a, h);
            const bool rb = font_set_remove_handle(b.raw(), h);
            CHECK(ra && rb);
            CHECK(a.bytes == b.bytes);
            t = b;
        }
        CHECK(t.count() == 0);
    }
}

// Two handles of one key - the old one and the new one a live change made,
// added at the head: the game's removal of the old one unlinks the new one's
// entry and keeps the old (freed) one; ours removes the old one's.
void test_duplicates() {
    {
        Table t(7);
        const std::uint64_t old_h = new_handle(3, 1), new_h = new_handle(3, 1);
        game_add(t, old_h);
        game_add(t, new_h);
        std::uint64_t found = 0;
        CHECK(game_lookup(t, new_h, &found) && found == new_h);  // the newest first
        Table g = t;
        CHECK(game_remove(g, old_h));
        CHECK(handles_in(g) == std::set<std::uint64_t>{old_h});  // the bug: the freed handle stays
        CHECK(font_set_remove_handle(t.raw(), old_h));
        CHECK(handles_in(t) == std::set<std::uint64_t>{new_h});
        CHECK(t.count() == 1);
        CHECK(game_lookup(t, new_h, &found) && found == new_h);
        CHECK(font_set_remove_handle(t.raw(), new_h));
        CHECK(t.count() == 0 && handles_in(t).empty());
    }
    {
        // The new one (the head) removed first: the old one moves into the head.
        Table t(7);
        const std::uint64_t other = new_handle(3, 9), old_h = new_handle(3, 1), new_h = new_handle(3, 1);
        game_add(t, other);
        game_add(t, old_h);
        game_add(t, new_h);
        CHECK(font_set_remove_handle(t.raw(), new_h));
        CHECK(handles_in(t) == (std::set<std::uint64_t>{other, old_h}));
        std::uint64_t found = 0;
        CHECK(game_lookup(t, old_h, &found) && found == old_h);
        CHECK(game_lookup(t, other, &found) && found == other);
    }
    {
        // A key no entry holds: nothing changes (the game's own removal runs).
        Table t(7);
        const std::uint64_t a = new_handle(2, 5), stranger = new_handle(2, 5);
        game_add(t, a);
        const std::vector<std::uint8_t> before = t.bytes;
        CHECK(!font_set_remove_handle(t.raw(), stranger));
        CHECK(t.bytes == before);
        CHECK(!font_set_remove_handle(nullptr, a) && !font_set_remove_handle(t.raw(), 0));
    }
}

// A movie's set over a long run of live changes: texts made at the scale the
// map has now (a handle reused only at that scale, else a new one added, as
// FindOrCreateHandle does), dropped in random order, each handle removed as it
// is freed. With the removal by pointer the set always holds exactly the
// handles alive; with the game's removal it ends up holding a freed one.
int long_run(bool ours, int changes) {
    std::mt19937 rng(11);
    Table t(63);
    std::map<std::uint64_t, int> refs;         // handle -> its texts
    std::map<std::uint64_t, float> scale_of;   // handle -> its scale
    std::vector<std::uint64_t> texts;          // each text's handle
    std::set<std::uint64_t> freed;
    const std::uint64_t key_proto = new_handle(5, 77);  // MenuFont_01's key
    float scale = 0.75f;
    int freed_in_set = 0;
    std::uint64_t most = 0;
    // Other fonts' handles, alive throughout, crossing the same slots.
    for (int k = 0; k < 6; ++k) game_add(t, new_handle(static_cast<std::uint64_t>(k), 200 + k));
    for (int step = 0; step < changes * 40; ++step) {
        if (step % 40 == 0) scale = 0.5f + 0.05f * static_cast<float>(rng() % 11);  // a live change
        if (rng() % 2 == 0 || texts.empty()) {
            std::uint64_t h = 0;
            if (!game_lookup(t, key_proto, &h) || scale_of[h] != scale) {
                h = new_handle(5, 77);
                scale_of[h] = scale;
                game_add(t, h);
            }
            ++refs[h];
            texts.push_back(h);
        } else {
            const std::size_t i = rng() % texts.size();
            const std::uint64_t h = texts[i];
            texts.erase(texts.begin() + static_cast<std::ptrdiff_t>(i));
            if (--refs[h] == 0) {
                refs.erase(h);
                freed.insert(h);
                if (ours)
                    font_set_remove_handle(t.raw(), h);
                else
                    game_remove(t, h);
            }
        }
        for (const std::uint64_t h : handles_in(t)) freed_in_set += freed.count(h) != 0;
        most = std::max<std::uint64_t>(most, t.count());
        if (ours) {
            std::set<std::uint64_t> alive;
            for (const auto& [h, n] : refs) alive.insert(h);
            std::set<std::uint64_t> in = handles_in(t);
            int key_entries = 0;
            for (const std::uint64_t h : in) key_entries += g_keys[h].second == 77;
            CHECK(key_entries == static_cast<int>(alive.size()));  // MenuFont_01's entries: its live handles, no more
            CHECK(t.count() == alive.size() + 6);
        }
    }
    if (ours) {
        CHECK(freed_in_set == 0);
        std::printf("font_handle_set_test: %d live changes, the set at most %llu handles (6 of other fonts)\n", changes,
                    static_cast<unsigned long long>(most));
    }
    return freed_in_set;
}

}  // namespace

int main() {
    test_stock_identical();
    test_duplicates();
    long_run(true, 60);
    CHECK(long_run(false, 60) > 0);  // the game's own removal: the use-after-free this fixes
    if (g_failed) {
        std::printf("font_handle_set_test: %d failed\n", g_failed);
        return 1;
    }
    std::printf("font_handle_set_test: ok\n");
    return 0;
}
