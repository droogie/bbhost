// Scaleform's font-handle set through the game's own code (tests/decomp/eboot_kit.h):
// engine/font_handle_set.cpp's removal by the handle's pointer, which
// engine/graphics_patch.cpp's hook on HashSet RemoveAlt (0x27d6a80) runs first,
// against RemoveAlt itself. Sets are built in the test's memory by the game's
// HashSet add (0x27d6250) from handles as FindOrCreateHandle makes them - a
// name (a String with its text inline), flags, a font with flags of its own -
// keyed by the game's hash (0x45e510, the name's, xor the flags' low two bits).
// Where every key has one entry (the stock game), every removal by pointer
// leaves the bytes RemoveAlt leaves, in any order, chains crossing slots. Where
// two handles share a key (a live Text scale change), RemoveAlt's removal of
// the older one takes the newer one's entry and leaves the older (freed) one in
// the set; ours takes its own. A key no entry holds is left to RemoveAlt, as
// the hook leaves it. Skips without the 1.09 eboot.
#include "engine/font_handle_set.h"

#include "eboot_kit.h"

#include <strings.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kHash = 0x45e510, kAdd = 0x27d6250, kRemoveAlt = 0x27d6a80;

using HashFn = std::uint64_t (*)(const char*, std::uint64_t, std::uint64_t);
using AddFn = void (*)(void*, void*, void*, std::uint64_t);
using RemoveFn = std::uint8_t (*)(void*, void*);

int g_failed = 0;
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_failed;                                             \
        }                                                           \
    } while (0)

int host_strcasecmp(const char* a, const char* b) { return strcasecmp(a, b); }

// A String as the handle names it: {size (nonzero: the text is inline), refs}, the text at +0xc.
struct Name {
    alignas(16) std::uint8_t bytes[64] = {};
    explicit Name(const std::string& s) {
        const std::uint64_t size = s.size() ? s.size() : 1;
        std::memcpy(bytes, &size, 8);
        std::memcpy(bytes + 0xc, s.c_str(), s.size() + 1);
    }
};
struct Font {
    alignas(16) std::uint8_t bytes[0x40] = {};
};
struct Handle {
    alignas(16) std::uint8_t bytes[0x40] = {};
};

struct World {
    std::vector<Name*> names;
    std::vector<Font*> fonts;
    std::vector<Handle*> handles;
    ~World() {
        for (Name* n : names) delete n;
        for (Font* f : fonts) delete f;
        for (Handle* h : handles) delete h;
    }
    Handle* make(const std::string& name, std::uint32_t flags, std::uint32_t font_flags) {
        Name* n = new Name(name);
        Font* f = new Font();
        Handle* h = new Handle();
        names.push_back(n);
        fonts.push_back(f);
        handles.push_back(h);
        std::memcpy(f->bytes + 0x18, &font_flags, 4);
        std::memcpy(h->bytes + 0x18, &flags, 4);
        const std::uint64_t name_ptr = reinterpret_cast<std::uint64_t>(n->bytes), font_ptr = reinterpret_cast<std::uint64_t>(f->bytes);
        std::memcpy(h->bytes + 0x20, &name_ptr, 8);
        std::memcpy(h->bytes + 0x30, &font_ptr, 8);
        return h;
    }
};

std::uint64_t key_hash(Handle* h) {
    std::uint64_t name_ptr = 0;
    std::memcpy(&name_ptr, h->bytes + 0x20, 8);
    const char* name = reinterpret_cast<const char*>(name_ptr + 0xc);
    std::uint32_t f = 0, g = 0;
    std::uint64_t font_ptr = 0;
    std::memcpy(&f, h->bytes + 0x18, 4);
    std::memcpy(&font_ptr, h->bytes + 0x30, 8);
    std::memcpy(&g, reinterpret_cast<const void*>(font_ptr + 0x18), 4);
    return eboot_kit::fn<HashFn>(kHash)(name, std::strlen(name), 0x1505) ^ ((f | g) & 3);
}

// A set: its table pointer, the table {count, mask 15, 16 entries} - room for 12
// handles before the game's add would grow it.
struct Set {
    std::vector<std::uint8_t> table;
    std::uint64_t ptr = 0;  // the set object: its first (only) word
    Set() : table(16 + 32 * 16, 0) {
        const std::uint64_t mask = 15;
        std::memcpy(table.data() + 8, &mask, 8);
        for (int i = 0; i < 16; ++i) {
            const std::int64_t empty = -2;
            std::memcpy(table.data() + 16 + 32 * i, &empty, 8);
        }
        ptr = reinterpret_cast<std::uint64_t>(table.data());
    }
    Set(const Set& o) : table(o.table) { ptr = reinterpret_cast<std::uint64_t>(table.data()); }
    void add(Handle* h) {
        void* hp = h->bytes;
        eboot_kit::fn<AddFn>(kAdd)(&ptr, &ptr, &hp, key_hash(h));
    }
    bool game_remove(Handle* h) {
        void* hp = h->bytes;
        return eboot_kit::fn<RemoveFn>(kRemoveAlt)(&ptr, &hp) != 0;
    }
    bool our_remove(Handle* h) { return font_set_remove_handle(table.data(), reinterpret_cast<std::uint64_t>(h->bytes)); }
    std::set<std::uint64_t> held() const {
        std::set<std::uint64_t> out;
        for (int i = 0; i < 16; ++i) {
            std::int64_t next = 0;
            std::uint64_t h = 0;
            std::memcpy(&next, table.data() + 16 + 32 * i, 8);
            std::memcpy(&h, table.data() + 16 + 32 * i + 16, 8);
            if (next != -2) out.insert(h);
        }
        return out;
    }
    std::uint64_t count() const {
        std::uint64_t c = 0;
        std::memcpy(&c, table.data(), 8);
        return c;
    }
};

const char* const kNames[] = {"MenuFont_01", "MenuFont_02", "TalkFont", "MenuFont_11", "Bold", "Title",
                              "SubTitle", "Ui_Small", "Ui_Large", "Num", "Glyphs", "Symbols"};

}  // namespace

int main() {
    eboot_kit::bytes("decomp_font_set_removal");
    eboot_kit::load("decomp_font_set_removal");
    if (!eboot_kit::bind("strcasecmp", reinterpret_cast<const void*>(&host_strcasecmp))) {
        std::printf("decomp_font_set_removal: no strcasecmp import to bind\n");
        return 1;
    }
    eboot_kit::Rng rng(0xf0a7);
    World w;
    // The stock game: one entry a key.
    int removals = 0;
    for (int round = 0; round < 300; ++round) {
        Set s;
        std::vector<Handle*> hs;
        const int n = rng.range(1, 12);
        for (int k = 0; k < n; ++k) {
            // Distinct keys: each name once a set (in either case), flags at random.
            std::string name = kNames[k];
            if (rng.chance(0.3)) name[0] = static_cast<char>(name[0] ^ 0x20);
            Handle* h = w.make(name, rng.u32() & 0x3ff, rng.u32() & 0x313);
            hs.push_back(h);
            s.add(h);
        }
        CHECK(s.count() == static_cast<std::uint64_t>(n));
        for (int k = n - 1; k > 0; --k) std::swap(hs[k], hs[rng.range(0, k)]);
        for (Handle* h : hs) {
            Set a = s, b = s;
            const bool ra = a.game_remove(h), rb = b.our_remove(h);
            CHECK(ra && rb);
            CHECK(a.table == b.table);
            s = b;
            ++removals;
        }
        CHECK(s.count() == 0);
    }
    // Two handles of one key: the older one's removal.
    int wrong = 0;
    for (int round = 0; round < 200; ++round) {
        Set s;
        const int others = rng.range(0, 8);
        for (int k = 0; k < others; ++k) s.add(w.make(kNames[1 + k], rng.u32() & 0x3ff, rng.u32() & 0x313));
        const std::uint32_t flags = rng.u32() & 0x3ff, font_flags = rng.u32() & 0x313;
        Handle* old_h = w.make("MenuFont_01", flags, font_flags);
        Handle* new_h = w.make("MenuFont_01", flags, font_flags);  // a live change's: same name, flags and font flags
        s.add(old_h);
        s.add(new_h);
        Set game = s, ours = s;
        CHECK(game.game_remove(old_h));
        CHECK(ours.our_remove(old_h));
        const std::uint64_t o = reinterpret_cast<std::uint64_t>(old_h->bytes), nw = reinterpret_cast<std::uint64_t>(new_h->bytes);
        wrong += game.held().count(o) == 1 && game.held().count(nw) == 0;  // the game's: the freed one kept
        CHECK(ours.held().count(o) == 0 && ours.held().count(nw) == 1);
        CHECK(ours.count() == static_cast<std::uint64_t>(others + 1));
        // The newer one, then the rest, by pointer: the game's removal agrees from here (one entry a key).
        Set a = ours, b = ours;
        CHECK(a.game_remove(new_h) && b.our_remove(new_h));
        CHECK(a.table == b.table);
        // A key no entry holds (the hook leaves it to RemoveAlt): ours changes nothing.
        Handle* stranger = w.make("MenuFont_01", flags, font_flags);
        Set c = b;
        CHECK(!c.our_remove(stranger) && c.table == b.table);
    }
    CHECK(wrong == 200);  // RemoveAlt took the newer handle's entry every time
    if (g_failed) {
        std::printf("decomp_font_set_removal: %d failed\n", g_failed);
        return 1;
    }
    std::printf("decomp_font_set_removal: ok - %d removals the same as the game's; with two of a key, the game's kept the freed "
                "handle %d of 200 times, ours never\n",
                removals, wrong);
    return 0;
}
