// Window imports let go while idle (host/import_retire.h): the switch, which
// imports a check finds idle - both views naming them, no window used within
// the time, a window not stamped yet counted as used now - and the check's cap.
#include "host/import_retire.h"

#include <atomic>
#include <cstdint>
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

struct Imp {
    int id = 0;
};

constexpr std::uint32_t kN = 64;

struct Chunk {
    Imp* window[kN] = {};
    Imp* table[kN] = {};
    std::atomic<std::uint64_t> used[kN] = {};
    void cover(std::uint32_t a, std::uint32_t b, Imp* c, std::uint64_t at) {
        for (std::uint32_t w = a; w <= b; ++w) {
            window[w] = table[w] = c;
            used[w].store(at);
        }
    }
    bool idle(std::uint32_t a, std::uint32_t b, const Imp* c, std::uint64_t now, std::uint64_t after = 60000) {
        return import_retire::idle(window, table, used, a, b, c, now, after);
    }
};

void test_parse() {
    using import_retire::parse;
    import_retire::Switch s = parse(nullptr, nullptr, nullptr);
    CHECK(s.mode == import_retire::kOff && s.after_ms == 60000 && s.per_check == (128ull << 20));
    CHECK(parse("1", nullptr, nullptr).mode == import_retire::kOn);
    CHECK(parse("2", nullptr, nullptr).mode == import_retire::kCount);
    CHECK(parse("0", nullptr, nullptr, import_retire::kOn).mode == import_retire::kOff);  // the switch over the default
    CHECK(parse(nullptr, nullptr, nullptr, import_retire::kOn).mode == import_retire::kOn);
    CHECK(parse("", nullptr, nullptr, import_retire::kOn).mode == import_retire::kOn);
    CHECK(parse("12", nullptr, nullptr).mode == import_retire::kOff);  // not a mode: the default
    CHECK(parse("yes", nullptr, nullptr, import_retire::kCount).mode == import_retire::kCount);
    CHECK(parse("1", "30", nullptr).after_ms == 30000);
    CHECK(parse("1", "2.5", nullptr).after_ms == 2500);
    CHECK(parse("1", "0.5", nullptr).after_ms == 60000);  // under a second: the default
    CHECK(parse("1", "x", nullptr).after_ms == 60000);
    CHECK(parse("1", nullptr, "64").per_check == (64ull << 20));
    CHECK(parse("1", nullptr, "8").per_check == (128ull << 20));  // under 16 MiB: the default
}

void test_idle() {
    Imp c{1};
    Chunk k;
    k.cover(3, 5, &c, 1000);
    CHECK(!k.idle(3, 5, &c, 1000 + 59999));
    CHECK(k.idle(3, 5, &c, 1000 + 60000));
    CHECK(k.idle(3, 5, &c, 1000 + 600000));
    // One window used later keeps the whole import.
    k.used[4].store(50000);
    CHECK(!k.idle(3, 5, &c, 1000 + 60000));
    CHECK(k.idle(3, 5, &c, 50000 + 60000));
    // The time of its own.
    CHECK(!k.idle(3, 5, &c, 50000 + 1000, 2000));
    CHECK(k.idle(3, 5, &c, 50000 + 2000, 2000));
    // A clock behind a stamp (no wrap to a huge age).
    Chunk late;
    late.cover(3, 5, &c, 50000);
    CHECK(!late.idle(3, 5, &c, 40000));
}

void test_views() {
    Imp c{1}, older{2}, newer{3};
    Chunk k;
    k.cover(10, 12, &c, 1000);
    CHECK(k.idle(10, 12, &c, 100000));
    // A window whose page-table entries still name an older import (a merge the queue has not seen): not yet.
    k.table[11] = &older;
    CHECK(!k.idle(10, 12, &c, 100000));
    // Entries naming nothing yet (the sink) are fine.
    k.table[11] = nullptr;
    CHECK(k.idle(10, 12, &c, 100000));
    // A window another import took over: this one is a merge's older view, not the window's.
    k.window[12] = &newer;
    CHECK(!k.idle(10, 12, &c, 100000));
    // The older import of a merge, named only by the page-table view.
    Chunk m;
    m.cover(0, 1, &newer, 1000);
    m.table[0] = &older;
    CHECK(!m.idle(0, 1, &older, 100000));
    // An empty range never.
    CHECK(!k.idle(5, 4, &c, 100000));
}

void test_unstamped() {
    Imp c{1};
    Chunk k;
    k.cover(0, 2, &c, 1000);
    k.used[1].store(0);  // made before the clock's first tick
    CHECK(!k.idle(0, 2, &c, 100000));
    CHECK(k.used[1].load() == 100000);  // stamped now: its time starts
    CHECK(!k.idle(0, 2, &c, 100000 + 59999));
    CHECK(k.idle(0, 2, &c, 100000 + 60000));
    // Every unstamped window of an import is stamped by one check, not only the first.
    Chunk m;
    m.cover(0, 3, &c, 0);
    CHECK(!m.idle(0, 3, &c, 5000));
    for (std::uint32_t w = 0; w <= 3; ++w) CHECK(m.used[w].load() == 5000);
}

void test_fits() {
    CHECK(import_retire::fits(0, 128ull << 20));
    CHECK(import_retire::fits((128ull << 20) - 1, 128ull << 20));
    CHECK(!import_retire::fits(128ull << 20, 128ull << 20));
    // The first import fits whatever its size: a check lets go of at least one.
    std::uint64_t so_far = 0, n = 0;
    const std::uint64_t sizes[] = {512ull << 20, 16ull << 20, 16ull << 20};
    for (const std::uint64_t s : sizes) {
        if (!import_retire::fits(so_far, 128ull << 20)) continue;
        so_far += s;
        ++n;
    }
    CHECK(n == 1 && so_far == (512ull << 20));
}

}  // namespace

int main() {
    test_parse();
    test_idle();
    test_views();
    test_unstamped();
    test_fits();
    if (g_fail) {
        std::printf("import_retire_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("import_retire_test: ok\n");
    return 0;
}
