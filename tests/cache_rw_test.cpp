// Stage 4b.1a's lock (host/cache_rw.h): readers on eight threads find entries
// in a map one writer keeps inserting into and rewriting in place, as the
// command processor does with the caches a draw reads. No reader may see a
// torn entry or lose one it was told is there; the writer's nested writes take
// nothing more; a writer gets in while readers keep coming (it is preferred);
// each write moves its cache's generation.
//
// CACHE_RW_TEST_MUTANT=1 (a compile definition): the writer takes no lock, which
// this test must catch.
#include "host/cache_rw.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <unordered_map>
#include <vector>

// The pool's switch, on for the test (host/gxvk_lists_leaf.cpp's in the game).
namespace gxvk_lists {
int pool_threads() { return 2; }
}  // namespace gxvk_lists

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

// An entry whose fields say what it should hold: a torn read shows.
struct Entry {
    std::uint64_t key = 0, version = 0, check = 0;
    std::vector<std::uint32_t> words;  // version + 1 words, each the version
};
Entry make(std::uint64_t key, std::uint64_t version) {
    Entry e;
    e.key = key;
    e.version = version;
    e.check = key * 0x9e3779b97f4a7c15ull ^ version;
    e.words.assign(static_cast<std::size_t>(version % 23 + 1), static_cast<std::uint32_t>(version));
    return e;
}
bool whole(const Entry& e, std::uint64_t key) {
    if (e.key != key || e.check != (key * 0x9e3779b97f4a7c15ull ^ e.version)) return false;
    if (e.words.size() != e.version % 23 + 1) return false;
    for (const std::uint32_t w : e.words) {
        if (w != static_cast<std::uint32_t>(e.version)) return false;
    }
    return true;
}

std::unordered_map<std::uint64_t, Entry> g_map;
std::atomic<std::uint64_t> g_inserted{0};  // keys [0, n) are in the map once this reads n
std::atomic<bool> g_done{false};

void write_entry(std::uint64_t key, std::uint64_t version) {
#ifndef CACHE_RW_TEST_MUTANT
    cache_rw::Write cw(cache_rw::kPrograms);
    {
        cache_rw::Write nested(cache_rw::kPaths);  // a write inside a write takes nothing more
        CHECK(cache_rw::t_write_depth == 2);
    }
#endif
    g_map[key] = make(key, version);
}

void reader(std::atomic<std::uint64_t>* torn, std::atomic<std::uint64_t>* lost, std::atomic<std::uint64_t>* finds) {
    std::uint64_t seed = reinterpret_cast<std::uintptr_t>(&seed);
    while (!g_done.load(std::memory_order_acquire)) {
        const std::uint64_t n = g_inserted.load(std::memory_order_acquire);
        if (!n) continue;
        cache_rw::Read rd;
        for (int k = 0; k < 16; ++k) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const std::uint64_t key = (seed >> 33) % n;
            const auto it = g_map.find(key);
            if (it == g_map.end()) {
                lost->fetch_add(1, std::memory_order_relaxed);
            } else if (!whole(it->second, key)) {
                torn->fetch_add(1, std::memory_order_relaxed);
            }
            finds->fetch_add(1, std::memory_order_relaxed);
        }
    }
}

}  // namespace

int main() {
    CHECK(cache_rw::g_readers);
    std::atomic<std::uint64_t> torn{0}, lost{0}, finds{0};
    std::vector<std::thread> readers;
    for (int i = 0; i < 8; ++i) readers.emplace_back(reader, &torn, &lost, &finds);
    const std::uint64_t gen0 = cache_rw::gen(cache_rw::kPrograms);
    // The writer: 20,000 inserts (the map rehashes many times), each followed by
    // rewrites of an earlier entry in place (its words reallocated).
    constexpr std::uint64_t kKeys = 20000;
    std::uint64_t writes = 0;
    const auto t0 = std::chrono::steady_clock::now();
    double most_ms = 0;
    for (std::uint64_t key = 0; key < kKeys; ++key) {
        const auto w0 = std::chrono::steady_clock::now();
        write_entry(key, 0);
        ++writes;
        g_inserted.store(key + 1, std::memory_order_release);
        if (key > 8) {
            write_entry(key / 2, key);
            ++writes;
        }
        most_ms = std::max(most_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count());
    }
    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    g_done.store(true, std::memory_order_release);
    for (std::thread& t : readers) t.join();
    std::printf("cache_rw_test: %llu writes in %.1f ms (the slowest %.3f ms), %llu finds by 8 readers: %llu torn, %llu lost\n",
                static_cast<unsigned long long>(writes), total_ms, most_ms, static_cast<unsigned long long>(finds.load()),
                static_cast<unsigned long long>(torn.load()), static_cast<unsigned long long>(lost.load()));
    CHECK(finds.load() > 0);
    CHECK(torn.load() == 0);
    CHECK(lost.load() == 0);
#ifndef CACHE_RW_TEST_MUTANT
    CHECK(cache_rw::gen(cache_rw::kPrograms) - gen0 == writes);
    CHECK(cache_rw::t_write_depth == 0);
    const cache_rw::Counts c = cache_rw::counts();
    CHECK(c.writes == writes);  // the nested writes took nothing
    CHECK(c.reads > 0);
#endif
    // Every entry whole at the end.
    for (std::uint64_t key = 0; key < kKeys; ++key) {
        const auto it = g_map.find(key);
        CHECK(it != g_map.end() && whole(it->second, key));
    }
    // A reader's thread is marked while it holds the lock, and only then.
    CHECK(!cache_rw::reader());
    {
        cache_rw::Read rd;
        CHECK(cache_rw::reader());
        cache_rw::Write none(cache_rw::kViews);  // a reader's write guard takes nothing (it would wait for itself)
        CHECK(!none.held);
    }
    CHECK(!cache_rw::reader());
    if (g_fail) {
        std::printf("cache_rw_test: %d failures\n", g_fail);
        return 1;
    }
    std::printf("cache_rw_test: ok\n");
    return 0;
}
