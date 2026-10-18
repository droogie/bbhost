// The buffer shadow gathers its mirror pages' copies into runs for one
// vkCmdCopyBuffer (host/copy_runs.h). With window imports each page is copied
// into upload staging first, so the sources of two copies follow one another
// even when valid pages lie between their destinations: a run may grow only
// where both ends meet (on lavapipe the Dream drew its meshes from the wrong
// bytes when it grew on the source alone).
#include "host/copy_runs.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

struct Region {  // VkBufferCopy's members
    std::uint64_t srcOffset, dstOffset, size;
};

constexpr std::uint64_t kPage = 64 << 10;

// The copy loop's way of adding one page: grow the last run, else a new one.
void add(std::vector<Region>& runs, std::uint64_t src, std::uint64_t dst) {
    if (!runs.empty() && copy_run_extends(runs.back(), src, dst, kPage)) return;
    runs.push_back({src, dst, kPage});
}

// Copies as the GPU would run them: page index of destination -> page index of source.
std::vector<int> land(const std::vector<Region>& runs, int pages) {
    std::vector<int> from(pages, -1);
    for (const Region& r : runs) {
        for (std::uint64_t k = 0; k < r.size / kPage; ++k) from[(r.dstOffset / kPage) + k] = static_cast<int>(r.srcOffset / kPage + k);
    }
    return from;
}

void test_staged_skip() {
    // Pages 0, 2 and 3 copied, page 1 valid (skipped): staging hands out
    // placements 0, 1, 2 one after another.
    std::vector<Region> runs;
    add(runs, 0 * kPage, 0 * kPage);
    add(runs, 1 * kPage, 2 * kPage);
    add(runs, 2 * kPage, 3 * kPage);
    CHECK(runs.size() == 2);
    const std::vector<int> from = land(runs, 4);
    CHECK(from[0] == 0);
    CHECK(from[1] == -1);  // the valid page is not written
    CHECK(from[2] == 1);
    CHECK(from[3] == 2);
}

void test_import_runs() {
    // From an import that starts at the chunk's byte 3 pages in: the source
    // is the page's own offset less the import's start, both ends follow on.
    std::vector<Region> runs;
    const std::uint64_t lo = 3 * kPage;
    for (std::uint64_t p = 5; p <= 9; ++p) {
        if (p == 7) continue;  // valid
        add(runs, p * kPage - lo, p * kPage);
    }
    CHECK(runs.size() == 2);
    CHECK(runs[0].size == 2 * kPage && runs[1].size == 2 * kPage);
    const std::vector<int> from = land(runs, 10);
    CHECK(from[5] == 2 && from[6] == 3 && from[7] == -1 && from[8] == 5 && from[9] == 6);
}

void test_staging_chunk_change() {
    // A new staging chunk starts again at offset 0: no run joins across it
    // unless both ends follow on, which they do not here.
    std::vector<Region> runs;
    add(runs, 63 * kPage, 10 * kPage);
    add(runs, 0, 11 * kPage);
    CHECK(runs.size() == 2);
}

}  // namespace

int main() {
    test_staged_skip();
    test_import_runs();
    test_staging_chunk_change();
    if (g_fail) return 1;
    std::printf("copy_runs_test: ok\n");
    return 0;
}
