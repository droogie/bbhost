// The list arena's memo (host/list_cb_memo.h), without the game: a copy made,
// the guest's bytes at its address rewritten - a rename carved where the last
// one was, as GX does once the submission that used its block has run - and a
// later draw of the same address and flip, its segment untaken, asking for it.
// It must be refused (the old rule, the address and the flip alone, bound it),
// and bound while the bytes are the ones the copy took: at the buffer sizes
// the stale shares had at the Hunter's Dream and at odd ones, with any one
// byte of the buffer changed, and whatever lies after the buffer in the lines
// the copy ran on into. Another address, flip or segment take, or a binding
// longer than the copy, is another copy.
#include "host/list_cb_memo.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;
void check(bool ok, const char* what, unsigned size, unsigned at = 0) {
    if (ok) return;
    if (++g_failures <= 24) std::printf("FAIL: %s (%u bytes, at %u)\n", what, size, at);
}

std::uint64_t g_rng = 0x2545f4914f6cdd1dull;
std::uint32_t next() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return static_cast<std::uint32_t>(g_rng);
}

using list_cb_memo::Share;

}  // namespace

int main() {
    // The sizes the stale shares had (864, 768, 416, 84, 24, 20, 16, 64), the ones around the hash's steps, and odd ones.
    const unsigned sizes[] = {1, 3, 4, 7, 8, 16, 20, 24, 31, 32, 33, 64, 84, 96, 255, 416, 768, 864, 1000, 4096, 65536};
    std::uint64_t lap_now[16] = {};
    for (int s = 0; s < 16; ++s) lap_now[s] = 3 + s;
    const auto lap_of = [&](std::uint32_t seg) { return lap_now[seg]; };
    int cases = 0;
    for (unsigned size : sizes) {
        const unsigned lines = (size + 63) & ~63u;  // the copy runs on to its last line's end
        std::vector<unsigned char> mem(lines + 64);
        for (auto& b : mem) b = static_cast<unsigned char>(next());
        const std::uint64_t base = reinterpret_cast<std::uintptr_t>(mem.data());
        const std::uint32_t seg = next() % 16;
        const list_cb_memo::Entry e = list_cb_memo::make(base, lines, size, 1234, lap_now[seg], seg, 4096);
        // The same bytes: bound.
        check(list_cb_memo::shares(e, base, size, lines, 1234, lap_of) == Share::kYes, "the same bytes not shared", size);
        // A shorter binding of the same buffer, its bytes the same: bound.
        if (size > 1) check(list_cb_memo::shares(e, base, size - 1, lines, 1234, lap_of) == Share::kYes, "a shorter binding not shared", size);
        // Another flip, address or segment take, or a binding the copy does not cover: another copy.
        check(list_cb_memo::shares(e, base, size, lines, 1235, lap_of) == Share::kNo, "another flip shared", size);
        check(list_cb_memo::shares(e, base + 16, size, lines, 1234, lap_of) == Share::kNo, "another address shared", size);
        check(list_cb_memo::shares(e, base, size + 1, lines, 1234, lap_of) == Share::kNo, "a longer binding shared", size);
        check(list_cb_memo::shares(e, base, size, lines + 64, 1234, lap_of) == Share::kNo, "a longer copy shared", size);
        ++lap_now[seg];
        check(list_cb_memo::shares(e, base, size, lines, 1234, lap_of) == Share::kNo, "a segment taken again shared", size);
        --lap_now[seg];
        // What lies after the buffer in its last line may change: still the buffer's bytes, bound.
        for (unsigned k = size; k < lines; ++k) mem[k] ^= 0x5a;
        check(list_cb_memo::shares(e, base, size, lines, 1234, lap_of) == Share::kYes, "a change past the buffer refused", size);
        // A new rename at the address: every byte rewritten, then any one byte - refused, as kChanged.
        std::vector<unsigned char> keep(mem.begin(), mem.begin() + size);
        for (unsigned k = 0; k < size; ++k) mem[k] = static_cast<unsigned char>(next());
        if (std::memcmp(keep.data(), mem.data(), size) != 0)
            check(list_cb_memo::shares(e, base, size, lines, 1234, lap_of) == Share::kChanged, "a new rename's bytes shared", size);
        std::memcpy(mem.data(), keep.data(), size);
        for (unsigned trial = 0; trial < 64; ++trial) {
            const unsigned at = next() % size;
            const unsigned char was = mem[at];
            mem[at] = static_cast<unsigned char>(was ^ (1u << (next() % 8)));
            check(list_cb_memo::shares(e, base, size, lines, 1234, lap_of) == Share::kChanged, "a changed byte shared", size, at);
            mem[at] = was;
            ++cases;
        }
        check(list_cb_memo::shares(e, base, size, lines, 1234, lap_of) == Share::kYes, "the bytes put back not shared", size);
    }
    if (g_failures) {
        std::printf("list_cb_memo_test: %d failures\n", g_failures);
        return 1;
    }
    std::printf("list_cb_memo_test: ok (%zu sizes, %d changed bytes refused)\n", sizeof(sizes) / sizeof(sizes[0]), cases);
    return 0;
}
