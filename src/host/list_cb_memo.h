// The list arena's memo (stage 4b.1b/4b.2's draw pool, render.cpp
// resolve_stage_buffers): a list-mode draw's constant buffer that is copied
// into the pool's arena - a GX upload-heap rename or a table in the GX ring -
// is bound again by a later draw of the same address whose GX call saw the
// same count of flips done, while the guest's bytes are the ones the copy
// took. The address and the flip alone do not say so: a renamed copy's bytes
// are final at its call, but GX takes an upload-heap block back as soon as
// the submission that used it has run and carves the next renames where the
// last ones were, and the count of flips done can stand still from one of the
// game's frames to the next - a draw then bound the frame before's constants.
// So each copy keeps a hash of the buffer's bytes as copied, and a share asks
// it of the guest's bytes now. tests/list_cb_memo_test.cpp drives the rule
// without the game.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>

namespace list_cb_memo {

// A hash of the guest's bytes [va, +n), 32 bytes a step in four lanes.
inline std::uint64_t sum(std::uint64_t va, std::uint64_t n) {
    const auto* p = reinterpret_cast<const unsigned char*>(static_cast<std::uintptr_t>(va));
    constexpr std::uint64_t k1 = 0x9e3779b97f4a7c15ull, k2 = 0xc2b2ae3d27d4eb4full;
    std::uint64_t h[4] = {n ^ k1, n ^ k2, ~n, n * k1};
    std::uint64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        std::uint64_t w[4];
        std::memcpy(w, p + i, sizeof(w));
        for (int l = 0; l < 4; ++l) {
            h[l] = (h[l] ^ w[l]) * k1;
            h[l] ^= h[l] >> 32;
        }
    }
    std::uint64_t r = h[0] ^ std::rotl(h[1], 16) ^ std::rotl(h[2], 32) ^ std::rotl(h[3], 48);
    for (; i + 8 <= n; i += 8) {
        std::uint64_t w;
        std::memcpy(&w, p + i, sizeof(w));
        r = (r ^ w) * k2;
        r ^= r >> 29;
    }
    for (; i < n; ++i) r = (r ^ p[i]) * k1;
    r ^= r >> 33;
    r *= k2;
    return r ^ (r >> 29);
}

// One copy: the guest address its bytes came from and how many it copied (the
// buffer's, run on to the end of its last 64-byte line), its GX call's flip,
// the take of the arena segment it went into, that segment and its place in
// the arena, and the hash of the buffer's own bytes as copied with their count.
struct Entry {
    std::uint64_t base = 0, bytes = 0, flip = 0, lap = 0;
    std::uint32_t seg = 0;
    std::uint64_t at = 0;
    std::uint64_t sum = 0, summed = 0;
};

// The entry for a copy of `buffer_bytes` at `base` (`copy_bytes` copied) just made.
inline Entry make(std::uint64_t base, std::uint64_t copy_bytes, std::uint64_t buffer_bytes, std::uint64_t flip, std::uint64_t lap,
                  std::uint32_t seg, std::uint64_t at) {
    return Entry{base, copy_bytes, flip, lap, seg, at, sum(base, buffer_bytes), buffer_bytes};
}

enum class Share { kNo, kChanged, kYes };

// Whether a binding of `bytes` at `base` (`copy_bytes` to copy) in a draw
// whose GX call saw `flip` binds the entry's copy: kNo when the entry is
// another copy (another address, fewer bytes, another flip, or its segment
// taken again since - `lap_of(seg)` is the segment's take now), kChanged when
// it is this copy but the guest's bytes are no longer the ones it took, kYes
// to bind it.
template <typename LapOf>
Share shares(const Entry& e, std::uint64_t base, std::uint64_t bytes, std::uint64_t copy_bytes, std::uint64_t flip, LapOf&& lap_of) {
    if (e.base != base || e.bytes < copy_bytes || bytes > e.summed || e.flip != flip || lap_of(e.seg) != e.lap) return Share::kNo;
    return sum(base, e.summed) == e.sum ? Share::kYes : Share::kChanged;
}

}  // namespace list_cb_memo
