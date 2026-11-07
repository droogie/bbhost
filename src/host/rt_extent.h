#pragma once
// The real size of a render target (render.cpp target_image). CB_COLOR*_PITCH
// and CB_COLOR*_SLICE (and DB_DEPTH_SIZE for depth) carry only the
// tile-padded size, but a target sampled as a texture is addressed in
// normalized coordinates over its image: a 392-wide target held in a 512-wide
// image makes every u 392/512 too small, and down a bloom pyramid that
// compounds. What says the size, in order: the texture of the draw's view
// (GpuDrawInputs::color_extent); the live texture the registry has at the
// target's base, GX's (the T# GX built) or YEBIS's (the T# its device built,
// when its render target is the one the draw's registers describe:
// registered_fits); then what the target's scissors taught (Table). A scissor
// edge is believed when padding it to a tiling granularity gives exactly the
// padded size (plausible); an edge equal to the padded size says nothing,
// since many passes scissor to the whole surface.
//
// YEBIS's draws bind no view, and the registry gave no size for its textures,
// so every YEBIS target was sized from its scissors - and those sizes were
// kept by address alone, while YEBIS gives one address to targets of another
// geometry. At 2560x1440 its other glare pyramid (the one the Hunter's Dream
// draws: 392x331 at 1920x1080) starts 523x441, at addresses its glare later
// draws 640x360 levels into - a 640-pixel pitch and 384 rows at the Nightmare
// of Mensis. Such a level scissors to its whole pitch, 640, which may not
// replace a learned size, so the 523 stayed for the session: the next pass
// read the 523-wide image through a 640-wide T#, stretching it by 640/523,
// twice down the chain - the sky through a window, bright at x 217 of 640,
// glared at 264 and then 322, a soft bright rectangle to the right of it. At
// 1920x1080 the level is 480 in a 512 pitch and its scissor replaced the stale
// width; the stale heights left on its smaller levels there came out larger
// than the levels, and a T# smaller than its target reads its corner - which
// is why only 1440p showed it. YEBIS's targets now take the size of the
// texture YEBIS made (yebis_texture), and a learned size belongs to the
// geometry it was learned under: the target's base, pitch and padded height
// (Key). Two targets of one geometry at one address still share a learned
// size, and a scissor the padded size still cannot replace it there
// (Step::padded_over, counted in the exit report). Pure;
// tests/rt_extent_test.cpp checks it.

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#ifdef RT_EXTENT_MUTANTS
extern int g_rt_extent_mutant;
#define RT_EXTENT_MUTANT(n) (g_rt_extent_mutant == (n))
#else
#define RT_EXTENT_MUTANT(n) false
#endif

namespace rt_extent {

// True when `v` rounded up to a tiling granularity is exactly `padded`: that is
// what makes a scissor edge believable as the target's real extent rather than
// a pass drawing into part of a larger one. Colour pitch pads to 128 or 256
// pixels (392 -> 512, 960 -> 1024, 1176 -> 1280) and height to 64 rows
// (1080 -> 1088).
inline bool plausible(std::uint32_t v, std::uint32_t padded) {
    if (!v || v > padded) return false;
    for (std::uint32_t g : {64u, 128u, 256u}) {
        if (((v + g - 1) & ~(g - 1)) == padded) return true;
    }
    return false;
}

// One dimension's answer: the width over the pitch or the height over the
// padded height.
struct Step {
    std::uint32_t extent = 0;  // the size to make the image
    bool learned = false;      // `known` changed: answers given before are old
    bool padded_over = false;  // a scissor the padded size over a smaller learned size, which stays
};

// `known`: what was learned for this dimension of the target (0: nothing).
// A size once learned stands against smaller scissors - a later pass that
// draws into part of the target cannot resize its image, which costs a
// re-creation and the contents. A larger scissor that is plausible replaces
// it: a pass cannot draw past its target. An edge equal to the padded size
// neither teaches nor replaces.
inline Step step(std::uint32_t& known, std::uint32_t padded, std::uint32_t edge) {
    if (plausible(known, padded) || (RT_EXTENT_MUTANT(4) && known)) {
        if (edge > known && (edge != padded || RT_EXTENT_MUTANT(2)) && plausible(edge, padded) && !RT_EXTENT_MUTANT(5)) {
            known = edge;
            return {known, true, false};
        }
        return {known, false, edge > known && edge == padded && !RT_EXTENT_MUTANT(6)};
    }
    if (plausible(edge, padded) && (edge != padded || RT_EXTENT_MUTANT(3))) {
        known = edge;
        return {known, true, false};
    }
    return {padded, false, false};
}

// A target's geometry as its registers give it: where it starts and its padded
// size (the pitch and CB_COLOR*_SLICE / pitch; DB_DEPTH_SIZE's width and height).
struct Key {
    std::uint64_t base = 0;
    std::uint32_t pitch = 0, padded_h = 0;
    bool operator==(const Key&) const = default;
};
struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
        const std::uint64_t g = (static_cast<std::uint64_t>(k.pitch) << 32) | k.padded_h;
        std::uint64_t h = k.base * 0x9e3779b97f4a7c15ull ^ g * 0xc2b2ae3d27d4eb4full;
        h ^= h >> 29;
        return static_cast<std::size_t>(h);
    }
};

// A draw's answer for both dimensions, and what the census needs of it.
struct Answer {
    std::uint32_t w = 0, h = 0;
    bool changed = false;                            // a size learned (target_image's memo is old)
    bool padded_over_w = false, padded_over_h = false;  // Step::padded_over, per dimension
    std::uint32_t known_w = 0, known_h = 0;          // what was learned before this draw (0: nothing)
};

// Every target's learned sizes. by_geometry false keys them by the base alone,
// the rule before (BBHOST_RT_LEARN_BY_GEOMETRY=0).
class Table {
public:
    explicit Table(bool by_geometry = true) : by_geometry_(by_geometry) {}

    // A draw (or a clear, with no scissor: 0) into the target at `base` with
    // that pitch and padded height: the size to make its image.
    Answer draw(std::uint64_t base, std::uint32_t pitch, std::uint32_t padded_h, std::uint32_t scissor_right,
                std::uint32_t scissor_bottom) {
        const Key k = key(base, pitch, padded_h);
        const auto it = map_.find(k);
        Sizes s = it != map_.end() ? it->second : Sizes{};
        Answer a;
        a.known_w = s.w;
        a.known_h = s.h;
        const Step sw = step(s.w, pitch, scissor_right), sh = step(s.h, padded_h, scissor_bottom);
        a.w = sw.extent;
        a.h = sh.extent;
        a.padded_over_w = sw.padded_over;
        a.padded_over_h = sh.padded_over;
        a.changed = sw.learned || sh.learned;
        if (a.changed) map_[k] = s;  // (nothing stored for a draw that teaches nothing)
        return a;
    }

    // A texture says the target's size (its view's, or the registry's live
    // texture at its base): what a draw with no view and no registered texture
    // then makes the same target. True when that changed what was known.
    bool view(std::uint64_t base, std::uint32_t pitch, std::uint32_t padded_h, std::uint32_t w, std::uint32_t h) {
        Sizes& s = map_[key(base, pitch, padded_h)];
        if (s.w == w && s.h == h) return false;
        s = Sizes{w, h};
        return true;
    }

    std::size_t size() const { return map_.size(); }

private:
    struct Sizes {
        std::uint32_t w = 0, h = 0;  // 0: not learned
    };
    Key key(std::uint64_t base, std::uint32_t pitch, std::uint32_t padded_h) const {
        if (!by_geometry_) return Key{base, 0, 0};
        if (RT_EXTENT_MUTANT(1)) return Key{base, pitch, 0};  // the padded height left out of the key
        return Key{base, pitch, padded_h};
    }
    bool by_geometry_;
    std::unordered_map<Key, Sizes, KeyHash> map_;
};

// A YEBIS 2D texture as its object holds it (YEBIS's CreateTexture2D,
// 0x15df570, and its 0x15df840; the registry reads it, gx_resources.cpp): the
// T# its device built at +0xc4 (sub_1487ee0, from the description's width and
// height) - its base in 256-byte units in word 0 (set when the memory is
// handed out, rebased by a discard Map, zeroed by the destructor's free
// 0x2e4b230), width - 1 and height - 1 in word 2, the surface's pitch - 1 in
// word 4 (bits 13-26). (The sce::Gnm render target at +0x108 is a depth
// target's only: a colour target's comes with its view.)
constexpr std::uint32_t kYebisTsharp = 0xc4;
struct YebisTexture {
    std::uint64_t memory = 0;   // the T#'s base
    std::uint32_t w = 0, h = 0;  // the T#'s size
    std::uint32_t pitch = 0;     // the T#'s pitch: the surface's padded width
};
inline YebisTexture yebis_texture(const std::uint32_t (&t)[8]) {
    YebisTexture y;
    y.memory = static_cast<std::uint64_t>(t[0]) << 8;
    y.w = (t[2] & 0x3fff) + 1;
    y.h = ((t[2] >> 14) & 0x3fff) + 1;
    y.pitch = ((t[4] >> 13) & 0x3fff) + 1;
    return y;
}
// Whether the object at a registered texture's address is still that live
// texture: its vtable the 2D texture's (the destructor's base classes put
// theirs) and its T# on the memory the registry holds for it (the free zeroes
// the base). A texture whose destructor the registry missed never sizes a
// target.
inline bool yebis_live(std::uint64_t vtable, std::uint64_t texture_vtable, const YebisTexture& t, std::uint64_t memory) {
    if (RT_EXTENT_MUTANT(8)) return vtable == texture_vtable;  // the T#'s base not checked
    return vtable == texture_vtable && t.memory == memory && memory;
}

// Whether `padded_h` is what the tiling pads `h` rows to: whole 8-row micro
// tiles, or a 64-, 128- or 256-row alignment (plausible).
inline bool rows_pad_to(std::uint32_t h, std::uint32_t padded_h) {
    return padded_h == h || padded_h == ((h + 7) & ~7u) || plausible(h, padded_h);
}

// The registry's live 2D texture at a target's base (gx_texture_extent).
struct Registered {
    std::uint32_t w = 0, h = 0;  // 0: none
    bool yebis = false;
    std::uint32_t pitch = 0;  // YEBIS's: its T#'s pitch
};
// Whether it sizes a draw's target whose registers pad it to pitch x
// padded_h. GX's when it fits, as GX's T# names the texture the memory holds.
// YEBIS's when it is the surface the registers describe - its pitch the
// draw's, its rows what the tiling pads to the draw's padded height - and it
// fits: YEBIS's textures share memory with others of other sizes, and its
// draws bind no view to say which they mean (a 640x360 texture there must not
// size the 523x441 target of a 640-wide, 448-row surface).
inline bool registered_fits(const Registered& r, std::uint32_t pitch, std::uint32_t padded_h) {
    if (!r.w || !r.h || r.w > pitch || r.h > padded_h) return false;
    if (!r.yebis || RT_EXTENT_MUTANT(7)) return true;  // (7: YEBIS's surface not compared)
    return r.pitch == pitch && rows_pad_to(r.h, padded_h);
}

}  // namespace rt_extent
