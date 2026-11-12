// YEBIS's texture updates kept for the texture made for them
// (host/tex_update.h): a texture takes an update only when it is the
// resource the update was for - its generation, at its memory - and the
// update's box lies in the level and layer it names, with the texels to fill
// it; and it takes the texels as rows of the box's width. The textures seen
// in play take theirs (CTextureUtil's black and gray 1x1s, its 4x4
// dither, a mip of its 256x256 noise); the memory's next resource, a level or
// layer the texture lacks, a box past the level's edge and short data do not.
// Then mutants of the rule, each of which a case here must catch, and the
// packing of rows given with a wider pitch.
#include "host/tex_update.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

using Rule = bool (*)(const gpu::TexUpdate&, std::uint64_t, std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
                      std::uint32_t);

// CTextureUtil's gray 1x1 of a Hunter's Dream visit's set (flip 286): RGBA8, its YEBIS resource 50114 at
// 0x15fb9e700, 0x7f in every byte.
constexpr std::uint64_t kBase = 0x15fb9e700ull, kGen = 50114;

gpu::TexUpdate update(std::uint32_t w, std::uint32_t h, std::uint32_t pitch, std::uint32_t mip = 0, std::uint32_t x = 0, std::uint32_t y = 0,
                      std::uint64_t gen = kGen, std::size_t rows_bytes = 0) {
    gpu::TexUpdate u;
    u.memory = kBase;
    u.gen = gen;
    u.mip = mip;
    u.x = x;
    u.y = y;
    u.w = w;
    u.h = h;
    u.pitch = pitch;
    u.rows.assign(rows_bytes ? rows_bytes : static_cast<std::size_t>(pitch) * h, 0x7f);
    return u;
}

struct Case {
    const char* what;
    gpu::TexUpdate u;
    std::uint64_t base, gen;
    std::uint32_t levels, layers, mip_w, mip_h, texel_bytes;
    bool takes;
};

int wrong_answers(Rule rule, bool say) {
    const Case cases[] = {
        {"the gray 1x1 its update was for", update(1, 1, 4), kBase, kGen, 1, 1, 1, 1, 4, true},
        {"the 4x4 dither", update(4, 4, 16), kBase, kGen, 1, 1, 4, 4, 4, true},
        {"mip 3 of the 256x256 noise (32x32)", update(32, 32, 128, 3), kBase, kGen, 9, 1, 32, 32, 4, true},
        {"a box inside the level", update(2, 2, 8, 0, 2, 2), kBase, kGen, 1, 1, 4, 4, 4, true},
        {"rows given with a wider pitch", update(4, 4, 64), kBase, kGen, 1, 1, 4, 4, 4, true},
        {"the last row only as long as the box", update(4, 4, 64, 0, 0, 0, kGen, 64 * 3 + 16), kBase, kGen, 1, 1, 4, 4, 4, true},
        {"the memory's next resource (a newer generation)", update(1, 1, 4), kBase, kGen + 1, 1, 1, 1, 1, 4, false},
        {"an older generation", update(1, 1, 4), kBase, kGen - 1, 1, 1, 1, 1, 4, false},
        {"an update made over no resource", update(1, 1, 4, 0, 0, 0, 0), kBase, 0, 1, 1, 1, 1, 4, false},
        {"another texture's memory", update(1, 1, 4), kBase + 0x400, kGen, 1, 1, 1, 1, 4, false},
        {"a level the texture lacks", update(1, 1, 4, 1), kBase, kGen, 1, 1, 1, 1, 4, false},
        {"a layer the texture lacks", [] { auto u = update(1, 1, 4); u.slice = 1; return u; }(), kBase, kGen, 1, 1, 1, 1, 4, false},
        {"a box wider than the level", update(8, 4, 32), kBase, kGen, 1, 1, 4, 4, 4, false},
        {"a box past the level's edge", update(2, 2, 8, 0, 3, 0), kBase, kGen, 1, 1, 4, 4, 4, false},
        {"a box starting outside the level", update(1, 1, 4, 0, 0, 4), kBase, kGen, 1, 1, 4, 4, 4, false},
        {"an empty box", update(0, 4, 16), kBase, kGen, 1, 1, 4, 4, 4, false},
        {"a pitch shorter than the box's rows", update(4, 4, 8), kBase, kGen, 1, 1, 4, 4, 4, false},
        {"data shorter than the box", update(4, 4, 16, 0, 0, 0, kGen, 16 * 3), kBase, kGen, 1, 1, 4, 4, 4, false},
    };
    int wrong = 0;
    for (const Case& c : cases) {
        if (rule(c.u, c.base, c.gen, c.levels, c.layers, c.mip_w, c.mip_h, c.texel_bytes) == c.takes) continue;
        ++wrong;
        if (say) std::fprintf(stderr, "tex_update_test: wrong for %s (expected %s)\n", c.what, c.takes ? "taken" : "not taken");
    }
    return wrong;
}

// Mutants: the rule with one condition left out.
bool mutant_no_gen(const gpu::TexUpdate& u, std::uint64_t base, std::uint64_t, std::uint32_t levels, std::uint32_t layers, std::uint32_t w,
                   std::uint32_t h, std::uint32_t tb) {
    gpu::TexUpdate v = u;
    if (!v.gen) v.gen = 1;
    return gpu::tex_update_takes(v, base, v.gen, levels, layers, w, h, tb);
}
bool mutant_no_base(const gpu::TexUpdate& u, std::uint64_t, std::uint64_t gen, std::uint32_t levels, std::uint32_t layers, std::uint32_t w,
                    std::uint32_t h, std::uint32_t tb) {
    return gpu::tex_update_takes(u, u.memory, gen, levels, layers, w, h, tb);
}
bool mutant_no_level(const gpu::TexUpdate& u, std::uint64_t base, std::uint64_t gen, std::uint32_t, std::uint32_t, std::uint32_t w,
                     std::uint32_t h, std::uint32_t tb) {
    return gpu::tex_update_takes(u, base, gen, 64, 64, w, h, tb);
}
bool mutant_no_box(const gpu::TexUpdate& u, std::uint64_t base, std::uint64_t gen, std::uint32_t levels, std::uint32_t layers, std::uint32_t,
                   std::uint32_t, std::uint32_t tb) {
    return gpu::tex_update_takes(u, base, gen, levels, layers, 1u << 20, 1u << 20, tb);
}
bool mutant_no_size(const gpu::TexUpdate& u, std::uint64_t base, std::uint64_t gen, std::uint32_t levels, std::uint32_t layers, std::uint32_t w,
                    std::uint32_t h, std::uint32_t tb) {
    gpu::TexUpdate v = u;
    if (v.pitch < v.w * tb) v.pitch = v.w * tb;
    v.rows.resize(static_cast<std::size_t>(v.pitch) * v.h);
    return gpu::tex_update_takes(v, base, gen, levels, layers, w, h, tb);
}

}  // namespace

int main() {
    const int wrong = wrong_answers(&gpu::tex_update_takes, true);
    CHECK(wrong == 0);
    std::printf("tex_update_test: the rule, %d cases wrong\n", wrong);
    struct Mutant {
        const char* name;
        Rule rule;
    } mutants[] = {{"no generation check", &mutant_no_gen},
                   {"no memory check", &mutant_no_base},
                   {"no level or layer check", &mutant_no_level},
                   {"no box check", &mutant_no_box},
                   {"no pitch or length check", &mutant_no_size}};
    for (const Mutant& m : mutants) {
        const int caught = wrong_answers(m.rule, false);
        std::printf("tex_update_test: mutant '%s': %d cases wrong%s\n", m.name, caught, caught ? " (caught)" : " - NOT CAUGHT");
        CHECK(caught > 0);
    }
    // The rows as the image takes them: a 2x2 RGBA8 box given 16 bytes apart comes out 8 bytes a row.
    gpu::TexUpdate u = update(2, 2, 16);
    for (std::size_t i = 0; i < u.rows.size(); ++i) u.rows[i] = static_cast<std::uint8_t>(i);
    const std::vector<std::uint8_t> packed = gpu::tex_update_packed(u, 4);
    CHECK(packed.size() == 16);
    for (std::size_t i = 0; i < 8 && packed.size() == 16; ++i) {
        CHECK(packed[i] == i);
        CHECK(packed[8 + i] == 16 + i);
    }
    if (g_failures) {
        std::fprintf(stderr, "tex_update_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("tex_update_test: ok\n");
    return 0;
}
