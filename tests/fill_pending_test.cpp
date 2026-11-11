// A texture made after a fill that met no image (host/fill_pending.h): it
// takes the fill's colour only when the fill was made for it - the live GX
// texture whose memory begins at the fill's base, by generation, unwritten by
// GX since - and covers it, and the fill is a colour. The water's interaction
// height map of 2026-11-10 takes it; the memory's next owner (a newer
// generation), the resource after a Map, a texture inside the fill but not at
// its base, one longer than the fill, a fill made over no resource and an
// HTILE word's fill do not. Then five mutants of the rule, each of which a
// case here must catch.
#include "host/fill_pending.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

using Rule = bool (*)(const gpu::PendingFillFor&, std::uint64_t, std::size_t, std::uint64_t, std::uint64_t);

// The water's interaction height map at Moonside Lake's second load (run r20-1):
// a 2x2 R16G16 float texture at 0x23ecf8000 whose 256 bytes the fill of its
// resource's 32 KiB covers, cleared to 0, 0, 0, 1.
constexpr std::uint64_t kBase = 0x23ecf8000ull, kGen = 74702, kSeq = 9310;
constexpr std::size_t kFillBytes = 32768, kMapBytes = 256;

gpu::PendingFillFor fill(std::uint64_t gen = kGen, float r = 0.0f, float b = 0.0f) {
    gpu::PendingFillFor f;
    f.base = kBase;
    f.bytes = kFillBytes;
    f.gen = gen;
    f.write_seq = kSeq;
    f.rgba[0] = r;
    f.rgba[1] = 0.0f;
    f.rgba[2] = b;
    f.rgba[3] = 1.0f;
    return f;
}

struct Case {
    const char* what;
    gpu::PendingFillFor f;
    std::uint64_t base;
    std::size_t bytes;
    std::uint64_t gen;
    std::uint64_t seq;
    bool takes;
};

// How many cases a rule answers wrongly; `say` names them.
int wrong_answers(Rule rule, bool say) {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    const Case cases[] = {
        {"the texture the fill was made for", fill(), kBase, kMapBytes, kGen, kSeq, true},
        {"a texture as long as the fill", fill(), kBase, kFillBytes, kGen, kSeq, true},
        {"the title's grey fill", fill(kGen, 0.5f, 0.5f), kBase, kMapBytes, kGen, kSeq, true},
        {"the memory's next owner (a newer resource)", fill(), kBase, kMapBytes, kGen + 1, kSeq + 4, false},
        {"the next owner, its write sequence the same", fill(), kBase, kMapBytes, kGen + 1, kSeq, false},
        {"an older generation", fill(), kBase, kMapBytes, kGen - 1, kSeq, false},
        {"no live resource at the base now", fill(), kBase, kMapBytes, 0, 0, false},
        {"GX wrote the resource since (a Map)", fill(), kBase, kMapBytes, kGen, kSeq + 1, false},
        {"a texture inside the fill, not at its base (0x23f21d100's case)", fill(), kBase + 0x5100, kMapBytes, kGen, kSeq, false},
        {"a texture starting before the fill", fill(), kBase - 0x100, kMapBytes, kGen, kSeq, false},
        {"a texture longer than the fill", fill(), kBase, kFillBytes + 1, kGen, kSeq, false},
        {"a texture of no bytes", fill(), kBase, 0, kGen, kSeq, false},
        {"a fill made over no resource", fill(0), kBase, kMapBytes, 0, kSeq, false},
        {"a fill made over no resource, a texture there now", fill(0), kBase, kMapBytes, kGen, kSeq, false},
        {"an HTILE word's fill (not a colour)", fill(kGen, nan), kBase, kMapBytes, kGen, kSeq, false},
        {"an infinite channel", fill(kGen, 0.0f, inf), kBase, kMapBytes, kGen, kSeq, false},
    };
    int wrong = 0;
    for (const Case& c : cases) {
        if (rule(c.f, c.base, c.bytes, c.gen, c.seq) == c.takes) continue;
        ++wrong;
        if (say) std::fprintf(stderr, "fill_pending_test: wrong for %s (expected %s)\n", c.what, c.takes ? "taken" : "not taken");
    }
    return wrong;
}

// Mutants: the rule with one condition left out.
bool mutant_no_gen(const gpu::PendingFillFor& f, std::uint64_t base, std::size_t bytes, std::uint64_t, std::uint64_t seq) {
    gpu::PendingFillFor g = f;
    if (!g.gen) g.gen = 1;
    return gpu::pending_fill_takes(g, base, bytes, g.gen, seq);
}
bool mutant_no_write(const gpu::PendingFillFor& f, std::uint64_t base, std::size_t bytes, std::uint64_t gen, std::uint64_t) {
    return gpu::pending_fill_takes(f, base, bytes, gen, f.write_seq);
}
bool mutant_no_base(const gpu::PendingFillFor& f, std::uint64_t, std::size_t bytes, std::uint64_t gen, std::uint64_t seq) {
    return gpu::pending_fill_takes(f, f.base, bytes, gen, seq);
}
bool mutant_no_length(const gpu::PendingFillFor& f, std::uint64_t base, std::size_t bytes, std::uint64_t gen, std::uint64_t seq) {
    gpu::PendingFillFor g = f;
    if (bytes > g.bytes) g.bytes = bytes;
    return gpu::pending_fill_takes(g, base, bytes, gen, seq);
}
bool mutant_no_colour(const gpu::PendingFillFor& f, std::uint64_t base, std::size_t bytes, std::uint64_t gen, std::uint64_t seq) {
    gpu::PendingFillFor g = f;
    for (float& c : g.rgba) {
        if (!std::isfinite(c)) c = 0.0f;
    }
    return gpu::pending_fill_takes(g, base, bytes, gen, seq);
}

}  // namespace

int main() {
    const int wrong = wrong_answers(&gpu::pending_fill_takes, true);
    CHECK(wrong == 0);
    std::printf("fill_pending_test: the rule, %d cases wrong\n", wrong);
    struct Mutant {
        const char* name;
        Rule rule;
    } mutants[] = {{"no generation check", &mutant_no_gen},
                   {"no write check", &mutant_no_write},
                   {"no base check", &mutant_no_base},
                   {"no length check", &mutant_no_length},
                   {"no colour check", &mutant_no_colour}};
    for (const Mutant& m : mutants) {
        const int caught = wrong_answers(m.rule, false);
        std::printf("fill_pending_test: mutant '%s': %d cases wrong%s\n", m.name, caught, caught ? " (caught)" : " - NOT CAUGHT");
        CHECK(caught > 0);
    }
    if (g_failures) {
        std::fprintf(stderr, "fill_pending_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("fill_pending_test: ok\n");
    return 0;
}
