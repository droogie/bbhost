// spvrelax: spv::relax_contraction, as bbhost applies it before MoltenVK.
//
//   spvrelax <in.spv> <out.spv> [--graphics | --invariant] [--exact-discards]
//
// Default: fragment stages only. --graphics: vertex stages too, Position kept
// exact; --invariant: Position made Invariant instead, as bbhost does on
// MoltenVK. --exact-discards: as BBHOST_PS_DISCARD_EXACT=1.
#include "host/spirv_relax.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: spvrelax <in.spv> <out.spv> [--graphics | --invariant] [--exact-discards]\n");
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.empty() || bytes.size() % 4) {
        std::fprintf(stderr, "spvrelax: cannot read %s\n", argv[1]);
        return 2;
    }
    std::vector<std::uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    spv::RelaxStages stages = spv::RelaxStages::Fragment;
    bool exact_discards = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--graphics") == 0) stages = spv::RelaxStages::Graphics;
        if (std::strcmp(argv[i], "--invariant") == 0) stages = spv::RelaxStages::GraphicsInvariant;
        if (std::strcmp(argv[i], "--exact-discards") == 0) exact_discards = true;
    }
    spv::RelaxStats st;
    const std::vector<std::uint32_t> out = spv::relax_contraction(words, stages, &st, exact_discards);
    std::ofstream o(argv[2], std::ios::binary);
    if (!o.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * 4))) {
        std::fprintf(stderr, "spvrelax: cannot write %s\n", argv[2]);
        return 2;
    }
    std::printf("%s: %u NoContraction dropped, %u kept\n", argv[1], st.dropped, st.kept);
    return 0;
}
