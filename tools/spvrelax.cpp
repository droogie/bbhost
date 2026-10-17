// spvrelax: spv::relax_contraction, as bbhost applies it before MoltenVK.
//
//   spvrelax <in.spv> <out.spv> [--graphics]
//
// Without --graphics only a fragment stage changes, as bbhost does by default.
// Prints how many decorations were dropped and kept.
#include "host/spirv_relax.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: spvrelax <in.spv> <out.spv> [--graphics]\n");
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
    const bool graphics = argc > 3 && std::strcmp(argv[3], "--graphics") == 0;
    spv::RelaxStats st;
    const std::vector<std::uint32_t> out = spv::relax_contraction(words, graphics ? spv::RelaxStages::Graphics : spv::RelaxStages::Fragment, &st);
    std::ofstream o(argv[2], std::ios::binary);
    if (!o.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * 4))) {
        std::fprintf(stderr, "spvrelax: cannot write %s\n", argv[2]);
        return 2;
    }
    std::printf("%s: %u NoContraction dropped, %u kept\n", argv[1], st.dropped, st.kept);
    return 0;
}
