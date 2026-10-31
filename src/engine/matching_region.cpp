#include "engine/matching_region.h"

#include "core/elf.h"
#include "core/memory.h"
#include "engine/addr.h"
#include "log.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

// Each site: its address and the instruction as the 1.09 eboot has it; the
// last byte is the value written.
struct Site {
    std::uint64_t at;
    std::uint8_t bytes[4];
    std::size_t n;
};
constexpr Site kSites[] = {
    {0x208af63, {0xc6, 0x43, 0x0f, 0x00}, 4},  // sub_208af20, the options block's constructor: mov byte [rbx+0xf], 0
};

}  // namespace

void matching_region_install(ElfImage* image) {
    if (!image || image->sha256 != kEboot109Sha256) return;
    if (const char* e = std::getenv("BBHOST_MATCHING_REGION"); e && std::strcmp(e, "game") == 0) {
        host_log("network: Matching Regions starts where the game has it (BBHOST_MATCHING_REGION=game)");
        return;
    }
    const std::uint64_t slide = image->mem.slide;
    int done = 0;
    for (const Site& s : kSites) {
        const std::uint64_t va = slide + (s.at - kPreferredGuestSlide);
        if (va < slide || va + s.n > slide + image->mem.size) continue;
        auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, va));
        if (std::memcmp(p, s.bytes, s.n) != 0) {
            host_log("network: Matching Regions' default at 0x%llx is not the expected code; left as it is", static_cast<unsigned long long>(s.at));
            continue;
        }
        if (!guest_protect_rwx(&image->mem, va & ~0xfffull, ((va + s.n - 1) | 0xfffull) + 1 - (va & ~0xfffull))) continue;
        p[s.n - 1] = 1;  // Worldwide
        guest_protect_rx(&image->mem, va & ~0xfffull, ((va + s.n - 1) | 0xfffull) + 1 - (va & ~0xfffull));
        ++done;
    }
    host_log("network: Matching Regions starts at Worldwide (%d of 1 site; BBHOST_MATCHING_REGION=game keeps the game's)", done);
}
