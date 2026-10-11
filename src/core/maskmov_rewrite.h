// Rosetta 2 runs vmaskmovdqu as a 16-byte read, blend and write back, so a
// store another thread makes to an unselected byte in between is lost. The
// eboot stores vec3s and vec2s this way, so under Rosetta each site jumps to
// a stub that stores only the selected bytes, flags and registers kept, below
// the red zone; the bytes the jump covers run from the stub, re-aimed.
#pragma once

#include <cstddef>
#include <cstdint>

struct ElfImage;

// VEX form only (data = ModRM.reg, mask = ModRM.rm); the eboot has no SSE2 one.
struct MaskmovInsn {
    std::uint8_t len = 0;
    std::uint8_t data = 0;
    std::uint8_t mask = 0;
};
bool maskmov_decode(const std::uint8_t* p, std::size_t avail, MaskmovInsn* out);

// Stub for the whole instructions at `region` (run at `from`) holding a
// vmaskmovdqu at `site_off`, written to `dst` (run at `to`). `counter` counts
// stores, not atomically. Returns the bytes written, 0 on failure.
std::size_t maskmov_emit_stub(const std::uint8_t* region, std::size_t region_len, std::size_t site_off, std::uint64_t from,
                              std::uint8_t* dst, std::uint64_t to, std::size_t cap, std::uint64_t* counter);

// The region's first five bytes become a jump to `stub`, the rest int3.
bool maskmov_patch(std::uint8_t* region, std::size_t region_len, std::uint64_t from, std::uint64_t stub);

// Under Rosetta 2 (BBHOST_MASKMOV=0 leaves the sites as they are; =1 does
// it on any x86 too, for testing).
bool maskmov_rewrite_wanted();

struct MaskmovRewriteStats {
    std::size_t found = 0;      // vmaskmovdqu sites (53 on 1.09)
    std::size_t rewritten = 0;  // sent through a stub
};
// One executable segment, while still writable. Only 1.09's sites (`is_109`)
// were checked, so other eboots are left alone.
void maskmov_rewrite(ElfImage* image, std::uint64_t va, std::size_t size, bool is_109, MaskmovRewriteStats* st);
void maskmov_rewrite_report(const MaskmovRewriteStats& st);
// At exit: how often the stubs stored, all sites together.
void maskmov_rewrite_exit_report();
