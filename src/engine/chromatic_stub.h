#pragma once
#include <cstdint>
#include <cstring>

namespace engine {
// Bloodborne 1.09's tone-map copy maps GPARAM lateral/uniform dispersion
// to these four YEBIS fields. The enable integer alone does not remove
// the visible colour separation. Apply to a freshly copied view block;
// enabling leaves the area's values intact on the next view.
inline void apply_chromatic_setting(std::uint8_t* block, bool enabled) {
    if (enabled) return;
    const std::uint32_t zero = 0;
    constexpr unsigned fields[] = {0xacu, 0xb4u, 0xb8u, 0xbcu, 0xc0u};
    for (unsigned offset : fields)
        std::memcpy(block + offset, &zero, sizeof(zero));
}
inline constexpr unsigned ca_mask_offset = 0x40;
// Used at Bloodborne 1.09's verified two-instruction amount copy. The entry
// consumes exactly its 12 bytes; the body preserves flags and returns by an
// indirect jump so the copied value remains in eax.
inline void emit_ca_absolute_stub(std::uint8_t* entry, std::uint8_t* body, std::uint64_t resume) {
    constexpr std::uint8_t load[] = {0x8b,0x85,0x90,0xf5,0xff,0xff};
    constexpr std::uint8_t save[] = {0x89,0x83,0xac,0,0,0};
    body[0] = 0x9c;
    std::memcpy(body + 1, load, 6);
    body[7] = 0x23; body[8] = 0x05;
    const std::int32_t mask_delta = ca_mask_offset - 13;
    std::memcpy(body + 9, &mask_delta, 4);
    body[13] = 0x9d;
    std::memcpy(body + 14, save, 6);
    body[20] = 0xff; body[21] = 0x25;
    std::memset(body + 22, 0, 4);
    std::memcpy(body + 26, &resume, 8);
    const auto target = reinterpret_cast<std::uint64_t>(body);
    entry[0] = 0x48; entry[1] = 0xb8;
    std::memcpy(entry + 2, &target, 8);
    entry[10] = 0xff; entry[11] = 0xe0;
}
} // namespace engine
