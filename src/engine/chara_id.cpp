#include "guest_abi.h"
#include "engine/chara_id.h"

#include "core/elf.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstring>

namespace {

// The network login's character step (0x2043ee0; addresses are Binary
// Ninja's): when GameDataMan's local PlayerGameData (data_593b130 -> +8) holds
// a character id at +0x690 it hands it to FrpgNetMan (+0xa90, the CharaId of
// every later request) and goes on; when it holds 0 it calls
// calls_sync_chara_id_web_api (0x228be50), whose answer's first
// PublishCharacterIdList entry the callback stores at +0x690 with no check -
// 0x8000000000000000 when the list is empty, which the PlayerGameData
// serializers (0x18f23a0, 0x18f2660) then write into the save.
constexpr std::uint64_t kCharaStep = 0x2043ee0;
constexpr std::uint8_t kCharaStepPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56,
                                               0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x40};
constexpr std::uint64_t kGameDataMan = 0x593b130;
constexpr std::uint64_t kPlayerGameData = 0x8;
constexpr std::uint64_t kCharaIdField = 0x690;
constexpr std::uint64_t kUnset = 0x8000000000000000ull;

std::uint64_t g_slide = 0;
std::atomic<int> g_cleared{0};

GUEST_ABI std::int64_t chara_step_hook(std::uint64_t, const std::uint64_t*) {
    std::uint64_t gdm = 0, pgd = 0, id = 0;
    std::memcpy(&gdm, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(g_slide + (kGameDataMan - kPreferredGuestSlide))),
                sizeof(gdm));
    if (!gdm) return 0;
    std::memcpy(&pgd, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(gdm + kPlayerGameData)), sizeof(pgd));
    if (!pgd) return 0;
    auto* field = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(pgd + kCharaIdField));
    std::memcpy(&id, field, sizeof(id));
    if (id == kUnset) {
        const std::uint64_t none = 0;
        std::memcpy(field, &none, sizeof(none));
        if (g_cleared.fetch_add(1) < 4) {
            host_log("chara id: this character has none from the server (0x8000000000000000); asking for one");
        }
    } else if (id) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) host_log("chara id: %llu", static_cast<unsigned long long>(id));
    }
    return 0;  // the step runs as it would
}

}  // namespace

void chara_id_install(ElfImage* image) {
    if (!image || image->sha256 != kEboot109Sha256) {
        host_log("chara id: off - not the 1.09 eboot");
        return;
    }
    g_slide = image->mem.slide;
    const std::uint64_t at = g_slide + (kCharaStep - kPreferredGuestSlide);
    if (!engine_prologue_hook(image, at, kCharaStepPrologue, sizeof(kCharaStepPrologue),
                              reinterpret_cast<void*>(&chara_step_hook))) {
        host_log("chara id: off - the login's character step (0x%llx) is not the expected code",
                 static_cast<unsigned long long>(kCharaStep));
    }
}
