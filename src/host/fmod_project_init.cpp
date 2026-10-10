// EventProjectI's constructor leaves the language count uninitialized. The
// LANG reader assigns it only after allocation and reads succeed, but failed
// loading can reach cleanup first. Start the count at zero on every new object.
#include "host/fmod_project_init.h"
#include "decomp/decomp.h"
#include "core/thunk.h"
#include "hle/common.h"
#include "log.h"

#include <cstdlib>
#include <cstring>

namespace fmod_project_init {
namespace {
bool active = false;
void* original = nullptr;
constexpr const char* name = "fmod-project-language-init";

GUEST_ABI std::uint64_t construct(std::uint64_t project) {
    // Constructor contract: valid, exclusively owned 0x288-byte storage.
    // Successful loading overwrites this four-byte member with the real count.
    const std::uint32_t zero = 0;
    std::memcpy(reinterpret_cast<void*>(project + 0x170), &zero, sizeof(zero));
    return hle_call_guest<std::uint64_t>(original, project);
}
}

void add_hooks() {
    const char* setting = std::getenv("BBHOST_FMOD_PROJECT_INIT");
    active = setting && !std::strcmp(setting, "1");
    if (!active) return;

    static constexpr std::uint8_t entry[] = {
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53,
        0x48, 0x89, 0xfb, 0x48, 0x8d, 0x43, 0x08,
    };
    decomp_add({name, "FMOD project initialization", 0x11f3ce0,
                entry, sizeof(entry), reinterpret_cast<void*>(&construct),
                DecompKind::Hosted, &original});
}

void qualify() {
    if (!active) return;
    if (!original || !decomp_placed(name)) {
        host_log("FMOD project initialization: required constructor hook missing; stopping");
        std::abort();
    }
    host_log("FMOD project initialization: language count starts at zero");
}
}
