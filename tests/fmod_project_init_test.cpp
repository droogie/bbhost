// Owned storage and a constructor stub; no game files or failing cleanup code.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "host/fmod_project_init.h"
#include "decomp/decomp.h"
#include "core/thunk.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
DecompFunction hook{};
unsigned registrations = 0, calls = 0;
bool placed = true;

void setting(const char* value) {
#ifdef _WIN32
    _putenv_s("BBHOST_FMOD_PROJECT_INIT", value);
#else
    setenv("BBHOST_FMOD_PROJECT_INIT", value, 1);
#endif
}

GUEST_ABI std::uint64_t constructor(std::uint64_t object) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(object);
    for (unsigned i = 0; i < 0x288; ++i)
        assert(bytes[i] == (i >= 0x170 && i < 0x174 ? 0 : 0xa5));
    ++calls;
    return 0x12345678;
}
}

void decomp_add(const DecompFunction& fn) { hook = fn; ++registrations; }
int decomp_placed(const char* name) {
    assert(!std::strcmp(name, "fmod-project-language-init"));
    return placed;
}
void host_log(const char*, ...) {}
GUEST_ABI std::int64_t hle_call_guest6(void* fn, std::int64_t a,
    std::int64_t b, std::int64_t c, std::int64_t d, std::int64_t e, std::int64_t f) {
    assert(!b && !c && !d && !e && !f);
    return reinterpret_cast<std::uint64_t(GUEST_ABI*)(std::uint64_t)>(fn)(a);
}

int main(int argc, char** argv) {
    for (const char* disabled : {"", "0", "true"}) {
        setting(disabled);
        fmod_project_init::add_hooks();
        fmod_project_init::qualify();
        assert(registrations == 0);
    }
    setting("1");
    fmod_project_init::add_hooks();
    assert(registrations == 1 && hook.bn == 0x11f3ce0);
    assert(hook.kind == DecompKind::Hosted && hook.original && hook.entry_len == 14);
    if (argc > 1 && !std::strcmp(argv[1], "missing-trampoline")) {
        fmod_project_init::qualify();
        return 99; // Must abort before returning.
    }
    *hook.original = reinterpret_cast<void*>(&constructor);
    if (argc > 1 && !std::strcmp(argv[1], "missing-hook")) {
        placed = false;
        fmod_project_init::qualify();
        return 99;
    }
    fmod_project_init::qualify();
    std::array<unsigned char, 0x288> object;
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        object.fill(0xa5);
        const auto invoke = reinterpret_cast<std::uint64_t(GUEST_ABI*)(std::uint64_t)>(hook.ours);
        assert(invoke(reinterpret_cast<std::uint64_t>(object.data())) == 0x12345678);
        std::uint32_t count = 99;
        std::memcpy(&count, object.data() + 0x170, sizeof(count));
        assert(count == 0);
        // A successful loader remains free to set the actual count afterward.
        count = 2;
        std::memcpy(object.data() + 0x170, &count, sizeof(count));
        assert(object[0x170] == 2);
    }
    assert(calls == 2);
    std::puts("PASS: exact four-byte initialization before constructor, forwarding, reuse, opt-in");
}
