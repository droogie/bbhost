// Native Windows fixture for the actual installer, without booting a guest.
// Build with Clang, -std=c++20 -Wno-invalid-constexpr and -Isrc.
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <windows.h>

namespace {
bool fail_alloc = false;
int protect_calls = 0, fail_protect_at = 0;
int flush_calls = 0, fail_flush_at = 0;
int released = 0;
void* allocated = nullptr;

LPVOID test_alloc(LPVOID address, SIZE_T size, DWORD type, DWORD protection) {
    if (fail_alloc) return nullptr;
    void* p = VirtualAlloc(address, size, type, protection);
    if (p) allocated = p;
    return p;
}
BOOL test_protect(LPVOID address, SIZE_T size, DWORD protection, PDWORD old) {
    if (++protect_calls == fail_protect_at) return FALSE;
    return VirtualProtect(address, size, protection, old);
}
BOOL test_flush(HANDLE process, LPCVOID address, SIZE_T size) {
    if (++flush_calls == fail_flush_at) return FALSE;
    return FlushInstructionCache(process, address, size);
}
BOOL test_free(LPVOID address, SIZE_T size, DWORD type) {
    ++released;
    return VirtualFree(address, size, type);
}
}  // namespace

#define VirtualAlloc test_alloc
#define VirtualProtect test_protect
#define FlushInstructionCache test_flush
#define VirtualFree test_free
#include "../src/engine/graphics_patch.cpp"
#undef VirtualAlloc
#undef VirtualProtect
#undef FlushInstructionCache
#undef VirtualFree

// Unrelated hooks must never execute in this fixture.
bool guest_protect_rwx(GuestMemory*, std::uint64_t, std::size_t) { std::abort(); }
bool guest_protect_rx(GuestMemory*, std::uint64_t, std::size_t) { std::abort(); }
bool guest_protect_rw(GuestMemory*, std::uint64_t, std::size_t) { std::abort(); }
void* thunk_wrap(void*) { std::abort(); }
std::uint8_t* thunk_emit_prologue_stub(std::uint8_t*, std::uint64_t, void*, std::uint64_t,
                                      const std::uint8_t*, std::size_t, bool) { std::abort(); }
bool live_resolution_install(ElfImage*) { std::abort(); }
void menu_memory_install(ElfImage*) { std::abort(); }
void frame_rate_install(ElfImage*) { std::abort(); }
void camera_install(ElfImage*) { std::abort(); }
void camera_tick() { std::abort(); }
void fmod_probe_install(ElfImage*) { std::abort(); }
void sf_heap_probe_install(ElfImage*) { std::abort(); }
bool host_opt_resolution(int*, int*) { std::abort(); }
std::uint64_t hle_video_flip_count() { std::abort(); }
bool host_window_pixels(int*, int*) { std::abort(); }
bool hle_kernel_va_mapped(std::uint64_t, std::size_t) { std::abort(); }
extern "C" GUEST_ABI std::int64_t hle_call_guest6(void*, std::int64_t, std::int64_t, std::int64_t,
                                                std::int64_t, std::int64_t, std::int64_t) { std::abort(); }

static HostSettings fake_settings;
static std::uint64_t fake_serial = 0;
std::uint64_t host_opt_serial() { return fake_serial; }
HostSettings host_settings() { return fake_settings; }

namespace {
constexpr SIZE_T kSize = 0x2000;
void reset_faults() {
    fail_alloc = false;
    protect_calls = fail_protect_at = 0;
    flush_calls = fail_flush_at = 0;
    released = 0;
    allocated = nullptr;
    g_ca_mask = nullptr;
}

DWORD protection(void* address) {
    MEMORY_BASIC_INFORMATION info{};
    assert(VirtualQuery(address, &info, sizeof(info)) == sizeof(info));
    return info.Protect;
}

struct Result {
    std::uint32_t amount = 0, padding = 0;
    std::uint64_t flags = 0, rax = 0;
};
using Runner = void (*)(void* entry, const std::uint32_t* source, Result* result, std::uint64_t flags);

// Windows ABI wrapper: point RBP-0xa70 at source, RBX+0xac at result,
// set arithmetic flags, call the guest's patched copy, capture flags and RAX.
// Preserve the host's nonvolatile registers and provide call shadow space.
const std::uint8_t kRunner[] = {
    0x55, 0x53, 0x48, 0x83, 0xec, 0x28,
    0x48, 0x89, 0xd5, 0x48, 0x81, 0xc5, 0x70, 0x0a, 0x00, 0x00,
    0x4c, 0x89, 0xc3, 0x48, 0x81, 0xeb, 0xac, 0x00, 0x00, 0x00,
    0x41, 0x51, 0x9d, 0xff, 0xd1, 0x9c, 0x5a,
    0x49, 0x89, 0x50, 0x08, 0x49, 0x89, 0x40, 0x10,
    0x48, 0x83, 0xc4, 0x28, 0x5b, 0x5d, 0xc3,
};

void prepare(ElfImage& image, std::size_t offset) {
    DWORD old = 0;
    assert(VirtualProtect(image.mem.base, kSize, PAGE_EXECUTE_READWRITE, &old));
    auto* entry = static_cast<std::uint8_t*>(image.mem.base) + offset;
    std::memcpy(entry, kCaCopyBytes, sizeof(kCaCopyBytes));
    entry[sizeof(kCaCopyBytes)] = 0xc3;
    assert(VirtualProtect(image.mem.base, kSize, PAGE_EXECUTE_READ, &old));
    assert(FlushInstructionCache(GetCurrentProcess(), image.mem.base, kSize));
}

void run_at(std::uint64_t preferred, std::size_t offset, Runner runner) {
    ElfImage image;
    image.mem.base = VirtualAlloc(reinterpret_cast<void*>(preferred), kSize,
                                  MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    assert(image.mem.base);
    image.mem.slide = reinterpret_cast<std::uint64_t>(image.mem.base);
    image.mem.size = kSize;
    const auto at = image.mem.slide + offset;
    auto* entry = static_cast<std::uint8_t*>(image.mem.base) + offset;

    // Fail every OS operation in the installation sequence, including the
    // post-write flush and restoration of guest code's original protection.
    for (int failure = 0; failure < 7; ++failure) {
        prepare(image, offset);
        reset_faults();
        if (failure == 0) fail_alloc = true;
        if (failure >= 1 && failure <= 4) fail_protect_at = failure;
        if (failure >= 5) fail_flush_at = failure - 4;
        assert(!install_ca_stub(&image, at));
        assert(!g_ca_mask);
        assert(std::memcmp(entry, kCaCopyBytes, sizeof(kCaCopyBytes)) == 0);
        assert(protection(entry) == PAGE_EXECUTE_READ);
        assert(released == (failure == 0 ? 0 : 1));
        if (allocated) {
            MEMORY_BASIC_INFORMATION info{};
            assert(VirtualQuery(allocated, &info, sizeof(info)) == sizeof(info));
            assert(info.State == MEM_FREE);
        }
    }

    prepare(image, offset);
    reset_faults();
    assert(install_ca_stub(&image, at));
    assert(g_ca_mask && entry[0] == 0xe9);
    assert(protection(allocated) == PAGE_EXECUTE_READ);
    assert(protection(g_ca_mask) == PAGE_READWRITE);
    assert(protection(entry) == PAGE_EXECUTE_READ);
    assert(!released);
    const std::uint32_t amounts[] = {0, 0x3f800000, 0x3e800000, 0x80000000, 0x7fc12345, 0xffffffff};
    constexpr std::uint64_t kArithmeticFlags = 0x8d5;
    for (auto amount : amounts) {
        for (auto flags : {0x202ull, 0x202ull | kArithmeticFlags}) {
            for (bool on : {true, false, true}) {
                fake_settings.chromatic_aberration = on;
                ++fake_serial;
                refresh_settings();
                Result result;
                runner(entry, &amount, &result, flags);
                assert(result.amount == (on ? amount : 0));
                assert(result.rax == result.amount);  // MOV EAX must zero-extend
                assert((result.flags & kArithmeticFlags) == (flags & kArithmeticFlags));
            }
        }
    }
    g_ca_mask = nullptr;
    assert(VirtualFree(allocated, 0, MEM_RELEASE));
    assert(VirtualFree(image.mem.base, 0, MEM_RELEASE));
}
}  // namespace

int main() {
    void* wrapper = VirtualAlloc(nullptr, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    assert(wrapper);
    std::memcpy(wrapper, kRunner, sizeof(kRunner));
    DWORD old = 0;
    assert(VirtualProtect(wrapper, 0x1000, PAGE_EXECUTE_READ, &old));
    assert(FlushInstructionCache(GetCurrentProcess(), wrapper, sizeof(kRunner)));
    auto runner = reinterpret_cast<Runner>(wrapper);
    run_at(0x400000, 0x100, runner);  // preferred guest slide
    run_at(0x300000000, 0xffb, runner);  // Windows fallback; copy crosses a page
    assert(VirtualFree(wrapper, 0, MEM_RELEASE));
    std::puts("graphics_ca_windows_test: live mask, flags, low/high slides, rollback and page protections passed");
}
