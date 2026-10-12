#include "engine/chromatic_stub.h"
#include <array>
#include <atomic>
#include <cassert>
#include <new>
void test_view_fields() {
    std::array<std::uint8_t, 0x370> area;
    area.fill(0x5a);
    const float dispersion = 0.006f;
    std::memcpy(area.data() + 0xb4, &dispersion, 4);
    auto view = area;
    engine::apply_chromatic_setting(view.data(), true);
    assert(view == area);
    engine::apply_chromatic_setting(view.data(), false);
    for (unsigned i = 0; i < view.size(); ++i) {
        const bool suppressed = (i >= 0xac && i < 0xb0) || (i >= 0xb4 && i < 0xc4);
        assert(view[i] == (suppressed ? 0 : area[i]));
    }
    // The next view is rebuilt by the game, so turning on restores the area's
    // dispersion without retaining or editing its GPARAM entries.
    view = area;
    engine::apply_chromatic_setting(view.data(), true);
    assert(view == area);
}
#if defined(_WIN32) && defined(__x86_64__)
#include <windows.h>
int main() {
    test_view_fields();
    auto* code = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    assert(code);
    // Windows ABI harness: rbp is the synthetic stack-block base, rbx the
    // synthetic post-process block. Verify eax, the destination and carry flag.
    constexpr std::uint8_t prefix[] = {0x53,0x55,0x48,0x89,0xcd,0x48,0x89,0xd3,0xf9};
    constexpr std::uint8_t suffix[] = {0x9c,0x5a,0x83,0xe2,0x01,0x89,0x93,0xb0,0,0,0,0x5d,0x5b,0xc3};
    std::memcpy(code, prefix, sizeof(prefix));
    auto* entry = code + sizeof(prefix);
    auto* resume = entry + 12;
    std::memcpy(resume, suffix, sizeof(suffix));
    auto* body = code + 256;
    engine::emit_ca_absolute_stub(entry, body, reinterpret_cast<std::uint64_t>(resume));
    auto* mask = new (body + engine::ca_mask_offset) std::atomic<std::uint32_t>(~0u);
    FlushInstructionCache(GetCurrentProcess(), code, 4096);
    std::array<std::uint8_t, 0xa80> stack{};
    std::array<std::uint8_t, 0xc0> object{};
    const std::uint32_t amount = 0x3f99999a;
    std::memcpy(stack.data(), &amount, 4);
    const auto call = reinterpret_cast<std::uint32_t(*)(void*,void*)>(code);
    const auto read = [&](unsigned offset) { std::uint32_t v; std::memcpy(&v, object.data()+offset,4); return v; };
    assert(call(stack.data()+0xa70,object.data()) == amount);
    assert(read(0xac) == amount && read(0xb0) == 1);
    mask->store(0);
    assert(call(stack.data()+0xa70,object.data()) == 0);
    assert(read(0xac) == 0 && read(0xb0) == 1);
    mask->store(~0u);
    assert(call(stack.data()+0xa70,object.data()) == amount);
    assert(read(0xac) == amount && read(0xb0) == 1);
    VirtualFree(code, 0, MEM_RELEASE);
}
#else
int main() { test_view_fields(); }
#endif
