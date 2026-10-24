// WAIT_REG_MEM's compare and its can-never-pass rule (hle/wait_reg_mem.h)
// on synthetic packets: the packets written the way Gnm's wait writer in the
// 1.09 eboot writes them (0x1477b30: the function and the memory bit, the
// address's two dwords, the reference, the mask, a poll interval of 10) and
// read the way the command processor reads them (hle/gnm_exec.cpp); every
// function's compare against the masked value; the rule against a search of
// every value under small masks; and the packet a Steam Deck once walked -
// a YEBIS flush's wait for its count 0xdd with its mask written over by 1.
#include "hle/wait_reg_mem.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

namespace {

int g_failed = 0;
void check(bool ok, const char* what, int line) {
    if (ok) return;
    if (++g_failed <= 40) std::printf("FAIL (line %d): %s\n", line, what);
}
#define CHECK(x) check((x), #x, __LINE__)

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// The packet's body as Gnm's waitOnAddress(cb, address, mask, function,
// reference) writes it after the header.
struct Body {
    u32 d[6];
};
Body gnm_wait(u64 address, u32 mask, u32 function, u32 reference) {
    const bool memory = address > 0xffff;
    return Body{{(static_cast<u32>(memory) << 4) | (function & 7), static_cast<u32>(address) & (memory ? 0xfffffffcu : 0xffffu),
                 static_cast<u32>((address >> 32) & 0xffff), reference, mask, 0xa}};
}

// What the command processor takes from it.
struct Read {
    unsigned fn;
    bool memory;
    u64 address;
    u32 ref, mask;
};
Read cp_read(const Body& b) {
    return Read{b.d[0] & 7, (b.d[0] & 0x10) != 0, (static_cast<u64>(b.d[2]) << 32) | b.d[1], b.d[3], b.d[4]};
}

void test_fields() {
    // A label wait as the game writes them: the mask all ones.
    const Body b = gnm_wait(0x152403a50ull, 0xffffffffu, 3, 0xdd);
    const Read r = cp_read(b);
    CHECK(b.d[0] == 0x13 && b.d[1] == 0x52403a50u && b.d[2] == 1 && b.d[3] == 0xdd && b.d[4] == 0xffffffffu && b.d[5] == 0xa);
    CHECK(r.fn == 3 && r.memory && r.address == 0x152403a50ull && r.ref == 0xdd && r.mask == 0xffffffffu);
    CHECK(wait_reg_mem::can_pass(r.fn, r.ref, r.mask));
    CHECK(wait_reg_mem::passes(r.fn, 0xdd, r.ref, r.mask));
    CHECK(!wait_reg_mem::passes(r.fn, 0xdc, r.ref, r.mask));
    CHECK(!wait_reg_mem::passes(r.fn, 0x1dd, r.ref, r.mask));
    // A register wait (an address at or below 0xffff): no memory bit.
    const Read reg = cp_read(gnm_wait(0xc03f, 1, 4, 0));
    CHECK(!reg.memory && reg.address == 0xc03f && reg.ref == 0 && reg.mask == 1);

    // The Deck's packet: 00000013 52403a50 00000001 000000dd 00000001
    // 0000000a. With its mask a 1, no label value passes - not 0xdd, which
    // the flush's event wrote, nor 1.
    const Body deck{{0x13, 0x52403a50u, 1, 0xdd, 1, 0xa}};
    const Read d = cp_read(deck);
    CHECK(d.fn == 3 && d.ref == 0xdd && d.mask == 1);
    CHECK(!wait_reg_mem::can_pass(d.fn, d.ref, d.mask));
    CHECK(!wait_reg_mem::passes(d.fn, 0xdd, d.ref, d.mask));
    CHECK(!wait_reg_mem::passes(d.fn, 1, d.ref, d.mask));
}

// Each function against the masked value, the reference as it is.
void test_compare() {
    using wait_reg_mem::passes;
    CHECK(passes(0, 0, 1, 0) && passes(7, 0, 1, 0));                       // always; reserved
    CHECK(passes(1, 4, 5, ~0u) && !passes(1, 5, 5, ~0u) && passes(1, 0x105, 6, 0xff));
    CHECK(passes(2, 5, 5, ~0u) && !passes(2, 6, 5, ~0u) && passes(2, 0x106, 6, 0xff));
    CHECK(passes(3, 5, 5, ~0u) && !passes(3, 4, 5, ~0u) && passes(3, 0xab05, 5, 0xff) && !passes(3, 0xab05, 0xab05, 0xff));
    CHECK(passes(4, 4, 5, ~0u) && !passes(4, 5, 5, ~0u) && !passes(4, 0x105, 5, 0xff));
    CHECK(passes(5, 5, 5, ~0u) && !passes(5, 4, 5, ~0u) && !passes(5, 0x104, 5, 0xff));
    CHECK(passes(6, 6, 5, ~0u) && !passes(6, 5, 5, ~0u) && !passes(6, 0x105, 5, 0xff));
    // The edges of the range, unsigned.
    CHECK(passes(6, 0xffffffffu, 0x7fffffffu, ~0u) && !passes(1, 0xffffffffu, 0x7fffffffu, ~0u));
}

// The rule is exactly "some value passes": every function, every 4-bit mask,
// references past it, every value under the mask; then masks and references
// at the 32-bit edges, with the values that decide them.
void test_can_pass() {
    int disagree = 0;
    for (unsigned fn = 0; fn < 8; ++fn) {
        for (u32 mask = 0; mask < 16; ++mask) {
            for (u32 ref = 0; ref < 40; ++ref) {
                for (u32 hi : {0u, 0xfffffff0u}) {  // the same in the high bits
                    const u32 m = mask | (hi & 0x30000000u), rf = ref | (hi & 0x10000000u);
                    bool some = false;
                    for (u32 v = 0; v < 16 && !some; ++v) {
                        for (u32 top : {0u, 0x10000000u, 0x20000000u, 0x30000000u}) some = some || wait_reg_mem::passes(fn, v | top, rf, m);
                    }
                    if (some != wait_reg_mem::can_pass(fn, rf, m)) ++disagree;
                }
            }
        }
    }
    CHECK(disagree == 0);
    using wait_reg_mem::can_pass;
    CHECK(can_pass(3, 0xffffffffu, 0xffffffffu) && !can_pass(3, 0x80000000u, 0x7fffffffu));
    CHECK(can_pass(5, 0xffffffffu, 0xffffffffu) && !can_pass(5, 0x80000000u, 0x7fffffffu));
    CHECK(!can_pass(6, 0xffffffffu, 0xffffffffu) && can_pass(6, 0xfffffffeu, 0xffffffffu));
    CHECK(!can_pass(1, 0, 0xffffffffu) && can_pass(1, 1, 0));
    CHECK(can_pass(2, 0, 0) && !can_pass(4, 0, 0) && can_pass(4, 1, 0) && can_pass(4, 0, 1));
}

}  // namespace

int main() {
    test_fields();
    test_compare();
    test_can_pass();
    if (g_failed) {
        std::printf("wait_reg_mem_test: %d failure(s)\n", g_failed);
        return 1;
    }
    std::printf("wait_reg_mem_test: ok\n");
    return 0;
}
