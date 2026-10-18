// GX's pipeline drain as a barrier token (decomp/gx/drain_token.cpp) beside
// the game's own gx_command_buffer_wait_for_eop (tests/decomp/eboot_kit.h).
// Ours writes something else on purpose - a two-dword NOP {kGxBarrierMagic}
// where the game writes its label, end-of-pipe event and wait - so this checks
// what they must share: the command buffer they write through, read the same
// way ({+0 chunk, +0x8 end, +0x10 write pointer, +0x18 refill, +0x20 its
// argument}), the room asked of the refill when there is none, and that
// neither writes past the room it has. The game's version runs in the loaded
// eboot on the same buffers: its packets are where ours reads them to be (the
// write pointer moves by the EOP's six dwords and the wait's seven, the label
// comes off the end), and it calls the same refill. Skips without the 1.09
// eboot.
#include "decomp/gx/drain_token.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr u64 kWaitForEop = 0x2ab09e0;
using WaitForEop = GUEST_ABI void (*)(u64 cb, u32 event, u32 cache);

// A command buffer the size GX's is (the arena allocator it reads the chunk
// size through sits at +0xb4c8), over chunks of the test's.
struct Buffer {
    std::vector<u8> object = std::vector<u8>(0xb500, 0);
    std::vector<std::vector<u8>> chunks;
    u64 allocator[2] = {0, 512};  // +8: the chunk size in qwords (4 KiB)
    u64 cb() { return reinterpret_cast<u64>(object.data()); }
    void chunk(std::size_t room_bytes) {
        chunks.emplace_back(4096, 0xcd);
        const u64 base = reinterpret_cast<u64>(chunks.back().data());
        store<u64>(cb() + 0x0, base);
        store<u64>(cb() + 0x8, base + 4096);
        store<u64>(cb() + 0x10, base + 4096 - room_bytes);
    }
};

Buffer* g_buffer = nullptr;
int g_stub_refills = 0;
u32 g_stub_refill_dwords = 0;
GUEST_ABI u8 stub_refill(u64 cb, u32 dwords, u64 arg, u32) {
    ++g_stub_refills;
    g_stub_refill_dwords = dwords;
    if (arg != 0x1234 || cb != g_buffer->cb()) return 0;
    g_buffer->chunk(4096);
    return 1;
}

Buffer make(std::size_t room) {
    Buffer b;
    store<u64>(b.cb() + 0x18, reinterpret_cast<u64>(&stub_refill));
    store<u64>(b.cb() + 0x20, 0x1234);
    store<u64>(b.cb() + 0xb4c8, reinterpret_cast<u64>(b.allocator));
    b.chunk(room);
    return b;
}

int g_failed = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failed;
    }
}

}  // namespace

int main() {
    eboot_kit::load("gx_drain_token_test");
    const auto game = eboot_kit::fn<WaitForEop>(kWaitForEop);

    // Room for everything: the game's packets and label, and ours.
    {
        Buffer b = make(1024);
        g_buffer = &b;
        g_stub_refills = 0;
        const u64 wp = load<u64>(b.cb() + 0x10), end = load<u64>(b.cb() + 0x8);
        game(b.cb(), 0x28, 0);
        const u64 wp2 = load<u64>(b.cb() + 0x10), end2 = load<u64>(b.cb() + 0x8);
        check(g_stub_refills == 0, "game: no refill with room");
        check(wp2 - wp == (6 + 7) * 4, "game: an EVENT_WRITE_EOP and a WAIT_REG_MEM at the write pointer");
        check(load<u32>(wp) >> 30 == 3 && ((load<u32>(wp) >> 8) & 0xff) == 0x47, "game: the first packet is an EOP");
        check(((load<u32>(wp + 24) >> 8) & 0xff) == 0x3c, "game: the second is a WAIT_REG_MEM");
        check(end2 < end && end - end2 <= 16, "game: the label comes off the end");

        Buffer o = make(1024);
        g_buffer = &o;
        g_stub_refills = 0;
        const u64 owp = load<u64>(o.cb() + 0x10), oend = load<u64>(o.cb() + 0x8);
        gx_drain_token(o.cb(), 0x28, 0);
        check(g_stub_refills == 0, "ours: no refill with room");
        check(load<u64>(o.cb() + 0x10) == owp + 8, "ours: two dwords");
        check(load<u64>(o.cb() + 0x8) == oend, "ours: nothing off the end");
        check(load<u32>(owp) == 0xc0001000u && load<u32>(owp + 4) == kGxBarrierMagic, "ours: the barrier token");
    }
    // No room at all: both ask the refill, and write into the new chunk.
    {
        Buffer b = make(4);
        g_buffer = &b;
        g_stub_refills = 0;
        game(b.cb(), 0x28, 0);
        check(g_stub_refills >= 1, "game: a refill without room");
        const u64 base = load<u64>(b.cb());
        check(base == reinterpret_cast<u64>(b.chunks.back().data()), "game: into the new chunk");

        Buffer o = make(4);
        g_buffer = &o;
        g_stub_refills = 0;
        gx_drain_token(o.cb(), 0x28, 0);
        check(g_stub_refills == 1 && g_stub_refill_dwords == 2, "ours: one refill, for two dwords");
        const u64 nb = reinterpret_cast<u64>(o.chunks.back().data());
        check(load<u32>(nb) == 0xc0001000u && load<u32>(nb + 4) == kGxBarrierMagic && load<u64>(o.cb() + 0x10) == nb + 8,
              "ours: the token at the new chunk's start");
        check(o.chunks.front()[4096 - 4] == 0xcd, "ours: nothing written in the old chunk's last dword");
    }
    // A refill that gives nothing: ours writes nothing.
    {
        Buffer o = make(4);
        store<u64>(o.cb() + 0x20, 0x9999);  // the stub refuses another argument
        g_buffer = &o;
        g_stub_refills = 0;
        const u64 owp = load<u64>(o.cb() + 0x10);
        gx_drain_token(o.cb(), 4, 0);
        check(g_stub_refills == 1 && load<u64>(o.cb() + 0x10) == owp, "ours: nothing written when the refill refuses");
    }
    std::printf("gx_drain_token: %s\n", g_failed ? "FAILED" : "ok");
    return g_failed ? 1 : 0;
}
