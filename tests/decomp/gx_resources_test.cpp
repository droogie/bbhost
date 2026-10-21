// GX's CreateBuffer (decomp/gx/resources.cpp) beside the game's own
// (tests/decomp/eboot_kit.h). The resource maker (gx_buffer_create_internal)
// and the rename-slot release are recorders, scripted per case (the
// resource's bytes, its result), and the allocators are test objects whose
// vtables record their calls (the holder's allocation failing now and then).
// On generated descriptions - every usage, bind, CPU access and misc flag
// the creator tests and garbage beside them, structured strides on both
// sides of each limit, initial data or none, allocator overrides, a device
// half made - ours must make the game's calls with the game's arguments in
// the game's order, leave the holder and the out pointer as the game's
// leaves them and return what the game's returns, and tell the host once
// what it made. Five deliberate mistakes must be caught, and the entry
// holds the instructions written against. Skips without the 1.09 eboot.
#define DECOMP_GX_RESOURCES_MUTANTS
int g_resources_mutant = 0;
#include "decomp/gx/resources.cpp"

#include "eboot_kit.h"

#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <vector>

namespace {

eboot_kit::Rng g_rng(0x6b5f);

struct Call {
    int who;
    u64 a[5];
    bool operator==(const Call&) const = default;
};
std::vector<Call> g_rec, g_told;

enum Who { kMakeId, kAllocId, kFreeId, kFreeSlotId, kToldId };

// What the resource maker writes and returns, and whether the holder's
// allocation fails, this case.
struct Script {
    u32 make_result = 0;
    u8 resource[0x40] = {};
    bool writes_tail = false;  // the rename slots' allocator and slots at +0x30 / +0x38
    bool alloc_fails = false;
} g_script;

alignas(64) u8 g_holder[0x80];
alignas(16) u16 g_slots[16];

GUEST_ABI u32 fake_make(u64 resource, u64 unused, u64 desc, u64 initial_data, u64 object_alloc, u64 memory_alloc, u64 placement) {
    (void)unused;  // the creator leaves whatever it last held in rsi; the maker reads its stack argument there
    g_rec.push_back({kMakeId, {desc, initial_data, object_alloc, memory_alloc, placement}});
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(resource)), g_script.resource, 0x30);
    if (g_script.writes_tail) std::memcpy(reinterpret_cast<u8*>(static_cast<std::uintptr_t>(resource)) + 0x30, g_script.resource + 0x30, 0x10);
    return g_script.make_result;
}
GUEST_ABI u64 fake_alloc(u64 allocator, u64 bytes, u64 align) {
    g_rec.push_back({kAllocId, {allocator, bytes & 0xffffffffull, align & 0xffffffffull}});
    return g_script.alloc_fails ? 0 : reinterpret_cast<u64>(g_holder);
}
GUEST_ABI void fake_free(u64 allocator, u64 memory) { g_rec.push_back({kFreeId, {allocator, memory}}); }
GUEST_ABI void fake_free_slot(u64 at, u64 slot) { g_rec.push_back({kFreeSlotId, {at, slot & 0xffffffffull}}); }
GUEST_ABI void fake_told(u64 holder, u64 desc, u64 initial_data, u64 result) {
    g_told.push_back({kToldId, {holder, desc, initial_data, result & 0xffffffffull}});
}

// Four allocators sharing one vtable (+0x58 allocate, +0x70 free).
alignas(16) u64 g_vtable[32];
alignas(16) u64 g_allocators[4][2];
alignas(64) u8 g_device[0xdc40];
alignas(16) u32 g_desc[6];
alignas(16) u64 g_initial[2];
u64 g_out;

int g_failed = 0;
bool g_quiet = false;
void fail(int trial, const char* why) {
    if (++g_failed <= 40 && !g_quiet) {
        std::printf("FAIL: case %d: %s (%u bytes, usage %u, bind 0x%x, cpu %u, misc 0x%x, stride %u)\n", trial, why, g_desc[0], g_desc[1],
                    g_desc[2], g_desc[3], g_desc[4], g_desc[5]);
    }
}

u32 pick(const u32* v, int n) { return v[g_rng.range(0, n - 1)]; }

struct Case {
    u64 init, object_override, memory_override, placement;
};

Case generate() {
    static const u32 kUsage[] = {0, 1, 2, 3, 0, 1, 2, 3, 4, 7, 0x80000001u};
    static const u32 kBind[] = {0x1, 0x2, 0x3, 0x4, 0x8, 0x9, 0xa, 0xb, 0x10, 0x11, 0x20, 0x40, 0x80, 0x88, 0x90, 0xc0,
                                0x60, 0x5, 0x6, 0xc, 0x0, 0x100, 0x101, 0xf0, 0x8a, 0x18, 0x1, 0x2, 0x3, 0x8};
    static const u32 kCpu[] = {0, 1, 2, 3, 0, 0, 1, 3, 4, 0x10000, 0x100};
    static const u32 kMisc[] = {0, 0, 0, 0x20, 0x20, 0x2, 0x8, 0x10, 0x18, 0x4000, 0x4020, 0x40, 0x1, 0x4, 0x80, 0x100,
                                0x200, 0x400, 0x8000, 0x10000, 0x800, 0x1000, 0x2000, 0x22, 0x28};
    static const u32 kStride[] = {0, 4, 8, 12, 16, 32, 0x7fc, 0x800, 0x804, 0x1000, 3, 6, 1, 2};
    u32 usage, bind, cpu, misc, stride, bytes;
    if (g_rng.chance(0.6)) {
        // A description the creator takes - then, a quarter of the time, one field changed.
        static const u32 kDefaultBind[] = {0x1, 0x2, 0x3, 0x8, 0x9, 0xa, 0x80, 0x88, 0x18, 0x10, 0x11, 0x0, 0x4};
        static const u32 kImmutableBind[] = {0x1, 0x2, 0x3, 0x8, 0x9, 0xa, 0xb, 0x4};
        static const u32 kDynamicBind[] = {0x1, 0x2, 0x3, 0x8, 0x9, 0xa, 0x4};
        static const u32 kGoodMisc[] = {0, 0, 0, 0x20, 0x20, 0x4000, 0x4020};
        usage = static_cast<u32>(g_rng.range(0, 3));
        misc = pick(kGoodMisc, static_cast<int>(std::size(kGoodMisc)));
        switch (usage) {
        case 0: cpu = 0, bind = pick(kDefaultBind, static_cast<int>(std::size(kDefaultBind))); break;
        case 1: cpu = 0, bind = pick(kImmutableBind, static_cast<int>(std::size(kImmutableBind))); break;
        case 2: cpu = g_rng.chance(0.8) ? 1 : 3, bind = pick(kDynamicBind, static_cast<int>(std::size(kDynamicBind))); break;
        default: cpu = static_cast<u32>(g_rng.range(1, 3)), bind = 0; break;
        }
        stride = 4u * static_cast<u32>(g_rng.range(1, 0x200));
        bytes = (misc & 0x20) ? stride * (g_rng.chance(0.2) ? 1u : static_cast<u32>(g_rng.range(1, 2048))) : static_cast<u32>(g_rng.range(1, 1 << 22));
        if (g_rng.chance(0.25)) {
            switch (g_rng.range(0, 5)) {
            case 0: usage = g_rng.chance(0.5) ? static_cast<u32>(g_rng.range(0, 7)) : g_rng.u32(); break;
            case 1: bind ^= 1u << g_rng.range(0, 9); break;
            case 2: cpu = g_rng.chance(0.5) ? static_cast<u32>(g_rng.range(0, 4)) : g_rng.u32(); break;
            case 3: misc ^= 1u << g_rng.range(0, 16); break;
            case 4: stride = g_rng.chance(0.5) ? stride + static_cast<u32>(g_rng.range(-3, 3)) : g_rng.u32() & 0x1fff; break;
            default: bytes = g_rng.chance(0.3) ? 0 : bytes + static_cast<u32>(g_rng.range(-4, 4)); break;
            }
        }
    } else {
        usage = g_rng.chance(0.03) ? g_rng.u32() : pick(kUsage, static_cast<int>(std::size(kUsage)));
        bind = g_rng.chance(0.04) ? g_rng.u32() & 0x3ff : pick(kBind, static_cast<int>(std::size(kBind)));
        cpu = g_rng.chance(0.03) ? g_rng.u32() : pick(kCpu, static_cast<int>(std::size(kCpu)));
        misc = g_rng.chance(0.03) ? g_rng.u32() : pick(kMisc, static_cast<int>(std::size(kMisc)));
        stride = g_rng.chance(0.05) ? g_rng.u32() : pick(kStride, static_cast<int>(std::size(kStride)));
        switch (g_rng.range(0, 5)) {
        case 0: bytes = 0; break;
        case 1: bytes = static_cast<u32>(g_rng.range(1, 64)); break;
        case 2: bytes = stride ? stride * static_cast<u32>(g_rng.range(1, 4096)) : 16; break;
        case 3: bytes = stride ? stride * static_cast<u32>(g_rng.range(1, 64)) + static_cast<u32>(g_rng.range(1, 3)) : 1; break;
        case 4: bytes = stride ? stride - static_cast<u32>(g_rng.range(0, 1)) : 4; break;
        default: bytes = g_rng.u32(); break;
        }
    }
    g_desc[0] = bytes;
    g_desc[1] = usage;
    g_desc[2] = bind;
    g_desc[3] = cpu;
    g_desc[4] = misc;
    g_desc[5] = stride;
    // The device: made (its three objects there) or half made, its two allocators.
    std::memset(g_device, 0, sizeof g_device);
    const auto obj = [](int k) { return reinterpret_cast<u64>(g_allocators[k]); };
    store<u64>(reinterpret_cast<u64>(g_device), g_rng.chance(0.95) ? 0x1111 : 0);
    store<u64>(reinterpret_cast<u64>(g_device) + 8, g_rng.chance(0.95) ? 0x2222 : 0);
    store<u64>(reinterpret_cast<u64>(g_device) + 0x20, g_rng.chance(0.95) ? 0x3333 : 0);
    store<u64>(reinterpret_cast<u64>(g_device) + 0xdc10, g_rng.chance(0.96) ? obj(0) : 0);
    store<u64>(reinterpret_cast<u64>(g_device) + 0xdc20, g_rng.chance(0.96) ? obj(1) : 0);
    // GX's contexts (the first 16 of the table): some there, some not.
    for (int i = 0; i < 16; ++i)
        store<u64>(game_address(0x5aa63c0) + 8 * i, g_rng.chance(0.6) ? 0x7000000000ull + 0x10000ull * static_cast<u64>(i) : 0);
    Case c;
    c.init = g_rng.chance(0.65) ? reinterpret_cast<u64>(g_initial) : 0;
    c.object_override = g_rng.chance(0.2) ? obj(2) : 0;
    c.memory_override = g_rng.chance(0.2) ? obj(3) : 0;
    c.placement = g_rng.chance(0.5) ? 0 : g_rng.next();
    // The resource maker's answer and bytes: its allocator, memory, id, the
    // description's fields, the renamed flag; a constant buffer's slots.
    static const u32 kResults[] = {5, 8, 9};
    g_script.make_result = g_rng.chance(0.85) ? 0 : pick(kResults, 3);
    for (u8& b : g_script.resource) b = static_cast<u8>(g_rng.u32());
    // Its allocator (an object with the allocators' vtable, or none) and memory.
    const u64 resource_alloc = g_rng.chance(0.9) ? obj(1) : 0, memory = g_rng.chance(0.9) ? g_rng.next() | 1 : 0;
    std::memcpy(g_script.resource, &resource_alloc, 8);
    std::memcpy(g_script.resource + 8, &memory, 8);
    g_script.resource[0x23] = static_cast<u8>(bind);
    g_script.writes_tail = (bind & 4) != 0 || g_rng.chance(0.3);
    if (bind & 4) {
        const u64 slot_alloc = g_rng.chance(0.9) ? obj(0) : 0;
        const u64 slots = reinterpret_cast<u64>(g_slots);
        std::memcpy(g_script.resource + 0x30, &slot_alloc, 8);
        std::memcpy(g_script.resource + 0x38, &slots, 8);
    }
    for (u16& s : g_slots) s = static_cast<u16>(g_rng.u32());
    g_script.alloc_fails = g_rng.chance(0.12);
    return c;
}

using CreateFn = GUEST_ABI u32 (*)(u64, u64, u64, u64, u64, u64, u64);

// The paths the game's took: refused at the description, by the resource, by the holder; made.
long g_paths[4] = {};
long run(int cases) {
    long n = 0;
    u8 pattern[sizeof g_holder], game_holder[sizeof g_holder];
    for (int t = 0; t < cases; ++t) {
        const Case c = generate();
        for (u8& b : pattern) b = static_cast<u8>(g_rng.u32());
        const u64 out_before = g_rng.next();
        const u64 device = reinterpret_cast<u64>(g_device), desc = reinterpret_cast<u64>(g_desc), out = reinterpret_cast<u64>(&g_out);

        std::memcpy(g_holder, pattern, sizeof g_holder);
        g_out = out_before;
        g_rec.clear();
        decomp_gx::g_buffer_made_entry = nullptr;
        const u32 game_ret = eboot_kit::fn<CreateFn>(kCreateBuffer)(device, out, desc, c.init, c.object_override, c.memory_override, c.placement);
        const std::vector<Call> game = g_rec;
        const u64 game_out = g_out;
        std::memcpy(game_holder, g_holder, sizeof g_holder);

        std::memcpy(g_holder, pattern, sizeof g_holder);
        g_out = out_before;
        g_rec.clear();
        g_told.clear();
        decomp_gx::g_buffer_made_entry = reinterpret_cast<void*>(&fake_told);
        const u32 our_ret = reinterpret_cast<CreateFn>(&gx_create_buffer)(device, out, desc, c.init, c.object_override, c.memory_override, c.placement);

        if (our_ret != game_ret) fail(t, "the return value");
        if (g_rec != game) fail(t, "the calls");
        if (g_out != game_out) fail(t, "the out pointer");
        if (std::memcmp(g_holder, game_holder, sizeof g_holder) != 0) {
            // The game's holder +0x41 and +0x44..+0x47 are its frame's, which
            // the resource maker never writes: the script's bytes stand in
            // for both, so every byte must agree.
            fail(t, "the holder");
        }
        const Call told{kToldId, {game_ret == 0 ? game_out : 0, desc, c.init ? 1ull : 0ull, game_ret}};
        if (g_told.size() != 1 || !(g_told[0] == told)) fail(t, "what the host was told");
        bool made_resource = false, allocated = false;
        for (const Call& k : game) made_resource |= k.who == kMakeId, allocated |= k.who == kAllocId;
        if (!g_quiet) ++g_paths[game_ret == 0 ? 3 : allocated ? 2 : made_resource ? 1 : 0];
        ++n;
    }
    return n;
}

bool caught(int mutant) {
    const int before = g_failed;
    g_resources_mutant = mutant;
    g_quiet = true;
    run(4000);
    g_quiet = false;
    g_resources_mutant = 0;
    const bool found = g_failed > before;
    g_failed = before;
    return found;
}

}  // namespace

int main(int argc, char** argv) {
    eboot_kit::load("gx_resources_test");
    const int cases = argc > 1 ? std::atoi(argv[1]) : 40000;
    eboot_kit::stub(kMakeResource, reinterpret_cast<const void*>(&fake_make));
    eboot_kit::stub(kFreeRenameSlot, reinterpret_cast<const void*>(&fake_free_slot));
    g_vtable[0x58 / 8] = reinterpret_cast<u64>(&fake_alloc);
    g_vtable[0x70 / 8] = reinterpret_cast<u64>(&fake_free);
    for (auto& a : g_allocators) a[0] = reinterpret_cast<u64>(g_vtable);
    const u8* at = eboot_kit::file_at("gx_resources_test", kCreateBuffer);
    if (!at || std::memcmp(at, kEntryCreateBuffer, sizeof kEntryCreateBuffer) != 0) fail(-1, "the entry's instructions");
    const long total = run(cases);
    for (const int mutant : {1, 2, 3, 4, 5}) {
        if (!caught(mutant)) {
            ++g_failed;
            std::printf("FAIL: mutant %d not caught\n", mutant);
        }
    }
    if (g_failed) {
        std::printf("gx_resources_test: %d failures (%ld cases)\n", g_failed, total);
        return 1;
    }
    std::printf("gx_resources_test: ok - %ld cases (%ld refused at the description, %ld by the resource, %ld for the "
                "holder, %ld made), 5 mutants caught\n",
                total, g_paths[0], g_paths[1], g_paths[2], g_paths[3]);
    return 0;
}
