#include "engine/np_test.h"

#include "core/asm_sym.h"
#include "core/elf.h"
#include "core/thunk.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "hle/modules.h"
#include "guest_abi.h"
#include "log.h"

#include <atomic>
#include <map>
#include <mutex>
#include "hle/hle.h"
#include "core/portable.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Binary Ninja addresses (guest VA at the preferred slide).
constexpr std::uint64_t kLuaDispatchByName = 0x1739870;   // (ctx = *(SprjLuaEventMan+8), const char* name)
constexpr std::uint64_t kSprjLuaEventMan = 0x593b0c8;     // SprjLuaEventMan*
constexpr std::uint64_t kWorldChrMan = 0x593e878;         // WorldChrMan*
constexpr std::uint64_t kSprjSessionManager = 0x5940290;  // SprjSessionManager*
constexpr std::uint64_t kGameDataMan = 0x593b130;         // GameDataMan*; local player record at +0x08
constexpr std::uint32_t kRecInsight = 0x84, kRecLevel = 0x90, kRecEchoes = 0x94;  // the record
// The bytes the menu Lua asks before a bell rings (lua_cli_IsOnline ->
// sub_10b4320 -> *(data_5ac7058 + 0xd8); lua_cli_IsOnlineMode -> the network
// flow object's +0x1590; CSNetworkFlowStep::STEP_OnlineMode also gates on its
// +0x1591 and FrpgNetMan +0x9f6/+0x9f8/+0x9fb/+0xa05).
constexpr std::uint64_t kNpMan = 0x5ac7058;              // the NP manager object (data_5ac7048 points at it)
constexpr std::uint64_t kNetFlowPtr = 0x5956678;         // large_unk_network_related*
constexpr std::uint64_t kFrpgNetMan = 0x593b120;         // FrpgNetMan*
constexpr std::uint64_t kSprjEventMan = 0x593b108;       // SprjEventMan*; +0x60 area state; its +0x30 the summon selector

std::uint64_t g_slide = 0;
int g_insight = -1;  // BBHOST_TEST_INSIGHT: written once the world is up (the bells need it)
int g_level = -1;    // BBHOST_TEST_LEVEL: the Chime Maiden wants a host of level 30 or more (common event 9220)
// BBHOST_TEST_FLAGS=id=val,id?,...: event flags written (or read, with "?") once the world is up and again
// 30 s later, after the map's own init event has run. The Chime Maiden's common event 9220 wants area flag
// 2400 off, which m24's event 0 derives from the boss flags 12400160/12401800.
struct FlagReq { std::uint32_t id; int value; };  // value -1 = read only
std::vector<FlagReq> g_flags;
std::vector<std::uint32_t> g_watch;  // BBHOST_NP_WATCH_FLAGS: logged with every change
bool g_flags_again = false;
std::string g_role;
double g_delay = 0;
bool g_done = false;
std::chrono::steady_clock::time_point g_world_at{};
bool g_world_seen = false;
int g_logged = 0;

std::uint64_t guest_of(std::uint64_t bn) { return g_slide + (bn - kPreferredGuestSlide); }
std::uint64_t rd64(std::uint64_t va) {
    std::uint64_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), 8);
    return v;
}

}  // namespace

// BBHOST_NP_TEST=probe: the event side's summon filter, as its selection
// state calls it for a fetched sign (sub_1c72360 at 0x1c7338a ->
// sub_1c74710(state, request, force)). The request is null when the state
// dropped the candidate first (its name in WorldSessionObjectMan+0xb0's
// list); otherwise the filter's answer is the request (summon it) or null.
// Logged with what the area check compares: the request's +0x18 and the
// player's +0x278/+0x27c (ChrIns), each / 10.
namespace {
constexpr std::uint64_t kSosFilterCall = 0x1c7338a, kSosFilter = 0x1c74710;
GUEST_ABI std::int64_t sos_filter_call_hook(std::uint64_t, std::uint64_t* saved) {
    const std::uint64_t req = saved[4];
    const std::int64_t r = hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest_of(kSosFilter))),
                                                        saved[5], req, saved[3]);
    char name[18] = "";
    for (int i = 0; req && i < 17; ++i) {
        const std::uint16_t c = *reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(req + 0x24 + 2 * i));
        if (!c) break;
        name[i] = c < 0x80 ? static_cast<char>(c) : '?';
    }
    const std::uint64_t wcm = rd64(guest_of(kWorldChrMan));
    const std::uint64_t player = wcm ? rd64(wcm + 0x60) : 0;
    auto rd32s = [](std::uint64_t va) { return va ? *reinterpret_cast<const std::int32_t*>(static_cast<std::uintptr_t>(va)) : -1; };
    // The player's vtable slot +0x628 (PlayerIns: sub_1cfe8f0, bit 4 of +0x52c): nonzero makes
    // the filter read the region at +0x27c, always -|region| (sub_1cbba70), so refused, and
    // refuses kinds with flag 2 outright. And the event area's byte the filter needs set,
    // *(*(SprjEventMan+0x60)+8).
    const std::uint64_t vt = player ? rd64(player) : 0;
    const std::uint64_t f628 = vt ? rd64(vt + 0x628) : 0;
    const int v628 = f628 ? static_cast<int>(hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(f628)), player) & 0xff) : -1;
    const std::uint64_t evm = rd64(guest_of(kSprjEventMan));
    const std::uint64_t area = evm ? rd64(evm + 0x60) : 0;
    // The rest of what the filter reads: the request's character key (+0x88, -1: the player);
    // the region's Lua/flag gate (sub_171d7b0(SprjLuaEventMan, &region), nonzero refuses); an
    // effect with stateInfo 191 on the player (the SpEffect container +0x1c8: summary bits
    // +0x21 & 0xe, entries from +8 linked at +0x58, flags +0x1c, param +0x48, stateInfo
    // +0x156); the online switch 0x55282e4; the four-guest sum (cooperators + invaders +
    // others + outgoing work) and the outgoing work.
    std::int32_t region = player ? rd32s(player + (v628 > 0 ? 0x27c : 0x278)) : -1;
    const std::uint64_t lua = rd64(guest_of(kSprjLuaEventMan));
    const int lua_gate = lua ? static_cast<int>(hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest_of(0x171d7b0))),
                                                                              lua, reinterpret_cast<std::uint64_t>(&region)) & 0xff)
                             : -1;
    const std::uint64_t effects = player ? rd64(player + 0x1c8) : 0;
    int bits = -1, state191 = 0;
    if (effects) {
        bits = *reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(effects + 0x21)) & 0xe;
        for (std::uint64_t e = rd64(effects + 8), n = 0; e && n < 64; e = rd64(e + 0x58), ++n) {
            if (*reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(e + 0x1c)) & 0x800c0003u) continue;
            const std::uint64_t param = rd64(e + 0x48);
            if (param && *reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(param + 0x156)) == 0xbf) state191 = 1;
        }
    }
    const std::uint64_t flow = rd64(guest_of(kNetFlowPtr));
    const std::uint64_t slots = flow ? rd64(flow + 0x16f8) : 0;
    const std::uint64_t sel = saved[5];
    const int guests = slots ? rd32s(slots + 8) + rd32s(slots + 0xc) + rd32s(slots + 0x10) + rd32s(sel + 0x1c8) : -1;
    host_log("np probe: summon filter: request %llx kind %d name '%s' force %llu -> %s; area: request %d, player %d / %d; "
             "player +0x628 %llx -> %d; event area byte %d; request +0x88 %d; lua gate %d; stateInfo 191 %d (bits %d); "
             "online switch %d; guests %d, outgoing %d, invaders %d of %d",
             static_cast<unsigned long long>(req), req ? *reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(req + 0x22)) : -1,
             name, static_cast<unsigned long long>(saved[3]), r ? "summon" : "refused", req ? rd32s(req + 0x18) : -1,
             player ? rd32s(player + 0x278) : -1, player ? rd32s(player + 0x27c) : -1,
             static_cast<unsigned long long>(f628 ? f628 - g_slide + kPreferredGuestSlide : 0), v628,
             area ? *reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(area + 8)) : -1,
             req ? rd32s(req + 0x88) : -1, lua_gate, state191, bits, rd32s(guest_of(0x55282e4)), guests, rd32s(sel + 0x1c8),
             slots ? rd32s(slots + 0xc) : -1, rd32s(sel + 0x200));
    saved[-1] = static_cast<std::uint64_t>(r);  // what the caller gets
    return 1;                                     // the call was made here
}
}  // namespace

// BBHOST_SAVE_DUMP=1: a probe on the save config's prefix getter
// (sub_2c328c0: the wide string at +0xb0 of the config, heap when +0xc8 >= 8),
// for the 6.3 Windows boot where the save directory name came out without it.
namespace {
void live_lookup_log(std::uint64_t key);
GUEST_ABI std::int64_t save_prefix_probe(std::uint64_t, const std::uint64_t* saved) {
    const auto* obj = reinterpret_cast<const unsigned char*>(static_cast<std::uintptr_t>(saved[5]));  // rdi
    if (!obj) return 0;
    char hex[0x30 * 3 + 1];
    int n = 0;
    for (int i = 0; i < 0x30; ++i) n += std::snprintf(hex + n, sizeof(hex) - n, "%02x ", obj[0xb0 + i]);
    std::uint64_t cap = 0;
    std::memcpy(&cap, obj + 0xc8, 8);
    char heap[16 * 3 + 1] = "";
    if (cap >= 8) {
        std::uint64_t ptr = 0;
        std::memcpy(&ptr, obj + 0xb0, 8);
        const auto* h = reinterpret_cast<const unsigned char*>(static_cast<std::uintptr_t>(ptr));
        int m = 0;
        for (int i = 0; i < 16; ++i) m += std::snprintf(heap + m, sizeof(heap) - m, "%02x ", h[i]);
    }
    host_log("save-probe: config %p +0xb0: %s heap: %s", static_cast<const void*>(obj), hex, heap);
    // The sibling getter's string (sub_2c327b0): +0x10, capacity at +0x28.
    n = 0;
    for (int i = 0; i < 0x30; ++i) n += std::snprintf(hex + n, sizeof(hex) - n, "%02x ", obj[0x10 + i]);
    std::memcpy(&cap, obj + 0x28, 8);
    heap[0] = 0;
    if (cap >= 8) {
        std::uint64_t ptr = 0;
        std::memcpy(&ptr, obj + 0x10, 8);
        const auto* h = reinterpret_cast<const unsigned char*>(static_cast<std::uintptr_t>(ptr));
        int m = 0;
        for (int i = 0; i < 16; ++i) m += std::snprintf(heap + m, sizeof(heap) - m, "%02x ", h[i]);
    }
    host_log("save-probe: config %p +0x10: %s heap: %s", static_cast<const void*>(obj), hex, heap);
    live_lookup_log(reinterpret_cast<std::uint64_t>(obj));
    {
        // obj_mgmt (0x5940438): the object allocator the config came from, and its vtable.
        const std::uint64_t slide = g_slide ? g_slide : hle_image_slide();
        std::uint64_t mgr = 0, vt = 0;
        std::memcpy(&mgr, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(slide + 0x5940438 - kPreferredGuestSlide)), 8);
        if (mgr) std::memcpy(&vt, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(mgr)), 8);
        std::string e;
        for (int i = 0; vt && i < 27; ++i) {
            std::uint64_t fn = 0;
            std::memcpy(&fn, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(vt + 8 * i)), 8);
            char one[32];
            std::snprintf(one, sizeof(one), " [%x]=0x%llx", 8 * i, static_cast<unsigned long long>(fn - slide + kPreferredGuestSlide));
            e += one;
        }
        host_log("save-probe: obj_mgmt 0x%llx vtable 0x%llx:%s", static_cast<unsigned long long>(mgr - slide + kPreferredGuestSlide),
                 static_cast<unsigned long long>(vt - slide + kPreferredGuestSlide), e.c_str());
    }
    {
        std::uint64_t head[3] = {};
        std::memcpy(head, obj - 8, 24);  // the block header before it, then its first two qwords (vtable?)
        const std::uint64_t slide = g_slide ? g_slide : hle_image_slide();
        host_log("save-probe: header 0x%llx obj[0]=0x%llx (elf 0x%llx) obj[8]=0x%llx", static_cast<unsigned long long>(head[0]),
                 static_cast<unsigned long long>(head[1]), static_cast<unsigned long long>(head[1] - slide + 0x400000),
                 static_cast<unsigned long long>(head[2]));
    }
    // The allocator behind the config (+0x30) and its vtable, as 0x400000-based addresses.
    {
        std::uint64_t alloc = 0;
        std::memcpy(&alloc, obj + 0x30, 8);
        std::uint64_t vt = 0;
        if (alloc) std::memcpy(&vt, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(alloc)), 8);
        const std::uint64_t slide = g_slide ? g_slide : hle_image_slide();
        std::string entries;
        for (int i = 0; vt && i < 20; ++i) {
            std::uint64_t fn = 0;
            std::memcpy(&fn, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(vt + 8 * i)), 8);
            char one[32];
            std::snprintf(one, sizeof(one), " [%x]=0x%llx", 8 * i, static_cast<unsigned long long>(fn - slide + 0x400000));
            entries += one;
        }
        host_log("save-probe: allocator 0x%llx vtable 0x%llx:%s", static_cast<unsigned long long>(alloc - slide + 0x400000),
                 static_cast<unsigned long long>(vt - slide + 0x400000), entries.c_str());
        // The heap's lock object (allocator+8 -> +0x60) and its vtable: what primitive guards allocation.
        if (alloc) {
            std::uint64_t heapobj = 0;
            std::memcpy(&heapobj, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(alloc + 8)), 8);
            const std::uint64_t lock = heapobj + 0x60;
            std::uint64_t lvt = 0;
            std::memcpy(&lvt, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(lock)), 8);
            std::string le;
            for (int i = 0; lvt && i < 8; ++i) {
                std::uint64_t fn = 0;
                std::memcpy(&fn, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(lvt + 8 * i)), 8);
                char one[32];
                std::snprintf(one, sizeof(one), " [%x]=0x%llx", 8 * i, static_cast<unsigned long long>(fn - slide + 0x400000));
                le += one;
            }
            host_log("save-probe: heap lock 0x%llx vtable 0x%llx:%s", static_cast<unsigned long long>(lock),
                     static_cast<unsigned long long>(lvt - slide + 0x400000), le.c_str());
        }
    }
    return 0;
}
}  // namespace

namespace {
// The Dantelion allocator's alloc (sub_2486810: heap, size) and free
// (sub_2487b50: heap, ptr); the save config lives in [0x20aab1000, +0x1000)
// on both platforms, so frees there and mid-sized allocs are logged with
// their caller.
constexpr std::uint64_t kDlAlloc = 0x2486810, kDlFree = 0x2487b50;
// The object allocator class the save config (and Scaleform's root
// SysAlloc) use: the vtable wrappers calls_object_allocator_mb (obj, size,
// align) and its free (+0x70).
constexpr std::uint64_t kObjAlloc = 0x23c5f30, kObjFree = 0x23c5fc0;
constexpr std::uint8_t kObjFreePrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x49, 0x89, 0xf6};
}  // namespace
extern "C" GUEST_ABI std::uint64_t dl_alloc_result(std::uint64_t block);
extern "C" GUEST_ABI void dl_free_note(std::uint64_t ptr);
extern "C" void dl_alloc_ret();
namespace {
constexpr std::uint8_t kDlPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};
std::uint64_t elf_of(std::uint64_t va) { return va - (g_slide ? g_slide : hle_image_slide()) + kPreferredGuestSlide; }
GUEST_ABI std::int64_t dl_free_probe(std::uint64_t, const std::uint64_t* saved) {
    const std::uint64_t ptr = saved[4];  // rsi
    dl_free_note(ptr);
    std::uint64_t hdr = 0;
    if (ptr >= 0x200000000ull && ptr < 0x230000000ull) {
        std::memcpy(&hdr, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(ptr - 8)), 8);
    }
    const std::uint64_t size = hdr & ~3ull;
    constexpr std::uint64_t kTarget = 0x20aab1728ull;  // the config under wine (0x20aab1388 on Linux)
    if ((ptr <= kTarget && kTarget < ptr + size && size < (64u << 20)) || (ptr >= 0x20aab1000ull && ptr < 0x20aab2000ull)) {
        host_log("dl-free: 0x%llx size 0x%llx heap 0x%llx from 0x%llx", static_cast<unsigned long long>(ptr),
                 static_cast<unsigned long long>(hdr & ~3ull), static_cast<unsigned long long>(saved[5]),
                 static_cast<unsigned long long>(elf_of(saved[6])));
    }
    return 0;
}
// Live blocks of the allocator, from a return hook on alloc: the entry hook
// swaps the caller's return address for dl_alloc_ret (asm below), which
// hands rax to dl_alloc_result and jumps on to the real return address.
// An allocation overlapping a live block is the double allocation the
// Windows save-name fault looks like.
struct Pending {
    std::uint64_t size, ret, heap;
};
thread_local std::vector<Pending> t_pending;
struct Live {
    std::uint64_t size, heap, site;
    std::uint32_t tid;
};
std::mutex g_live_mu;
std::map<std::uint64_t, Live> g_live;  // by block start
std::atomic<int> g_overlaps{0};

void live_lookup_log(std::uint64_t key) {
    // Which live allocator block holds the config, if any.
    std::lock_guard<std::mutex> lk(g_live_mu);
    auto it = g_live.upper_bound(key);
    if (it != g_live.begin() && std::prev(it)->first + std::prev(it)->second.size > key) {
        auto b = std::prev(it);
        host_log("save-probe: config is in live block 0x%llx+0x%llx allocated from 0x%llx on tid %u (%zu live blocks)",
                 static_cast<unsigned long long>(b->first), static_cast<unsigned long long>(b->second.size),
                 static_cast<unsigned long long>(elf_of(b->second.site)), b->second.tid, g_live.size());
    } else {
        host_log("save-probe: config is in NO live allocator block (%zu live blocks)", g_live.size());
    }
}

GUEST_ABI std::int64_t dl_alloc_probe(std::uint64_t, const std::uint64_t* saved) {
    t_pending.push_back({saved[4], saved[6], saved[5]});
    const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&dl_alloc_ret);
    return 0;
}
}  // namespace

extern "C" GUEST_ABI std::uint64_t dl_alloc_result(std::uint64_t block) {
    const Pending p = t_pending.back();
    t_pending.pop_back();
    if (block) {
        std::lock_guard<std::mutex> lk(g_live_mu);
        auto it = g_live.upper_bound(block);
        if (it != g_live.begin()) {
            auto prev = std::prev(it);
            if (prev->first + prev->second.size > block && g_overlaps.fetch_add(1) < 20) {
                host_log("dl-OVERLAP: new block 0x%llx+0x%llx (heap 0x%llx, from 0x%llx, tid %u) inside live 0x%llx+0x%llx (from 0x%llx, tid %u)",
                         static_cast<unsigned long long>(block), static_cast<unsigned long long>(p.size),
                         static_cast<unsigned long long>(p.heap), static_cast<unsigned long long>(elf_of(p.ret)),
                         host_thread_id(), static_cast<unsigned long long>(prev->first),
                         static_cast<unsigned long long>(prev->second.size),
                         static_cast<unsigned long long>(elf_of(prev->second.site)), prev->second.tid);
            }
        }
        if (it != g_live.end() && block + p.size > it->first && g_overlaps.fetch_add(1) < 20) {
            host_log("dl-OVERLAP: new block 0x%llx+0x%llx (from 0x%llx, tid %u) runs into live 0x%llx (from 0x%llx, tid %u)",
                     static_cast<unsigned long long>(block), static_cast<unsigned long long>(p.size),
                     static_cast<unsigned long long>(elf_of(p.ret)), host_thread_id(),
                     static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(elf_of(it->second.site)),
                     it->second.tid);
        }
        g_live[block] = Live{p.size, p.heap, p.ret, host_thread_id()};
    }
    return p.ret;
}
extern "C" GUEST_ABI void dl_free_note(std::uint64_t ptr) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    g_live.erase(ptr);
}
extern "C" void* g_dl_alloc_result_thunk = nullptr;
extern "C" void dl_alloc_ret();
asm(".text\n.globl " ASM_SYM(dl_alloc_ret) "\n" ASM_TYPE_FUNCTION(dl_alloc_ret) ASM_SYM(dl_alloc_ret) ":\n"
    "push %rax\n push %rdx\n mov %rax, %rdi\n call *" ASM_SYM(g_dl_alloc_result_thunk) "(%rip)\n"
    "mov %rax, %r11\n pop %rdx\n pop %rax\n jmp *%r11\n");

void np_test_install(ElfImage* image) {
    g_slide = 0;
    if (const char* d = std::getenv("BBHOST_SAVE_DUMP"); d && d[0] == '1') {
        g_slide = image->mem.slide;
        const bool f = engine_prologue_hook(image, image->mem.slide + (kDlFree - kPreferredGuestSlide), kDlPrologue,
                                            sizeof(kDlPrologue), reinterpret_cast<void*>(&dl_free_probe));
        g_dl_alloc_result_thunk = hle_wrap_fn(reinterpret_cast<void*>(&dl_alloc_result));
        const bool a = engine_prologue_hook(image, image->mem.slide + (kDlAlloc - kPreferredGuestSlide), kDlPrologue,
                                            sizeof(kDlPrologue), reinterpret_cast<void*>(&dl_alloc_probe));
        const bool of = engine_prologue_hook(image, image->mem.slide + (kObjFree - kPreferredGuestSlide), kObjFreePrologue,
                                             sizeof(kObjFreePrologue), reinterpret_cast<void*>(&dl_free_probe));
        const bool oa = engine_prologue_hook(image, image->mem.slide + (kObjAlloc - kPreferredGuestSlide), kDlPrologue,
                                             sizeof(kDlPrologue), reinterpret_cast<void*>(&dl_alloc_probe));
        host_log("save-probe: allocator hooks free %s alloc %s; object allocator free %s alloc %s", f ? "on" : "REFUSED",
                 a ? "on" : "REFUSED", of ? "on" : "REFUSED", oa ? "on" : "REFUSED");
        // cmp qword [rdi+0xc8], 8; lea rax, [rdi+0xb0]: 15 bytes, no relative jumps.
        static const std::uint8_t kGetterPrologue[] = {0x48, 0x83, 0xbf, 0xc8, 0x00, 0x00, 0x00, 0x08,
                                                       0x48, 0x8d, 0x87, 0xb0, 0x00, 0x00, 0x00};
        const std::uint64_t at = image->mem.slide + (0x2c328c0 - kPreferredGuestSlide);  // a 0x400000-based address
        if (!engine_prologue_hook(image, at, kGetterPrologue, sizeof(kGetterPrologue),
                                  reinterpret_cast<void*>(&save_prefix_probe))) {
            host_log("save-probe: the prefix getter did not take the hook");
        }
    }
    const char* e = std::getenv("BBHOST_NP_TEST");
    const char* ins = std::getenv("BBHOST_TEST_INSIGHT");
    const char* lvl = std::getenv("BBHOST_TEST_LEVEL");
    if ((!e || !e[0]) && (!ins || !ins[0]) && (!lvl || !lvl[0])) return;
    if (!e || !e[0]) e = "none";  // Insight only, no event
    if (!image || image->sha256 != kEboot109Sha256) {
        host_log("np test: not the 1.09 build; ignored");
        return;
    }
    std::string spec = e;
    const std::size_t colon = spec.find(':');
    g_role = spec.substr(0, colon);
    g_delay = g_role == "host" ? 60.0 : 20.0;
    if (colon != std::string::npos) g_delay = std::atof(spec.c_str() + colon + 1);
    if (g_role != "guest" && g_role != "host" && g_role != "none" && g_role != "probe") {
        host_log("np test: BBHOST_NP_TEST=%s is not guest or host; ignored", e);
        return;
    }
    g_slide = image->mem.slide;
    if (g_role == "probe" &&
        !engine_call_site_hook(image, guest_of(kSosFilterCall), guest_of(kSosFilter), reinterpret_cast<void*>(&sos_filter_call_hook))) {
        host_log("np probe: the summon filter's call site did not take the hook");
    }
    if (const char* i = std::getenv("BBHOST_TEST_INSIGHT"); i && i[0]) g_insight = std::atoi(i);
    if (lvl && lvl[0]) g_level = std::atoi(lvl);
    if (const char* w = std::getenv("BBHOST_NP_WATCH_FLAGS"); w && w[0]) {
        for (const char* q = w; *q;) {
            char* end = nullptr;
            const unsigned long id = std::strtoul(q, &end, 10);
            if (end == q) break;
            if (id) g_watch.push_back(static_cast<std::uint32_t>(id));
            q = *end == ',' ? end + 1 : end;
        }
    }
    if (const char* f = std::getenv("BBHOST_TEST_FLAGS"); f && f[0]) {
        std::string list = f;
        std::size_t at = 0;
        while (at < list.size()) {
            std::size_t end = list.find(',', at);
            if (end == std::string::npos) end = list.size();
            std::string item = list.substr(at, end - at);
            at = end + 1;
            if (item.empty()) continue;
            FlagReq r{static_cast<std::uint32_t>(std::atoll(item.c_str())), -1};
            if (const std::size_t eq = item.find('='); eq != std::string::npos) r.value = std::atoi(item.c_str() + eq + 1);
            g_flags.push_back(r);
        }
    }
    host_log("np test: %s, %.0f s after the world is up%s%s", g_role.c_str(), g_delay,
             g_insight >= 0 ? " (Insight written first)" : "", g_level >= 0 ? " (level written first)" : "");
}

namespace {
void give_insight() {
    if (g_insight < 0 && g_level < 0) return;
    const std::uint64_t gdm = rd64(guest_of(kGameDataMan));
    const std::uint64_t rec = gdm ? rd64(gdm + 8) : 0;
    if (!rec) {
        host_log("np test: no GameDataMan player record; Insight/level not written");
        return;
    }
    std::uint32_t insight = 0, level = 0, echoes = 0;
    std::memcpy(&insight, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(rec + kRecInsight)), 4);
    std::memcpy(&level, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(rec + kRecLevel)), 4);
    std::memcpy(&echoes, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(rec + kRecEchoes)), 4);
    const std::uint32_t want = g_insight >= 0 ? static_cast<std::uint32_t>(g_insight) : insight;
    const std::uint32_t want_level = g_level >= 0 ? static_cast<std::uint32_t>(g_level) : level;
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(rec + kRecInsight)), &want, 4);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(rec + kRecLevel)), &want_level, 4);
    host_log("np test: player record 0x%llx: level %u -> %u, echoes %u, Insight %u -> %u",
             static_cast<unsigned long long>(rec), level, want_level, echoes, insight, want);
}
}  // namespace

// SprjEventFlagMan (0x593b100): +0x1c flags per block, +0x20 bytes per block, +0x28 the flat blocks, +0x38 an
// rb-tree of {block index +0x20, kind +0x28 (1 flat, 2 pointer at +0x30), +0x30}; bit (~offset)&7 of byte offset>>3
// (the layout lua_cli_SetEventFlag 0x172e5a0 walks).
constexpr std::uint64_t kEventFlagMan = 0x593b100, kEventState = 0x593b0d8;
constexpr std::uint64_t kGetEventFlag = 0x17cfd80;  // (SprjEventFlagMan*, flag, 8) -> u8
std::uint32_t rd32(std::uint64_t va) {
    std::uint32_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), 4);
    return v;
}
bool flag_byte(std::uint32_t flag, std::uint64_t* byte, std::uint8_t* mask) {
    if (!rd64(guest_of(kEventState))) return false;
    const std::uint64_t mgr = rd64(guest_of(kEventFlagMan));
    if (!mgr) return false;
    const std::uint32_t per_block = rd32(mgr + 0x1c);
    if (!per_block) return false;
    const std::uint32_t block = flag / per_block, off = flag - block * per_block;
    const std::uint64_t head = rd64(mgr + 0x38);
    if (!head) return false;
    std::uint64_t node = rd64(head + 8), found = head;
    while (node && *reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(node + 0x19)) == 0) {
        if (rd32(node + 0x20) >= block) {
            found = node;
            node = rd64(node);
        } else {
            node = rd64(node + 0x10);
        }
    }
    if (found == head || rd32(found + 0x20) != block) return false;
    const std::uint32_t kind = rd32(found + 0x28);
    std::uint64_t data = 0;
    if (kind == 2) data = rd64(found + 0x30);
    else if (kind == 1) data = rd64(mgr + 0x28) + static_cast<std::uint64_t>(rd32(mgr + 0x20)) * rd32(found + 0x30);
    if (!data) return false;
    *byte = data + (off >> 3);
    *mask = static_cast<std::uint8_t>(1u << ((~off) & 7));
    return true;
}
void apply_flags(const char* when) {
    for (const FlagReq& r : g_flags) {
        std::uint64_t byte = 0;
        std::uint8_t mask = 0;
        if (!flag_byte(r.id, &byte, &mask)) {
            host_log("np test: flag %u: no block (%s)", r.id, when);
            continue;
        }
        auto* p = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(byte));
        const int before = (*p & mask) ? 1 : 0;
        // The game's own reader (what lua_cli_GetEventFlagValue calls), as a check on the walk above.
        const std::uint64_t mgr = rd64(guest_of(kEventFlagMan));
        const int game_says = static_cast<int>(hle_call_guest<std::int64_t>(
                                  reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest_of(kGetEventFlag))), mgr,
                                  static_cast<std::uint64_t>(r.id), 8ull) & 0xff);
        if (r.value >= 0) {
            if (r.value) *p |= mask; else *p &= static_cast<std::uint8_t>(~mask);
        }
        host_log("np test: flag %u: %d (game reads %d) -> %d (%s)", r.id, before, game_says, (*p & mask) ? 1 : 0, when);
    }
}

void np_test_tick() {
    if (!g_slide || g_done) return;
    const std::uint64_t wcm = rd64(guest_of(kWorldChrMan));
    const std::uint64_t ssm = rd64(guest_of(kSprjSessionManager));
    if (!wcm || !ssm) {
        g_world_seen = false;
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!g_world_seen) {
        g_world_seen = true;
        g_world_at = now;
        host_log("np test: the world is up; %s event in %.0f s", g_role.c_str(), g_delay);
        give_insight();
        apply_flags("world up");
        g_flags_again = !g_flags.empty();
        if (g_role == "none") g_done = true;
        return;
    }
    if (g_flags_again && std::chrono::duration<double>(now - g_world_at).count() > 30.0) {
        g_flags_again = false;
        apply_flags("30 s later");
    }
    if (g_role == "probe") {
        const double t = std::chrono::duration<double>(now - g_world_at).count();
        // Every second: where this instance thinks everyone is (BBHOST_NP_POS=0 turns it off). The
        // line carries the wall clock so tools/desync.py can line up two instances' logs: "np pos
        // <epoch ms> me x y z | <slot> <handle> x y z ...". Position is the physics module's
        // (ChrIns+0x58 -> +8 -> +0x3b0 -> +0x68 -> +0x1e0, what the floating plates use); the net
        // players come from the session's slot table (net flow +0x16f8: count +0x14, handles
        // +0x1c+i*0x14) through WorldChrMan's handle table at +0x850, as sub_1a439b0 walks them.
        static double last_pos = -1;
        static const bool pos_on = !(std::getenv("BBHOST_NP_POS") && std::getenv("BBHOST_NP_POS")[0] == '0');
        static const double pos_every = [] {  // BBHOST_NP_POS_HZ: 1 by default, up to 20 for a lag estimate
            const char* hz = std::getenv("BBHOST_NP_POS_HZ");
            const double v = hz && hz[0] ? std::atof(hz) : 1.0;
            return 1.0 / (v > 0 && v <= 20 ? v : 1.0);
        }();
        if (pos_on && t - last_pos >= pos_every) {
            last_pos = t;
            auto chr_pos = [](std::uint64_t chr, float* xyz) {
                const std::uint64_t a = chr ? rd64(chr + 0x58) : 0;
                const std::uint64_t b = a ? rd64(a + 8) : 0;
                const std::uint64_t c = b ? rd64(b + 0x3b0) : 0;
                const std::uint64_t d = c ? rd64(c + 0x68) : 0;
                if (!d) return false;
                std::memcpy(xyz, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(d + 0x1e0)), 12);
                return true;
            };
            const std::uint64_t me = rd64(wcm + 0x60);
            float mp[3] = {0, 0, 0};
            const bool have_me = chr_pos(me, mp);
            std::string others;
            const std::uint64_t flow = rd64(guest_of(kNetFlowPtr));
            const std::uint64_t slots = flow ? rd64(flow + 0x16f8) : 0;
            const int count = slots ? static_cast<int>(rd32(slots + 0x14)) : 0;
            for (int i = 0; i < count && i < 7; ++i) {
                const std::uint32_t h = rd32(slots + 0x1c + static_cast<std::uint64_t>(i) * 0x14);
                if (h == 0xffffffffu) continue;
                const std::uint64_t set = rd64(wcm + 0x850 + static_cast<std::uint64_t>((h >> 14) & 0x3f) * 8);
                if (!set || (h & 0x3fff) >= rd32(set)) continue;
                const std::uint64_t entry = rd64(set + 8) + static_cast<std::uint64_t>(h & 0x3fff) * 0x38;
                float p[3] = {0, 0, 0};
                if (!chr_pos(rd64(entry), p)) continue;
                char b[96];
                std::snprintf(b, sizeof(b), " | %d %08x %.3f %.3f %.3f", i, h, p[0], p[1], p[2]);
                others += b;
            }
            const auto wall = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count();
            host_log("np pos %lld me %.3f %.3f %.3f%s", static_cast<long long>(wall), have_me ? mp[0] : 0.f,
                     have_me ? mp[1] : 0.f, have_me ? mp[2] : 0.f, others.c_str());
        }
        // Every frame: the summon handshake's own state, logged when it changes, so a summon
        // that stalls between the guest's request and the host's room payload shows where. The
        // event side's SprjEventSosSelectionState (*(SprjEventMan+0x60)+0x30):
        // pending guests +0x78, the descriptor being invited +0x90, accept +0xa4, result +0xa8,
        // the first outgoing work entry (list +0x1c0: SOSID +0, state +0xc) and their count
        // +0x1c8, +0x230; the native SosSignMan (FrpgNetMan+0xc50): the staged room payload
        // (+0x130 member, +0x140 length - sub_18baec0) and the passive entries +0x160; the
        // session state (SprjSessionManager+0x124).
        {
            auto rd32 = [](std::uint64_t va) { return *reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va)); };
            const std::uint64_t evm = rd64(guest_of(kSprjEventMan));
            const std::uint64_t area = evm ? rd64(evm + 0x60) : 0;
            const std::uint64_t sel = area ? rd64(area + 0x30) : 0;
            const std::uint64_t fnm = rd64(guest_of(kFrpgNetMan));
            const std::uint64_t step = fnm ? rd64(fnm + 0xc50) : 0;
            std::uint32_t v[12] = {};
            if (sel) {
                v[0] = rd32(sel + 0x78);
                v[1] = rd32(sel + 0x90);
                v[2] = rd32(sel + 0xa4);
                v[3] = rd32(sel + 0xa8);
                v[4] = rd32(sel + 0x1c8);
                v[5] = rd32(sel + 0x230);
                const std::uint64_t head = rd64(sel + 0x1c0);
                const std::uint64_t node = head ? rd64(head) : 0;
                const std::uint64_t work = node && node != head ? rd64(node + 0x10) : 0;
                v[6] = work ? rd32(work) : 0xffffffffu;
                v[7] = work ? rd32(work + 0xc) : 0xffffffffu;
            }
            if (step) {
                v[8] = rd32(step + 0x130);
                v[9] = rd32(step + 0x140);
                v[10] = rd32(step + 0x160);
            }
            v[11] = rd32(ssm + 0x124);
            static std::uint32_t prev[12];
            static bool have = false;
            if (!have || std::memcmp(prev, v, sizeof(v)) != 0) {
                have = true;
                std::memcpy(prev, v, sizeof(v));
                const auto wall = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch()).count();
                host_log("np probe: summon state %lld: pending %u inviting %d accept %u result %u outgoing %u "
                         "(first SOSID %d state %d) +0x230 %u; staged member %u len %u; passive %u; session %u",
                         static_cast<long long>(wall), v[0], static_cast<int>(v[1]), v[2], v[3], v[4],
                         static_cast<int>(v[6]), static_cast<int>(v[7]), v[5], v[8], v[9], v[10], v[11]);
            }
        }
        // Every frame, BBHOST_NP_WATCH_FLAGS (a comma list of event flags, e.g. a maiden's
        // 12414220-12414223 in Central Yharnam) and whether the player carries an effect with
        // stateInfo 191 (the maiden's SpEffect 9020, "invadable"): one line whenever any changes.
        if (!g_watch.empty()) {
            std::string sig;
            for (const std::uint32_t id : g_watch) {
                std::uint64_t byte = 0;
                std::uint8_t mask = 0;
                const bool known = flag_byte(id, &byte, &mask);
                sig += " " + std::to_string(id) + "=" +
                       (known ? ((*reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(byte)) & mask) ? "1" : "0") : "?");
            }
            int state191 = 0;
            const std::uint64_t me = rd64(wcm + 0x60);
            const std::uint64_t effects = me ? rd64(me + 0x1c8) : 0;
            for (std::uint64_t e = effects ? rd64(effects + 8) : 0, n = 0; e && n < 64; e = rd64(e + 0x58), ++n) {
                if (*reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(e + 0x1c)) & 0x800c0003u) continue;
                const std::uint64_t param = rd64(e + 0x48);
                if (param && *reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(param + 0x156)) == 0xbf) state191 = 1;
            }
            sig += " stateInfo191=" + std::to_string(state191);
            // What the maiden's 3[29] counts (sub_18510a0): client type 0, cooperators
            // (sub_19bdc20: invade types 1/5/7/19), and 1, invaders (sub_19bde10: invade type 8),
            // over the session's slots (net flow +0x16f8: handles at +0x1c + 0x14 * i, five of
            // them; ChrIns vtable +0x558 the invade type), the player's own left out.
            const std::uint64_t flow = rd64(guest_of(kNetFlowPtr));
            const std::uint64_t slots = flow ? rd64(flow + 0x16f8) : 0;
            if (slots) {
                auto call1 = [](std::uint64_t fn, std::uint64_t a) {
                    return static_cast<int>(hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest_of(fn))), a) & 0xffffffff);
                };
                sig += " coop=" + std::to_string(call1(0x19bdc20, slots)) + " invaders=" + std::to_string(call1(0x19bde10, slots)) + " slots";
                for (int i = 0; i < 5; ++i) {
                    const std::uint32_t h = rd32(slots + 0x1c + 0x14 * static_cast<std::uint64_t>(i));
                    if (h == 0xffffffffu) {
                        sig += " -";
                        continue;
                    }
                    const std::uint64_t set = rd64(wcm + 0x850 + static_cast<std::uint64_t>((h >> 14) & 0x3f) * 8);
                    std::uint64_t chr = 0;
                    if (set && (h & 0x3fff) < rd32(set)) chr = rd64(rd64(set + 8) + static_cast<std::uint64_t>(h & 0x3fff) * 0x38);
                    int type = -1;
                    if (chr) {
                        const std::uint64_t f = rd64(rd64(chr) + 0x558);
                        type = static_cast<int>(hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(f)), chr) & 0xffffffff);
                    }
                    char b[40];
                    std::snprintf(b, sizeof(b), " %x:%d%s", h, type, me && h == rd32(me + 8) ? "(me)" : "");
                    sig += b;
                }
            }
            static std::string prev_sig;
            if (sig != prev_sig) {
                prev_sig = sig;
                const auto wall = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch()).count();
                host_log("np probe: watch %lld:%s", static_cast<long long>(wall), sig.c_str());
            }
        }
        // Every 5 s: the online gates, so a refused bell says which one.
        static double last = -1;
        if (t - last < 5.0) return;
        last = t;
        const std::uint64_t flow = rd64(guest_of(kNetFlowPtr));
        const std::uint64_t fnm = rd64(guest_of(kFrpgNetMan));
        auto rd8 = [](std::uint64_t va) { return va ? *reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(va)) : 0xff; };
        host_log("np probe: IsOnline %u; flow online_mode %u +1591 %u +15a0 %u; FrpgNetMan 9f6 %u 9f8 %u 9fb %u a05 %02x; session state %u",
                 rd8(guest_of(kNpMan) + 0xd8), rd8(flow ? flow + 0x1590 : 0), rd8(flow ? flow + 0x1591 : 0),
                 rd8(flow ? flow + 0x15a0 : 0), rd8(fnm ? fnm + 0x9f6 : 0), rd8(fnm ? fnm + 0x9f8 : 0),
                 rd8(fnm ? fnm + 0x9fb : 0), rd8(fnm ? fnm + 0xa05 : 0),
                 static_cast<unsigned>(rd64(ssm + 0x124) & 0xffffffffu));
        // The SummonStepManager (FrpgNetMan+0xc50), for the summon that stalls:
        // the staged type-1 item (+0x130 member, +0x138 data, +0x140 len, from
        // sub_18baec0), the guest's passive entries (+0x160) and the first
        // one's sessionReady (+0xc8). (sub_1c72360's own +0x78/+0x90/+0xa4
        // are on its argument, a different object - not read here.)
        // The summon selector, the game's SosSignMan (sub_1c72360's object,
        // *(SprjEventMan+0x60)+0x30): the host's pending guests +0x78, the id
        // being invited +0x90, the kick id +0x88, accept +0xa4, result +0xa8,
        // outgoing list +0x1c8, the guest's accept gate +0x1f8, the most
        // cooperators +0x1fc and invaders +0x200 a summon may make (3 and 3
        // from its constructor, sub_1c70c50), the retry timer +0x210. The
        // candidate list is the SummonStepManager's (FrpgNetMan+0xc50, +0x1a8):
        // {sign (kind +8, ack flag +0xef), request (row +0x22, the kind),
        // processed}. Plus SprjSessionManager+0x280's state (vtable slot 0xb),
        // which masks new invites while 2..3.
        const std::uint64_t evm = rd64(guest_of(kSprjEventMan));
        const std::uint64_t area = evm ? rd64(evm + 0x60) : 0;
        const std::uint64_t sel = area ? rd64(area + 0x30) : 0;
        if (sel) {
            auto rd32 = [](std::uint64_t va) { return *reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va)); };
            auto rdf = [](std::uint64_t va) { return *reinterpret_cast<const float*>(static_cast<std::uintptr_t>(va)); };
            std::string cands;
            const std::uint64_t step = fnm ? rd64(fnm + 0xc50) : 0;
            const std::uint64_t head = step ? rd64(step + 0x1a8) : 0;
            int n = 0;
            for (std::uint64_t node = head ? rd64(head) : 0; node && node != head && n < 4; node = rd64(node), ++n) {
                const std::uint64_t e = rd64(node + 0x10);
                if (!e) continue;
                const std::uint64_t sign = rd64(e), req = rd64(e + 8);
                char b[96];
                std::snprintf(b, sizeof(b), " [sign %llx kind %d ack %u row %u done %u]", static_cast<unsigned long long>(sign),
                              sign ? static_cast<int>(rd32(sign + 8) & 0xff) : -1,
                              sign ? (*reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(sign + 0xef)) & 1u) : 0u,
                              req ? *reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(req + 0x22)) : 0u,
                              *reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(e + 0x10)));
                cands += b;
            }
            const std::uint64_t so = rd64(ssm + 0x280);
            long long so_state = -1;
            if (so && !(*reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(so + 0xd8)) & 1u)) {
                const std::uint64_t vt = rd64(so);
                const std::uint64_t fn = vt ? rd64(vt + 0x58) : 0;
                if (fn) so_state = hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(fn)), so) & 0xffffffff;
            }
            host_log("np probe: selector %llx: pending %u inviting %d kick %d accept %u result %u outgoing %u gate %u max coop %d invaders %d timer %.1f; SO %llx state %lld; candidates%s",
                     static_cast<unsigned long long>(sel), rd32(sel + 0x78), static_cast<int>(rd32(sel + 0x90)),
                     static_cast<int>(rd32(sel + 0x88)), rd32(sel + 0xa4), rd32(sel + 0xa8), rd32(sel + 0x1c8), rd32(sel + 0x1f8),
                     static_cast<int>(rd32(sel + 0x1fc)), static_cast<int>(rd32(sel + 0x200)),
                     rdf(sel + 0x210), static_cast<unsigned long long>(so), so_state, cands.empty() ? " none" : cands.c_str());
            // What sub_1c74710 checks before it starts a summon of a kind-8
            // (Sinister Resonant Bell) sign: the session's members by
            // category (flow+0x16f8: +8 cooperators, +0xc invaders, +0x10
            // others, +0x14 all, +0x98 must be -1) against the room's most
            // (*0x593d6d0 + 0xc) and four guests; flow+0x1592 set; the
            // player's state (+0x78: 0); the kind's cooldown (0x596e560 +
            // 12 * kind: armed, then two timers).
            const std::uint64_t slots = flow ? rd64(flow + 0x16f8) : 0;
            const std::uint64_t room = rd64(guest_of(0x593d6d0));
            // The selection state drops a candidate whose character name (request +0x24) is in
            // WorldSessionObjectMan+0xb0's list (+0x10; each node's +0x10 the UTF-16 name) before
            // the filter sees it (sub_1c72360 at 0x1c732f8). Names, ASCII only.
            auto utf16 = [](std::uint64_t p) {
                std::string s;
                for (int i = 0; p && i < 17; ++i) {
                    const std::uint16_t c = *reinterpret_cast<const std::uint16_t*>(static_cast<std::uintptr_t>(p + 2 * i));
                    if (!c) break;
                    s += c < 0x80 ? static_cast<char>(c) : '?';
                }
                return s;
            };
            std::string names;
            const std::uint64_t helper = room ? rd64(room + 0xb0) : 0;
            const std::uint64_t list = helper ? rd64(helper + 0x10) : 0;
            int listed = 0;
            for (std::uint64_t node = list ? rd64(list) : 0; node && node != list && listed < 6; node = rd64(node), ++listed) {
                names += " '" + utf16(rd64(node + 0x10)) + "'";
            }
            for (std::uint64_t node = head ? rd64(head) : 0, k = 0; node && node != head && k < 4; node = rd64(node), ++k) {
                const std::uint64_t e = rd64(node + 0x10);
                const std::uint64_t req = e ? rd64(e + 8) : 0;
                if (req) names += " cand '" + utf16(req + 0x24) + "'";
            }
            host_log("np probe: summon names: %d listed%s", listed, names.c_str());
            const std::uint64_t player = wcm ? rd64(wcm + 0x60) : 0;
            const std::uint64_t cd = guest_of(0x596e560 + 12 * 8);
            host_log("np probe: summon gates: members coop %d invaders %d other %d all %d (+0x98 %d) room max %d; "
                     "flow+0x1592 %u; player state %d; kind 8 cooldown %u %.1f %.1f",
                     slots ? static_cast<int>(rd32(slots + 8)) : -1, slots ? static_cast<int>(rd32(slots + 0xc)) : -1,
                     slots ? static_cast<int>(rd32(slots + 0x10)) : -1, slots ? static_cast<int>(rd32(slots + 0x14)) : -1,
                     slots ? static_cast<int>(rd32(slots + 0x98)) : -1, room ? static_cast<int>(rd32(room + 0xc)) : -1,
                     rd8(flow ? flow + 0x1592 : 0), player ? static_cast<int>(rd32(player + 0x78)) : -1, rd32(cd),
                     rdf(cd + 4), rdf(cd + 8));
        }
        const std::uint64_t mgr = fnm ? rd64(fnm + 0xc50) : 0;
        if (mgr) {
            auto rd32 = [](std::uint64_t va) { return *reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(va)); };
            const std::uint64_t head = rd64(mgr + 0x158);
            const std::uint64_t first = head ? rd64(head) : 0;
            const std::uint64_t entry = (first && first != head) ? rd64(first + 0x10) : 0;
            host_log("np probe: summon mgr: staged member %u data %llx len %u; passive entries %u first ready %u",
                     rd32(mgr + 0x130), static_cast<unsigned long long>(rd64(mgr + 0x138)), rd32(mgr + 0x140),
                     rd32(mgr + 0x160), entry ? rd32(entry + 0xc8) : 0);
        }
        return;
    }
    if (std::chrono::duration<double>(now - g_world_at).count() < g_delay) return;
    g_done = true;
    const std::uint64_t man = rd64(guest_of(kSprjLuaEventMan));
    if (!man) {
        host_log("np test: no SprjLuaEventMan; nothing dispatched");
        return;
    }
    const std::uint64_t ctx = rd64(man + 8);
    const char* name = g_role == "host" ? "OnEvent_Call_SOS" : "OnEvent_SendSoulSign_NormalCoop";
    const auto r = hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest_of(kLuaDispatchByName))),
                                                ctx, name);
    host_log("np test: dispatched Lua event %s -> %lld (session state %u)", name, static_cast<long long>(r),
             static_cast<unsigned>(rd64(ssm + 0x124) & 0xffffffffu));
    (void)g_logged;
}
