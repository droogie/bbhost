// The effects' node emission, sub_2c4e6b0 at Binary Ninja 0x2c4e6b0 (docs/decomp.md,
// the effects area): a staged node's emitting state, which the node update
// (sub_2c4df80, decomp/sfx/node_update.cpp) calls from its states 0 (after the
// record's set-up), 1 and 3 (the controller back on the record), as ours - every
// read, write and call in the game's order with the game's arguments - so that its
// two halves can be called apart (node_emit.h):
//   - the edges: the controller's state (+0xc8: sub_2c82190, sub_2c821b0) against
//     the node's record (+0xd8): a switch to another record index, a reset past the
//     definition's records (3) or the new record's set-up (sub_2c4f660; 9 when it
//     fails); then by the record's flags (+4): bit 0, a record that waits for its end
//     (+0xec) and lingers (sub_2c4ec70), the root of an ending instance or a
//     stopping node too under bit 8; bit 1, children, which end (6) past +0xec or,
//     under bit 8, when the instance ends or the node stops, and linger under bit 9 or
//     a controller whose state no longer holds; bit 2, CPU particles, which linger
//     unless the controller holds, the time is before +0xec and, under bits 8-9, the
//     instance goes on;
//   - the event: once the time reaches the next (+0xe4) with fewer than +0xe8 made
//     (+0xe0 counts them; past that bit 4 parks the next at FLT_MAX, else the node
//     lingers): the emitter's count (record +0x28's slot +0x58, given the time)
//     added to the container (sub_2c4c7f0), or the children (sub_2c77050 under
//     bit 5, else sub_2c6d780), then the record's binder (+0x98, sub_2c54280) and
//     the next time, +0xe4 = the time + the interval (+0xf0).
// One update makes at most one event, so the game's emission rate is its update
// rate: 30 a second at 30, though an interval of 1/60 s - 8,421 of the game's
// 15,330 CPU emitter records - asks for 60. The emission clock runs the edges on
// the node's whole steps and the events on 1/60 s sub-steps of its time (the node
// update): the effects emit as at 60 updates a second, the rest counts as at 30.
#include "decomp/sfx/node_emit.h"

#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <atomic>
#include <cstdint>

using namespace decomp;

#ifdef DECOMP_SFX_NODE_EMIT_MUTANTS
extern int g_sfx_emit_mutant;
#define MUTANT(n) (g_sfx_emit_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kNoTime = 0x5ac82e8;        // the effects' "no time" float
constexpr u64 kStateOf = 0x2c82190;       // controller -> the state it is in
constexpr u64 kStateHolds = 0x2c821b0;    // (controller, state) -> whether that state holds
constexpr u64 kSetUp = 0x2c4f660;         // (node, the instance's definition, index table, record) -> 0 on failure
constexpr u64 kEndModules = 0x2c4c6c0;    // (container)
constexpr u64 kLinger = 0x2c4ec70;        // (node, the instance's definition, context) -> the next state
constexpr u64 kContainerAdd = 0x2c4c7f0;  // (container, count): the particles its next update bears
constexpr u64 kBind = 0x2c54280;          // (the instance's definition, node, id, binder data)
constexpr u64 kChildren = 0x2c6d780;      // (id, the instance's definition, node, data, definition + 0x18)
constexpr u64 kChildrenAt = 0x2c77050;    // the same, and the record's +0x70

using StateOfFn = GUEST_ABI u32 (*)(u64 controller);
using StateHoldsFn = GUEST_ABI u8 (*)(u64 controller, u32 state);
using SetUpFn = GUEST_ABI u8 (*)(u64 node, u64 instance_definition, u64 index, u64 record);
using ContainerOnlyFn = GUEST_ABI void (*)(u64 container);
using NextStateFn = GUEST_ABI u32 (*)(u64 node, u64 instance_definition, u64 context);
using CountFn = GUEST_ABI s32 (*)(u64 emitter, u64 node, float* time);
using AddFn = GUEST_ABI void (*)(u64 container, u32 count);
using BindFn = GUEST_ABI void (*)(u64 instance_definition, u64 node, u32 id, u64 data);
using ChildrenFn = GUEST_ABI void (*)(u32 id, u64 instance_definition, u64 node, u64 data, u64 definition_18);
using ChildrenAtFn = GUEST_ABI void (*)(u32 id, u64 instance_definition, u64 node, u64 data, u64 definition_18, u64 extra);

// The game's vaddss x, x, [y]: x + y with x as the first operand.
inline float add(float x, float y) {
    __asm__("addss %1, %0" : "+x"(x) : "x"(y));
    return x;
}

// vucomiss a, b then jb/sbb: the carry, set when a < b or either is a NaN.
inline bool below(float a, float b) {
    bool c;
    __asm__("ucomiss %2, %1" : "=@ccb"(c) : "x"(a), "x"(b));
    return c;
}

void* g_game_emit = nullptr;
std::atomic<int> g_clock{0};

}  // namespace

namespace sfx_node {

void set_emission_clock(int subs) { g_clock.store(subs, std::memory_order_relaxed); }
int emission_clock() { return g_clock.load(std::memory_order_relaxed); }

DECOMP_LEAF u32 emit(u64 node, u64 definition, u64 context, Emit mode, float t_event, bool children, float slack) {
    const bool event = mode == Emit::Event && !MUTANT(1);
    const bool edges = mode == Emit::Edges && !MUTANT(2);
    // A state an event leaves to the whole step: nothing done, nothing made.
    const u32 none = 0;
    auto linger = [&]() -> u32 { return game_function<NextStateFn>(kLinger)(node, definition, context); };

    const u32 st = game_function<StateOfFn>(kStateOf)(load<u64>(node + 0xc8));
    u8 cl = 1;  // the controller's state does not hold
    if (game_function<StateHoldsFn>(kStateHolds)(load<u64>(node + 0xc8), st)) {
        const u32 cur = load<u32>(node + 0xd8);
        if (cur == st) {
            cl = 0;
        } else {
            if (event && !MUTANT(3)) return none;  // a switch of record: the whole step's
            const u64 d = load<u64>(node + 0xd0);
            const u64 index = load<u64>(d + 8);
            const u32 from = load<u32>(index + 4 * static_cast<u64>(cur));
            const u32 to = load<u32>(index + 4 * static_cast<u64>(st));
            store<u32>(node + 0xd8, st);
            if (from == to) {
                cl = 0;
            } else if (MUTANT(4) ? static_cast<u32>(load<u64>(d)) <= to : load<u64>(d) <= static_cast<u64>(to)) {
                store<float>(node + 0x7c, load<float>(game_address(kNoTime)));
                store<u32>(node + 0xf0, 0);
                store<u64>(node + 0xe8, 0);
                store<u64>(node + 0xe0, 0);
                game_function<ContainerOnlyFn>(kEndModules)(node + 0x80);
                return 3;
            } else {
                if (!game_function<SetUpFn>(kSetUp)(node, definition, index, load<u64>(d + 0x10) + static_cast<u64>(to) * 0xa0)) return 9;
                cl = 0;
            }
        }
    }

    const u64 d = load<u64>(node + 0xd0);
    const u64 records = load<u64>(d + 0x10);
    const u64 rec = records + static_cast<u64>(load<u32>(load<u64>(d + 8) + 4 * static_cast<u64>(load<u32>(node + 0xd8)))) * 0xa0;
    const float t = event ? t_event : load<float>(node + 0x7c);
    const u32 f = load<u32>(rec + 4);
    const u64 inst = load<u64>(node + 0x58);

    if (f & 1) {
        // A record that waits for its end.
        if (event) return none;
        u8 al = cl ? 0 : static_cast<u8>((load<u8>(inst + 0x88) & 8) == 0);
        const u8 not_root = load<u64>(node + 0x60) != load<u64>(inst + 0x78);
        const u8 not_stopping = (load<u8>(node + 0x78) & 2) == 0;
        if (f & 0x100) {
            al = static_cast<u8>((al | not_root) & not_stopping);
            if (!al) return linger();
        }
        if (below(t, load<float>(node + 0xec))) return 1;
        return linger();
    }

    if (!(f & 2)) {
        // CPU particles (bit 2), or a record that emits nothing.
        if (!(f & 4)) return event ? none : 1;
        const u8 iflags = load<u8>(inst + 0x88);
        const u8 c = cl ? 0 : static_cast<u8>((iflags & 8) == 0);
        const u8 al = static_cast<u8>((load<u64>(node + 0x60) != load<u64>(inst + 0x78)) | c);
        const u8 on = static_cast<u8>(((load<u8>(node + 0x78) & 2) == 0) & al);
        u8 held = 1;  // the controller's state no longer holds (xor 1 of its answer)
        if (iflags & 2) {
            const u64 ctl = load<u64>(node + 0xc8);
            held = static_cast<u8>(game_function<StateHoldsFn>(kStateHolds)(ctl, game_function<StateOfFn>(kStateOf)(ctl)) ^ 1);
        }
        const u8 dl = static_cast<u8>(((f & 0x300) == 0) | on);
        const u8 before_end = below(t, load<float>(node + 0xec)) ? 0xff : 0;
        if (dl != 1 || !(before_end & static_cast<u8>(held ^ 1))) return event ? none : linger();
        if (event ? below(add(t, slack), load<float>(node + 0xe4)) : below(t, load<float>(node + 0xe4))) return event ? none : 1;
        const u32 made = load<u32>(node + 0xe0);
        if (MUTANT(5) ? static_cast<s32>(made) >= static_cast<s32>(load<u32>(node + 0xe8)) : made >= load<u32>(node + 0xe8)) {
            if (event) return none;  // the limit's edge: the whole step's
            if (f & 0x10) {
                store<u32>(node + 0xe4, 0x7f7fffffu);
                return 1;
            }
            return linger();
        }
        if (edges) return 1;  // the event is the clock's
        store<u32>(node + 0xe0, made + 1);
        if (const u64 emitter = load<u64>(rec + 0x28)) {
            float time = MUTANT(6) ? load<float>(node + 0x7c) : t;
            const s32 n = reinterpret_cast<CountFn>(static_cast<std::uintptr_t>(load<u64>(load<u64>(emitter) + 0x58)))(emitter, node, &time);
            if (n > 0) game_function<AddFn>(kContainerAdd)(node + 0x80, static_cast<u32>(n));
        }
        if (const u64 binder = load<u64>(rec + 0x98)) game_function<BindFn>(kBind)(definition, node, load<u32>(rec + 0x90), binder);
        store<float>(node + 0xe4, add(t, load<float>(node + 0xf0)));
        return 1;
    }

    // Children (bit 1).
    if (event && !children && !MUTANT(7)) return none;  // they wait for a whole step's frame
    u8 al = 1;
    u8 sil = 1;
    if (!cl) al = static_cast<u8>((load<u8>(inst + 0x88) & 8) >> 3);
    if (load<u64>(node + 0x60) == load<u64>(inst + 0x78)) {
        al ^= 1;
        sil = al;
    }
    const u32 nf = load<u32>(node + 0x78);
    al = 1;
    if (load<u8>(inst + 0x88) & 2) {
        const u64 ctl = load<u64>(node + 0xc8);
        al = static_cast<u8>(game_function<StateHoldsFn>(kStateHolds)(ctl, game_function<StateOfFn>(kStateOf)(ctl)) ^ 1);
    }
    if (!below(t, load<float>(node + 0xec))) return event ? none : 6;
    const u8 stopping = (nf & 2) != 0;
    const u8 ends = static_cast<u8>((sil == 0) | stopping);
    if ((f & 0x100) && ends) return event ? none : 6;
    al = static_cast<u8>(al | (((f & 0x200) != 0) & ends));
    if (al == 1) return event ? none : linger();
    if (event ? below(add(t, slack), load<float>(node + 0xe4)) : below(t, load<float>(node + 0xe4))) return event ? none : 1;
    const u32 made = load<u32>(node + 0xe0);
    if (made >= load<u32>(node + 0xe8)) {
        if (event) return none;  // the limit's edge: the whole step's
        store<u32>(node + 0xe0, made + 1);
        return linger();
    }
    if (edges && !MUTANT(8)) return 1;  // the event is the clock's
    store<u32>(node + 0xe0, made + 1);
    const u64 d18 = load<u64>(node + 0xd0) + 0x18;
    const u32 id = load<u32>(rec + 0x80);
    const u64 data = load<u64>(rec + 0x88);
    if (f & 0x20) game_function<ChildrenAtFn>(kChildrenAt)(id, definition, node, data, d18, load<u64>(rec + 0x70));
    else game_function<ChildrenFn>(kChildren)(id, definition, node, data, d18);
    if (const u64 binder = load<u64>(rec + 0x98)) game_function<BindFn>(kBind)(definition, node, load<u32>(rec + 0x90), binder);
    store<float>(node + 0xe4, add(t, load<float>(node + 0xf0)));
    return 1;
}

}  // namespace sfx_node

// sub_2c4e6b0 itself.
DECOMP_LEAF u32 sfx_node_emit(u64 node, u64 definition, u64 context) {
    return sfx_node::emit(node, definition, context, sfx_node::Emit::Game, 0.0f, true);
}

namespace {

// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, 0x48.
constexpr std::uint8_t kEmitEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                       0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x48};

}  // namespace

void decomp_sfx_node_emit_add() {
    decomp_add({"sub_2c4e6b0", "sfx", 0x2c4e6b0, kEmitEntry, sizeof(kEmitEntry), reinterpret_cast<void*>(&sfx_node_emit), DecompKind::Leaf,
                &g_game_emit});
}
