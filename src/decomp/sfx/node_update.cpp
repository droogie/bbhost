// The effects' node update, sub_2c4df80 at Binary Ninja 0x2c4df80 (docs/decomp.md,
// the effects area): the game's sequence as ours, every read, write and call in
// the game's order with the game's arguments, so the smooth steps can give its
// counted parts and its container calls different steps (engine/sfx_schedule.h).
//
// sub_2c4df80(node, &step, the instance's definition, render context) -> keep:
//   - a dead node (+0x78 bit 2) answers 0, to be freed; an inactive one (bit 0
//     clear) answers 1 and nothing else happens;
//   - a plain node (bit 3 clear) asks its controller (+0xc8) whether it has the
//     state it is in, and the root node of an instance asked to end (+0x88 bit
//     3) with no switch object (+0xd8, or that object's +0x11 set) dies; then
//     the definition pick (sub_2c81730 from +0xd0) - a new one resets the
//     definition's time +0xe0 to the "no time" value (0x5ac82e8) and is applied
//     (sub_2c81790) - its container update (sub_2c4c8c0, node+0x80) with a
//     copy of the step, and +0xe0 += step;
//   - a staged node (bit 3) runs its state +0xdc against its time +0x7c as it
//     was on entry: 0 waits for its record's start (+8), binds (sub_2c54280),
//     sets up (sub_2c4f660; failed: 9) and emits; 1 emits (sub_2c4e6b0); 2
//     resets its times and emission fields, ends its container's modules
//     (sub_2c4c6c0) and goes to 3; 3 waits for its controller's state, its
//     modules done when stopping (+0x78 bit 1, sub_2c4c6b0), switching records
//     or emitting; 4 lingers (sub_2c4ec70); 5 holds until +0xf0 or, for a
//     record with bit 1, while its children and the instance allow; 6 ends the
//     modules and, like 7, waits for the children (sub_2c4cdd0: 7, or 9 when
//     none live); above 7 only 9 matters - dead. A node asking for its render
//     items (+0x78 bit 4) has them made for its record (sub_2c4c240 for one,
//     sub_2c4c430 for three; a failure kills it) and bits 4-6 cleared;
//   - every active node then runs its container update once more with a copy
//     of the step - with no render context if it died here - and +0x7c += step.
// The adds are the game's addss with its operand order (the node's value
// first), so a NaN keeps the payload it has in the game.
#include "decomp/sfx/node_update.h"

#include "decomp/sfx/node_emit.h"

#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <atomic>
#include <cstdint>

using namespace decomp;

#ifdef DECOMP_SFX_NODE_UPDATE_MUTANTS
extern int g_sfx_node_mutant;
#define MUTANT(n) (g_sfx_node_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kNoTime = 0x5ac82e8;           // the effects' "no time" float
constexpr u64 kContainerUpdate = 0x2c4c8c0;  // (container, &step, definition, render context)
constexpr u64 kStateOf = 0x2c82190;          // controller -> the state it is in
constexpr u64 kStateHolds = 0x2c821b0;       // (controller, state) -> whether that state holds
constexpr u64 kPick = 0x2c81730;             // (definition, the instance's, node) -> a new definition or 0
constexpr u64 kApply = 0x2c81790;            // (definition, the instance's, node)
constexpr u64 kBind = 0x2c54280;             // (the instance's definition, node, id, binder data)
constexpr u64 kSetUp = 0x2c4f660;            // (node, the instance's definition, -, record) -> 0 on failure
constexpr u64 kEmit = 0x2c4e6b0;             // (node, the instance's definition, context) -> the next state
constexpr u64 kLinger = 0x2c4ec70;           // (node, the instance's definition, context) -> the next state
constexpr u64 kEndModules = 0x2c4c6c0;       // (container)
constexpr u64 kModulesDone = 0x2c4c6b0;      // (container) -> whether its modules are done
constexpr u64 kChildrenGone = 0x2c4cdd0;     // (container) -> 1 when no child lives
constexpr u64 kItemData = 0x2c69560;         // (an item's data) -> its definition
constexpr u64 kOneItem = 0x2c4c240;          // (container, context, definition, flag, allocator) -> made
constexpr u64 kThreeItems = 0x2c4c430;       // (container, context, a, b, c, kind, flag, flag, &time, allocator) -> made

using ContainerFn = GUEST_ABI void (*)(u64 container, float* step, u64 definition, u64 context);
using StateOfFn = GUEST_ABI u32 (*)(u64 controller);
using StateHoldsFn = GUEST_ABI u8 (*)(u64 controller, u32 state);
using PickFn = GUEST_ABI u64 (*)(u64 definition, u64 instance_definition, u64 node);
using ApplyFn = GUEST_ABI void (*)(u64 definition, u64 instance_definition, u64 node);
using BindFn = GUEST_ABI void (*)(u64 instance_definition, u64 node, u32 id, u64 data);
using SetUpFn = GUEST_ABI u8 (*)(u64 node, u64 instance_definition, u64 records, u64 record);
using NextStateFn = GUEST_ABI u32 (*)(u64 node, u64 instance_definition, u64 context);
using ContainerOnlyFn = GUEST_ABI void (*)(u64 container);
using ContainerAskFn = GUEST_ABI u8 (*)(u64 container);
using ItemDataFn = GUEST_ABI u64 (*)(u64 data);
using AllocatorFn = GUEST_ABI u64 (*)(u64 object, u64 context, u64 index, u64 vtable);
using OneItemFn = GUEST_ABI u8 (*)(u64 container, u64 context, u64 definition, u32 flag, u64 allocator);
using ThreeItemsFn = GUEST_ABI u8 (*)(u64 container, u64 context, u64 a, u64 b, u64 c, u32 kind, u32 flag20, u32 flag40, float* time,
                                      u64 allocator);

// The game's vaddss x, x, [y]: x + y with x as the first operand.
inline float add(float x, float y) {
    __asm__("addss %1, %0" : "+x"(x) : "x"(y));
    return x;
}

inline void kill(u64 node) { store<u32>(node + 0x78, (load<u32>(node + 0x78) & 0xfffffffau) | 4u); }

// The record a staged node is on: its definition's (+0xd0) index table (+8)
// at +0xd8, into the 0xa0-byte records (+0x10).
inline u64 record_offset(u64 node) {
    const u64 d = load<u64>(node + 0xd0);
    return static_cast<u64>(load<u32>(load<u64>(d + 8) + 4 * static_cast<u64>(load<u32>(node + 0xd8)))) * 0xa0;
}

void* g_game_node = nullptr;

// The smooth steps' container call in progress (sfx_node::smooth_call): its node, the node's time of the whole step and
// whether the frame has one. Written before the call (the node last) and cleared after it.
std::atomic<std::uint64_t> g_call_node{0};
std::atomic<std::uint32_t> g_call_time{0}, g_call_tick{0};

// A container call of the smooth steps' frame domain: the frame's step, and the
// node's time (+0x7c) - and a plain node's definition time (+0xe0) - moved on
// by an offset for the call (the part of a whole step the frame is past) and
// put back after it, unless the call changed them; an offset of zero (either
// sign) or a time that is the "no time" value is left alone, so a call with
// no offsets is the game's. A smooth call is published for its modules (the
// node and its time before the offset).
inline void call_container(u64 node, float step, u64 definition, u64 context, float node_offset, float definition_offset,
                           const sfx_node::Steps& st) {
    const u32 none = load<u32>(game_address(kNoTime));
    const u32 t = load<u32>(node + 0x7c), e = load<u32>(node + 0xe0);
    if (st.smooth && !MUTANT(18)) {
        g_call_time.store(t, std::memory_order_relaxed);
        g_call_tick.store(st.tick ? 1u : 0u, std::memory_order_relaxed);
        g_call_node.store(node, std::memory_order_release);
    }
    const bool shift_t = node_offset != 0.0f && (t != none || MUTANT(16));
    const bool shift_e = definition_offset != 0.0f && (e != none || MUTANT(16));
    u32 t_shifted = t, e_shifted = e;
    if (shift_t) {
        store<float>(node + 0x7c, add(load<float>(node + 0x7c), node_offset));
        t_shifted = load<u32>(node + 0x7c);
    }
    if (shift_e) {
        store<float>(node + 0xe0, add(load<float>(node + 0xe0), definition_offset));
        e_shifted = load<u32>(node + 0xe0);
    }
    float s = step;
    game_function<ContainerFn>(kContainerUpdate)(node + 0x80, &s, definition, context);
    if (st.smooth && !MUTANT(19)) g_call_node.store(0, std::memory_order_release);
    if (shift_t && load<u32>(node + 0x7c) == t_shifted && !MUTANT(12)) store<u32>(node + 0x7c, t);
    if (shift_e && load<u32>(node + 0xe0) == e_shifted) store<u32>(node + 0xe0, e);
}

// A staged node's emitting state on a whole step: the game's (sub_2c4e6b0), or under the emission clock its edges only,
// the clock making the events (node_emit.h).
inline u32 emit_edges(u64 node, u64 definition, u64 context, const sfx_node::Steps& st) {
    if (st.sub > 0.0f && !MUTANT(20)) return sfx_node::emit(node, definition, context, sfx_node::Emit::Edges, 0.0f, true);
    return game_function<NextStateFn>(kEmit)(node, definition, context);
}

// The emission clock (node_emit.h): an emitting staged node (state 1) makes its emission events at the 1/60 s
// sub-steps of its time - those of its whole step (from its time +0x7c) and, on a whole step's frame, of the next one
// (from the time the step leaves it) - each at its own time and before the container call that bears its particles:
//   - CPU particles: the sub-steps in this frame's span of the node's time, from the carry before the frame to the
//     carry after it (past the whole step's end on a whole step's frame), so each sub-step is made on the one frame
//     whose span holds it; a node that began emitting on this whole step (its state was not 1) catches up from the
//     step's own time;
//   - children: on a whole step's frame only (the node trees change there), the whole step's own sub-steps.
inline void clock_events(u64 node, u64 definition, u64 context, const sfx_node::Steps& st, bool entered) {
    if (!(st.sub > 0.0f) || load<u32>(node + 0xdc) != 1 || (load<u32>(node + 0x78) & 4)) return;
    const int subs = sfx_node::emission_clock() > 1 ? 2 : 1;
    const u64 d = load<u64>(node + 0xd0);
    const u32 f =
        load<u32>(load<u64>(d + 0x10) + static_cast<u64>(load<u32>(load<u64>(d + 8) + 4 * static_cast<u64>(load<u32>(node + 0xd8)))) * 0xa0 + 4);
    if (f & 1) return;  // a record that waits for its end makes no event
    const float base = load<float>(node + 0x7c);
    // A sub-step meets the next within a few roundings of the node's time: the sub-steps lie on the whole steps (+0x7c
    // moved by the whole step), the next on the last event's time and the interval, and an add rounding the other way
    // must not skip a sub-step (it skipped one in two of the 1/60 s emitters' events).
    const float mag = base > 1.0f ? base : base < -1.0f ? -base : 1.0f;
    const float slack = MUTANT(24) ? 0.0f : mag * (1.0f / 1048576.0f);
    if (f & 2) {
        if (!st.tick && !MUTANT(22)) return;
        float s = base;
        for (int k = 0; k < subs; ++k, s = add(s, st.sub)) sfx_node::emit(node, definition, context, sfx_node::Emit::Event, s, true, slack);
        return;
    }
    if (!(f & 4)) return;
    const float start = add(base, MUTANT(21) ? 0.0f : st.from);
    const float next_base = st.tick ? add(base, st.counted) : base;  // the next whole step's time, as the step leaves it
    const float end = add(next_base, st.into);
    for (int a = 0; a < (st.tick ? 2 : 1); ++a) {
        float s = a == 0 ? base : next_base;
        for (int k = 0; k < subs; ++k, s = add(s, st.sub)) {
            const bool in_span = !(s < start) && s < end;
            const bool catch_up = entered && a == 0 && s < start && !MUTANT(23);
            if (in_span || catch_up) sfx_node::emit(node, definition, context, sfx_node::Emit::Event, s, st.tick, slack);
        }
    }
}

}  // namespace

namespace sfx_node {

SmoothCall smooth_call() {
    SmoothCall c;
    const u64 node = g_call_node.load(std::memory_order_acquire);
    if (!node) return c;
    c.time = g_call_time.load(std::memory_order_relaxed);
    c.tick = g_call_tick.load(std::memory_order_relaxed) != 0;
    if (g_call_node.load(std::memory_order_acquire) == node) c.node = node;
    return c;
}

// The node update with its two kinds of step apart (decomp/sfx/node_update.h).
DECOMP_LEAF u8 update(u64 node, const Steps& st, u64 definition, u64 context) {
    const bool counts = st.tick || MUTANT(11);  // the counted parts run
    const float container_step = MUTANT(13) ? st.counted : st.frame;
    const u32 flags = load<u32>(node + 0x78);
    const u32 state_in = load<u32>(node + 0xdc);  // a staged node's state on entry: whether its emitting began this step
    if (flags & 4) return st.tick || MUTANT(14) ? 0 : 1;  // a dead node: freed by a whole step, kept between them
    u8 keep = 1;
    if (!(flags & 1)) return keep;
    const bool plain = !(flags & 8);
    if (plain) {
        // A plain node: its definition's switch and pick, its first container call.
        if (counts) {
            const u64 ctl = load<u64>(node + 0xc8);
            const u8 holds = game_function<StateHoldsFn>(kStateHolds)(ctl, game_function<StateOfFn>(kStateOf)(ctl));
            const u64 inst = load<u64>(node + 0x58);
            u8 ending = 1;
            if (holds) ending = static_cast<u8>((load<u8>(inst + 0x88) & 8) >> 3);
            if (load<u64>(node + 0x60) == load<u64>(inst + 0x78) && ending) {
                const u64 sw = load<u64>(node + 0xd8);
                if (!sw || load<u8>(sw + 0x11) || MUTANT(10)) {
                    kill(node);
                    keep = 0;
                }
            }
            const u64 current = (load<u8>(node + 0x78) & 8) ? 0 : load<u64>(node + 0xd0);
            if (const u64 picked = game_function<PickFn>(kPick)(current, definition, node)) {
                store<float>(node + 0xe0, load<float>(game_address(kNoTime)));
                store<u64>(node + 0xd0, picked);
                game_function<ApplyFn>(kApply)(picked, definition, node);
            }
        }
        call_container(node, container_step, definition, context, st.from, st.from, st);
        if (counts) store<float>(node + 0xe0, MUTANT(1) ? load<float>(node + 0xe0) : add(load<float>(node + 0xe0), st.counted));
    } else if (counts) {

        // A staged node on a whole step: its state machine against its time on entry.
        const float t = load<float>(node + 0x7c);
        const u32 state = load<u32>(node + 0xdc);
        u32 next = state;
        bool counted = true;  // the state stored and 9 checked (2's own path skips both)
        bool stored = false;  // stored on its own path (5's)
        switch (state) {
        case 0: {
            const u64 d = load<u64>(node + 0xd0);
            const u64 records = load<u64>(d + 0x10);
            const u64 rec = records + record_offset(node);
            next = 0;
            if (load<float>(rec + 8) > t) break;
            if (const u64 binder = load<u64>(rec + 0x38)) game_function<BindFn>(kBind)(definition, node, load<u32>(rec + 0x30), binder);
            next = 9;
            if (!game_function<SetUpFn>(kSetUp)(node, definition, records, rec)) break;
            next = emit_edges(node, definition, context, st);
            break;
        }
        case 1:
            next = emit_edges(node, definition, context, st);
            break;
        case 2:
            store<float>(node + 0x7c, load<float>(game_address(kNoTime)));
            store<u32>(node + 0xf0, 0);
            store<u64>(node + 0xe8, 0);
            store<u64>(node + 0xe0, 0);
            game_function<ContainerOnlyFn>(kEndModules)(node + 0x80);
            store<u32>(node + 0xdc, MUTANT(2) ? 2u : 3u);
            counted = false;
            break;
        case 3: {
            if (flags & 2) {
                const u8 done = game_function<ContainerAskFn>(kModulesDone)(node + 0x80);
                next = 6;
                if (!done) break;
            }
            const u32 want = game_function<StateOfFn>(kStateOf)(load<u64>(node + 0xc8));
            const u8 holds = game_function<StateHoldsFn>(kStateHolds)(load<u64>(node + 0xc8), want);
            next = 6;
            if (!holds) break;
            const u32 at = load<u32>(node + 0xd8);
            next = 3;
            if (at == want) break;
            const u64 d = load<u64>(node + 0xd0);
            const u64 index = load<u64>(d + 8);
            const u32 from = load<u32>(index + 4 * static_cast<u64>(at)), to = load<u32>(index + 4 * static_cast<u64>(want));
            const bool past = MUTANT(3) ? static_cast<u32>(load<u64>(d)) <= to : load<u64>(d) <= static_cast<u64>(to);
            if (from == to || past) {
                store<u32>(node + 0xd8, want);
                break;
            }
            next = emit_edges(node, definition, context, st);
            break;
        }
        case 4:
            next = game_function<NextStateFn>(kLinger)(node, definition, context);
            break;
        case 5: {
            const u64 d = load<u64>(node + 0xd0);
            const u32 rf = load<u32>(load<u64>(d + 0x10) + record_offset(node) + 4);
            if (!(rf & 2)) {
                next = t >= load<float>(node + 0xf0) ? 6u : 5u;
            } else {
                const u64 ctl = load<u64>(node + 0xc8);
                u8 allow = 0;
                if (game_function<StateHoldsFn>(kStateHolds)(ctl, game_function<StateOfFn>(kStateOf)(ctl))) {
                    const u64 inst = load<u64>(node + 0x58);
                    allow = 1;
                    if (load<u64>(node + 0x60) == load<u64>(inst + 0x78)) allow = !(load<u8>(inst + 0x88) & 8);
                }
                next = 6;
                if (load<u64>(node + 0x70)) {
                    const u8 not_stopping = !(load<u8>(node + 0x78) & 2);
                    if (!(rf & 0x100) || (allow & not_stopping)) next = 5;
                }
            }
            store<u32>(node + 0xdc, next);
            stored = true;
            counted = false;
            break;
        }
        case 6:
            game_function<ContainerOnlyFn>(kEndModules)(node + 0x80);
            [[fallthrough]];
        case 7:
            if (MUTANT(9)) next = game_function<ContainerAskFn>(kChildrenGone)(node + 0x80) ? 9u : 7u;
            else next = static_cast<u32>(game_function<ContainerAskFn>(kChildrenGone)(node + 0x80)) * 2 + 7;
            break;
        default:
            if (MUTANT(8)) next = 9;  // every state above 7 taken for 9
            break;                    // above 7: not stored, only 9 counts
        }
        (void)stored;
        if (counted) {
            if (state <= 7) store<u32>(node + 0xdc, next);
            if (next == 9) {
                kill(node);
                keep = 0;
                call_container(node, container_step, definition, 0, st.from, 0.0f, st);
                store<float>(node + 0x7c, add(load<float>(node + 0x7c), st.counted));
                return keep;
            }
        }
        // Its render items, once asked for (+0x78 bit 4).
        const u32 f = load<u32>(node + 0x78);
        keep = 1;
        if (f & 0x10) {
            const u64 d = load<u64>(node + 0xd0);
            const u64 index = load<u64>(d + 8);
            const u32 i = load<u32>(index + 4 * static_cast<u64>(load<u32>(node + 0xd8)));
            if (context) {
                const u64 records = load<u64>(d + 0x10);
                const u64 off = static_cast<u64>(i) * 0xa0;
                if (load<u64>(records + off + 0x58)) {
                    const u64 object = load<u64>(definition + 8);
                    const u64 vtable = load<u64>(object);
                    const u64 allocator =
                        reinterpret_cast<AllocatorFn>(static_cast<std::uintptr_t>(load<u64>(vtable + 0x70)))(object, context, index, vtable);
                    const u32 rf = load<u32>(records + off + 4);
                    const u32 high = f >> 6;
                    if (rf & 1) {
                        const u64 a = game_function<ItemDataFn>(kItemData)(load<u64>(records + off + 0x58));
                        if (!a) {
                            game_function<BindFn>(kBind)(definition, node, load<u32>(records + off + 0x50), load<u64>(records + off + 0x58));
                        } else if (!game_function<OneItemFn>(kOneItem)(node + 0x80, context, a, high & 1, allocator)) {
                            kill(node);
                            keep = 0;
                        }
                    } else if (rf & 4) {
                        const u64 a = game_function<ItemDataFn>(kItemData)(load<u64>(records + off + 0x58));
                        const u64 b = game_function<ItemDataFn>(kItemData)(load<u64>(records + off + 0x88));
                        const u64 c = game_function<ItemDataFn>(kItemData)(load<u64>(records + off + 0x78));
                        if (a && b && c) {
                            float time = MUTANT(7) ? load<float>(node + 0x7c) : t;
                            if (!game_function<ThreeItemsFn>(kThreeItems)(node + 0x80, context, a, b, c, load<u32>(records + off),
                                                                          (f & 0x20) ? 1u : 0u, high & 1, &time, allocator)) {
                                kill(node);
                                keep = 0;
                            }
                        } else {
                            kill(node);
                            keep = 0;
                        }
                    }
                }
            }
            store<u8>(node + 0x78, static_cast<u8>(load<u8>(node + 0x78) & (MUTANT(4) ? 0x9fu : 0x8fu)));
        }
    }
    // An emitting staged node's events of this frame, on the emission clock.
    if (!plain && keep) clock_events(node, definition, context, st, counts && state_in != 1);
    // Every active node: its container once more, and its time.
    call_container(node, container_step, definition, keep && !MUTANT(5) ? context : 0, st.from,
                   plain ? (MUTANT(15) ? st.from : st.into) : 0.0f, st);
    if (st.tick || MUTANT(17))
        store<float>(node + 0x7c, MUTANT(6) ? add(st.counted, load<float>(node + 0x7c)) : add(load<float>(node + 0x7c), st.counted));
    return keep;
}

}  // namespace sfx_node

// sub_2c4df80 itself: the game's sequence, one step for both kinds and no offsets.
DECOMP_LEAF u8 sfx_node_update(u64 node, float* step, u64 definition, u64 context) {
    const float s = *step;
    sfx_node::Steps st{s, s, 0.0f, 0.0f, true};
    // Under the emission clock the update's step holds sfx_node::emission_clock() sub-steps of 1/60 s (two in the
    // game's 1/30 s, one in a whole 1/60 s step); a step of 0 (the game's own still frame) keeps the game's event.
    const int subs = sfx_node::emission_clock();
    if (subs > 0 && s > 0.0f) st.sub = subs == 2 ? s * 0.5f : s;
    return sfx_node::update(node, st, definition, context);
}

namespace {

// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, 0x88.
constexpr std::uint8_t kNodeEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                       0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x88, 0x00, 0x00, 0x00};

}  // namespace

void decomp_sfx_node_update_add() {
    decomp_add({"sub_2c4df80", "sfx", 0x2c4df80, kNodeEntry, sizeof(kNodeEntry), reinterpret_cast<void*>(&sfx_node_update), DecompKind::Leaf,
                &g_game_node});
}
