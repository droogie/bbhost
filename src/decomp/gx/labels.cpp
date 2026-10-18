// Gnm's label, wait, inline-data, predication, indirect-buffer and counter
// writers as our source (docs/decomp.md). Twelve small leaf methods of Gnm's
// command buffers - every writer of a packet the command processor acts on,
// besides our tokens, that the game calls - through which every completion
// the game asks the GPU for is written, every wait it makes the GPU do on one
// - GX's block pools and command arena, Map and its queries, YEBIS's drains
// and compute utilities, Scaleform's fences, the compute ring's retirement,
// the flip - the compute ring's calls of the command buffers it runs, and the
// draw engine's side of the constant engine's counters:
//
// - EVENT_WRITE_EOP (0x14778f0): a label written when the work before it has
//   left the pipe - 4.4M a 12-place tour;
// - EVENT_WRITE_EOS (0x1478320): one written when the shaders before it are
//   done - 93k;
// - RELEASE_MEM (0x148b0f0): the compute command buffer's label - the compute
//   ring's retirement, 79;
// - WAIT_REG_MEM, the draw command buffer's (0x1477b30) and the compute
//   one's (0x148b430), the same code: the GPU waits until memory or a
//   register compares as asked - 343k;
// - WRITE_DATA (0x1473b80 to memory, 0x1473ce0 through L2 with a cache
//   policy): dwords from the caller written inline - none in a tour;
// - SET_PREDICATION on an occlusion result, and cleared (0x1477570,
//   0x1477600), with the command buffer's predication flag - none in a tour;
// - INDIRECT_BUFFER on the compute command buffer (0x148b330,
//   GXComputeQueue::Submit's) - 79;
// - the draw engine's side of the constant engine's counters,
//   INCREMENT_DE_COUNTER (0x1478220) and WAIT_ON_CE_COUNTER (0x14781d0) -
//   YEBIS's draws through its constant engine, 75 in a six-place session.
//
// Each is the game's function as source, instruction for instruction: the
// same room check (the low 32 bits of the dwords left, against the packet's
// size), the same refill call when there is no room (cb, dwords, its
// argument, the room left as a 64-bit count), the same packet, the same
// write-pointer moves and the same return value (the packet's address, or for
// WRITE_DATA with a payload the write pointer after it; the refill's result
// when it gives nothing). They write the game's packets always: the command
// processor walks them. Each also notes its packet in its chunk's host list
// (host/host_lists.h; nothing unless BBHOST_HOST_LISTS is on).
// tests/decomp/gx_labels_test.cpp checks them against the game's code on
// generated inputs.
//
// The command buffer they write through: +0x8 the chunk's end, +0x10 the write
// pointer, +0x18 the refill, +0x20 its argument, +0x28 its flags (bit 0
// predication on, bit 1 the predicate the end-of-shader header carries).
//
// Leaves: entered straight from the game's code, they touch only the game's
// memory. With the writers' census (BBHOST_CP_WRITERS=1), which hooks the
// game's prologues of three of them, the game's versions stay;
// BBHOST_DECOMP_COMPARE=1 runs the game's version of each in their place and
// checks ours against every call (hosted).
#include "core/thunk.h"
#include "decomp/decomp.h"
#include "decomp/guest.h"
#include "host/host_lists.h"
#include "log.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

using namespace decomp;

namespace {

enum Writer : std::size_t {
    kEndOfPipe,
    kEndOfShader,
    kReleaseMem,
    kWait,
    kComputeWait,
    kWriteData,
    kWriteDataL2,
    kSetPredication,
    kClearPredication,
    kIndirectBuffer,
    kIncrementDeCounter,
    kWaitOnCeCounter,
    kWriters
};

struct LabelWriter {
    const char* name;  // on the decomp list (BBHOST_DECOMP_OFF), the symbol file's where it has one
    u64 bn;            // the entry, as Binary Ninja addresses it
    u8 n;              // the prologue below: whole instructions, none a branch target
    u8 bytes[17];
    u8 dwords;         // the packet's size (WRITE_DATA: 4, and its payload)
};

// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx, rax (14 bytes).
#define PUSH6 0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50
const LabelWriter kTable[kWriters] = {
    {"gnm_write_end_of_pipe_event", 0x14778f0, 14, {PUSH6}, 6},
    // push rbp; mov rbp, rsp; push r15, r14, r12, rbx; mov r14d, ecx.
    {"gnm_write_end_of_shader_event", 0x1478320, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xce}, 5},
    {"gnm_dispatch_write_release_mem", 0x148b0f0, 14, {PUSH6}, 7},
    {"gnm_wait_reg_mem", 0x1477b30, 14, {PUSH6}, 7},
    {"gnm_dispatch_wait_reg_mem", 0x148b430, 14, {PUSH6}, 7},
    {"gnm_write_data_inline", 0x1473b80, 14, {PUSH6}, 4},
    {"gnm_write_data_inline_l2", 0x1473ce0, 14, {PUSH6}, 4},
    // push rbp; mov rbp, rsp; push r15, r14, r12, rbx; mov r15d, ecx.
    {"gnm_set_predication", 0x1477570, 14, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x54, 0x53, 0x41, 0x89, 0xcf}, 3},
    // push rbp; mov rbp, rsp; push rbx, rax; mov rbx, rdi; mov rcx, [rbx+8]; mov rax, [rbx+0x10].
    {"gnm_clear_predication", 0x1477600, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 3},
    // push rbp; mov rbp, rsp; push r15, r14, rbx, rax; mov r14, rdx; mov r15, rsi.
    {"gnm_dispatch_write_indirect_buffer", 0x148b330, 16,
     {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd6, 0x49, 0x89, 0xf7}, 4},
    // push rbp; mov rbp, rsp; push rbx, rax; mov rbx, rdi; mov rcx, [rbx+8]; mov rax, [rbx+0x10].
    {"gnm_increment_de_counter", 0x1478220, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 2},
    {"gnm_wait_on_ce_counter", 0x14781d0, 17,
     {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x4b, 0x08, 0x48, 0x8b, 0x43, 0x10}, 2},
};
#undef PUSH6

// The refill every writer calls when there is no room: (cb, dwords, its
// argument, the room left in dwords); its result is a bool in the low byte,
// and a writer that gets false returns the whole register as it came back.
using Refill = GUEST_ABI u64 (*)(u64 cb, u32 dwords, u64 arg, u64 room);

// The room for `need` dwords: true with the write pointer to write at; false
// with the refill's result when it gave nothing. The game's check is the low
// 32 bits of the dwords left against the packet's size (`cmp ecx, n - 1; ja`,
// or `cmp ecx, n; jae` for WRITE_DATA's, the same for n > 0), and after a
// refill it reads the write pointer again without a second check.
__attribute__((always_inline)) inline bool make_room(u64 cb, u32 need, u64* wp, u64* failed) {
    u64 p = load<u64>(cb + 0x10);
    const u64 room = (load<u64>(cb + 0x8) - p) >> 2;
    if (static_cast<u32>(room) < need) {
        const auto refill = reinterpret_cast<Refill>(static_cast<std::uintptr_t>(load<u64>(cb + 0x18)));
        const u64 r = refill(cb, need, load<u64>(cb + 0x20), room);
        if (static_cast<u8>(r) == 0) {
            *failed = r;
            return false;
        }
        p = load<u64>(cb + 0x10);
    }
    *wp = p;
    return true;
}

__attribute__((always_inline)) inline void put(u64 at, u32 v) { store<u32>(at, v); }

// The game's functions, written out. Each returns what the game's returns.

// EVENT_WRITE_EOP: the event, the cache actions (bits 12-17) and the cache
// policy (25-26); `flags` bit 4 into bit 27 and bit 0 into the destination's
// select (bit 16 of the address's high dword); the destination dword-pair
// aligned, its high 16 bits; the data select and the value.
inline u64 w_end_of_pipe(u64 cb, u32 event, u32 flags, u64 addr, u32 data_sel, u64 value, u32 cache_action, u32 cache_policy) {
    u64 wp, failed;
    if (!make_room(cb, 6, &wp, &failed)) return failed;
    put(wp, 0xc0044700u);
    put(wp + 4, ((flags & 0x10) << 23) | (event & 0x3f) | ((cache_action & 0x3f) << 12) | ((cache_policy & 3) << 25) | 0x500u);
    put(wp + 8, static_cast<u32>(addr) & 0xfffffff8u);
    put(wp + 12, (data_sel << 29) | ((flags & 1) << 16) | static_cast<u32>((addr >> 32) & 0xffff));
    put(wp + 16, static_cast<u32>(value));
    put(wp + 20, static_cast<u32>(value >> 32));
    store<u64>(cb + 0x10, wp + 0x18);
    host_lists::note(cb, wp, 0x47);
    return wp;
}

// EVENT_WRITE_EOS: the event, the destination (dword aligned, its high 8
// bits) and a 32-bit value; the header takes the command buffer's predicate
// bit (+0x28 bit 1). The game's writes the packet, reads the write pointer
// again and writes the packet there once more, then moves it.
inline u64 w_end_of_shader(u64 cb, u32 event, u64 addr, u32 value) {
    u64 wp, failed;
    if (!make_room(cb, 5, &wp, &failed)) return failed;
    const u32 header = (load<u32>(cb + 0x28) & 2) | 0xc0034800u;
    const u32 cntl = (event & 0x3f) | 0x600u;
    const u32 lo = static_cast<u32>(addr) & 0xfffffffcu;
    const u32 hi = static_cast<u32>((addr >> 32) & 0xff) | 0x40000000u;
    put(wp, header);
    put(wp + 4, cntl);
    put(wp + 8, lo);
    put(wp + 12, hi);
    put(wp + 16, value);
    const u64 at = load<u64>(cb + 0x10);
    put(at, header);
    put(at + 4, cntl);
    put(at + 8, lo);
    put(at + 12, hi);
    put(at + 16, value);
    store<u64>(cb + 0x10, load<u64>(cb + 0x10) + 0x14);
    host_lists::note(cb, at, 0x48);
    return at;
}

// RELEASE_MEM on the compute command buffer: as the end-of-pipe label, with
// the event's index 6 for 0x2f (an end-of-shader event) and 5 otherwise, an
// interrupt select of 3, and the whole 64-bit destination.
inline u64 w_release_mem(u64 cb, u32 event, u32 flags, u64 addr, u32 data_sel, u64 value, u32 cache_action, u32 cache_policy) {
    u64 wp, failed;
    if (!make_room(cb, 7, &wp, &failed)) return failed;
    const u32 index = event == 0x2f ? 0x600u : 0x500u;
    put(wp, 0xc0054902u);
    put(wp + 4, ((flags & 0x10) << 23) | (event & 0x3f) | index | ((cache_action & 0x3f) << 12) | ((cache_policy & 3) << 25));
    put(wp + 8, (data_sel << 29) | ((flags & 1) << 16) | 0x3000000u);
    put(wp + 12, static_cast<u32>(addr));
    put(wp + 16, static_cast<u32>(addr >> 32));
    put(wp + 20, static_cast<u32>(value));
    put(wp + 24, static_cast<u32>(value >> 32));
    store<u64>(cb + 0x10, wp + 0x1c);
    host_lists::note(cb, wp, 0x49);
    return wp;
}

// WAIT_REG_MEM: memory past 0xffff (a dword address), a register at or below
// it; the compare function, the reference and the mask; a poll interval of 10.
inline u64 w_wait(u64 cb, u64 addr, u32 mask, u32 function, u32 reference) {
    u64 wp, failed;
    if (!make_room(cb, 7, &wp, &failed)) return failed;
    const bool memory = addr > 0xffff;
    put(wp, 0xc0053c00u);
    put(wp + 4, (static_cast<u32>(memory) << 4) | (function & 7));
    put(wp + 8, static_cast<u32>(addr) & (memory ? 0xfffffffcu : 0xffffu));
    put(wp + 12, static_cast<u32>((addr >> 32) & 0xffff));
    put(wp + 16, reference);
    put(wp + 20, mask);
    put(wp + 24, 0xa);
    store<u64>(cb + 0x10, wp + 0x1c);
    host_lists::note(cb, wp, 0x3c);
    return wp;
}

// WRITE_DATA's payload: `dwords` dwords from `src` to `dst`, as the game's
// copy goes - when the two do not overlap by its test, sixteen dwords at a
// time (each block read whole before it is written), then one at a time;
// when they do, one at a time from the first. The same bytes for any
// overlap and alignment.
inline void copy_payload(u64 dst, u64 src, u32 dwords) {
    const u64 n = dwords;
    const u64 blocks = n & 0x1fffffff0ull;
    u64 done = 0;
    if (blocks && (dst > src + 4 * (n - 1) || dst + 4 * (n - 1) < src)) {
        for (; done < blocks; done += 16) {
            u32 block[16];
            for (u64 k = 0; k < 16; ++k) block[k] = load<u32>(src + 4 * (done + k));
            for (u64 k = 0; k < 16; ++k) put(dst + 4 * (done + k), block[k]);
        }
    }
    for (u64 k = done; k < n; ++k) put(dst + 4 * k, load<u32>(src + 4 * k));
}

// WRITE_DATA: the header's count from the payload's size in 32-bit
// arithmetic, the control dword, the destination (dword aligned) - then the
// write pointer read again and moved past those four dwords, the payload
// after them, and the write pointer after the payload returned (the packet's
// address when there is none).
inline u64 write_data(u64 cb, u64 dest, u64 src, u32 dwords, u32 control) {
    u64 wp, failed;
    if (!make_room(cb, dwords + 4, &wp, &failed)) return failed;
    put(wp, ((dwords << 16) + 0x20000u) | 0xc0003700u);
    put(wp + 4, control);
    put(wp + 8, static_cast<u32>(dest) & 0xfffffffcu);
    put(wp + 12, static_cast<u32>(dest >> 32));
    const u64 p = load<u64>(cb + 0x10);
    store<u64>(cb + 0x10, p + 0x10);
    host_lists::note(cb, wp, 0x37);
    if (dwords == 0) return p;
    copy_payload(p + 0x10, src, dwords);
    const u64 end = p + 4 * static_cast<u64>(dwords - 1) + 0x14;
    store<u64>(cb + 0x10, end);
    return end;
}

// To memory (the control's destination select 5), a cache policy of 2, the
// write confirmed if asked.
inline u64 w_write_data(u64 cb, u64 dest, u64 src, u32 dwords, u32 confirm) {
    return write_data(cb, dest, src, dwords, ((confirm & 1) << 20) | 0x4000500u);
}

// Through L2 (the select 2) with the caller's cache policy.
inline u64 w_write_data_l2(u64 cb, u64 dest, u64 src, u32 dwords, u32 cache_policy, u32 confirm) {
    return write_data(cb, dest, src, dwords, ((cache_policy & 3) << 25) | ((confirm & 1) << 20) | 0x200u);
}

// SET_PREDICATION on the occlusion result at `block` (16-byte aligned, its
// high 8 bits), drawing if it was visible or not (bit 12), the hint (bit 8);
// the command buffer's predication flag set first.
inline u64 w_set_predication(u64 cb, u64 block, u32 draw_if_visible, u32 hint) {
    u64 wp, failed;
    if (!make_room(cb, 3, &wp, &failed)) return failed;
    store<u8>(cb + 0x28, static_cast<u8>(load<u8>(cb + 0x28) | 1));
    put(wp, 0xc0012000u);
    put(wp + 4, static_cast<u32>(block) & 0xfffffff0u);
    put(wp + 8, ((hint & 1) << 8) | ((draw_if_visible & 1) << 12) | static_cast<u32>((block >> 32) & 0xff) | 0x10000u);
    store<u64>(cb + 0x10, wp + 12);
    host_lists::note(cb, wp, 0x20);
    return wp;
}

// INDIRECT_BUFFER on the compute command buffer: the callee's address (its
// high 16 bits) and size in dwords (20 bits), valid, as the compute ring's
// submissions call the command buffers they run (GXComputeQueue::Submit).
inline u64 w_indirect_buffer(u64 cb, u64 addr, u32 dwords) {
    u64 wp, failed;
    if (!make_room(cb, 4, &wp, &failed)) return failed;
    put(wp, 0xc0023f02u);
    put(wp + 4, static_cast<u32>(addr));
    put(wp + 8, static_cast<u32>((addr >> 32) & 0xffff));
    put(wp + 12, (dwords & 0xfffff) | 0x1800000u);
    store<u64>(cb + 0x10, wp + 0x10);
    host_lists::note(cb, wp, 0x3f);
    return wp;
}

// The draw engine's side of the constant engine's counters: INCREMENT_DE_COUNTER,
// and WAIT_ON_CE_COUNTER (its one dword 1), a qword each.
inline u64 w_counter(u64 cb, u64 packet, u32 op) {
    u64 wp, failed;
    if (!make_room(cb, 2, &wp, &failed)) return failed;
    store<u64>(wp, packet);
    store<u64>(cb + 0x10, wp + 8);
    host_lists::note(cb, wp, op);
    return wp;
}

// SET_PREDICATION cleared, and the flag.
inline u64 w_clear_predication(u64 cb) {
    u64 wp, failed;
    if (!make_room(cb, 3, &wp, &failed)) return failed;
    store<u8>(cb + 0x28, static_cast<u8>(load<u8>(cb + 0x28) & 0xfe));
    store<u64>(wp, 0x00000000c0012000ull);
    put(wp + 8, 0);
    store<u64>(cb + 0x10, wp + 12);
    host_lists::note(cb, wp, 0x20);
    return wp;
}

}  // namespace

// The leaves on the decomp list.
DECOMP_LEAF u64 gnm_write_end_of_pipe_event(u64 cb, u32 event, u32 flags, u64 addr, u32 data_sel, u64 value, u32 cache_action,
                                            u32 cache_policy) {
    return w_end_of_pipe(cb, event, flags, addr, data_sel, value, cache_action, cache_policy);
}
DECOMP_LEAF u64 gnm_write_end_of_shader_event(u64 cb, u32 event, u64 addr, u32 value) { return w_end_of_shader(cb, event, addr, value); }
DECOMP_LEAF u64 gnm_dispatch_write_release_mem(u64 cb, u32 event, u32 flags, u64 addr, u32 data_sel, u64 value, u32 cache_action,
                                               u32 cache_policy) {
    return w_release_mem(cb, event, flags, addr, data_sel, value, cache_action, cache_policy);
}
DECOMP_LEAF u64 gnm_wait_reg_mem(u64 cb, u64 addr, u32 mask, u32 function, u32 reference) {
    return w_wait(cb, addr, mask, function, reference);
}
DECOMP_LEAF u64 gnm_dispatch_wait_reg_mem(u64 cb, u64 addr, u32 mask, u32 function, u32 reference) {
    return w_wait(cb, addr, mask, function, reference);
}
DECOMP_LEAF u64 gnm_write_data_inline(u64 cb, u64 dest, u64 src, u32 dwords, u32 confirm) {
    return w_write_data(cb, dest, src, dwords, confirm);
}
DECOMP_LEAF u64 gnm_write_data_inline_l2(u64 cb, u64 dest, u64 src, u32 dwords, u32 cache_policy, u32 confirm) {
    return w_write_data_l2(cb, dest, src, dwords, cache_policy, confirm);
}
DECOMP_LEAF u64 gnm_set_predication(u64 cb, u64 block, u32 draw_if_visible, u32 hint) {
    return w_set_predication(cb, block, draw_if_visible, hint);
}
DECOMP_LEAF u64 gnm_clear_predication(u64 cb) { return w_clear_predication(cb); }
DECOMP_LEAF u64 gnm_dispatch_write_indirect_buffer(u64 cb, u64 addr, u32 dwords) { return w_indirect_buffer(cb, addr, dwords); }
// The counters pace the draw engine against a constant engine that writes
// nothing when nothing dumps its RAM (decomp/gx/constant_engine.cpp): then
// they write nothing either (gnm_ce_configure sets it).
u8 g_ce_counters_dead = 0;
std::atomic<u64> g_ce_counters_skipped{0};
DECOMP_LEAF u64 gnm_increment_de_counter(u64 cb) {
    if (g_ce_counters_dead) {
        g_ce_counters_skipped.fetch_add(1, std::memory_order_relaxed);
        return load<u64>(cb + 0x10);
    }
    return w_counter(cb, 0x00000000c0008500ull, 0x85);
}
DECOMP_LEAF u64 gnm_wait_on_ce_counter(u64 cb) {
    if (g_ce_counters_dead) {
        g_ce_counters_skipped.fetch_add(1, std::memory_order_relaxed);
        return load<u64>(cb + 0x10);
    }
    return w_counter(cb, 0x00000001c0008600ull, 0x86);
}

namespace {

// Every writer's arguments as the registers carry them: the six integer
// registers and the two stack arguments the widest takes.
struct Args {
    u64 r[6];
    u64 s[2];
};

// The game's function, written out, on those registers.
u64 write_as_game(std::size_t i, const Args& a) {
    const u64 cb = a.r[0];
    const auto d = [&](int k) { return static_cast<u32>(a.r[k]); };
    const auto s = [&](int k) { return static_cast<u32>(a.s[k]); };
    switch (i) {
        case kEndOfPipe: return w_end_of_pipe(cb, d(1), d(2), a.r[3], d(4), a.r[5], s(0), s(1));
        case kEndOfShader: return w_end_of_shader(cb, d(1), a.r[2], d(3));
        case kReleaseMem: return w_release_mem(cb, d(1), d(2), a.r[3], d(4), a.r[5], s(0), s(1));
        case kWait:
        case kComputeWait: return w_wait(cb, a.r[1], d(2), d(3), d(4));
        case kWriteData: return w_write_data(cb, a.r[1], a.r[2], d(3), d(4));
        case kWriteDataL2: return w_write_data_l2(cb, a.r[1], a.r[2], d(3), d(4), d(5));
        case kSetPredication: return w_set_predication(cb, a.r[1], d(2), d(3));
        case kClearPredication: return w_clear_predication(cb);
        case kIndirectBuffer: return w_indirect_buffer(cb, a.r[1], d(2));
        case kIncrementDeCounter: return w_counter(cb, 0x00000000c0008500ull, 0x85);
        case kWaitOnCeCounter: return w_counter(cb, 0x00000001c0008600ull, 0x86);
        default: return 0;
    }
}

bool has_payload(std::size_t i) { return i == kWriteData || i == kWriteDataL2; }
// The predication writers change the command buffer's flags (+0x28); the
// others leave it alone, and the compare does too (it may not be theirs).
bool sets_flags(std::size_t i) { return i == kSetPredication || i == kClearPredication; }

// The room the game's version asks for, in dwords (32-bit, as it adds).
u32 room_for(std::size_t i, const Args& a) { return has_payload(i) ? static_cast<u32>(a.r[3]) + 4 : kTable[i].dwords; }

// BBHOST_DECOMP_COMPARE=1: the game's version runs in ours' place on every
// call, and ours is run on the same command buffer beside it - the packet's
// bytes, the write pointer, the predication flag and the return value
// compared - then the game's result is left in place. A call the refill would
// serve runs the game's alone (the refill cannot run twice), and so does a
// WRITE_DATA of more than the compare's buffer or one whose payload lies where
// the packet goes.
DecompCompare g_counts[kWriters];
void* g_original[kWriters] = {};
std::atomic<std::uint64_t> g_compare_alone{0};
std::atomic<int> g_compare_logs{0};
constexpr u64 kCompareMax = 256;  // dwords

u64 compare(std::size_t i, const Args& a) {
    const u64 cb = a.r[0];
    GuestCallRegs g{};
    g.fn = g_original[i];
    for (int k = 0; k < 6; ++k) g.ints[k] = a.r[k];
    g.stack[0] = a.s[0];
    g.stack[1] = a.s[1];
    const u64 wp = load<u64>(cb + 0x10);
    const u64 dwords = has_payload(i) ? static_cast<u64>(static_cast<u32>(a.r[3])) + 4 : kTable[i].dwords;
    const std::size_t bytes = static_cast<std::size_t>(dwords) * 4;
    const u64 src = a.r[2];
    if (static_cast<u32>((load<u64>(cb + 0x8) - wp) >> 2) < room_for(i, a) || dwords > kCompareMax ||
        (has_payload(i) && src < wp + bytes && wp < src + bytes)) {
        g_compare_alone.fetch_add(1, std::memory_order_relaxed);
        hle_call_guest_regs(&g);
        // The packet the game's wrote, for the host list (it ends at the write
        // pointer, wherever the refill put it).
        const u64 after = load<u64>(cb + 0x10), packet = after - bytes;
        if (after != wp && after >= bytes && (load<u32>(packet) >> 30) == 3) host_lists::note(cb, packet, (load<u32>(packet) >> 8) & 0xff);
        return g.ret;
    }
    u8 before[kCompareMax * 4], game[kCompareMax * 4];
    const bool flagged = sets_flags(i);
    const u8 flags = flagged ? load<u8>(cb + 0x28) : 0;
    std::memcpy(before, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), bytes);
    hle_call_guest_regs(&g);
    const u64 game_ret = g.ret, game_wp = load<u64>(cb + 0x10);
    const u8 game_flags = flagged ? load<u8>(cb + 0x28) : 0;
    std::memcpy(game, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), bytes);
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), before, bytes);
    store<u64>(cb + 0x10, wp);
    if (flagged) store<u8>(cb + 0x28, flags);
    const u64 our_ret = write_as_game(i, a);
    const u64 our_wp = load<u64>(cb + 0x10);
    const u8 our_flags = flagged ? load<u8>(cb + 0x28) : 0;
    const bool same = our_ret == game_ret && our_wp == game_wp && our_flags == game_flags &&
                      std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(wp)), game, bytes) == 0;
    g_counts[i].calls.fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        g_counts[i].differ.fetch_add(1, std::memory_order_relaxed);
        if (g_compare_logs.fetch_add(1) < 16) {
            host_log("decomp: %s differs from the game's: returned 0x%llx (game 0x%llx), write pointer +%lld (game +%lld), flags 0x%x "
                     "(game 0x%x)",
                     kTable[i].name, static_cast<ull>(our_ret), static_cast<ull>(game_ret), static_cast<long long>(our_wp - wp),
                     static_cast<long long>(game_wp - wp), our_flags, game_flags);
        }
    }
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(wp)), game, bytes);
    store<u64>(cb + 0x10, game_wp);
    if (flagged) store<u8>(cb + 0x28, game_flags);
    return game_ret;
}

template <std::size_t I>
GUEST_ABI u64 compare_writer(u64 r0, u64 r1, u64 r2, u64 r3, u64 r4, u64 r5, u64 s0, u64 s1) {
    return compare(I, Args{{r0, r1, r2, r3, r4, r5}, {s0, s1}});
}

template <std::size_t... I>
std::array<void*, kWriters> compare_table(std::index_sequence<I...>) {
    return {reinterpret_cast<void*>(&compare_writer<I>)...};
}

void report() {
    static bool once = false;
    if (once) return;
    once = true;
    std::size_t ours = 0, compared = 0;
    for (std::size_t i = 0; i < kWriters; ++i) {
        const int placed = decomp_placed(kTable[i].name);
        if (placed == 1) ++ours;
        if (placed == 2) ++compared;
    }
    if (compared) {
        host_log("decomp: Gnm label, wait and predication writers: %zu compared (BBHOST_DECOMP_COMPARE=1), %llu calls ran the game's "
                 "alone (the refill served them, or a payload too large or under the packet)",
                 compared, static_cast<ull>(g_compare_alone.load()));
    } else {
        host_log("decomp: Gnm label, wait and predication writers: %zu of %zu ours, writing the game's packets%s", ours,
                 static_cast<std::size_t>(kWriters),
                 g_ce_counters_dead ? (std::string("; the draw engine's counter steps not written with the constant engine's: ") +
                                       std::to_string(g_ce_counters_skipped.load()))
                                          .c_str()
                                    : "");
    }
}

}  // namespace

// The census of who calls the packet writers (BBHOST_CP_WRITERS=1) hooks the
// game's own prologues of the end-of-pipe, end-of-shader and wait writers, so
// with it the game's versions stay.
void decomp_gx_labels_add() {
    if (const char* e = std::getenv("BBHOST_CP_WRITERS"); e && e[0] == '1') return;
    static void* const kOurs[kWriters] = {
        reinterpret_cast<void*>(&gnm_write_end_of_pipe_event),
        reinterpret_cast<void*>(&gnm_write_end_of_shader_event),
        reinterpret_cast<void*>(&gnm_dispatch_write_release_mem),
        reinterpret_cast<void*>(&gnm_wait_reg_mem),
        reinterpret_cast<void*>(&gnm_dispatch_wait_reg_mem),
        reinterpret_cast<void*>(&gnm_write_data_inline),
        reinterpret_cast<void*>(&gnm_write_data_inline_l2),
        reinterpret_cast<void*>(&gnm_set_predication),
        reinterpret_cast<void*>(&gnm_clear_predication),
        reinterpret_cast<void*>(&gnm_dispatch_write_indirect_buffer),
        reinterpret_cast<void*>(&gnm_increment_de_counter),
        reinterpret_cast<void*>(&gnm_wait_on_ce_counter),
    };
    static const std::array<void*, kWriters> kCompare = compare_table(std::make_index_sequence<kWriters>{});
    for (std::size_t i = 0; i < kWriters; ++i) {
        const LabelWriter& w = kTable[i];
        DecompFunction fn{w.name, "gx", w.bn, w.bytes, w.n, kOurs[i], DecompKind::Leaf};
        fn.original = &g_original[i];
        fn.compare = kCompare[i];
        fn.counts = &g_counts[i];
        fn.report = &report;
        decomp_add(fn);
    }
}
