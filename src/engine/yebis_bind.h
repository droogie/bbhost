#pragma once

// YEBIS's resource binding, as our source instead of the game's.
//
// The YEBIS draw flip (BBHOST_GX_NATIVE_YEBIS=5) already draws from the state
// we read ourselves, but it still made seven guest calls a draw - and every one
// of them exists only to build a PM4 packet:
//
//   0x15fd230  ring open    WAIT_ON_DE_COUNTER_DIFF   (CE)
//   0x15f7840  the builder  WRITE_CONST_RAM + DUMP_CONST_RAM per stage (CE)
//   0x1473b80  WRITE_DATA                             (DE)
//   0x1477f40  tail         DMA_DATA                  (DE)
//   0x14781d0  tail         WAIT_ON_CE_COUNTER        (DE)
//   0x15fd190  ring close   INCREMENT_CE_COUNTER      (CE)
//   0x15f9d10  post-draw    INCREMENT_DE_COUNTER      (DE), and one state store
//
// What the builder is doing underneath is a double-buffered upload. Each
// bindable resource has a window in YEBIS's per-stage array at *(state + 8),
// and a bitmap saying which records in that window the game has changed.
// WRITE_CONST_RAM copies the changed records into constant RAM *at the same
// offset* - constant RAM is a mirror of the array - and DUMP_CONST_RAM copies
// the whole window out to the next slot of a 16-slot ring, so the shader reads
// a snapshot that the CPU is free to go on editing. The counters are what stop
// the ring lapping work the GPU has not finished.
//
// So the native version keeps a mirror of its own, refreshes the changed
// records into it, and hands the window to the command processor to write at
// the draw's own token - the point where the renderer resolves the descriptors,
// and the same place in the stream the dump would have been. The mirror is what
// makes it byte-for-byte: a record the game changed without marking it dirty
// must not reach the shader early, exactly as on the hardware.
//
// BBHOST_YEBIS_BIND: 0 the guest builder, 1 ours (default), 2 compare - ours
// predicts, the guest's runs, and every address is checked against it.
// BBHOST_YEBIS_BIND_VERIFY=1 also checks every window against constant RAM.

#include <cstdint>

namespace yebis_bind {

// A window waiting to be written to its ring slot, held in the transit arena
// until the command processor reaches the draw.
struct Copy {
    std::uint64_t dst;
    std::uint64_t seq;  // the arena's count before this one, to catch a lapped window
    std::uint32_t at;   // into the transit arena
    std::uint32_t bytes;
};
// The live tables need three (one 0x17 on the vertex side, a rare 0x13 and the
// extended block on the pixel side); a stage that wants more falls back.
constexpr int kMaxCopies = 12;
struct Work {
    Copy copies[kMaxCopies];
    int n = 0;
};

int mode();  // 0 guest, 1 native, 2 compare

// Whether every descriptor in a stage's table is one build_stage() knows, which
// has to be settled for all of a draw's stages before any of them is built: a
// stage it cannot finish has already moved the rings by the time it finds out.
bool table_known(std::uint64_t table, std::uint32_t count);
void note_fallback();

// One stage's descriptor table, resolved the way sub_15f7840 resolves it:
// `block[i]` gets the address descriptor i resolved to, the extended user-data
// dwords are staged into YEBIS's own array, the rings advance, and the windows
// are stashed in `work`. `dry` predicts and keeps the mirror current without
// touching any guest state, which is what compare mode runs.
//
// False means the table holds something this has not been verified against, and
// the caller must run the guest builder for that stage.
bool build_stage(std::uint64_t state, int stage, std::uint64_t table, std::uint32_t count, std::uint64_t* block, Work& work,
                 bool dry = false);

// The user data a stage's table gives its shader, as sub_15f9190 writes it
// (user-data slots 0-15): a plain descriptor's record dwords out of the stage's
// array, the others' addresses from `block` (what build_stage resolved). The
// slots it wrote, as a mask.
std::uint16_t user_data(std::uint64_t state, int stage, std::uint64_t table, std::uint32_t count, const std::uint64_t* block,
                        std::uint32_t* user);

// One more deferred write, for a payload the guest would have carried inside a
// WRITE_DATA packet: a copy is taken now, written at the token. False when the
// draw has no room left for it.
bool add_copy(Work& work, std::uint64_t dst, const void* src, std::uint32_t bytes);

// The windows, written where the descriptors are read: on the command-processor
// thread, at the draw's token.
void run(const Work& work);
// The same for a dispatch, which reads guest memory where the GPU runs it: the
// windows go on the GPU's timeline in stream order, before it.
void run_ordered(const Work& work);

// Compare mode: what a stage predicted, against what the guest builder left.
void compare(int stage, const std::uint64_t* mine, const std::uint64_t* theirs, std::uint32_t count);

// BBHOST_YEBIS_BIND_VERIFY=1: the window we stashed, against the one the
// guest's constant engine dumped for the same draw.
void check_dump(std::uint32_t off, const void* ce, std::uint32_t bytes);
bool verifying();

// The counters, for the 300-flip report.
void report();

}  // namespace yebis_bind
