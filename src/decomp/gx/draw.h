// GX's immediate context as source (decomp/gx/draw.cpp): what the GX trace
// (hle/gx_trace.cpp) and the methods share.
#pragma once

#include <cstdint>

namespace decomp_gx {

// The GX trace's entry, gx_hook(kind, frame), behind the FS-switching thunk
// (thunk_wrap): what the prologue stubs called before these methods were
// ours, and now ours calls first. Set once by hle_gx_trace_arm, before the
// game draws; null when the trace and the GX backend are off, and ours is
// then the game's method alone. The frame is the stubs' - r9, r8, rcx, rdx,
// rsi, rdi, the return address, a stack argument - with a return slot below
// it; the kind is the trace's (these values, hle/gx_trace.cpp checks them).
extern void* g_gx_hook_entry;
enum MethodKind : std::uint64_t {
    kGxDraw = 0,
    kGxDrawIndexed = 1,
    kGxDrawInstanced = 2,
    kGxDrawIndexedInstanced = 3,
    kGxDrawInstancedIndirect = 4,
    kGxDrawIndexedInstancedIndirect = 5,
    kGxCommit = 6,
    kGxDispatch = 7,
    kGxDispatchIndirect = 8,
};
// The methods' names on the decomp list, by kind: hle_gx_trace_arm installs
// no stub over one that is placed (it would not find the game's prologue).
extern const char* const kMethodNames[9];

// Hands a call to the GX trace as a prologue stub over the game's function
// did - the arguments' registers as the frame has them, the caller's return
// address - and says whether it took the call whole (nothing more runs).
// False when the trace is not armed. Leaf-safe: the entry switches to the
// host's FS itself.
bool hand_to_trace(std::uint64_t kind, std::uint64_t rdi, std::uint64_t rsi, std::uint64_t rdx, std::uint64_t rcx, std::uint64_t r8,
                   std::uint64_t r9, std::uint64_t ret);
// The trace's kinds for YEBIS's Gnm draw and dispatch wrappers
// (decomp/yebis/gnm.cpp), and whether it hooks the dispatch at all
// (BBHOST_GX_YEBIS_DISPATCH): set by hle_gx_trace_arm.
extern std::uint64_t g_yebis_draw_kind, g_yebis_dispatch_kind;
extern bool g_yebis_dispatch_hooked;

// What the GX trace takes for the caller of a Gnm writer it hooks. It knows a
// GX method's packets, a YEBIS draw's and the like by the return address of
// the writer call; ours make those calls from the host's code. So every such
// function of ours lives in one section (DECOMP_GX_WRITER_CALLER, the bounds
// below), is told with the game's return address from the same writer call
// (note_writer_caller, as it is added to the decomp list), and a return
// address inside the section is read as that game address. Any other address
// is itself. A guest address either way.
#ifdef _WIN32
#define DECOMP_GX_WRITER_CALLER __attribute__((section("bbgxm$m")))
#else
#define DECOMP_GX_WRITER_CALLER __attribute__((section("bbhost_gx_methods")))
#endif
void note_writer_caller(const void* ours, std::uint64_t game_return_bn);
extern std::uint64_t g_method_code_lo, g_method_code_hi;
std::uint64_t game_return_in_methods(std::uint64_t ret);
inline std::uint64_t game_return(std::uint64_t ret) {
    return ret - g_method_code_lo < g_method_code_hi - g_method_code_lo ? game_return_in_methods(ret) : ret;
}

}  // namespace decomp_gx
