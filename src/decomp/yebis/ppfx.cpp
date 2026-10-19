// YEBIS's PPFX entry points as source (docs/decomp.md): the two calls the
// engine runs its whole post-processing chain with - ppfxBindFramebuffer
// (0xfb68f0), which gives YEBIS's single post effect the frame's colour and
// depth to read and write (its SetRenderTarget, 0xfd82f0), and
// ppfxApplyEffects (0xfb6980), which runs every effect in it (its
// ApplyEffects, 0xfe95e0: tone mapping, glare, depth of field, motion blur,
// colour grading, each a run of passes through YEBIS's D3D11-shaped context,
// decomp/yebis/context.cpp). Names as the binary's own messages give them
// (ppfx.cpp's). Each returns true when the effect's call succeeded; a failure
// is printed when YEBIS's debug output is on, and a failed ApplyEffects
// asserts, as the game's do.
//
// That is the post-processing chain as one pass: what a mod would hook to
// run a pass of its own before or after the whole chain, or in its place
// (docs/modding.md). Leaves: the game's calls in the game's order.
#include "decomp/decomp.h"
#include "decomp/guest.h"

#include <cstdint>

using namespace decomp;

#ifdef DECOMP_YEBIS_PPFX_MUTANTS
extern int g_ppfx_mutant;
#define MUTANT(n) (g_ppfx_mutant == (n))
#else
#define MUTANT(n) false
#endif

namespace {

constexpr u64 kSinglePostEffect = 0x5865ed0;     // s_pSinglePostEffect
constexpr u64 kDebugOut = 0x5519a68, kDebugOutToo = 0x5519a70;  // u32s: YEBIS's debug output on (both)
constexpr u64 kSetRenderTarget = 0xfd82f0;       // CPostEffect::SetRenderTarget(effect, a, b, c, -1, f)
constexpr u64 kApplyEffects = 0xfe95e0;          // CPostEffect::ApplyEffects(effect, 0, 0)
constexpr u64 kPrint = 0xfb1a90;                 // YEBIS's debug print (format, ...)
constexpr u64 kErrorText = 0x104ede0;            // an HRESULT's text
constexpr u64 kAssert = 0x2fbf178;               // _Assert(text, function)
// The messages (ppfx.cpp's).
constexpr u64 kErrorPrefix = 0x4bfc59d;          // "...ppfx.cpp(%d) : PPFX ERROR: "
constexpr u64 kBindFailed = 0x4bfe3af;           // "ppfxBindFramebuffer(): s_pSinglePostEffect->SetRenderTarget() failed (0x%x: %s).\n"
constexpr u64 kApplyFailed = 0x4bfe401;          // "ppfxBindFramebuffer(): s_pSinglePostEffect->ApplyEffects() failed (0x%x: %s).\n"
constexpr u64 kApplyAssert = 0x4bfe450, kApplyAssertFunction = 0x4bfe4aa;  // ppfx.cpp(6845), "ppfxApplyEffects"
constexpr u32 kBindLine = 0x1aa6, kApplyLine = 0x1abb;

using Print = GUEST_ABI u64 (*)(u64, ...);
using One = GUEST_ABI u64 (*)(u64);
using Two = GUEST_ABI u64 (*)(u64, u64);
using Three = GUEST_ABI u64 (*)(u64, u64, u64);
using Six = GUEST_ABI u64 (*)(u64, u64, u64, u64, u64, u64);

bool debug_out() { return load<u32>(game_address(kDebugOut)) && load<u32>(game_address(kDebugOutToo)); }

// The failure's two lines: where, then what and its text.
void print_failure(u32 line, u64 message, s32 hr) {
    game_function<Print>(kPrint)(game_address(kErrorPrefix), static_cast<u64>(line));
    const u64 text = game_function<One>(kErrorText)(static_cast<u32>(hr));
    game_function<Print>(kPrint)(game_address(message), static_cast<u64>(static_cast<u32>(hr)), text);
}

}  // namespace

// ppfxBindFramebuffer(a, b, c, _, _, f): the effect's SetRenderTarget(a, b,
// c, -1, f) - the fourth and fifth arguments are not passed on.
DECOMP_LEAF bool ppfxBindFramebuffer(u32 a, u64 b, u32 c, u64, u64, u64 f) {
    const s32 hr = static_cast<s32>(
        game_function<Six>(kSetRenderTarget)(load<u64>(game_address(kSinglePostEffect)), a, b, c, MUTANT(1) ? 0 : 0xffffffffu, f));
    if (hr >= 0) return true;
    if (debug_out()) print_failure(kBindLine, kBindFailed, hr);
    return false;
}

// ppfxApplyEffects(): the effect's ApplyEffects(0, 0).
DECOMP_LEAF bool ppfxApplyEffects() {
    const s32 hr = static_cast<s32>(game_function<Three>(kApplyEffects)(load<u64>(game_address(kSinglePostEffect)), 0, 0));
    if (MUTANT(2) ? hr > 0 : hr >= 0) return true;
    if (debug_out()) print_failure(kApplyLine, kApplyFailed, hr);
    game_function<Two>(kAssert)(game_address(kApplyAssert), game_address(kApplyAssertFunction));
    return false;
}

namespace {
// push rbp; mov rbp, rsp; push r14; push rbx; mov eax, edx; mov rcx, rsi
constexpr u8 kEntryBind[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x89, 0xd0, 0x48, 0x89, 0xf1};
// push rbp; mov rbp, rsp; push rbx; push rax (the rip-relative load after is not taken)
constexpr u8 kEntryApply[] = {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50};
}  // namespace

void decomp_yebis_ppfx_add() {
    const DecompFunction fns[] = {
        {"ppfxBindFramebuffer", "yebis", 0xfb68f0, kEntryBind, sizeof kEntryBind, reinterpret_cast<void*>(&ppfxBindFramebuffer), DecompKind::Leaf},
        {"ppfxApplyEffects", "yebis", 0xfb6980, kEntryApply, sizeof kEntryApply, reinterpret_cast<void*>(&ppfxApplyEffects), DecompKind::Leaf},
    };
    for (const DecompFunction& fn : fns) decomp_add(fn);
}
