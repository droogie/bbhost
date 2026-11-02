#pragma once

// The post-processing switches, as byte patches on the eboot.
//
// Three sessions went into looking for these in data - the YEBIS debug block
// (a mirror the renderer ignores), Yebis2ParamFile.filterparam (never opened)
// and GPARAM (loaded, edited, ignored). They are not data at all: they are a
// constructor's immediate and a branch.
//
//   mov dword [rbx+0x2968], 0x01010101
//
// is one instruction writing four adjacent bools - SSAO, motion blur,
// anti-alias and a fourth - so each is a byte of that immediate, and zeroing
// one turns the effect off at construction. Depth of field is a branch
// instead: `cmp dword [rbx+0x1ac], 0; je +0xde` around its setup, forced.
//
// The addresses come from the shadPS4 community patch list for 1.09 and were
// checked against the eboot before being used: the flag site really does hold
// `01 01 01 01` and the DoF site really is that `je`. Every patch is gated on
// the eboot's SHA-256 and refuses on an unexpected byte, like the rest.
//
// Motion blur is the exception: its bool is left alone, because clearing it
// blows the frame out here, and the blur shader is neutralised instead
// (host/shader_patch.h).
//
// All five follow their settings live now (graphics_patch.cpp, "Live"): the
// bools are given to each renderer at its next view after a change (and to a
// renderer made since), chromatic aberration's 1.0 is rewritten in place, and
// motion blur and depth of field are shader patches. The render resolution
// changes live too (engine/live_resolution.h).

#include <cstddef>
#include <cstdint>
#include <string>

struct ElfImage;

void graphics_patch_install(ElfImage* image);
// The flip count when the scene's view (sub_269e990) was last drawn, plus
// one; 0 before any. The title, loading screens and movies draw none.
std::uint64_t engine_scene_view_flip();
// For the 300-flip report: the Bloom setting, the YEBIS views since the last
// call and how many had a glare luminance above 0 (YEBIS draws a camera's
// glare only then; Bloom 0 makes it 0 in every view); empty when no view was
// recorded.
std::string graphics_glare_report();
// A scene context - the renderer whose SSAO and AA bytes the settings write -
// is being constructed at `ctx` (engine/live_resolution.cpp's constructor
// hook): its constructor turns both on, so it is given the settings at its
// next view, even where an earlier context stood at the same address.
void graphics_scene_context_made(std::uint64_t ctx);

// A hook at a guest function's entry (core/thunk.h's prologue stub): `host`
// runs with the argument registers first. `prologue` is the `n` (>= 14) bytes
// expected there, none rip-relative; false when they are not. Only after the
// eboot's SHA-256 check.
bool engine_prologue_hook(ElfImage* image, std::uint64_t at, const std::uint8_t* prologue, std::size_t n, void* host,
                          std::uint64_t id = 0);
// The same, running `exec` (m bytes) in the stub instead of the displaced
// prologue: for a prologue whose instructions are not position-independent
// (a rip-relative lea becomes a movabs of the slid address).
bool engine_prologue_hook_exec(ElfImage* image, std::uint64_t at, const std::uint8_t* prologue, std::size_t n,
                               const std::uint8_t* exec, std::size_t m, void* host, std::uint64_t id = 0);  // id: what the stub passes host as its first argument
// A hook inside a function body, between two instructions: `bytes` (n >= 14,
// whole instructions, none rip-relative, no branch landing inside them) are
// moved into the stub, which keeps rsp aligned as a body has it. `host` runs
// with the argument registers as they are there and must return 0.
bool engine_site_hook(ElfImage* image, std::uint64_t at, const std::uint8_t* bytes, std::size_t n, void* host,
                      std::uint64_t id = 0);
// A `call rel32` (or `jmp rel32`) at `at`, whose destination must be
// `target`, sent through a stub: `host` runs with the argument registers and,
// returning nonzero, stands for the call (it made the call itself, around
// its own work); 0 goes on to `target`.
bool engine_call_site_hook(ElfImage* image, std::uint64_t at, std::uint64_t target, void* host);
// The same, for a variadic guest function: the stub keeps rax, which SysV uses
// to tell the callee how many vector registers hold arguments (core/thunk.h).
bool engine_prologue_hook_variadic(ElfImage* image, std::uint64_t at, const std::uint8_t* prologue, std::size_t n,
                                   void* host, std::uint64_t id = 0);
