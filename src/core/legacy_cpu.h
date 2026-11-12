// Processors without BMI1, LZCNT or MOVBE: Intel's 2nd and 3rd generation
// Core (Sandy Bridge, Ivy Bridge) and AMD FX before Excavator. The game's
// code uses all three - 6,327 instructions of 1.09 - and such a processor
// either faults on them (BMI1, MOVBE: #UD) or, worse, runs them as something
// else without a word (LZCNT and TZCNT decode as BSR and BSF there: lzcnt(1)
// is 31, bsr(1) is 0). So at load, when CPUID says one is missing, every one
// of them is rewritten: a jmp rel32 in its place to a routine of ours
// (core/legacy_cpu_code.h) in an arena after the image, inside the image's
// own allocation and so within a rel32 of all of it, which computes the same
// result and flags and jumps back. The sites come from the eboot itself
// (tools/legacy_cpu_sites.py writes core/legacy_cpu_sites.inc), each checked
// to decode to its family before anything is written.
//
// An instruction shorter than the 5-byte jump (some LZCNT, TZCNT and MOVBE)
// takes whole instructions next to it into its routine - after it, else
// before it - which the tool proved nothing branches into; the bytes of a
// region past its jump are int3, and a trap there (something did branch
// in) is sent to the routine's copy of that instruction and logged once.
//
// The rewrite runs right after the eboot is loaded and relocated, before any
// of bbhost's patches, hooks and decomp entries and before any plugin's: a
// patch that expects the game's bytes where a site was rewritten is refused
// as it would be for any other change, and hooks and decomp trampolines,
// which copy the game's instructions elsewhere, take them as the game has
// them (legacy_cpu_original) with these instructions expanded into routines
// (legacy_relocate). legacy_cpu_verify checks before the guest starts that
// every jump is still in place.
//
// Any instruction of the families left (a site a later hook cut in two, an
// eboot that is not 1.09) still works where it faults: the illegal-
// instruction handler computes it in the thread's context and steps over it,
// logged once a site. LZCNT and TZCNT do not fault, which is why the rewrite
// covers every site rather than leaving them to that path.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ElfImage;

// The floor: AVX the operating system has turned on (CPUID's AVX and OSXSAVE,
// XGETBV's XMM and YMM state), SSE4.2 and POPCNT - Intel's 2nd generation
// Core, AMD FX. What is missing, by name; empty when nothing is.
std::string legacy_cpu_floor_missing();
// The check from main, before the eboot loads: false when the processor is
// below the floor, after saying so (a message box on Windows; stderr and the
// log on Linux).
bool legacy_cpu_check_floor();

// Whether the rewrite runs: when CPUID lacks BMI1 (leaf 7 EBX bit 3), LZCNT
// (0x80000001 ECX bit 5) or MOVBE (leaf 1 ECX bit 22). BBHOST_LEGACY_CPU=1
// forces it on, =0 off. legacy_cpu_log says which and why, once.
bool legacy_cpu_on();
void legacy_cpu_log();

// The bytes core/elf.cpp reserves after the image's tail for the routines
// (0 when the rewrite is off).
std::size_t legacy_cpu_arena_size();

// Rewrites every site of the 1.09 eboot in the loaded image. Once, after the
// SHA-256 check, before any patch or hook and before the guest starts.
void legacy_cpu_rewrite(ElfImage* image);
// Every rewritten site's jump still in place; logs any that a later write
// changed (`when`: "before the guest starts", "at exit").
void legacy_cpu_verify(const char* when);

// For code that writes or copies the game's code after the rewrite.
// The bytes [va, va + n) as the game has them: a rewritten region's own bytes
// where its jump is. Reads the image; without the rewrite, a plain copy.
void legacy_cpu_original(std::uint64_t va, std::size_t n, std::uint8_t* out);
// How [va, va + n) meets the rewritten regions: 0 none, 1 it holds whole
// regions only, -1 it cuts one (a write there would break its jump).
int legacy_cpu_meets(std::uint64_t va, std::size_t n);
// [va, va + n) is about to be written over (a hook's jump, a decomp entry,
// a patch) and holds whole regions only: they are given up, their routines
// left unused, so the check above no longer counts them. `who` is logged.
void legacy_cpu_release(std::uint64_t va, std::size_t n, const char* who);

// The fault paths. Linux: from the SIGILL and SIGTRAP handlers, with their
// ucontext_t; true when handled (the thread goes on). Windows: a vectored
// handler, first in line, for EXCEPTION_ILLEGAL_INSTRUCTION and
// EXCEPTION_BREAKPOINT (legacy_cpu_install_handler, once, from main).
bool legacy_cpu_on_sigill(void* ucontext);
bool legacy_cpu_on_sigtrap(void* ucontext);
void legacy_cpu_install_handler();

// For tests/legacy_cpu_sites_test.cpp: the rewritten regions in order -
// [lo, hi) in the image, the routine's [code, code_end), the bytes they held.
struct LegacyCpuRegion {
    std::uint64_t lo, hi, code, code_end;
    const std::uint8_t* orig;
    bool live;
};
std::size_t legacy_cpu_regions(LegacyCpuRegion* out, std::size_t cap);

// A pc inside a routine: the game instruction it stands for (its runtime
// address), else pc itself. Lock-free, for the fault handlers and the
// write watch's census.
std::uint64_t legacy_cpu_guest_pc(std::uint64_t pc);
// For a crash report: a line naming the site a routine's pc belongs to
// (nothing when pc is not in one).
void legacy_cpu_crash_line(std::uint64_t pc);
