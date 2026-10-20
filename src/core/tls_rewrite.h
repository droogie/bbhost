// Guest TLS through GS instead of FS (the Windows shape).
//
// The eboot reaches its thread block with `mov rax, fs:[0]` (FreeBSD's TCB
// self pointer at FS:0): 17,127 sites, all the same nine bytes
// `64 48 8b 04 25 00 00 00 00`. Windows x64 keeps the TEB at GS and lets
// nothing set FS, so on Windows those become `mov rax, gs:[TEB.TlsSlots+8*n]`
// (`65 48 8b 04 25 disp32`), the same length, with the guest TCB stored in
// TLS slot n. The rewrite is done at load, while the text is still writable.
//
// On Linux the same rewrite runs behind BBHOST_TLS_GS=1: GS gets a block the
// size of a TEB whose slot n holds the TCB, FS stays glibc's for good, and
// the thunks stop switching FS - so the Windows-shaped path is proven on
// the game here, before a Windows build exists.
//
// macOS lets no program set FS or GS either, but libpthread keeps each
// thread's TSD at GS (key k at gs:[8*k]), under Rosetta 2 too. There the
// slot is a pthread key of ours whose value is the TCB.
#pragma once

#include <cstddef>
#include <cstdint>

struct ElfImage;

// TEB.TlsSlots on Windows x64, and the slot this host uses when it picks one
// itself (Linux); on Windows the slot comes from TlsAlloc.
constexpr std::uint32_t kTebTlsSlots = 0x1480;
constexpr std::uint32_t kTebTlsSlotCount = 64;
constexpr std::size_t kTebStandInSize = 0x1800;

// Where the slots start in the GS segment: TEB.TlsSlots on Windows and in
// the Linux stand-in; on macOS GS is the TSD array itself, key 0 at gs:0.
#if defined(__APPLE__)
constexpr std::uint32_t kGsSlotsBase = 0;
// libpthread's TSD array (_INTERNAL_POSIX_THREAD_KEYS_END); a key past it is
// not at gs:[8*key].
constexpr std::uint32_t kTsdSlotCount = 768;
#else
constexpr std::uint32_t kGsSlotsBase = kTebTlsSlots;
#endif

// True when the guest reads its TCB through GS (BBHOST_TLS_GS=1 on Linux,
// always on Windows and macOS). Read anywhere; fixed before the eboot is
// loaded.
bool tls_gs_mode();
// The displacement the rewritten sites read: kGsSlotsBase + 8 * slot, and
// the slot itself (Windows: from TlsAlloc, below 64 so it is in the TEB;
// macOS: a pthread key).
std::uint32_t tls_gs_disp();
std::uint32_t tls_gs_slot();

struct TlsRewriteStats {
    std::size_t rewritten = 0;  // mov rax, fs:[0] sites turned to GS (17,127 on 1.09)
};
// Rewrites every `mov rax, fs:[0]` in one executable segment (guest VA and
// size) to the GS form, adding to `st`. Call before the segment is made
// read-only; tls_rewrite_report logs the totals.
void tls_rewrite_to_gs(ElfImage* image, std::uint64_t va, std::size_t size, TlsRewriteStats* st);
void tls_rewrite_report(const TlsRewriteStats& st);
