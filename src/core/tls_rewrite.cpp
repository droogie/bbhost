#include "core/tls_rewrite.h"

#include "core/elf.h"
#include "core/memory.h"
#include "log.h"

#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <cstdio>
#endif

namespace {

#if defined(_WIN32)
const std::uint32_t kSlot = [] {
    const DWORD slot = TlsAlloc();
    if (slot == TLS_OUT_OF_INDEXES || slot >= kTebTlsSlotCount) std::abort();  // must sit in TEB.TlsSlots
    return static_cast<std::uint32_t>(slot);
}();
#elif defined(__APPLE__)
// The key's value is the guest TCB. Probe that it reads back at gs:[8*key]
// before any site depends on it: without that the guest finds no TCB.
const std::uint32_t kSlot = [] {
    pthread_key_t key = 0;
    const int e = pthread_key_create(&key, nullptr);
    if (e != 0 || key >= kTsdSlotCount) {
        std::fprintf(stderr, "[bbhost] tls: no pthread key for the guest's TCB (error %d, key %lu of %u)\n", e,
                     static_cast<unsigned long>(key), kTsdSlotCount);
        std::abort();
    }
    static const char probe = 0;
    pthread_setspecific(key, &probe);
    const void* seen = nullptr;
    asm volatile("movq %%gs:(%1), %0" : "=r"(seen) : "r"(static_cast<std::uint64_t>(key) * 8));
    pthread_setspecific(key, nullptr);
    if (seen != &probe) {
        std::fprintf(stderr,
                     "[bbhost] tls: pthread key %lu does not read back at gs:0x%lx (%p, want %p); this macOS keeps "
                     "its TSD elsewhere and the guest's TLS cannot work\n",
                     static_cast<unsigned long>(key), static_cast<unsigned long>(key) * 8, seen,
                     static_cast<const void*>(&probe));
        std::abort();
    }
    return static_cast<std::uint32_t>(key);
}();
#else
constexpr std::uint32_t kSlot = 60;  // the Linux stand-in's slot
#endif

// GS mode (the guest's fs:[0] reads rewritten to a GS slot, FS left to the
// host) is what Windows and macOS always run, and the Linux default since
// 2026-10-04: the thunk no longer switches FS twice a call (four
// alternating soaks: main-loop work 11.34/11.62 ms against 11.63/11.73, p95
// lower), and host code that touches thread-local state - libm's errno - can
// run raw. BBHOST_TLS_GS=0 goes back to switching FS, on Linux only.
const bool g_gs_mode = [] {
#if defined(_WIN32) || defined(__APPLE__)
    return true;
#else
    const char* e = std::getenv("BBHOST_TLS_GS");
    return !(e && e[0] == '0');
#endif
}();

}  // namespace

bool tls_gs_mode() { return g_gs_mode; }

std::uint32_t tls_gs_disp() { return kGsSlotsBase + 8 * kSlot; }

std::uint32_t tls_gs_slot() { return kSlot; }

void tls_rewrite_to_gs(ElfImage* image, std::uint64_t va, std::size_t size, TlsRewriteStats* st) {
    // mov rax, fs:[0]   64 48 8b 04 25 00 00 00 00
    static const std::uint8_t kFrom[9] = {0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0};
    const std::uint32_t disp = tls_gs_disp();
    auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, va));
    for (std::size_t i = 0; i + 9 <= size; ++i) {
        if (p[i] != 0x64) continue;
        if (std::memcmp(p + i, kFrom, 9) == 0) {
            p[i] = 0x65;  // gs
            std::memcpy(p + i + 5, &disp, 4);
            ++st->rewritten;
            i += 8;
            continue;
        }
    }
    // No other FS form exists in the code: a linear disassembly of every
    // executable segment (capstone, 2026-09-22) finds 17,127 `mov rax,
    // fs:[0]` and nothing else FS-relative below the end of the text at
    // 0x2fc0000; every other hit is string data the segment carries.
}

void tls_rewrite_report(const TlsRewriteStats& st) {
    host_log("tls: %zu `mov rax, fs:[0]` sites now read gs:[0x%x] (slot %u)%s", st.rewritten, tls_gs_disp(), kSlot,
             st.rewritten == 17127 ? "" : " - not the 17,127 of the 1.09 build");
}
