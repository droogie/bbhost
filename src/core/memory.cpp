#include "core/memory.h"
#include "log.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "core/win_vm.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

bool guest_alloc(GuestMemory* mem, std::uint64_t slide, std::size_t size) {
    mem->slide = slide;
    mem->size = size;
#if defined(_WIN32)
    // The preferred slide when it is free (it is not under wine, whose
    // process heap sits there before main runs), else anywhere below 4 GiB:
    // every address-based patch converts through the real slide.
    const std::size_t round = (size + 0xffff) & ~std::size_t(0xffff);
    mem->base = VirtualAlloc(reinterpret_cast<void*>(slide), round, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!mem->base) {
        // The fixed fallback window kernel.cpp reserved (0x300000000), else anywhere below 4 GiB.
        mem->base = win_vm_commit(reinterpret_cast<void*>(0x300000000ull), round, PAGE_READWRITE, 0xffffffffull);
        if (!mem->base) mem->base = win_vm_commit(nullptr, round, PAGE_READWRITE, 0xffffffffull);
        if (mem->base) {
            mem->slide = reinterpret_cast<std::uint64_t>(mem->base);
            host_log("guest slide is 0x%llx (preferred 0x%llx was taken)", static_cast<unsigned long long>(mem->slide),
                     static_cast<unsigned long long>(slide));
        }
    }
    if (!mem->base) {
        host_log("VirtualAlloc(0x%llx, %zu) failed: %lu",
                 static_cast<unsigned long long>(slide), size, GetLastError());
        return false;
    }
#else
    mem->base = mmap(reinterpret_cast<void*>(slide), size,
                     PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem->base == MAP_FAILED) {
        host_log("mmap(%zu) failed", size);
        mem->base = nullptr;
        return false;
    }
    if (mem->base != reinterpret_cast<void*>(slide)) {
        mem->slide = reinterpret_cast<std::uint64_t>(mem->base);
        host_log("guest slide is 0x%llx (preferred 0x%llx was a hint)",
                 static_cast<unsigned long long>(mem->slide),
                 static_cast<unsigned long long>(slide));
    }
#endif
    std::memset(mem->base, 0, size);
    return true;
}

void guest_free(GuestMemory* mem) {
    if (!mem->base) {
        return;
    }
#if defined(_WIN32)
    VirtualFree(mem->base, 0, MEM_RELEASE);
#else
    munmap(mem->base, mem->size);
#endif
    mem->base = nullptr;
}

#if defined(_WIN32)
static bool win_protect(void* p, std::size_t size, DWORD prot) {
    DWORD old = 0;
    return VirtualProtect(p, size, prot, &old) != 0;
}
#endif

// The pages guest_keep_writable keeps writable (host addresses), and their
// protection given back after a change that covered one.
static std::mutex g_kept_mu;
static std::vector<std::uintptr_t> g_kept;

static void protect_page_rwx(std::uintptr_t page) {
#if defined(_WIN32)
    win_protect(reinterpret_cast<void*>(page), 0x1000, PAGE_EXECUTE_READWRITE);
#else
    mprotect(reinterpret_cast<void*>(page), 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC);
#endif
}

static void keep_kept_writable(const void* p, std::size_t size) {
    std::lock_guard<std::mutex> lock(g_kept_mu);
    const auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + size;
    for (const std::uintptr_t page : g_kept) {
        if (page < hi && page + 0x1000 > lo) protect_page_rwx(page);
    }
}

void guest_keep_writable(GuestMemory* mem, std::uint64_t va) {
    const auto page = reinterpret_cast<std::uintptr_t>(guest_ptr(*mem, va & ~0xfffull));
    {
        std::lock_guard<std::mutex> lock(g_kept_mu);
        bool have = false;
        for (const std::uintptr_t k : g_kept) have = have || k == page;
        if (!have) g_kept.push_back(page);
    }
    protect_page_rwx(page);
}

bool guest_protect_rx(GuestMemory* mem, std::uint64_t va, std::size_t size) {
    void* p = guest_ptr(*mem, va);
#if defined(_WIN32)
    if (!win_protect(p, size, PAGE_EXECUTE_READ)) {
        host_log("PAGE_EXECUTE_READ failed at 0x%llx", static_cast<unsigned long long>(va));
        return false;
    }
#else
    if (mprotect(p, size, PROT_READ | PROT_EXEC) != 0) {
        host_log("mprotect RX failed at 0x%llx", static_cast<unsigned long long>(va));
        return false;
    }
#endif
    keep_kept_writable(p, size);
    return true;
}

// The executable segments guest_mark_executable named (host addresses).
static std::vector<std::pair<std::uintptr_t, std::uintptr_t>> g_exec;

void guest_mark_executable(GuestMemory* mem, std::uint64_t va, std::size_t size) {
    const auto lo = reinterpret_cast<std::uintptr_t>(guest_ptr(*mem, va));
    std::lock_guard<std::mutex> lock(g_kept_mu);
    g_exec.emplace_back(lo, lo + size);
}

static bool protect_rw_range(void* p, std::size_t size, bool exec) {
#if defined(_WIN32)
    return win_protect(p, size, exec ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
#else
    return mprotect(p, size, PROT_READ | PROT_WRITE | (exec ? PROT_EXEC : 0)) == 0;
#endif
}

bool guest_protect_rw(GuestMemory* mem, std::uint64_t va, std::size_t size) {
    void* p = guest_ptr(*mem, va);
    // Within an executable segment the pages keep execute (guest_mark_executable).
    const auto lo = reinterpret_cast<std::uintptr_t>(p), hi = lo + size;
    constexpr std::uintptr_t kPage = 0xfff;
    std::uintptr_t elo = 0, ehi = 0;  // the executable part, in whole pages
    {
        std::lock_guard<std::mutex> lock(g_kept_mu);
        for (const auto& [a, b] : g_exec) {
            if (a < hi && lo < b) {
                elo = std::max(a, lo) & ~kPage;
                ehi = (std::min(b, hi) + kPage) & ~kPage;
                break;
            }
        }
    }
    bool ok = true;
    if (ehi <= elo) {
        ok = protect_rw_range(p, size, false);
    } else {
        if (elo > lo) ok = protect_rw_range(p, elo - lo, false);
        ok = protect_rw_range(reinterpret_cast<void*>(elo), ehi - elo, true) && ok;
        if (ehi < hi) ok = protect_rw_range(reinterpret_cast<void*>(ehi), hi - ehi, false) && ok;
    }
    keep_kept_writable(p, size);
    return ok;
}

bool guest_protect_rwx(GuestMemory* mem, std::uint64_t va, std::size_t size) {
    void* p = guest_ptr(*mem, va);
#if defined(_WIN32)
    return win_protect(p, size, PAGE_EXECUTE_READWRITE);
#else
    return mprotect(p, size, PROT_READ | PROT_WRITE | PROT_EXEC) == 0;
#endif
}
