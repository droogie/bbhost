#include "hle/equeue.h"
#include "hle/sync.h"
#include "core/futex.h"
#include "core/host_clock.h"
#include "core/tls_rewrite.h"
#include "host/frame_stats.h"
#include "engine/gx_resources.h"
#include "core/write_watch.h"
#include "hle/modules.h"
#include "hle/common.h"
#include "hle/platform.h"
#include "hle/hle.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <csetjmp>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include "core/win_vm.h"
#include <pthread_time.h>  // winpthreads: clock_gettime and the POSIX clock ids
#else
#include <fcntl.h>
#ifndef SYS_memfd_create
#define SYS_memfd_create 319
#endif
#endif

namespace {

std::mutex g_dmem_mu;

// The direct memory the game is told it has (sceKernelGetDirectMemorySize):
// 6 GiB, more when the render-target heap is grown past what fits in it
// (hle_kernel_set_dmem_size, engine/live_resolution.cpp), up to what the
// window reserved for it holds - it ends below the image's fallback base.
constexpr std::uint64_t kDmemBaseSize = 0x180000000ull;
constexpr std::uint64_t kDmemMaxSize = 0x1f0000000ull;
std::uint64_t g_dmem_size = kDmemBaseSize;  // fixed once the backing exists
// Gnm addresses are 40-bit. Place CPU dmem below 2^40 so (va >> 8) << 8 is identity.
constexpr std::uint64_t kDmemVaBase = 0x100000000ull;
// GX also fills command buffers at this window + phys when phys fits in 4 GiB.
constexpr std::uint64_t kGpuDmemBase = 0xFF00000000ull;
constexpr std::uint64_t kGnmVaMask = 0xFFFFFFFFFFull;
// Windows: where the image goes when its preferred slide is taken (memory.cpp).
constexpr std::uint64_t kImageFallbackBase = 0x300000000ull;
constexpr std::uint64_t kPage = 0x4000;  // Orbis page

// SCE_KERNEL_PROT_*
constexpr int kProtCpuRead = 1;
constexpr int kProtCpuWrite = 2;
constexpr int kProtCpuExec = 4;
constexpr int kProtGpuRead = 0x10;
constexpr int kProtGpuWrite = 0x20;

// Direct memory: a sorted list of live allocations over the physical range.
struct DmemAlloc {
    std::uint64_t phys = 0;
    std::uint64_t len = 0;
    int type = 0;
};
std::vector<DmemAlloc> g_dmem_allocs;  // sorted by phys, non-overlapping
std::uint64_t g_dmem_used = 0;
std::uint64_t g_dmem_high = 0;

// Virtual mappings. A direct mapping may carry a GPU alias record whose
// `alias_of` points back at the CPU mapping's start.
struct MappedRegion {
    std::uintptr_t start = 0;
    std::uint64_t len = 0;
    std::int64_t phys = -1;   // -1 for flexible / reserved
    int prot = 0x33;
    int type = 0;
    bool dmem = false;
    std::uintptr_t alias_of = 0;  // non-zero: this is the GPU alias of that CPU va
};
std::vector<MappedRegion> g_maps;
std::atomic<std::uint64_t> g_maps_gen{1};  // bumped on every change to g_maps

#if defined(_WIN32)
HANDLE g_dmem_map = nullptr;
#else
int g_dmem_fd = -1;
#endif

std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) & ~(a - 1); }
bool pow2(std::uint64_t a) { return a && (a & (a - 1)) == 0; }

int dmem_type_at_locked(std::int64_t phys) {
    for (const auto& a : g_dmem_allocs) {
        if (phys >= 0 && static_cast<std::uint64_t>(phys) >= a.phys &&
            static_cast<std::uint64_t>(phys) < a.phys + a.len) {
            return a.type;
        }
    }
    return 0;
}

// First-fit over the gaps between live allocations inside [start, end).
int dmem_alloc_locked(std::uint64_t start, std::uint64_t end, std::uint64_t len, std::uint64_t align,
                      int type, std::uint64_t* out) {
    if (end == 0 || end > g_dmem_size) {
        end = g_dmem_size;
    }
    if (align < kPage) {
        align = kPage;
    }
    if (!pow2(align) || len == 0 || (len & (kPage - 1)) != 0) {
        return EINVAL;
    }
    std::uint64_t cand = align_up(start, align);
    std::size_t i = 0;
    for (; i < g_dmem_allocs.size(); ++i) {
        const auto& a = g_dmem_allocs[i];
        if (a.phys + a.len <= cand) {
            continue;
        }
        if (cand + len <= a.phys) {
            break;
        }
        cand = align_up(a.phys + a.len, align);
    }
    if (cand + len > end) {
        return EAGAIN;
    }
    g_dmem_allocs.insert(g_dmem_allocs.begin() + static_cast<std::ptrdiff_t>(i), DmemAlloc{cand, len, type});
    g_dmem_used += len;
    if (g_dmem_used > g_dmem_high) {
        g_dmem_high = g_dmem_used;
    }
    *out = cand;
    return 0;
}

void dmem_release_locked(std::uint64_t phys, std::uint64_t len) {
    const std::uint64_t lo = phys;
    const std::uint64_t hi = phys + len;
    std::vector<DmemAlloc> keep;
    keep.reserve(g_dmem_allocs.size() + 1);
    for (const auto& a : g_dmem_allocs) {
        const std::uint64_t alo = a.phys;
        const std::uint64_t ahi = a.phys + a.len;
        if (ahi <= lo || alo >= hi) {
            keep.push_back(a);
            continue;
        }
        if (alo < lo) {
            keep.push_back(DmemAlloc{alo, lo - alo, a.type});
        }
        if (ahi > hi) {
            keep.push_back(DmemAlloc{hi, ahi - hi, a.type});
        }
        const std::uint64_t cut = std::min(ahi, hi) - std::max(alo, lo);
        g_dmem_used -= cut;
    }
    g_dmem_allocs.swap(keep);
}

bool ensure_dmem_backing() {
#if defined(_WIN32)
    if (g_dmem_map) {
        return true;
    }
    const DWORD hi = static_cast<DWORD>(g_dmem_size >> 32);
    const DWORD lo = static_cast<DWORD>(g_dmem_size);
    g_dmem_map = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, hi, lo,
                                    nullptr);
    if (!g_dmem_map) {
        host_log("CreateFileMapping dmem 0x%llx failed: %lu",
                 static_cast<unsigned long long>(g_dmem_size), GetLastError());
        return false;
    }
    return true;
#else
    if (g_dmem_fd >= 0) {
        return true;
    }
    const long fd = syscall(SYS_memfd_create, "bb-dmem", 1 | 2 /* MFD_CLOEXEC | MFD_ALLOW_SEALING */);
    if (fd < 0) {
        host_log("memfd_create dmem failed errno=%d", errno);
        return false;
    }
    if (ftruncate(static_cast<int>(fd), static_cast<off_t>(g_dmem_size)) != 0) {
        host_log("ftruncate dmem 0x%llx failed errno=%d",
                 static_cast<unsigned long long>(g_dmem_size), errno);
        close(static_cast<int>(fd));
        return false;
    }
    // /dev/udmabuf, the GPU's way in where the driver refuses the pages as a
    // host pointer (host/gpu.cpp, dmabuf_span), takes only a memfd sealed
    // against shrinking. It never changes size once made.
    if (fcntl(static_cast<int>(fd), F_ADD_SEALS, F_SEAL_SHRINK) != 0) {
        host_log("dmem: sealing the memfd failed errno=%d", errno);
    }
    g_dmem_fd = static_cast<int>(fd);
    return true;
#endif
}

MappedRegion* maps_find_locked(std::uintptr_t a) {
    for (auto& r : g_maps) {
        if (a >= r.start && a < r.start + r.len) {
            return &r;
        }
    }
    return nullptr;
}

int host_prot(int sce_prot) {
#if defined(_WIN32)
    const bool r = sce_prot & kProtCpuRead, w = sce_prot & kProtCpuWrite, x = sce_prot & kProtCpuExec;
    if (x) {
        return w ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    }
    if (w) {
        return PAGE_READWRITE;
    }
    return r ? PAGE_READONLY : PAGE_NOACCESS;
#else
    int prot = 0;
    if (sce_prot & kProtCpuRead) {
        prot |= PROT_READ;
    }
    if (sce_prot & kProtCpuWrite) {
        prot |= PROT_WRITE;
    }
    if (sce_prot & kProtCpuExec) {
        prot |= PROT_EXEC;
    }
    return prot;
#endif
}

enum class Fix { None, Exact, NoReplace };

// Anonymous host memory. `Fix::Exact` replaces whatever is at hint (guest
// asked for MAP_FIXED on a range it owns); `Fix::NoReplace` fails instead of
// clobbering host memory (addresses we computed ourselves).
void* host_map_anon(void* hint, std::uint64_t len, int sce_prot, Fix fix) {
    if (!len) {
        return nullptr;
    }
#if defined(_WIN32)
    // Windows: placeholders and section views (core/win_vm.h). A guest
    // reservation (no protection) is a placeholder; a mapping is a private
    // section so parts of it can be released later. Guest memory stays
    // below the 40-bit Gnm address mask.
    const WinVmPlace place = fix == Fix::Exact ? WinVmPlace::Exact
                             : fix == Fix::NoReplace ? WinVmPlace::NoReplace
                                                     : WinVmPlace::Anywhere;
    const std::uint64_t round = (len + 0xffff) & ~std::uint64_t(0xffff);
    if (sce_prot == 0) return win_vm_place(hint, round, place, kGnmVaMask);
    return win_vm_alloc(hint, round, static_cast<DWORD>(host_prot(sce_prot)), place, kGnmVaMask);
#else
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    if (hint && fix == Fix::Exact) {
        flags |= MAP_FIXED;
    } else if (hint && fix == Fix::NoReplace) {
        flags |= MAP_FIXED_NOREPLACE;
    }
    void* p = mmap(hint, static_cast<std::size_t>(len), host_prot(sce_prot), flags, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    if (hint && fix != Fix::None && p != hint) {
        munmap(p, static_cast<std::size_t>(len));  // old kernels ignore NOREPLACE
        return nullptr;
    }
    return p;
#endif
}

void* host_map_dmem(void* hint, std::uint64_t len, int sce_prot, Fix fix, std::int64_t phys) {
    if (!len || phys < 0 || static_cast<std::uint64_t>(phys) + len > g_dmem_size) {
        return nullptr;
    }
    if (!ensure_dmem_backing()) {
        return nullptr;
    }
#if defined(_WIN32)
    const WinVmPlace place = fix == Fix::Exact ? WinVmPlace::Exact
                             : fix == Fix::NoReplace ? WinVmPlace::NoReplace
                                                     : WinVmPlace::Anywhere;
    const std::uint64_t round = (len + 0xffff) & ~std::uint64_t(0xffff);
    if (static_cast<std::uint64_t>(phys) + round > g_dmem_size) return nullptr;
    return win_vm_map(g_dmem_map, hint, round, static_cast<std::uint64_t>(phys),
                      static_cast<DWORD>(host_prot(sce_prot)), place, kGnmVaMask);
#else
    int flags = MAP_SHARED;
    if (hint && fix == Fix::Exact) {
        flags |= MAP_FIXED;
    } else if (hint && fix == Fix::NoReplace) {
        flags |= MAP_FIXED_NOREPLACE;
    }
    void* p = mmap(hint, static_cast<std::size_t>(len), host_prot(sce_prot), flags, g_dmem_fd,
                   static_cast<off_t>(phys));
    if (p == MAP_FAILED) {
        return nullptr;
    }
    if (hint && fix != Fix::None && p != hint) {
        munmap(p, static_cast<std::size_t>(len));
        return nullptr;
    }
    return p;
#endif
}

void host_unmap(std::uintptr_t p, std::uint64_t len, bool dmem) {
    if (!p || !len) {
        return;
    }
#if defined(_WIN32)
    (void)dmem;
    win_vm_unmap(reinterpret_cast<void*>(p), (len + 0xffff) & ~std::uint64_t(0xffff));
#else
    (void)dmem;
    munmap(reinterpret_cast<void*>(p), static_cast<std::size_t>(len));
#endif
}

bool host_protect(std::uintptr_t p, std::uint64_t len, int sce_prot) {
#if defined(_WIN32)
    DWORD old = 0;
    return VirtualProtect(reinterpret_cast<void*>(p), static_cast<std::size_t>(len),
                          static_cast<DWORD>(host_prot(sce_prot)), &old) != 0;
#else
    return mprotect(reinterpret_cast<void*>(p), static_cast<std::size_t>(len), host_prot(sce_prot)) == 0;
#endif
}

std::uintptr_t map_gpu_alias(std::int64_t phys, std::uint64_t len) {
    if (phys < 0 || static_cast<std::uint64_t>(phys) + len > 0x100000000ull) {
        return 0;
    }
    void* gpu = reinterpret_cast<void*>(kGpuDmemBase + static_cast<std::uint64_t>(phys));
    void* p = host_map_dmem(gpu, len, kProtCpuRead | kProtCpuWrite, Fix::NoReplace, phys);
    if (!p) {
        host_log("GPU dmem alias failed va=%p phys=0x%llx len=0x%llx", gpu,
                 static_cast<unsigned long long>(phys), static_cast<unsigned long long>(len));
        return 0;
    }
    return reinterpret_cast<std::uintptr_t>(p);
}

// Remove [lo, hi) from the mapping records, splitting partial overlaps.
// Returns the pieces actually removed so the caller can unmap them.
std::vector<MappedRegion> maps_cut_locked(std::uintptr_t lo, std::uintptr_t hi) {
    std::vector<MappedRegion> removed;
    std::vector<MappedRegion> keep;
    keep.reserve(g_maps.size() + 2);
    for (const auto& r : g_maps) {
        const std::uintptr_t rlo = r.start;
        const std::uintptr_t rhi = r.start + r.len;
        if (rhi <= lo || rlo >= hi) {
            keep.push_back(r);
            continue;
        }
        const std::uintptr_t clo = std::max(rlo, lo);
        const std::uintptr_t chi = std::min(rhi, hi);
        if (rlo < clo) {
            MappedRegion left = r;
            left.len = clo - rlo;
            keep.push_back(left);
        }
        if (rhi > chi) {
            MappedRegion right = r;
            right.start = chi;
            right.len = rhi - chi;
            if (r.phys >= 0) {
                right.phys = r.phys + static_cast<std::int64_t>(chi - rlo);
            }
            if (r.alias_of) {
                right.alias_of = r.alias_of + (chi - rlo);
            }
            keep.push_back(right);
        }
        MappedRegion cut = r;
        cut.start = clo;
        cut.len = chi - clo;
        if (r.phys >= 0) {
            cut.phys = r.phys + static_cast<std::int64_t>(clo - rlo);
        }
        if (r.alias_of) {
            cut.alias_of = r.alias_of + (clo - rlo);
        }
        removed.push_back(cut);
    }
    g_maps.swap(keep);
    g_maps_gen.fetch_add(1);
    // Whatever the texture cache watched there (core/write_watch.h) is no
    // longer what it uploaded, and the GX resources whose memory it was are
    // dead as far as their memory goes (engine/gx_resources.h).
    write_watch_forget(lo, hi - lo);
    gx_resources_memory_released(lo, hi - lo);
    return removed;
}

int unmap_range(std::uintptr_t addr, std::uint64_t len) {
    if (!addr || !len) {
        return sce_err(EINVAL);
    }
    std::vector<MappedRegion> cut;
    {
        std::lock_guard<std::mutex> lock(g_dmem_mu);
        cut = maps_cut_locked(addr, addr + len);
        // Direct mappings carry a GPU alias at a fixed offset; drop it too.
        std::vector<MappedRegion> alias_cut;
        for (const auto& c : cut) {
            if (c.dmem && !c.alias_of && c.phys >= 0 &&
                static_cast<std::uint64_t>(c.phys) + c.len <= 0x100000000ull) {
                const std::uintptr_t g = static_cast<std::uintptr_t>(kGpuDmemBase + static_cast<std::uint64_t>(c.phys));
                auto more = maps_cut_locked(g, g + c.len);
                alias_cut.insert(alias_cut.end(), more.begin(), more.end());
            }
        }
        cut.insert(cut.end(), alias_cut.begin(), alias_cut.end());
    }
    for (const auto& c : cut) {
        host_unmap(c.start, c.len, c.dmem);
    }
    return 0;
}

int protect_range(std::uintptr_t addr, std::uint64_t len, int prot) {
    if (!addr || !len) {
        return sce_err(EINVAL);
    }
    std::vector<MappedRegion> pieces;
    {
        std::lock_guard<std::mutex> lock(g_dmem_mu);
        pieces = maps_cut_locked(addr, addr + len);
        for (auto& p : pieces) {
            p.prot = prot;
            g_maps.push_back(p);
            g_maps_gen.fetch_add(1);
        }
    }
    if (pieces.empty()) {
        return sce_err(EACCES);
    }
    bool ok = true;
    for (const auto& p : pieces) {
        if (p.alias_of) {
            continue;  // GX writes through the alias regardless of CPU prot
        }
        ok = host_protect(p.start, p.len, prot) && ok;
    }
    // Again, now the protection is the guest's: a watch armed between the
    // records changing and the mprotect above has just been undone by it.
    write_watch_forget(addr, len);
    gx_resources_memory_protected(addr, len, prot);
    return ok ? 0 : sce_err(EACCES);
}

}  // namespace

// Arms a write watch on [va, va+len) (core/write_watch.h), under the map lock
// so no mapping changes between the check and the protection: only direct
// memory mapped CPU read/write (not execute) is watched, every page of it.
bool hle_kernel_write_watch(std::uint64_t va, std::size_t len, WriteWatch& w) {
    w.armed = 0;
    if (!write_watch_enabled() || !len) {
        return false;
    }
    const std::uint64_t lo = va & ~0xfffull, hi = (va + len + 0xfff) & ~0xfffull;
    std::lock_guard<std::mutex> lock(g_dmem_mu);
    for (std::uint64_t at = lo; at < hi;) {
        const MappedRegion* m = nullptr;
        for (const auto& r : g_maps) {
            if (r.dmem && !r.alias_of && at >= r.start && at < r.start + r.len) {
                m = &r;
                break;
            }
        }
        if (!m || (m->prot & (kProtCpuRead | kProtCpuWrite | kProtCpuExec)) != (kProtCpuRead | kProtCpuWrite)) {
            return false;
        }
        at = m->start + m->len;
    }
    return write_watch_arm_locked(va, len, w);
}

// The same for the GPU alias of [va, va+len) (map_gpu_alias): GX writes
// through it, and a watch on the CPU view alone never sees those writes.
// Returns the alias address it armed, or 0 when va has no alias or it could
// not be armed.
std::uint64_t hle_kernel_write_watch_alias(std::uint64_t va, std::size_t len, WriteWatch& w) {
    w.armed = 0;
    if (!write_watch_enabled() || !len) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_dmem_mu);
    for (const auto& r : g_maps) {
        if (r.alias_of && va >= r.alias_of && va + len <= r.alias_of + r.len) {
            const std::uint64_t alias = r.start + (va - r.alias_of);
            return write_watch_arm_locked(alias, len, w) ? alias : 0;
        }
    }
    return 0;
}

std::uint64_t hle_kernel_dmem_high() {
    std::lock_guard<std::mutex> lock(g_dmem_mu);
    return g_dmem_high;
}
std::uint64_t hle_kernel_maps_generation() { return g_maps_gen.load(); }

// Host-owned read/write memory the GPU page table reaches: 64 KiB aligned,
// below 1 TiB (the 40-bit GPU VA space), clear of the dmem ranges. Hints go
// down from 960 GiB a GiB at a time and never replace a mapping. Recorded as
// an anonymous mapping, so the page tables import it on their next rebuild.
std::uint64_t hle_kernel_map_host(std::uint64_t len) {
    const std::uint64_t page = 0x10000;
    len = (len + page - 1) & ~(page - 1);
    for (std::uint64_t hint = 0xF000000000ull; len && hint >= 0xC000000000ull; hint -= 0x40000000ull) {
        const int prot = kProtCpuRead | kProtCpuWrite;
        if (!host_map_anon(reinterpret_cast<void*>(static_cast<std::uintptr_t>(hint)), len, prot, Fix::NoReplace)) continue;
        MappedRegion r{};
        r.start = static_cast<std::uintptr_t>(hint);
        r.len = len;
        r.phys = -1;
        r.prot = prot | kProtGpuRead | kProtGpuWrite;  // for the GPU page table: imported like the game's GPU mappings
        r.dmem = false;
        {
            std::lock_guard<std::mutex> lock(g_dmem_mu);
            maps_cut_locked(r.start, r.start + r.len);
            g_maps.push_back(r);
            g_maps_gen.fetch_add(1);
        }
        host_log("kernel: host mapping 0x%llx len 0x%llx", static_cast<unsigned long long>(hint), static_cast<unsigned long long>(len));
        return hint;
    }
    return 0;
}

// Every mapping of the physical memory behind a direct-memory address, logged
// (address of the same byte, protection, alias or not); returns a CPU address
// of that byte other than `va` and the GPU alias, or 0. For finding writers
// that go through a mapping a write watch on `va` does not cover.
std::uint64_t hle_kernel_mappings_of(std::uint64_t va) {
    std::lock_guard<std::mutex> lock(g_dmem_mu);
    std::int64_t phys = -1;
    for (const auto& r : g_maps) {
        if (r.dmem && r.phys >= 0 && va >= r.start && va < r.start + r.len) {
            phys = r.phys + static_cast<std::int64_t>(va - r.start);
            break;
        }
    }
    if (phys < 0) {
        host_log("kernel: 0x%llx is not in a direct mapping", static_cast<unsigned long long>(va));
        return 0;
    }
    std::uint64_t other = 0;
    for (const auto& r : g_maps) {
        if (!r.dmem || r.phys < 0 || phys < r.phys || phys >= r.phys + static_cast<std::int64_t>(r.len)) continue;
        const std::uint64_t at = r.start + static_cast<std::uint64_t>(phys - r.phys);
        host_log("kernel: phys 0x%llx is mapped at 0x%llx (mapping 0x%llx len 0x%llx, prot 0x%x%s)",
                 static_cast<unsigned long long>(phys), static_cast<unsigned long long>(at),
                 static_cast<unsigned long long>(r.start), static_cast<unsigned long long>(r.len), r.prot,
                 r.alias_of ? ", GPU alias" : "");
        if (!r.alias_of && at != va && !other) other = at;
    }
    return other;
}

// The GPU alias of a CPU address in direct memory, when its mapping has one:
// map_gpu_alias maps a direct mapping a second time at kGpuDmemBase + phys
// only when it ends below 4 GiB, and GX writes through it regardless of CPU
// protection. 0 when the address has no alias.
std::uint64_t hle_kernel_gpu_alias(std::uint64_t va) {
    std::lock_guard<std::mutex> lock(g_dmem_mu);
    for (const auto& r : g_maps) {
        if (r.alias_of && va >= r.alias_of && va < r.alias_of + r.len) return r.start + (va - r.alias_of);
    }
    return 0;
}

void hle_kernel_snapshot_maps(std::vector<GuestMapInfo>& out) {
    std::lock_guard<std::mutex> lock(g_dmem_mu);
    out.clear();
    for (const MappedRegion& r : g_maps) {
        out.push_back({static_cast<std::uint64_t>(r.start), r.len, r.phys, r.dmem, r.prot});
    }
}

int hle_kernel_dmem_fd() {
#if defined(_WIN32)
    return -1;
#else
    return ensure_dmem_backing() ? g_dmem_fd : -1;
#endif
}
std::uint64_t hle_kernel_dmem_size() { return g_dmem_size; }

bool hle_kernel_set_dmem_size(std::uint64_t bytes) {
    bytes = std::clamp<std::uint64_t>((bytes + 0xffffffull) & ~0xffffffull, kDmemBaseSize, kDmemMaxSize);  // whole 16 MiB
    std::lock_guard<std::mutex> lock(g_dmem_mu);
#if defined(_WIN32)
    const bool made = g_dmem_map != nullptr;
#else
    const bool made = g_dmem_fd >= 0;
#endif
    if (made) {
        if (bytes != g_dmem_size) host_log("dmem: already made at %llu MiB; not resized", static_cast<unsigned long long>(g_dmem_size >> 20));
        return bytes == g_dmem_size;
    }
    if (bytes != g_dmem_size)
        host_log("dmem: %llu MiB (was %llu)", static_cast<unsigned long long>(bytes >> 20), static_cast<unsigned long long>(g_dmem_size >> 20));
    g_dmem_size = bytes;
    return true;
}

void* hle_kernel_dmem_mirror() {
    if (!ensure_dmem_backing()) return nullptr;
#if defined(_WIN32)
    void* p = MapViewOfFile(g_dmem_map, FILE_MAP_ALL_ACCESS, 0, 0, static_cast<std::size_t>(g_dmem_size));
    if (!p) host_log("dmem mirror view failed: error %lu", GetLastError());
    return p;
#else
    void* p = mmap(nullptr, g_dmem_size, PROT_READ | PROT_WRITE, MAP_SHARED, g_dmem_fd, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

bool hle_kernel_reserve_guest_windows() {
#if defined(_WIN32)
    // The CPU dmem window and the GPU alias window. (The eboot image is not
    // fixed: elf.cpp takes any slide, memory.cpp picks one below 4 GiB.)
    bool ok = win_vm_reserve(kDmemVaBase, kDmemMaxSize);
    ok = win_vm_reserve(kGpuDmemBase, 0x100000000ull) && ok;
    // A second home for the eboot image when 0x400000 is taken (it is under
    // wine): fixed, so a run's guest addresses are the same every time.
    win_vm_reserve(kImageFallbackBase, 0x10000000ull);
    // BBHOST_TEST_RESERVE_FAIL=1: the first process pretends the reservation
    // lost, to see the relaunch below work.
    char relaunch_env[16] = {};
    const DWORD relaunches = GetEnvironmentVariableA("BBHOST_RELAUNCH", relaunch_env, sizeof(relaunch_env))
                                 ? static_cast<DWORD>(std::atoi(relaunch_env))
                                 : 0;
    if (ok && relaunches == 0 && GetEnvironmentVariableA("BBHOST_TEST_RESERVE_FAIL", relaunch_env, sizeof(relaunch_env)) &&
        relaunch_env[0] == '1') {
        host_log("win-vm: BBHOST_TEST_RESERVE_FAIL: pretending the reservation lost");
        ok = false;
    }
    host_log("win-vm: guest windows [0x%llx, +0x%llx) and [0x%llx, +4 GiB) %s",
             static_cast<unsigned long long>(kDmemVaBase), static_cast<unsigned long long>(kDmemMaxSize),
             static_cast<unsigned long long>(kGpuDmemBase), ok ? "reserved" : "NOT reserved");
    if (!ok) {
        // The windows are fixed addresses, and what took one of them (the
        // heap, a DLL, the stack: bottom-up ASLR places them before main
        // runs) cannot be moved. A fresh process is placed anew, so the run
        // restarts itself, up to three times, with the same command line and
        // handles; its exit code becomes ours.
        if (relaunches >= 3) {
            host_log("win-vm: the guest windows were taken in %lu launches; giving up", static_cast<unsigned long>(relaunches + 1));
            return false;
        }
        std::snprintf(relaunch_env, sizeof(relaunch_env), "%lu", static_cast<unsigned long>(relaunches + 1));
        SetEnvironmentVariableA("BBHOST_RELAUNCH", relaunch_env);
        host_log("win-vm: relaunching (%s of 3) so the guest windows are placed anew", relaunch_env);
        std::fflush(nullptr);
        STARTUPINFOW si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        wchar_t* cmd = _wcsdup(GetCommandLineW());
        if (!CreateProcessW(nullptr, cmd, nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
            host_log("win-vm: relaunch failed: error %lu", GetLastError());
            return false;
        }
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD code = 1;
        GetExitCodeProcess(pi.hProcess, &code);
        _exit(static_cast<int>(code));
    }
    return ok;
#else
    return true;
#endif
}

namespace {

// The mapping list as one thread last saw it: sorted, disjoint [start, end)
// ranges, refreshed when g_maps_gen moves (map and unmap only), and the last
// few ranges lookups landed in. A draw's lookups fall in a handful of heap
// mappings (state, shaders, views, buffers), and the binary search each one
// did was ~5% of the GX workers' time.
struct MapSnapshot {
    static constexpr int kHits = 4;
    std::uint64_t gen = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::uint64_t hit_lo[kHits] = {}, hit_hi[kHits] = {};
    unsigned next_hit = 0;
};

// readable_only: only CPU-readable mappings, and touching ranges merge (a read
// may cross from one into the next); otherwise every mapping, and only
// overlapping ranges merge, as the scan under the lock had them.
// `lo`, `hi` (optional): the mapped range that holds [va, va+len), when it is.
bool va_in_maps(MapSnapshot& snap, bool readable_only, std::uint64_t va, std::size_t len, std::uint64_t* lo = nullptr,
                std::uint64_t* hi = nullptr) {
    if (len == 0) {
        return false;
    }
    const std::uint64_t end = va + len;
    if (end < va) {
        return false;
    }
    const std::uint64_t gen = g_maps_gen.load(std::memory_order_acquire);
    if (snap.gen == gen) {
        for (int i = 0; i < MapSnapshot::kHits; ++i) {
            if (va >= snap.hit_lo[i] && end <= snap.hit_hi[i]) {
                if (lo) *lo = snap.hit_lo[i];
                if (hi) *hi = snap.hit_hi[i];
                return true;
            }
        }
    } else {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> v;
        {
            std::lock_guard<std::mutex> lock(g_dmem_mu);
            v.reserve(g_maps.size());
            for (const auto& r : g_maps) {
                if (!readable_only || (r.prot & kProtCpuRead)) {
                    v.push_back({static_cast<std::uint64_t>(r.start), static_cast<std::uint64_t>(r.start) + r.len});
                }
            }
        }
        std::sort(v.begin(), v.end());
        snap.ranges.clear();
        for (const auto& r : v) {
            if (!snap.ranges.empty() &&
                (readable_only ? r.first <= snap.ranges.back().second : r.first < snap.ranges.back().second)) {
                if (r.second > snap.ranges.back().second) snap.ranges.back().second = r.second;
            } else {
                snap.ranges.push_back(r);
            }
        }
        snap.gen = gen;
        for (int i = 0; i < MapSnapshot::kHits; ++i) {
            snap.hit_lo[i] = snap.hit_hi[i] = 0;
        }
    }
    auto it = std::upper_bound(snap.ranges.begin(), snap.ranges.end(), va,
                               [](std::uint64_t v, const std::pair<std::uint64_t, std::uint64_t>& r) { return v < r.first; });
    if (it == snap.ranges.begin()) {
        return false;
    }
    --it;
    if (end > it->second) {
        return false;
    }
    const unsigned slot = snap.next_hit++ % MapSnapshot::kHits;
    snap.hit_lo[slot] = it->first;
    snap.hit_hi[slot] = it->second;
    if (lo) *lo = it->first;
    if (hi) *hi = it->second;
    return true;
}

thread_local MapSnapshot t_mapped_snap;

}  // namespace

// Called millions of times a frame - every resource resolve, every shader
// fetch, every command packet - so the mapping list is not scanned under the
// dmem lock any more: each thread keeps its own snapshot (va_in_maps).
bool hle_kernel_va_mapped(std::uint64_t va, std::size_t len) { return va_in_maps(t_mapped_snap, false, va, len); }

// The same, and the mapped range [*lo, *hi) that holds it: a caller checking
// a burst of pointers into a few heaps tests the next one against that range
// first (the GX threads' draw snapshots, which the game's job barrier waits for).
bool hle_kernel_va_mapped_in(std::uint64_t va, std::size_t len, std::uint64_t* lo, std::uint64_t* hi) {
    return va_in_maps(t_mapped_snap, false, va, len, lo, hi);
}

// Like hle_kernel_va_mapped, but only CPU-readable mappings count: memory a
// host read can touch without faulting (reserved ranges are mapped but not
// readable). Memory the loader mapped outside the map list is not covered.
static thread_local MapSnapshot t_readable_snap;  // this file's own, like t_mapped_snap
bool hle_kernel_va_readable(std::uint64_t va, std::size_t len) { return va_in_maps(t_readable_snap, true, va, len); }

// The same, and the readable range [*lo, *hi) that holds it (see hle_kernel_va_mapped_in).
bool hle_kernel_va_readable_in(std::uint64_t va, std::size_t len, std::uint64_t* lo, std::uint64_t* hi) {
    return va_in_maps(t_readable_snap, true, va, len, lo, hi);
}

// Printed at exit / on SIGTERM. Reads racy values on purpose.
// Every eight-byte-aligned word of mapped guest memory that holds one of
// `values`, for finding who still points at an object the game has lost
// track of. Slow (it walks everything mapped), so it is for a diagnostic
// that runs once: the hang watchdog's.
int hle_kernel_find_pointers(const std::uint64_t* values, int n, int cap, int* counts, std::uint64_t* first_outside) {
    if (!values || n <= 0) return 0;
    std::vector<MappedRegion> maps;
    {
        std::lock_guard<std::mutex> lk(g_dmem_mu);
        maps = g_maps;
    }
    const auto start = std::chrono::steady_clock::now();
    struct Hit {
        std::uint64_t at;
        int which;
    };
    std::vector<Hit> hits;
    std::uint64_t scanned = 0;
    for (const MappedRegion& r : maps) {
        if (r.alias_of || !r.start || r.len < 8 || !(r.prot & 1)) continue;  // the GPU alias is the same memory
        const auto* p = reinterpret_cast<const std::uint64_t*>(r.start);
        const std::uint64_t words = r.len / 8;
        scanned += r.len;
        for (std::uint64_t i = 0; i < words; ++i) {
            const std::uint64_t v = p[i];
            for (int k = 0; k < n; ++k) {
                if (v == values[k]) {
                    hits.push_back({r.start + i * 8, k});
                    if (counts) ++counts[k];
                    break;
                }
            }
        }
    }
    // Which hits are bookkeeping rather than a holder: a pointer beside the
    // entry itself, and one in a list of the entries (another of them eight
    // bytes away). What is left is something that still holds the entry.
    const auto listed = [&](const Hit& h) {
        for (const Hit& o : hits) {
            if (o.which != h.which && (o.at == h.at + 8 || o.at + 8 == h.at)) return true;
        }
        return false;
    };
    int shown = 0;
    for (const Hit& h : hits) {
        const bool beside = h.at > values[h.which] - 0x4000 && h.at < values[h.which] + 0x4000;
        const bool in_list = listed(h);
        if (!beside && !in_list && first_outside && !first_outside[h.which]) first_outside[h.which] = h.at;
        if (shown++ < cap) {
            host_log("hang:   entry %d (0x%llx) is pointed at from 0x%llx%s", h.which,
                     static_cast<unsigned long long>(values[h.which]), static_cast<unsigned long long>(h.at),
                     beside ? " (beside itself)" : in_list ? " (in a list of them)" : " (a holder)");
        }
    }
    host_log("hang: %zu pointer(s) into the pool's entries in %.1f MiB of guest memory (%.1f s)", hits.size(),
             static_cast<double>(scanned) / (1 << 20),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return static_cast<int>(hits.size());
}

void hle_kernel_memory_report() {
    std::uint64_t pk = 0, wr = 0;
    hle_gnm_exec_stats(&pk, &wr);
    hle_gnm_exec_histogram();
    hle_gnm_wait_report();
    host_log("dmem: used=%llu MiB high=%llu MiB allocs=%zu maps=%zu; gnm packets=%llu writes=%llu",
             static_cast<unsigned long long>(g_dmem_used >> 20),
             static_cast<unsigned long long>(g_dmem_high >> 20), g_dmem_allocs.size(), g_maps.size(),
             static_cast<unsigned long long>(pk), static_cast<unsigned long long>(wr));
}

namespace {

GUEST_ABI std::uint64_t hle_sce_direct_mem_size() { return g_dmem_size; }

GUEST_ABI int hle_alloc_direct(std::int64_t search_start, std::int64_t search_end, std::uint64_t len,
                               std::uint64_t align, int type, std::int64_t* phys_out) {
    if (!phys_out || len == 0 || search_start < 0 || search_end < 0) {
        return sce_err(EINVAL);
    }
    std::uint64_t p = 0;
    int e;
    {
        std::lock_guard<std::mutex> lock(g_dmem_mu);
        e = dmem_alloc_locked(static_cast<std::uint64_t>(search_start), static_cast<std::uint64_t>(search_end),
                              len, align, type, &p);
    }
    if (e) {
        host_log("AllocateDirectMemory failed len=0x%llx align=0x%llx e=%d",
                 static_cast<unsigned long long>(len), static_cast<unsigned long long>(align), e);
        return sce_err(e);
    }
    *phys_out = static_cast<std::int64_t>(p);
    host_log("AllocateDirectMemory phys=0x%llx len=0x%llx type=%d",
             static_cast<unsigned long long>(p), static_cast<unsigned long long>(len), type);
    return 0;
}

int map_va(void** addr, std::uint64_t len, int prot, int flags, std::int64_t phys, std::uint64_t align,
           bool dmem) {
    if (!addr || len == 0 || (len & 0x3fff) != 0) {
        return sce_err(EINVAL);
    }
    if (align == 0) {
        align = kPage;
    }
    if (!pow2(align)) {
        return sce_err(EINVAL);
    }
    void* hint = *addr;
    Fix fix = Fix::None;
    if ((flags & 0x10) != 0 && hint != nullptr) {
        fix = Fix::Exact;  // SCE_KERNEL_MAP_FIXED on a range the guest owns
    } else if (dmem && phys >= 0) {
        const std::uint64_t cpu_va = kDmemVaBase + static_cast<std::uint64_t>(phys);
        if (cpu_va + len - 1 <= kGnmVaMask) {
            hint = reinterpret_cast<void*>(cpu_va);
            fix = Fix::NoReplace;
        }
    }
    void* p = nullptr;
    if (dmem) {
        p = host_map_dmem(hint, len, prot, fix, phys);
    } else if (fix == Fix::None && align > 0x1000) {
        // Over-allocate and trim to honour the alignment.
        void* raw = host_map_anon(nullptr, len + align, prot, Fix::None);
        if (raw) {
            const auto r = reinterpret_cast<std::uintptr_t>(raw);
            const std::uintptr_t a = align_up(r, align);
#if !defined(_WIN32)
            if (a > r) {
                munmap(raw, a - r);
            }
            if (r + len + align > a + len) {
                munmap(reinterpret_cast<void*>(a + len), (r + len + align) - (a + len));
            }
#endif
            p = reinterpret_cast<void*>(a);
        }
    } else {
        p = host_map_anon(hint, len, prot, fix);
    }
    if (!p) {
        host_log("map failed dmem=%d va=%p len=0x%llx phys=0x%llx flags=0x%x fix=%d", static_cast<int>(dmem),
                 hint, static_cast<unsigned long long>(len), static_cast<unsigned long long>(phys), flags,
                 static_cast<int>(fix));
        return sce_err(ENOMEM);
    }
    std::uintptr_t gpu = 0;
    int type = 0;
    if (dmem) {
        gpu = map_gpu_alias(phys, len);
    }
    MappedRegion r{};
    r.start = reinterpret_cast<std::uintptr_t>(p);
    r.len = len;
    r.phys = dmem ? phys : -1;
    r.prot = prot;
    r.dmem = dmem;
    {
        std::lock_guard<std::mutex> lock(g_dmem_mu);
        if (dmem) {
            type = dmem_type_at_locked(phys);
        }
        r.type = type;
        // Replace any stale records under an Exact mapping.
        maps_cut_locked(r.start, r.start + r.len);
        g_maps.push_back(r);
        g_maps_gen.fetch_add(1);
        if (gpu) {
            MappedRegion g = r;
            g.start = gpu;
            g.alias_of = r.start;
            g.prot = kProtCpuRead | kProtCpuWrite;
            maps_cut_locked(gpu, gpu + len);
            g_maps.push_back(g);
            g_maps_gen.fetch_add(1);
        }
    }
    *addr = p;
    if (dmem) {
        host_log("MapDirectMemory va=%p gpu=0x%llx len=0x%llx phys=0x%llx prot=0x%x flags=0x%x align=0x%llx",
                 p, static_cast<unsigned long long>(gpu), static_cast<unsigned long long>(len),
                 static_cast<unsigned long long>(phys), prot, flags, static_cast<unsigned long long>(align));
    } else {
        host_log("MapMemory va=%p len=0x%llx prot=0x%x flags=0x%x", p,
                 static_cast<unsigned long long>(len), prot, flags);
    }
    return 0;
}

GUEST_ABI int hle_map_direct(void** addr, std::uint64_t len, int prot, int flags, std::int64_t start,
                             std::uint64_t align) {
    return map_va(addr, len, prot, flags, start, align, true);
}

GUEST_ABI int hle_map_flex(void** addr, std::uint64_t len, int prot, int flags) {
    return map_va(addr, len, prot, flags, -1, kPage, false);
}

GUEST_ABI int hle_reserve_va(void** addr, std::uint64_t len, int flags, std::uint64_t align) {
    // Reserved ranges are inaccessible until mapped over with MAP_FIXED.
    return map_va(addr, len, 0, flags, -1, align, false);
}

GUEST_ABI int hle_munmap(void* addr, std::uint64_t len) {
    return unmap_range(reinterpret_cast<std::uintptr_t>(addr), len);
}

GUEST_ABI int hle_release_direct(std::int64_t phys, std::uint64_t len) {
    if (phys < 0 || len == 0 || static_cast<std::uint64_t>(phys) + len > g_dmem_size) {
        return sce_err(EINVAL);
    }
    // Any CPU mapping of the released range goes away with it.
    std::vector<MappedRegion> victims;
    {
        std::lock_guard<std::mutex> lock(g_dmem_mu);
        for (const auto& r : g_maps) {
            if (r.dmem && !r.alias_of && r.phys >= 0) {
                const std::uint64_t rlo = static_cast<std::uint64_t>(r.phys);
                const std::uint64_t rhi = rlo + r.len;
                const std::uint64_t lo = static_cast<std::uint64_t>(phys);
                const std::uint64_t hi = lo + len;
                if (rhi > lo && rlo < hi) {
                    MappedRegion v = r;
                    const std::uint64_t clo = std::max(rlo, lo);
                    const std::uint64_t chi = std::min(rhi, hi);
                    v.start = r.start + (clo - rlo);
                    v.len = chi - clo;
                    victims.push_back(v);
                }
            }
        }
        dmem_release_locked(static_cast<std::uint64_t>(phys), len);
    }
    for (const auto& v : victims) {
        unmap_range(v.start, v.len);
    }
    host_log("ReleaseDirectMemory phys=0x%llx len=0x%llx", static_cast<unsigned long long>(phys),
             static_cast<unsigned long long>(len));
    return 0;
}

GUEST_ABI int hle_release_flex(void* addr, std::uint64_t len) {
    return unmap_range(reinterpret_cast<std::uintptr_t>(addr), len);
}

GUEST_ABI int hle_mprotect(void* addr, std::uint64_t len, int prot) {
    return protect_range(reinterpret_cast<std::uintptr_t>(addr), len, prot);
}

// SceKernelBatchMapEntry: start, offset, length, protection, type, pad, operation.
struct BatchEntry {
    void* start;
    std::int64_t offset;
    std::uint64_t length;
    std::uint8_t protection;
    std::uint8_t type;
    std::int16_t pad;
    std::int32_t operation;
};
static_assert(sizeof(BatchEntry) == 32, "SceKernelBatchMapEntry is 32 bytes");

GUEST_ABI int hle_batch_map(BatchEntry* entries, int num, int* processed) {
    if (!entries || num <= 0) {
        return sce_err(EINVAL);
    }
    int done = 0;
    int err = 0;
    for (int i = 0; i < num && !err; ++i) {
        BatchEntry& e = entries[i];
        void* va = e.start;
        switch (e.operation) {
            case 0:  // MAP_DIRECT
                err = map_va(&va, e.length, e.protection, 0x10, e.offset, kPage, true);
                break;
            case 1:  // UNMAP
                err = unmap_range(reinterpret_cast<std::uintptr_t>(e.start), e.length);
                break;
            case 2:  // PROTECT
            case 4:  // TYPE_PROTECT
                err = protect_range(reinterpret_cast<std::uintptr_t>(e.start), e.length, e.protection);
                break;
            case 3:  // MAP_FLEXIBLE
                err = map_va(&va, e.length, e.protection, 0x10, -1, kPage, false);
                break;
            default:
                err = sce_err(EINVAL);
                break;
        }
        if (!err) {
            ++done;
        }
    }
    if (processed) {
        *processed = done;
    }
    host_log("BatchMap %d entries, %d done, err=0x%x", num, done, static_cast<unsigned>(err));
    return err;
}

// The lock and the sleep are core/futex.h's (WaitOnAddress on Windows, not
// winpthreads' semaphores and critical sections; BBHOST_FUTEX=0 for those).
// A signal wakes as many waiters as it gives tokens when every waiter wants
// one (it woke them all, and all but one went back to sleep);
// when a waiter wants more than one, all wake and each checks its own need,
// since waking the wrong one would leave a satisfiable waiter asleep.
struct HostSema {
    HostLock mu;
    HostCondVar cv;
    int count = 0;
    int max = 0x7fffffff;
    int waiters = 0;
    int multi_waiters = 0;  // waiters that want more than one token
    bool dead = false;
};
std::atomic<std::uint64_t> g_sema_sleeps{0}, g_sema_wakes{0}, g_eq_sleeps{0};

GUEST_ABI int hle_create_sema(HostSema** out, const char* name, unsigned attr, int init, int max, void*) {
    if (!out || init < 0 || max <= 0 || init > max) {
        return sce_err(EINVAL);
    }
    auto* h = new HostSema();
    h->count = init;
    h->max = max;
    *out = h;
    static std::atomic<int> sema_logs{0};
    if (sema_logs.fetch_add(1) < 12) {
        host_log("CreateSema %s init=%d max=%d attr=%u -> %p", name ? name : "", init, max, attr,
                 static_cast<void*>(h));
    }
    return 0;
}

// Waiters own the object once it is marked dead; the last one out frees it.
GUEST_ABI int hle_delete_sema(HostSema* h) {
    if (!h) {
        return sce_err(EINVAL);
    }
    bool free_now = false;
    {
        std::lock_guard<HostLock> lk(h->mu);
        h->dead = true;
        free_now = h->waiters == 0;
        h->cv.notify_all();
    }
    if (free_now) {
        delete h;
    }
    return 0;
}

// timeout: SceKernelUseconds* (relative). nullptr blocks; 0 polls. On return
// it holds the remaining time.
GUEST_ABI int hle_wait_sema(HostSema* h, int need, std::uint32_t* timeout) {
    MainThreadWait timed(0);
    if (!h || need <= 0) {
        return sce_err(EINVAL);
    }
    std::unique_lock<HostLock> lk(h->mu);
    if (h->dead) {
        return sce_err(EINVAL);
    }
    if (h->count >= need) {
        h->count -= need;
        return 0;
    }
    if (timeout && *timeout == 0) {
        return sce_err(EBUSY);
    }
    ++h->waiters;
    if (need > 1) ++h->multi_waiters;
    g_sema_sleeps.fetch_add(1, std::memory_order_relaxed);
    bool ok = true;
    const auto start = std::chrono::steady_clock::now();
    auto ready = [&] { return h->dead || h->count >= need; };
    if (timeout) {
        ok = h->cv.wait_for(lk, std::chrono::microseconds(*timeout), ready);
    } else {
        h->cv.wait(lk, ready);
    }
    --h->waiters;
    if (need > 1) --h->multi_waiters;
    if (h->dead) {
        const bool last = h->waiters == 0;
        lk.unlock();
        if (last) {
            delete h;
        }
        return sce_err(EINVAL);
    }
    if (timeout) {
        const auto spent = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
        *timeout = spent >= *timeout ? 0u : *timeout - static_cast<std::uint32_t>(spent);
    }
    if (!ok) {
        return sce_err(ETIMEDOUT);
    }
    h->count -= need;
    return 0;
}

GUEST_ABI int hle_signal_sema(HostSema* h, int count) {
    if (!h || count <= 0) {
        return sce_err(EINVAL);
    }
    std::lock_guard<HostLock> lk(h->mu);
    if (h->dead) {
        return sce_err(EINVAL);
    }
    if (h->count + count > h->max) {
        return sce_err(EINVAL);  // SCE_KERNEL_ERROR_EINVAL: over max
    }
    h->count += count;
    if (h->waiters) {
        // Every logged CreateSema is binary (init 0, max 1) with one waiter
        // at most, so this is mostly one wake for one token either way.
        static const bool wake_all = [] {
            const char* e = std::getenv("BBHOST_SEMA_WAKE_ALL");
            return e && e[0] == '1';
        }();
        if (wake_all || h->multi_waiters || count >= h->waiters) {
            h->cv.notify_all();
            g_sema_wakes.fetch_add(static_cast<std::uint64_t>(h->waiters), std::memory_order_relaxed);
        } else {
            for (int i = 0; i < count; ++i) h->cv.notify_one();
            g_sema_wakes.fetch_add(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
        }
    }
    return 0;
}

GUEST_ABI int hle_poll_sema(HostSema* h, int need) {
    std::uint32_t zero = 0;
    return hle_wait_sema(h, need, &zero);
}

GUEST_ABI int hle_cancel_sema(HostSema* h, int set_count, int* num_waiters) {
    if (!h) {
        return sce_err(EINVAL);
    }
    std::lock_guard<HostLock> lk(h->mu);
    if (num_waiters) {
        *num_waiters = h->waiters;
    }
    if (set_count >= 0 && set_count <= h->max) {
        h->count = set_count;
    }
    h->cv.notify_all();
    return 0;
}

GUEST_ABI int hle_clock_gettime(int clock_id, void* ts) {
    if (!ts) {
        return sce_err(EINVAL);
    }
    g_hle_clock_reads.clock_gettime.add();
    // FreeBSD's ids: the monotonic and uptime family from the host's
    // monotonic count, the CPU-time ones from the system (winpthreads asks
    // GetThreadTimes/GetProcessTimes), everything else wall-clock time.
    std::int64_t ns = 0;
    switch (clock_id) {
        case 4:
        case 5:
        case 7:
        case 8:
        case 11:
        case 12:
            ns = static_cast<std::int64_t>(host_clock_monotonic_ns());
            break;
        case 14:
        case 2:
        case 15: {
            timespec host{};
            if (clock_gettime(clock_id == 14 ? CLOCK_THREAD_CPUTIME_ID : CLOCK_PROCESS_CPUTIME_ID, &host) != 0) {
                clock_gettime(CLOCK_REALTIME, &host);
            }
            ns = static_cast<std::int64_t>(host.tv_sec) * 1000000000ll + static_cast<std::int64_t>(host.tv_nsec);
            break;
        }
        default:
            ns = host_clock_realtime_ns();
            break;
    }
    auto* out = static_cast<std::int64_t*>(ts);
    out[0] = ns / 1000000000ll;
    out[1] = ns % 1000000000ll;
    return 0;
}

GUEST_ABI int hle_query_prot(void* addr, void** start, void** end, int* prot) {
    std::lock_guard<std::mutex> lock(g_dmem_mu);
    MappedRegion* r = maps_find_locked(reinterpret_cast<std::uintptr_t>(addr));
    if (!r) {
        return sce_err(EACCES);
    }
    if (start) {
        *start = reinterpret_cast<void*>(r->start);
    }
    if (end) {
        *end = reinterpret_cast<void*>(r->start + r->len);
    }
    if (prot) {
        *prot = r->prot;
    }
    return 0;
}

// SceKernelVirtualQueryInfo: start, end, offset, protection, memoryType,
// flags (isFlexible:1 isDirect:1 isStack:1 isPooled:1 isCommitted:1), name[32].
GUEST_ABI int hle_virtual_query(void* addr, int flags, void* info, std::uint64_t infosz) {
    if (!info || infosz < 0x48) {
        return sce_err(EINVAL);
    }
    std::memset(info, 0, static_cast<std::size_t>(infosz));
    auto* out = static_cast<std::uint64_t*>(info);
    std::lock_guard<std::mutex> lock(g_dmem_mu);
    MappedRegion* r = maps_find_locked(reinterpret_cast<std::uintptr_t>(addr));
    if (!r && (flags & 1)) {
        // SCE_KERNEL_VQ_FIND_NEXT: first mapping above addr.
        std::uintptr_t best = ~static_cast<std::uintptr_t>(0);
        for (auto& m : g_maps) {
            if (m.start >= reinterpret_cast<std::uintptr_t>(addr) && m.start < best) {
                best = m.start;
                r = &m;
            }
        }
    }
    if (!r) {
        return sce_err(EACCES);
    }
    out[0] = r->start;
    out[1] = r->start + r->len;
    out[2] = r->phys >= 0 ? static_cast<std::uint64_t>(r->phys) : 0;
    auto* i32 = reinterpret_cast<std::int32_t*>(out + 3);
    i32[0] = r->prot;
    i32[1] = r->type;
    i32[2] = (r->dmem ? 0x2 : 0x1) | 0x10;
    return 0;
}

GUEST_ABI int hle_direct_mem_type(std::int64_t start, int* type, std::int64_t* region_start,
                                 std::int64_t* region_end) {
    std::lock_guard<std::mutex> lock(g_dmem_mu);
    for (const auto& a : g_dmem_allocs) {
        if (start >= 0 && static_cast<std::uint64_t>(start) >= a.phys &&
            static_cast<std::uint64_t>(start) < a.phys + a.len) {
            if (type) {
                *type = a.type;
            }
            if (region_start) {
                *region_start = static_cast<std::int64_t>(a.phys);
            }
            if (region_end) {
                *region_end = static_cast<std::int64_t>(a.phys + a.len);
            }
            return 0;
        }
    }
    return sce_err(ENOENT);
}

std::mutex g_eq_mu;
std::unordered_set<HostEqueue*> g_eqs;
GUEST_ABI int hle_create_equeue(HostEqueue** out, const char* name) {
    if (!out) {
        return sce_err(EINVAL);
    }
    auto* eq = new HostEqueue();
    if (name) {
        eq->name = name;
    }
    {
        std::lock_guard<std::mutex> lock(g_eq_mu);
        g_eqs.insert(eq);
    }
    *out = eq;
    host_log("CreateEqueue %s -> %p", name ? name : "", static_cast<void*>(eq));
    return 0;
}

GUEST_ABI int hle_delete_equeue(HostEqueue* eq) {
    if (!eq || !equeue_live(eq)) {
        return sce_err(EBADF);
    }
    hle_video_detach_equeue(eq);
    hle_gnm_detach_equeue(eq);
    {
        std::lock_guard<std::mutex> lock(g_eq_mu);
        g_eqs.erase(eq);
    }
    bool free_now = false;
    {
        std::lock_guard<HostLock> lock(eq->mu);
        eq->dead = true;
        free_now = eq->waiters == 0;
        eq->cv.notify_all();
    }
    if (free_now) {
        delete eq;
    }
    return 0;
}

GUEST_ABI int hle_wait_equeue(HostEqueue* eq, HostEvent* ev, int num, int* out, std::uint32_t* timeout) {
    MainThreadWait timed(0);
    if (!eq || !equeue_live(eq) || num <= 0) {
        return sce_err(EINVAL);
    }
    std::unique_lock<HostLock> lk(eq->mu);
    auto ready = [&] { return eq->dead || !eq->q.empty(); };
    bool ok = true;
    ++eq->waiters;
    if (!ready() && !(timeout && *timeout == 0)) g_eq_sleeps.fetch_add(1, std::memory_order_relaxed);
    if (timeout == nullptr) {
        eq->cv.wait(lk, ready);
    } else if (*timeout == 0) {
        ok = !eq->q.empty();
    } else {
        ok = eq->cv.wait_for(lk, std::chrono::microseconds(*timeout), ready);
    }
    --eq->waiters;
    if (eq->dead) {
        const bool last = eq->waiters == 0;
        lk.unlock();
        if (last) {
            delete eq;
        }
        return sce_err(EBADF);
    }
    if (!ok) {
        return sce_err(ETIMEDOUT);
    }
    int n = 0;
    while (n < num && !eq->q.empty()) {
        if (ev) {
            ev[n] = eq->q.front();
        }
        eq->q.pop_front();
        ++n;
    }
    if (out) {
        *out = n;
    }
    return n > 0 ? 0 : sce_err(ETIMEDOUT);
}
GUEST_ABI std::uint64_t hle_kernel_time() {
    g_hle_clock_reads.process_time.add();
    return host_clock_monotonic_ns() / 1000ull;
}

GUEST_ABI std::uint64_t hle_read_tsc() {
    g_hle_clock_reads.tsc.add();
    return rdtsc_now();
}
GUEST_ABI std::uint64_t hle_tsc_frequency() { return kGuestTscHz; }

}  // namespace

HleClockReads g_hle_clock_reads;

// The clock entries need no thread-local state, so the guest calls them
// without the FS- and stack-switching thunk, as it calls memcpy (hle/libc.cpp
// REG_RAW) - only in GS mode (Windows always), where the host's FS (errno,
// the C library's thread data, which the CPU-time clocks may touch) stays in
// place while the guest runs. BBHOST_RAW_CLOCK=0 thunks them again.
bool hle_raw_clock() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_RAW_CLOCK");
        return tls_gs_mode() && !(e && e[0] == '0');
    }();
    return on;
}

std::string hle_timing_window(double secs, std::string* sync) {
    // Totals since the start; the lines give rates over this window. Called
    // from the 300-flip report only (one thread).
    struct Snap {
        std::uint64_t v[16];
    };
    static Snap last{};
    const gsync::SyncCounts s = gsync::counts();
    const HostFutexCounts f = host_futex_counts();
    const Snap now{{g_hle_clock_reads.gettimeofday.total(), g_hle_clock_reads.clock_gettime.total(),
                    g_hle_clock_reads.process_time.total(), g_hle_clock_reads.tsc.total(), g_hle_clock_reads.time.total(),
                    g_hle_clock_reads.clock.total(), s.mutex_contended, s.cond_waits, s.cond_signals, s.rw_sleeps,
                    g_sema_sleeps.load(), g_sema_wakes.load(), g_eq_sleeps.load(), f.sleeps, f.timeouts, f.wakes}};
    const double per = secs > 0.0 ? 1.0 / secs : 0.0;
    const auto rate = [&](int i) { return static_cast<double>(now.v[i] - last.v[i]) * per; };
    char buf[320];
    std::snprintf(buf, sizeof(buf), "clock reads/s gettimeofday=%.0f clock_gettime=%.0f process-time=%.0f tsc=%.0f time=%.0f clock=%.0f (%s)",
                  rate(0), rate(1), rate(2), rate(3), rate(4), rate(5), hle_raw_clock() ? "raw" : "thunked");
    if (sync) {
        char sb[400];
        std::snprintf(sb, sizeof(sb),
                      "waits/s mutex=%.0f cond=%.0f (signals %.0f) rwlock=%.0f sema=%.0f (woken %.0f) equeue=%.0f; "
                      "sleeps/s %.0f (timed out %.0f), wakes/s %.0f, on %s",
                      rate(6), rate(7), rate(8), rate(9), rate(10), rate(11), rate(12), rate(13), rate(14), rate(15),
                      host_futex_native() ? "the word sleep" : "parked condition variables (BBHOST_FUTEX=0)");
        *sync = sb;
    }
    last = now;
    return buf;
}

bool equeue_live(HostEqueue* eq) {
    std::lock_guard<std::mutex> lock(g_eq_mu);
    return g_eqs.count(eq) != 0;
}

void equeue_post(HostEqueue* eq, const HostEvent& ev) {
    if (!eq) {
        return;
    }
    std::lock_guard<HostLock> lock(eq->mu);
    if (!eq->dead) {
        eq->q.push_back(ev);
        eq->cv.notify_all();
    }
}

void hle_register_kernel() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
    REG("sceKernelGetDirectMemorySize", hle_sce_direct_mem_size);
    REG("sceKernelAllocateDirectMemory", hle_alloc_direct);
    REG("sceKernelMapDirectMemory", hle_map_direct);
    REG("sceKernelMapFlexibleMemory", hle_map_flex);
    REG("sceKernelReserveVirtualRange", hle_reserve_va);
    REG("sceKernelMunmap", hle_munmap);
    REG("sceKernelReleaseDirectMemory", hle_release_direct);
    REG("sceKernelReleaseFlexibleMemory", hle_release_flex);
    REG("sceKernelMprotect", hle_mprotect);
    REG("sceKernelBatchMap", hle_batch_map);
    REG("sceKernelCreateSema", hle_create_sema);
    REG("sceKernelDeleteSema", hle_delete_sema);
    REG("sceKernelWaitSema", hle_wait_sema);
    REG("sceKernelSignalSema", hle_signal_sema);
    REG("sceKernelPollSema", hle_poll_sema);
    REG("sceKernelCancelSema", hle_cancel_sema);
    REG("sceKernelCreateEqueue", hle_create_equeue);
    REG("sceKernelDeleteEqueue", hle_delete_equeue);
    REG("sceKernelWaitEqueue", hle_wait_equeue);
#define REG_CLOCK(name, fn) \
    (hle_raw_clock() ? register_hle_fn_raw(name, reinterpret_cast<void*>(fn)) : register_hle_fn(name, reinterpret_cast<void*>(fn)))
    rdtsc_now();  // the clock statics set up here, not in a raw entry's first call
    REG_CLOCK("sceKernelClockGettime", hle_clock_gettime);
    REG("sceKernelQueryMemoryProtection", hle_query_prot);
    REG("sceKernelVirtualQuery", hle_virtual_query);
    REG("sceKernelGetDirectMemoryType", hle_direct_mem_type);
    REG_CLOCK("sceKernelGetProcessTime", hle_kernel_time);
    REG_CLOCK("sceKernelReadTsc", hle_read_tsc);
    REG_CLOCK("sceKernelGetTscFrequency", hle_tsc_frequency);
#undef REG_CLOCK
#undef REG
    host_log("timing: the guest's clocks (gettimeofday, time, clock, sceKernelClockGettime, GetProcessTime, ReadTsc) %s; "
             "its mutexes, condition variables, read/write locks, semaphores and event queues sleep on %s, %u pauses of spin "
             "before a contended lock sleeps (BBHOST_FUTEX_SPIN)",
             hle_raw_clock() ? "bound raw, read from the performance counter (BBHOST_RAW_CLOCK=0: thunked)" : "thunked (BBHOST_RAW_CLOCK=0 or FS mode)",
#if defined(_WIN32)
             host_futex_native() ? "WaitOnAddress (BBHOST_FUTEX=0: winpthreads)" : "winpthreads' condition variables (BBHOST_FUTEX=0)",
#else
             host_futex_native() ? "futex (BBHOST_FUTEX=0: std::condition_variable)" : "std::condition_variable (BBHOST_FUTEX=0)",
#endif
             host_futex_spin());
}
