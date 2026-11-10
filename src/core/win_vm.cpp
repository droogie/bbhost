#include "core/win_vm.h"

#if defined(_WIN32)

#include "log.h"

#include <windows.h>

#include <map>
#include <mutex>
#include <vector>

namespace {

constexpr std::uint64_t kGran = 0x10000;  // the allocation granularity

using PFN_VirtualAlloc2 = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
using PFN_MapViewOfFile3 = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG,
                                          MEM_EXTENDED_PARAMETER*, ULONG);
using PFN_UnmapViewOfFile2 = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);

struct Api {
    PFN_VirtualAlloc2 VirtualAlloc2 = nullptr;
    PFN_MapViewOfFile3 MapViewOfFile3 = nullptr;
    PFN_UnmapViewOfFile2 UnmapViewOfFile2 = nullptr;
    bool ok = false;
};

const Api& api() {
    static const Api a = [] {
        Api r;
        HMODULE kb = GetModuleHandleW(L"kernelbase.dll");
        if (!kb) kb = LoadLibraryW(L"kernelbase.dll");
        if (kb) {
            r.VirtualAlloc2 = reinterpret_cast<PFN_VirtualAlloc2>(GetProcAddress(kb, "VirtualAlloc2"));
            r.MapViewOfFile3 = reinterpret_cast<PFN_MapViewOfFile3>(GetProcAddress(kb, "MapViewOfFile3"));
            r.UnmapViewOfFile2 = reinterpret_cast<PFN_UnmapViewOfFile2>(GetProcAddress(kb, "UnmapViewOfFile2"));
        }
        r.ok = r.VirtualAlloc2 && r.MapViewOfFile3 && r.UnmapViewOfFile2;
        if (!r.ok) host_log("win-vm: VirtualAlloc2/MapViewOfFile3 unavailable; fixed guest addresses are best effort");
        return r;
    }();
    return a;
}

// Every view this layer made: base -> section, offset, length, ownership.
struct View {
    HANDLE section = nullptr;
    std::uint64_t offset = 0;
    std::uint64_t len = 0;
    bool owned = false;  // a private section: closed when its last view goes
};
std::mutex g_mu;
std::map<std::uintptr_t, View> g_views;

bool aligned(std::uint64_t v) { return (v & (kGran - 1)) == 0; }

const char* state_name(DWORD s) {
    return s == MEM_FREE ? "free" : s == MEM_RESERVE ? "reserved" : s == MEM_COMMIT ? "committed" : "?";
}

bool is_placeholder(const MEMORY_BASIC_INFORMATION& m) {
    return m.State == MEM_RESERVE && m.Type == MEM_PRIVATE;
}

int section_users_locked(HANDLE s) {
    int n = 0;
    for (const auto& [b, v] : g_views) n += v.section == s;
    return n;
}

// Makes [at, at+len) exactly one placeholder: our own views there are
// unmapped first when `replace`, foreign memory fails. Caller holds g_mu.
bool carve_locked(std::uintptr_t at, std::uint64_t len, bool replace);

// Unmaps [at, at+len) of one view [vb, vb+v.len): the whole view when it
// covers it, otherwise the view whole and the kept parts again, with their
// protections. Caller holds g_mu.
bool unmap_view_part_locked(std::uintptr_t vb, const View& v, std::uintptr_t at, std::uint64_t len) {
    const std::uintptr_t lo = at > vb ? at : vb;
    const std::uintptr_t hi = (at + len) < (vb + v.len) ? (at + len) : (vb + v.len);
    if (lo >= hi) return true;
    const bool whole = lo == vb && hi == vb + v.len;
    if (!whole && (!aligned(lo) || !aligned(hi))) {
        host_log("win-vm: partial unmap [%p,%p) of the view at %p is not 64 KiB aligned; it stays mapped",
                 reinterpret_cast<void*>(lo), reinterpret_cast<void*>(hi), reinterpret_cast<void*>(vb));
        return false;
    }
    struct Part {
        std::uintptr_t base;
        std::uint64_t len;
        DWORD protect;
    };
    std::vector<Part> kept;
    if (!whole) {
        const std::uintptr_t ranges[2][2] = {{vb, lo}, {hi, vb + v.len}};
        for (const auto& r : ranges) {
            for (std::uintptr_t p = r[0]; p < r[1];) {
                MEMORY_BASIC_INFORMATION m{};
                if (!VirtualQuery(reinterpret_cast<void*>(p), &m, sizeof(m))) break;
                const std::uintptr_t end = reinterpret_cast<std::uintptr_t>(m.BaseAddress) + m.RegionSize;
                const std::uintptr_t e = end < r[1] ? end : r[1];
                kept.push_back({p, e - p, m.Protect ? m.Protect : PAGE_READWRITE});
                p = e;
            }
        }
    }
    const View copy = v;
    if (!api().UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(vb), MEM_PRESERVE_PLACEHOLDER)) {
        host_log("win-vm: UnmapViewOfFile2 %p: error %lu", reinterpret_cast<void*>(vb), GetLastError());
        return false;
    }
    g_views.erase(vb);
    // The kept parts come back as their own views. Parts that only differ in
    // protection are remapped as one run per 64 KiB-aligned piece.
    std::uintptr_t run_base = 0;
    std::uint64_t run_len = 0;
    auto flush = [&]() {
        if (!run_len) return;
        if (!(run_base == vb && run_len == copy.len)) {
            VirtualFree(reinterpret_cast<void*>(run_base), run_len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
        }
        void* p = api().MapViewOfFile3(copy.section, GetCurrentProcess(), reinterpret_cast<void*>(run_base),
                                       copy.offset + (run_base - vb), run_len, MEM_REPLACE_PLACEHOLDER,
                                       PAGE_EXECUTE_READWRITE, nullptr, 0);
        if (!p) {
            host_log("win-vm: remapping the kept part at %p (0x%llx) failed: error %lu",
                     reinterpret_cast<void*>(run_base), static_cast<unsigned long long>(run_len), GetLastError());
        } else {
            g_views[run_base] = View{copy.section, copy.offset + (run_base - vb), run_len, copy.owned};
            for (const Part& k : kept) {
                if (k.base >= run_base && k.base < run_base + run_len && k.protect != PAGE_EXECUTE_READWRITE) {
                    DWORD old = 0;
                    VirtualProtect(reinterpret_cast<void*>(k.base), k.len, k.protect, &old);
                }
            }
        }
        run_len = 0;
    };
    for (const Part& k : kept) {
        if (run_len && run_base + run_len == k.base) {
            run_len += k.len;
        } else {
            flush();
            run_base = k.base;
            run_len = k.len;
        }
    }
    flush();
    if (copy.owned && section_users_locked(copy.section) == 0) CloseHandle(copy.section);
    return true;
}

bool unmap_locked(std::uintptr_t at, std::uint64_t len) {
    bool ok = true;
    for (;;) {
        bool any = false;
        for (const auto& [vb, v] : g_views) {
            if (vb < at + len && vb + v.len > at) {
                ok = unmap_view_part_locked(vb, v, at, len) && ok;
                any = true;
                break;  // g_views changed
            }
        }
        if (!any) break;
        if (!ok) break;
    }
    return ok;
}

bool carve_locked(std::uintptr_t at, std::uint64_t len, bool replace) {
    if (!aligned(at) || !aligned(len)) {
        host_log("win-vm: [%p, +0x%llx) is not 64 KiB aligned", reinterpret_cast<void*>(at),
                 static_cast<unsigned long long>(len));
        return false;
    }
    if (replace && !unmap_locked(at, len)) return false;
    // Free memory in the range (a guest hint outside the reserved windows,
    // as Linux's MAP_FIXED_NOREPLACE would take) becomes placeholder first.
    for (std::uintptr_t p = at; p < at + len;) {
        MEMORY_BASIC_INFORMATION m{};
        if (!VirtualQuery(reinterpret_cast<void*>(p), &m, sizeof(m))) return false;
        const auto base = reinterpret_cast<std::uintptr_t>(m.BaseAddress);
        const std::uintptr_t end = base + m.RegionSize;
        if (m.State == MEM_FREE) {
            const std::uintptr_t lo = p, hi = end < at + len ? end : at + len;
            if (!api().VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(lo), hi - lo,
                                     MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0)) {
                host_log("win-vm: reserving free [%p, +0x%llx) as a placeholder: error %lu", reinterpret_cast<void*>(lo),
                         static_cast<unsigned long long>(hi - lo), GetLastError());
                return false;
            }
        }
        p = end;
    }
    // Everything in the range must now be placeholder.
    std::uintptr_t first = 0, last_end = 0;
    int regions = 0;
    for (std::uintptr_t p = at; p < at + len;) {
        MEMORY_BASIC_INFORMATION m{};
        if (!VirtualQuery(reinterpret_cast<void*>(p), &m, sizeof(m))) return false;
        if (!is_placeholder(m)) {
            host_log("win-vm: %p is %s memory (type 0x%lx), not a placeholder; the guest range [%p, +0x%llx) is taken",
                     reinterpret_cast<void*>(p), state_name(m.State), m.Type, reinterpret_cast<void*>(at),
                     static_cast<unsigned long long>(len));
            return false;
        }
        const auto base = reinterpret_cast<std::uintptr_t>(m.BaseAddress);
        if (!regions) first = base;
        last_end = base + m.RegionSize;
        ++regions;
        p = last_end;
    }
    if (regions > 1 &&
        !VirtualFree(reinterpret_cast<void*>(first), last_end - first, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
        host_log("win-vm: coalescing %d placeholders at %p: error %lu", regions, reinterpret_cast<void*>(first),
                 GetLastError());
        return false;
    }
    if (first != at || last_end != at + len) {
        if (!VirtualFree(reinterpret_cast<void*>(at), len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            host_log("win-vm: splitting the placeholder at %p (0x%llx): error %lu", reinterpret_cast<void*>(at),
                     static_cast<unsigned long long>(len), GetLastError());
            return false;
        }
    }
    return true;
}

// A fresh placeholder of len below limit, anywhere; its base a multiple of
// `align` where that is above the granularity.
void* place_anywhere_locked(std::uint64_t len, std::uint64_t limit, std::uint64_t align = 0) {
    MEM_ADDRESS_REQUIREMENTS req{};
    req.HighestEndingAddress = reinterpret_cast<void*>(static_cast<std::uintptr_t>(limit));
    if (align > kGran) req.Alignment = static_cast<SIZE_T>(align);
    MEM_EXTENDED_PARAMETER ep{};
    ep.Type = MemExtendedParameterAddressRequirements;
    ep.Pointer = &req;
    void* p = api().VirtualAlloc2(GetCurrentProcess(), nullptr, len, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                  PAGE_NOACCESS, &ep, 1);
    if (!p) {
        p = api().VirtualAlloc2(GetCurrentProcess(), nullptr, len, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                PAGE_NOACCESS, nullptr, 0);
        if (p && reinterpret_cast<std::uintptr_t>(p) + len > limit) {
            host_log("win-vm: a placeholder landed above the guest's address limit (%p)", p);
        }
    }
    if (p && align > kGran && (reinterpret_cast<std::uintptr_t>(p) & (align - 1)) != 0) {
        host_log("win-vm: a placeholder of 0x%llx at %p is not 0x%llx-aligned", static_cast<unsigned long long>(len), p,
                 static_cast<unsigned long long>(align));
        VirtualFree(p, 0, MEM_RELEASE);
        return nullptr;
    }
    return p;
}

}  // namespace

bool win_vm_available() { return api().ok; }

bool win_vm_reserve(std::uint64_t base, std::uint64_t len) {
    if (!api().ok) return false;
    void* p = api().VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(static_cast<std::uintptr_t>(base)),
                                  len, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    if (!p) {
        MEMORY_BASIC_INFORMATION m{};
        VirtualQuery(reinterpret_cast<void*>(static_cast<std::uintptr_t>(base)), &m, sizeof(m));
        host_log("win-vm: reserving [0x%llx, +0x%llx) failed: error %lu (there: %s, base %p, 0x%llx)",
                 static_cast<unsigned long long>(base), static_cast<unsigned long long>(len), GetLastError(),
                 state_name(m.State), m.AllocationBase, static_cast<unsigned long long>(m.RegionSize));
        return false;
    }
    return true;
}

void* win_vm_map(void* section, void* at, std::uint64_t len, std::uint64_t offset, unsigned long protect,
                 WinVmPlace place, std::uint64_t limit, std::uint64_t align) {
    if (!len || !aligned(offset)) {
        host_log("win-vm: view offset 0x%llx is not 64 KiB aligned", static_cast<unsigned long long>(offset));
        return nullptr;
    }
    if (!api().ok) {  // best effort: MapViewOfFileEx at the hint
        if (place == WinVmPlace::Anywhere && align > kGran) return nullptr;
        return MapViewOfFileEx(section, FILE_MAP_ALL_ACCESS | FILE_MAP_EXECUTE, static_cast<DWORD>(offset >> 32),
                               static_cast<DWORD>(offset), len, place == WinVmPlace::Anywhere ? nullptr : at);
    }
    std::lock_guard<std::mutex> lk(g_mu);
    std::uintptr_t base;
    if (place == WinVmPlace::Anywhere) {
        void* p = place_anywhere_locked(len, limit, align);
        if (!p) return nullptr;
        base = reinterpret_cast<std::uintptr_t>(p);
    } else {
        base = reinterpret_cast<std::uintptr_t>(at);
        if (!carve_locked(base, len, place == WinVmPlace::Exact)) return nullptr;
    }
    // Views take a real page protection; NOACCESS is applied afterwards.
    const DWORD view_prot = protect == PAGE_NOACCESS ? PAGE_READWRITE : protect;
    void* p = api().MapViewOfFile3(section, GetCurrentProcess(), reinterpret_cast<void*>(base), offset, len,
                                   MEM_REPLACE_PLACEHOLDER, view_prot, nullptr, 0);
    if (!p) {
        host_log("win-vm: MapViewOfFile3 at %p offset 0x%llx len 0x%llx prot 0x%lx: error %lu",
                 reinterpret_cast<void*>(base), static_cast<unsigned long long>(offset),
                 static_cast<unsigned long long>(len), protect, GetLastError());
        if (place == WinVmPlace::Anywhere) VirtualFree(reinterpret_cast<void*>(base), 0, MEM_RELEASE);
        return nullptr;
    }
    if (protect == PAGE_NOACCESS) {
        DWORD old = 0;
        VirtualProtect(p, len, PAGE_NOACCESS, &old);
    }
    g_views[base] = View{section, offset, len, false};
    return p;
}

void* win_vm_alloc(void* at, std::uint64_t len, unsigned long protect, WinVmPlace place, std::uint64_t limit) {
    if (!api().ok) {
        return VirtualAlloc(place == WinVmPlace::Anywhere ? nullptr : at, len, MEM_RESERVE | MEM_COMMIT, protect);
    }
    HANDLE s = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, static_cast<DWORD>(len >> 32),
                                  static_cast<DWORD>(len), nullptr);
    if (!s) {
        host_log("win-vm: private section of 0x%llx failed: error %lu", static_cast<unsigned long long>(len),
                 GetLastError());
        return nullptr;
    }
    void* p = win_vm_map(s, at, len, 0, protect, place, limit);
    if (!p) {
        CloseHandle(s);
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    g_views[reinterpret_cast<std::uintptr_t>(p)].owned = true;
    return p;
}

void* win_vm_commit(void* at, std::uint64_t len, unsigned long protect, std::uint64_t limit) {
    if (!api().ok) return VirtualAlloc(at, len, MEM_RESERVE | MEM_COMMIT, protect);
    std::lock_guard<std::mutex> lk(g_mu);
    if (at) {
        if (!carve_locked(reinterpret_cast<std::uintptr_t>(at), len, true)) return nullptr;
    } else {
        at = place_anywhere_locked(len, limit);
        if (!at) return nullptr;
    }
    void* p = api().VirtualAlloc2(GetCurrentProcess(), at, len, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                                  protect, nullptr, 0);
    if (!p) host_log("win-vm: committing %p (0x%llx): error %lu", at, static_cast<unsigned long long>(len), GetLastError());
    return p;
}

void* win_vm_place(void* at, std::uint64_t len, WinVmPlace place, std::uint64_t limit) {
    if (!api().ok) return VirtualAlloc(place == WinVmPlace::Anywhere ? nullptr : at, len, MEM_RESERVE, PAGE_NOACCESS);
    std::lock_guard<std::mutex> lk(g_mu);
    if (place == WinVmPlace::Anywhere) return place_anywhere_locked(len, limit);
    return carve_locked(reinterpret_cast<std::uintptr_t>(at), len, place == WinVmPlace::Exact) ? at : nullptr;
}

void win_vm_dump_views() {
    std::lock_guard<std::mutex> lk(g_mu);
    int overlaps = 0;
    for (const auto& [b, v] : g_views) {
        host_log("win-vm view: base 0x%llx len 0x%llx section %p offset 0x%llx%s", static_cast<unsigned long long>(b),
                 static_cast<unsigned long long>(v.len), v.section, static_cast<unsigned long long>(v.offset),
                 v.owned ? " (private)" : "");
        for (const auto& [b2, v2] : g_views) {
            if (b2 <= b || v2.section != v.section) continue;
            if (v.offset < v2.offset + v2.len && v2.offset < v.offset + v.len) {
                ++overlaps;
                host_log("win-vm ALIAS: views at 0x%llx and 0x%llx share section offsets [0x%llx, 0x%llx)",
                         static_cast<unsigned long long>(b), static_cast<unsigned long long>(b2),
                         static_cast<unsigned long long>(v.offset > v2.offset ? v.offset : v2.offset),
                         static_cast<unsigned long long>((v.offset + v.len) < (v2.offset + v2.len) ? v.offset + v.len
                                                                                                    : v2.offset + v2.len));
            }
        }
    }
    host_log("win-vm: %zu views, %d overlapping pairs", g_views.size(), overlaps);
}

bool win_vm_unmap(void* at, std::uint64_t len) {
    if (!api().ok) {
        MEMORY_BASIC_INFORMATION m{};
        VirtualQuery(at, &m, sizeof(m));
        return m.Type == MEM_MAPPED ? UnmapViewOfFile(at) != 0 : VirtualFree(at, 0, MEM_RELEASE) != 0;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    return unmap_locked(reinterpret_cast<std::uintptr_t>(at), len);
}

#endif
