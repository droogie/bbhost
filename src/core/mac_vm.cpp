#include "core/mac_vm.h"

#if defined(__APPLE__)

#include "log.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <mutex>
#include <vector>

namespace {

using ull = unsigned long long;

constexpr std::uint64_t kPage = 0x1000;
// A placeholder's tag in the kernel's map: one of the sixteen kept for
// applications (vmmap shows it as "Memory Tag 255").
constexpr int kTag = VM_MEMORY_APPLICATION_SPECIFIC_16;
// A rel32's reach, less a page so a stub's own length stays inside it.
constexpr std::uint64_t kReach = 0x7fff0000ull;

struct Window {
    std::uint64_t lo, hi;
};
std::mutex g_mu;
std::vector<Window> g_windows;  // set at start, never shrinks

std::uint64_t page_up(std::uint64_t a) { return (a + kPage - 1) & ~(kPage - 1); }

// The kernel's map entry holding `at`, or the first one above it; false when
// there is none.
struct Region {
    std::uint64_t lo, hi;
    bool placeholder;
};
bool region_at(std::uint64_t at, Region* out) {
    mach_vm_address_t a = at;
    mach_vm_size_t size = 0;
    natural_t depth = 0;
    vm_region_submap_short_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_SUBMAP_SHORT_INFO_COUNT_64;
    if (mach_vm_region_recurse(mach_task_self(), &a, &size, &depth, reinterpret_cast<vm_region_recurse_info_t>(&info),
                               &count) != KERN_SUCCESS) {
        return false;
    }
    out->lo = a;
    out->hi = a + size;
    out->placeholder = !info.is_submap && info.user_tag == kTag && info.protection == VM_PROT_NONE;
    return true;
}

// Every page of [at, at+len) is a placeholder.
bool placeholders(std::uint64_t at, std::uint64_t len) {
    for (std::uint64_t p = at; p < at + len;) {
        Region r{};
        if (!region_at(p, &r) || r.lo > p || !r.placeholder) return false;
        p = r.hi;
    }
    return true;
}

// [at, at+len) as placeholders: over what is there (replace), or only where
// nothing is (then it fails when anything is).
bool place(std::uint64_t at, std::uint64_t len, bool replace) {
    if (replace) {
        return mmap(reinterpret_cast<void*>(at), len, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, VM_MAKE_TAG(kTag), 0) !=
               MAP_FAILED;
    }
    mach_vm_address_t a = at;
    return mach_vm_map(mach_task_self(), &a, len, 0, VM_FLAGS_FIXED | VM_MAKE_TAG(kTag), MACH_PORT_NULL, 0, FALSE,
                       VM_PROT_NONE, VM_PROT_ALL, VM_INHERIT_DEFAULT) == KERN_SUCCESS;
}

}  // namespace

bool mac_vm_adopt(std::uint64_t lo, std::uint64_t hi) {
    lo = page_up(lo);
    hi &= ~(kPage - 1);
    std::lock_guard<std::mutex> lock(g_mu);
    if (hi <= lo || !place(lo, hi - lo, true)) {
        host_log("mac-vm: the image's reservation [0x%llx, 0x%llx) could not be made placeholders: errno %d",
                 static_cast<ull>(lo), static_cast<ull>(hi), errno);
        return false;
    }
    g_windows.push_back({lo, hi});
    return true;
}

bool mac_vm_reserve(std::uint64_t base, std::uint64_t len) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!place(base, len, false)) {
        Region r{};
        if (region_at(base, &r) && r.lo < base + len) {
            host_log("mac-vm: [0x%llx, 0x%llx) is taken: something is mapped at [0x%llx, 0x%llx)", static_cast<ull>(base),
                     static_cast<ull>(base + len), static_cast<ull>(r.lo), static_cast<ull>(r.hi));
        } else {
            host_log("mac-vm: [0x%llx, 0x%llx) could not be reserved", static_cast<ull>(base), static_cast<ull>(base + len));
        }
        return false;
    }
    g_windows.push_back({base, base + len});
    return true;
}

void* mac_vm_map(void* at, std::uint64_t len, int prot, int flags, int fd, std::uint64_t offset, MacVmPlace place_as) {
    if (!len) return nullptr;
    const auto a = reinterpret_cast<std::uint64_t>(at);
    void* p = MAP_FAILED;
    if (!at || place_as == MacVmPlace::Anywhere) {
        p = mmap(at, len, prot, flags, fd, static_cast<off_t>(offset));
    } else if (place_as == MacVmPlace::Exact) {
        p = mmap(at, len, prot, flags | MAP_FIXED, fd, static_cast<off_t>(offset));
    } else {
        // Placeholders are ours to map over. Free space is made placeholders
        // first, which fails when anything is there, so nothing is replaced
        // that was not ours.
        std::lock_guard<std::mutex> lock(g_mu);
        const bool ours = placeholders(a, len);
        if (!ours && !place(a, len, false)) return nullptr;
        p = mmap(at, len, prot, flags | MAP_FIXED, fd, static_cast<off_t>(offset));
        if (p == MAP_FAILED && !ours) munmap(at, len);
    }
    return p == MAP_FAILED ? nullptr : p;
}

bool mac_vm_unmap(void* at, std::uint64_t len) {
    const auto a = reinterpret_cast<std::uint64_t>(at);
    const std::uint64_t end = a + len;
    std::lock_guard<std::mutex> lock(g_mu);
    bool ok = true;
    for (std::uint64_t p = a; p < end;) {
        // The piece up to the next window edge: inside a window, placeholders
        // again; outside, unmapped.
        const Window* in = nullptr;
        std::uint64_t next = end;
        for (const Window& w : g_windows) {
            if (p >= w.lo && p < w.hi) in = &w;
            else if (w.lo > p) next = std::min(next, w.lo);
        }
        const std::uint64_t q = in ? std::min(end, in->hi) : next;
        ok = (in ? place(p, q - p, true) : munmap(reinterpret_cast<void*>(p), q - p) == 0) && ok;
        p = q;
    }
    return ok;
}

void* mac_vm_map_near(std::uint64_t lo, std::uint64_t hi, std::uint64_t len, int prot) {
    len = page_up(len);
    if (!len || hi < lo) return nullptr;
    // Starts from which every byte of [lo, hi) reaches every byte of the
    // range, and back.
    const std::uint64_t lowest = hi > kReach ? page_up(hi - kReach) : kPage;
    const std::uint64_t highest = (lo + kReach - len) & ~(kPage - 1);
    const auto take = [&](std::uint64_t a) -> void* {
        void* p = mmap(reinterpret_cast<void*>(a), len, prot, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
        return p == MAP_FAILED ? nullptr : p;
    };
    std::lock_guard<std::mutex> lock(g_mu);
    // Above the range, the first placeholders that hold len.
    for (std::uint64_t p = page_up(hi); p <= highest;) {
        Region r{};
        if (!region_at(p, &r) || r.lo > highest) break;
        const std::uint64_t a = std::max(p, r.lo);
        if (r.placeholder && a + len <= r.hi) return take(a);
        p = r.hi;
    }
    // Below it, the highest.
    std::uint64_t best = 0;
    for (std::uint64_t p = lowest; p < lo;) {
        Region r{};
        if (!region_at(p, &r) || r.lo >= lo) break;
        const std::uint64_t top = std::min(r.hi, lo);
        if (r.placeholder && top >= len && top - len >= std::max(p, r.lo)) best = top - len;
        p = r.hi;
    }
    return best ? take(best) : nullptr;
}

#endif
