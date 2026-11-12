#include "core/linux_vm.h"

#if !defined(_WIN32)

#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/types.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
#ifndef PR_SET_VMA
#define PR_SET_VMA 0x53564d41
#endif
#ifndef PR_SET_VMA_ANON_NAME
#define PR_SET_VMA_ANON_NAME 0
#endif

namespace {

constexpr std::uintptr_t kPage = 0x1000;

struct Window {
    std::uintptr_t lo = 0;
    std::uintptr_t hi = 0;
};

std::mutex g_mu;
std::vector<Window> g_windows;                    // sorted by lo, disjoint
std::map<std::uintptr_t, std::uintptr_t> g_used;  // lo -> hi mapped here inside the windows: disjoint, touching ones merged

std::uintptr_t page_up(std::uintptr_t v) { return (v + kPage - 1) & ~(kPage - 1); }

// [lo, hi) PROT_NONE and MAP_NORESERVE: no memory and no commit. `over`
// replaces what is there (MAP_FIXED: only ever our own); otherwise only where
// nothing is mapped.
bool reserve_range(std::uintptr_t lo, std::uintptr_t hi, bool over) {
    void* want = reinterpret_cast<void*>(lo);
    void* p = mmap(want, hi - lo, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | (over ? MAP_FIXED : MAP_FIXED_NOREPLACE), -1, 0);
    if (p == MAP_FAILED) {
        return false;
    }
    if (p != want) {  // a kernel before 4.17 takes MAP_FIXED_NOREPLACE for a hint
        munmap(p, hi - lo);
        errno = EEXIST;
        return false;
    }
    // Named in /proc/<pid>/maps ([anon:bbhost guest window]) for whoever reads
    // one; kernels before 5.17, or without the option, refuse, which is fine.
    prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, lo, hi - lo, "bbhost guest window");
    return true;
}

void add_window_locked(std::uintptr_t lo, std::uintptr_t hi) {
    const auto at = std::lower_bound(g_windows.begin(), g_windows.end(), lo,
                                     [](const Window& w, std::uintptr_t v) { return w.lo < v; });
    g_windows.insert(at, Window{lo, hi});
}

bool overlaps_window_locked(std::uintptr_t lo, std::uintptr_t hi) {
    return std::any_of(g_windows.begin(), g_windows.end(), [&](const Window& w) { return lo < w.hi && hi > w.lo; });
}

bool in_one_window_locked(std::uintptr_t lo, std::uintptr_t hi) {
    return std::any_of(g_windows.begin(), g_windows.end(), [&](const Window& w) { return lo >= w.lo && hi <= w.hi; });
}

bool used_overlaps_locked(std::uintptr_t lo, std::uintptr_t hi) {
    auto it = g_used.upper_bound(lo);
    if (it != g_used.begin() && std::prev(it)->second > lo) {
        return true;
    }
    return it != g_used.end() && it->first < hi;
}

void used_add_locked(std::uintptr_t lo, std::uintptr_t hi) {
    auto it = g_used.upper_bound(lo);
    if (it != g_used.begin() && std::prev(it)->second >= lo) {
        --it;
    }
    while (it != g_used.end() && it->first <= hi) {
        lo = std::min(lo, it->first);
        hi = std::max(hi, it->second);
        it = g_used.erase(it);
    }
    g_used.emplace(lo, hi);
}

void used_sub_locked(std::uintptr_t lo, std::uintptr_t hi) {
    auto it = g_used.upper_bound(lo);
    if (it != g_used.begin()) {
        --it;
    }
    while (it != g_used.end() && it->first < hi) {
        const std::uintptr_t a = it->first, b = it->second;
        if (b <= lo) {
            ++it;
            continue;
        }
        it = g_used.erase(it);
        if (a < lo) {
            g_used.emplace(a, lo);
        }
        if (b > hi) {
            g_used.emplace(hi, b);
            break;
        }
    }
}

// The parts of [lo, hi) inside the windows, recorded as mapped.
void note_mapped_locked(std::uintptr_t lo, std::uintptr_t hi) {
    for (const Window& w : g_windows) {
        const std::uintptr_t a = std::max(lo, w.lo), b = std::min(hi, w.hi);
        if (a < b) {
            used_add_locked(a, b);
        }
    }
}

// After a failed MAP_FIXED in a window (which may have unmapped the range
// before it failed): the reservation back where nothing of ours was.
void restore_free_locked(std::uintptr_t lo, std::uintptr_t hi) {
    std::uintptr_t cur = lo;
    auto it = g_used.upper_bound(lo);
    if (it != g_used.begin()) {
        --it;  // the one that may hold lo
    }
    for (; it != g_used.end() && it->first < hi; ++it) {
        if (it->second <= cur) {
            continue;
        }
        if (it->first > cur) {
            reserve_range(cur, it->first, true);
        }
        cur = it->second;
    }
    if (cur < hi) {
        reserve_range(cur, hi, true);
    }
}

struct MapsLine {
    std::uintptr_t lo = 0;
    std::uintptr_t hi = 0;
    std::string text;
};

// The process's mappings that overlap [lo, hi), from /proc/self/maps (sorted).
std::vector<MapsLine> maps_overlapping(std::uintptr_t lo, std::uintptr_t hi) {
    std::vector<MapsLine> out;
    FILE* f = std::fopen("/proc/self/maps", "re");
    if (!f) {
        return out;
    }
    char* line = nullptr;
    std::size_t cap = 0;
    ssize_t n = 0;
    while ((n = getline(&line, &cap, f)) > 0) {
        unsigned long long a = 0, b = 0;
        if (std::sscanf(line, "%llx-%llx", &a, &b) != 2 || a >= hi || b <= lo) {
            continue;
        }
        std::string text;
        for (ssize_t i = 0; i < n; ++i) {
            const char c = line[i];
            if (c == '\n') {
                break;
            }
            if (c == ' ' && !text.empty() && text.back() == ' ') {
                continue;  // the padding before the path, one space
            }
            text += c;
        }
        out.push_back({static_cast<std::uintptr_t>(a), static_cast<std::uintptr_t>(b), text});
    }
    std::free(line);
    std::fclose(f);
    return out;
}

}  // namespace

bool linux_vm_reserve(std::uint64_t base, std::uint64_t len) {
    const auto lo = static_cast<std::uintptr_t>(base);
    const std::uintptr_t hi = page_up(lo + len);
    if (!len || (lo & (kPage - 1)) != 0 || hi <= lo) {
        errno = EINVAL;
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (overlaps_window_locked(lo, hi)) {
        errno = EEXIST;
        return false;
    }
    if (!reserve_range(lo, hi, false)) {
        return false;
    }
    add_window_locked(lo, hi);
    return true;
}

std::uint64_t linux_vm_reserve_free(std::uint64_t base, std::uint64_t len) {
    const auto lo = static_cast<std::uintptr_t>(base);
    const std::uintptr_t hi = page_up(lo + len);
    if (!len || (lo & (kPage - 1)) != 0 || hi <= lo) {
        return 0;
    }
    const std::vector<MapsLine> taken = maps_overlapping(lo, hi);  // ours among them
    std::lock_guard<std::mutex> lock(g_mu);
    std::uint64_t got = 0;
    const auto take = [&](std::uintptr_t a, std::uintptr_t b) {
        // Something mapped meanwhile fails it (MAP_FIXED_NOREPLACE): left out.
        if (a < b && !overlaps_window_locked(a, b) && reserve_range(a, b, false)) {
            add_window_locked(a, b);
            got += b - a;
        }
    };
    std::uintptr_t cur = lo;
    for (const MapsLine& m : taken) {
        take(cur, std::max(cur, std::min(m.lo, hi)));
        cur = std::max(cur, m.hi);
    }
    take(cur, hi);
    return got;
}

std::vector<std::string> linux_vm_occupants(std::uint64_t base, std::uint64_t len) {
    std::vector<std::string> out;
    for (MapsLine& m : maps_overlapping(static_cast<std::uintptr_t>(base), static_cast<std::uintptr_t>(base + len))) {
        out.push_back(std::move(m.text));
    }
    return out;
}

bool linux_vm_owns(std::uint64_t p, std::uint64_t len) {
    const auto lo = static_cast<std::uintptr_t>(p);
    const std::uintptr_t hi = page_up(lo + len);
    if (!len || hi <= lo) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    return in_one_window_locked(lo, hi);
}

void* linux_vm_mmap(void* at, std::size_t len, int prot, int flags, int fd, std::int64_t off, LinuxVmFix fix) {
    flags &= ~(MAP_FIXED | MAP_FIXED_NOREPLACE);
    if (!at || !len || fix == LinuxVmFix::None) {
        return mmap(at, len, prot, flags, fd, static_cast<off_t>(off));
    }
    const auto lo = reinterpret_cast<std::uintptr_t>(at);
    const std::uintptr_t hi = page_up(lo + len);
    std::lock_guard<std::mutex> lock(g_mu);
    if (hi > lo && in_one_window_locked(lo, hi)) {
        // Ours: the reservation is replaced; another mapping made here is not
        // unless asked (the check and the map under the one lock, as atomic
        // as MAP_FIXED_NOREPLACE is).
        if (fix == LinuxVmFix::NoReplace && used_overlaps_locked(lo, hi)) {
            errno = EEXIST;
            return MAP_FAILED;
        }
        void* p = mmap(at, len, prot, flags | MAP_FIXED, fd, static_cast<off_t>(off));
        if (p == MAP_FAILED) {
            const int e = errno;
            restore_free_locked(lo, hi);
            errno = e;
            return p;
        }
        used_add_locked(lo, hi);
        return p;
    }
    // Outside the windows (or across an edge of one, which a NoReplace cannot
    // take: the reservation is there) the plain call.
    void* p = mmap(at, len, prot, flags | (fix == LinuxVmFix::Exact ? MAP_FIXED : MAP_FIXED_NOREPLACE), fd,
                   static_cast<off_t>(off));
    if (p == at) {
        note_mapped_locked(lo, hi);  // an Exact one across a window's edge took that part of it
    }
    return p;
}

int linux_vm_munmap(void* at, std::size_t len) {
    const auto lo = reinterpret_cast<std::uintptr_t>(at);
    const std::uintptr_t hi = page_up(lo + len);
    if (!len || (lo & (kPage - 1)) != 0 || hi <= lo) {
        return munmap(at, len);  // what munmap says to that (EINVAL)
    }
    std::lock_guard<std::mutex> lock(g_mu);
    int rc = 0;
    std::uintptr_t cur = lo;
    for (const Window& w : g_windows) {
        if (w.hi <= cur) {
            continue;
        }
        if (w.lo >= hi) {
            break;
        }
        const std::uintptr_t a = std::max(cur, w.lo), b = std::min(hi, w.hi);
        if (cur < a && munmap(reinterpret_cast<void*>(cur), a - cur) != 0) {
            rc = -1;
        }
        // Back to the reservation (MAP_FIXED replaces the mapping in one step:
        // the range is never free for something else to take).
        if (!reserve_range(a, b, true)) {
            rc = -1;
        }
        used_sub_locked(a, b);
        cur = b;
    }
    if (cur < hi && munmap(reinterpret_cast<void*>(cur), hi - cur) != 0) {
        rc = -1;
    }
    return rc;
}

bool linux_vm_populate(void* p, std::size_t len) {
    return madvise(p, len, MADV_POPULATE_WRITE) == 0;
}

std::uint64_t linux_vm_reserved_bytes() {
    std::lock_guard<std::mutex> lock(g_mu);
    std::uint64_t n = 0;
    for (const Window& w : g_windows) {
        n += w.hi - w.lo;
    }
    return n;
}

std::uint64_t linux_vm_mapped_bytes() {
    std::lock_guard<std::mutex> lock(g_mu);
    std::uint64_t n = 0;
    for (const auto& [lo, hi] : g_used) {
        n += hi - lo;
    }
    return n;
}

#endif
