#include "decomp/decomp.h"

#include "core/elf.h"
#include "core/memory.h"
#include "core/thunk.h"
#include "decomp/insn.h"
#include "decomp/name_lines.h"
#include "engine/addr.h"
#include "log.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

using ull = unsigned long long;

std::vector<DecompFunction> g_added;

struct Placed {
    const DecompFunction* fn;
    bool compared;
};
std::vector<Placed> g_placed;
bool g_comparing = false;

// Trampolines are carved from executable pages within a rel32 of the whole
// image: the displaced instructions run there re-aimed (decomp/insn.h), then
// a jump back after them. Each entry's own jump is 5 bytes, a rel32 to its
// far leg on the same page - jmp [rip+0] and the address of ours - so an
// entry gives up as few as 5 bytes of whole instructions.
constexpr std::size_t kPage = 0x1000;
constexpr std::uint64_t kReach = 0x7fff0000ull;  // a rel32, with room for the page itself
std::uint8_t* g_page = nullptr;
std::size_t g_page_used = 0;

// An executable page from which a rel32 reaches all of [lo, hi), and back.
std::uint8_t* page_near(std::uint64_t lo, std::uint64_t hi) {
    const auto reaches = [&](std::uint64_t a) { return a >= hi ? a + kPage - lo < kReach : a + kReach > hi && a < lo; };
#if defined(_WIN32)
    for (std::uint64_t a = (hi + 0xffff) & ~0xffffull; reaches(a); a += 0x10000)
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(a), kPage, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE))
            return static_cast<std::uint8_t*>(p);
    for (std::uint64_t a = (lo & ~0xffffull) - 0x10000; a > 0x10000 && reaches(a); a -= 0x10000)
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(a), kPage, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE))
            return static_cast<std::uint8_t*>(p);
    return nullptr;
#else
    constexpr int kProt = PROT_READ | PROT_WRITE | PROT_EXEC;
    // The low 2 GiB first: they hold the image at its preferred slide.
    void* p = mmap(nullptr, kPage, kProt, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    if (p != MAP_FAILED) {
        if (reaches(reinterpret_cast<std::uint64_t>(p))) return static_cast<std::uint8_t*>(p);
        munmap(p, kPage);
    }
    const auto at = [&](std::uint64_t a) -> std::uint8_t* {
        void* q = mmap(reinterpret_cast<void*>(a), kPage, kProt, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (q == MAP_FAILED) return nullptr;
        if (reinterpret_cast<std::uint64_t>(q) == a) return static_cast<std::uint8_t*>(q);
        munmap(q, kPage);  // a kernel that took it as a hint
        return nullptr;
    };
    for (std::uint64_t a = (hi + 0xffff) & ~0xffffull; reaches(a); a += 0x10000)
        if (std::uint8_t* q = at(a)) return q;
    for (std::uint64_t a = (lo & ~0xffffull) - 0x10000; a > 0x10000 && reaches(a); a -= 0x10000)
        if (std::uint8_t* q = at(a)) return q;
    return nullptr;
#endif
}

// Room for one function's trampoline and far leg, taken only once it is
// written (carve_commit).
std::uint8_t* carve(const GuestMemory& mem, std::size_t need) {
    if (!g_page || g_page_used + need > kPage) {
        g_page = page_near(mem.slide, mem.slide + mem.size);
        g_page_used = 0;
        if (!g_page) return nullptr;
    }
    return g_page + g_page_used;
}

void carve_commit(std::size_t used) { g_page_used += (used + 15) & ~std::size_t{15}; }

void put_rel32(std::uint8_t* at, std::uint8_t op, std::uint64_t target) {
    at[0] = op;
    const auto rel = static_cast<std::int32_t>(static_cast<std::int64_t>(target - (reinterpret_cast<std::uint64_t>(at) + 5)));
    std::memcpy(at + 1, &rel, 4);
}

bool rel32_reaches(const std::uint8_t* from, std::uint64_t target) {
    const std::int64_t d = static_cast<std::int64_t>(target - (reinterpret_cast<std::uint64_t>(from) + 5));
    return d >= INT32_MIN && d <= INT32_MAX;
}

// `name` in a comma-separated list.
bool listed(const char* list, const char* name) {
    if (!list) return false;
    const std::size_t len = std::strlen(name);
    for (const char* p = list; *p;) {
        const char* end = std::strchr(p, ',');
        const std::size_t n = end ? static_cast<std::size_t>(end - p) : std::strlen(p);
        if (n == len && std::strncmp(p, name, n) == 0) return true;
        if (!end) break;
        p = end + 1;
    }
    return false;
}

bool place(ElfImage* image, const DecompFunction& fn, bool compare) {
    const std::uint64_t at = image->mem.slide + (fn.bn - kPreferredGuestSlide);
    if (fn.entry_len < 5 || at < image->mem.slide || at + fn.entry_len > image->mem.slide + image->mem.size) return false;
    auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, at));
    if (!p || std::memcmp(p, fn.entry, fn.entry_len) != 0) {
        host_log("decomp: %s refused, 0x%llx does not hold the instructions it was written against; the game's stays",
                 fn.name, static_cast<ull>(fn.bn));
        return false;
    }
    if (fn.body_ok && !fn.body_ok(p)) {
        host_log("decomp: %s refused, its body after 0x%llx is not the one ours mirrors; the game's stays", fn.name,
                 static_cast<ull>(fn.bn));
        return false;
    }
    // The trampoline: the displaced instructions, re-aimed, at most 64 bytes
    // (a short branch grows to its rel32 form); the jump back; the far leg.
    constexpr std::size_t kMoved = 64, kNeed = kMoved + 5 + 14;
    std::uint8_t* t = carve(image->mem, kNeed);
    if (!t) {
        host_log("decomp: %s refused, no executable page within reach of the image", fn.name);
        return false;
    }
    const std::size_t moved = x86_relocate(fn.entry, fn.entry_len, at, t, reinterpret_cast<std::uint64_t>(t), kMoved);
    std::uint8_t* back = t + moved;
    std::uint8_t* leg = back + 5;
    if (!moved || !rel32_reaches(back, at + fn.entry_len) || !rel32_reaches(p, reinterpret_cast<std::uint64_t>(leg))) {
        host_log("decomp: %s refused, the first %zu bytes at 0x%llx cannot run from a trampoline (decomp/insn.h)", fn.name,
                 fn.entry_len, static_cast<ull>(fn.bn));
        return false;
    }
    put_rel32(back, 0xe9, at + fn.entry_len);
    void* dest = compare                              ? thunk_wrap(fn.compare)
                 : fn.kind == DecompKind::Leaf        ? fn.ours
                 : fn.kind == DecompKind::HostedFrame ? thunk_wrap_capture_frame(fn.ours)
                                                      : thunk_wrap(fn.ours);
    const std::uint64_t d = reinterpret_cast<std::uint64_t>(dest);
    leg[0] = 0xff;
    leg[1] = 0x25;  // jmp [rip+0]
    std::memset(leg + 2, 0, 4);
    std::memcpy(leg + 6, &d, 8);
    carve_commit(moved + 5 + 14);
    const std::uint64_t lo = at & ~0xfffull, hi = (at + fn.entry_len + 0xfff) & ~0xfffull;
    if (!guest_protect_rwx(&image->mem, lo, hi - lo)) return false;
    // The trampoline is published before the jump that can lead to it.
    if (fn.original) *fn.original = t;
    std::memset(p + 5, 0xcc, fn.entry_len - 5);
    put_rel32(p, 0xe9, reinterpret_cast<std::uint64_t>(leg));
    guest_protect_rx(&image->mem, lo, hi - lo);
    return true;
}

// The install report's names in parts (decomp/name_lines.h): a part and its
// prefix stay well inside the 1023 characters one log line holds.
constexpr std::size_t kNameLineWidth = 900;

void log_names(const char* what, const std::vector<std::string>& names) {
    const std::vector<std::string> lines = decomp_name_lines(names, kNameLineWidth);
    for (std::size_t i = 0; i < lines.size(); ++i)
        host_log("decomp: %s (%zu/%zu): %s", what, i + 1, lines.size(), lines[i].c_str());
}

}  // namespace

std::uint64_t g_decomp_slide = kPreferredGuestSlide;

void decomp_add(const DecompFunction& fn) { g_added.push_back(fn); }

bool decomp_comparing() { return g_comparing; }

int decomp_placed(const char* name) {
    for (const Placed& p : g_placed)
        if (std::strcmp(p.fn->name, name) == 0) return p.compared ? 2 : 1;
    return 0;
}

std::uint64_t decomp_guest(std::uint64_t bn) { return g_decomp_slide + (bn - kPreferredGuestSlide); }

void decomp_install(ElfImage* image) {
    if (!image || image->sha256 != kEboot109Sha256 || g_added.empty()) return;
    g_decomp_slide = image->mem.slide;
    if (const char* e = std::getenv("BBHOST_DECOMP"); e && e[0] == '0') {
        host_log("decomp: off (BBHOST_DECOMP=0): the game's own code for all %zu functions", g_added.size());
        return;
    }
    const char* off = std::getenv("BBHOST_DECOMP_OFF");
    const char* cmp = std::getenv("BBHOST_DECOMP_COMPARE");
    g_comparing = cmp && cmp[0] == '1';
    std::vector<std::string> ours, kept;
    for (const DecompFunction& fn : g_added) {
        if (listed(off, fn.name)) {
            kept.emplace_back(fn.name);
            continue;
        }
        const bool compare = g_comparing && fn.compare && fn.counts;
        if (!place(image, fn, compare)) {
            kept.emplace_back(fn.name);
            continue;
        }
        g_placed.push_back({&fn, compare});
        ours.push_back(compare ? std::string(fn.name) + " (compared)" : std::string(fn.name));
    }
    // The count first, then the names in parts: the game's (switched off or not placed), then ours.
    host_log("decomp: %zu of %zu functions ours, %zu the game's", g_placed.size(), g_added.size(), kept.size());
    log_names("the game's", kept);
    log_names("ours", ours);
}

void decomp_report() {
    for (const Placed& p : g_placed) {
        if (p.fn->report) p.fn->report();
        if (!p.compared) continue;
        host_log("decomp: %s (%s, 0x%llx): %llu calls compared with the game's, %llu differ", p.fn->name, p.fn->area,
                 static_cast<ull>(p.fn->bn), static_cast<ull>(p.fn->counts->calls.load()),
                 static_cast<ull>(p.fn->counts->differ.load()));
    }
}
