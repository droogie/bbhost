#include "core/legacy_cpu.h"

#include "core/elf.h"
#include "core/legacy_cpu_code.h"
#include "core/memory.h"
#include "core/thunk.h"
#include "decomp/insn.h"
#include "engine/addr.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cpuid.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <signal.h>
#include <ucontext.h>
#endif

namespace {

// The table tools/legacy_cpu_sites.py writes: every site, with the bytes of
// whole instructions a short one's routine takes before and after it.
struct LegacySite {
    std::uint32_t at;  // Binary Ninja's address
    std::uint8_t len;
    LegacyFamily family;
    std::uint8_t before, after;
};
constexpr LegacyFamily kAndn = LegacyFamily::kAndn, kBextr = LegacyFamily::kBextr, kBlsi = LegacyFamily::kBlsi,
                       kBlsr = LegacyFamily::kBlsr, kLzcnt = LegacyFamily::kLzcnt, kTzcnt = LegacyFamily::kTzcnt,
                       kMovbe = LegacyFamily::kMovbe;
[[maybe_unused]] constexpr LegacyFamily kBlsmsk = LegacyFamily::kBlsmsk;  // none in 1.09
constexpr LegacySite kSites[] = {
#include "core/legacy_cpu_sites.inc"
};
constexpr std::size_t kSiteCount = sizeof(kSites) / sizeof(kSites[0]);

// The longest region and the most instructions in one (the table's are 12
// bytes and 3 instructions at most).
constexpr std::size_t kMaxRegion = 32, kMaxInsns = 8;
// The arena: every routine is under 100 bytes; 6,327 of them fit in 1 MiB.
constexpr std::size_t kArena = 2u << 20;

// A rewritten region: [lo, hi) in the image (runtime addresses), its jump to
// [code, code_end) in the arena, its instructions and where each one's code
// starts in the routine (for the int3 net and the crash report), and the
// bytes it held.
struct Region {
    std::uint64_t lo = 0, hi = 0;
    std::uint64_t code = 0, code_end = 0;
    std::uint32_t site = 0;  // its first site, in kSites
    std::uint8_t n = 0;
    std::uint8_t at[kMaxInsns] = {};
    std::uint16_t out[kMaxInsns] = {};
    std::uint8_t orig[kMaxRegion] = {};
    bool live = false;  // its jump is in place (a hook or patch over it gives it up)
};

// Fixed once the rewrite is done (the fault paths read them without a lock);
// `live` changes only while patches and hooks go in, before the guest starts,
// or under g_write_mu for a plugin's late write.
std::vector<Region> g_regions;  // by lo, and so by code too
GuestMemory* g_mem = nullptr;
std::uint64_t g_code_lo = 0, g_code_hi = 0;    // the image's code (the fault path's range)
std::uint64_t g_arena_lo = 0, g_arena_hi = 0;  // the routines written

struct Cpu {
    bool bmi1 = false, lzcnt = false, movbe = false;
    bool on = false, forced = false;
};

const Cpu& cpu() {
    static const Cpu c = [] {
        Cpu r;
        unsigned a = 0, b = 0, cx = 0, d = 0;
        if (__get_cpuid(1, &a, &b, &cx, &d)) r.movbe = (cx >> 22) & 1;
        if (__get_cpuid_count(7, 0, &a, &b, &cx, &d)) r.bmi1 = (b >> 3) & 1;
        if (__get_cpuid(0x80000001u, &a, &b, &cx, &d)) r.lzcnt = (cx >> 5) & 1;
        r.on = !(r.bmi1 && r.lzcnt && r.movbe);
        if (const char* e = std::getenv("BBHOST_LEGACY_CPU"); e && (e[0] == '0' || e[0] == '1')) {
            r.on = e[0] == '1';
            r.forced = true;
        }
        return r;
    }();
    return c;
}

std::uint64_t xgetbv0() {
    std::uint32_t lo = 0, hi = 0;
    asm volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
}

// The region holding `va` (lo <= va < hi), or null. Regions do not overlap.
const Region* region_at(std::uint64_t va) {
    auto it = std::upper_bound(g_regions.begin(), g_regions.end(), va, [](std::uint64_t v, const Region& r) { return v < r.lo; });
    if (it == g_regions.begin()) return nullptr;
    --it;
    return va < it->hi ? &*it : nullptr;
}

// The region whose routine holds `pc`, or null.
const Region* routine_at(std::uint64_t pc) {
    if (pc < g_arena_lo || pc >= g_arena_hi) return nullptr;
    auto it = std::upper_bound(g_regions.begin(), g_regions.end(), pc, [](std::uint64_t v, const Region& r) { return v < r.code; });
    if (it == g_regions.begin()) return nullptr;
    --it;
    return pc < it->code_end ? &*it : nullptr;
}

// The region's instruction whose code holds `pc`.
int insn_of(const Region& r, std::uint64_t pc) {
    int k = 0;
    for (int j = 0; j < r.n; ++j)
        if (pc >= r.code + r.out[j]) k = j;
    return k;
}

std::uint64_t bn_of(std::uint64_t va) { return g_mem ? va - g_mem->slide + kPreferredGuestSlide : va; }

// Once a site: the fault paths log the first time each address comes by.
constexpr int kSeen = 512;
std::atomic<std::uint64_t> g_seen[kSeen];

bool first_time(std::uint64_t pc) {
    for (int i = 0; i < kSeen; ++i) {
        std::uint64_t have = g_seen[i].load(std::memory_order_relaxed);
        if (have == pc) return false;
        if (!have) {
            if (g_seen[i].compare_exchange_strong(have, pc)) return true;
            if (have == pc) return false;
        }
    }
    return false;  // the table is full: quiet from here
}

// The log line from a fault handler on a guest thread: with the host's FS
// when the guest's is in (BBHOST_TLS_GS=0), and back after. No stack
// protector here: FS changes under it.
template <typename... A>
__attribute__((no_stack_protector)) void fault_log(const char* fmt, A... a) {
    hle_fs_host();
    host_log(fmt, a...);
    hle_fs_guest();
}

// The instruction at a faulting pc, computed in its thread's registers
// (regs[16] by encoding, as both platforms' contexts can be read) and
// stepped over. False when it is not one of ours to take.
__attribute__((no_stack_protector)) bool emulate(std::uint64_t pc, std::uint64_t regs[16], std::uint64_t* flags,
                                                 std::uint64_t* next) {
    if (pc < g_code_lo || pc >= g_code_hi) return false;
    LegacyInsn in;
    const std::size_t avail = static_cast<std::size_t>(std::min<std::uint64_t>(15, g_code_hi - pc));
    if (!legacy_decode(reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(pc)), avail, &in)) return false;
    // LZCNT and TZCNT never fault here (they run as BSR and BSF).
    if (in.family == LegacyFamily::kLzcnt || in.family == LegacyFamily::kTzcnt) return false;
    legacy_apply(in, regs, flags, pc + in.len);
    *next = pc + in.len;
    if (first_time(pc)) {
#if defined(_WIN32)
        fault_log("legacy cpu: the game's %s at 0x%llx was computed by the illegal-instruction handler (not rewritten; on "
                  "Windows the fault's frame overwrote the 128 bytes under rsp there); once a site",
                  legacy_family_name(in.family), static_cast<unsigned long long>(bn_of(pc)));
#else
        fault_log("legacy cpu: the game's %s at 0x%llx was computed by the illegal-instruction handler (not rewritten); once "
                  "a site",
                  legacy_family_name(in.family), static_cast<unsigned long long>(bn_of(pc)));
#endif
    }
    return true;
}

// A trap on an int3 at `at`, past a region's jump: something branched into
// the region's bytes. When `at` starts one of its instructions, the thread
// goes on in that instruction's code in the routine.
__attribute__((no_stack_protector)) bool redirect(std::uint64_t at, std::uint64_t* next) {
    const Region* r = region_at(at);
    if (!r || !r->live || at < r->lo + 5) return false;
    for (int j = 0; j < r->n; ++j) {
        if (r->lo + r->at[j] != at) continue;
        *next = r->code + r->out[j];
        if (first_time(at))
            fault_log("legacy cpu: a branch landed on 0x%llx, inside the rewritten site 0x%llx's bytes - sent to its "
                      "routine; once a site",
                      static_cast<unsigned long long>(bn_of(at)), static_cast<unsigned long long>(bn_of(r->lo)));
        return true;
    }
    if (first_time(at))
        fault_log("legacy cpu: a branch landed inside the rewritten site 0x%llx's bytes, not on one of its instructions "
                  "(its jump's bytes ran as code up to the int3 at 0x%llx)",
                  static_cast<unsigned long long>(bn_of(r->lo)), static_cast<unsigned long long>(bn_of(at)));
    return false;
}

#if defined(_WIN32)
LONG CALLBACK legacy_veh(EXCEPTION_POINTERS* ep) {
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    CONTEXT* c = ep->ContextRecord;
    if (code == EXCEPTION_ILLEGAL_INSTRUCTION) {
        // CONTEXT keeps Rax..R15 in encoding order.
        std::uint64_t* regs = &c->Rax;
        std::uint64_t flags = c->EFlags, next = 0;
        if (!emulate(c->Rip, regs, &flags, &next)) return EXCEPTION_CONTINUE_SEARCH;
        c->EFlags = static_cast<DWORD>(flags);
        c->Rip = next;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (code == EXCEPTION_BREAKPOINT) {
        std::uint64_t next = 0;
        if (!redirect(reinterpret_cast<std::uint64_t>(ep->ExceptionRecord->ExceptionAddress), &next)) return EXCEPTION_CONTINUE_SEARCH;
        c->Rip = next;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#else
// ucontext's general registers in encoding order (rax, rcx, ... r15).
constexpr int kGreg[16] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
                           REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};
#endif

const char* missing_names(const Cpu& c, char* buf, std::size_t cap) {
    buf[0] = 0;
    const char* names[3] = {c.bmi1 ? nullptr : "BMI1", c.lzcnt ? nullptr : "LZCNT", c.movbe ? nullptr : "MOVBE"};
    int n = 0;
    for (const char* s : names) n += s != nullptr;
    int k = 0;
    for (const char* s : names) {
        if (!s) continue;
        std::strncat(buf, k == 0 ? "" : k + 1 == n ? " and " : ", ", cap - std::strlen(buf) - 1);
        std::strncat(buf, s, cap - std::strlen(buf) - 1);
        ++k;
    }
    return buf;
}

}  // namespace

std::string legacy_cpu_floor_missing() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return "CPUID's feature leaf";
    std::string miss;
    const auto add = [&](const char* s) {
        if (!miss.empty()) miss += ", ";
        miss += s;
    };
    if (!((c >> 20) & 1)) add("SSE4.2");
    if (!((c >> 23) & 1)) add("POPCNT");
    if (!((c >> 28) & 1)) {
        add("AVX");
    } else if (!((c >> 27) & 1) || (xgetbv0() & 6) != 6) {
        add("AVX turned on by the operating system");
    }
    return miss;
}

bool legacy_cpu_check_floor() {
    const std::string miss = legacy_cpu_floor_missing();
    if (miss.empty()) return true;
    const std::string text = "bbhost cannot run on this processor: it lacks " + miss +
                             ".\n\nThe game's code needs AVX, SSE4.2 and POPCNT: an Intel Core processor of the 2nd "
                             "generation (Sandy Bridge) or newer, or an AMD FX or newer, with an operating system that "
                             "turns AVX on.";
    host_log_decide(true);  // the lines so far (the version, the processor's name) written too
    host_log("cpu: %s", text.c_str());
#if defined(_WIN32)
    std::wstring w(text.begin(), text.end());  // the text is ASCII
    MessageBoxW(nullptr, w.c_str(), L"bbhost", MB_OK | MB_ICONERROR);
#endif
    std::fflush(nullptr);
    return false;
}

bool legacy_cpu_on() { return cpu().on; }

void legacy_cpu_log() {
    const Cpu& c = cpu();
    char miss[64];
    missing_names(c, miss, sizeof miss);
    const bool lacks = !(c.bmi1 && c.lzcnt && c.movbe);
    if (!lacks && !c.forced) return;
    if (c.on && lacks) {
        host_log("cpu: no %s on this processor: the game's %zu instructions that use them (and TZCNT) run as routines "
                 "of bbhost's, written in at load (BBHOST_LEGACY_CPU=0 leaves them)",
                 miss, kSiteCount);
    } else if (c.on) {
        host_log("cpu: BBHOST_LEGACY_CPU=1: the game's %zu BMI1, LZCNT, TZCNT and MOVBE instructions run as routines of "
                 "bbhost's although this processor has them",
                 kSiteCount);
    } else if (lacks) {
        host_log("cpu: no %s on this processor, and BBHOST_LEGACY_CPU=0 leaves the game's instructions as they are: BMI1 "
                 "and MOVBE ones go through the illegal-instruction handler, LZCNT and TZCNT give BSR's and BSF's answers "
                 "(wrong ones)",
                 miss);
    } else {
        host_log("cpu: BBHOST_LEGACY_CPU=0: the game's instructions as they are (this processor has BMI1, LZCNT and MOVBE)");
    }
}

std::size_t legacy_cpu_arena_size() { return legacy_cpu_on() ? kArena : 0; }

void legacy_cpu_rewrite(ElfImage* image) {
    g_mem = &image->mem;
    for (const ElfSegment& s : image->segments) {
        if (s.flags & 1) {
            g_code_lo = image->mem.slide + s.va;
            g_code_hi = g_code_lo + s.filesz;
        }
    }
    if (!legacy_cpu_on()) return;
    if (image->sha256 != kEboot109Sha256) {
        host_log("legacy cpu: the eboot is not 1.09, whose sites are the ones known: nothing rewritten (BMI1 and MOVBE go "
                 "through the illegal-instruction handler, LZCNT and TZCNT give wrong answers)");
        return;
    }
    if (image->arena_size < kArena) {
        host_log("legacy cpu: no room was reserved for the routines; nothing rewritten");
        return;
    }
    GuestMemory& mem = image->mem;
    const std::uint64_t arena = mem.slide + image->arena_va;
    auto* out = static_cast<std::uint8_t*>(guest_ptr(mem, arena));
    std::size_t used = 0;
    std::vector<Region> regions;
    regions.reserve(kSiteCount);
    std::uint64_t covered = 0;
    std::size_t sites = 0, taken = 0, skipped = 0;
    std::size_t by_family[8] = {};
    for (std::size_t i = 0; i < kSiteCount; ++i) {
        const LegacySite& site = kSites[i];
        const std::uint64_t at = mem.slide + (site.at - kPreferredGuestSlide);
        if (at < covered) continue;  // inside the region before (decoded there)
        Region r;
        r.lo = at - site.before;
        r.hi = at + site.len + site.after;
        r.site = static_cast<std::uint32_t>(i);
        const std::size_t n = static_cast<std::size_t>(r.hi - r.lo);
        const auto* p = static_cast<const std::uint8_t*>(guest_ptr(mem, r.lo));
        bool ok = n <= kMaxRegion && r.lo >= g_code_lo && r.hi <= g_code_hi;
        // Whole instructions, the site where the table says it is and of its
        // family; the others movable (no call, no loop).
        std::size_t off = 0;
        bool site_seen = false;
        std::size_t family_sites = 0;
        while (ok && off < n) {
            LegacyInsn li;
            std::size_t len = 0;
            if (legacy_decode(p + off, n - off, &li)) {
                len = li.len;
                ++family_sites;
                if (off == site.before) site_seen = li.family == site.family && li.len == site.len;
            } else {
                const X86Insn x = x86_decode(p + off, n - off);
                len = x.len;
                if (!len || x.kind == X86Insn::Call || x.kind == X86Insn::Loop) ok = false;
                if (x.rel_at) {
                    // A branch among them must not land inside the region (its
                    // start is the jump to the routine: that is fine).
                    std::int64_t rel = 0;
                    if (x.rel_size == 1) {
                        rel = static_cast<std::int8_t>(p[off + x.rel_at]);
                    } else {
                        std::int32_t r32;
                        std::memcpy(&r32, p + off + x.rel_at, 4);
                        rel = r32;
                    }
                    const std::uint64_t t = r.lo + off + len + static_cast<std::uint64_t>(rel);
                    if (t > r.lo && t < r.hi) ok = false;
                }
            }
            if (!ok || r.n == kMaxInsns) {
                ok = false;
                break;
            }
            r.at[r.n++] = static_cast<std::uint8_t>(off);
            off += len;
        }
        ok = ok && off == n && site_seen;
        // The routine: each instruction in turn - a site as its routine, the
        // others moved - then the jump back.
        std::size_t k = 0;
        const std::uint64_t code = arena + used;
        for (int j = 0; ok && j < r.n; ++j) {
            const std::size_t len = (j + 1 < r.n ? r.at[j + 1] : n) - r.at[j];
            r.out[j] = static_cast<std::uint16_t>(k);
            const std::size_t w =
                legacy_relocate(p + r.at[j], len, r.lo + r.at[j], out + used + k, code + k, kArena - used - k);
            if (!w) ok = false;
            k += w;
        }
        if (ok && used + k + 5 <= kArena) {
            std::uint8_t* j = out + used + k;
            j[0] = 0xe9;
            const auto rel = static_cast<std::int32_t>(static_cast<std::int64_t>(r.hi - (code + k + 5)));
            std::memcpy(j + 1, &rel, 4);
            k += 5;
        } else {
            ok = false;
        }
        if (!ok) {
            ++skipped;
            host_log("legacy cpu: %s at 0x%x left as it is: its bytes are not the ones the table was made from",
                     legacy_family_name(site.family), site.at);
            continue;
        }
        std::memcpy(r.orig, p, n);
        r.code = code;
        r.code_end = code + k;
        used = (used + k + 15) & ~std::size_t{15};
        sites += family_sites;
        if (site.before || site.after) ++taken;
        by_family[static_cast<int>(site.family)] += family_sites;
        regions.push_back(r);
        covered = r.hi;
    }
    // The jumps, written over the whole code at once while it is writable.
    const std::uint64_t lo = g_code_lo & ~0xfffull, hi = (g_code_hi + 0xfff) & ~0xfffull;
    if (!guest_protect_rw(&mem, lo, hi - lo)) {
        host_log("legacy cpu: the code cannot be made writable; nothing rewritten");
        return;
    }
    for (Region& r : regions) {
        auto* p = static_cast<std::uint8_t*>(guest_ptr(mem, r.lo));
        const auto rel = static_cast<std::int32_t>(static_cast<std::int64_t>(r.code - (r.lo + 5)));
        p[0] = 0xe9;
        std::memcpy(p + 1, &rel, 4);
        std::memset(p + 5, 0xcc, static_cast<std::size_t>(r.hi - r.lo) - 5);
        r.live = true;
    }
    guest_protect_rx(&mem, lo, hi - lo);
    guest_protect_rx(&mem, arena, kArena);
    g_regions = std::move(regions);
    g_arena_lo = arena;
    g_arena_hi = arena + used;
    host_log("legacy cpu: %zu of the game's %zu sites rewritten (ANDN %zu, BEXTR %zu, BLSI %zu, BLSR %zu, LZCNT %zu, TZCNT %zu, "
             "MOVBE %zu) in %zu jumps, %zu of them short sites' with the instructions next to them; routines %zu KiB at "
             "0x%llx%s",
             sites, kSiteCount, by_family[0], by_family[1], by_family[2], by_family[4], by_family[5], by_family[6], by_family[7],
             g_regions.size(), taken, used >> 10, static_cast<unsigned long long>(arena),
             skipped ? "; some left as they are (above)" : "");
}

void legacy_cpu_verify(const char* when) {
    if (g_regions.empty()) return;
    std::size_t live = 0, changed = 0;
    for (const Region& r : g_regions) {
        if (!r.live) continue;
        ++live;
        const auto* p = static_cast<const std::uint8_t*>(guest_ptr(*g_mem, r.lo));
        const auto rel = static_cast<std::int32_t>(static_cast<std::int64_t>(r.code - (r.lo + 5)));
        bool same = p[0] == 0xe9 && std::memcmp(p + 1, &rel, 4) == 0;
        for (std::uint64_t a = r.lo + 5; same && a < r.hi; ++a) same = p[a - r.lo] == 0xcc;
        if (same) continue;
        if (++changed <= 20) {
            host_log("legacy cpu: the jump at 0x%llx (the game's %s) was written over (%s): a patch or hook wrote into a "
                     "rewritten site without legacy_cpu_release",
                     static_cast<unsigned long long>(bn_of(r.lo)), legacy_family_name(kSites[r.site].family), when);
        }
    }
    host_log("legacy cpu: %zu of %zu jumps in place %s%s", live - changed, g_regions.size(), when,
             live < g_regions.size() ? " (the rest given up to hooks and patches over them)" : "");
}

void legacy_cpu_original(std::uint64_t va, std::size_t n, std::uint8_t* out) {
    std::memcpy(out, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), n);  // guest addresses are the host's
    if (g_regions.empty() || va >= g_code_hi || va + n <= g_code_lo) return;
    auto it = std::upper_bound(g_regions.begin(), g_regions.end(), va, [](std::uint64_t v, const Region& r) { return v < r.lo; });
    if (it != g_regions.begin()) --it;
    for (; it != g_regions.end() && it->lo < va + n; ++it) {
        if (!it->live || it->hi <= va) continue;
        const std::uint64_t a = std::max(va, it->lo), b = std::min(va + n, it->hi);
        std::memcpy(out + (a - va), it->orig + (a - it->lo), static_cast<std::size_t>(b - a));
    }
}

int legacy_cpu_meets(std::uint64_t va, std::size_t n) {
    if (g_regions.empty()) return 0;
    auto it = std::upper_bound(g_regions.begin(), g_regions.end(), va, [](std::uint64_t v, const Region& r) { return v < r.lo; });
    if (it != g_regions.begin()) --it;
    int meets = 0;
    for (; it != g_regions.end() && it->lo < va + n; ++it) {
        if (!it->live || it->hi <= va) continue;
        if (it->lo < va || it->hi > va + n) return -1;
        meets = 1;
    }
    return meets;
}

void legacy_cpu_release(std::uint64_t va, std::size_t n, const char* who) {
    if (g_regions.empty()) return;
    auto it = std::upper_bound(g_regions.begin(), g_regions.end(), va, [](std::uint64_t v, const Region& r) { return v < r.lo; });
    if (it != g_regions.begin()) --it;
    for (; it != g_regions.end() && it->lo < va + n; ++it) {
        if (!it->live || it->hi <= va || it->lo < va || it->hi > va + n) continue;
        it->live = false;
        host_log("legacy cpu: the rewritten %s at 0x%llx given up to %s, which takes its place (and copies it as a routine)",
                 legacy_family_name(kSites[it->site].family), static_cast<unsigned long long>(bn_of(it->lo)), who);
    }
}

#if defined(_WIN32)
bool legacy_cpu_on_sigill(void*) { return false; }
bool legacy_cpu_on_sigtrap(void*) { return false; }

void legacy_cpu_install_handler() {
    static PVOID h = AddVectoredExceptionHandler(1, legacy_veh);
    (void)h;
}
#else
__attribute__((no_stack_protector)) bool legacy_cpu_on_sigill(void* ctx) {
    auto* uc = static_cast<ucontext_t*>(ctx);
    greg_t* g = uc->uc_mcontext.gregs;
    std::uint64_t regs[16];
    for (int i = 0; i < 16; ++i) regs[i] = static_cast<std::uint64_t>(g[kGreg[i]]);
    std::uint64_t flags = static_cast<std::uint64_t>(g[REG_EFL]), next = 0;
    if (!emulate(static_cast<std::uint64_t>(g[REG_RIP]), regs, &flags, &next)) return false;
    for (int i = 0; i < 16; ++i) g[kGreg[i]] = static_cast<greg_t>(regs[i]);
    g[REG_EFL] = static_cast<greg_t>(flags);
    g[REG_RIP] = static_cast<greg_t>(next);
    return true;
}

__attribute__((no_stack_protector)) bool legacy_cpu_on_sigtrap(void* ctx) {
    auto* uc = static_cast<ucontext_t*>(ctx);
    std::uint64_t next = 0;
    // The trap reports the address after the int3.
    if (!redirect(static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RIP]) - 1, &next)) return false;
    uc->uc_mcontext.gregs[REG_RIP] = static_cast<greg_t>(next);
    return true;
}

void legacy_cpu_install_handler() {}
#endif

std::size_t legacy_cpu_regions(LegacyCpuRegion* out, std::size_t cap) {
    std::size_t n = 0;
    for (const Region& r : g_regions) {
        if (n < cap) out[n] = {r.lo, r.hi, r.code, r.code_end, r.orig, r.live};
        ++n;
    }
    return n;
}

std::uint64_t legacy_cpu_guest_pc(std::uint64_t pc) {
    const Region* r = routine_at(pc);
    return r ? r->lo + r->at[insn_of(*r, pc)] : pc;
}

void legacy_cpu_crash_line(std::uint64_t pc) {
    const Region* r = routine_at(pc);
    if (!r) return;
    const int j = insn_of(*r, pc);
    LegacyInsn li;
    const bool site = legacy_decode(r->orig + r->at[j], static_cast<std::size_t>(r->hi - r->lo) - r->at[j], &li);
    host_log("  pc %llx is in the legacy-cpu routine of the game's %s at 0x%llx (+%llu into its code; the site's region "
             "0x%llx..0x%llx)",
             static_cast<unsigned long long>(pc), site ? legacy_family_name(li.family) : "instruction taken along with a site",
             static_cast<unsigned long long>(bn_of(r->lo + r->at[j])),
             static_cast<unsigned long long>(pc - (r->code + r->out[j])), static_cast<unsigned long long>(bn_of(r->lo)),
             static_cast<unsigned long long>(bn_of(r->hi)));
}
