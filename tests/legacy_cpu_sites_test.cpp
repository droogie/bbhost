// legacy_cpu_sites_test: every site of the 1.09 eboot, rewritten as bbhost
// does it (core/legacy_cpu.h), against the game's own bytes. Skips (77)
// without the eboot.
//
// The image is loaded with bbhost's loader (tests/decomp/eboot_kit.h) and
// legacy_cpu_rewrite runs on it, BBHOST_LEGACY_CPU=1. Then each rewritten
// region - one instruction, or a short one with its neighbours - runs from
// random states three ways, which must agree on every general and xmm
// register, the flags (the ones the region's last instruction defines), the
// memory its operands reach and the stack from 128 bytes under rsp up:
//   rewritten  - the region's jump in the image, its routine in the arena,
//                and the jump back (caught after the region);
//   stepped    - the region's own bytes, moved into the test, with each site
//                an int3 whose trap computes it with legacy_apply;
//   native     - the region's own bytes as they are, where the processor has
//                the three (not under qemu-x86_64 -cpu IvyBridge).
// The memory operands are aimed at the test's buffers: base registers at
// one, index registers small; a rip-relative one reads the image.
//
// Then the illegal-instruction path: a sample of BMI1 and MOVBE sites given
// back their bytes and run in place - computed by legacy_cpu_on_sigill where
// the processor faults on them (under qemu), run natively here - against
// the stepped run.
#include "legacy_cpu_harness.h"

#include "core/elf.h"
#include "core/legacy_cpu.h"
#include "core/legacy_cpu_code.h"
#include "decomp/eboot_kit.h"
#include "decomp/insn.h"
#include "guest_abi.h"
#include "log.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include <sys/mman.h>
#include <ucontext.h>

// The fault path's log switches FS on a guest thread (core/thunk.h); this
// test's thread has the host's FS throughout.
extern "C" GUEST_ABI void hle_fs_host() {}
extern "C" GUEST_ABI void hle_fs_guest() {}

using namespace harness;

namespace {

constexpr const char* kTest = "legacy_cpu_sites_test";
constexpr int kRsp = 4;
constexpr std::size_t kBlock = 1u << 21;
constexpr std::size_t kStepped = 0, kNative = 0x8000, kRewritten = 0x10000, kData = 0x40000, kStack = 0x80000;
constexpr std::size_t kDataSize = 0x10000, kStackSize = 0x4000;
constexpr std::uint64_t kStackRsp = kStack + 0x2000;
constexpr std::size_t kMaxRegionBytes = 32;

std::uint8_t* g_block = nullptr;
Slots g_slots;
std::uint64_t at(std::size_t off) { return reinterpret_cast<std::uint64_t>(g_block + off); }

const int kGreg[16] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
                       REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};

// The stepped run's sites: an int3's address (after it) -> the instruction
// and where it ended in the image (for a rip-relative operand).
struct Step {
    LegacyInsn in;
    std::uint64_t end_va;
};
std::map<std::uint64_t, Step> g_steps;

void on_trap(int, siginfo_t*, void* ctx) {
    auto* uc = static_cast<ucontext_t*>(ctx);
    greg_t* g = uc->uc_mcontext.gregs;
    if (legacy_cpu_on_sigtrap(ctx)) return;
    const auto it = g_steps.find(static_cast<std::uint64_t>(g[REG_RIP]));
    if (it == g_steps.end()) {
        std::printf("%s: an unexpected trap at %llx\n", kTest, static_cast<unsigned long long>(g[REG_RIP]));
        std::_Exit(1);
    }
    std::uint64_t regs[16];
    for (int i = 0; i < 16; ++i) regs[i] = static_cast<std::uint64_t>(g[kGreg[i]]);
    std::uint64_t flags = static_cast<std::uint64_t>(g[REG_EFL]);
    legacy_apply(it->second.in, regs, &flags, it->second.end_va);
    for (int i = 0; i < 16; ++i) g[kGreg[i]] = static_cast<greg_t>(regs[i]);
    g[REG_EFL] = static_cast<greg_t>(flags);
}

void on_segv(int sig, siginfo_t* info, void* ctx) {
    auto* uc = static_cast<ucontext_t*>(ctx);
    std::printf("%s: signal %d at pc %llx, address %llx, rsp %llx\n", kTest, sig,
                static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RIP]),
                reinterpret_cast<unsigned long long>(info->si_addr), static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RSP]));
    std::fflush(stdout);
    std::_Exit(128 + sig);
}

std::size_t g_ills = 0;

void on_ill(int sig, siginfo_t*, void* ctx) {
    if (legacy_cpu_on_sigill(ctx)) {
        ++g_ills;
        return;
    }
    auto* uc = static_cast<ucontext_t*>(ctx);
    std::printf("%s: an illegal instruction at %llx not taken by the fallback\n", kTest,
                static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RIP]));
    std::_Exit(128 + sig);
}

// A memory operand of an instruction that is not a site, from its ModRM.
bool operand(const std::uint8_t* p, const X86Insn& x, LegacyMem* m) {
    if (!x.modrm_at) return false;
    const std::uint8_t modrm = p[x.modrm_at];
    const int mod = modrm >> 6, low = modrm & 7;
    if (mod == 3) return false;
    const int b = x.rex & 1, xx = (x.rex >> 1) & 1;
    std::size_t i = x.modrm_at + 1u;
    bool disp32 = mod == 2;
    if (low == 4) {
        const std::uint8_t sib = p[i++];
        const int idx = ((sib >> 3) & 7) | (xx << 3);
        m->index = static_cast<std::int8_t>(idx == 4 ? LegacyMem::kNone : idx);
        m->scale = static_cast<std::uint8_t>(1u << (sib >> 6));
        if ((sib & 7) == 5 && mod == 0) {
            m->base = LegacyMem::kNone;
            disp32 = true;
        } else {
            m->base = static_cast<std::int8_t>((sib & 7) | (b << 3));
        }
    } else if (low == 5 && mod == 0) {
        m->base = LegacyMem::kRip;
        disp32 = true;
    } else {
        m->base = static_cast<std::int8_t>(low | (b << 3));
    }
    if (mod == 1) {
        m->disp = static_cast<std::int8_t>(p[i]);
    } else if (disp32) {
        std::memcpy(&m->disp, p + i, 4);
    }
    return true;
}

// Whether an instruction that is not a site writes the flags: the ALU
// group, test, inc/dec, the shifts. (The regions' other instructions - mov,
// lea, movzx, vmovd, ret - leave them.)
bool writes_flags(const std::uint8_t* p, std::size_t n) {
    std::size_t i = 0;
    while (i < n && (p[i] == 0x66 || p[i] == 0xf2 || p[i] == 0xf3 || (p[i] & 0xf0) == 0x40)) ++i;
    if (i >= n) return false;
    const std::uint8_t op = p[i];
    if (op == 0x0f || op == 0xc4 || op == 0xc5) return false;  // the two-byte and VEX maps: movzx, vmovd
    return op < 0x40 || (op >= 0x80 && op <= 0x85) || op == 0xa8 || op == 0xa9 || (op >= 0xc0 && op <= 0xc1) ||
           (op >= 0xd0 && op <= 0xd3) || op == 0xf6 || op == 0xf7 || op == 0xfe || op == 0xff;
}

struct Insn {
    std::uint64_t va;  // in the image
    std::size_t len;
    bool site;
    LegacyInsn in;
    bool mem;
    LegacyMem m;
    bool lea;
};

std::vector<Insn> decode_region(const LegacyCpuRegion& r) {
    std::vector<Insn> out;
    const std::size_t n = static_cast<std::size_t>(r.hi - r.lo);
    for (std::size_t off = 0; off < n;) {
        Insn i{};
        i.va = r.lo + off;
        if (legacy_decode(r.orig + off, n - off, &i.in)) {
            i.site = true;
            i.len = i.in.len;
            i.mem = i.in.mem;
            i.m = i.in.m;
        } else {
            const X86Insn x = x86_decode(r.orig + off, n - off);
            if (!x.len) return {};
            i.len = x.len;
            i.mem = operand(r.orig + off, x, &i.m);
            std::size_t o = 0;
            while (o < x.len && (r.orig[off + o] & 0xf0) == 0x40) ++o;
            i.lea = r.orig[off + o] == 0x8d;
        }
        out.push_back(i);
        off += i.len;
    }
    return out;
}

struct Outcome {
    State st;
    std::vector<std::uint8_t> data, stack;
};

void load_mem(const std::vector<std::uint8_t>& data, const std::vector<std::uint8_t>& stack) {
    std::memcpy(g_block + kData, data.data(), kDataSize);
    std::memcpy(g_block + kStack, stack.data(), kStackSize);
}

Outcome take() {
    Outcome o;
    o.st = g_slots.out;
    o.data.assign(g_block + kData, g_block + kData + kDataSize);
    o.stack.assign(g_block + kStack, g_block + kStack + kStackSize);
    return o;
}

std::string differ(const Outcome& a, const Outcome& b, std::uint32_t flags_mask) {
    char buf[160];
    for (int r = 0; r < 16; ++r) {
        if (a.st.r[r] != b.st.r[r]) {
            std::snprintf(buf, sizeof buf, "register %d: %llx vs %llx", r, static_cast<unsigned long long>(a.st.r[r]),
                          static_cast<unsigned long long>(b.st.r[r]));
            return buf;
        }
    }
    if (std::memcmp(a.st.x, b.st.x, sizeof a.st.x)) return "xmm";
    if ((a.st.flags ^ b.st.flags) & flags_mask) {
        std::snprintf(buf, sizeof buf, "flags %llx vs %llx", static_cast<unsigned long long>(a.st.flags),
                      static_cast<unsigned long long>(b.st.flags));
        return buf;
    }
    if (a.data != b.data) return "memory";
    const std::size_t from = kStackRsp - kStack - 128;
    if (std::memcmp(a.stack.data() + from, b.stack.data() + from, kStackSize - from)) return "stack";
    return "";
}

// 14 bytes: jmp [rip+0]; dq target.
void far_jump(std::uint8_t* p, std::uint64_t target) {
    p[0] = 0xff;
    p[1] = 0x25;
    std::memset(p + 2, 0, 4);
    std::memcpy(p + 6, &target, 8);
}

}  // namespace

int main() {
    host_log_decide(true);    // the rewrite's own lines in the test's output
    eboot_kit::bytes(kTest);  // exits 77 without the eboot
    setenv("BBHOST_LEGACY_CPU", "1", 1);
    eboot_kit::reserve_arena(legacy_cpu_arena_size());
    eboot_kit::load(kTest);
    ElfImage* image = eboot_kit::image();
    legacy_cpu_rewrite(image);
    std::vector<LegacyCpuRegion> regions(8192);
    const std::size_t count = legacy_cpu_regions(regions.data(), regions.size());
    regions.resize(count);
    if (count < 6000) {
        std::printf("%s: FAIL: only %zu regions rewritten\n", kTest, count);
        return 1;
    }
    // Within a rel32 of the image (at 0x400000): a rip-relative operand moved
    // into the test still reaches what it read. (Asked for by address: qemu's
    // user mode does not keep MAP_32BIT.)
    // A pc inside a routine is the game instruction's it stands for, in the
    // crash report and the census: every region's code maps back to its own
    // instructions.
    for (const LegacyCpuRegion& r : regions) {
        for (std::uint64_t pc = r.code; pc < r.code_end; ++pc) {
            const std::uint64_t g = legacy_cpu_guest_pc(pc);
            if (g < r.lo || g >= r.hi) {
                std::printf("%s: FAIL: routine pc %llx maps to %llx, outside its region %llx..%llx\n", kTest,
                            static_cast<unsigned long long>(pc), static_cast<unsigned long long>(g),
                            static_cast<unsigned long long>(r.lo), static_cast<unsigned long long>(r.hi));
                return 1;
            }
        }
        if (legacy_cpu_guest_pc(r.lo) != r.lo) {
            std::printf("%s: FAIL: an image pc does not map to itself\n", kTest);
            return 1;
        }
    }
    legacy_cpu_crash_line(regions[0].code + 3);  // the crash report's line, shown once
    g_block = nullptr;
    for (std::uint64_t a = 0x20000000; !g_block && a < 0x70000000; a += 0x1000000) {
        void* p = mmap(reinterpret_cast<void*>(a), kBlock, PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == reinterpret_cast<void*>(a)) {
            g_block = static_cast<std::uint8_t*>(p);
        } else if (p != MAP_FAILED) {
            munmap(p, kBlock);
        }
    }
    if (!g_block) {
        std::printf("%s: no memory within reach of the image\n", kTest);
        return 1;
    }
    const Features feat = features();
    {
        struct sigaction sa {};
        sa.sa_sigaction = on_trap;
        sa.sa_flags = SA_SIGINFO;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGTRAP, &sa, nullptr);
        sa.sa_sigaction = on_ill;
        sigaction(SIGILL, &sa, nullptr);
        sa.sa_sigaction = on_segv;
        sigaction(SIGSEGV, &sa, nullptr);
        sigaction(SIGBUS, &sa, nullptr);
    }
    // The image's code writable: the catch after a region is written there.
    for (const ElfSegment& s : image->segments)
        if (s.flags & 1)
            mprotect(reinterpret_cast<void*>(image->mem.slide + s.va), (s.memsz + 0xfff) & ~0xfffull, PROT_READ | PROT_WRITE | PROT_EXEC);

    Rng rng{0x2545f4914f6cdd1dull};
    std::size_t runs = 0, native_runs = 0, failures = 0, multi = 0, sites = 0, fallback_regions = 0;
    const int kInputs = std::getenv("LEGACY_CPU_SITES_INPUTS") ? std::atoi(std::getenv("LEGACY_CPU_SITES_INPUTS")) : 16;
    std::vector<std::uint8_t> data(kDataSize), stack(kStackSize);
    // fallback: the region's own bytes put back in the image, run in place.
    auto test_region = [&](const LegacyCpuRegion& r, bool fallback) {
        const std::vector<Insn> insns = decode_region(r);
        if (insns.empty()) {
            std::printf("%s: FAIL: region %llx does not decode\n", kTest, static_cast<unsigned long long>(r.lo));
            ++failures;
            return;
        }
        if (insns.size() > 1 && !fallback) ++multi;
        bool has_native_site = true, ret = false;
        for (const Insn& i : insns) {
            if (i.site) {
                if (!fallback) ++sites;
                const LegacyFamily f = i.in.family;
                if (f == LegacyFamily::kMovbe ? !feat.movbe : f == LegacyFamily::kLzcnt ? !feat.lzcnt : !feat.bmi1)
                    has_native_site = false;
            }
            ret = ret || (!i.site && r.orig[i.va - r.lo] == 0xc3);
        }
        // The flags compared: what the last instruction to write them defines
        // (a site its family's; another arithmetic one all of them).
        std::uint32_t mask = kArith;
        for (const Insn& i : insns) {
            if (i.site) {
                if (i.in.family != LegacyFamily::kMovbe) mask = legacy_defined_flags(i.in);
            } else if (writes_flags(r.orig + (i.va - r.lo), i.len)) {
                mask = kArith & ~kAF;
            }
        }
        // The harnesses: stepped, native, rewritten.
        std::uint8_t* hs = prologue(g_block + kStepped, &g_slots);
        g_steps.clear();
        std::uint8_t* p = hs;
        bool ok = true;
        for (const Insn& i : insns) {
            if (i.site) {
                *p++ = 0xcc;
                g_steps[reinterpret_cast<std::uint64_t>(p)] = {i.in, i.va + i.len};
            } else {
                const std::size_t k = x86_relocate(r.orig + (i.va - r.lo), i.len, i.va, p, reinterpret_cast<std::uint64_t>(p), 64);
                if (!k) ok = false;
                p += k;
            }
        }
        epilogue(p, &g_slots);
        std::uint8_t* hn = prologue(g_block + kNative, &g_slots);
        const std::size_t kn = x86_relocate(r.orig, static_cast<std::size_t>(r.hi - r.lo), r.lo, hn, reinterpret_cast<std::uint64_t>(hn), 256);
        if (!kn) ok = false;
        epilogue(hn + kn, &g_slots);
        std::uint8_t* hr = prologue(g_block + kRewritten, &g_slots);
        far_jump(hr, r.lo);
        std::uint8_t* catcher = hr + 14;
        epilogue(catcher, &g_slots);
        if (!ok) {
            std::printf("%s: FAIL: region %llx cannot be moved into the test\n", kTest, static_cast<unsigned long long>(r.lo));
            ++failures;
            return;
        }
        if (std::getenv("LEGACY_CPU_SITES_TRACE")) std::fprintf(stderr, "region %llx\n", static_cast<unsigned long long>(r.lo));
        // The catch after the region, in the image (put back after).
        auto* after = reinterpret_cast<std::uint8_t*>(r.hi);
        std::uint8_t saved[14];
        std::memcpy(saved, after, 14);
        far_jump(after, reinterpret_cast<std::uint64_t>(catcher));
        auto* region = reinterpret_cast<std::uint8_t*>(r.lo);
        std::uint8_t jump[kMaxRegionBytes];
        const std::size_t rn = static_cast<std::size_t>(r.hi - r.lo);
        std::memcpy(jump, region, rn);
        if (fallback) std::memcpy(region, r.orig, rn);
        for (int k = 0; k < kInputs; ++k) {
            State in{};
            for (auto& v : in.r) v = rng.next();
            for (auto& x : in.x)
                for (auto& b : x) b = static_cast<std::uint8_t>(rng.next());
            in.flags = rng.next() & kArith;
            in.r[kRsp] = at(kStackRsp);
            for (auto& b : data) b = static_cast<std::uint8_t>(rng.next());
            for (auto& b : stack) b = static_cast<std::uint8_t>(rng.next());
            // Edges for the sites' sources, now and then.
            static const std::uint64_t kEdges[] = {0, 1, ~0ull, 0x8000000000000000ull, 0x80000000ull, 0xffffffffull};
            for (const Insn& i : insns)
                if (i.site && !i.in.mem && k % 3 == 0) in.r[i.in.rm] = kEdges[rng.below(6)];
            // Aim the memory operands: index registers small, bases into the data.
            for (const Insn& i : insns)
                if (i.mem && !i.lea && i.m.index >= 0) in.r[i.m.index] = rng.below(4);
            bool aimed = true;
            std::uint64_t set = 0;
            for (const Insn& i : insns) {
                if (!i.mem || i.lea || i.m.base < 0 || i.m.base == LegacyMem::kRip || i.m.base == kRsp) continue;
                if (!(set & (1ull << i.m.base))) {
                    in.r[i.m.base] = at(kData) + 0x8000 + rng.below(0x100) - static_cast<std::uint64_t>(static_cast<std::int64_t>(i.m.disp));
                    set |= 1ull << i.m.base;
                }
            }
            // Every operand's address inside the test's memory (or the image's, rip-relative).
            for (const Insn& i : insns) {
                if (!i.mem || i.lea) continue;
                LegacyInsn li{};
                li.m = i.m;
                const std::uint64_t ea = legacy_address(li, in.r, i.va + i.len);
                const bool in_data = ea >= at(kData) && ea + 16 <= at(kData) + kDataSize;
                const bool in_stack = ea >= at(kStack) && ea + 16 <= at(kStack) + kStackSize;
                const bool in_image = i.m.base == LegacyMem::kRip;
                if (!in_data && !in_stack && !in_image) aimed = false;
            }
            if (!aimed) {
                std::printf("%s: FAIL: region %llx's operands cannot be aimed\n", kTest, static_cast<unsigned long long>(r.lo));
                ++failures;
                break;
            }
            if (ret) {
                const std::uint64_t back = reinterpret_cast<std::uint64_t>(catcher);
                std::memcpy(stack.data() + (kStackRsp - kStack), &back, 8);  // a ret in the region returns to the catch
            }
            g_slots.in = in;
            load_mem(data, stack);
            if (ret) {
                const std::uint64_t back = reinterpret_cast<std::uint64_t>(p);
                std::memcpy(g_block + kStackRsp, &back, 8);
            }
            run(g_block + kStepped, &g_slots);
            const Outcome stepped = take();
            g_slots.in = in;
            load_mem(data, stack);
            run(g_block + kRewritten, &g_slots);
            Outcome rewritten = take();
            ++runs;
            std::string d = differ(rewritten, stepped, mask);
            if (ret) {
                // The stepped run's ret came back to its own catch, the rewritten
                // one's to the rewritten catch: the word they returned to differs.
                const std::size_t w = kStackRsp - kStack;
                std::memcpy(rewritten.stack.data() + w, stepped.stack.data() + w, 8);
                d = differ(rewritten, stepped, mask);
            }
            if (!d.empty()) {
                std::printf("%s: FAIL: region %llx (%zu instructions) rewritten vs stepped: %s\n", kTest,
                            static_cast<unsigned long long>(r.lo), insns.size(), d.c_str());
                ++failures;
                break;
            }
            if (has_native_site && !ret) {
                g_slots.in = in;
                load_mem(data, stack);
                run(g_block + kNative, &g_slots);
                const Outcome native = take();
                ++native_runs;
                d = differ(native, stepped, mask);
                if (!d.empty()) {
                    std::printf("%s: FAIL: region %llx native vs stepped: %s\n", kTest, static_cast<unsigned long long>(r.lo),
                                d.c_str());
                    ++failures;
                    break;
                }
            }
        }
        std::memcpy(after, saved, 14);
        std::memcpy(region, jump, rn);
        if (fallback) ++fallback_regions;
    };
    for (const LegacyCpuRegion& r : regions) test_region(r, false);
    // The illegal-instruction path: every 8th single BMI1 or MOVBE site (LZCNT
    // and TZCNT do not fault) in its own bytes.
    std::size_t k = 0;
    for (const LegacyCpuRegion& r : regions) {
        LegacyInsn li;
        const std::size_t rn = static_cast<std::size_t>(r.hi - r.lo);
        if (!legacy_decode(r.orig, rn, &li) || li.len != rn || li.family == LegacyFamily::kLzcnt ||
            li.family == LegacyFamily::kTzcnt || k++ % 8)
            continue;
        test_region(r, true);
    }
    std::printf("%s: the illegal-instruction path: %zu sites run in their own bytes, %zu illegal instructions computed by it\n",
                kTest, fallback_regions, g_ills);
    std::printf("%s: %zu regions (%zu of more than one instruction), %zu sites; %zu runs rewritten against stepped, %zu "
                "against the processor's own%s; %zu failures\n",
                kTest, count, multi, sites, runs, native_runs,
                (feat.bmi1 && feat.lzcnt && feat.movbe) ? "" : " (it lacks BMI1, LZCNT or MOVBE: those regions not run natively)",
                failures);
    return failures ? 1 : 0;
}
