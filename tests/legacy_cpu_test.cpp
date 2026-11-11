// legacy_cpu_test: every routine core/legacy_cpu_code.h writes, against the
// instruction it stands for. No game files.
//
// Each case is one instruction - a family (ANDN, BEXTR, BLSI, BLSMSK, BLSR,
// LZCNT, TZCNT, MOVBE load and store), an operand size, a form (registers;
// memory through every base, index, scale and displacement size, rsp-based
// ones reaching above rsp and into its red zone, rip-relative, no base) and
// its registers, aliased every way (the destination as a source, as the
// address's base or index, rcx and the scratch registers as operands). The
// test encodes it on its own, checks that legacy_decode reads it back, and
// runs it three ways from the same state - every general register, the flags,
// the memory it reaches and the stack around rsp: the instruction itself
// (where the processor has it), its routine, and legacy_apply (the
// illegal-instruction path's arithmetic). All registers, rsp included, the
// flags the manual defines, the memory and the stack from 128 bytes under rsp
// up must agree. The inputs are the edges (zero, one, the top bit, all ones;
// BEXTR's start and length at 0, 1, the size, past it, 255) and random ones.
//
// Then mutants: each representative routine is changed one instruction at a
// time - a condition reversed, an stc made clc, an immediate off by one, an
// instruction turned to nops - and every mutant must fail against the same
// inputs, so the comparison is shown to see what a wrong routine does.
//
// Without BMI1, LZCNT or MOVBE (qemu-x86_64 -cpu IvyBridge; or
// LEGACY_CPU_TEST_NO_NATIVE=1) the routine is compared with legacy_apply
// alone: the run itself shows the routines need none of the three.
// LEGACY_CPU_TEST_SCALE=N runs N times the random cases (default 1).
#include "legacy_cpu_harness.h"

#include "core/legacy_cpu_code.h"
#include "decomp/insn.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace harness;

namespace {

constexpr int kRsp = 4;

// ---- An encoder of its own for the instructions under test ----------------

struct Spec {
    LegacyFamily family = LegacyFamily::kAndn;
    unsigned size = 4;
    int dst = 0;   // the register written (a store's source)
    int vex = 0;   // ANDN's inverted operand, BEXTR's control
    bool store = false;
    bool mem = false;
    int rm = 0;
    int base = -1;   // -1 none, 16 rip
    int index = -1;  // -1 none
    int scale = 1;
    std::int32_t disp = 0;
    int disp_size = 0;  // 0, 1, 4 (as encoded)
};

bool needs_vex(LegacyFamily f) { return f != LegacyFamily::kLzcnt && f != LegacyFamily::kTzcnt && f != LegacyFamily::kMovbe; }

// ModRM, SIB and displacement, and the REX/VEX extension bits (r, x, b).
struct ModRm {
    std::vector<std::uint8_t> bytes;
    int r = 0, x = 0, b = 0;
    int disp_at = -1;  // where a rip-relative disp32 goes
};

ModRm modrm(const Spec& s, int reg) {
    ModRm m;
    m.r = (reg >> 3) & 1;
    if (!s.mem) {
        m.b = (s.rm >> 3) & 1;
        m.bytes.push_back(static_cast<std::uint8_t>(0xc0 | ((reg & 7) << 3) | (s.rm & 7)));
        return m;
    }
    if (s.base == 16) {
        m.bytes.push_back(static_cast<std::uint8_t>(((reg & 7) << 3) | 5));
        m.disp_at = 1;
        for (int i = 0; i < 4; ++i) m.bytes.push_back(0);
        return m;
    }
    const bool sib = s.index >= 0 || s.base < 0 || (s.base & 7) == 4;
    int mod = s.disp_size == 0 ? 0 : s.disp_size == 1 ? 1 : 2;
    if (s.base < 0) mod = 0;
    m.bytes.push_back(static_cast<std::uint8_t>((mod << 6) | ((reg & 7) << 3) | (sib ? 4 : (s.base & 7))));
    if (sib) {
        const int ss = s.scale == 8 ? 3 : s.scale == 4 ? 2 : s.scale == 2 ? 1 : 0;
        const int idx = s.index < 0 ? 4 : (s.index & 7);
        const int base = s.base < 0 ? 5 : (s.base & 7);
        m.bytes.push_back(static_cast<std::uint8_t>((ss << 6) | (idx << 3) | base));
        if (s.index >= 8) m.x = 1;
    }
    if (s.base >= 8) m.b = 1;
    if (mod == 1) m.bytes.push_back(static_cast<std::uint8_t>(s.disp));
    if (mod == 2 || s.base < 0) {
        for (int i = 0; i < 4; ++i) m.bytes.push_back(static_cast<std::uint8_t>(static_cast<std::uint32_t>(s.disp) >> (8 * i)));
    }
    return m;
}

// The instruction's bytes; for a rip operand, its disp32 aimed at `target`
// from `at` (where the bytes will run).
std::vector<std::uint8_t> encode(const Spec& s, std::uint64_t at = 0, std::uint64_t target = 0) {
    std::vector<std::uint8_t> out;
    int reg = s.dst;
    if (s.family == LegacyFamily::kBlsr) reg = 1;
    if (s.family == LegacyFamily::kBlsmsk) reg = 2;
    if (s.family == LegacyFamily::kBlsi) reg = 3;
    const ModRm m = modrm(s, reg);
    if (needs_vex(s.family)) {
        const bool bls = reg != s.dst || s.family == LegacyFamily::kBlsr || s.family == LegacyFamily::kBlsmsk ||
                         s.family == LegacyFamily::kBlsi;
        const int vvvv = bls ? s.dst : s.vex;
        const std::uint8_t op = s.family == LegacyFamily::kAndn ? 0xf2 : s.family == LegacyFamily::kBextr ? 0xf7 : 0xf3;
        out.push_back(0xc4);
        out.push_back(static_cast<std::uint8_t>(((!m.r) << 7) | ((!m.x) << 6) | ((!m.b) << 5) | 0x02));
        out.push_back(static_cast<std::uint8_t>(((s.size == 8) << 7) | ((~vvvv & 15) << 3)));
        out.push_back(op);
    } else {
        if (s.size == 2) out.push_back(0x66);
        if (s.family != LegacyFamily::kMovbe) out.push_back(0xf3);
        const int rex = 0x40 | ((s.size == 8) << 3) | (m.r << 2) | (m.x << 1) | m.b;
        if (rex != 0x40) out.push_back(static_cast<std::uint8_t>(rex));
        out.push_back(0x0f);
        if (s.family == LegacyFamily::kMovbe) {
            out.push_back(0x38);
            out.push_back(s.store ? 0xf1 : 0xf0);
        } else {
            out.push_back(s.family == LegacyFamily::kLzcnt ? 0xbd : 0xbc);
        }
    }
    const std::size_t modrm_at = out.size();
    out.insert(out.end(), m.bytes.begin(), m.bytes.end());
    if (m.disp_at >= 0) {
        const std::int64_t d = static_cast<std::int64_t>(target - (at + out.size()));
        const auto d32 = static_cast<std::int32_t>(d);
        std::memcpy(out.data() + modrm_at + m.disp_at, &d32, 4);
    }
    return out;
}

// ---- The test's machine -----------------------------------------------------

constexpr std::size_t kBlock = 1u << 22;
constexpr std::size_t kNative = 0, kRoutine = 0x10000, kRipData = 0x40000, kData = 0x80000,
                      kStack = 0x100000;
constexpr std::size_t kDataSize = 0x400, kStackSize = 0xc00, kStackTop = kStack + kStackSize;
constexpr std::uint64_t kStackRsp = kStack + 0x400;  // 2 KiB above for the caller's frame, 1 KiB below

struct Machine {
    std::uint8_t* block = nullptr;
    Slots slots{};
    Features feat;
    std::uint64_t at(std::size_t off) const { return reinterpret_cast<std::uint64_t>(block + off); }
};

struct Outcome {
    State st;
    std::uint8_t data[kDataSize];
    std::uint8_t rip[256];
    std::uint8_t stack[kStackSize];
};

struct Input {
    State st;
    std::uint8_t data[kDataSize];
    std::uint8_t rip[256];
    std::uint8_t stack[kStackSize];
};

void load(Machine& m, const Input& in) {
    std::memcpy(m.block + kData, in.data, kDataSize);
    std::memcpy(m.block + kRipData, in.rip, sizeof(in.rip));
    std::memcpy(m.block + kStack, in.stack, kStackSize);
    m.slots.in = in.st;
}

void save(Machine& m, Outcome* o, const State& st) {
    o->st = st;
    std::memcpy(o->data, m.block + kData, kDataSize);
    std::memcpy(o->rip, m.block + kRipData, sizeof(o->rip));
    std::memcpy(o->stack, m.block + kStack, kStackSize);
}

// Differences between two outcomes, as text ("" when they agree): registers,
// the defined flags, memory, and the stack from the red zone up.
std::string differ(const Outcome& a, const Outcome& b, std::uint32_t defined) {
    static const char* kNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                     "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
    char buf[256];
    for (int r = 0; r < 16; ++r) {
        if (a.st.r[r] != b.st.r[r]) {
            std::snprintf(buf, sizeof buf, "%s %llx vs %llx", kNames[r], static_cast<unsigned long long>(a.st.r[r]),
                          static_cast<unsigned long long>(b.st.r[r]));
            return buf;
        }
    }
    if (std::memcmp(a.st.x, b.st.x, sizeof(a.st.x))) return "xmm";
    if ((a.st.flags ^ b.st.flags) & defined) {
        std::snprintf(buf, sizeof buf, "flags %llx vs %llx (defined %x)", static_cast<unsigned long long>(a.st.flags),
                      static_cast<unsigned long long>(b.st.flags), defined);
        return buf;
    }
    if (std::memcmp(a.data, b.data, kDataSize)) return "memory";
    if (std::memcmp(a.rip, b.rip, sizeof(a.rip))) return "rip data";
    const std::size_t from = kStackRsp - kStack - 128;
    if (std::memcmp(a.stack + from, b.stack + from, kStackSize - from)) return "stack (red zone or above)";
    return "";
}

// ---- Inputs -----------------------------------------------------------------

const std::uint64_t kEdges[] = {0,
                                1,
                                2,
                                3,
                                0x80,
                                0xff,
                                0x8000,
                                0xffff,
                                0x7fffffffull,
                                0x80000000ull,
                                0xffffffffull,
                                0x100000000ull,
                                0x7fffffffffffffffull,
                                0x8000000000000000ull,
                                ~0ull,
                                0x0123456789abcdefull,
                                0xfedcba9876543210ull,
                                0x8000000000000001ull,
                                0x0000000100000001ull};
const unsigned kBits[] = {0, 1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 127, 128, 200, 255};

// The value an operand takes in input i: the edges first, then random ones
// (dense, sparse, one bit).
std::uint64_t value(Rng& rng, int i, int which) {
    const int n = static_cast<int>(sizeof(kEdges) / sizeof(kEdges[0]));
    const int k = (i + which * 7) % (n + 6);
    if (k < n) return kEdges[k];
    switch (k - n) {
    case 0: return rng.next();
    case 1: return rng.next() & rng.next();
    case 2: return 1ull << rng.below(64);
    case 3: return rng.next() & 0xffffffffull;
    case 4: return (rng.next() | 1) << rng.below(63);
    default: return rng.next() >> rng.below(64);
    }
}

std::uint64_t control(Rng& rng, int i) {
    const int n = static_cast<int>(sizeof(kBits) / sizeof(kBits[0]));
    if (i < n * n) {
        const std::uint64_t start = kBits[i % n], len = kBits[i / n];
        return start | (len << 8) | ((i & 1) ? (rng.next() << 16) : 0);
    }
    return rng.next();
}

// Registers aimed so that the memory operand reaches `target`.
bool aim(const Spec& s, std::uint64_t target, State* st, Rng& rng) {
    if (s.base == 16) return true;
    const std::int64_t d = s.disp;
    if (s.base == kRsp) {
        if (s.index >= 0) {
            // rsp + index*scale + disp: the index takes the rest.
            const std::int64_t rest = static_cast<std::int64_t>(target) - static_cast<std::int64_t>(st->r[kRsp]) - d;
            if (rest % s.scale) return false;
            st->r[s.index] = static_cast<std::uint64_t>(rest / s.scale);
        }
        return true;
    }
    if (s.base < 0) {
        const std::int64_t rest = static_cast<std::int64_t>(target) - d;
        if (rest % s.scale) return false;
        st->r[s.index] = static_cast<std::uint64_t>(rest / s.scale);
        return true;
    }
    if (s.index < 0) {
        st->r[s.base] = target - static_cast<std::uint64_t>(d);
        return true;
    }
    if (s.index == s.base) {
        const std::int64_t rest = static_cast<std::int64_t>(target) - d;
        if (rest % (1 + s.scale)) return false;
        st->r[s.base] = static_cast<std::uint64_t>(rest / (1 + s.scale));
        return true;
    }
    const std::uint64_t idx = rng.below(64);
    st->r[s.index] = idx;
    st->r[s.base] = target - static_cast<std::uint64_t>(d) - idx * static_cast<std::uint64_t>(s.scale);
    return true;
}

// ---- One case ---------------------------------------------------------------

struct Totals {
    std::uint64_t cases = 0, runs = 0, native = 0, failures = 0, mutants = 0, killed = 0;
    std::size_t routine_bytes_max = 0;
} g_tot;

bool native_ok(const Machine& m, LegacyFamily f) {
    if (f == LegacyFamily::kMovbe) return m.feat.movbe;
    if (f == LegacyFamily::kLzcnt) return m.feat.lzcnt;
    return m.feat.bmi1;
}

// Builds a harness at `off` around `code` (n bytes).
std::uint8_t* harness_at(Machine& m, std::size_t off, const std::uint8_t* code, std::size_t n) {
    std::uint8_t* p = prologue(m.block + off, &m.slots);
    std::memcpy(p, code, n);
    epilogue(p + n, &m.slots);
    return m.block + off;
}

// Where the code under test starts in a harness at `off`.
std::uint64_t code_va(Machine& m, std::size_t off) {
    std::uint8_t* p = prologue(m.block + off, &m.slots);
    return reinterpret_cast<std::uint64_t>(p);
}

// Inputs for a case: state, memory, stack.
std::vector<Input> inputs(Machine& m, const Spec& s, Rng& rng, int count) {
    std::vector<Input> out;
    const bool bextr = s.family == LegacyFamily::kBextr;
    for (int i = 0; i < count; ++i) {
        Input in;
        for (auto& r : in.st.r) r = rng.next();
        for (auto& x : in.st.x)
            for (auto& b : x) b = static_cast<std::uint8_t>(rng.next());
        in.st.flags = rng.next() & kArith;
        for (auto& b : in.data) b = static_cast<std::uint8_t>(rng.next());
        for (auto& b : in.rip) b = static_cast<std::uint8_t>(rng.next());
        for (auto& b : in.stack) b = static_cast<std::uint8_t>(rng.next());
        in.st.r[kRsp] = m.at(kStackRsp);
        // The destination's old value in a pattern of its own (a routine that
        // leaves it, or writes half of it, shows), then the operands: edges
        // first. BEXTR's value is dense across its grid of starts and lengths.
        static const std::uint64_t kOld[] = {0, ~0ull, 0x5555555555555555ull, 0xaaaaaaaaaaaaaaaaull, 0xfffffffffffffffeull};
        in.st.r[s.dst] = (i / 3) % 6 < 5 ? kOld[(i / 3) % 6] : rng.next();
        // BEXTR's grid of starts and lengths runs twice: on all ones (every
        // bit the mask keeps shows) and on dense random values.
        const int grid = static_cast<int>(sizeof(kBits) / sizeof(kBits[0]) * sizeof(kBits) / sizeof(kBits[0]));
        // A case with fewer inputs takes grid points spread over it, on all
        // ones, random and edge values in turn.
        const bool full = count >= 2 * grid;
        const std::uint64_t v = !bextr ? value(rng, i, 0)
                                : full  ? (i < grid ? ~0ull : rng.next())
                                        : (i % 3 == 0 ? ~0ull : i % 3 == 1 ? rng.next() : value(rng, i, 0));
        if (needs_vex(s.family) && s.family != LegacyFamily::kBlsi && s.family != LegacyFamily::kBlsmsk &&
            s.family != LegacyFamily::kBlsr) {
            in.st.r[s.vex] = !bextr ? value(rng, i, 1)
                             : full  ? control(rng, i % grid + (i >= 2 * grid ? 2 * grid : 0))
                                     : control(rng, i < 20 ? (i * 37 + static_cast<int>(rng.below(grid))) % grid : grid + i);
        }
        if (s.store) in.st.r[s.dst] = v;
        std::uint64_t target = 0;
        if (s.mem) {
            if (s.base == 16) {
                target = m.at(kRipData) + 64;  // where encode() aimed it
            } else if (s.base == kRsp) {
                target = in.st.r[kRsp] + static_cast<std::uint64_t>(static_cast<std::int64_t>(s.disp));
            } else {
                target = m.at(kData) + 256 + rng.below(kDataSize - 512);
            }
            if (s.base == kRsp && s.index >= 0) target = m.at(kStackRsp) + rng.below(512) - 128;
            if (!aim(s, target, &in.st, rng)) {
                // An alignment the scale cannot reach: move the target.
                bool ok = false;
                for (int t = 1; t < 16 && !ok; ++t) ok = aim(s, target + static_cast<std::uint64_t>(t), &in.st, rng);
                if (!ok) continue;
            }
            // The memory operand's value, at the effective address.
            std::uint64_t regs[16];
            std::memcpy(regs, in.st.r, sizeof regs);
            const std::uint64_t ea = s.base == 16 ? target
                                                  : (s.base >= 0 ? regs[s.base] : 0) + (s.index >= 0 ? regs[s.index] * s.scale : 0) +
                                                        static_cast<std::uint64_t>(static_cast<std::int64_t>(s.disp));
            if (!s.store) {
                std::uint8_t* base = nullptr;
                std::size_t off = 0;
                if (ea >= m.at(kData) && ea + 8 <= m.at(kData) + kDataSize) {
                    base = in.data;
                    off = ea - m.at(kData);
                } else if (ea >= m.at(kRipData) && ea + 8 <= m.at(kRipData) + sizeof(in.rip)) {
                    base = in.rip;
                    off = ea - m.at(kRipData);
                } else if (ea >= m.at(kStack) && ea + 8 <= m.at(kStackTop)) {
                    base = in.stack;
                    off = ea - m.at(kStack);
                } else {
                    std::printf("FAIL: an input's address %llx is outside the test's memory\n", static_cast<unsigned long long>(ea));
                    ++g_tot.failures;
                    continue;
                }
                std::memcpy(base + off, &v, s.size);
            }
        } else {
            in.st.r[s.rm] = v;
        }
        out.push_back(in);
    }
    return out;
}

// The reference: legacy_apply on the input's state, in the machine's memory.
void reference(Machine& m, const LegacyInsn& li, const Input& in, std::uint64_t end_va, Outcome* o) {
    load(m, in);
    State st = in.st;
    std::uint64_t flags = st.flags;
    legacy_apply(li, st.r, &flags, end_va);
    st.flags = flags & kArith;
    save(m, o, st);
}

void run_code(Machine& m, std::uint8_t* code, const Input& in, Outcome* o) {
    load(m, in);
    run(code, &m.slots);
    save(m, o, m.slots.out);
}

std::string describe(const Spec& s) {
    static const char* kNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                     "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
    char buf[200];
    std::string m;
    if (s.mem) {
        char mb[96];
        std::snprintf(mb, sizeof mb, "[%s%s%s*%d%+d/%d]", s.base == 16 ? "rip" : s.base < 0 ? "" : kNames[s.base],
                      s.index >= 0 ? "+" : "", s.index >= 0 ? kNames[s.index] : "", s.scale, s.disp, s.disp_size);
        m = mb;
    } else {
        m = kNames[s.rm];
    }
    std::snprintf(buf, sizeof buf, "%s%s size %u dst %s vex %s %s", legacy_family_name(s.family), s.store ? " store" : "", s.size,
                  kNames[s.dst], kNames[s.vex], m.c_str());
    return buf;
}

// One case: encode, decode, emit, run every input three ways. Returns false on
// a failure (printed). `mutate`: also the mutants.
bool test_case(Machine& m, const Spec& s, Rng& rng, int count, bool mutate) {
    ++g_tot.cases;
    const std::uint64_t native_code = code_va(m, kNative);
    const std::uint64_t rip_target = m.at(kRipData) + 64;
    const std::vector<std::uint8_t> bytes = encode(s, native_code, rip_target);
    LegacyInsn li;
    if (!legacy_decode(bytes.data(), bytes.size(), &li) || li.len != bytes.size() || li.family != s.family ||
        li.size != s.size || li.dst != s.dst || li.mem != s.mem || li.store != s.store ||
        (!s.mem && li.rm != s.rm) || (s.mem && (li.m.base != (s.base == 16 ? LegacyMem::kRip : s.base) || li.m.index != s.index ||
                                                (s.index >= 0 && li.m.scale != s.scale)))) {
        std::printf("FAIL decode: %s\n", describe(s).c_str());
        ++g_tot.failures;
        return false;
    }
    if (needs_vex(s.family) && s.family != LegacyFamily::kBlsi && s.family != LegacyFamily::kBlsmsk &&
        s.family != LegacyFamily::kBlsr && li.vex != s.vex) {
        std::printf("FAIL decode (vex): %s\n", describe(s).c_str());
        ++g_tot.failures;
        return false;
    }
    const std::uint64_t end_va = native_code + bytes.size();
    std::uint8_t routine[512];
    const std::uint64_t routine_va = code_va(m, kRoutine);
    const std::size_t n = legacy_emit(li, end_va, routine, routine_va, sizeof routine);
    if (!n) {
        std::printf("FAIL emit: %s\n", describe(s).c_str());
        ++g_tot.failures;
        return false;
    }
    if (n > g_tot.routine_bytes_max) g_tot.routine_bytes_max = n;
    std::uint8_t* hn = harness_at(m, kNative, bytes.data(), bytes.size());
    std::uint8_t* hr = harness_at(m, kRoutine, routine, n);
    const bool native = native_ok(m, s.family);
    const std::uint32_t defined = legacy_defined_flags(li);
    const std::vector<Input> ins = inputs(m, s, rng, count);
    std::vector<Outcome> refs(ins.size());
    bool ok = true;
    for (std::size_t i = 0; i < ins.size() && ok; ++i) {
        Outcome o_ref, o_rt, o_nat;
        reference(m, li, ins[i], end_va, &o_ref);
        run_code(m, hr, ins[i], &o_rt);
        ++g_tot.runs;
        std::string d = differ(o_rt, o_ref, defined);
        if (!d.empty()) {
            std::printf("FAIL routine vs reference: %s: %s\n", describe(s).c_str(), d.c_str());
            ok = false;
        }
        if (native) {
            run_code(m, hn, ins[i], &o_nat);
            ++g_tot.native;
            d = differ(o_nat, o_rt, defined);
            if (!d.empty() && ok) {
                std::printf("FAIL routine vs native: %s: %s\n", describe(s).c_str(), d.c_str());
                ok = false;
            }
            d = differ(o_nat, o_ref, defined);
            if (!d.empty() && ok) {
                std::printf("FAIL reference vs native: %s: %s\n", describe(s).c_str(), d.c_str());
                ok = false;
            }
            refs[i] = o_nat;
        } else {
            refs[i] = o_ref;
        }
    }
    if (!ok) {
        ++g_tot.failures;
        return false;
    }
    if (!mutate) return true;
    // Mutants: one change at a time; each must fail on some input.
    std::vector<std::pair<std::size_t, std::string>> insns;  // offset, kind
    for (std::size_t off = 0; off < n;) {
        const X86Insn x = x86_decode(routine + off, n - off);
        if (!x.len) {
            std::printf("FAIL: the routine does not decode at +%zu: %s\n", off, describe(s).c_str());
            ++g_tot.failures;
            return false;
        }
        insns.push_back({off, ""});
        off += x.len;
    }
    int cmps = 0;
    for (std::size_t k = 0; k < insns.size(); ++k) {
        const std::size_t off = insns[k].first;
        const X86Insn x = x86_decode(routine + off, n - off);
        std::vector<std::vector<std::uint8_t>> variants;
        auto with = [&](auto change) {
            std::vector<std::uint8_t> v(routine, routine + n);
            change(v.data() + off);
            variants.push_back(v);
        };
        with([&](std::uint8_t* p) { std::memset(p, 0x90, x.len); });  // gone
        if (x.kind == X86Insn::Jcc && x.rel_size == 1) with([&](std::uint8_t* p) { p[0] ^= 1; });
        if (routine[off] == 0xf9) with([&](std::uint8_t* p) { p[0] = 0xf8; });
        // An imm8 at the end (83 /x ib, c1 /x ib): off by one either way.
        const std::uint8_t* q = routine + off;
        std::size_t o = 0;
        while (o < x.len && (q[o] == 0x66 || (q[o] & 0xf0) == 0x40)) ++o;
        if (o < x.len && (q[o] == 0x83 || q[o] == 0xc1)) {
            // One is equivalent and left out: BEXTR's `cmp ecx, bits; jae keep`
            // (the length past which nothing is masked) made bits+1 - a length
            // of exactly the size then shifts by 0, which keeps the value too.
            const bool length_check = s.family == LegacyFamily::kBextr && q[o] == 0x83 && q[o + 1] == 0xf9 &&
                                      ++cmps == 2;
            if (!length_check) with([&](std::uint8_t* p) { ++p[x.len - 1]; });
            with([&](std::uint8_t* p) { --p[x.len - 1]; });
        }
        for (auto& v : variants) {
            ++g_tot.mutants;
            // In the routine's own place: a rip-relative operand stays aimed.
            std::uint8_t* hm = harness_at(m, kRoutine, v.data(), v.size());
            bool killed = false;
            for (std::size_t i = 0; i < ins.size() && !killed; ++i) {
                // A mutant may fault or run away: only ones that stay inside
                // their routine are made (no jumps out), and a dropped push or
                // lea leaves rsp wrong - which the comparison catches.
                Outcome o_m;
                run_code(m, hm, ins[i], &o_m);
                killed = !differ(o_m, refs[i], defined).empty();
            }
            if (killed) {
                ++g_tot.killed;
            } else {
                std::printf("FAIL: a mutant lived (instruction %zu of %s, bytes at +%zu):", k, describe(s).c_str(), off);
                for (std::size_t b = 0; b < v.size(); ++b) std::printf("%s%02x", b == off ? " [" : " ", v[b]);
                std::printf("\n");
                ++g_tot.failures;
                ok = false;
            }
        }
    }
    return ok;
}

// ---- The cases ----------------------------------------------------------------

std::vector<int> regs_but_rsp() {
    std::vector<int> r;
    for (int i = 0; i < 16; ++i)
        if (i != kRsp) r.push_back(i);
    return r;
}

}  // namespace

// LEGACY_CPU_TEST_BENCH=1: what a routine costs against the instruction -
// eight sites in a loop, each its own instruction, or its own jump to its own
// routine and back as the rewrite makes them - in ns an execution. Prints a
// table; checks nothing.
int bench(Machine& m) {
    struct Case {
        const char* name;
        Spec s;
    };
    std::vector<Case> cases;
    auto add = [&](const char* name, LegacyFamily f, unsigned size, int dst, int vex, int rm, bool mem = false, bool store = false) {
        Spec s;
        s.family = f;
        s.size = size;
        s.dst = dst;
        s.vex = vex;
        s.rm = rm;
        s.mem = mem;
        s.store = store;
        if (mem) {
            s.base = 3;  // rbx: the buffer
            s.disp = 8;
            s.disp_size = 1;
        }
        cases.push_back({name, s});
    };
    add("ANDN r32", LegacyFamily::kAndn, 4, 0, 1, 2);
    add("ANDN r64, m64", LegacyFamily::kAndn, 8, 0, 1, 0, true);
    add("BEXTR r32", LegacyFamily::kBextr, 4, 0, 1, 2);
    add("BEXTR r64, m64", LegacyFamily::kBextr, 8, 0, 1, 0, true);
    add("BLSI r64", LegacyFamily::kBlsi, 8, 0, 0, 2);
    add("BLSR r64", LegacyFamily::kBlsr, 8, 0, 0, 2);
    add("LZCNT r32", LegacyFamily::kLzcnt, 4, 0, 0, 2);
    add("TZCNT r64", LegacyFamily::kTzcnt, 8, 0, 0, 2);
    add("MOVBE r32, m32", LegacyFamily::kMovbe, 4, 0, 0, 0, true);
    add("MOVBE m32, r32", LegacyFamily::kMovbe, 4, 2, 0, 0, true, true);
    add("MOVBE m64, r64", LegacyFamily::kMovbe, 8, 2, 0, 0, true, true);
    constexpr int kUnroll = 8;
    const long iters = std::getenv("LEGACY_CPU_TEST_BENCH_ITERS") ? std::atol(std::getenv("LEGACY_CPU_TEST_BENCH_ITERS")) : 2000000;
    alignas(64) static std::uint8_t buffer[256];
    std::printf("legacy_cpu_test: ns an execution, %d sites in a loop of %ld (native where the processor has it)\n", kUnroll, iters);
    for (const Case& c : cases) {
        double ns[2] = {-1, -1};
        for (int routine = 0; routine < 2; ++routine) {
            if (!routine && !native_ok(m, c.s.family)) continue;
            std::uint8_t* code = m.block + kRoutine;
            std::uint8_t* p = code;
            // push rbx rbp r12-r15; rbx = buffer; rcx = rdi (count); values in rdx, rsi
            const std::uint8_t head[] = {0x53, 0x55, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57};
            std::memcpy(p, head, sizeof head);
            p += sizeof head;
            *p++ = 0x48;
            *p++ = 0xbb;  // movabs rbx, buffer
            const std::uint64_t buf = reinterpret_cast<std::uint64_t>(buffer);
            std::memcpy(p, &buf, 8);
            p += 8;
            const std::uint8_t setup[] = {0x49, 0x89, 0xff,                                // mov r15, rdi
                                          0x48, 0xba, 0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00,  // movabs rdx, 0x12345678
                                          0xb9, 0x04, 0x08, 0x00, 0x00};                    // mov ecx, 0x0804
            std::memcpy(p, setup, sizeof setup);
            p += sizeof setup;
            std::uint8_t* loop = p;
            std::uint8_t* slots[kUnroll];
            for (int k = 0; k < kUnroll; ++k) {
                slots[k] = p;
                // The instruction (or its jump, filled in below, padded to the instruction's length).
                const std::vector<std::uint8_t> bytes = encode(c.s, reinterpret_cast<std::uint64_t>(p), 0);
                std::memcpy(p, bytes.data(), bytes.size());
                p += std::max<std::size_t>(bytes.size(), 5);
            }
            // dec r15; jnz loop
            *p++ = 0x49;
            *p++ = 0xff;
            *p++ = 0xcf;
            *p++ = 0x0f;
            *p++ = 0x85;
            const std::int32_t back = static_cast<std::int32_t>(loop - (p + 4));
            std::memcpy(p, &back, 4);
            p += 4;
            const std::uint8_t tail[] = {0x41, 0x5f, 0x41, 0x5e, 0x41, 0x5d, 0x41, 0x5c, 0x5d, 0x5b, 0xc3};
            std::memcpy(p, tail, sizeof tail);
            p += sizeof tail;
            if (routine) {
                p = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(p) + 15) & ~std::uintptr_t{15});
                for (int k = 0; k < kUnroll; ++k) {
                    const std::vector<std::uint8_t> bytes = encode(c.s, reinterpret_cast<std::uint64_t>(slots[k]), 0);
                    LegacyInsn li;
                    legacy_decode(bytes.data(), bytes.size(), &li);
                    const std::size_t slot = std::max<std::size_t>(bytes.size(), 5);
                    std::uint8_t* r = p;
                    const std::size_t n = legacy_emit(li, reinterpret_cast<std::uint64_t>(slots[k]) + bytes.size(), r,
                                                      reinterpret_cast<std::uint64_t>(r), 256);
                    p += n;
                    *p++ = 0xe9;  // back after the site
                    const std::int32_t rel = static_cast<std::int32_t>((slots[k] + slot) - (p + 4));
                    std::memcpy(p, &rel, 4);
                    p += 4;
                    p = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(p) + 15) & ~std::uintptr_t{15});
                    slots[k][0] = 0xe9;  // the site's jump
                    const std::int32_t to = static_cast<std::int32_t>(r - (slots[k] + 5));
                    std::memcpy(slots[k] + 1, &to, 4);
                    for (std::size_t b = 5; b < slot; ++b) slots[k][b] = 0xcc;
                }
            } else {
                for (int k = 0; k < kUnroll; ++k) {
                    const std::vector<std::uint8_t> bytes = encode(c.s, reinterpret_cast<std::uint64_t>(slots[k]), 0);
                    for (std::size_t b = bytes.size(); b < 5; ++b) slots[k][b] = 0x90;  // a short one's pad
                }
            }
            using Fn = void (*)(long);
            const auto fn = reinterpret_cast<Fn>(code);
            fn(iters / 10);  // warm
            const auto t0 = std::chrono::steady_clock::now();
            fn(iters);
            const auto t1 = std::chrono::steady_clock::now();
            ns[routine] = std::chrono::duration<double, std::nano>(t1 - t0).count() / (static_cast<double>(iters) * kUnroll);
        }
        if (ns[0] >= 0) {
            std::printf("  %-28s native %6.2f ns   routine %6.2f ns   (x%.1f)\n", c.name, ns[0], ns[1], ns[1] / ns[0]);
        } else {
            std::printf("  %-28s native   -        routine %6.2f ns\n", c.name, ns[1]);
        }
    }
    return 0;
}

int main() {
    Machine m;
    m.block = alloc_exec(kBlock);
    if (!m.block) {
        std::printf("no executable memory\n");
        return 1;
    }
    m.feat = features();
    if (const char* e = std::getenv("LEGACY_CPU_TEST_BENCH"); e && e[0] == '1') return bench(m);
    std::printf("legacy_cpu_test: native BMI1 %s, LZCNT %s, MOVBE %s%s\n", m.feat.bmi1 ? "yes" : "no", m.feat.lzcnt ? "yes" : "no",
                m.feat.movbe ? "yes" : "no", (m.feat.bmi1 && m.feat.lzcnt && m.feat.movbe) ? "" : " - those compared with legacy_apply only");
    int scale = 1;
    if (const char* e = std::getenv("LEGACY_CPU_TEST_SCALE")) scale = std::max(1, std::atoi(e));
    Rng rng{0x9e3779b97f4a7c15ull};
    const std::vector<int> regs = regs_but_rsp();
    const LegacyFamily kVexFamilies[] = {LegacyFamily::kAndn, LegacyFamily::kBextr};
    const LegacyFamily kBls[] = {LegacyFamily::kBlsi, LegacyFamily::kBlsmsk, LegacyFamily::kBlsr};
    const LegacyFamily kCounts[] = {LegacyFamily::kLzcnt, LegacyFamily::kTzcnt};
    std::set<std::string> mutated;  // a family, size and form mutated once each
    auto key = [](const Spec& s) {
        char b[64];
        std::snprintf(b, sizeof b, "%d/%u/%d/%d/%d", static_cast<int>(s.family), s.size, s.mem, s.store,
                      s.mem ? (s.base == 4 ? 1 : s.base == 16 ? 2 : 0) : 0);
        return std::string(b);
    };
    const int bextr_inputs = 2 * static_cast<int>(sizeof(kBits) / sizeof(kBits[0]) * sizeof(kBits) / sizeof(kBits[0])) + 8;
    // Mutants are made for the first case of each kind whose operands are
    // distinct registers: one register in two roles ties two inputs together
    // (BEXTR's value and control), and the edges then do not all occur.
    auto distinct = [](const Spec& s) {
        unsigned seen = 1u << kRsp;
        auto add = [&](int r) {
            if (r < 0 || r >= 16) return true;
            if (seen & (1u << r)) return false;
            seen |= 1u << r;
            return true;
        };
        const bool vex = needs_vex(s.family) && s.family != LegacyFamily::kBlsi && s.family != LegacyFamily::kBlsmsk &&
                         s.family != LegacyFamily::kBlsr;
        return add(s.dst) && (!vex || add(s.vex)) && (s.mem ? (add(s.base) && add(s.index)) : add(s.rm)) && add(1);
    };
    auto one = [&](const Spec& s) {
        const bool mut = distinct(s) && mutated.insert(key(s)).second;
        const int count = s.family == LegacyFamily::kBextr ? (mut ? bextr_inputs : 24) : (mut ? 48 : 24);
        test_case(m, s, rng, count, mut);
    };

    // Registers: every destination, source and VEX operand, every size.
    for (LegacyFamily f : kVexFamilies)
        for (unsigned size : {4u, 8u})
            for (int d : regs)
                for (int v : regs)
                    for (int r : regs) {
                        Spec s;
                        s.family = f;
                        s.size = size;
                        s.dst = d;
                        s.vex = v;
                        s.rm = r;
                        one(s);
                    }
    for (LegacyFamily f : kBls)
        for (unsigned size : {4u, 8u})
            for (int d : regs)
                for (int r : regs) {
                    Spec s;
                    s.family = f;
                    s.size = size;
                    s.dst = d;
                    s.rm = r;
                    one(s);
                }
    for (LegacyFamily f : kCounts)
        for (unsigned size : {2u, 4u, 8u})
            for (int d : regs)
                for (int r : regs) {
                    Spec s;
                    s.family = f;
                    s.size = size;
                    s.dst = d;
                    s.rm = r;
                    one(s);
                }

    // Memory: every base (rsp, rbp, r12, r13 included; none; rip) with every
    // index, and the destination and VEX operand on each of them in turn.
    const int kScales[] = {1, 2, 4, 8};
    std::vector<int> bases = regs;
    bases.push_back(kRsp);
    bases.push_back(-1);
    bases.push_back(16);
    std::vector<int> indexes = regs;
    indexes.push_back(-1);
    const LegacyFamily kAll[] = {LegacyFamily::kAndn,  LegacyFamily::kBextr, LegacyFamily::kBlsi, LegacyFamily::kBlsmsk,
                                 LegacyFamily::kBlsr,  LegacyFamily::kLzcnt, LegacyFamily::kTzcnt, LegacyFamily::kMovbe};
    for (LegacyFamily f : kAll) {
        const bool counts = f == LegacyFamily::kLzcnt || f == LegacyFamily::kTzcnt;
        const bool movbe = f == LegacyFamily::kMovbe;
        for (int store = 0; store < (movbe ? 2 : 1); ++store) {
            for (unsigned size : {2u, 4u, 8u}) {
                if (size == 2 && !counts && !movbe) continue;
                for (int base : bases) {
                    for (int index : indexes) {
                        if (base == 16 && index >= 0) continue;  // rip takes no index
                        if (base < 0 && index < 0) continue;     // an absolute address: no site has one
                        // The operand registers: each of the address's, and others.
                        std::vector<int> picks = {regs[rng.below(regs.size())]};
                        if (base >= 0 && base < 16 && base != kRsp) picks.push_back(base);
                        if (index >= 0) picks.push_back(index);
                        picks.push_back(1);  // rcx: BEXTR's shift count
                        for (int d : picks) {
                            if (d == kRsp) continue;
                            for (int v : {d, base >= 0 && base < 16 && base != kRsp ? base : 2, 1, static_cast<int>(9 + rng.below(7))}) {
                                Spec s;
                                s.family = f;
                                s.size = size;
                                s.store = store;
                                s.dst = d;
                                s.vex = v == kRsp ? 3 : v;
                                s.mem = true;
                                s.base = base;
                                s.index = index;
                                s.scale = kScales[rng.below(4)];
                                const int ds = static_cast<int>(rng.below(3));
                                s.disp_size = base < 0 || base == 16 ? 4 : ds == 0 ? 0 : ds == 1 ? 1 : 4;
                                if (base >= 0 && base < 16 && (base & 7) == 5 && s.disp_size == 0) s.disp_size = 1;
                                s.disp = s.disp_size == 0 ? 0
                                         : s.disp_size == 1 ? static_cast<std::int32_t>(rng.below(256)) - 128
                                                            : static_cast<std::int32_t>(rng.below(0x20000)) - 0x10000;
                                if (base == kRsp) {
                                    // Above rsp (the caller's frame) or in its red zone.
                                    s.disp = s.disp_size == 0 ? 0 : static_cast<std::int32_t>(rng.below(1024)) - 128;
                                    if (s.disp_size == 1) s.disp = static_cast<std::int32_t>(rng.below(240)) - 120;
                                }
                                if (!needs_vex(f) || f == LegacyFamily::kBlsi || f == LegacyFamily::kBlsmsk || f == LegacyFamily::kBlsr)
                                    s.vex = 0;
                                one(s);
                                if (!needs_vex(f) || f == LegacyFamily::kBlsi || f == LegacyFamily::kBlsmsk || f == LegacyFamily::kBlsr)
                                    break;
                            }
                        }
                    }
                }
            }
        }
    }
    // Random combinations beyond those, LEGACY_CPU_TEST_SCALE times as many.
    for (int i = 0; i < 4000 * scale; ++i) {
        Spec s;
        s.family = kAll[rng.below(8)];
        const bool counts = s.family == LegacyFamily::kLzcnt || s.family == LegacyFamily::kTzcnt;
        const bool movbe = s.family == LegacyFamily::kMovbe;
        s.size = (counts || movbe) && rng.below(3) == 0 ? 2 : rng.below(2) ? 8 : 4;
        s.store = movbe && rng.below(2);
        s.dst = regs[rng.below(regs.size())];
        s.vex = regs[rng.below(regs.size())];
        if (!needs_vex(s.family) || s.family == LegacyFamily::kBlsi || s.family == LegacyFamily::kBlsmsk ||
            s.family == LegacyFamily::kBlsr)
            s.vex = 0;
        s.mem = movbe || rng.below(2);
        if (s.mem) {
            s.base = bases[rng.below(bases.size())];
            s.index = s.base == 16 ? -1 : indexes[rng.below(indexes.size())];
            if (s.base < 0 && s.index < 0) s.index = 3;
            s.scale = kScales[rng.below(4)];
            s.disp_size = s.base < 0 || s.base == 16 ? 4 : static_cast<int>(rng.below(2)) * 3 + (rng.below(2) ? 1 : 0);
            if (s.disp_size == 3) s.disp_size = 4;
            if (s.base >= 0 && s.base < 16 && (s.base & 7) == 5 && s.disp_size == 0) s.disp_size = 1;
            s.disp = s.disp_size == 0 ? 0 : s.disp_size == 1 ? static_cast<std::int32_t>(rng.below(256)) - 128 : static_cast<std::int32_t>(rng.below(0x20000)) - 0x10000;
            if (s.base == kRsp) s.disp = s.disp_size == 0 ? 0 : s.disp_size == 1 ? static_cast<std::int32_t>(rng.below(240)) - 120 : static_cast<std::int32_t>(rng.below(1024)) - 128;
        } else {
            s.rm = regs[rng.below(regs.size())];
        }
        one(s);
    }
    std::printf("legacy_cpu_test: %llu cases, %llu runs (%llu against the processor's own), routines up to %zu bytes; "
                "%llu mutants, %llu killed; %llu failures\n",
                static_cast<unsigned long long>(g_tot.cases), static_cast<unsigned long long>(g_tot.runs),
                static_cast<unsigned long long>(g_tot.native), g_tot.routine_bytes_max,
                static_cast<unsigned long long>(g_tot.mutants), static_cast<unsigned long long>(g_tot.killed),
                static_cast<unsigned long long>(g_tot.failures));
    return g_tot.failures ? 1 : 0;
}
