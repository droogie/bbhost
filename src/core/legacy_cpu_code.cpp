#include "core/legacy_cpu_code.h"

#include "decomp/insn.h"

#include <cstring>
#include <initializer_list>

namespace {

constexpr std::uint32_t kCF = 0x1, kZF = 0x40, kSF = 0x80, kOF = 0x800;
constexpr int kRax = 0, kRcx = 1, kRdx = 2, kRbx = 3, kRsp = 4;

bool fits8(std::int64_t v) { return v >= -128 && v <= 127; }
bool fits32(std::int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }

std::uint64_t mask_of(unsigned size) { return size == 8 ? ~0ull : (1ull << (size * 8)) - 1; }

std::uint64_t byteswap(std::uint64_t v, unsigned size) {
    std::uint64_t r = 0;
    for (unsigned i = 0; i < size; ++i) r |= ((v >> (8 * i)) & 0xff) << (8 * (size - 1 - i));
    return r;
}

// The registers an r/m operand reads: its register, or its address's.
unsigned rm_regs(const LegacyInsn& in) {
    if (!in.mem) return 1u << in.rm;
    unsigned r = 0;
    if (in.m.base >= 0 && in.m.base < 16) r |= 1u << in.m.base;
    if (in.m.index >= 0) r |= 1u << in.m.index;
    return r;
}

// A small x86-64 assembler: the instructions the routines are made of, with
// rel8 labels inside a routine.
class Asm {
public:
    Asm(std::uint8_t* out, std::uint64_t va, std::size_t cap) : p_(out), start_(out), lim_(out + cap), va_(va) {}

    bool ok() const { return ok_; }
    std::size_t size() const { return static_cast<std::size_t>(p_ - start_); }
    std::uint64_t here() const { return va_ + size(); }

    void byte(std::uint8_t b) {
        if (p_ < lim_) {
            *p_++ = b;
        } else {
            ok_ = false;
        }
    }
    void bytes(std::initializer_list<std::uint8_t> bs) {
        for (std::uint8_t b : bs) byte(b);
    }
    void imm32(std::int32_t v) {
        for (int i = 0; i < 4; ++i) byte(static_cast<std::uint8_t>(static_cast<std::uint32_t>(v) >> (8 * i)));
    }

    // Labels: a forward rel8 branch is patched when its label is bound.
    struct Label {
        std::uint8_t* fix[4] = {};
        int n = 0;
        std::uint8_t* at = nullptr;
    };
    void jcc8(std::uint8_t cc, Label& l) { branch8(static_cast<std::uint8_t>(0x70 | cc), l); }
    void jmp8(Label& l) { branch8(0xeb, l); }
    void bind(Label& l) {
        l.at = p_;
        for (int i = 0; i < l.n; ++i) {
            const std::int64_t d = p_ - (l.fix[i] + 1);
            if (!fits8(d)) ok_ = false;
            *l.fix[i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(d));
        }
    }

    // [66] [REX] opcode ModRM for a register r/m. byte_regs: a byte register
    // in rm (or reg) of 4-7 means spl..dil, which needs a REX.
    void rr(unsigned size, std::initializer_list<std::uint8_t> op, int reg, int rm, bool byte_rm = false, bool byte_reg = false) {
        if (size == 2) byte(0x66);
        std::uint8_t rex = 0x40;
        if (size == 8) rex |= 8;
        if (reg & 8) rex |= 4;
        if (rm & 8) rex |= 1;
        const bool need = rex != 0x40 || (byte_rm && (rm & 7) >= 4 && rm < 8) || (byte_reg && (reg & 7) >= 4 && reg < 8);
        if (need) byte(rex);
        for (std::uint8_t b : op) byte(b);
        byte(static_cast<std::uint8_t>(0xc0 | ((reg & 7) << 3) | (rm & 7)));
    }

    // [66] [REX] opcode ModRM [SIB] [disp] for a memory r/m. delta: how far
    // rsp is below the guest's here (an rsp-based address moves with it);
    // rip_target: where a rip-relative operand points; tail: bytes after the
    // displacement (an immediate).
    void rm(unsigned size, std::initializer_list<std::uint8_t> op, int reg, const LegacyMem& m, int delta,
            std::uint64_t rip_target, int tail = 0) {
        if (size == 2) byte(0x66);
        std::uint8_t rex = 0x40;
        if (size == 8) rex |= 8;
        if (reg & 8) rex |= 4;
        if (m.index >= 8) rex |= 2;
        if (m.base >= 8 && m.base < 16) rex |= 1;
        if (rex != 0x40) byte(rex);
        for (std::uint8_t b : op) byte(b);
        if (m.base == LegacyMem::kRip) {
            byte(static_cast<std::uint8_t>(((reg & 7) << 3) | 5));
            const std::int64_t d = static_cast<std::int64_t>(rip_target - (here() + 4 + static_cast<std::uint64_t>(tail)));
            if (!fits32(d)) ok_ = false;
            imm32(static_cast<std::int32_t>(d));
            return;
        }
        std::int64_t disp = m.disp;
        if (m.base == kRsp) disp += delta;
        if (!fits32(disp)) ok_ = false;
        const bool sib = m.index != LegacyMem::kNone || m.base == LegacyMem::kNone || (m.base & 7) == 4;
        int mod;
        if (m.base == LegacyMem::kNone) {
            mod = 0;  // SIB base 101 with mod 00: disp32, no base
        } else if (disp == 0 && (m.base & 7) != 5) {
            mod = 0;
        } else if (fits8(disp)) {
            mod = 1;
        } else {
            mod = 2;
        }
        byte(static_cast<std::uint8_t>((mod << 6) | ((reg & 7) << 3) | (sib ? 4 : (m.base & 7))));
        if (sib) {
            const int ss = m.scale == 8 ? 3 : m.scale == 4 ? 2 : m.scale == 2 ? 1 : 0;
            const int idx = m.index == LegacyMem::kNone ? 4 : (m.index & 7);
            const int base = m.base == LegacyMem::kNone ? 5 : (m.base & 7);
            byte(static_cast<std::uint8_t>((ss << 6) | (idx << 3) | base));
        }
        if (mod == 1) {
            byte(static_cast<std::uint8_t>(static_cast<std::int8_t>(disp)));
        } else if (mod == 2 || m.base == LegacyMem::kNone) {
            imm32(static_cast<std::int32_t>(disp));
        }
    }

    // The instructions themselves.
    void push(int r) {
        if (r & 8) byte(0x41);
        byte(static_cast<std::uint8_t>(0x50 | (r & 7)));
    }
    void pop(int r) {
        if (r & 8) byte(0x41);
        byte(static_cast<std::uint8_t>(0x58 | (r & 7)));
    }
    void lea_rsp(int d) {  // lea rsp, [rsp + d]
        LegacyMem m;
        m.base = kRsp;
        m.disp = d;
        rm(8, {0x8d}, kRsp, m, 0, 0);
    }
    void mov_rr(unsigned size, int dst, int src) { rr(size, {0x89}, src, dst); }   // mov dst, src
    void op_rr(unsigned size, std::uint8_t op, int dst, int src) { rr(size, {op}, src, dst); }  // op r/m=dst, reg=src
    void test_rr(unsigned size, int a, int b) { rr(size, {0x85}, b, a); }
    void not_r(unsigned size, int r) { rr(size, {0xf7}, 2, r); }
    void neg_r(unsigned size, int r) { rr(size, {0xf7}, 3, r); }
    void shift_cl(unsigned size, int ext, int r) { rr(size, {0xd3}, ext, r); }  // ext 4 shl, 5 shr
    void shift_imm(unsigned size, int ext, int r, std::uint8_t n) {
        rr(size, {0xc1}, ext, r);
        byte(n);
    }
    void alu_imm8(unsigned size, int ext, int r, std::int8_t v) {  // ext 0 add, 1 or, 6 xor, 7 cmp
        rr(size, {0x83}, ext, r);
        byte(static_cast<std::uint8_t>(v));
    }
    void mov_imm32(int r, std::uint32_t v) {  // mov r32, imm32 (zero-extends)
        if (r & 8) byte(0x41);
        byte(static_cast<std::uint8_t>(0xb8 | (r & 7)));
        imm32(static_cast<std::int32_t>(v));
    }
    void mov_imm16(int r, std::uint16_t v) {  // mov r16, imm16 (keeps the upper bits)
        byte(0x66);
        if (r & 8) byte(0x41);
        byte(static_cast<std::uint8_t>(0xb8 | (r & 7)));
        byte(static_cast<std::uint8_t>(v));
        byte(static_cast<std::uint8_t>(v >> 8));
    }
    void bswap(unsigned size, int r) {
        std::uint8_t rex = 0x40;
        if (size == 8) rex |= 8;
        if (r & 8) rex |= 1;
        if (rex != 0x40) byte(rex);
        byte(0x0f);
        byte(static_cast<std::uint8_t>(0xc8 | (r & 7)));
    }
    void movzx8(int dst, int src) { rr(4, {0x0f, 0xb6}, dst, src, true); }  // movzx dst32, src8
    void xchg_hl(int r) { rr(4, {0x86}, (r & 3) | 4, r & 3); }             // xchg rl, rh (rax..rbx)
    void stc() { byte(0xf9); }

private:
    void branch8(std::uint8_t op, Label& l) {
        byte(op);
        if (l.n < 4 && p_ < lim_) {
            l.fix[l.n++] = p_;
        } else {
            ok_ = false;
        }
        byte(0);
    }

    std::uint8_t* p_;
    std::uint8_t* start_;
    std::uint8_t* lim_;
    std::uint64_t va_;
    bool ok_ = true;
};

constexpr std::uint8_t kCcZ = 0x4, kCcAe = 0x3;

// A scratch register outside `avoid` (a bit per register; rsp is never one).
// legacy: only rax..rbx, whose second byte has a name (xchg cl, ch).
int pick(unsigned avoid, bool legacy = false) {
    static const int kOrder[] = {kRax, kRcx, kRdx, kRbx, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    for (int r : kOrder) {
        if (legacy && r > kRbx) break;
        if (!(avoid & (1u << r))) return r;
    }
    return -1;
}

// The scratch registers' frame: rsp below the red zone, then the pushes.
struct Frame {
    int regs[3] = {};
    int n = 0;
    int delta() const { return n ? 128 + 8 * n : 0; }
};

void open(Asm& a, Frame& f, std::initializer_list<int> regs) {
    if (regs.size() == 0) return;
    a.lea_rsp(-128);
    for (int r : regs) {
        a.push(r);
        f.regs[f.n++] = r;
    }
}

void close(Asm& a, const Frame& f) {
    if (!f.n) return;
    for (int i = f.n - 1; i >= 0; --i) a.pop(f.regs[i]);
    a.lea_rsp(128);
}

// The r/m operand into the routine: `op` with it as r/m (for a register r/m,
// as `rr` does with it in rm).
void with_rm(Asm& a, const LegacyInsn& in, unsigned size, std::initializer_list<std::uint8_t> op, int reg, int delta,
             std::uint64_t rip_target) {
    if (in.mem) {
        a.rm(size, op, reg, in.m, delta, rip_target);
    } else {
        a.rr(size, op, reg, in.rm);
    }
}

void emit_count(Asm& a, const LegacyInsn& in, std::uint64_t rip) {
    // bsr/bsf: ZF when the source is zero, else the bit's index.
    const unsigned bits = in.size * 8u;
    Asm::Label zero, done;
    with_rm(a, in, in.size, {0x0f, static_cast<std::uint8_t>(in.family == LegacyFamily::kLzcnt ? 0xbd : 0xbc)}, in.dst, 0, rip);
    a.jcc8(kCcZ, zero);
    if (in.family == LegacyFamily::kLzcnt) {
        a.alu_imm8(in.size, 6, in.dst, static_cast<std::int8_t>(bits - 1));  // xor: bits-1 - index; ZF, CF 0
    } else {
        a.test_rr(in.size, in.dst, in.dst);  // ZF when bit 0 was set, CF 0
    }
    a.jmp8(done);
    a.bind(zero);
    if (in.size == 2) {
        a.mov_imm16(in.dst, static_cast<std::uint16_t>(bits));
    } else {
        a.mov_imm32(in.dst, bits);
    }
    a.test_rr(in.size, in.dst, in.dst);  // ZF 0
    a.stc();                             // CF: the source was zero
    a.bind(done);
}

void emit_movbe(Asm& a, const LegacyInsn& in, std::uint64_t rip) {
    Frame f;
    if (!in.store) {
        if (in.size != 2) {
            a.rm(in.size, {0x8b}, in.dst, in.m, 0, rip);  // mov dst, [m]
            a.bswap(in.size, in.dst);
            return;
        }
        const int x = pick((1u << kRsp) | (1u << in.dst), true);
        open(a, f, {x});
        a.rm(4, {0x0f, 0xb7}, x, in.m, f.delta(), rip);  // movzx x32, word [m]
        a.xchg_hl(x);
        a.mov_rr(2, in.dst, x);
        close(a, f);
        return;
    }
    const unsigned avoid = (1u << kRsp) | (1u << in.dst) | rm_regs(in);
    const int x = pick(avoid, in.size == 2);
    open(a, f, {x});
    a.mov_rr(in.size == 8 ? 8 : 4, x, in.dst);
    if (in.size == 2) {
        a.xchg_hl(x);
    } else {
        a.bswap(in.size, x);
    }
    a.rm(in.size, {0x89}, x, in.m, f.delta(), rip);  // mov [m], x
    close(a, f);
}

void emit_andn(Asm& a, const LegacyInsn& in, std::uint64_t rip) {
    const int d = in.dst, s1 = in.vex;
    const unsigned sregs = rm_regs(in);
    if (!(sregs & (1u << d))) {
        if (d != s1) a.mov_rr(in.size, d, s1);
        a.not_r(in.size, d);
        with_rm(a, in, in.size, {0x23}, d, 0, rip);  // and d, src: SF ZF, CF OF 0
        return;
    }
    Frame f;
    const int x = pick((1u << kRsp) | (1u << d) | (1u << s1) | sregs);
    open(a, f, {x});
    a.mov_rr(in.size, x, s1);
    a.not_r(in.size, x);
    with_rm(a, in, in.size, {0x23}, x, f.delta(), rip);
    a.mov_rr(in.size, d, x);
    close(a, f);
}

void emit_bls(Asm& a, const LegacyInsn& in, std::uint64_t rip) {
    const int d = in.dst;
    Frame f;
    int r;
    if (!in.mem && in.rm != d) {
        r = in.rm;
    } else {
        r = pick((1u << kRsp) | (1u << d) | rm_regs(in));
        open(a, f, {r});
        with_rm(a, in, in.size, {0x8b}, r, f.delta(), rip);  // mov r, src
    }
    Asm::Label zero, done;
    LegacyMem minus1;
    minus1.base = static_cast<std::int8_t>(r);
    minus1.disp = -1;
    switch (in.family) {
    case LegacyFamily::kBlsr:  // (src - 1) & src; CF: src was zero
        a.test_rr(in.size, r, r);
        a.jcc8(kCcZ, zero);
        a.rm(in.size, {0x8d}, d, minus1, 0, 0);
        a.op_rr(in.size, 0x21, d, r);  // and d, r
        a.jmp8(done);
        a.bind(zero);
        a.op_rr(4, 0x31, d, d);  // xor d32, d32: 0, ZF
        a.stc();
        a.bind(done);
        break;
    case LegacyFamily::kBlsmsk:  // (src - 1) ^ src; CF: src was zero; ZF 0
        a.test_rr(in.size, r, r);
        a.jcc8(kCcZ, zero);
        a.rm(in.size, {0x8d}, d, minus1, 0, 0);
        a.op_rr(in.size, 0x31, d, r);  // xor d, r
        a.jmp8(done);
        a.bind(zero);
        a.alu_imm8(in.size, 1, d, -1);  // or d, -1: all ones, SF
        a.stc();
        a.bind(done);
        break;
    default:  // BLSI: -src & src; CF: src was not zero
        a.mov_rr(in.size, d, r);
        a.neg_r(in.size, d);
        a.op_rr(in.size, 0x21, d, r);
        a.jcc8(kCcZ, done);
        a.stc();
        a.bind(done);
        break;
    }
    close(a, f);
}

void emit_bextr(Asm& a, const LegacyInsn& in, std::uint64_t rip) {
    // The shifts need cl: rcx is saved with two scratch registers, the value
    // (t) and the control's copy (k), and the result goes to the destination
    // - into rcx's saved slot when rcx is the destination.
    const int d = in.dst, c = in.vex;
    const unsigned bits = in.size * 8u;
    const unsigned avoid = (1u << kRsp) | (1u << kRcx) | (1u << d) | (1u << c) | rm_regs(in);
    const int t = pick(avoid);
    const int k = pick(avoid | (1u << t));
    Frame f;
    open(a, f, {kRcx, t, k});
    with_rm(a, in, in.size, {0x8b}, t, f.delta(), rip);  // mov t, value
    a.mov_rr(4, k, c);                                    // k = control
    Asm::Label zero, keep, store;
    a.movzx8(kRcx, k);  // start
    a.alu_imm8(4, 7, kRcx, static_cast<std::int8_t>(bits));
    a.jcc8(kCcAe, zero);
    a.shift_cl(in.size, 5, t);  // t >>= start
    a.mov_rr(4, kRcx, k);
    a.shift_imm(4, 5, kRcx, 8);
    a.movzx8(kRcx, kRcx);  // length
    a.test_rr(4, kRcx, kRcx);
    a.jcc8(kCcZ, zero);
    a.alu_imm8(4, 7, kRcx, static_cast<std::int8_t>(bits));
    a.jcc8(kCcAe, keep);
    a.neg_r(4, kRcx);  // the shifts take it modulo bits: bits - length
    a.shift_cl(in.size, 4, t);
    a.shift_cl(in.size, 5, t);
    a.bind(keep);
    a.test_rr(in.size, t, t);  // ZF; CF OF 0
    a.jmp8(store);
    a.bind(zero);
    a.op_rr(4, 0x31, t, t);  // 0: ZF; CF OF 0
    a.bind(store);
    if (d == kRcx) {
        LegacyMem slot;
        slot.base = kRsp;
        slot.disp = 16;  // rcx's push, under t's and k's
        a.rm(8, {0x89}, t, slot, 0, 0);
    } else {
        a.mov_rr(in.size, d, t);
    }
    close(a, f);
}

}  // namespace

const char* legacy_family_name(LegacyFamily f) {
    switch (f) {
    case LegacyFamily::kAndn: return "ANDN";
    case LegacyFamily::kBextr: return "BEXTR";
    case LegacyFamily::kBlsi: return "BLSI";
    case LegacyFamily::kBlsmsk: return "BLSMSK";
    case LegacyFamily::kBlsr: return "BLSR";
    case LegacyFamily::kLzcnt: return "LZCNT";
    case LegacyFamily::kTzcnt: return "TZCNT";
    case LegacyFamily::kMovbe: return "MOVBE";
    }
    return "?";
}

bool legacy_decode(const std::uint8_t* p, std::size_t avail, LegacyInsn* out) {
    if (avail > 15) avail = 15;
    std::size_t i = 0;
    bool p66 = false, pf3 = false, pf2 = false;
    for (; i < avail; ++i) {
        const std::uint8_t b = p[i];
        if (b == 0x66) {
            p66 = true;
        } else if (b == 0xf3) {
            pf3 = true;
        } else if (b == 0xf2) {
            pf2 = true;
        } else if (b == 0x67 || b == 0xf0 || b == 0x64 || b == 0x65 || b == 0x2e || b == 0x36 || b == 0x3e || b == 0x26) {
            return false;  // 32-bit addressing, lock, a segment: not a form the routines cover
        } else {
            break;
        }
    }
    LegacyInsn r;
    std::uint8_t rex = 0;
    if (i < avail && p[i] == 0xc4) {
        // VEX.LZ.0F38 W0/W1 with pp 00: ANDN (F2), BLSR/BLSMSK/BLSI (F3 /1 /2 /3), BEXTR (F7).
        if (p66 || pf3 || pf2 || i + 4 >= avail) return false;
        const std::uint8_t b1 = p[i + 1], b2 = p[i + 2], op = p[i + 3];
        if ((b1 & 0x1f) != 2 || (b2 & 0x07) != 0) return false;
        rex = static_cast<std::uint8_t>(0x40 | ((b2 & 0x80) ? 8 : 0) | (~b1 & 0xe0) >> 5);
        r.vex = static_cast<std::uint8_t>((~b2 >> 3) & 15);
        r.size = (b2 & 0x80) ? 8 : 4;
        i += 4;
        const int ext = (p[i] >> 3) & 7;
        r.dst = static_cast<std::uint8_t>(ext | ((rex & 4) << 1));
        if (op == 0xf2) {
            r.family = LegacyFamily::kAndn;
        } else if (op == 0xf7) {
            r.family = LegacyFamily::kBextr;
        } else if (op == 0xf3 && ext >= 1 && ext <= 3) {
            r.family = ext == 1 ? LegacyFamily::kBlsr : ext == 2 ? LegacyFamily::kBlsmsk : LegacyFamily::kBlsi;
            r.dst = r.vex;
        } else {
            return false;
        }
    } else {
        if (i < avail && (p[i] & 0xf0) == 0x40) rex = p[i++];
        if (i + 2 >= avail || p[i] != 0x0f) return false;
        if ((p[i + 1] == 0xbd || p[i + 1] == 0xbc) && pf3 && !pf2) {
            r.family = p[i + 1] == 0xbd ? LegacyFamily::kLzcnt : LegacyFamily::kTzcnt;
            i += 2;
        } else if (p[i + 1] == 0x38 && (p[i + 2] == 0xf0 || p[i + 2] == 0xf1) && !pf3 && !pf2) {
            r.family = LegacyFamily::kMovbe;  // F2 0F 38 F0/F1 is CRC32
            r.store = p[i + 2] == 0xf1;
            i += 3;
        } else {
            return false;
        }
        r.size = (rex & 8) ? 8 : p66 ? 2 : 4;
        r.dst = static_cast<std::uint8_t>(((p[i] >> 3) & 7) | ((rex & 4) << 1));
    }
    if (i >= avail) return false;
    const std::uint8_t modrm = p[i++];
    const int mod = modrm >> 6, low = modrm & 7;
    const int b = rex & 1, x = (rex >> 1) & 1;
    if (mod == 3) {
        if (r.family == LegacyFamily::kMovbe) return false;
        r.rm = static_cast<std::uint8_t>(low | (b << 3));
    } else {
        r.mem = true;
        bool disp32 = mod == 2;
        if (low == 4) {
            if (i >= avail) return false;
            const std::uint8_t sib = p[i++];
            const int idx = ((sib >> 3) & 7) | (x << 3);
            r.m.index = static_cast<std::int8_t>(idx == 4 ? LegacyMem::kNone : idx);
            r.m.scale = static_cast<std::uint8_t>(1u << (sib >> 6));
            if ((sib & 7) == 5 && mod == 0) {
                r.m.base = LegacyMem::kNone;
                disp32 = true;
            } else {
                r.m.base = static_cast<std::int8_t>((sib & 7) | (b << 3));
            }
        } else if (low == 5 && mod == 0) {
            r.m.base = LegacyMem::kRip;
            disp32 = true;
        } else {
            r.m.base = static_cast<std::int8_t>(low | (b << 3));
        }
        if (mod == 1) {
            if (i >= avail) return false;
            r.m.disp = static_cast<std::int8_t>(p[i++]);
        } else if (disp32) {
            if (i + 4 > avail) return false;
            std::int32_t d;
            std::memcpy(&d, p + i, 4);
            r.m.disp = d;
            i += 4;
        }
    }
    // rsp as a register operand is no compiler's, and a routine's frame moves
    // it: refused (an rsp-based address is fine - the frame's offset is added).
    if (r.dst == kRsp || (!r.mem && r.rm == kRsp)) return false;
    if ((r.family == LegacyFamily::kAndn || r.family == LegacyFamily::kBextr) && r.vex == kRsp) return false;
    r.len = static_cast<std::uint8_t>(i);
    *out = r;
    return true;
}

std::uint32_t legacy_defined_flags(const LegacyInsn& in) {
    switch (in.family) {
    case LegacyFamily::kAndn:
    case LegacyFamily::kBlsi:
    case LegacyFamily::kBlsmsk:
    case LegacyFamily::kBlsr: return kCF | kZF | kSF | kOF;
    case LegacyFamily::kBextr: return kCF | kZF | kOF;
    case LegacyFamily::kLzcnt:
    case LegacyFamily::kTzcnt: return kCF | kZF;
    case LegacyFamily::kMovbe: return 0;
    }
    return 0;
}

std::uint64_t legacy_eval(const LegacyInsn& in, std::uint64_t rm, std::uint64_t vex, std::uint64_t old, std::uint32_t flags,
                          std::uint32_t* flags_out) {
    const unsigned bits = in.size * 8u;
    const std::uint64_t mask = mask_of(in.size);
    rm &= mask;
    vex &= mask;
    std::uint64_t res = 0;
    bool cf = false, zf = false, sf = false;
    switch (in.family) {
    case LegacyFamily::kAndn: res = ~vex & rm & mask; break;
    case LegacyFamily::kBextr: {
        const unsigned start = vex & 0xff, len = (vex >> 8) & 0xff;
        res = start >= bits ? 0 : rm >> start;
        if (len < bits) res &= (1ull << len) - 1;
        break;
    }
    case LegacyFamily::kBlsi:
        res = (0 - rm) & rm & mask;
        cf = rm != 0;
        break;
    case LegacyFamily::kBlsmsk:
        res = ((rm - 1) ^ rm) & mask;
        cf = rm == 0;
        break;
    case LegacyFamily::kBlsr:
        res = ((rm - 1) & rm) & mask;
        cf = rm == 0;
        break;
    case LegacyFamily::kLzcnt:
    case LegacyFamily::kTzcnt: {
        unsigned n = 0;
        if (rm == 0) {
            n = bits;
        } else if (in.family == LegacyFamily::kLzcnt) {
            while (!((rm >> (bits - 1 - n)) & 1)) ++n;
        } else {
            while (!((rm >> n) & 1)) ++n;
        }
        res = n;
        cf = rm == 0;
        break;
    }
    case LegacyFamily::kMovbe: res = byteswap(rm, in.size); break;
    }
    zf = res == 0;
    sf = (res >> (bits - 1)) & 1;
    std::uint32_t f = flags;
    const std::uint32_t defined = legacy_defined_flags(in);
    const auto put = [&](std::uint32_t bit, bool v) {
        if (defined & bit) f = v ? (f | bit) : (f & ~bit);
    };
    put(kCF, cf);
    put(kZF, zf && in.family != LegacyFamily::kBlsmsk);
    put(kSF, sf);
    put(kOF, false);
    *flags_out = f;
    if (in.family == LegacyFamily::kMovbe && in.store) return res;
    if (in.size == 2) return (old & ~0xffffull) | (res & 0xffff);
    return res;
}

std::size_t legacy_emit(const LegacyInsn& in, std::uint64_t end_va, std::uint8_t* out, std::uint64_t out_va, std::size_t cap) {
    Asm a(out, out_va, cap);
    const std::uint64_t rip = end_va + static_cast<std::uint64_t>(static_cast<std::int64_t>(in.m.disp));
    switch (in.family) {
    case LegacyFamily::kLzcnt:
    case LegacyFamily::kTzcnt: emit_count(a, in, rip); break;
    case LegacyFamily::kMovbe: emit_movbe(a, in, rip); break;
    case LegacyFamily::kAndn: emit_andn(a, in, rip); break;
    case LegacyFamily::kBextr: emit_bextr(a, in, rip); break;
    default: emit_bls(a, in, rip); break;
    }
    return a.ok() ? a.size() : 0;
}

std::uint64_t legacy_address(const LegacyInsn& in, const std::uint64_t regs[16], std::uint64_t end_va) {
    std::uint64_t a = static_cast<std::uint64_t>(static_cast<std::int64_t>(in.m.disp));
    if (in.m.base == LegacyMem::kRip) return end_va + a;
    if (in.m.base >= 0) a += regs[in.m.base];
    if (in.m.index >= 0) a += regs[in.m.index] * in.m.scale;
    return a;
}

void legacy_apply(const LegacyInsn& in, std::uint64_t regs[16], std::uint64_t* flags, std::uint64_t end_va) {
    std::uint64_t rm = 0;
    void* at = nullptr;
    if (in.mem) {
        at = reinterpret_cast<void*>(static_cast<std::uintptr_t>(legacy_address(in, regs, end_va)));
        if (!in.store) std::memcpy(&rm, at, in.size);
    } else {
        rm = regs[in.rm];
    }
    if (in.store) rm = regs[in.dst];
    std::uint32_t f = 0;
    const std::uint64_t v = legacy_eval(in, rm, regs[in.vex], regs[in.dst], static_cast<std::uint32_t>(*flags), &f);
    if (in.store) {
        std::memcpy(at, &v, in.size);
    } else {
        regs[in.dst] = v;
    }
    *flags = (*flags & ~0xffffffffull) | f;
}

std::size_t legacy_relocate(const std::uint8_t* src, std::size_t n, std::uint64_t from, std::uint8_t* dst, std::uint64_t to,
                            std::size_t cap) {
    std::size_t in = 0, out = 0;
    while (in < n) {
        LegacyInsn li;
        if (legacy_decode(src + in, n - in, &li)) {
            if (!to && li.mem && li.m.base == LegacyMem::kRip) return 0;
            const std::size_t k = legacy_emit(li, from + in + li.len, dst + out, to + out, cap - out);
            if (!k) return 0;
            in += li.len;
            out += k;
            continue;
        }
        const X86Insn x = x86_decode(src + in, n - in);
        if (!x.len) return 0;
        if (x.rel_at) {
            // A branch into the moved bytes would land somewhere else.
            const std::int64_t rel = x.rel_size == 1 ? static_cast<std::int8_t>(src[in + x.rel_at])
                                                     : static_cast<std::int32_t>(src[in + x.rel_at] | src[in + x.rel_at + 1] << 8 |
                                                                                 src[in + x.rel_at + 2] << 16 |
                                                                                 static_cast<std::uint32_t>(src[in + x.rel_at + 3]) << 24);
            const std::uint64_t target = from + in + x.len + static_cast<std::uint64_t>(rel);
            if (target >= from && target < from + n) return 0;
        }
        if (!to && (x.rel_at || x.disp_at)) return 0;
        const std::size_t k = x86_relocate(src + in, x.len, from + in, dst + out, to + out, cap - out);
        if (!k) return 0;
        in += x.len;
        out += k;
    }
    return out;
}
