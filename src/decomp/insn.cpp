#include "decomp/insn.h"

#include <cstring>

namespace {

constexpr std::size_t kMaxInsn = 15;

std::int32_t read32(const std::uint8_t* p) {
    std::int32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

void write32(std::uint8_t* p, std::int32_t v) { std::memcpy(p, &v, 4); }

bool fits32(std::int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }

}  // namespace

X86Insn x86_decode(const std::uint8_t* p, std::size_t avail) {
    if (avail > kMaxInsn) avail = kMaxInsn;
    std::size_t i = 0;
    bool opsize = false, addr32 = false, rex_w = false;
    for (;; ++i) {
        if (i >= avail) return {};
        const std::uint8_t b = p[i];
        if (b == 0x66) opsize = true;
        else if (b == 0x67) addr32 = true;
        else if (b != 0xf0 && b != 0xf2 && b != 0xf3 && b != 0x2e && b != 0x36 && b != 0x3e && b != 0x26 && b != 0x64 &&
                 b != 0x65)
            break;
    }
    std::uint8_t rex = 0;
    if ((p[i] & 0xf0) == 0x40) {
        rex = p[i];
        rex_w = (p[i] & 0x08) != 0;
        if (++i >= avail) return {};
    }
    std::uint8_t op = p[i++];
    int map = 0;  // the one-byte map; 1: 0F, 2: 0F 38, 3: 0F 3A
    if (op == 0xc5 || op == 0xc4) {
        // VEX: in 64-bit mode these are never LDS/LES.
        if (op == 0xc5) {
            if (i + 1 >= avail) return {};
            map = 1;
            rex = static_cast<std::uint8_t>(0x40 | ((~p[i] & 0x80) >> 5));  // VEX2 has R only
            i += 1;
        } else {
            if (i + 2 >= avail) return {};
            map = p[i] & 0x1f;
            rex_w = (p[i + 1] & 0x80) != 0;
            rex = static_cast<std::uint8_t>(0x40 | (rex_w ? 8 : 0) | ((~p[i] & 0xe0) >> 5));
            i += 2;
            if (map < 1 || map > 3) return {};
        }
        op = p[i++];
    } else if (op == 0x0f) {
        if (i >= avail) return {};
        op = p[i++];
        map = 1;
        if (op == 0x38 || op == 0x3a) {
            map = op == 0x38 ? 2 : 3;
            if (i >= avail) return {};
            op = p[i++];
        }
    }

    X86Insn r;
    bool modrm = false;
    std::size_t imm = 0;
    const std::size_t iz = opsize ? 2 : 4;
    enum { None, Test8, TestZ } test_imm = None;  // F6 / F7: an immediate only for TEST (reg 0, 1)
    const auto branch = [&](X86Insn::Kind kind, std::uint8_t size) {
        r.kind = kind;
        r.rel_size = size;
        imm = size;
    };
    if (map == 0) {
        if (op <= 0x3f) {
            switch (op & 7) {
            case 0: case 1: case 2: case 3: modrm = true; break;
            case 4: imm = 1; break;
            case 5: imm = iz; break;
            default: return {};  // 06/07/0e/... are not instructions in 64-bit mode
            }
        } else if (op >= 0x50 && op <= 0x5f) {
        } else if (op == 0x63) {
            modrm = true;
        } else if (op == 0x68) {
            imm = iz;
        } else if (op == 0x69) {
            modrm = true;
            imm = iz;
        } else if (op == 0x6a) {
            imm = 1;
        } else if (op == 0x6b) {
            modrm = true;
            imm = 1;
        } else if (op >= 0x6c && op <= 0x6f) {
        } else if (op >= 0x70 && op <= 0x7f) {
            branch(X86Insn::Jcc, 1);
            r.cc = op & 0x0f;
        } else if (op == 0x80 || op == 0x83) {
            modrm = true;
            imm = 1;
        } else if (op == 0x81) {
            modrm = true;
            imm = iz;
        } else if (op >= 0x84 && op <= 0x8f) {
            modrm = true;
        } else if (op >= 0x90 && op <= 0x9f && op != 0x9a) {
        } else if (op >= 0xa0 && op <= 0xa3) {
            imm = addr32 ? 4 : 8;  // moffs
        } else if ((op >= 0xa4 && op <= 0xa7) || (op >= 0xaa && op <= 0xaf)) {
        } else if (op == 0xa8) {
            imm = 1;
        } else if (op == 0xa9) {
            imm = iz;
        } else if (op >= 0xb0 && op <= 0xb7) {
            imm = 1;
        } else if (op >= 0xb8 && op <= 0xbf) {
            imm = rex_w ? 8 : iz;
        } else if (op == 0xc0 || op == 0xc1 || op == 0xc6) {
            modrm = true;
            imm = 1;
        } else if (op == 0xc7) {
            modrm = true;
            imm = iz;
        } else if (op == 0xc2 || op == 0xca) {
            imm = 2;
        } else if (op == 0xc8) {
            imm = 3;
        } else if (op == 0xcd) {
            imm = 1;
        } else if (op == 0xc3 || op == 0xc9 || op == 0xcb || op == 0xcc || op == 0xcf) {
        } else if ((op >= 0xd0 && op <= 0xd3) || (op >= 0xd8 && op <= 0xdf)) {
            modrm = true;
        } else if (op == 0xd7) {
        } else if (op >= 0xe0 && op <= 0xe3) {
            branch(X86Insn::Loop, 1);
        } else if (op >= 0xe4 && op <= 0xe7) {
            imm = 1;
        } else if (op == 0xe8) {
            branch(X86Insn::Call, 4);
        } else if (op == 0xe9) {
            branch(X86Insn::Jmp, 4);
        } else if (op == 0xeb) {
            branch(X86Insn::Jmp, 1);
        } else if ((op >= 0xec && op <= 0xef) || op == 0xf1 || op == 0xf4 || op == 0xf5 || (op >= 0xf8 && op <= 0xfd)) {
        } else if (op == 0xf6) {
            modrm = true;
            test_imm = Test8;
        } else if (op == 0xf7) {
            modrm = true;
            test_imm = TestZ;
        } else if (op == 0xfe || op == 0xff) {
            modrm = true;
        } else {
            return {};
        }
    } else if (map == 1) {
        if (op <= 0x03 || op == 0x0d || (op >= 0x10 && op <= 0x23) || (op >= 0x28 && op <= 0x2f) ||
            (op >= 0x40 && op <= 0x6f) || (op >= 0x74 && op <= 0x76) || (op >= 0x78 && op <= 0x7f) ||
            (op >= 0x90 && op <= 0x9f) || op == 0xa3 || op == 0xa5 || op == 0xab || (op >= 0xad && op <= 0xb9) ||
            (op >= 0xbb && op <= 0xc1) || op == 0xc3 || op == 0xc7 || op >= 0xd0) {
            modrm = true;
        } else if ((op >= 0x70 && op <= 0x73) || op == 0xa4 || op == 0xac || op == 0xba || op == 0xc2 ||
                   (op >= 0xc4 && op <= 0xc6)) {
            modrm = true;
            imm = 1;
        } else if (op >= 0x80 && op <= 0x8f) {
            branch(X86Insn::Jcc, 4);
            r.cc = op & 0x0f;
        } else if ((op >= 0x05 && op <= 0x09) || op == 0x0b || op == 0x0e || (op >= 0x30 && op <= 0x37) || op == 0x77 ||
                   op == 0xa0 || op == 0xa1 || op == 0xa2 || (op >= 0xa8 && op <= 0xaa) || (op >= 0xc8 && op <= 0xcf)) {
        } else {
            return {};
        }
    } else {
        modrm = true;
        imm = map == 3 ? 1 : 0;
    }

    r.rex = rex;
    if (modrm) {
        if (i >= avail) return {};
        r.modrm_at = static_cast<std::uint8_t>(i);
        const std::uint8_t m = p[i++];
        const int mod = m >> 6, reg = (m >> 3) & 7, rm = m & 7;
        if (test_imm != None && reg <= 1) imm = test_imm == Test8 ? 1 : iz;
        std::size_t disp = 0;
        if (mod != 3) {
            if (rm == 4) {
                if (i >= avail) return {};
                const std::uint8_t sib = p[i++];
                if (mod == 0 && (sib & 7) == 5) disp = 4;
            }
            if (mod == 0 && rm == 5) {
                r.disp_at = static_cast<std::uint8_t>(i);  // [rip + disp32]
                disp = 4;
            } else if (mod == 1) {
                disp = 1;
            } else if (mod == 2) {
                disp = 4;
            }
        }
        i += disp;
    }
    if (r.rel_size) r.rel_at = static_cast<std::uint8_t>(i);
    i += imm;
    if (i > avail) return {};
    r.len = static_cast<std::uint8_t>(i);
    return r;
}

std::size_t x86_relocate(const std::uint8_t* src, std::size_t n, std::uint64_t from, std::uint8_t* dst, std::uint64_t to,
                         std::size_t cap) {
    std::size_t in = 0, out = 0;
    while (in < n) {
        const X86Insn x = x86_decode(src + in, n - in);
        if (!x.len || x.kind == X86Insn::Loop) return 0;
        const std::uint64_t old_end = from + in + x.len;
        if (x.rel_at) {
            const std::int64_t rel = x.rel_size == 1 ? static_cast<std::int8_t>(src[in + x.rel_at]) : read32(src + in + x.rel_at);
            const std::uint64_t target = old_end + static_cast<std::uint64_t>(rel);
            // Into what moves (its start included: that is now the jump to
            // ours) - the branch would land somewhere else.
            if (target >= from && target < from + n) return 0;
            std::uint8_t buf[kMaxInsn + 1];
            std::size_t len;
            if (x.rel_size == 4) {
                len = x.len;
                std::memcpy(buf, src + in, len);
            } else if (x.kind == X86Insn::Jmp) {
                buf[0] = 0xe9;  // jmp rel8 -> jmp rel32
                len = 5;
            } else {
                buf[0] = 0x0f;  // jcc rel8 -> jcc rel32, its prefixes (branch hints) left behind
                buf[1] = static_cast<std::uint8_t>(0x80 | x.cc);
                len = 6;
            }
            const std::int64_t now = static_cast<std::int64_t>(target - (to + out + len));
            if (!fits32(now) || out + len > cap) return 0;
            write32(buf + len - 4, static_cast<std::int32_t>(now));
            std::memcpy(dst + out, buf, len);
            out += len;
        } else {
            if (out + x.len > cap) return 0;
            std::memcpy(dst + out, src + in, x.len);
            if (x.disp_at) {
                const std::int64_t target = static_cast<std::int64_t>(old_end) + read32(src + in + x.disp_at);
                const std::int64_t now = target - static_cast<std::int64_t>(to + out + x.len);
                if (!fits32(now)) return 0;
                write32(dst + out + x.disp_at, static_cast<std::int32_t>(now));
            }
            out += x.len;
        }
        in += x.len;
    }
    return in == n ? out : 0;
}
