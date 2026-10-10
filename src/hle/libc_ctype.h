#pragma once

namespace bb::libc {

// Orbis/Dinkum C-locale flags, not the host CRT's ctype masks. In particular,
// 0x40 is additional whitespace, NOT control: Havok scans trailing whitespace
// with mask 0x144 and must stop at NUL (whose only flag is 0x80).
constexpr short ctype_flags(int c) {
    constexpr short kXD = 0x001;  // hex digit
    constexpr short kUP = 0x002;  // upper
    constexpr short kSP = 0x004;  // space
    constexpr short kPU = 0x008;  // punctuation
    constexpr short kLO = 0x010;  // lower
    constexpr short kDI = 0x020;  // digit
    constexpr short kXS = 0x040;  // additional whitespace (TAB through CR)
    constexpr short kCN = 0x080;  // control
    constexpr short kTB = 0x400;  // additional TAB flag

    if (c < 0 || c >= 128) return 0;  // EOF and non-ASCII in the C locale
    short bits = 0;
    if (c >= '0' && c <= '9') bits |= kDI | kXD;
    if (c >= 'A' && c <= 'Z') bits |= kUP;
    if (c >= 'a' && c <= 'z') bits |= kLO;
    if ((c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')) bits |= kXD;
    if (c == ' ') bits |= kSP;
    if (c >= '\t' && c <= '\r') bits |= kXS;
    if (c == '\t') bits |= kTB;
    if (c < 0x20 || c == 0x7f) bits |= kCN;
    if (c > 0x20 && c < 0x7f && !(bits & (kDI | kUP | kLO))) bits |= kPU;
    return bits;
}

}  // namespace bb::libc
