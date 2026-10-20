#include "hle/sysv_va.h"

#include "log.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr std::uint32_t kGpEnd = 6 * 8;        // rdi..r9
constexpr std::uint32_t kFpEnd = 48 + 8 * 16;  // xmm0..xmm7

template <class T>
T from_stack(SysvVaList* ap, std::size_t align = 8) {
    auto p = reinterpret_cast<std::uintptr_t>(ap->overflow_arg_area);
    p = (p + align - 1) & ~(align - 1);
    T v;
    std::memcpy(&v, reinterpret_cast<const void*>(p), sizeof(T));
    ap->overflow_arg_area = reinterpret_cast<void*>(p + ((sizeof(T) + 7) & ~std::size_t(7)));
    return v;
}

long double va_long_double(SysvVaList* ap) { return from_stack<long double>(ap, 16); }

// One conversion specification, as parsed from the format.
struct Spec {
    std::string flags;
    int width = -1;       // -2: from an argument
    int precision = -1;   // -2: from an argument; -1: none
    std::string length;   // hh h l ll j z t L q
    char conv = 0;
    bool star = false;    // scanf: assignment suppressed
    std::size_t len = 0;  // bytes of the format the spec took, 0 = malformed
};

// p points after the '%'. In scanf mode a leading '*' suppresses the
// assignment; in printf mode '*' after the flags takes the width from an
// argument.
template <typename Char>
Spec parse_spec(const Char* p, bool scanf) {
    Spec s;
    const Char* q = p;
    if (scanf && *q == '*') {
        s.star = true;
        ++q;
    }
    while (!scanf && *q && *q <= 127 && std::strchr("-+ #0'", *q))
        s.flags.push_back(static_cast<char>(*q++));
    if (!scanf && *q == '*') {
        s.width = -2;
        ++q;
    } else {
        while (*q >= '0' && *q <= '9') {
            s.width = (s.width < 0 ? 0 : s.width) * 10 + (*q - '0');
            ++q;
        }
    }
    if (*q == '.') {
        ++q;
        if (*q == '*') {
            s.precision = -2;
            ++q;
        } else {
            s.precision = 0;
            while (*q >= '0' && *q <= '9') s.precision = s.precision * 10 + (*q++ - '0');
        }
    }
    while (*q && *q <= 127 && std::strchr("hlLjzqt", *q))
        s.length.push_back(static_cast<char>(*q++));
    if (!*q || *q > 127) return s;
    s.conv = static_cast<char>(*q++);
    s.len = static_cast<std::size_t>(q - p);
    return s;
}

// The host spec for one conversion: the flags/width/precision as parsed
// (fixed numbers after `*` was resolved), then `ll` for every integer and
// no length for the rest, so the widened value formats the same everywhere.
std::string host_spec(const Spec& s, int width, int precision, const char* length, char conv) {
    std::string r = "%" + s.flags;
    if (width >= 0) r += std::to_string(width);
    if (precision >= 0) r += "." + std::to_string(precision);
    r += length;
    r.push_back(conv);
    return r;
}

void append_fmt(std::string& out, const char* spec, auto value) {
    char small[256];
    const int n = std::snprintf(small, sizeof(small), spec, value);
    if (n < 0) return;
    if (static_cast<std::size_t>(n) < sizeof(small)) {
        out.append(small, static_cast<std::size_t>(n));
        return;
    }
    std::string big(static_cast<std::size_t>(n) + 1, '\0');
    std::snprintf(big.data(), big.size(), spec, value);
    out.append(big.data(), static_cast<std::size_t>(n));
}

std::string narrow_utf16(const char16_t* w, int precision) {
    std::string r;
    if (!w) return "(null)";
    for (std::size_t i = 0; w[i] && (precision < 0 || i < static_cast<std::size_t>(precision)); ++i) {
        r.push_back(w[i] < 0x80 ? static_cast<char>(w[i]) : '?');
    }
    return r;
}

std::string format(const char* fmt, SysvVaList* ap) {
    std::string out;
    for (const char* p = fmt; *p;) {
        if (*p != '%') {
            const char* q = p;
            while (*q && *q != '%') ++q;
            out.append(p, static_cast<std::size_t>(q - p));
            p = q;
            continue;
        }
        if (p[1] == '%') {
            out.push_back('%');
            p += 2;
            continue;
        }
        Spec s = parse_spec(p + 1, false);
        if (!s.len) {  // malformed: copy the rest as text
            out.append(p);
            break;
        }
        p += 1 + s.len;
        int width = s.width;
        int precision = s.precision;
        if (width == -2) {
            width = static_cast<int>(sysv_va_gp(ap));
            if (width < 0) {
                s.flags.push_back('-');
                width = -width;
            }
        }
        if (precision == -2) precision = static_cast<int>(sysv_va_gp(ap));
        const std::string& L = s.length;
        switch (s.conv) {
            case 'd':
            case 'i': {
                const std::uint64_t v = sysv_va_gp(ap);
                long long ll;
                if (L == "hh") ll = static_cast<signed char>(v);
                else if (L == "h") ll = static_cast<short>(v);
                else if (L.empty()) ll = static_cast<int>(static_cast<std::uint32_t>(v));
                else ll = static_cast<long long>(v);
                append_fmt(out, host_spec(s, width, precision, "ll", s.conv).c_str(), ll);
                break;
            }
            case 'u':
            case 'o':
            case 'x':
            case 'X': {
                const std::uint64_t v = sysv_va_gp(ap);
                unsigned long long ull;
                if (L == "hh") ull = static_cast<unsigned char>(v);
                else if (L == "h") ull = static_cast<unsigned short>(v);
                else if (L.empty()) ull = static_cast<std::uint32_t>(v);
                else ull = v;
                append_fmt(out, host_spec(s, width, precision, "ll", s.conv).c_str(), ull);
                break;
            }
            case 'c': {
                const std::uint64_t v = sysv_va_gp(ap);
                const int c = L.empty() ? static_cast<int>(static_cast<unsigned char>(v))
                                        : (v < 0x80 ? static_cast<int>(v) : '?');
                append_fmt(out, host_spec(s, width, -1, "", 'c').c_str(), c);
                break;
            }
            case 's': {
                const std::uint64_t v = sysv_va_gp(ap);
                if (L == "l") {
                    const std::string n = narrow_utf16(reinterpret_cast<const char16_t*>(v), precision);
                    append_fmt(out, host_spec(s, width, -1, "", 's').c_str(), n.c_str());
                } else {
                    const char* str = reinterpret_cast<const char*>(v);
                    append_fmt(out, host_spec(s, width, precision, "", 's').c_str(), str ? str : "(null)");
                }
                break;
            }
            case 'p': {  // FreeBSD prints 0x%llx
                const std::uint64_t v = sysv_va_gp(ap);
                char hex[32];
                std::snprintf(hex, sizeof(hex), "0x%llx", static_cast<unsigned long long>(v));
                Spec t = s;
                t.flags.erase(std::remove(t.flags.begin(), t.flags.end(), '0'), t.flags.end());
                append_fmt(out, host_spec(t, width, -1, "", 's').c_str(), hex);
                break;
            }
            case 'f':
            case 'F':
            case 'e':
            case 'E':
            case 'g':
            case 'G':
            case 'a':
            case 'A': {
                const double d = L == "L" ? static_cast<double>(va_long_double(ap)) : sysv_va_fp(ap);
                append_fmt(out, host_spec(s, width, precision, "", s.conv).c_str(), d);
                break;
            }
            case 'n': {
                void* dst = reinterpret_cast<void*>(sysv_va_gp(ap));
                const long long n = static_cast<long long>(out.size());
                if (!dst) break;
                if (L == "hh") *static_cast<signed char*>(dst) = static_cast<signed char>(n);
                else if (L == "h") *static_cast<short*>(dst) = static_cast<short>(n);
                else if (L.empty()) *static_cast<int*>(dst) = static_cast<int>(n);
                else *static_cast<long long*>(dst) = n;
                break;
            }
            default:  // unknown conversion: keep the text
                out.append(p - 1 - s.len, s.len + 1);
                break;
        }
    }
    return out;
}

struct WideFormat {
    std::u16string text;
    bool encoding_error = false;
};

// The native default multibyte decoder (0xf0e0..0xf2ff) accepts 1/2/3-byte
// sequences, including overlong encodings and surrogate code units. Do not
// substitute a modern strict UTF-8 codec or Microsoft's wide %s convention.
bool native_multibyte(const unsigned char*& p, char16_t& value) {
    unsigned code = *p++;
    unsigned extra = 0;
    if (code < 128) { value = static_cast<char16_t>(code); return true; }
    if ((code & 0xe0) == 0xc0) { code &= 0x1f; extra = 1; }
    else if ((code & 0xf0) == 0xe0) { code &= 0x0f; extra = 2; }
    else return false;
    while (extra--) {
        if ((*p & 0xc0) != 0x80) return false;
        code = (code << 6) | (*p++ & 0x3f);
    }
    value = static_cast<char16_t>(code);
    return true;
}

// Literal text and %ls/%lc stay UTF-16. Numeric directives use the existing
// LP64/SysV formatter; %s/%c follow the guest's multibyte decoder.

WideFormat format_wide(const char16_t* fmt, SysvVaList* ap) {
    WideFormat result;
    auto& out = result.text;
    for (const char16_t* p = fmt; *p;) {
        if (*p != u'%') { out.push_back(*p++); continue; }
        if (p[1] == u'%') { out.push_back(u'%'); p += 2; continue; }
        const char16_t* begin = p++;
        Spec s = parse_spec(p, false);
        if (!s.len) break;
        p += s.len;
        if (s.conv == 'n') {
            void* dst = reinterpret_cast<void*>(sysv_va_gp(ap));
            if (!dst) continue;
            const auto n = out.size();
            if (s.length == "hh") *static_cast<signed char*>(dst) = static_cast<signed char>(n);
            else if (s.length == "h") *static_cast<short*>(dst) = static_cast<short>(n);
            else if (s.length.empty()) *static_cast<int*>(dst) = static_cast<int>(n);
            else *static_cast<std::int64_t*>(dst) = static_cast<std::int64_t>(n);
            continue;
        }
        if (s.conv != 's' && s.conv != 'c') {
            std::string directive;
            for (const char16_t* q = begin; q != p; ++q) directive.push_back(static_cast<char>(*q));
            const auto numeric = format(directive.c_str(), ap);
            out.append(numeric.begin(), numeric.end());
            continue;
        }
        int width = s.width, precision = s.precision;
        if (width == -2) {
            width = static_cast<int>(sysv_va_gp(ap));
            if (width < 0) {
                s.flags.push_back('-');
                width = width == INT_MIN ? INT_MAX : -width;
            }
        }
        if (precision == -2) precision = static_cast<int>(sysv_va_gp(ap));
        std::u16string text;
        if (s.conv == 'c') {
            const auto c = sysv_va_gp(ap);
            if (s.length == "l") text.push_back(static_cast<char16_t>(c));
            else {
                const auto byte = static_cast<unsigned char>(c);
                if (byte > 127) {
                    // A valid multibyte lead is incomplete in a one-byte
                    // %c conversion (-2); other bytes are EILSEQ (-1).
                    if ((byte & 0xe0) != 0xc0 && (byte & 0xf0) != 0xe0) errno = EILSEQ;
                    result.encoding_error = true;
                    break;
                }
                text.push_back(static_cast<char16_t>(byte));
            }
        } else {
            const auto arg = sysv_va_gp(ap);
            if (!arg) {
                text = u"(null)";
                if (s.length != "l" && precision >= 0 && text.size() > static_cast<std::size_t>(precision))
                    text.resize(static_cast<std::size_t>(precision));
            } else if (s.length == "l") {
                const auto* w = reinterpret_cast<const char16_t*>(arg);
                while (*w && (precision < 0 || text.size() < static_cast<std::size_t>(precision)))
                    text.push_back(*w++);
            } else {
                const auto* str = reinterpret_cast<const unsigned char*>(arg);
                while (*str && (precision < 0 || text.size() < static_cast<std::size_t>(precision))) {
                    char16_t value;
                    if (!native_multibyte(str, value)) {
                        errno = EILSEQ;
                        result.encoding_error = true;
                        break;
                    }
                    if (!value) break;
                    text.push_back(value);
                }
                if (result.encoding_error) break;
            }
        }
        const auto padding = width > 0 && static_cast<std::size_t>(width) > text.size()
            ? static_cast<std::size_t>(width) - text.size() : 0;
        const bool left = s.flags.find('-') != std::string::npos;
        if (!left) out.append(padding, u' ');
        out += text;
        if (left) out.append(padding, u' ');
    }
    return result;
}

// scanf: each directive (or literal run) becomes one host call with `%n`
// appended, so the host does the matching and reports how far it got.
struct ScanIn {
    const char* s = nullptr;  // sscanf: the input and the position
    std::size_t pos = 0;
    std::FILE* f = nullptr;   // fscanf
    // Runs one sub-format on the input. Returns the host's count (0/1/EOF);
    // *consumed is -1 when the directive did not match.
    int run(const std::string& sub, void* target, int* consumed) {
        const std::string fmt = sub + "%n";
        *consumed = -1;
        int r;
        if (s) {
            r = target ? std::sscanf(s + pos, fmt.c_str(), target, consumed) : std::sscanf(s + pos, fmt.c_str(), consumed);
            if (*consumed >= 0) pos += static_cast<std::size_t>(*consumed);
        } else {
            r = target ? std::fscanf(f, fmt.c_str(), target, consumed) : std::fscanf(f, fmt.c_str(), consumed);
            if (*consumed >= 0) pos += static_cast<std::size_t>(*consumed);
        }
        return r;
    }
};

int scan(ScanIn& in, const char* fmt, SysvVaList* ap) {
    int assigned = 0;
    for (const char* p = fmt; *p;) {
        if (*p != '%' || p[1] == '%') {
            // A run of literal text (with %% kept as %%); whitespace in it
            // matches any amount of input whitespace, as the host does.
            std::string lit;
            while (*p && (*p != '%' || p[1] == '%')) {
                if (*p == '%') {
                    lit += "%%";
                    p += 2;
                } else {
                    lit.push_back(*p++);
                }
            }
            int consumed;
            const int r = in.run(lit, nullptr, &consumed);
            if (consumed < 0) return (r == EOF && !assigned) ? EOF : assigned;
            continue;
        }
        Spec s = parse_spec(p + 1, true);
        if (!s.len) return assigned;
        p += 1 + s.len;
        const bool star = s.star;
        std::string sub = "%";
        if (star) sub += "*";
        if (s.width >= 0) sub += std::to_string(s.width);
        std::string length = s.length;
        const char c = s.conv;
        if (c == 'n') {
            if (star) continue;
            void* dst = reinterpret_cast<void*>(sysv_va_gp(ap));
            if (!dst) continue;
            const long long n = static_cast<long long>(in.pos);
            if (length == "hh") *static_cast<signed char*>(dst) = static_cast<signed char>(n);
            else if (length == "h") *static_cast<short*>(dst) = static_cast<short>(n);
            else if (length.empty()) *static_cast<int*>(dst) = static_cast<int>(n);
            else *static_cast<long long*>(dst) = n;
            continue;
        }
        if (std::strchr("diouxX", c) && (length == "l" || length == "q")) length = "ll";  // guest long is 64-bit
        if (std::strchr("sc[", c) && length == "l") return assigned;  // wide targets: not supported
        if (c == '[') {  // copy the set through its closing bracket
            const char* q = p;
            if (*q == '^') ++q;
            if (*q == ']') ++q;
            while (*q && *q != ']') ++q;
            if (!*q) return assigned;
            sub += length;
            sub.append("[", 1);
            sub.append(p, static_cast<std::size_t>(q + 1 - p));
            p = q + 1;
        } else {
            sub += length;
            sub.push_back(c);
        }
        void* target = star ? nullptr : reinterpret_cast<void*>(sysv_va_gp(ap));
        int consumed;
        const int r = in.run(sub, target, &consumed);
        if (consumed < 0) return (r == EOF && !assigned) ? EOF : assigned;
        if (!star) ++assigned;
    }
    return assigned;
}

}  // namespace

std::uint64_t sysv_va_gp(SysvVaList* ap) {
    if (ap->gp_offset < kGpEnd) {
        std::uint64_t v;
        std::memcpy(&v, static_cast<const char*>(ap->reg_save_area) + ap->gp_offset, 8);
        ap->gp_offset += 8;
        return v;
    }
    return from_stack<std::uint64_t>(ap);
}

double sysv_va_fp(SysvVaList* ap) {
    if (ap->fp_offset < kFpEnd) {
        double v;
        std::memcpy(&v, static_cast<const char*>(ap->reg_save_area) + ap->fp_offset, 8);
        ap->fp_offset += 16;
        return v;
    }
    return from_stack<double>(ap);
}

// BBHOST_PRINTF_LOG=1: every guest printf-family call, format and result.
const bool g_printf_log = [] {
    const char* e = std::getenv("BBHOST_PRINTF_LOG");
    return e && e[0] == '1';
}();

int sysv_vsnprintf(char* out, std::size_t n, const char* fmt, SysvVaList* ap) {
    const std::string s = format(fmt, ap);
    if (g_printf_log) host_log("printf: \"%s\" -> \"%s\"", fmt, s.c_str());
    if (out && n) {
        const std::size_t c = s.size() < n - 1 ? s.size() : n - 1;
        std::memcpy(out, s.data(), c);
        out[c] = 0;
    }
    return static_cast<int>(s.size());
}

std::string sysv_vformat(const char* fmt, SysvVaList* ap) { return format(fmt, ap); }

int sysv_vfprintf(std::FILE* f, const char* fmt, SysvVaList* ap) {
    const std::string s = format(fmt, ap);
    if (std::fwrite(s.data(), 1, s.size(), f) != s.size()) return -1;
    return static_cast<int>(s.size());
}

int sysv_vswprintf(char16_t* out, std::size_t n, const char16_t* fmt, SysvVaList* ap) {
    if (!fmt) { if (out && n) *out = 0; return -1; }
    const auto result = format_wide(fmt, ap);
    if (out && n) {
        const auto count = std::min(result.text.size(), n - 1);
        std::memcpy(out, result.text.data(), count * sizeof(char16_t));
        out[count] = 0;
    }
    if (result.encoding_error) return -1;
    if ((!result.text.empty() && result.text.size() >= n) || result.text.size() > INT_MAX) return -1;
    return static_cast<int>(result.text.size());
}

int sysv_vsscanf(const char* s, const char* fmt, SysvVaList* ap) {
    ScanIn in;
    in.s = s;
    const int r = scan(in, fmt, ap);
    if (g_printf_log) host_log("sscanf: \"%s\" on \"%.60s\" -> %d", fmt, s ? s : "(null)", r);
    return r;
}

int sysv_vfscanf(std::FILE* f, const char* fmt, SysvVaList* ap) {
    ScanIn in;
    in.f = f;
    return scan(in, fmt, ap);
}
