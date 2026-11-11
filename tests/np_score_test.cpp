// sceNpScoreSanitizeComment's answer (hle/np_score.h): the game sends its own
// character's name through it before the other players get it (they show it
// under Player Name Display: Name), so the text has to come back in the
// output buffer - whole, in UTF-8, cut only at the library's 255 bytes and
// then never inside a character - and a missing pointer is the library's
// invalid-argument error with the buffer untouched. An answer that writes
// nothing (as before: every name sent empty, shown blank) fails here.
#include "hle/np_score.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::printf("FAIL: %s\n", what);
    }
}

// The game's call: a 256-byte buffer (zeroed by the game; filled with a
// marker here, so an answer that leaves it alone cannot pass by luck).
struct Out {
    char buf[np_score::kSanitizeMaxLen + 1 + 16];
    Out() { std::memset(buf, 0x5a, sizeof buf); }
    std::string text() const {
        const void* nul = std::memchr(buf, 0, np_score::kSanitizeMaxLen + 1);
        return std::string(buf, nul ? static_cast<const char*>(nul) - buf : np_score::kSanitizeMaxLen + 1);
    }
    bool terminated() const { return std::memchr(buf, 0, np_score::kSanitizeMaxLen + 1) != nullptr; }
    bool tail_untouched() const {
        for (std::size_t i = np_score::kSanitizeMaxLen + 1; i < sizeof buf; ++i)
            if (buf[i] != 0x5a) return false;
        return true;
    }
};

std::string sanitized(const std::string& in, int* rc = nullptr) {
    Out o;
    const int r = np_score::sanitize_copy(in.c_str(), o.buf);
    if (rc) *rc = r;
    check(o.terminated(), "the answer is NUL-terminated inside the 256 bytes");
    check(o.tail_untouched(), "nothing is written past the 256 bytes");
    return o.text();
}

bool whole_utf8(const std::string& s) {
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        const std::size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (!len || i + len > s.size()) return false;
        for (std::size_t k = 1; k < len; ++k)
            if ((static_cast<unsigned char>(s[i + k]) & 0xc0) != 0x80) return false;
        i += len;
    }
    return true;
}

void names() {
    int rc = -1;
    check(sanitized("Hunter", &rc) == "Hunter", "a character's name comes back as it went");
    check(rc == 0, "and the call succeeds");
    check(sanitized("Lady Maria of the Astral Clocktower") == "Lady Maria of the Astral Clocktower", "spaces and length kept");
    // The game converts the UTF-16 name to UTF-8 before the call and back after it.
    const std::string jp = "\xe7\x8b\xa9\xe4\xba\xba";  // 狩人
    check(sanitized(jp) == jp, "a name outside ASCII comes back byte for byte");
    const std::string mixed = "Gehrman \xf0\x9f\x8c\x99";  // a 4-byte character
    check(sanitized(mixed) == mixed, "a 4-byte character comes back whole");
    check(sanitized("").empty(), "an empty text comes back empty");
}

void limits() {
    // The game sends at most 254 bytes (it checks strlen + 1 <= 255 first).
    const std::string longest(254, 'a');
    check(sanitized(longest) == longest, "the game's longest text comes back whole");
    const std::string max(np_score::kSanitizeMaxLen, 'b');
    check(sanitized(max) == max, "255 bytes, the library's limit, come back whole");
    const std::string over(300, 'c');
    check(sanitized(over) == std::string(np_score::kSanitizeMaxLen, 'c'), "a longer text is cut at 255 bytes");
    // 253 ASCII bytes and a 3-byte character: the cut at 255 falls inside it.
    const std::string cut = std::string(253, 'd') + "\xe7\x8b\xa9" + "tail";
    const std::string got = sanitized(cut);
    check(got == std::string(253, 'd'), "a cut inside a character drops the whole character");
    check(whole_utf8(got), "the cut leaves whole UTF-8");
    // 252 bytes and a 3-byte character end exactly at 255: kept.
    const std::string edge = std::string(252, 'e') + "\xe4\xba\xba" + "more";
    check(sanitized(edge) == std::string(252, 'e') + "\xe4\xba\xba", "a character that ends at the limit is kept");
}

void bad_arguments() {
    Out o;
    check(np_score::sanitize_copy(nullptr, o.buf) == np_score::kErrInvalidArgument, "no comment: invalid argument");
    check(static_cast<unsigned char>(o.buf[0]) == 0x5a, "and the buffer is left alone");
    check(np_score::sanitize_copy("Hunter", nullptr) == np_score::kErrInvalidArgument, "no buffer: invalid argument");
    check(np_score::kErrInvalidArgument == static_cast<int>(0x80550704u), "SCE_NP_COMMUNITY_ERROR_INVALID_ARGUMENT");
}

}  // namespace

int main() {
    names();
    limits();
    bad_arguments();
    if (g_fail) {
        std::printf("np_score_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("np_score_test: ok\n");
    return 0;
}
