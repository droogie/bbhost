// sceNpScoreSanitizeComment's answer (hle/system.cpp), apart so a test can
// hold it to its contract (tests/np_score_test.cpp).
//
// Playing online, the game puts its own character's name through it before
// the name goes to the other players: the name holder (sub_225b8e0) queues
// it for the NP worker, which converts it to UTF-8 and calls
// sceNpScoreSanitizeComment(request, name, out, NULL) with a zeroed 256-byte
// buffer (sub_10d09a0); whatever comes back in that buffer is the name the
// session's player data carries (sub_1b81ac0), and the other games show it
// under System > Network > Player Name Display "Name". PSN gave the text back
// with the words its filter matched starred out; the private server has no
// filter, so the text comes back as it went. Writing nothing - this answered
// success with the buffer untouched - sent every name empty.
#pragma once

#include <cstddef>
#include <cstring>

namespace np_score {

// SCE_NP_SCORE_SANITIZE_COMMENT_MAXLEN: the output holds this many bytes and
// a NUL.
constexpr std::size_t kSanitizeMaxLen = 255;

// SCE_NP_COMMUNITY_ERROR_INVALID_ARGUMENT, the score library's answer to a
// missing comment or output buffer.
constexpr int kErrInvalidArgument = static_cast<int>(0x80550704u);

// Copies `comment` into `out` (kSanitizeMaxLen + 1 bytes) as a server with no
// word filter answers: the same text, cut at kSanitizeMaxLen bytes - back to
// the start of a UTF-8 sequence, so a cut never leaves half a character -
// and NUL-terminated. Returns 0, or kErrInvalidArgument for a null pointer
// (out untouched).
inline int sanitize_copy(const char* comment, char* out) {
    if (!comment || !out) return kErrInvalidArgument;
    std::size_t n = 0;
    while (n < kSanitizeMaxLen && comment[n]) ++n;
    if (n == kSanitizeMaxLen && comment[n]) {
        // Cut inside the text: drop the continuation bytes (10xxxxxx) at the
        // cut and the lead byte they belong to.
        std::size_t k = n;
        while (k > 0 && (static_cast<unsigned char>(comment[k]) & 0xc0) == 0x80) --k;
        n = k;
    }
    std::memcpy(out, comment, n);
    out[n] = '\0';
    return 0;
}

}  // namespace np_score
