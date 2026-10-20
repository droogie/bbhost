// The game's console prints through the repeat filter (hle/guest_console.h):
// a line the game prints every frame is written once and then counted, a
// line in two pieces (YEBIS's file and line, then its message) is one line,
// threads' pieces do not mix, other lines and a line after a quiet window
// are written as they come, the counts come every window and at exit, and
// memory stays bounded under a flood of different lines. On the filter's own
// clock: no sleeping.
#include "hle/guest_console.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using bb::guest_console::Filter;

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

std::size_t count_of(const std::string& hay, const std::string& needle) {
    std::size_t n = 0;
    for (std::size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + needle.size())) ++n;
    return n;
}

const std::string kPrefix = "G:\\workspace\\yebis_git\\yebis_draft\\src\\GPU\\GPUPostEffect.cpp(31102) : PPFX ERROR: ";
const std::string kMessage = "CPostEffect::SetDepthOfFieldEdgeQuality(): Specified edge quality could not be used on this context.\n";

}  // namespace

int main() {
    // 1. YEBIS's error each frame at 60 fps for 25 s, in its two pieces: the
    // line once, a count at 10 s and at 20 s, and the rest at exit.
    {
        Filter f;
        std::string out;
        std::uint64_t t = 0;
        for (int frame = 0; frame < 1500; ++frame, t += 16) {
            out += f.feed(7, kPrefix, t);
            out += f.feed(7, kMessage, t);
        }
        CHECK(count_of(out, kPrefix + kMessage) == 3);  // written once, whole, and in the two counts
        CHECK(count_of(out, "the game printed this ") == 2);  // at 10 s and at 20 s
        CHECK(out.find("[bbhost] the game printed this 629 more times (over 10.1 s): " + kPrefix + kMessage) != std::string::npos);
        CHECK(out.find("[bbhost] the game printed this 630 more times (over 10.1 s): " + kPrefix + kMessage) != std::string::npos);
        const std::string rest = f.flush_all(t);
        CHECK(rest == "[bbhost] the game printed this 240 more times (over 3.8 s): " + kPrefix + kMessage);
        CHECK(f.suppressed_total() == 1499);
        std::uint64_t counted = 0;
        for (const std::string& part : {out, rest}) {
            for (std::size_t at = part.find("printed this "); at != std::string::npos; at = part.find("printed this ", at + 1)) {
                counted += std::strtoull(part.c_str() + at + 13, nullptr, 10);
            }
        }
        CHECK(counted == 1499);  // every repeat is in a count
    }

    // 2. Other lines are written as they come, between the repeats; two
    // threads' pieces do not mix.
    {
        Filter f;
        std::string out;
        out += f.feed(1, "loading ", 0);
        out += f.feed(2, "first\n", 0);
        out += f.feed(1, "map m24\n", 1);
        out += f.feed(2, "first\n", 2);     // a repeat: counted
        out += f.feed(2, "second\n", 3);    // a new line: written
        CHECK(out == "first\nloading map m24\nsecond\n");
        CHECK(f.suppressed_total() == 1);
    }

    // 3. After a quiet window the same line is written again - its count
    // first when one is pending.
    {
        Filter f;
        std::string out;
        out += f.feed(1, "tick\n", 0);
        out += f.feed(1, "tick\n", 500);
        out += f.feed(1, "tick\n", 600);  // the sweep runs once a second: no count yet
        CHECK(out == "tick\n");
        out += f.feed(1, "tick\n", 20000);  // quiet since 600: the count of two, then the line
        CHECK(out == "tick\n[bbhost] the game printed this 2 more times (over 0.6 s): tick\ntick\n");
    }

    // 4. A thread's unfinished line: written when it flushes, at exit, or
    // once it is too long to wait for.
    {
        Filter::Options o;
        o.max_partial = 64;
        Filter f(o);
        CHECK(f.feed(3, "progress: ", 0).empty());
        CHECK(f.flush_thread(3, 1) == "progress: ");
        CHECK(f.feed(3, "at exit", 2).empty());
        CHECK(f.flush_all(3) == "at exit\n");
        const std::string long_line(100, 'x');
        CHECK(f.feed(4, long_line, 4) == long_line);
    }

    // 5. A flood of different lines keeps the filter's memory bounded, and
    // every one of them is written.
    {
        Filter::Options o;
        o.max_lines = 256;
        Filter f(o);
        std::size_t written = 0;
        for (int i = 0; i < 20000; ++i) {
            const std::string out = f.feed(1, "line " + std::to_string(i) + "\n", static_cast<std::uint64_t>(i));
            written += count_of(out, "line ");
        }
        CHECK(written == 20000);
        CHECK(f.flush_all(20000).empty());
    }

    // 6. BBHOST_GUEST_PRINTS=all's filter: everything as it comes.
    {
        Filter::Options o;
        o.collapse = false;
        Filter f(o);
        std::string out;
        for (int i = 0; i < 3; ++i) out += f.feed(1, "same\n", static_cast<std::uint64_t>(i));
        CHECK(out == "same\nsame\nsame\n");
        CHECK(f.flush_all(3).empty());
    }

    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::printf("guest console: ok\n");
    return 0;
}
