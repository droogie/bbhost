// The game's own prints to the console (printf, puts, fputs ... to stdout or
// stderr): a line the game prints again and again is written once, then a
// count - every ten seconds while it keeps coming, and at exit - so a print
// in a loop cannot flood the console or cost the frames a console write takes
// (slow on Windows). YEBIS prints an error each frame where a frame's
// parameters ask for something its context cannot do (Cainhurst's
// "CPostEffect::SetDepthOfFieldEdgeQuality(): Specified edge quality could not
// be used on this context.", 2026-10-19), as it would on a PS4's debug
// output. Other lines - and the same line again after ten quiet seconds -
// are written as they come. Text is filtered by whole lines: a line printed
// in pieces (YEBIS's file and line, then its message) is written when its
// newline comes, or when its thread flushes the stream.
// BBHOST_GUEST_PRINTS=all writes everything as it comes, as before.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <unordered_map>

namespace bb::guest_console {

// The filter, on its own clock (milliseconds) and text: what to write for
// what a thread printed. One per stream.
class Filter {
public:
    struct Options {
        bool collapse = true;              // false: everything as it comes
        std::uint64_t window_ms = 10000;   // a repeat within this of the last is counted, and the counts written this often
        std::size_t max_lines = 1024;      // lines remembered (those not repeating are forgotten first)
        std::size_t max_partial = 4096;    // a thread's unfinished line written once it is this long
    };
    Filter() : Filter(Options{}) {}
    explicit Filter(const Options& o) : o_(o) {}

    // `text` printed by `thread` at `now_ms`: what to write now (often nothing).
    std::string feed(std::uint64_t thread, std::string_view text, std::uint64_t now_ms);
    // `thread` flushed the stream: its unfinished line, filtered as a line.
    std::string flush_thread(std::uint64_t thread, std::uint64_t now_ms);
    // At exit: every unfinished line and every count not written yet.
    std::string flush_all(std::uint64_t now_ms);

    std::uint64_t suppressed_total() const { return suppressed_total_; }

private:
    struct Seen {
        std::uint64_t last_ms = 0;     // the last time it came
        std::uint64_t since_ms = 0;    // the start of the count not written yet
        std::uint64_t count = 0;       // repeats not written
    };
    void line(std::string_view l, std::uint64_t now_ms, std::string& out);
    void counts(std::uint64_t now_ms, bool all, std::string& out);
    static void count_line(std::string_view l, const Seen& s, std::uint64_t now_ms, std::string& out);

    Options o_;
    std::unordered_map<std::uint64_t, std::string> partial_;
    std::unordered_map<std::string, Seen> seen_;
    std::uint64_t last_sweep_ms_ = 0;
    std::uint64_t suppressed_total_ = 0;
};

// The guest wrote `n` bytes to `f`: through stdout's or stderr's filter when
// `f` is one of them (true: written or counted), else nothing is done (false).
bool write(std::FILE* f, const char* data, std::size_t n);
// The guest flushed `f`: its thread's unfinished line on stdout or stderr.
void flush(std::FILE* f);
// The counts and unfinished lines, written now (also at exit).
void flush_all();

}  // namespace bb::guest_console
