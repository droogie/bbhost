// The setup window's side of "Compile shaders now" (host/shader_compile_read.h):
// the child's stdout read to its end on a scripted stream - an end of the
// stream or a failed read while the child runs, the lines after it, a last
// line without its newline, a child that exits having said nothing - and the
// result taken from the child's record when its last line did not arrive,
// only from a record of this run.
#include "host/shader_compile_read.h"

#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

// A child's stdout as a script, on a clock that the reader's pauses move:
// what each read returns while the child runs (bytes, or none and why), the
// pause at which it exits (whether or not anyone asks), and what is still in
// the pipe then - its unread bytes, and `left` - read before the end.
struct Script final : ShaderCompileStream {
    struct Read {
        std::string bytes;
        Status st = kReady;  // when no bytes
    };
    std::deque<Read> running;
    int exit_at = 0;               // pauses until it has exited
    std::deque<std::string> left;  // also in the pipe at its exit
    int code = 0;
    int pauses = 0, reads_after_exit = 0;
    bool gone = false;
    std::size_t give(std::string& b, char* p, std::size_t n) {
        const std::size_t k = b.size() < n ? b.size() : n;
        std::memcpy(p, b.data(), k);
        b.erase(0, k);
        return k;
    }
    void tick() {
        if (gone || pauses < exit_at) return;
        gone = true;
        // What it wrote and nobody read yet is still in the pipe.
        for (auto it = running.rbegin(); it != running.rend(); ++it)
            if (!it->bytes.empty()) left.push_front(it->bytes);
        running.clear();
    }
    std::size_t read(char* p, std::size_t n, Status& st) override {
        tick();
        if (gone) {
            ++reads_after_exit;
            if (left.empty()) {
                st = kEnd;
                return 0;
            }
            const std::size_t k = give(left.front(), p, n);
            if (left.front().empty()) left.pop_front();
            return k;
        }
        if (running.empty()) {
            st = kNotReady;
            return 0;
        }
        Read& r = running.front();
        if (r.bytes.empty()) {
            st = r.st;
            running.pop_front();
            return 0;
        }
        const std::size_t k = give(r.bytes, p, n);
        if (r.bytes.empty()) running.pop_front();
        return k;
    }
    bool exited(int& c) override {
        tick();
        if (gone) c = code;
        return gone;
    }
    void pause() override {
        if (++pauses > 100000) {
            std::fprintf(stderr, "the reader never stopped\n");
            std::exit(1);
        }
    }
};

struct Run {
    ShaderCompileView v;
    std::vector<std::string> lines;
    std::vector<ShaderCompileStream::Status> notes;
    int code = -99;
};

Run read_all(Script& s) {
    Run r;
    r.code = shader_compile_read(
        s,
        [&r](const std::string& l) {
            r.lines.push_back(l);
            shader_compile_take_line(r.v, l);
        },
        [&r](ShaderCompileStream::Status st) { r.notes.push_back(st); });
    return r;
}

const char* kProgress = "compile-shaders: progress 3 12 0 2 0 0.4 0 3\n";
const char* kReason = "compile-shaders: reason VS translation: LDS use outside a compute shader x2\n";
const char* kDone = "compile-shaders: done 12 12 2 10 1 1.5 3 3\n";

// A record as the child writes it, of run `run`.
std::vector<std::string> record(const std::string& run, const std::string& result = "done") {
    std::vector<std::string> r{"result " + result};
    if (!run.empty()) r.push_back("run " + run);
    r.insert(r.end(), {"when 2026-11-06 12:00", "bbhost v0.2.27", "translator 0123", "device 10de:2c02 a GPU", "shaders HLSL",
                       "stages 4682 of 4682, 167 failed", "hlsl 4005 with DXC's module (3708 compiled now), 3 kept the lift",
                       "pipelines 1518 of 1518 from earlier sessions and the shipped list", "seconds 48",
                       "failed VS translation: LDS use outside a compute shader x167", "lift an image query x3"});
    return r;
}

}  // namespace

int main() {
    // The lines as they come: all of them, split across reads, CRLF or LF.
    {
        Script s;
        s.running = {{"compile-shaders: phase compi"}, {"ling\r\n"}, {kProgress}, {std::string(kReason) + kDone}};
        s.exit_at = 3;
        const Run r = read_all(s);
        CHECK(r.code == 0);
        CHECK(r.lines.size() == 4 && r.lines[0] == "compile-shaders: phase compiling");
        CHECK(r.notes.empty());
        CHECK(r.v.finished && r.v.done == 12 && r.v.total == 12 && r.v.failed == 2 && r.v.hlsl == 10 && r.v.lift == 1);
        CHECK(r.v.recipes_done == 3 && r.v.recipes_total == 3 && r.v.seconds == 1.5);
        CHECK(r.v.reasons.size() == 1 && r.v.reasons[0] == "VS translation: LDS use outside a compute shader x2");
    }
    // The end of the stream while the child runs (a stdout taken from the
    // window), then its lines after all: read on, said once, nothing lost.
    for (const ShaderCompileStream::Status why : {ShaderCompileStream::kEnd, ShaderCompileStream::kError}) {
        Script s;
        Script::Read gap;
        gap.st = why;
        s.running = {{kProgress}, gap, gap, gap, {kReason}, gap, {kDone}};
        s.exit_at = 20;
        const Run r = read_all(s);
        CHECK(r.notes.size() == 1 && r.notes[0] == why);
        CHECK(r.lines.size() == 3);
        CHECK(r.v.finished && r.v.done == 12);
        CHECK(r.v.reasons.size() == 1);
        // Waited for its exit rather than taken the end at its word.
        CHECK(s.gone && s.pauses >= 4);
    }
    // What it wrote before it exited is all read, after the exit too: its
    // last lines unread at its exit, and a line split across the pipe.
    {
        Script s;
        s.running = {{kProgress}};
        s.exit_at = 2;
        s.left = {std::string(kReason), "compile-shaders: done 12 12 2 10 1", " 1.5 3 3\n"};
        const Run r = read_all(s);
        CHECK(r.lines.size() == 3 && r.v.finished && r.v.recipes_total == 3);
        CHECK(s.reads_after_exit >= 4);
    }
    // The last line without its newline is a line.
    {
        Script s;
        s.running = {{kProgress}, {"compile-shaders: done 12 12 0 12 0 2.0 0 0\r"}};
        s.exit_at = 2;
        const Run r = read_all(s);
        CHECK(r.lines.size() == 2 && r.lines[1] == "compile-shaders: done 12 12 0 12 0 2.0 0 0");
        CHECK(r.v.finished && r.v.done == 12);
    }
    {
        Script s;
        s.running = {{kProgress}};
        s.left = {"compile-shaders: cancelled 5 12 0 4 0 1.0 0 3"};
        s.code = 1;
        const Run r = read_all(s);
        CHECK(r.code == 1 && r.v.cancelled && !r.v.finished && r.v.done == 5);
    }
    // A child that exits having said nothing.
    {
        Script s;
        Script::Read end;
        end.st = ShaderCompileStream::kEnd;
        s.running = {end, end};
        s.exit_at = 5;
        s.code = 0;
        Run r = read_all(s);
        CHECK(r.lines.empty() && r.code == 0 && r.notes.size() == 1);
        // No record: it ended early.
        ShaderCompileView v = r.v;
        CHECK(!shader_compile_finish(v, r.code, {}, "00ff"));
        CHECK(!v.running && !v.finished && v.error == "the compile ended early (exit 0)" && v.phase == "stopped");
        // A record of another run, or of a child started without one, or an
        // older one without a run line: not this run's.
        v = r.v;
        CHECK(!shader_compile_finish(v, 0, record("1234"), "00ff") && !v.finished && !v.from_record);
        v = r.v;
        CHECK(!shader_compile_finish(v, 0, record("00ff"), "") && !v.finished);
        v = r.v;
        CHECK(!shader_compile_finish(v, 0, record(""), "00ff") && !v.finished);
        // This run's record: its result, counts and reasons.
        v = r.v;
        CHECK(shader_compile_finish(v, 0, record("00ff"), "00ff"));
        CHECK(v.finished && v.from_record && v.error.empty() && v.phase == "done" && !v.running);
        CHECK(v.done == 4682 && v.total == 4682 && v.failed == 167 && v.hlsl == 4005 && v.lift == 3);
        CHECK(v.recipes_done == 1518 && v.recipes_total == 1518 && v.seconds == 48);
        CHECK(v.reasons.size() == 2 && v.reasons[0] == "VS translation: LDS use outside a compute shader x167" &&
              v.reasons[1] == "the lift kept: an image query x3");
        // A cancelled one, with the cancelled exit; a result and an exit
        // that do not agree, or an exit no compile ends with, is no result.
        v = r.v;
        CHECK(shader_compile_finish(v, 1, record("00ff", "cancelled"), "00ff") && v.cancelled && !v.finished && v.phase == "cancelled");
        v = r.v;
        CHECK(!shader_compile_finish(v, 1, record("00ff"), "00ff") && v.error == "the compile ended early (exit 1)");
        v = r.v;
        CHECK(!shader_compile_finish(v, 3, record("00ff"), "00ff") && v.error == "the compile ended early (exit 3)");
    }
    // Its own last line wins over any record, and its own error stays.
    {
        ShaderCompileView v;
        shader_compile_take_line(v, "compile-shaders: done 12 12 0 12 0 2.0 0 0");
        CHECK(!shader_compile_finish(v, 0, record("00ff"), "00ff") && v.finished && !v.from_record && v.done == 12);
        ShaderCompileView e;
        shader_compile_take_line(e, "compile-shaders: error no Vulkan device could be started");
        CHECK(!shader_compile_finish(e, 3, {}, "00ff") && e.error == "no Vulkan device could be started" && e.phase == "stopped");
    }
    // Lines that are not the child's change nothing.
    {
        ShaderCompileView v;
        shader_compile_take_line(v, "a driver's own line");
        shader_compile_take_line(v, "xcompile-shaders: done 1 1 0 0 0 0.1 0 0");
        CHECK(!v.finished && v.done == 0 && v.phase.empty());
    }
    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::printf("shader compile read: ok\n");
    return 0;
}
