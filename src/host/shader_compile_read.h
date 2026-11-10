// The setup window's side of "Compile shaders now" (host/shader_compile.h):
// the child's stdout read to its end, the lines it prints, and the record it
// leaves for a compile whose last lines did not reach the window. Free of
// SDL, so a test drives it with a stream of its own
// (tests/shader_compile_read_test.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// What the child said last, as the setup window shows it.
struct ShaderCompileView {
    bool running = false, finished = false, cancelled = false, failed_to_run = false;
    bool from_record = false;  // the result is its record's: the last lines did not arrive
    std::uint64_t done = 0, total = 0, failed = 0, hlsl = 0, lift = 0;
    std::uint64_t recipes_done = 0, recipes_total = 0;  // pipelines from the manifest and the shipped list
    double seconds = 0, eta = -1;
    std::string phase, error;
    std::vector<std::string> reasons;
};

// One line of the child's stdout: "compile-shaders: progress|done|cancelled
// <done> <total> <failed> <hlsl> <lift> <seconds> <pipelines done> <of>",
// "... phase <what>", "... reason <why>", "... error <what>". Other lines
// are not the child's.
inline void shader_compile_take_line(ShaderCompileView& v, const std::string& line) {
    static const std::string pre = "compile-shaders: ";
    if (line.rfind(pre, 0) != 0) return;
    const std::string rest = line.substr(pre.size());
    unsigned long long d = 0, t = 0, f = 0, h = 0, l = 0, rd = 0, rt = 0;
    double secs = 0;
    if (std::sscanf(rest.c_str(), "progress %llu %llu %llu %llu %llu %lf %llu %llu", &d, &t, &f, &h, &l, &secs, &rd, &rt) >= 6 ||
        std::sscanf(rest.c_str(), "done %llu %llu %llu %llu %llu %lf %llu %llu", &d, &t, &f, &h, &l, &secs, &rd, &rt) >= 6 ||
        std::sscanf(rest.c_str(), "cancelled %llu %llu %llu %llu %llu %lf %llu %llu", &d, &t, &f, &h, &l, &secs, &rd, &rt) >= 6) {
        v.recipes_done = rd;
        v.recipes_total = rt;
        v.done = d;
        v.total = t;
        v.failed = f;
        v.hlsl = h;
        v.lift = l;
        v.seconds = secs;
        // Time left from the rate since the first shader was done.
        v.eta = d > 20 && t > d ? secs * static_cast<double>(t - d) / static_cast<double>(d) : -1;
        if (rest.rfind("done", 0) == 0) v.finished = true;
        if (rest.rfind("cancelled", 0) == 0) v.cancelled = true;
    } else if (rest.rfind("phase ", 0) == 0) {
        v.phase = rest.substr(6);
    } else if (rest.rfind("reason ", 0) == 0) {
        v.reasons.push_back(rest.substr(7));
    } else if (rest.rfind("error ", 0) == 0) {
        v.error = rest.substr(6);
    }
}

// The child's stdout as the reader sees it: a read's bytes and, when it
// read none, why (SDL_ReadIO and SDL_GetIOStatus), and whether the process
// has ended (SDL_WaitProcess without blocking).
struct ShaderCompileStream {
    enum Status { kReady, kNotReady, kEnd, kError };
    virtual ~ShaderCompileStream() = default;
    virtual std::size_t read(char* p, std::size_t n, Status& st) = 0;
    virtual bool exited(int& code) = 0;
    virtual void pause() = 0;  // between empty reads
};

// The size of each read. SDL's Windows stream serves reads under 1024 bytes
// from a buffer of its own, and drops what that buffer still held when the
// next ReadFile fails (a FIXME in SDL_iostream.c); a read of 1024 or more
// goes to ReadFile itself, into the caller's buffer, and loses nothing.
constexpr std::size_t kShaderCompileReadChunk = 4096;
static_assert(kShaderCompileReadChunk >= 1024, "reads under 1024 bytes go through SDL's Windows readahead");

// Reads the child's stdout to its end and returns the child's exit code.
// Every line goes to `line` - the last one too, if it has no newline. An
// empty read is the end only once the process has exited: the end of the
// stream or a failed read while the child runs (a pipe its stdout no longer
// is, a read that went wrong) is said once to `note` and the reading goes
// on, so a child that runs on is waited for and anything it still gets
// through is read. Once it has exited, everything it wrote is in the pipe:
// that is read to the stream's end.
template <class Line, class Note>
int shader_compile_read(ShaderCompileStream& s, Line&& line, Note&& note) {
    std::string buf;
    char chunk[kShaderCompileReadChunk];
    const auto feed = [&](std::size_t n) {
        buf.append(chunk, n);
        for (std::size_t nl; (nl = buf.find('\n')) != std::string::npos;) {
            std::string l = buf.substr(0, nl);
            if (!l.empty() && l.back() == '\r') l.pop_back();
            buf.erase(0, nl + 1);
            line(l);
        }
    };
    int code = 0;
    bool noted = false;
    for (;;) {
        ShaderCompileStream::Status st = ShaderCompileStream::kReady;
        const std::size_t n = s.read(chunk, sizeof(chunk), st);
        if (n > 0) {
            feed(n);
            continue;
        }
        if (s.exited(code)) break;
        if (st != ShaderCompileStream::kNotReady && !noted) {
            noted = true;
            note(st);
        }
        s.pause();
    }
    for (;;) {
        ShaderCompileStream::Status st = ShaderCompileStream::kReady;
        const std::size_t n = s.read(chunk, sizeof(chunk), st);
        if (n == 0) break;
        feed(n);
    }
    if (!buf.empty()) {
        if (buf.back() == '\r') buf.pop_back();
        line(buf);
    }
    return code;
}

// The record a compile leaves (<data>/bbhost/shader-compile.txt, written
// before its last lines), into the view: the result, the counts and the
// reasons. Only a record of this run - its "run" line names `run`, the id
// the window started the child with - and only a finished or cancelled one.
inline bool shader_compile_take_record(ShaderCompileView& v, const std::vector<std::string>& record, const std::string& run) {
    if (run.empty()) return false;
    bool mine = false;
    std::string result;
    ShaderCompileView r = v;
    r.reasons.clear();
    for (const std::string& l : record) {
        const std::size_t sp = l.find(' ');
        const std::string k = l.substr(0, sp), val = sp == std::string::npos ? std::string() : l.substr(sp + 1);
        unsigned long long a = 0, b = 0, c = 0;
        if (k == "run") {
            mine = val == run;
        } else if (k == "result") {
            result = val;
        } else if (k == "stages" && std::sscanf(val.c_str(), "%llu of %llu, %llu failed", &a, &b, &c) == 3) {
            r.done = a;
            r.total = b;
            r.failed = c;
        } else if (k == "hlsl" && std::sscanf(val.c_str(), "%llu with DXC's module (%llu compiled now), %llu kept the lift", &a, &b, &c) == 3) {
            r.hlsl = a;
            r.lift = c;
        } else if (k == "pipelines" && std::sscanf(val.c_str(), "%llu of %llu", &a, &b) == 2) {
            r.recipes_done = a;
            r.recipes_total = b;
        } else if (k == "seconds") {
            r.seconds = std::atof(val.c_str());
        } else if (k == "failed") {
            r.reasons.push_back(val);
        } else if (k == "lift") {
            r.reasons.push_back("the lift kept: " + val);
        }
    }
    if (!mine || (result != "done" && result != "cancelled")) return false;
    r.finished = result == "done";
    r.cancelled = result == "cancelled";
    r.from_record = true;
    r.eta = -1;
    v = r;
    return true;
}

// The end of a compile, once its stdout is read and it has exited with
// `code`: its result from its last line, or - when that did not arrive and
// it ended as a compile ends (0 done, 1 cancelled) - from its record (read
// from disk by the caller) if the record is this run's, else that it ended
// early. Returns whether the record was used.
inline bool shader_compile_finish(ShaderCompileView& v, int code, const std::vector<std::string>& record, const std::string& run) {
    v.running = false;
    bool used = false;
    if (!v.finished && !v.cancelled && v.error.empty() && (code == 0 || code == 1)) {
        ShaderCompileView r = v;
        if (shader_compile_take_record(r, record, run) && (r.finished ? code == 0 : code == 1)) {
            v = r;
            used = true;
        }
    }
    if (!v.finished && !v.cancelled && v.error.empty()) v.error = "the compile ended early (exit " + std::to_string(code) + ")";
    v.phase = v.finished ? "done" : v.cancelled ? "cancelled" : "stopped";
    return used;
}
