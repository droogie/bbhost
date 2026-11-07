// "Compile shaders now" in the setup window (host/launcher.h).
//
// The game compiles its shaders itself - at the start, and DXC's modules
// behind the game for the next start (host/translation_cache.h) - and a player
// who skips this loses nothing. The button does all of it up front: it starts
// bbhost again as a child process, `bbhost --compile-shaders` with the same
// arguments, which starts the GPU device with no game and compiles every
// shader the game's bundles hold (the debug menu's aside) and every stage and
// pipeline the manifests name - translations, lifts, DXC's modules and the
// driver's pipelines - into the same caches, on the same precompile workers
// the game uses (host_gpu_set_compile_session). The child prints its
// progress on stdout ("compile-shaders: ..." lines), logs to
// <data>/bbhost/shader-compile.log, ends with the caches written, and on
// "cancel" drops what it has not begun and writes what it has. What it did
// is kept in <data>/bbhost/shader-compile.txt, written before its last lines,
// with the run id the window started it with (--compile-shaders=<id>): a
// window that did not get those lines takes the result from there
// (host/shader_compile_read.h).
#pragma once

#include "host/shader_compile_read.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// main.cpp, before the configuration is read: whether the arguments ask for
// the compile session (--compile-shaders or --compile-shaders=<run id>; the
// flag is taken out of them), and the arguments the setup window starts its
// child with.
bool shader_compile_take_flag(int& argc, char** argv);
void shader_compile_set_args(int argc, char** argv);
// This executable started again with the arguments it was started with,
// `flag` first: the compile's child (--compile-shaders), the game the setup
// window starts while it stays open (--no-setup), the setup window again
// (--setup), its output piped to this one when `pipe_stdio`. The SDL_Process,
// or null with `error` set.
void* host_start_again(const char* flag, bool pipe_stdio, std::string& error);
// The child's run, once the game's folders are known; its exit code.
// BBHOST_TEST_COMPILE pretends the compile (no GPU) for the self-test below.
int shader_compile_session();
// `bbhost --compile-selftest DIR` (the shader_compile_e2e test): the setup
// window's side end to end - this executable started again as the window
// starts it, its stdout read, its record taken - with children that pretend
// the compile (BBHOST_TEST_COMPILE), in DIR (a game folder, eboot and
// configuration of their own). The exit code says; the lines are also kept
// in DIR/selftest.txt. "" when the arguments do not ask for it.
std::string shader_compile_selftest_dir(int argc, char** argv);
int shader_compile_selftest(const std::string& dir);

// What the setup window shows before anything runs.
struct ShaderCacheState {
    std::string dir;                // <data>/bbhost
    bool translator_current = false;  // shader-cache-build.txt names this bbhost's translator
    bool have_caches = false;
    std::string cache_translator;   // the one the caches were made by ("" none)
    int device_current = -1;        // the pipeline cache's device among this machine's: 1 yes, 0 no, -1 not known (yet)
    std::string cache_device;       // "vendor:device uuid" from the pipeline cache's header
    std::uint64_t translation_bytes = 0, pipeline_bytes = 0, stage_manifest_bytes = 0, pipeline_manifest_bytes = 0;
    // shader-compile.txt: the last compile session's result.
    std::vector<std::string> last;  // its lines
};
// `data` is the data folder the game will use ("" = the default). The device
// check enumerates this machine's Vulkan devices: done on a thread of its own
// by ShaderCompileJob::refresh.
ShaderCacheState shader_cache_state(const std::string& data, const std::vector<std::string>& devices);

// Whether the setup window recommends "Compile shaders now" (its Shaders tab,
// and a line on its main tab): when the caches were not made by this bbhost's
// translator or for this machine's graphics driver, or there are none - a new
// install, an update, a new driver. Only a recommendation, one click: the game
// compiles what it lacks itself. `time` is what the compile is likely to take:
// the last finished one's seconds if there was one (shader-compile.txt), else a
// guess from the machine: a Steam Deck took ~150 s, a laptop's 8-core Ryzen
// (16 threads) ~90 s, a 16-worker Xeon ~60 s.
struct ShaderCompileAdvice {
    bool recommend = false;
    std::string why;   // a sentence without its full stop
    std::string time;  // "about 2-3 minutes on a Steam Deck"
};
inline std::string shader_compile_minutes(double seconds) {
    if (seconds < 50) return "about a minute";
    const int lo = seconds < 120 ? 1 : static_cast<int>(seconds / 60.0);
    return "about " + std::to_string(lo) + "-" + std::to_string(lo + 1) + " minutes";
}
inline ShaderCompileAdvice shader_compile_advice(const ShaderCacheState& s, bool steam_deck, unsigned hw_threads) {
    ShaderCompileAdvice a;
    if (!s.have_caches) a.why = "There are no shader caches yet";
    else if (!s.translator_current) a.why = "The shader caches were made by another bbhost version";
    else if (s.device_current == 0) a.why = "The shader caches were made for another GPU or graphics driver";
    a.recommend = !a.why.empty();
    double last = -1;
    bool done = false;
    for (const std::string& l : s.last) {
        if (l == "result done") done = true;
        else if (l.rfind("seconds ", 0) == 0) last = std::atof(l.c_str() + 8);
    }
    if (done && last > 0) a.time = shader_compile_minutes(last) + " (the last compile here took " + std::to_string(static_cast<long long>(last)) + " s)";
    else if (steam_deck) a.time = "about 2-3 minutes on a Steam Deck";
    else if (hw_threads >= 16) a.time = "about 1-2 minutes on this PC";
    else if (hw_threads >= 8) a.time = "about 2-3 minutes on this PC";
    else a.time = "about 3-5 minutes on this PC";
    return a;
}
// This machine's Vulkan devices, "vendor:device uuid" each (pipelineCacheUUID).
std::vector<std::string> shader_vulkan_devices();

// The child, from the setup window.
class ShaderCompileJob {
public:
    ~ShaderCompileJob();
    // `data` is the data folder the child's configuration names, as for
    // shader_cache_state: where its record is.
    bool start(const std::string& data, std::string& error);
    void cancel();
    bool running() const;
    // Under the job's own lock, what the child said last.
    using View = ShaderCompileView;
    View view() const;

private:
    void reader();
    mutable std::mutex mu_;
    View v_;
    void* proc_ = nullptr;  // SDL_Process
    std::thread thread_;
    std::string run_;     // the id the child was started with, which its record carries
    std::string record_;  // <data>/bbhost/shader-compile.txt
};
