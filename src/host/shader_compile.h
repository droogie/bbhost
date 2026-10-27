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
// "cancel" (or its stdin closing) drops what it has not begun and writes what
// it has. What it did is kept in <data>/bbhost/shader-compile.txt.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// main.cpp, before the configuration is read: whether the arguments ask for
// the compile session (the flag is taken out of them), and the arguments the
// setup window starts its child with.
bool shader_compile_take_flag(int& argc, char** argv);
void shader_compile_set_args(int argc, char** argv);
// The child's run, once the game's folders are known; its exit code.
int shader_compile_session();

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
    if (seconds < 50) return "約１分";
    const int lo = seconds < 120 ? 1 : static_cast<int>(seconds / 60.0);
    return "約 " + std::to_string(lo) + "-" + std::to_string(lo + 1) + " 分";
}
inline ShaderCompileAdvice shader_compile_advice(const ShaderCacheState& s, bool steam_deck, unsigned hw_threads) {
    ShaderCompileAdvice a;
    if (!s.have_caches) a.why = "シェーダーキャッシュがありません";
    else if (!s.translator_current) a.why = "別バージョンの bbhost のキャッシュです";
    else if (s.device_current == 0) a.why = "別の GPU またはドライバーのキャッシュです";
    a.recommend = !a.why.empty();
    double last = -1;
    bool done = false;
    for (const std::string& l : s.last) {
        if (l == "result done") done = true;
        else if (l.rfind("seconds ", 0) == 0) last = std::atof(l.c_str() + 8);
    }
    if (done && last > 0) a.time = shader_compile_minutes(last) + " （前回の所要時間: " + std::to_string(static_cast<long long>(last)) + " 秒）";
    else if (steam_deck) a.time = "Steam Deck で約２～３分";
    else if (hw_threads >= 16) a.time = "この PC で約１～２分";
    else if (hw_threads >= 8) a.time = "この PC で約２～３分";
    else a.time = "この PC で約３～５分";
    return a;
}
// This machine's Vulkan devices, "vendor:device uuid" each (pipelineCacheUUID).
std::vector<std::string> shader_vulkan_devices();

// The child, from the setup window.
class ShaderCompileJob {
public:
    ~ShaderCompileJob();
    bool start(std::string& error);
    void cancel();
    bool running() const;
    // Under the job's own lock, what the child said last.
    struct View {
        bool running = false, finished = false, cancelled = false, failed_to_run = false;
        std::uint64_t done = 0, total = 0, failed = 0, hlsl = 0, lift = 0;
        std::uint64_t recipes_done = 0, recipes_total = 0;  // pipelines from the manifest and the shipped list
        double seconds = 0, eta = -1;
        std::string phase, error;
        std::vector<std::string> reasons;
    };
    View view() const;

private:
    void reader();
    mutable std::mutex mu_;
    View v_;
    void* proc_ = nullptr;  // SDL_Process
    std::thread thread_;
};
