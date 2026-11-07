#include "host/shader_compile.h"

#include "bbhost_version.h"
#include "core/config.h"
#include "hle/fs.h"
#include "host/gpu.h"
#include "host/options.h"
#include "host/translation_cache.h"
#include "log.h"
#include "translator_fingerprint.h"

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

std::vector<std::string> g_args;  // the arguments the setup window's child is started with
std::string g_exe_name;          // this executable's file name (argv[0]'s)
std::string g_run;               // the child's: the run id it was started with (--compile-shaders=<id>), for its record

std::string data_dir(const std::string& data) {
    return (data.empty() ? config_default_data_dir() : data) + "/bbhost";
}

std::uint64_t size_of(const fs::path& p) {
    std::error_code ec;
    const auto n = fs::file_size(p, ec);
    return ec ? 0 : static_cast<std::uint64_t>(n);
}

std::string hex(const std::uint8_t* p, std::size_t n) {
    std::string s;
    char b[3];
    for (std::size_t i = 0; i < n; ++i) {
        std::snprintf(b, sizeof(b), "%02x", p[i]);
        s += b;
    }
    return s;
}

}  // namespace

bool shader_compile_take_flag(int& argc, char** argv) {
    bool found = false;
    int out = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--compile-shaders") == 0) {
            found = true;
            continue;
        }
        if (std::strncmp(argv[i], "--compile-shaders=", 18) == 0) {
            found = true;
            const std::string id = argv[i] + 18;
            if (!id.empty() && id.size() <= 32 && id.find_first_not_of("0123456789abcdef") == std::string::npos) g_run = id;
            continue;
        }
        argv[out++] = argv[i];
    }
    if (found) {
        argc = out;
        argv[argc] = nullptr;
    }
    return found;
}

void shader_compile_set_args(int argc, char** argv) {
    g_args.clear();
    if (argc > 0 && argv[0]) g_exe_name = fs::path(argv[0]).filename().string();
    for (int i = 1; i < argc; ++i) {
        // The setup window's own switch would open it again in the child.
        if (std::strcmp(argv[i], "--setup") == 0) continue;
        g_args.emplace_back(argv[i]);
    }
}

// ---- The child ----------------------------------------------------------------

int shader_compile_session() {
    const char* root = hle_fs_data_root();
    const std::string dir = std::string(root && *root ? root : ".") + "/bbhost";
    std::error_code ec;
    fs::create_directories(dir, ec);
    const std::string log_path = dir + "/shader-compile.log";
    // The log to its own file: stdout is the setup window's.
#if defined(_WIN32)
    std::freopen(log_path.c_str(), "w", stderr);
#else
    if (const int fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644); fd >= 0) {
        ::dup2(fd, 2);
        ::close(fd);
    }
#endif
    host_log("compile-shaders: bbhost %s, every shader into the caches in %s", BBHOST_VERSION, dir.c_str());
    const auto say = [](const std::string& line) {
        std::fputs(("compile-shaders: " + line + "\n").c_str(), stdout);
        std::fflush(stdout);
    };
    host_gpu_set_compile_session(true);
    say("phase starting the GPU");
    if (!host_gpu_init(nullptr, 0, false)) {
        say("error no Vulkan device could be started (see " + log_path + ")");
        std::fflush(stderr);
        std::_Exit(3);
    }
    const std::string device = host_gpu_device_identity();
    say("device " + device);
    // "cancel" on stdin ends it early: what has not begun is dropped, what
    // has is written. Stdin closing does not (a run from a script with no
    // input goes to its end; the setup window sends "cancel" when it closes).
    static std::atomic<bool> cancelled{false};
    std::thread([] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.rfind("cancel", 0) == 0) {
                cancelled.store(true);
                host_gpu_compile_cancel();
                return;
            }
        }
    }).detach();
    const auto t0 = std::chrono::steady_clock::now();
    const auto seconds = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    say("phase compiling");
    HostShaderCompileProgress p;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        p = host_gpu_compile_progress();
        std::uint64_t taken = 0, compiled = 0, kept = 0;
        std::string reasons;
        gpu::hlsl_counts(taken, compiled, kept, reasons);
        char b[240];
        std::snprintf(b, sizeof(b), "progress %llu %llu %llu %llu %llu %.1f %llu %llu", static_cast<unsigned long long>(p.done),
                      static_cast<unsigned long long>(p.total), static_cast<unsigned long long>(p.failed),
                      static_cast<unsigned long long>(taken), static_cast<unsigned long long>(kept), seconds(),
                      static_cast<unsigned long long>(p.recipes_done), static_cast<unsigned long long>(p.recipes_total));
        say(b);
        if (p.finished && (p.listed || cancelled.load())) break;
    }
    say("phase writing the caches");
    host_gpu_save_pipeline_cache_at_exit();
    std::uint64_t taken = 0, compiled = 0, kept = 0;
    std::string hlsl_reasons;
    gpu::hlsl_counts(taken, compiled, kept, hlsl_reasons);
    const bool hlsl = gpu::hlsl_on();
    // One answer for the record, the last line and the exit code, whenever
    // a "cancel" comes in.
    const bool was_cancelled = cancelled.load();
    // The record the setup window reads (shader_cache_state), and takes the
    // result from when the lines below do not reach it: before them, with
    // the run id it started this child with.
    {
        std::ofstream out(dir + "/shader-compile.txt", std::ios::trunc);
        const std::time_t now = std::time(nullptr);
        char when[64];
        std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M", std::localtime(&now));
        out << "result " << (was_cancelled ? "cancelled" : "done") << "\n";
        if (!g_run.empty()) out << "run " << g_run << "\n";
        out << "when " << when << "\n"
            << "bbhost " << BBHOST_VERSION << "\n"
            << "translator " << BBHOST_TRANSLATOR_FINGERPRINT << "\n"
            << "device " << device << "\n"
            << "shaders " << (hlsl ? "HLSL" : "Lifted") << "\n"
            << "stages " << p.done << " of " << p.total << ", " << p.failed << " failed\n"
            << "hlsl " << taken << " with DXC's module (" << compiled << " compiled now), " << kept << " kept the lift\n"
            << "pipelines " << p.recipes_done << " of " << p.recipes_total << " from earlier sessions and the shipped list\n"
            << "seconds " << static_cast<long long>(seconds()) << "\n";
        std::istringstream r1(p.reasons), r2(hlsl_reasons);
        for (std::string l; std::getline(r1, l);) out << "failed " << l << "\n";
        for (std::string l; std::getline(r2, l);) out << "lift " << l << "\n";
    }
    {
        std::istringstream r1(p.reasons), r2(hlsl_reasons);
        for (std::string l; std::getline(r1, l);) say("reason " + l);
        for (std::string l; std::getline(r2, l);) say("reason the lift kept: " + l);
    }
    char b[240];
    std::snprintf(b, sizeof(b), "%s %llu %llu %llu %llu %llu %.1f %llu %llu", was_cancelled ? "cancelled" : "done",
                  static_cast<unsigned long long>(p.done), static_cast<unsigned long long>(p.total), static_cast<unsigned long long>(p.failed),
                  static_cast<unsigned long long>(taken), static_cast<unsigned long long>(kept), seconds(),
                  static_cast<unsigned long long>(p.recipes_done), static_cast<unsigned long long>(p.recipes_total));
    say(b);
    host_log("compile-shaders: %s", b);
    // Not fflush(nullptr): the stdin reader holds stdin's lock in getline.
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(was_cancelled ? 1 : 0);
}

// ---- The setup window's side ------------------------------------------------------

std::vector<std::string> shader_vulkan_devices() {
    std::vector<std::string> out;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "bbhost launcher", 1, nullptr, 0, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return out;
    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    if (n) vkEnumeratePhysicalDevices(inst, &n, devs.data());
    for (VkPhysicalDevice d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        char b[64];
        std::snprintf(b, sizeof(b), "%04x:%04x ", p.vendorID, p.deviceID);
        out.push_back(b + hex(p.pipelineCacheUUID, VK_UUID_SIZE) + " " + p.deviceName);
    }
    vkDestroyInstance(inst, nullptr);
    return out;
}

ShaderCacheState shader_cache_state(const std::string& data, const std::vector<std::string>& devices) {
    ShaderCacheState s;
    s.dir = data_dir(data);
    const fs::path d(s.dir);
    {
        std::ifstream in(d / "shader-cache-build.txt");
        std::getline(in, s.cache_translator);
    }
    s.translation_bytes = size_of(d / "translation-cache.bin");
    s.pipeline_bytes = size_of(d / "vulkan-pipeline-cache.bin");
    s.stage_manifest_bytes = size_of(d / "stage-manifest.bin");
    s.pipeline_manifest_bytes = size_of(d / "pipeline-manifest.bin");
    s.have_caches = s.translation_bytes > 0 || s.pipeline_bytes > 0;
    s.translator_current = s.cache_translator == BBHOST_TRANSLATOR_FINGERPRINT;
    // The pipeline cache's header: length, version, vendor, device, UUID.
    {
        std::ifstream in(d / "vulkan-pipeline-cache.bin", std::ios::binary);
        std::uint8_t h[32] = {};
        if (in.read(reinterpret_cast<char*>(h), sizeof(h))) {
            std::uint32_t vendor, dev;
            std::memcpy(&vendor, h + 8, 4);
            std::memcpy(&dev, h + 12, 4);
            char b[32];
            std::snprintf(b, sizeof(b), "%04x:%04x ", vendor, dev);
            s.cache_device = b + hex(h + 16, 16);
        }
    }
    if (!s.cache_device.empty() && !devices.empty()) {
        s.device_current = 0;
        for (const std::string& dv : devices) {
            if (dv.compare(0, s.cache_device.size(), s.cache_device) == 0) s.device_current = 1;
        }
    }
    {
        std::ifstream in(d / "shader-compile.txt");
        for (std::string l; std::getline(in, l);) s.last.push_back(l);
    }
    return s;
}

ShaderCompileJob::~ShaderCompileJob() {
    cancel();
    if (thread_.joinable()) thread_.join();  // the child writes what it has done first
    if (proc_) SDL_DestroyProcess(static_cast<SDL_Process*>(proc_));
}

bool ShaderCompileJob::start(const std::string& data, std::string& error) {
    if (running()) return true;
    if (thread_.joinable()) thread_.join();
    if (proc_) {
        SDL_DestroyProcess(static_cast<SDL_Process*>(proc_));
        proc_ = nullptr;
    }
    // This run's id, which the child writes in its record.
    {
        std::random_device rd;
        const std::uint64_t clock = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        const std::uint64_t r = ((static_cast<std::uint64_t>(rd()) << 32) ^ rd()) ^ (clock * 0x9e3779b97f4a7c15ull);
        char id[17];
        std::snprintf(id, sizeof(id), "%016llx", static_cast<unsigned long long>(r));
        run_ = id;
    }
    record_ = data_dir(data) + "/shader-compile.txt";
    // This executable again, by its own path.
    std::string name = g_exe_name;
#if defined(_WIN32)
    if (name.empty()) name = "bbhost.exe";
    if (name.size() < 4 || name.compare(name.size() - 4, 4, ".exe") != 0) name += ".exe";
#else
    if (name.empty()) name = "bbhost";
#endif
    const std::string exe = config_exe_dir() + "/" + name;
    std::vector<std::string> args{exe, "--compile-shaders=" + run_};
    args.insert(args.end(), g_args.begin(), g_args.end());
    std::vector<const char*> argv;
    for (const std::string& a : args) argv.push_back(a.c_str());
    argv.push_back(nullptr);
    SDL_Process* p = SDL_CreateProcess(argv.data(), true);
    if (!p) {
        error = SDL_GetError();
        std::lock_guard<std::mutex> lk(mu_);
        v_ = View{};
        v_.failed_to_run = true;
        v_.error = "could not start " + exe + ": " + error;
        return false;
    }
    proc_ = p;
    {
        std::lock_guard<std::mutex> lk(mu_);
        v_ = View{};
        v_.running = true;
        v_.phase = "starting";
    }
    host_log("setup: compile-shaders started (%s, run %s)", exe.c_str(), run_.c_str());
    thread_ = std::thread([this] { reader(); });
    return true;
}

void ShaderCompileJob::cancel() {
    if (!proc_ || !running()) return;
    SDL_IOStream* in = SDL_GetProcessInput(static_cast<SDL_Process*>(proc_));
    if (in) {
        static const char kCancel[] = "cancel\n";
        SDL_WriteIO(in, kCancel, sizeof(kCancel) - 1);
        SDL_FlushIO(in);
    }
    std::lock_guard<std::mutex> lk(mu_);
    v_.phase = "cancelling (writing what is done)";
}

bool ShaderCompileJob::running() const {
    std::lock_guard<std::mutex> lk(mu_);
    return v_.running;
}

ShaderCompileJob::View ShaderCompileJob::view() const {
    std::lock_guard<std::mutex> lk(mu_);
    return v_;
}

namespace {

// The child's stdout and process, as shader_compile_read reads them.
struct SdlCompileStream final : ShaderCompileStream {
    SDL_Process* proc = nullptr;
    SDL_IOStream* out = nullptr;
    std::string read_error, wait_error;  // SDL's words for a failed read, a failed wait
    std::size_t read(char* p, std::size_t n, Status& st) override {
        if (!out) {
            st = kEnd;
            return 0;
        }
        const std::size_t got = SDL_ReadIO(out, p, n);
        if (got == 0) {
            const SDL_IOStatus s = SDL_GetIOStatus(out);
            st = s == SDL_IO_STATUS_NOT_READY ? kNotReady : s == SDL_IO_STATUS_EOF ? kEnd : kError;
            if (st == kError) read_error = SDL_GetError();
        }
        return got;
    }
    bool exited(int& code) override {
        if (SDL_WaitProcess(proc, false, &code)) return true;
        // Still running clears SDL's error; a wait that failed sets it, and
        // then nothing more can be known of the child.
        const char* e = SDL_GetError();
        if (!e || !*e) return false;
        wait_error = e;
        code = -1;
        return true;
    }
    void pause() override { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
};

}  // namespace

void ShaderCompileJob::reader() {
    SdlCompileStream s;
    s.proc = static_cast<SDL_Process*>(proc_);
    s.out = SDL_GetProcessOutput(s.proc);
    const int code = shader_compile_read(
        s,
        [this](const std::string& line) {
            std::lock_guard<std::mutex> lk(mu_);
            shader_compile_take_line(v_, line);
        },
        [&s](ShaderCompileStream::Status st) {
            host_log("setup: compile-shaders: the child's output %s while it runs%s%s; read on until it exits",
                     st == ShaderCompileStream::kEnd ? "ended" : "could not be read", s.read_error.empty() ? "" : ": ",
                     s.read_error.c_str());
        });
    host_log("setup: compile-shaders ended (exit %d)%s%s", code, s.wait_error.empty() ? "" : "; its wait failed: ",
             s.wait_error.c_str());
    // Without its last line, its record: written before it, read only then.
    bool need_record = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        need_record = !v_.finished && !v_.cancelled && v_.error.empty();
    }
    std::vector<std::string> record;
    if (need_record) {
        std::ifstream in(record_);
        for (std::string l; std::getline(in, l);) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            record.push_back(l);
        }
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (shader_compile_finish(v_, code, record, run_))
        host_log("setup: compile-shaders: its last line did not reach this window; the result (%s) is its record's: %s, run %s",
                 v_.finished ? "done" : "cancelled", record_.c_str(), run_.c_str());
    else if (need_record && (code == 0 || code == 1))
        host_log("setup: compile-shaders: no last line, and %s holds no record of run %s", record_.c_str(), run_.c_str());
}
