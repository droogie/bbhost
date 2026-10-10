#include "engine/frame_pool.h"
#include "engine/key_prompts.h"
#include "decomp/decomp.h"
#include "engine/mouse_camera.h"
#include "core/config.h"
#include "core/host_clock.h"
#include "core/portable.h"
#include "core/thunk.h"
#include "core/elf.h"
#include "core/imports.h"
#include "core/win_watch.h"
#include "core/write_watch.h"
#include "engine/addr.h"
#include "engine/guest.h"
#include "engine/np_test.h"
#include "engine/gx_resources.h"
#include "engine/params.h"
#include "engine/sf_heap_probe.h"
#include "guest_abi.h"
#include "hle/fs.h"
#include "hle/hle.h"
#include "hle/modules.h"
#include "host/gpu.h"
#include "host/options.h"
#include "host/updater.h"
#if defined(BBHOST_HAVE_SDL3)
#include "host/launcher.h"
#include "host/plugin_ui.h"
#endif
#include "host/settings.h"
#include "engine/debug_menu.h"
#include "host/plugins.h"
#include "host/sampler.h"
#include "host/audio.h"
#include "host/foreign_hooks.h"
#include "host/frame_stats.h"
#include "host/win_crash.h"
#include "host/window.h"
#include "log.h"

#include "bbhost_version.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <string>
#include <vector>

#include <cpuid.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <csignal>
#include <signal.h>
#include <ucontext.h>
#include <pthread.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <link.h>
#endif

void exit_reports();  // the reports a run prints when ended from outside (below)
[[noreturn]] void device_lost_exit(bool windowed);  // the GPU is gone: tell the player, end (below)

namespace {

struct GuestLaunch {
    GuestStart fn = nullptr;
    void* argv = nullptr;
};

#if defined(_WIN32)
DWORD WINAPI guest_main_thread(void* p) {
    auto* g = static_cast<GuestLaunch*>(p);
    sampler_note_main_thread();
    frame_stats_mark_main_thread();
    host_thread_set_name("main");  // Windows threads inherit no name, unlike Linux comm
    void* tcb = hle_thread_enter_guest();  // the TCB in this thread's TLS slot, which the rewritten fs:[0] reads
    g->fn(g->argv, nullptr);
    hle_thread_leave_guest(tcb);
    return 0;
}
#else
void* guest_main_thread(void* p) {
    auto* g = static_cast<GuestLaunch*>(p);
    sampler_note_main_thread();  // not renamed: the guest threads it creates would inherit the name
    frame_stats_mark_main_thread();
    void* tcb = hle_thread_enter_guest();
    g->fn(g->argv, nullptr);
    hle_thread_leave_guest(tcb);
    return nullptr;
}
#endif

// Boot thread: GX walk leaks ~8 bytes/node and visits millions of nodes.
constexpr std::size_t kGuestMainStack = 128ull * 1024 * 1024;

// Runs the guest on its own thread; the caller pumps the window until the
// guest returns or the user closes the window.
void run_guest_start(GuestStart fn, void* argv, bool windowed) {
    static GuestLaunch gl;
    gl = GuestLaunch{fn, argv};
    static const std::uint64_t exit_flip = [] {
        const char* e = std::getenv("BBHOST_EXIT_FLIP");
        return e ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    // BBHOST_EXIT_SECONDS=N: as BBHOST_EXIT_FLIP, after N seconds of the run
    // (an uncapped benchmark's flips come at any rate).
    static const double exit_seconds = [] {
        const char* e = std::getenv("BBHOST_EXIT_SECONDS");
        return e ? std::strtod(e, nullptr) : 0.0;
    }();
    static const auto run_start = std::chrono::steady_clock::now();
    static const std::uint64_t fault_flip = [] {
        const char* e = std::getenv("BBHOST_FAULT_AT_FLIP");
        return e ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    auto pump_until_done = [&](auto is_done) {
        while (!is_done()) {
            if (host_gpu_device_lost()) device_lost_exit(windowed);
            if (windowed && !host_window_pump()) {
                host_log("window closed");
                host_gpu_save_pipeline_cache_at_exit();
                hle_kernel_memory_report();
                host_window_stop();
                _exit(0);
            }
            if (fault_flip && hle_video_flip_count() >= fault_flip) {
                // BBHOST_FAULT_AT_FLIP: a deliberate null write on this thread,
                // to see the crash report and the exit code (Windows: the
                // vectored handler, the filter, exit 139). BBHOST_FAULT_KIND=
                // abort, terminate (an exception nothing catches) or invalid (a
                // C runtime call given a null stream) instead: the ends that
                // bypass those handlers (Windows: win_crash's, exit 134).
                static const char* const kind = std::getenv("BBHOST_FAULT_KIND");
                host_log("fault: %s at flip %llu (BBHOST_FAULT_AT_FLIP)", kind ? kind : "writing to address 0",
                         static_cast<unsigned long long>(fault_flip));
                if (kind && !std::strcmp(kind, "abort")) std::abort();
                if (kind && !std::strcmp(kind, "terminate")) std::thread([] { throw std::runtime_error("BBHOST_FAULT_KIND=terminate"); }).join();
                if (kind && !std::strcmp(kind, "invalid")) std::fclose(static_cast<FILE*>(nullptr));
                *static_cast<volatile int*>(nullptr) = 1;
            }
            if ((exit_flip && hle_video_flip_count() >= exit_flip) ||
                (exit_seconds > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count() >= exit_seconds)) {
                // The guest keeps running while this prints; the window is
                // not stopped (that waits for the GPU lock the command
                // processor may hold), the process just ends.
                host_log("exit: flip %llu reached (BBHOST_EXIT_FLIP)", static_cast<unsigned long long>(exit_flip));
                exit_reports();
                // The cache save takes the GPU lock, which the command
                // processor (still running) may hold for good: a helper
                // thread saves and the exit waits for it a bounded time.
                static std::atomic<bool> saved{false};
                std::thread([] {
                    host_gpu_save_pipeline_cache_at_exit();
                    saved.store(true);
                }).detach();
                for (int i = 0; i < 100 && !saved.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (!saved.load()) host_log("exit: the pipeline cache save did not finish in 5 s; leaving without it");
                std::fflush(nullptr);
#if defined(_WIN32)
                // Not ExitProcess: it detaches every DLL under the loader
                // lock, and a guest or driver thread inside one hangs it.
                TerminateProcess(GetCurrentProcess(), 0);
#endif
                _exit(0);
            }
            // The window's input and events every 4 ms, on the high-resolution
            // timer: sleep_for is winpthreads' Sleep on Windows, a 15.6 ms
            // step whenever the timer resolution is not raised (before the
            // first host_sleep_us used to raise it, or a hidden window on
            // Windows 11 without the power-throttling opt-out).
            host_sleep_us(windowed ? 4000 : 50000);
        }
    };
#if defined(_WIN32)
    // The whole stack committed, not only reserved. Windows commits a thread's
    // stack a page at a time, through a guard page that code must touch in
    // order (compilers for Windows probe large frames); the eboot was built
    // for a system that maps stacks whole and does not probe. Its GX setup
    // (sub_25676c0) opens a 46 KB frame and writes the bottom of it first:
    // past the guard page, into reserved memory, a fault the system cannot
    // even deliver - the stack it would write the exception to is that same
    // memory - so the process ended with 0xc0000005 and no handler ran. Every
    // run on a Windows 11 laptop did, at the same point; wine maps its
    // stacks whole. (The guest's other threads are winpthreads', which
    // commit what they are given.)
    HANDLE th = CreateThread(nullptr, kGuestMainStack, guest_main_thread, &gl, 0, nullptr);
    if (!th) {
        host_log("guest main thread: CreateThread with a %zu MiB stack failed (error %lu); running on this thread",
                 kGuestMainStack >> 20, static_cast<unsigned long>(GetLastError()));
        fn(argv, nullptr);
        return;
    }
    pump_until_done([&] { return WaitForSingleObject(th, 0) == WAIT_OBJECT_0; });
    CloseHandle(th);
#else
    const std::size_t guard = 0x1000;
    const std::size_t total = kGuestMainStack + guard;
    void* map = mmap(nullptr, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED) {
        host_log("guest stack mmap 0x%zx failed", total);
        fn(argv, nullptr);
        return;
    }
    void* stack = static_cast<std::uint8_t*>(map) + guard;
    if (mprotect(stack, kGuestMainStack, PROT_READ | PROT_WRITE) != 0) {
        host_log("guest stack mprotect failed");
        munmap(map, total);
        fn(argv, nullptr);
        return;
    }
    hle_set_guest_stack(stack, kGuestMainStack);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstack(&attr, stack, kGuestMainStack);
    pthread_t th;
    if (pthread_create(&th, &attr, guest_main_thread, &gl) != 0) {
        pthread_attr_destroy(&attr);
        munmap(map, total);
        fn(argv, nullptr);
        return;
    }
    pthread_attr_destroy(&attr);
    std::atomic<bool> done{false};
    std::thread joiner([&] {
        pthread_join(th, nullptr);
        done = true;
    });
    pump_until_done([&] { return done.load(); });
    joiner.join();
    munmap(map, total);
#endif
}

// The processor, for reports from other machines: its name, and its family,
// model and stepping as CPUID gives them, and whether it mixes core types.
void log_cpu() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    char brand[49] = {};
    if (__get_cpuid(0x80000000u, &a, &b, &c, &d) && a >= 0x80000004u) {
        for (unsigned i = 0; i < 3; ++i) {
            __get_cpuid(0x80000002u + i, &a, &b, &c, &d);
            const unsigned regs[4] = {a, b, c, d};
            std::memcpy(brand + 16 * i, regs, sizeof(regs));
        }
    }
    const char* name = brand;
    while (*name == ' ') ++name;
    unsigned family = 0, model = 0, stepping = 0;
    if (__get_cpuid(1, &a, &b, &c, &d)) {
        stepping = a & 0xf;
        family = (a >> 8) & 0xf;
        model = (a >> 4) & 0xf;
        if (family == 0xf) family += (a >> 20) & 0xff;
        if (family == 0x6 || family >= 0xf) model |= ((a >> 16) & 0xf) << 4;
    }
    const bool hybrid = __get_cpuid_count(7, 0, &a, &b, &c, &d) && ((d >> 15) & 1);
    host_log("cpu: %s (family 0x%x, model 0x%x, stepping %u%s), %u threads", *name ? name : "unknown", family, model, stepping,
             hybrid ? ", hybrid" : "", std::thread::hardware_concurrency());
}

}  // namespace

// The reports a run prints when it is ended from outside: SIGTERM from
// `timeout` on Linux, the console's close/Ctrl-C on Windows, and the flip
// BBHOST_EXIT_FLIP names on both (the run ends itself, with the reports, on
// a platform where the outside signal cannot reach it: `timeout` under wine).
// The GPU device was lost (gpu.cpp device_lost_locked): the game would go on
// - its logic and its sound need no GPU - behind a window nothing draws to
// again, which Windows paints white. The sound stops, the player is told why
// (and about the hooks that commonly cause it), and bbhost ends with exit
// code 3 once no save is being written.
void device_lost_exit(bool windowed) {
    host_log("exit: the GPU device is lost; bbhost ends (exit code 3)");
    host_audio_pause_all();
    // A shorter wait for the box under BBHOST_TEST_DEVICE_LOST, which harnesses
    // set with nobody there to close it.
    const bool test = std::getenv("BBHOST_TEST_DEVICE_LOST") != nullptr;
    if (windowed) {
        std::string text =
            "The graphics card stopped responding and the system reset it, so the game cannot draw anything more and "
            "bbhost has to close.\n\n";
        if (host_foreign_hooks_obs()) {
            text += "OBS's game capture is hooked into bbhost (graphics-hook64.dll), a common cause of this. Try again with "
                    "obs_capture = \"off\" under [video] in bbhost.toml (OBS's Window Capture still records bbhost), or set "
                    "DISABLE_VULKAN_OBS_CAPTURE=1 before starting it.\n\n";
        } else if (const std::string hooks = host_foreign_hooks(); !hooks.empty()) {
            text += "Hooked into bbhost: " + hooks + ". Overlays like these can cause this; try again without them.\n\n";
        }
        text += "The next starts record where the GPU was, so if it happens again the log says more. Please send the log "
                "(run-bbhost.bat writes it to the logs folder).";
        // The box on a thread of its own: the exit goes on after a while even
        // if nobody closes it.
        static std::atomic<bool> closed{false};
        static std::string box;
        box = text;
        std::thread([] {
            host_window_error_box("bbhost: the graphics card stopped responding", box.c_str());
            closed.store(true);
        }).detach();
        const int ticks = test ? 30 : 1200;  // 3 s / 2 min
        for (int i = 0; i < ticks && !closed.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // A save being written finishes first (up to 10 s).
    for (int i = 0; i < 100 && hle_save_writable_mounts() > 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    exit_reports();
    // The pipelines built so far, for a quicker next start: the save takes the
    // GPU lock a stuck thread may hold, so a helper does it, waited for 5 s.
    static std::atomic<bool> saved{false};
    std::thread([] {
        host_gpu_save_pipeline_cache_at_exit();
        saved.store(true);
    }).detach();
    for (int i = 0; i < 100 && !saved.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::fflush(nullptr);
#if defined(_WIN32)
    TerminateProcess(GetCurrentProcess(), 3);
#endif
    _exit(3);
}

void exit_reports() {
    hle_gx_scaleform_report();
    hle_gnm_cp_report();
    gx_resources_report();
    params_report();
    hle_bp_report();
    hle_watch_report();
    hle_kernel_memory_report();
    frame_pool_report();
    key_prompts_report();
    decomp_report();
    mouse_camera_report();
}

#if !defined(_WIN32)
// SIGTERM lands on whichever thread the kernel picks, often a guest thread
// running guest code with the guest's FS - where libc's thread state (errno,
// the stack guard, its locks' owner) is not glibc's. The reports then crashed
// in snprintf or write, and the run dumped core on its way out (two in ~40
// soaks, both with a guest thread inside a memcpy). Back to the host's FS
// first, as the crash handler does; this frame reads no guard, since it
// starts under the guest's FS and never returns.
__attribute__((no_stack_protector)) void on_term(int) {
    hle_fs_host();
    exit_reports();
    _exit(0);
}
#else
BOOL WINAPI on_console_ctrl(DWORD) {
    exit_reports();
    _exit(0);
    return TRUE;
}
#endif

#if !defined(_WIN32)

// --- debug soft-watchpoint (BBHOST_WATCH_FILL) ---------------------------
namespace {
std::atomic<std::uint64_t> g_wp_page{0}, g_wp_lo{0}, g_wp_hi{0};
std::atomic<bool> g_wp_armed{false};
std::atomic<int> g_wp_prot{PROT_READ};  // PROT_NONE when reads are watched too
std::atomic<int> g_wp_faults{0}, g_wp_n{0};
struct WpHit { std::uint64_t rip, addr, r13, r12, src_val, rsi, rdi; std::uint32_t src_ctx[8]; };
WpHit g_wp_ring[64];
std::uint64_t g_wp_stack[256];
std::atomic<int> g_wp_stack_n{0};
std::atomic<unsigned> g_wp_mxcsr{0xffffffff}, g_wp_cwd{0};
constexpr int kWpCap = 64;
constexpr int kWpFaultLimit = 100000;
}

namespace {
std::atomic<std::uint64_t> g_bp_va{0}, g_bp_rdx{0};
std::atomic<unsigned> g_bp_orig{0};
std::atomic<bool> g_bp_armed{false}, g_bp_rearm{false};
std::atomic<int> g_bp_hits{0};
constexpr int kBpCap = 8;
float g_bp_mat[kBpCap][8];
float g_bp_mat2[kBpCap][8];
std::uint64_t g_bp_rdx_raw[kBpCap];
std::uint64_t g_bp_ptr[kBpCap];
std::uint64_t g_bp_ret[kBpCap], g_bp_rdi[kBpCap], g_bp_rsi[kBpCap], g_bp_r8[kBpCap], g_bp_r9[kBpCap];
}

void hle_bp_arm(std::uint64_t va) {
    const std::uint64_t page = va & ~std::uint64_t(0xfff);
    if (mprotect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(page)), 0x2000,
                 PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        host_log("bp: mprotect failed for 0x%llx", static_cast<unsigned long long>(va));
        return;
    }
    auto* p = reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(va));
    g_bp_orig.store(*p);
    g_bp_va.store(va);
    *p = 0xcc;
    g_bp_armed.store(true);
    host_log("bp: int3 armed at 0x%llx (orig byte 0x%02x)", static_cast<unsigned long long>(va),
             g_bp_orig.load());
}

void hle_bp_report() {
    const int n = g_bp_hits.load();
    if (!g_bp_va.load() && n == 0) return;
    host_log("bp: %d hit(s) at 0x%llx", n, static_cast<unsigned long long>(g_bp_va.load()));
    for (int i = 0; i < n && i < kBpCap; ++i) {
        const float* m = g_bp_mat[i];
        // Matrix2x4 rows: [a b .. tx] [c d .. ty]; scale is the 2x2 part.
        host_log("  call[%d] ret=0x%llx arg1=0x%llx arg2=0x%llx arg5=0x%llx arg6=0x%llx", i,
                 static_cast<unsigned long long>(g_bp_ret[i]), static_cast<unsigned long long>(g_bp_rdi[i]),
                 static_cast<unsigned long long>(g_bp_rsi[i]), static_cast<unsigned long long>(g_bp_r8[i]),
                 static_cast<unsigned long long>(g_bp_r9[i]));
        host_log("  obj[%d] [R8+0x60] = [%g %g %g %g] [%g %g %g %g]", i,
                 g_bp_mat2[i][0], g_bp_mat2[i][1], g_bp_mat2[i][2], g_bp_mat2[i][3],
                 g_bp_mat2[i][4], g_bp_mat2[i][5], g_bp_mat2[i][6], g_bp_mat2[i][7]);
        host_log("  arg3[%d] @0x%llx = [%g %g %g %g] [%g %g %g %g]  |col0|=%g |col1|=%g", i,
                 static_cast<unsigned long long>(g_bp_ptr[i]), m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                 std::sqrt(m[0] * m[0] + m[4] * m[4]), std::sqrt(m[1] * m[1] + m[5] * m[5]));
    }
}

void hle_watch_arm(std::uint64_t lo, std::uint64_t hi) {
    g_wp_lo.store(lo);
    g_wp_hi.store(hi);
    g_wp_n.store(0);
    g_wp_faults.store(0);
    const std::uint64_t page = lo & ~std::uint64_t(0xfff);
    g_wp_page.store(page);
    g_wp_armed.store(true);
    mprotect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(page)), 0x1000, g_wp_prot.load());
    host_log("watch: armed [0x%llx,0x%llx) page 0x%llx%s", static_cast<unsigned long long>(lo),
             static_cast<unsigned long long>(hi), static_cast<unsigned long long>(page),
             g_wp_prot.load() == PROT_NONE ? " (reads too)" : "");
}

// The same watch over reads as well: the page is made inaccessible, so every
// access in [lo, hi) is recorded (as "watch write:" in the report).
void hle_watch_arm_reads(std::uint64_t lo, std::uint64_t hi) {
    g_wp_prot.store(PROT_NONE);
    hle_watch_arm(lo, hi);
}

void hle_watch_report() {
    const int n = g_wp_n.load();
    host_log("watch: %d in-range writes recorded, %d total faults on page 0x%llx; guest MXCSR=0x%04x cwd=0x%04x",
             n, g_wp_faults.load(), static_cast<unsigned long long>(g_wp_page.load()),
             g_wp_mxcsr.load(), g_wp_cwd.load());
    const int sn = g_wp_stack_n.load();
    if (sn) {
        host_log("watch: call-stack candidate return addresses (guest code 0x400000..0x5ad30f4):");
        for (int k = 0; k < sn; ++k) {
            const std::uint64_t v = g_wp_stack[k];
            if (v >= 0x400000 && v <= 0x5ad30f4)
                host_log("  stack[+0x%x] = 0x%llx", k * 8, static_cast<unsigned long long>(v));
        }
    }
    for (int i = 0; i < n && i < kWpCap; ++i) {
        host_log("  watch write: rip=0x%llx dst=0x%llx rsi=0x%llx rdi=0x%llx src(r13)=0x%llx srcbase=0x%llx ctx: %08x %08x %08x %08x %08x %08x %08x %08x",
                 static_cast<unsigned long long>(g_wp_ring[i].rip), static_cast<unsigned long long>(g_wp_ring[i].addr),
                 static_cast<unsigned long long>(g_wp_ring[i].rsi), static_cast<unsigned long long>(g_wp_ring[i].rdi),
                 static_cast<unsigned long long>(g_wp_ring[i].r13), static_cast<unsigned long long>(g_wp_ring[i].r13 & ~std::uint64_t(0x1f)),
                 g_wp_ring[i].src_ctx[0], g_wp_ring[i].src_ctx[1], g_wp_ring[i].src_ctx[2], g_wp_ring[i].src_ctx[3],
                 g_wp_ring[i].src_ctx[4], g_wp_ring[i].src_ctx[5], g_wp_ring[i].src_ctx[6], g_wp_ring[i].src_ctx[7]);
    }
}

// SIGTRAP: single-step landing after we allowed an out-of-range write; re-arm.
void on_trap(int sig, siginfo_t*, void* ctx) {
    auto* uc = static_cast<ucontext_t*>(ctx);
    const std::uint64_t bp = g_bp_va.load();
    if (bp) {
        const std::uint64_t rip = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RIP]);
        auto* bpp = reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(bp));
        if (g_bp_armed.load() && rip == bp + 1) {
            // int3 hit: capture arg3 (RDX) and the Matrix2x4 it points at.
            const std::uint64_t rdx = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RDX]);
            const int i = g_bp_hits.fetch_add(1);
            if (i < kBpCap) {
                g_bp_ptr[i] = rdx;
                const std::uint64_t rsp = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSP]);
                g_bp_ret[i] = *reinterpret_cast<const volatile std::uint64_t*>(static_cast<std::uintptr_t>(rsp));
                g_bp_rdi[i] = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RDI]);
                g_bp_rsi[i] = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSI]);
                g_bp_r8[i]  = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_R8]);
                g_bp_r9[i]  = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_R9]);
                g_bp_rdx_raw[i] = rdx;
                // Also snapshot [R8 + 0x60] as a Matrix2x4: for sub_4b7ab0 that is
                // the object field the packing matrix is copied from.
                const std::uint64_t r8 = g_bp_r8[i];
                for (int k = 0; k < 8; ++k) g_bp_mat2[i][k] = 0.0f;
                if (r8 > 0x10000) {
                    const auto* m2 = reinterpret_cast<const volatile float*>(static_cast<std::uintptr_t>(r8 + 0x60));
                    for (int k = 0; k < 8; ++k) g_bp_mat2[i][k] = m2[k];
                }
                if (rdx) {
                    const auto* m = reinterpret_cast<const volatile float*>(static_cast<std::uintptr_t>(rdx));
                    for (int k = 0; k < 8; ++k) g_bp_mat[i][k] = m[k];
                }
            }
            g_bp_rdx.store(rdx);
            *bpp = static_cast<std::uint8_t>(g_bp_orig.load());   // restore
            uc->uc_mcontext.gregs[REG_RIP] = static_cast<greg_t>(bp);  // re-run the real insn
            g_bp_armed.store(false);
            if (i + 1 < kBpCap) {
                g_bp_rearm.store(true);
                uc->uc_mcontext.gregs[REG_EFL] |= std::uint64_t(0x100);  // step, then re-arm
            }
            return;
        }
        if (g_bp_rearm.load()) {
            *bpp = 0xcc;
            g_bp_armed.store(true);
            g_bp_rearm.store(false);
            uc->uc_mcontext.gregs[REG_EFL] &= ~std::uint64_t(0x100);
            return;
        }
    }
    const std::uint64_t page = g_wp_page.load();
    if (g_wp_armed.load() && page) {
        mprotect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(page)), 0x1000, g_wp_prot.load());
        uc->uc_mcontext.gregs[REG_EFL] &= ~std::uint64_t(0x100);
        return;
    }
    std::signal(sig, SIG_DFL);
    raise(sig);
}

// On a fault, print where and what the command processor wrote recently,
// then re-raise so the core dump still happens.
void on_segv(int sig, siginfo_t* info, void* ctx) {
    auto* uc = static_cast<ucontext_t*>(ctx);
    if (g_wp_armed.load()) {
        const std::uint64_t fa = reinterpret_cast<std::uint64_t>(info->si_addr);
        const std::uint64_t page = g_wp_page.load();
        if (page && (fa & ~std::uint64_t(0xfff)) == page) {
            // Let the write through, then decide whether to keep watching.
            mprotect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(page)), 0x1000, PROT_READ | PROT_WRITE);
            g_wp_faults.fetch_add(1);
            if (fa >= g_wp_lo.load() && fa < g_wp_hi.load()) {
                const int i = g_wp_n.fetch_add(1);
                if (i < kWpCap) {
                    g_wp_ring[i].rip = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RIP]);
                    g_wp_ring[i].addr = fa;
                    // sub_4e5c00 keeps the source pointer in r13 and dest in r12.
                    const std::uint64_t r13 = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_R13]);
                    const std::uint64_t r12 = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_R12]);
                    g_wp_ring[i].r13 = r13;
                    g_wp_ring[i].r12 = r12;
                    // r13 was a source pointer only for the write this watch was
                    // built for (BBHOST_WATCH_FILL); nothing is dereferenced here -
                    // this runs with the guest's FS, so no libc either - and a
                    // memcpy's source and destination are kept as rsi and rdi.
                    g_wp_ring[i].src_val = 0;
                    g_wp_ring[i].rsi = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSI]);
                    g_wp_ring[i].rdi = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RDI]);
                    if (g_wp_stack_n.load() == 0) {
                        // Step 6: capture the guest thread's SSE control word
                        // (MXCSR) - logged later in hle_watch_report (host_log is
                        // not async-signal-safe). Default 0x1f80; FTZ/DAZ or a
                        // non-nearest rounding mode would corrupt tessellation.
                        if (uc->uc_mcontext.fpregs) {
                            g_wp_mxcsr.store(uc->uc_mcontext.fpregs->mxcsr);
                            g_wp_cwd.store(uc->uc_mcontext.fpregs->cwd);
                        }
                        // Only what lies on rsp's own page: a shallow stack
                        // ends before 256 words, and this runs on it.
                        const std::uint64_t rsp = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSP]);
                        const auto* sp = reinterpret_cast<const volatile std::uint64_t*>(static_cast<std::uintptr_t>(rsp));
                        const int words = static_cast<int>(std::min<std::uint64_t>(256, (((rsp + 0x1000) & ~0xfffull) - rsp) / 8));
                        for (int k = 0; k < words; ++k) g_wp_stack[k] = sp[k];
                        g_wp_stack_n.store(words);
                    }
                }
            }
            if (g_wp_n.load() >= kWpCap || g_wp_faults.load() >= kWpFaultLimit) {
                // Enough samples (or too many faults): stop, leave the page writable.
                g_wp_armed.store(false);
                g_wp_page.store(0);
            } else {
                // Keep watching: single-step past this write, re-arm on SIGTRAP.
                uc->uc_mcontext.gregs[REG_EFL] |= std::uint64_t(0x100);
            }
            return;
        }
    }
    hle_fs_host();  // the faulting thread may be running guest code with guest FS
    sampler_recover();  // a fault in the profiler's unwind drops its sample
    host_read_safe_recover();  // a fault in host_read_safe's copy fails the read
    const std::uint64_t pc = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RIP]);
    const std::uint64_t addr = reinterpret_cast<std::uint64_t>(info->si_addr);
    host_log("SIGSEGV pc=0x%llx addr=0x%llx (guest pc 0x%llx)", static_cast<unsigned long long>(pc),
             static_cast<unsigned long long>(addr), static_cast<unsigned long long>(pc));
    host_log("  rax=%llx rdi=%llx rsi=%llx rdx=%llx rcx=%llx r8=%llx r13=%llx r15=%llx rsp=%llx rbp=%llx",
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RAX]),
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RDI]),
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RSI]),
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RDX]),
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RCX]),
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_R8]),
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_R13]),
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_R15]),
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RSP]),
             static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RBP]));
    {
        // The GX heap's release pass (0x2aaa860) walks its nodes in r14. When the
        // node is guest memory, show it and every mapping of its physical page:
        // a node whose first word holds another address's value shares memory
        // with that address.
        const std::uint64_t r14 = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_R14]);
        host_log("  r12=%llx r14=%llx rbx=%llx", static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_R12]),
                 static_cast<unsigned long long>(r14), static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RBX]));
        if (r14 && hle_kernel_va_mapped(r14, 48)) {
            const auto* p = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(r14));
            host_log("  r14 %llx: %llx %llx %llx %llx %llx %llx", static_cast<unsigned long long>(r14),
                     static_cast<unsigned long long>(p[0]), static_cast<unsigned long long>(p[1]), static_cast<unsigned long long>(p[2]),
                     static_cast<unsigned long long>(p[3]), static_cast<unsigned long long>(p[4]), static_cast<unsigned long long>(p[5]));
            hle_kernel_mappings_of(r14);
            // Did the command processor write that value, or over this node?
            hle_gnm_find_writes(r14, 48, p[0]);
        }
    }
    std::uint64_t chain_returns[8] = {};  // the frame-pointer chain's first return addresses
    {
        const std::uint64_t rsp = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSP]);
        if (hle_kernel_va_mapped(rsp, 64) || rsp > 0x700000000000ull) {
            const auto* sp = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(rsp));
            host_log("  stack %llx: %llx %llx %llx %llx  %llx %llx %llx %llx",
                     static_cast<unsigned long long>(rsp),
                     static_cast<unsigned long long>(sp[0]), static_cast<unsigned long long>(sp[1]),
                     static_cast<unsigned long long>(sp[2]), static_cast<unsigned long long>(sp[3]),
                     static_cast<unsigned long long>(sp[4]), static_cast<unsigned long long>(sp[5]),
                     static_cast<unsigned long long>(sp[6]), static_cast<unsigned long long>(sp[7]));
        }
        // Guest code keeps frame pointers: the return addresses up the rbp chain,
        // for as long as each frame sits above the last within this stack.
        std::uint64_t rbp = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RBP]);
        char chain[512];
        int len = 0;
        int depth = 0;

        for (; rbp > rsp && rbp - rsp < (256u << 20) && (rbp & 7) == 0 &&
               (hle_kernel_va_mapped(rbp, 16) || rbp > 0x700000000000ull);
             ++depth) {
            const auto* f = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(rbp));
            if (depth < 16 && len < static_cast<int>(sizeof(chain)) - 20)
                len += std::snprintf(chain + len, sizeof(chain) - static_cast<std::size_t>(len), " %llx",
                                     static_cast<unsigned long long>(f[1]));
            if (depth < 8) chain_returns[depth] = f[1];
            if (f[0] <= rbp) break;
            rbp = f[0];
        }
        if (len) host_log("  frames (%d, over %llu KiB of stack):%s", depth + 1,
                          static_cast<unsigned long long>((rbp - rsp) >> 10), chain);
    }
    {
        // Host code: the module and offset (for addr2line) of the pc, and of the
        // stack words that point into a loaded module (return addresses among them).
        struct Module {
            std::uint64_t a;
            const char* name;
            std::uint64_t base;
            bool found;
        };
        const auto where = [](std::uint64_t a, const char* what) {
            if (a < 0x100000000000ull) return false;
            Module m{a, nullptr, 0, false};
            dl_iterate_phdr(
                [](dl_phdr_info* info, std::size_t, void* data) {
                    auto* m = static_cast<Module*>(data);
                    for (int k = 0; k < info->dlpi_phnum; ++k) {
                        const ElfW(Phdr)& ph = info->dlpi_phdr[k];
                        const std::uint64_t lo = info->dlpi_addr + ph.p_vaddr;
                        if (ph.p_type == PT_LOAD && (ph.p_flags & PF_X) && m->a >= lo && m->a < lo + ph.p_memsz) {
                            *m = {m->a, info->dlpi_name, info->dlpi_addr, true};
                            return 1;
                        }
                    }
                    return 0;
                },
                &m);
            if (!m.found) return false;
            Dl_info info{};
            const bool named = dladdr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(a)), &info) && info.dli_sname;
            host_log("  %s %llx: %s+0x%llx%s%s", what, static_cast<unsigned long long>(a), m.name && m.name[0] ? m.name : "bbhost",
                     static_cast<unsigned long long>(a - m.base), named ? " " : "", named ? info.dli_sname : "");
            return true;
        };
        where(pc, "pc");
        for (const std::uint64_t r : chain_returns) {
            if (r) where(r, "frame");
        }
        const std::uint64_t rsp = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSP]);
        if (rsp > 0x700000000000ull) {
            const auto* sp = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(rsp));
            for (int k = 0, shown = 0; k < 128 && shown < 16; ++k) shown += where(sp[k], "stack word") ? 1 : 0;
        }
    }
    {
        const std::uint64_t block = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSI]);
        if (block >= 16 && hle_kernel_va_mapped(block - 16, 64)) {
            const auto* p = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(block - 16));
            host_log("  rsi-16 %llx: %llx %llx %llx %llx  %llx %llx %llx %llx",
                     static_cast<unsigned long long>(block - 16),
                     static_cast<unsigned long long>(p[0]), static_cast<unsigned long long>(p[1]),
                     static_cast<unsigned long long>(p[2]), static_cast<unsigned long long>(p[3]),
                     static_cast<unsigned long long>(p[4]), static_cast<unsigned long long>(p[5]),
                     static_cast<unsigned long long>(p[6]), static_cast<unsigned long long>(p[7]));
        }
    }
    sf_heap_probe_crash_report();
    hle_fs_log_recent_opens();
    hle_gnm_dump_recent_writes(0, 48);
    signal(sig, SIG_DFL);
    raise(sig);
}

// The handler installed for SIGSEGV. A write to a page the texture cache
// write-protected (core/write_watch.h) is the common case and not an error:
// the page is made writable again and the write retried, without a lock, a
// log line or a thread-local - on a guest thread FS is the guest's, so this
// frame has no stack protector either.
__attribute__((no_stack_protector)) void on_segv_entry(int sig, siginfo_t* info, void* ctx) {
    auto* uc = static_cast<ucontext_t*>(ctx);
    const bool is_write = (uc->uc_mcontext.gregs[REG_ERR] & 2) != 0;  // the page fault's W bit
    // The writer, for the texture census: the pc, the top of the stack
    // (return addresses when the write is in a leaf like memcpy) and the
    // frame's return address. Plain reads of the guest's own stack, no libc.
    std::uint64_t frame[kWriteWatchFrame] = {};
    if (sig == SIGSEGV && is_write) {
        frame[0] = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RIP]);
        const std::uint64_t rsp = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSP]);
        const std::uint64_t rbp = static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RBP]);
        const auto page_ok = [](std::uint64_t a) { return a > 0x10000 && (a & 0xfff) <= 0xff8; };
        for (int k = 0; k < 6; ++k) {
            const std::uint64_t at = rsp + 8ull * static_cast<std::uint64_t>(k);
            if (!page_ok(at) || (at >> 12) != (rsp >> 12)) break;  // the stack's own page only
            frame[1 + k] = *reinterpret_cast<const std::uint64_t*>(at);
        }
        if (page_ok(rbp + 8) && ((rbp + 8) >> 12) == (rsp >> 12)) frame[7] = *reinterpret_cast<const std::uint64_t*>(rbp + 8);
    }
    if (sig == SIGSEGV && write_watch_on_fault(reinterpret_cast<std::uint64_t>(info->si_addr), is_write, frame)) {
        return;
    }
    on_segv(sig, info, ctx);
}
#endif

#if defined(_WIN32)
// The int3 breakpoint and the page watch on Windows, in their generic form
// (the Linux ones above grew captures for one investigation each): the
// breakpoint logs the guest pc and the argument registers on every hit, up
// to eight, and steps over; the watch is core/win_watch.cpp, which logs
// each write to the range with the writing instruction's address.
namespace {
std::atomic<std::uint64_t> g_bp_va{0};
std::atomic<unsigned> g_bp_orig{0};
std::atomic<int> g_bp_hits{0};
thread_local bool t_bp_stepping = false;

LONG CALLBACK bp_handler(EXCEPTION_POINTERS* ep) {
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    CONTEXT* c = ep->ContextRecord;
    const std::uint64_t bp = g_bp_va.load();
    if (code == EXCEPTION_BREAKPOINT && bp && c->Rip == bp) {
        const int n = g_bp_hits.fetch_add(1);
        if (n < 8) {
            const std::uint64_t slide = hle_image_slide();
            host_log("bp: hit %d at 0x%llx (guest 0x%llx): rdi=%llx rsi=%llx rdx=%llx rcx=%llx r8=%llx r9=%llx rsp=%llx", n,
                     static_cast<unsigned long long>(bp), static_cast<unsigned long long>(bp - slide + 0x400000),
                     static_cast<unsigned long long>(c->Rdi), static_cast<unsigned long long>(c->Rsi),
                     static_cast<unsigned long long>(c->Rdx), static_cast<unsigned long long>(c->Rcx),
                     static_cast<unsigned long long>(c->R8), static_cast<unsigned long long>(c->R9),
                     static_cast<unsigned long long>(c->Rsp));
        }
        // Run the original instruction: restore the byte, single-step, re-arm.
        *reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(bp)) = static_cast<std::uint8_t>(g_bp_orig.load());
        t_bp_stepping = true;
        c->EFlags |= 0x100;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (code == EXCEPTION_SINGLE_STEP && t_bp_stepping) {
        t_bp_stepping = false;
        c->EFlags &= ~static_cast<DWORD>(0x100);
        if (bp && g_bp_hits.load() < 8) *reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(bp)) = 0xcc;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
}  // namespace

void hle_bp_arm(std::uint64_t va) {
    DWORD old = 0;
    if (!VirtualProtect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(va & ~0xfffull)), 0x2000, PAGE_EXECUTE_READWRITE, &old)) {
        host_log("bp: cannot unprotect 0x%llx", static_cast<unsigned long long>(va));
        return;
    }
    static PVOID h = AddVectoredExceptionHandler(1, bp_handler);
    (void)h;
    auto* p = reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(va));
    g_bp_orig.store(*p);
    g_bp_va.store(va);
    *p = 0xcc;
    host_log("bp: int3 armed at 0x%llx (orig byte 0x%02x); the first 8 hits are logged", static_cast<unsigned long long>(va),
             g_bp_orig.load());
}
void hle_bp_report() {
    if (g_bp_va.load()) host_log("bp: %d hit(s) at 0x%llx", g_bp_hits.load(), static_cast<unsigned long long>(g_bp_va.load()));
}
void hle_watch_arm(std::uint64_t lo, std::uint64_t hi) { win_watch_arm(lo, hi, 64); }
void hle_watch_arm_reads(std::uint64_t lo, std::uint64_t hi) {
    host_log("watch: reads are not watched on Windows; writes to the range are");
    win_watch_arm(lo, hi, 64);
}
void hle_watch_report() {}
#endif

int main(int argc, char** argv) {
#if defined(_WIN32)
    hle_kernel_reserve_guest_windows();  // before anything else can take those addresses
    write_watch_install();               // the vectored handler for write-protected texture memory
    win_crash_install();                 // the crash report, behind it
    SetConsoleCtrlHandler(on_console_ctrl, TRUE);
    host_log("bbhost %s (%s, Windows x86-64)", BBHOST_VERSION, BBHOST_GIT_REV);
    log_cpu();
    {
        // The machine's memory, for reports from other machines: the GPU
        // locks the guest's direct memory (6 GiB) in RAM when it imports it.
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
            host_log("memory: %.1f GiB physical, %.1f GiB free; commit limit %.1f GiB, %.1f GiB of it free",
                     static_cast<double>(ms.ullTotalPhys) / kGiB, static_cast<double>(ms.ullAvailPhys) / kGiB,
                     static_cast<double>(ms.ullTotalPageFile) / kGiB, static_cast<double>(ms.ullAvailPageFile) / kGiB);
        }
    }
    host_timing_init();  // 1 ms timer resolution and no power throttling from here on (core/host_clock.h)
    {
        // The process's standing with Windows' schedulers, left as Windows
        // makes it unless asked.
        // BBHOST_PRIORITY_CLASS=above|high|normal: the priority class - shadPS4
        // sets above normal for itself at start, bbhost leaves normal.
        // BBHOST_POWER_THROTTLING=0: out of Windows' execution-speed throttling
        // (EcoQoS), which its power policy may put on a process it takes for
        // background work - a headless run has no window; =1 into it.
        const char* pc = std::getenv("BBHOST_PRIORITY_CLASS");
        if (pc && *pc) {
            const std::string v = pc;
            const DWORD cls = v == "above" ? ABOVE_NORMAL_PRIORITY_CLASS : v == "high" ? HIGH_PRIORITY_CLASS : v == "normal" ? NORMAL_PRIORITY_CLASS : 0;
            if (!cls) {
                host_log("process: BBHOST_PRIORITY_CLASS=%s is not above, high or normal; ignored", pc);
            } else if (SetPriorityClass(GetCurrentProcess(), cls)) {
                host_log("process: priority class %s (BBHOST_PRIORITY_CLASS)", pc);
            } else {
                host_log("process: priority class %s refused: error %lu", pc, GetLastError());
            }
        }
        const char* pt = std::getenv("BBHOST_POWER_THROTTLING");
        if (pt && (pt[0] == '0' || pt[0] == '1')) {
            PROCESS_POWER_THROTTLING_STATE st{};
            st.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
            st.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
            st.StateMask = pt[0] == '1' ? PROCESS_POWER_THROTTLING_EXECUTION_SPEED : 0;
            const bool ok = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st, sizeof(st));
            host_log("process: execution-speed throttling %s (BBHOST_POWER_THROTTLING=%c)%s", pt[0] == '1' ? "on" : "off", pt[0],
                     ok ? "" : ", refused");
        }
    }
#else
    host_log("bbhost %s (%s, Linux x86-64)", BBHOST_VERSION, BBHOST_GIT_REV);
    log_cpu();
    std::signal(SIGTERM, on_term);
    {
        struct sigaction sa{};
        sa.sa_sigaction = on_segv_entry;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGSEGV, &sa, nullptr);
        sigaction(SIGBUS, &sa, nullptr);
        struct sigaction st{};
        st.sa_sigaction = on_trap;
        st.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&st.sa_mask);
        sigaction(SIGTRAP, &st, nullptr);
    }
#endif
    {
        // --stream-selftest[=N] / BBHOST_STREAM_SELFTEST=N: the command
        // stream's self-test (host/stream_selftest.cpp), headless, before
        // any configuration or anything of the game; the exit code says.
        int rounds = 0;
        if (const char* e = std::getenv("BBHOST_STREAM_SELFTEST"); e && *e && *e != '0') rounds = std::max(1, std::atoi(e));
        for (int i = 1; i < argc; ++i) {
            if (std::strncmp(argv[i], "--stream-selftest", 17) == 0 && (argv[i][17] == 0 || argv[i][17] == '=')) {
                rounds = argv[i][17] == '=' ? std::max(1, std::atoi(argv[i] + 18)) : 16;
            }
        }
        if (rounds) {
            const int rc = host_gpu_stream_selftest(rounds);
            std::fflush(nullptr);
            std::_Exit(rc);  // the recorder and submission threads run on: no static destructors under them
        }
    }
    HostConfig cfg;
    std::string err;
    if (!config_load(argc, argv, &cfg, &err)) {
        host_log("%s", err.c_str());
        host_log("usage: %s [--config bbhost.toml] [--app0 DIR] [--data DIR] [--eboot FILE]"
                 " [--write-config PATH] [--headless] [--setup] [eboot]",
                 argv[0]);
        host_log("docs/running.md: where the configuration is looked for and what to put in it");
        return 2;
    }
#if defined(BBHOST_HAVE_SDL3)
    {
        // The setup window (host/launcher.h): when the game's paths are
        // missing or wrong, when asked for (--setup), or at every start
        // (startup.setup_window). Play saves and goes on with what it saved;
        // closing it ends the run.
        // Also for plugins the player turned on whose settings they have not
        // seen yet (or plugins.show_settings): it opens on the Plugins tab.
        const std::string help = config_setup_help(cfg);
        const bool plugins_tab = !cfg.headless && help.empty() && !cfg.setup_requested && plugin_ui_wants_attention();
        if (!cfg.headless && (cfg.setup_requested || cfg.setup_always || !help.empty() || plugins_tab)) {
            std::string reason = help.substr(0, help.find('\n'));
            if (reason.rfind("bbhost: ", 0) == 0) reason = reason.substr(8);
            if (launcher_run(cfg, reason, plugins_tab) == LauncherResult::Quit) return help.empty() ? 0 : 2;
            if (!config_load(argc, argv, &cfg, &err)) {
                host_log("%s", err.c_str());
                return 2;
            }
        }
    }
#endif
    // Nothing about the machine this was built on is compiled in: both paths
    // come from bbhost.toml or the command line. Say plainly what to set when
    // they are missing, and write a template when there is no file at all.
    if (const std::string help = config_setup_help(cfg); !help.empty()) {
        std::fputs(help.c_str(), stderr);
        std::fflush(stderr);
        return 2;
    }
    // Updates (host/updater.h): an older build's update leftover removed, a
    // release build's check begun on a thread of its own (not in a headless
    // run, nor when the setup window has checked).
    updater::start();
    const char* eboot = cfg.eboot.c_str();
    // Text entry defaults to typing in the window (the PC equivalent of the
    // PS4 on-screen keyboard); player.ime = "auto" answers with player.name
    // instead, which is what headless runs get anyway (no window, no typing).
    hle_dialog_set_default_name(cfg.player_name.empty() ? cfg.online_id.c_str() : cfg.player_name.c_str(),
                                cfg.ime_mode != "auto");
    hle_np_set_online_id(cfg.online_id.c_str());
    hle_video_set_fps_cap(cfg.fps_cap);
    // The PC port's own options (F10 in the window). They are read before the
    // guest is patched because one of them - the company logos - is a patch,
    // and they seed from bbhost.toml, so a run with no options file behaves
    // exactly as it did.
    host_options_load();
    {
        const HostSettings& h = host_startup_settings();
        host_log("display: render %dx%d -> output %dx%d, DLSS mode %d, frame generation %dx",
                 h.res_width, h.res_height, h.output_width, h.output_height,
                 h.dlss_mode, h.frame_generation ? h.frame_generation_factor : 1);
    }
    config_set_skip_intro(host_settings().skip_logos);
    // The PC enhancements are read the same way, once: each installs (or its
    // patch applies) while the guest is prepared, so a change waits for the
    // next start.
    {
        const HostSettings h = host_settings();
        config_set_enhancements(h.change_appearance, h.rebirth, h.five_players);
        const HostConfig& c = config();
        host_log("pc enhancements: mirror %s, rebirth %s, five players %s", c.change_appearance ? "on" : "off", c.rebirth ? "on" : "off",
                 c.five_players ? "on" : "off");
        // The debug menu was a PC Settings switch before the Debug Menu plugin
        // took its place: a player who had it on gets the plugin turned on,
        // once, and the old switch goes off.
        if (h.debug_menu) {
            if (config_value("plugins.debug_menu").empty()) {
                config_set_values(config_user_file(), {{"plugins", "debug_menu", "true"}});
                config_value_set("plugins.debug_menu", "true");
                host_log("debug-menu: PC Settings had it on; the Debug Menu plugin is turned on in its place");
            }
            host_opt_set("debug_menu", false);
            host_options_save_now();
        }
    }
    if (const App0Version v = config_app0_version(cfg.app0); v.app_ver.empty()) {
        host_log("app0: no sce_sys/param.sfo could be read in %s - is it the whole game folder?", cfg.app0.c_str());
    } else {
        host_log("app0: the game folder is version %s (category %s)%s%s", v.app_ver.c_str(), v.category.c_str(),
                 v.update.empty() ? "" : ", with its update read from ", v.update.c_str());
        if (v.app_ver != "01.09")
            host_log("app0: WARNING - the 1.09 eboot needs the 1.09 update's files, and this game folder is version %s. "
                     "Copy the update's files over it (its sce_sys/param.sfo then says APP_VER 01.09); without them "
                     "the game stops when it reads a file only the update has.",
                     v.app_ver.c_str());
    }
    hle_fs_set_roots(cfg.app0.c_str(), cfg.data.empty() ? nullptr : cfg.data.c_str(),
                     cfg.tmp.empty() ? nullptr : cfg.tmp.c_str(), eboot,
                     cfg.mods.empty() ? nullptr : cfg.mods.c_str());
    plugins_load();  // before the HLE table is bound: a plugin may replace an import
    if (host_settings().debug_camera && plugins_active("debug_menu"))
        host_log("debug camera: on with the debug menu - its LOAD TEST and DUNGEON MOVEMAP TEST crash the game "
                 "(the camera's code takes their step's place)");
    register_hle();

    ElfImage image{};
    if (!load_orbis_elf(eboot, &image)) {
        return 1;
    }
    // Only the 1.09 build runs: every patch, hook and engine layout is an
    // address in it, and another version fails far from the cause. A tester's
    // eboot of another version (2026-09-29) died before its first frame, on a
    // path pointer of 0xffffffff in the file layer. BBHOST_ANY_EBOOT=1 starts one
    // anyway, without the patches.
    if (image.sha256 != kEboot109Sha256) {
        const char* any = std::getenv("BBHOST_ANY_EBOOT");
        if (!(any && any[0] == '1')) {
            std::fputs(config_eboot_help(cfg, image.sha256, kEboot109Sha256).c_str(), stderr);
            std::fflush(stderr);
            return 3;
        }
        host_log("BBHOST_ANY_EBOOT=1: starting an eboot that is not 1.09");
    }
    sampler_start(image.mem.slide, image.mem.slide + image.mem.size);  // BBHOST_SAMPLE
    sampler_watchdog_start(image.mem.slide, image.mem.slide + image.mem.size);  // every thread's stack when the flips stop
#if defined(_WIN32)
    win_crash_note_image(image.mem.slide, image.mem.slide + image.mem.size);
#endif
    main_wait_guest_base(image.mem.slide);  // BBHOST_HLE_COUNT: wait sites as Binary Ninja addresses

    std::vector<std::int64_t> stack(static_cast<std::size_t>(argc) + 4, 0);
    // The guest sees no host flags: argc = 1, argv[0] = eboot path.
    stack[0] = 1;
    stack[1] = reinterpret_cast<std::int64_t>(eboot);
    stack[2] = 0;

    hle_patch_guest(&image);
    plugins_image(&image);  // the plugins' patches and hooks, after the host's own
    // The developers' debug menu, when the Debug Menu plugin is on: after the
    // plugins' image phase, which writes its font (engine/debug_menu.h).
    debug_menu_install(&image);

    bool windowed = false;
    host_log("config: %s headless=%d %dx%d", cfg.config_layers.empty() ? "(none)" : cfg.config_layers.c_str(),
             cfg.headless ? 1 : 0, cfg.width, cfg.height);
    if (!cfg.headless) {
        windowed = host_window_start(cfg.width, cfg.height, "Bloodborne (bbhost)");
        if (!windowed) {
            host_log("running headless");
        } else {
            host_options_apply();  // the display settings, now that there is one
        }
    }
    host_log("calling _start 0x%llx (CRT runs _init, stack=0x%zx)",
             static_cast<unsigned long long>(image.entry), kGuestMainStack);
    run_guest_start(reinterpret_cast<GuestStart>(image.entry), stack.data(), windowed);
    host_log("_start returned");
    host_window_stop();
    engine_bind(nullptr);
    unload_elf(&image);
    return 0;
}
