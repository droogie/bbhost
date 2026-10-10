#include "host/win_crash.h"

#if defined(_WIN32)

#include "core/portable.h"
#include "host/crash_dump.h"
#include "engine/sf_heap_probe.h"
#include "hle/fs.h"
#include "hle/modules.h"
#include "log.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <typeinfo>

namespace {

std::atomic<std::uint64_t> g_lo{0}, g_hi{0};
thread_local int t_reporting = 0;

bool in_guest(std::uint64_t a) { return a >= g_lo.load() && a < g_hi.load(); }

std::uint64_t guest_pc(std::uint64_t a) { return a - g_lo.load() + 0x400000ull; }

const char* code_name(DWORD c) {
    switch (c) {
    case EXCEPTION_ACCESS_VIOLATION: return "ACCESS_VIOLATION";
    case EXCEPTION_IN_PAGE_ERROR: return "IN_PAGE_ERROR";
    case EXCEPTION_ILLEGAL_INSTRUCTION: return "ILLEGAL_INSTRUCTION";
    case EXCEPTION_PRIV_INSTRUCTION: return "PRIV_INSTRUCTION";
    case EXCEPTION_STACK_OVERFLOW: return "STACK_OVERFLOW";
    case EXCEPTION_INT_DIVIDE_BY_ZERO: return "INT_DIVIDE_BY_ZERO";
    case EXCEPTION_INT_OVERFLOW: return "INT_OVERFLOW";
    case EXCEPTION_DATATYPE_MISALIGNMENT: return "DATATYPE_MISALIGNMENT";
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED: return "ARRAY_BOUNDS_EXCEEDED";
    case EXCEPTION_FLT_DIVIDE_BY_ZERO: return "FLT_DIVIDE_BY_ZERO";
    case EXCEPTION_FLT_INVALID_OPERATION: return "FLT_INVALID_OPERATION";
    case EXCEPTION_NONCONTINUABLE_EXCEPTION: return "NONCONTINUABLE";
    default: return "exception";
    }
}

// The codes that are crashes. Breakpoints and single steps belong to the
// diagnostics, 0x20474343 ('GCC') is a C++ throw in this toolchain,
// 0x406d1388 names a thread, and the RPC/C++ codes of libraries pass through
// to their own handlers.
bool reportable(DWORD c) {
    switch (c) {
    case EXCEPTION_ACCESS_VIOLATION:
    case EXCEPTION_IN_PAGE_ERROR:
    case EXCEPTION_ILLEGAL_INSTRUCTION:
    case EXCEPTION_PRIV_INSTRUCTION:
    case EXCEPTION_STACK_OVERFLOW:
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    case EXCEPTION_INT_OVERFLOW:
    case EXCEPTION_DATATYPE_MISALIGNMENT:
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
    case EXCEPTION_FLT_INVALID_OPERATION:
    case EXCEPTION_NONCONTINUABLE_EXCEPTION: return true;
    default: return false;
    }
}

bool rd(std::uint64_t a, std::uint64_t* v) {
    return host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), v, 8);
}

void where_log(std::uint64_t a, const char* what) {
    char buf[320];
    if (in_guest(a)) {
        host_log("  %s %llx: guest 0x%llx", what, static_cast<unsigned long long>(a),
                 static_cast<unsigned long long>(guest_pc(a)));
    } else if (win_crash_where(a, buf, sizeof(buf))) {
        host_log("  %s %llx: %s", what, static_cast<unsigned long long>(a), buf);
    }
}

void report(EXCEPTION_POINTERS* ep) {
    const EXCEPTION_RECORD* r = ep->ExceptionRecord;
    const CONTEXT* c = ep->ContextRecord;
    const std::uint64_t pc = c->Rip;
    std::uint64_t addr = 0;
    const char* kind = "";
    if ((r->ExceptionCode == EXCEPTION_ACCESS_VIOLATION || r->ExceptionCode == EXCEPTION_IN_PAGE_ERROR) &&
        r->NumberParameters >= 2) {
        addr = static_cast<std::uint64_t>(r->ExceptionInformation[1]);
        kind = r->ExceptionInformation[0] == 0 ? " (read)" : r->ExceptionInformation[0] == 1 ? " (write)" : " (execute)";
    }
    // The first line is the same shape as Linux's, so the log readers match.
    host_log("SIGSEGV pc=0x%llx addr=0x%llx (guest pc 0x%llx) [%s 0x%lx%s, thread %u]",
             static_cast<unsigned long long>(pc), static_cast<unsigned long long>(addr),
             static_cast<unsigned long long>(in_guest(pc) ? guest_pc(pc) : pc), code_name(r->ExceptionCode),
             static_cast<unsigned long>(r->ExceptionCode), kind, static_cast<unsigned>(GetCurrentThreadId()));
    host_log("  rax=%llx rdi=%llx rsi=%llx rdx=%llx rcx=%llx r8=%llx r13=%llx r15=%llx rsp=%llx rbp=%llx",
             static_cast<unsigned long long>(c->Rax), static_cast<unsigned long long>(c->Rdi),
             static_cast<unsigned long long>(c->Rsi), static_cast<unsigned long long>(c->Rdx),
             static_cast<unsigned long long>(c->Rcx), static_cast<unsigned long long>(c->R8),
             static_cast<unsigned long long>(c->R13), static_cast<unsigned long long>(c->R15),
             static_cast<unsigned long long>(c->Rsp), static_cast<unsigned long long>(c->Rbp));
    host_log("  r9=%llx r10=%llx r11=%llx r12=%llx r14=%llx rbx=%llx", static_cast<unsigned long long>(c->R9),
             static_cast<unsigned long long>(c->R10), static_cast<unsigned long long>(c->R11),
             static_cast<unsigned long long>(c->R12), static_cast<unsigned long long>(c->R14),
             static_cast<unsigned long long>(c->Rbx));
    if (in_guest(pc)) {
        std::uint8_t code[16] = {};
        if (host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(pc)), code, sizeof(code))) {
            host_log("  code: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x", code[0], code[1], code[2],
                     code[3], code[4], code[5], code[6], code[7], code[8], code[9], code[10], code[11]);
        }
    }
    std::uint64_t chain_returns[8] = {};
    {
        const std::uint64_t rsp = c->Rsp;
        std::uint64_t sp[8] = {};
        bool have = true;
        for (int k = 0; k < 8 && have; ++k) have = rd(rsp + 8u * static_cast<unsigned>(k), &sp[k]);
        if (have) {
            host_log("  stack %llx: %llx %llx %llx %llx  %llx %llx %llx %llx", static_cast<unsigned long long>(rsp),
                     static_cast<unsigned long long>(sp[0]), static_cast<unsigned long long>(sp[1]),
                     static_cast<unsigned long long>(sp[2]), static_cast<unsigned long long>(sp[3]),
                     static_cast<unsigned long long>(sp[4]), static_cast<unsigned long long>(sp[5]),
                     static_cast<unsigned long long>(sp[6]), static_cast<unsigned long long>(sp[7]));
        }
        // Guest code keeps frame pointers: the return addresses up the rbp
        // chain, while each frame sits above the last within this stack.
        // Host code does not (rbp is a general register there), so the
        // chain is only read for a fault in guest code.
        std::uint64_t rbp = c->Rbp, top = rbp;
        char chain[512];
        int len = 0;
        int depth = 0;
        for (; in_guest(pc) && rbp > rsp && rbp - rsp < (256u << 20) && (rbp & 7) == 0; ++depth) {
            std::uint64_t next = 0, ret = 0;
            if (!rd(rbp, &next) || !rd(rbp + 8, &ret)) break;
            if (depth < 16 && len < static_cast<int>(sizeof(chain)) - 20) {
                len += std::snprintf(chain + len, sizeof(chain) - static_cast<std::size_t>(len), " %llx",
                                     static_cast<unsigned long long>(in_guest(ret) ? guest_pc(ret) : ret));
            }
            if (depth < 8) chain_returns[depth] = ret;
            top = rbp;
            if (next <= rbp) break;
            rbp = next;
        }
        if (len) {
            host_log("  frames (%d, over %llu KiB of stack; guest addresses as Binary Ninja's):%s", depth,
                     static_cast<unsigned long long>((top - rsp) >> 10), chain);
        }
    }
    {
        where_log(pc, "pc");
        for (const std::uint64_t ret : chain_returns) {
            if (ret && !in_guest(ret)) where_log(ret, "frame");
        }
        // Stack words that point into a module: return addresses among them.
        const std::uint64_t rsp = c->Rsp;
        for (int k = 0, shown = 0; k < 128 && shown < 16; ++k) {
            std::uint64_t w = 0;
            if (!rd(rsp + 8u * static_cast<unsigned>(k), &w)) break;
            char buf[320];
            if (w > 0x10000 && !in_guest(w) && win_crash_where(w, buf, sizeof(buf))) {
                host_log("  stack word %llx: %s", static_cast<unsigned long long>(w), buf);
                ++shown;
            }
        }
    }
    sf_heap_probe_crash_report();
    hle_fs_log_recent_opens();
    hle_gnm_dump_recent_writes(0, 48);
}

LONG CALLBACK crash_veh(EXCEPTION_POINTERS* ep) {
    if (!reportable(ep->ExceptionRecord->ExceptionCode)) return EXCEPTION_CONTINUE_SEARCH;
    // A fault inside the report itself: say so and let the process go.
    if (t_reporting++ > 0) {
        host_log("crash: a second fault (0x%lx at 0x%llx) while reporting the first",
                 static_cast<unsigned long>(ep->ExceptionRecord->ExceptionCode),
                 static_cast<unsigned long long>(ep->ContextRecord->Rip));
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Save guest faults before reporting or a later filter can terminate us.
    // Earlier VEH handlers still get first chance to handle expected faults.
    if (in_guest(ep->ContextRecord->Rip)) crash_dump::capture(ep);
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_STACK_OVERFLOW) {
        // Little stack is left; the one line is what fits.
        host_log("SIGSEGV pc=0x%llx STACK_OVERFLOW rsp=0x%llx thread %u",
                 static_cast<unsigned long long>(ep->ContextRecord->Rip),
                 static_cast<unsigned long long>(ep->ContextRecord->Rsp), static_cast<unsigned>(GetCurrentThreadId()));
        return EXCEPTION_CONTINUE_SEARCH;
    }
    report(ep);
    return EXCEPTION_CONTINUE_SEARCH;
}

LONG WINAPI crash_filter(EXCEPTION_POINTERS* ep) {
    crash_dump::capture(ep);
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    // 0x20474343 ("CCG "): a C++ exception nothing caught, which reaches this
    // filter before std::terminate would. The vectored handler reports only
    // faults (every thrown exception passes it first), so the report of
    // where it was thrown is made here.
    if (!reportable(code) && t_reporting++ == 0) report(ep);
    hle_fs_log_recent_opens();
    host_log("crash: unhandled exception 0x%lx%s at 0x%llx; exiting 139", static_cast<unsigned long>(code),
             code == 0x20474343 ? " (a C++ exception nothing caught)" : "", static_cast<unsigned long long>(ep->ContextRecord->Rip));
    std::fflush(nullptr);
    TerminateProcess(GetCurrentProcess(), 139);  // not ExitProcess: DLL detach under the loader lock can hang (main.cpp)
    _exit(139);
    return EXCEPTION_EXECUTE_HANDLER;
}

// The calling thread's stack as far as unwind tables reach (to the HLE
// trampoline when a guest thread called in).
void log_own_stack() {
    void* frames[40];
    const USHORT n = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) where_log(reinterpret_cast<std::uint64_t>(frames[i]), "frame");
}

// Ends that pass no exception to the handlers above - the UCRT turns abort()
// and a CRT function's invalid argument into a fail-fast, and std::terminate
// calls abort() - so a run that took one stopped without a line. A Windows
// laptop's runs did, every time, at the same point. Each now says what it was
// and where, then leaves with 134.
[[noreturn]] void end_after(const char* what) {
    crash_dump::capture(nullptr);
    host_log("crash: %s (thread %lu)", what, static_cast<unsigned long>(GetCurrentThreadId()));
    log_own_stack();
    hle_fs_log_recent_opens();
    std::fflush(nullptr);
    TerminateProcess(GetCurrentProcess(), 134);  // not _exit: it goes through ExitProcess, which hung here under wine
    _exit(134);
}
void on_abort_signal(int) { end_after("abort() was called"); }
void on_terminate() {
    const char* what = "std::terminate with no exception";
    std::string text;
    if (std::exception_ptr e = std::current_exception()) {
        try {
            std::rethrow_exception(e);
        } catch (const std::exception& ex) {
            text = std::string("std::terminate: uncaught ") + typeid(ex).name() + ": " + ex.what();
            what = text.c_str();
        } catch (...) {
            what = "std::terminate: uncaught exception of an unknown type";
        }
    }
    end_after(what);
}
void on_invalid_parameter(const wchar_t*, const wchar_t*, const wchar_t*, unsigned, std::uintptr_t) {
    end_after("a C runtime function was given an invalid argument");
}
void on_purecall() { end_after("a pure virtual function was called"); }

}  // namespace

void win_crash_install() {
    if (!crash_dump::initialize()) {
        std::fprintf(stderr, "[bbhost] full dump: collector initialization FAILED; stopping before gameplay\n");
        std::fflush(stderr);
        TerminateProcess(GetCurrentProcess(), 78);
    }
    static PVOID h = AddVectoredExceptionHandler(0, crash_veh);  // last of the vectored handlers
    (void)h;
    SetUnhandledExceptionFilter(crash_filter);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    std::signal(SIGABRT, on_abort_signal);
    std::set_terminate(on_terminate);
    _set_invalid_parameter_handler(on_invalid_parameter);
    _set_purecall_handler(on_purecall);
    // The system's crash dialog would hold an unattended run open.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
}

void win_crash_note_image(std::uint64_t lo, std::uint64_t hi) {
    g_lo.store(lo);
    g_hi.store(hi);
}

bool win_crash_where(std::uint64_t a, char* out, std::size_t cap) {
    HMODULE h = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(static_cast<std::uintptr_t>(a)), &h) ||
        !h) {
        return false;
    }
    char path[MAX_PATH] = {};
    GetModuleFileNameA(h, path, sizeof(path));
    const char* name = std::strrchr(path, '\\');
    name = name ? name + 1 : path;
    const auto base = reinterpret_cast<std::uint64_t>(h);
    std::uint64_t link = 0;
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(h);
    if (dos->e_magic == IMAGE_DOS_SIGNATURE) {
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(reinterpret_cast<const std::uint8_t*>(h) + dos->e_lfanew);
        if (nt->Signature == IMAGE_NT_SIGNATURE) link = nt->OptionalHeader.ImageBase + (a - base);
    }
    std::snprintf(out, cap, "%s+0x%llx (link 0x%llx)", name[0] ? name : "?", static_cast<unsigned long long>(a - base),
                  static_cast<unsigned long long>(link));
    return true;
}

#endif
