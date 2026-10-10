// Benign exceptions and owned memory, using the production crash handlers.
// No game files or memory-corruption reproduction.
#include "host/crash_dump.h"
#include "host/win_crash.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

bool host_read_safe(const void *p, void *out, std::size_t n) {
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), p, out, n, &got) && got == n;
}
void sf_heap_probe_crash_report() {}
void hle_gnm_dump_recent_writes(std::uint64_t, unsigned) {}
void hle_fs_log_recent_opens() {}

int main(int argc, char **argv) {
    // Stand-in helper: ready, then exit when asked to capture.
    if (argc == 6) {
        if (std::getenv("BBHOST_TEST_HELPER_NOT_READY")) {
            WaitForSingleObject(reinterpret_cast<HANDLE>(strtoull(argv[5], nullptr, 10)), INFINITE);
            return 70;
        }
        SetEvent(reinterpret_cast<HANDLE>(strtoull(argv[4], nullptr, 10)));
        WaitForSingleObject(reinterpret_cast<HANDLE>(strtoull(argv[2], nullptr, 10)), INFINITE);
        return 70;
    }
    win_crash_install();
    auto *sentinel = static_cast<char *>(VirtualAlloc(nullptr, 65536, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!sentinel)
        return 71;
    constexpr char marker[] = "BBHOST-CAPTURE-CHECK";
    std::memcpy(sentinel, marker, sizeof(marker));
    std::printf("sentinel=%llx tid=%lu\n", reinterpret_cast<unsigned long long>(sentinel),
                static_cast<unsigned long>(GetCurrentThreadId()));
    std::fflush(stdout);
    const std::string mode = argc > 1 ? argv[1] : "exception";
    if (mode == "clean")
        return 0;
    if (mode == "forced-exit")
        TerminateProcess(GetCurrentProcess(), 0);
    if (mode == "triage-failure") {
        wchar_t path[32768]{};
        GetEnvironmentVariableW(L"BBHOST_FULL_DUMP", path, 32768);
        // A collision introduced after readiness must be reported at capture.
        if (!CreateDirectoryW((std::wstring(path) + L".triage.dmp.partial").c_str(), nullptr))
            return 73;
    }
    if (mode == "full-final-collision") {
        wchar_t path[32768]{};
        GetEnvironmentVariableW(L"BBHOST_FULL_DUMP", path, 32768);
        HANDLE existing = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (existing == INVALID_HANDLE_VALUE)
            return 73;
        DWORD written = 0;
        if (!WriteFile(existing, "KEEP", 4, &written, nullptr) || written != 4)
            return 73;
        CloseHandle(existing);
    }
    if (mode == "early") {
        // Treat this fixture's raising module as guest code, then install a
        // terminating downstream filter: the last VEH must save evidence first.
        win_crash_note_image(1, UINT64_MAX);
        SetUnhandledExceptionFilter([](EXCEPTION_POINTERS *) -> LONG {
            TerminateProcess(GetCurrentProcess(), 99);
            return EXCEPTION_EXECUTE_HANDLER;
        });
        RaiseException(EXCEPTION_INT_DIVIDE_BY_ZERO, EXCEPTION_NONCONTINUABLE, 0, nullptr);
        return 74;
    }
    if (mode == "abort")
        std::abort();
    if (mode == "concurrent") {
        std::atomic<unsigned> arrived{0};
        auto worker = [&] {
            ++arrived;
            while (arrived.load() < 2)
                SwitchToThread();
            crash_dump::capture(nullptr);
        };
        std::thread a(worker), b(worker);
        a.join();
        b.join();
        return 0;
    }
    RaiseException(0xE0424242, EXCEPTION_NONCONTINUABLE, 0, nullptr);
    return 72;
}
