#include "host/crash_dump.h"
#if defined(_WIN32)
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef BBHOST_DUMP_WAIT_MS
#define BBHOST_DUMP_WAIT_MS 600000
#endif
namespace crash_dump {
namespace {
Shared *shared = nullptr;
HANDLE mapping = nullptr, request = nullptr, done = nullptr, ready = nullptr, helper = nullptr;
std::atomic_flag capturing = ATOMIC_FLAG_INIT;
bool attempted = false;
void close(HANDLE &h) {
    if (h) {
        CloseHandle(h);
        h = nullptr;
    }
}
} // namespace
void shutdown() {
    // The helper also watches parent exit, so forced termination cannot orphan it.
    if (helper) {
        TerminateProcess(helper, 0);
        WaitForSingleObject(helper, 5000);
    }
    if (shared && shared->file_created && !attempted) {
        const std::wstring partial = std::wstring(shared->path) + L".partial";
        DeleteFileW(partial.c_str());
    }
    close(helper);
    close(request);
    close(done);
    close(ready);
    if (shared) {
        UnmapViewOfFile(shared);
        shared = nullptr;
    }
    close(mapping);
}
bool initialize() {
    wchar_t path[32768]{};
    const DWORD n = GetEnvironmentVariableW(L"BBHOST_FULL_DUMP", path, 32768);
    if (!n)
        return true;
    if (n >= 32768)
        return false;
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, &sa, PAGE_READWRITE, 0, sizeof(Shared), nullptr);
    if (mapping)
        shared = static_cast<Shared *>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Shared)));
    request = CreateEventW(&sa, FALSE, FALSE, nullptr);
    done = CreateEventW(&sa, TRUE, FALSE, nullptr);
    ready = CreateEventW(&sa, TRUE, FALSE, nullptr);
    HANDLE parent = nullptr;
    if (!shared || !request || !done || !ready ||
        !DuplicateHandle(GetCurrentProcess(), GetCurrentProcess(), GetCurrentProcess(), &parent,
                         PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | SYNCHRONIZE, TRUE, 0)) {
        shutdown();
        return false;
    }
    shared->pid = GetCurrentProcessId();
    shared->triage_error = ERROR_IO_PENDING;
    const DWORD full_n = GetFullPathNameW(path, 32768, shared->path, nullptr);
    if (!full_n || full_n >= 32768) {
        close(parent);
        shutdown();
        return false;
    }
    wchar_t own[32768]{};
    const DWORD own_n = GetModuleFileNameW(nullptr, own, 32768);
    if (!own_n || own_n >= 32768) {
        close(parent);
        shutdown();
        return false;
    }
    std::wstring exe(own);
    exe = exe.substr(0, exe.find_last_of(L"\\/") + 1) + L"crash_dump_helper.exe";
    wchar_t handles[256];
    swprintf_s(handles, L" %llu %llu %llu %llu %llu", (unsigned long long)mapping, (unsigned long long)request,
               (unsigned long long)done, (unsigned long long)ready, (unsigned long long)parent);
    std::wstring cmd = L"\"" + exe + L"\"" + handles;
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof(si);
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<unsigned char> attrs(bytes);
    si.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attrs.data());
    bool initialized = InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &bytes) != 0;
    HANDLE inherited[] = {mapping, request, done, ready, parent};
    bool configured = initialized && UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                                               inherited, sizeof(inherited), nullptr, nullptr);
    PROCESS_INFORMATION pi{};
    const bool started = configured && CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE,
                                                      CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
                                                      &si.StartupInfo, &pi);
    if (initialized)
        DeleteProcThreadAttributeList(si.lpAttributeList);
    close(parent);
    if (!started) {
        shutdown();
        return false;
    }
    CloseHandle(pi.hThread);
    helper = pi.hProcess;
    HANDLE wait[] = {ready, helper};
    const DWORD startup_result = WaitForMultipleObjects(2, wait, FALSE, 15000);
    if (startup_result != WAIT_OBJECT_0 || shared->error) {
        if (!shared->error)
            shared->error = startup_result == WAIT_TIMEOUT ? ERROR_TIMEOUT : ERROR_PROCESS_ABORTED;
        std::fprintf(stderr, "[bbhost] full dump: helper startup error=%lu\n", (unsigned long)shared->error);
        shutdown();
        return false;
    }
    std::atexit(shutdown);
    std::fprintf(stderr, "[bbhost] full dump: external collector ready; readable full memory; %u-ms capture deadline\n",
                 BBHOST_DUMP_WAIT_MS);
    return true;
}
void capture(EXCEPTION_POINTERS *ep) {
    if (!shared)
        return;
    // Serialize simultaneous failures. Later callers wait for the same capture;
    // they must not terminate the process while the first dump is in progress.
    if (capturing.test_and_set(std::memory_order_acquire)) {
        WaitForSingleObject(done, BBHOST_DUMP_WAIT_MS + 5000);
        return;
    }
    attempted = true;
    shared->thread = GetCurrentThreadId();
    if (ep) {
        shared->exception = *ep->ExceptionRecord;
        shared->context = *ep->ContextRecord;
    } else {
        RtlCaptureContext(&shared->context);
        shared->exception.ExceptionCode = 0xE0000001;
        shared->exception.ExceptionAddress = reinterpret_cast<void *>(shared->context.Rip);
    }
    shared->exception.ExceptionRecord = nullptr; // no remote linked record pointer
    SetEvent(request);
    HANDLE wait[] = {done, helper};
    DWORD result = WaitForMultipleObjects(2, wait, FALSE, BBHOST_DUMP_WAIT_MS);
    if (result != WAIT_OBJECT_0) {
        TerminateProcess(helper, 124);
        WaitForSingleObject(helper, 5000);
        shared->error = result == WAIT_TIMEOUT ? ERROR_TIMEOUT : ERROR_PROCESS_ABORTED;
        SetEvent(done);
    }
    std::fprintf(stderr, "[bbhost] triage dump: %s error=%lu\n", shared->triage_success ? "complete" : "FAILED",
                 (unsigned long)shared->triage_error);
    std::fprintf(stderr, "[bbhost] full dump: %s error=%lu (only finalized .dmp files are valid)\n",
                 shared->success ? "complete" : "FAILED", (unsigned long)shared->error);
    std::fflush(stderr);
}
} // namespace crash_dump
#endif
