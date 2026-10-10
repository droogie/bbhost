#include "host/crash_dump.h"
#include <cstdlib>
#include <dbghelp.h>
#include <string>

int wmain(int argc, wchar_t **argv) {
    if (argc != 6)
        return 64;
    HANDLE mapping = (HANDLE)wcstoull(argv[1], nullptr, 10), request = (HANDLE)wcstoull(argv[2], nullptr, 10);
    HANDLE done = (HANDLE)wcstoull(argv[3], nullptr, 10), ready = (HANDLE)wcstoull(argv[4], nullptr, 10);
    HANDLE parent = (HANDLE)wcstoull(argv[5], nullptr, 10);
    auto *s = static_cast<crash_dump::Shared *>(
        MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(crash_dump::Shared)));
    if (!s)
        return 65;
    // Open exclusively before acknowledging readiness. Never replace old evidence.
    const std::wstring final = s->path, partial = final + L".partial";
    const std::wstring triage = final + L".triage.dmp", triage_partial = triage + L".partial";
    const std::wstring active = final + L".capturing";
    for (const auto &path : {final, triage, triage_partial, active}) {
        if (GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES) {
            s->error = ERROR_FILE_EXISTS;
            SetEvent(ready);
            return 66;
        }
    }
    HANDLE file = CreateFileW(partial.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        s->error = GetLastError();
        SetEvent(ready);
        return 67;
    }
    s->file_created = 1;
    SetEvent(ready);
    HANDLE wait[] = {request, parent};
    if (WaitForMultipleObjects(2, wait, FALSE, INFINITE) != WAIT_OBJECT_0) {
        CloseHandle(file);
        DeleteFileW(partial.c_str());
        return 0;
    }
    HANDLE marker = CreateFileW(active.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    if (marker == INVALID_HANDLE_VALUE) {
        s->error = GetLastError();
        CloseHandle(file);
        SetEvent(done);
        return 69;
    }
    CloseHandle(marker); // lets the launcher give dump writing its own deadline
    EXCEPTION_POINTERS ep{&s->exception, &s->context};
    MINIDUMP_EXCEPTION_INFORMATION info{s->thread, &ep, FALSE};
    // Publish a small, independently usable dump BEFORE the expensive full
    // memory pass. A timeout must not discard exception/thread evidence.
    HANDLE small = CreateFileW(triage_partial.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
    if (small != INVALID_HANDLE_VALUE) {
        bool small_ok =
            MiniDumpWriteDump(parent, s->pid, small,
                              static_cast<MINIDUMP_TYPE>(MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules), &info,
                              nullptr, nullptr) != FALSE;
        DWORD small_error = small_ok ? 0 : GetLastError();
        if (small_ok && !FlushFileBuffers(small)) {
            small_ok = false;
            small_error = GetLastError();
        }
        CloseHandle(small);
        if (small_ok && !MoveFileExW(triage_partial.c_str(), triage.c_str(), MOVEFILE_WRITE_THROUGH)) {
            small_ok = false;
            small_error = GetLastError();
        }
        s->triage_error = small_error;
        s->triage_success = small_ok ? 1 : 0;
    } else
        s->triage_error = GetLastError();
#ifdef BBHOST_DUMP_TEST_DELAY_MS
    // Test-only: deterministically time out AFTER independently saving triage.
    Sleep(BBHOST_DUMP_TEST_DELAY_MS);
#endif
    const auto flags = static_cast<MINIDUMP_TYPE>(MiniDumpWithFullMemory | MiniDumpWithFullMemoryInfo |
                                                  MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules);
    bool ok = MiniDumpWriteDump(parent, s->pid, file, flags, &info, nullptr, nullptr) != FALSE;
    DWORD error = ok ? 0 : GetLastError();
    if (ok && !FlushFileBuffers(file)) {
        ok = false;
        error = GetLastError();
    }
    CloseHandle(file);
    if (ok && !MoveFileExW(partial.c_str(), final.c_str(), MOVEFILE_WRITE_THROUGH)) {
        ok = false;
        error = GetLastError();
    }
    s->success = ok ? 1 : 0;
    s->error = error;
    SetEvent(done);
    return ok ? 0 : 68;
}
