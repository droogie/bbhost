#pragma once
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Only used by the opt-in Windows crash collector. All handles are created
// before gameplay, and DbgHelp runs in a separate process.
namespace crash_dump {
struct Shared {
    DWORD pid, thread, error;
    LONG success, file_created;
    DWORD triage_error;
    LONG triage_success;
    EXCEPTION_RECORD exception;
    CONTEXT context;
    wchar_t path[32768];
};
bool initialize();
void capture(EXCEPTION_POINTERS *exception);
void shutdown();
} // namespace crash_dump
#endif
