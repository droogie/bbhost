/*
 * ffx_linux.h -- included first into every FidelityFX SDK source file of bbhost's Linux build of AMD's FidelityFX
 * library (-include, CMakeLists.txt), so AMD's sources build as they are. They are written for MSVC and AMD's own
 * build, and lack on Linux:
 *   - the MSVC library's bounded string functions (wcscpy_s, strcpy_s, strncpy_s with _TRUNCATE, swprintf_s), _countof
 *     and UNREFERENCED_PARAMETER, and the C++ headers MSVC's bring in with them (<cmath>, <locale>);
 *   - std::abs(float) in the global namespace, which MSVC's <cmath> declares there: without it lanczos2() in
 *     ffx_fsr3upscaler.cpp takes int abs(int), and the Lanczos table FSR 3 upsamples with on AMD GPUs reads 1 for every
 *     |x| < 1 (the build also fails on -Wabsolute-value);
 *   - __declspec(dllexport), which marks the FidelityFX API's five functions (ffx_api.h's FFX_API_ENTRY): here they are
 *     the library's default-visibility symbols (exports.map lets nothing else out);
 *   - room for 4-byte wchar_t: FSR 3's private context holds a name of 64 wchar_t for every binding of each of its
 *     pipelines, ~830 KB here against the 512 KB opaque context the SDK sizes for Windows' 2-byte wchar_t (a
 *     static_assert in ffx_fsr3upscaler.cpp); the opaque context is made 1 MB;
 *   - ffxProviderExternal, the provider AMD's driver can offer on Windows (a newer FSR from the driver), which the SDK's
 *     api/internal/ffx_provider.h and ffx_api.cpp name but its published source does not define: here none is ever
 *     valid, so FSR 3's own provider always runs.
 * The FidelityFX SDK's code is MIT, Copyright (C) Advanced Micro Devices, Inc.; this file: Copyright (c) 2026 the bbhost
 * authors, MIT (LICENSE in this folder).
 */
#pragma once
#if defined(__cplusplus)
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <locale>
#include <math.h>  // libstdc++'s: std::abs(float) and the other overloads in the global namespace too
#include <string>

#define FFX_UNUSED(x) ((void)(x))  // (ffx_util.h's own, which ffx_message.cpp uses without including it)
#define UNREFERENCED_PARAMETER(x) ((void)(x))
#define __declspec(x) __attribute__((visibility("default")))
#ifndef _TRUNCATE
#define _TRUNCATE (static_cast<size_t>(-1))
#endif
#ifndef _countof
#define _countof(a) (sizeof(a) / sizeof((a)[0]))
#endif

// MSVC's bounded copies: the destination is always terminated, and the copy cut short to fit.
inline int wcscpy_s(wchar_t* dest, size_t size, const wchar_t* src) {
    if (!dest || !size) return 22;
    if (!src) {
        dest[0] = 0;
        return 22;
    }
    size_t i = 0;
    for (; i + 1 < size && src[i]; ++i) dest[i] = src[i];
    dest[i] = 0;
    return src[i] ? 34 : 0;
}
template <size_t N>
inline int wcscpy_s(wchar_t (&dest)[N], const wchar_t* src) {
    return wcscpy_s(dest, N, src);
}
inline int strncpy_s(char* dest, size_t size, const char* src, size_t count) {
    if (!dest || !size) return 22;
    if (!src) {
        dest[0] = 0;
        return 22;
    }
    size_t i = 0;
    for (; i + 1 < size && i < count && src[i]; ++i) dest[i] = src[i];
    dest[i] = 0;
    return 0;
}
inline int strcpy_s(char* dest, size_t size, const char* src) { return strncpy_s(dest, size, src, _TRUNCATE); }
template <size_t N>
inline int strcpy_s(char (&dest)[N], const char* src) {
    return strcpy_s(dest, N, src);
}

// MSVC's wide printf reads %s and %c as wide, glibc's as narrow (%ls and %lc are wide in both): the format is
// rewritten into glibc's before it is used.
inline std::wstring ffx_linux_wide_format(const wchar_t* format) {
    std::wstring out;
    for (const wchar_t* p = format; *p; ++p) {
        out += *p;
        if (*p != L'%') continue;
        if (p[1] == L'%') {
            out += *++p;
            continue;
        }
        while (p[1] && std::wcschr(L"-+ #0123456789.*", p[1])) out += *++p;
        bool sized = false;
        while (p[1] && std::wcschr(L"hlLqjzt", p[1])) {
            out += *++p;
            sized = true;
        }
        if (!p[1]) break;
        const wchar_t c = *++p;
        if (!sized && (c == L's' || c == L'c')) out += L'l';
        out += c == L'S' ? L's' : c == L'C' ? L'c' : c;
    }
    return out;
}
inline int swprintf_s(wchar_t* buffer, size_t size, const wchar_t* format, ...) {
    va_list args;
    va_start(args, format);
    const int r = std::vswprintf(buffer, size, ffx_linux_wide_format(format).c_str(), args);
    va_end(args);
    if (r < 0 && buffer && size) buffer[size - 1] = 0;
    return r;
}
template <size_t N>
inline int swprintf_s(wchar_t (&buffer)[N], const wchar_t* format, ...) {
    va_list args;
    va_start(args, format);
    const int r = std::vswprintf(buffer, N, ffx_linux_wide_format(format).c_str(), args);
    va_end(args);
    if (r < 0) buffer[N - 1] = 0;
    return r;
}

#include <ffx_api_helper.h>
#include <ffx_internal_types.h>

// The opaque contexts' size (ffx_internal_types.h, included once): see above.
#undef FFX_SDK_DEFAULT_CONTEXT_SIZE
#define FFX_SDK_DEFAULT_CONTEXT_SIZE (1024 * 256)

// No provider from the driver: never valid, so ffx_provider.h's choice is always the library's own. A template on
// the provider class, which ffx_provider.h defines after this point and before the first use.
template <class Provider>
class ffxProviderExternalT : public Provider {
public:
    ffxProviderExternalT(void*, uint64_t) : Provider(0, 0, "") {}
    bool Valid() const { return false; }
    ffxReturnCode_t CreateContext(ffxContext*, ffxCreateContextDescHeader*, Allocator&) override { return FFX_API_RETURN_NO_PROVIDER; }
    ffxReturnCode_t DestroyContext(ffxContext*, Allocator&) override { return FFX_API_RETURN_NO_PROVIDER; }
    ffxReturnCode_t Configure(ffxContext*, const ffxConfigureDescHeader*) const override { return FFX_API_RETURN_NO_PROVIDER; }
    ffxReturnCode_t Query(ffxContext*, ffxQueryDescHeader*) const override { return FFX_API_RETURN_NO_PROVIDER; }
    ffxReturnCode_t Dispatch(ffxContext*, const ffxDispatchDescHeader*) const override { return FFX_API_RETURN_NO_PROVIDER; }
};
class ffxProvider;
using ffxProviderExternal = ffxProviderExternalT<ffxProvider>;
#endif
