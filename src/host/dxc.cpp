#include "host/dxc.h"

#include "gcn/hlsl.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace gpu {

// The few COM interfaces of dxcapi.h that a compile needs, by their vtable
// order (QueryInterface, AddRef, Release, then the interface's own methods):
// DXC's own headers are not needed to build bbhost. They are declared with
// external linkage (namespace dxc_com, not an anonymous one): in an anonymous
// namespace the compiler sees every class that could implement them - none -
// and devirtualizes the calls into __cxa_pure_virtual.
namespace dxc_com {
struct Guid {
    std::uint32_t a;
    std::uint16_t b, c;
    std::uint8_t d[8];
};
constexpr Guid kClsidCompiler = {0x73e22d93, 0xe6ce, 0x47f3, {0xb5, 0xbf, 0xf0, 0x66, 0x4f, 0x39, 0xc1, 0xb0}};
constexpr Guid kIidCompiler3 = {0x228b4687, 0x5a6a, 0x4730, {0x90, 0x0c, 0x97, 0x02, 0xb2, 0x20, 0x3f, 0x54}};
constexpr Guid kIidOperationResult = {0xcedb484a, 0xd4e9, 0x445a, {0xb9, 0x91, 0xca, 0x21, 0xca, 0x15, 0x7d, 0xc2}};
constexpr Guid kIidVersionInfo = {0xb04f5b50, 0x2059, 0x4f12, {0xa8, 0xff, 0xa1, 0xe0, 0xcd, 0xe1, 0xcc, 0x7e}};
constexpr Guid kIidVersionInfo2 = {0xfb6904c4, 0x42f0, 0x4b62, {0x9c, 0x46, 0x98, 0x3a, 0xf7, 0xda, 0x7c, 0x83}};

using HResult = std::int32_t;
#ifdef _WIN32
using WChar = wchar_t;  // UTF-16 on Windows
#else
using WChar = wchar_t;  // DXC's Linux build takes wchar_t strings too
#endif

struct Unknown {
    virtual HResult QueryInterface(const Guid& iid, void** out) = 0;
    virtual unsigned long AddRef() = 0;
    virtual unsigned long Release() = 0;
};
struct Blob : Unknown {
    virtual void* GetBufferPointer() = 0;
    virtual std::size_t GetBufferSize() = 0;
};
struct BlobEncoding : Blob {
    virtual HResult GetEncoding(bool* known, std::uint32_t* code_page) = 0;
};
struct OperationResult : Unknown {
    virtual HResult GetStatus(HResult* status) = 0;
    virtual HResult GetResult(Blob** result) = 0;
    virtual HResult GetErrorBuffer(BlobEncoding** errors) = 0;
};
struct Buffer {
    const void* ptr;
    std::size_t size;
    std::uint32_t encoding;  // DXC_CP_UTF8
};
struct Compiler3 : Unknown {
    virtual HResult Compile(const Buffer* source, const WChar** args, std::uint32_t argc, void* include_handler, const Guid& iid, void** result) = 0;
    virtual HResult Disassemble(const Buffer* object, const Guid& iid, void** result) = 0;
};
struct VersionInfo : Unknown {
    virtual HResult GetVersion(std::uint32_t* major, std::uint32_t* minor) = 0;
    virtual HResult GetFlags(std::uint32_t* flags) = 0;
};
struct VersionInfo2 : VersionInfo {
    virtual HResult GetCommitInfo(std::uint32_t* count, char** hash) = 0;
};
using CreateInstance = HResult (*)(const Guid& clsid, const Guid& iid, void** out);
constexpr std::uint32_t kCpUtf8 = 65001;

}  // namespace dxc_com
using namespace dxc_com;

namespace {

template <typename T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    void** out() { return reinterpret_cast<void**>(&p); }
};

std::wstring wide(const std::string& s) { return std::wstring(s.begin(), s.end()); }

// The directory of the running executable ("" when unknown).
std::string exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (!n || n >= MAX_PATH) return {};
    return std::filesystem::path(std::wstring(buf, n)).parent_path().string();
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    return std::filesystem::path(std::string(buf, static_cast<std::size_t>(n))).parent_path().string();
#endif
}

#ifdef _WIN32
constexpr const char* kLibName = "dxcompiler.dll";
#else
constexpr const char* kLibName = "libdxcompiler.so";
#endif

}  // namespace

std::vector<std::string> dxc_arguments(const DxcOptions& options) {
    return {
        "-spirv",
        "-fspv-target-env=vulkan1.2",  // SPIR-V 1.5, as the lifter writes; discard stays OpKill
        // Every binding and interface variable the lift declares, used or not:
        // the module declares what the pipeline layout and the other stages
        // were made for.
        "-fspv-preserve-bindings",
        "-fspv-preserve-interface",
        !options.passes.empty() ? "-Oconfig=" + options.passes
        : options.optimize <= 0 ? "-O0" : options.optimize == 1 ? "-O1" : options.optimize == 2 ? "-O2" : "-O3",
    };
}

Dxc::~Dxc() {
#ifdef _WIN32
    if (lib_) FreeLibrary(static_cast<HMODULE>(lib_));
#else
    if (lib_) dlclose(lib_);
#endif
}

namespace {

void close_library(void* h) {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(h));
#else
    dlclose(h);
#endif
}

// Opens `t` (a path, or a bare name for the system's search) and finds
// DxcCreateInstance in it; null with `why` when it cannot.
void* open_library(const std::string& t, void*& create, std::string& why) {
    create = nullptr;
#ifdef _WIN32
    const bool has_dir = t.find_first_of("\\/") != std::string::npos;
    if (has_dir && !std::filesystem::exists(std::filesystem::path(t))) {
        why = t + ": no such file";
        return nullptr;
    }
    // A path: its own directory first for what it loads in turn; a bare name:
    // Windows' search (the executable's directory, System32, PATH).
    HMODULE h = has_dir ? LoadLibraryExW(std::filesystem::path(t).wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)
                        : LoadLibraryW(std::filesystem::path(t).wstring().c_str());
    if (!h) {
        why = t + ": error " + std::to_string(GetLastError());
        return nullptr;
    }
    create = reinterpret_cast<void*>(GetProcAddress(h, "DxcCreateInstance"));
#else
    if (t.find('/') != std::string::npos && !std::filesystem::exists(t)) {
        why = t + ": no such file";
        return nullptr;
    }
    void* h = dlopen(t.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char* e = dlerror();
        why = e ? e : t;
        return nullptr;
    }
    create = dlsym(h, "DxcCreateInstance");
#endif
    if (!create) {
        why = t + ": no DxcCreateInstance";
        close_library(h);
        return nullptr;
    }
    return h;
}

// One compile through `create_fn`'s compiler: `args` besides the source.
bool run_compile(void* create_fn, const std::string& source, const std::vector<std::wstring>& args_w, std::vector<std::uint32_t>& spirv,
                 std::string& error) {
    spirv.clear();
    error.clear();
    const auto create = reinterpret_cast<CreateInstance>(create_fn);
    Ref<Compiler3> compiler;
    if (create(kClsidCompiler, kIidCompiler3, compiler.out()) != 0 || !compiler.p) {
        error = "no compiler instance";
        return false;
    }
    std::vector<const WChar*> args;
    for (const std::wstring& a : args_w) args.push_back(a.c_str());
    const Buffer src{source.data(), source.size(), kCpUtf8};
    Ref<OperationResult> result;
    const HResult hr = compiler.p->Compile(&src, args.data(), static_cast<std::uint32_t>(args.size()), nullptr, kIidOperationResult, result.out());
    if (hr != 0 || !result.p) {
        error = "DXC failed to run (" + std::to_string(hr) + ")";
        return false;
    }
    HResult status = -1;
    result.p->GetStatus(&status);
    if (status != 0) {
        Ref<BlobEncoding> errors;
        if (result.p->GetErrorBuffer(reinterpret_cast<BlobEncoding**>(errors.out())) == 0 && errors.p && errors.p->GetBufferSize()) {
            error.assign(static_cast<const char*>(errors.p->GetBufferPointer()), errors.p->GetBufferSize());
            while (!error.empty() && (error.back() == '\0' || error.back() == '\n')) error.pop_back();
        }
        if (error.empty()) error = "DXC failed (" + std::to_string(status) + ")";
        return false;
    }
    Ref<Blob> object;
    if (result.p->GetResult(reinterpret_cast<Blob**>(object.out())) != 0 || !object.p || object.p->GetBufferSize() < 20 ||
        object.p->GetBufferSize() % 4) {
        error = "DXC returned no SPIR-V";
        return false;
    }
    spirv.resize(object.p->GetBufferSize() / 4);
    std::memcpy(spirv.data(), object.p->GetBufferPointer(), spirv.size() * 4);
    if (spirv[0] != 0x07230203u) {
        spirv.clear();
        error = "DXC returned something other than SPIR-V";
        return false;
    }
    return true;
}

// A loaded library's version, "1.9 (5184 21d28f72)", and whether it compiles
// to SPIR-V at all: a DXC built without SPIR-V code generation (the DXIL-only
// builds some systems carry) cannot serve and is passed over.
bool probe(void* create_fn, std::string& version, std::string& why) {
    const auto create = reinterpret_cast<CreateInstance>(create_fn);
    Ref<Compiler3> compiler;
    if (create(kClsidCompiler, kIidCompiler3, compiler.out()) != 0 || !compiler.p) {
        why = "no compiler instance";
        return false;
    }
    Ref<VersionInfo> info;
    std::uint32_t major = 0, minor = 0;
    if (compiler.p->QueryInterface(kIidVersionInfo, info.out()) == 0 && info.p) info.p->GetVersion(&major, &minor);
    version = std::to_string(major) + "." + std::to_string(minor);
    Ref<VersionInfo2> info2;
    if (compiler.p->QueryInterface(kIidVersionInfo2, info2.out()) == 0 && info2.p) {
        std::uint32_t commits = 0;
        char* hash = nullptr;
        if (info2.p->GetCommitInfo(&commits, &hash) == 0) {
            version += " (" + std::to_string(commits) + (hash ? " " + std::string(hash) : std::string()) + ")";
            // The hash is DXC's allocation (CoTaskMemAlloc): a few bytes, once.
        }
    }
    std::vector<std::wstring> args = {L"-spirv", L"-fspv-target-env=vulkan1.2", L"-T", L"cs_6_0", L"-E", L"main"};
    std::vector<std::uint32_t> spirv;
    std::string error;
    if (!run_compile(create_fn, "[numthreads(1, 1, 1)] void main() {}\n", args, spirv, error)) {
        why = "DXC " + version + " does not compile to SPIR-V (" + error.substr(0, error.find('\n')) + ")";
        return false;
    }
    return true;
}

}  // namespace

bool Dxc::load(const std::string& path) {
    std::call_once(once_, [&] {
        tried_ = true;
        // `path`, else BBHOST_HLSL_DXC, and only that library when either
        // names one; else the one beside the executable (the packages carry
        // it there), then the system's by name.
        std::string wanted = path;
        if (wanted.empty()) {
            if (const char* e = std::getenv("BBHOST_HLSL_DXC"); e && *e) wanted = e;
        }
        std::vector<std::string> tries;
        if (!wanted.empty()) {
            tries.push_back(wanted);
        } else {
            const std::string dir = exe_dir();
            if (!dir.empty()) tries.push_back((std::filesystem::path(dir) / kLibName).string());
            tries.push_back(kLibName);
        }
        std::string why;
        for (const std::string& t : tries) {
            void* create = nullptr;
            void* h = open_library(t, create, why);
            if (!h) continue;
            std::string version, bad;
            if (!probe(create, version, bad)) {
                why = t + ": " + bad;
                close_library(h);
                continue;
            }
            lib_ = h;
            create_ = create;
            path_ = t;
            version_ = version;  // the shader cache's key holds it: another DXC compiles again
            break;
        }
        // `why` names the last library tried and what was wrong with it.
        if (!create_) error_ = "DXC not found (" + (why.empty() ? (wanted.empty() ? std::string(kLibName) : wanted) : why) + ")";
    });
    return create_ != nullptr;
}

bool Dxc::compile(const gcn::HlslResult& hlsl, const DxcOptions& options, std::vector<std::uint32_t>& spirv, std::string& error) const {
    spirv.clear();
    error.clear();
    if (!create_) {
        error = error_.empty() ? "DXC not loaded" : error_;
        return false;
    }
    std::vector<std::wstring> args_w;
    for (const std::string& a : dxc_arguments(options)) args_w.push_back(wide(a));
    for (const std::string& a : hlsl.arguments) args_w.push_back(wide(a));
    args_w.push_back(L"-HV");
    args_w.push_back(wide(hlsl.language));
    args_w.push_back(L"-T");
    args_w.push_back(wide(hlsl.profile));
    args_w.push_back(L"-E");
    args_w.push_back(wide(hlsl.entry));
    return run_compile(create_, hlsl.source, args_w, spirv, error);
}

}  // namespace gpu
