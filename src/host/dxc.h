#pragma once

// DXC, the HLSL compiler, loaded at run time: HLSL to SPIR-V for Vulkan.
//
// bbhost does not link it. libdxcompiler.so (dxcompiler.dll on Windows) is
// loaded from BBHOST_HLSL_DXC when that names one, else from the directory
// of the executable - the packages carry DXC's release library there
// (tools/fetch_dxc.sh) - then by name from the system's library path. A
// library that cannot make a compiler or does not compile to SPIR-V (a
// DXIL-only build) is passed over for the next. Nothing else of DXC is needed
// (not dxil.dll: no DXIL is signed or validated, the target is SPIR-V).
// Without it the HLSL path is off and every lifted shader keeps its SPIR-V
// from the lifter.
//
// One Dxc may compile from several threads at once: each call takes its own
// compiler instance.

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace gcn {
struct HlslResult;
}

namespace gpu {

struct DxcOptions {
    int optimize = 3;    // -O0 .. -O3
    std::string passes;  // non-empty: -Oconfig=<passes> (spirv-opt's, comma-separated) in place of -O
};

class Dxc {
public:
    Dxc() = default;
    ~Dxc();
    Dxc(const Dxc&) = delete;
    Dxc& operator=(const Dxc&) = delete;

    // Loads the library (once; later calls return the first result). `path`
    // empty: BBHOST_HLSL_DXC, else next to the executable, then by name from
    // the library path.
    bool load(const std::string& path = {});
    bool loaded() const { return create_ != nullptr; }
    const std::string& error() const { return error_; }
    // The library's version, "1.9 (5184 21d28f72)", and where it was loaded from.
    const std::string& version() const { return version_; }
    const std::string& path() const { return path_; }

    // Compiles `hlsl` (its source, profile and entry point) to SPIR-V for
    // Vulkan 1.2. False with `error` holding DXC's messages.
    bool compile(const gcn::HlslResult& hlsl, const DxcOptions& options, std::vector<std::uint32_t>& spirv, std::string& error) const;

private:
    void* lib_ = nullptr;
    void* create_ = nullptr;  // DxcCreateInstance
    std::string error_, version_, path_;
    std::once_flag once_;
    bool tried_ = false;
};

// The arguments compile() passes, besides the profile and entry point.
std::vector<std::string> dxc_arguments(const DxcOptions& options);

}  // namespace gpu
