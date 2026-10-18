#pragma once

// The game's shader bundles (/app0/dvdroot_ps4/shader/*.shaderbnd*.dcx, read
// through the file system's overlays as the game reads them): every shader
// container in them by its name - the footer's hash, as GX's creation hook
// names a shader (render.cpp record_shader_created). The precompile reads
// them for what a draw needs before GX has created the shader: the engine's
// reflection (a lift's binding names) for a pipeline built from the pipeline
// manifest, and every vertex, pixel and compute shader for the first start's
// compile of the whole game (shader_bundles_list).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu {

struct BundleShader {
    std::string name;  // "%08x", the footer's hash
    int type = 0;      // Shdr type: 1 vertex, 2 pixel, 4 compute (others are listed too)
    std::shared_ptr<const std::vector<std::uint8_t>> container;
    bool debug = false;  // from the debug menu's bundle
    std::string file;    // the bundle entry's own name ("...\\GXFlver_Sword_Gbuf_[A].ppo")
};

// Reads the bundles once, on the first call (the others wait for it); false
// when none could be read. Thread-safe.
bool shader_bundles_load();
// The container of the shader GX names `name`, or null (also before the
// bundles are read, or when none could be).
std::shared_ptr<const std::vector<std::uint8_t>> shader_bundle_container(const std::string& name);
// Every shader in the bundles, once each by name, in the bundles' order (the
// debug bundle last).
const std::vector<BundleShader>& shader_bundles_list();
// The bundle entry's file name for the shader GX names `name` (its base name,
// "GXFlver_Sword_Gbuf_[A].ppo"), or "" (also before the bundles are read).
std::string shader_bundle_file_name(const std::string& name);
// For the report: "N shaders in M bundles (x MiB), read in y ms", or why not.
std::string shader_bundles_summary();

// The shader containers the eboot carries itself (YEBIS's, Scaleform's, the
// engine's full-screen passes: programs GX never creates), found in the file
// at `path`, once each by name; vertex, pixel and compute ones only.
std::vector<BundleShader> eboot_shaders(const std::string& path);
// The eboot the game runs (set at the device's start), and the container of
// any program the game's files hold by its name: the bundles' first, then the
// eboot's own (read once, on the first call that needs it). Null when neither
// has it.
void shader_programs_set_eboot(const std::string& path);
std::shared_ptr<const std::vector<std::uint8_t>> shader_program_container(const std::string& name);

// The name record_shader_created gives a container ("" when it has no footer).
std::string shader_container_name(const std::uint8_t* container, std::size_t size);
// Its Shdr type byte (1 vertex, 2 pixel, 4 compute), or -1.
int shader_container_type(const std::uint8_t* container, std::size_t size);

}  // namespace gpu
