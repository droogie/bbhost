#pragma once

// Edited shaders: HLSL in place of a game shader's, from the player's
// <data>/bbhost/shaders (and other owners: the plugins'), each named as
// BBHOST_HLSL_DUMP names the shader's file: <stable>.<stage> (every variant
// of the program in that stage whose interface it declares),
// <stable>.<stage>~<lift hash> (one variant), <lift hash>.<stage> (a program
// whose name is not known).
//
// They apply while the game runs. A program's overrides are one set, named by
// a salt - the hash of its sources' names, identities and owners (a text's
// own hash; a file's path, size and modification time) - which the renderer
// keeps on the program (render.cpp CachedProgram, gpu.cpp CachedCompute) and
// mixes into its draws' and dispatches' keys and into the key of every lifted
// stage it builds from it. A stage compiles the sources its salt names
// (override_pick), and a set never changes once a salt names it: a stage's
// module is the source of its key whatever changed since, a change builds the
// pipelines of the one program it touches again, and a return to an earlier
// source finds the stages and pipelines made for it. Salted stages and
// pipelines are kept for the run only: never in the stage or pipeline
// manifests, the translation cache or the pipeline cache's file. With no
// override the generation (override_gen) stays 0 and every salt is 0: one
// load a draw.
//
// Owners: the player's folder (kOverridePlayer) comes first, then the others
// in their order. A later owner's source of the same name is not used
// (shadowed); within one owner its texts come before its folders' files, and
// a folder before the ones added after it.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu {

constexpr int kOverridePlayer = 0;  // the player's folder; other owners from 1, in their order
constexpr std::uint32_t kOverrideDataBytes = 65536;  // an owner's shader data (BB_SHADER_DATA_BYTES)

// What a stage compiles with: one source as a set holds it.
struct OverrideSource {
    std::string name;   // the file's name less ".hlsl": "GXGuiShader_Tex.ppo@6d921b0b.ps"
    int owner = kOverridePlayer;
    std::string path;   // a file's
    std::shared_ptr<const std::string> text;  // a text given as such (a file's is read where it compiles)
    std::uint64_t size = 0;
    std::int64_t mtime = 0;      // a file's, as listed
    std::uint64_t identity = 0;  // the text's hash, or the file's path, size and time
};

// At device start, after the Shaders setting (hlsl_set_setting): `on` - the
// HLSL path is on and DXC loads - else `why_off`. `bbhost_dir`: <data>/bbhost,
// whose shaders/ is the player's folder (listed here; empty: none).
void overrides_start(bool on, const std::string& why_off, const std::string& bbhost_dir);
bool overrides_on();

// An owner other than the player's folder: named for the logs, with a folder
// of its own that relative folders start from; off while `on` is false
// (its sources wait); gone with all its sources.
void overrides_owner(int owner, const std::string& name, const std::string& own_dir, bool on);
void overrides_owner_on(int owner, bool on);
void overrides_drop_owner(int owner);
// A text for a shader (`hlsl` null: none): 0, or 1 for a name that is no
// shader's, an unknown owner or a text over 4 MiB.
int overrides_set_text(int owner, const char* shader, const char* hlsl);
// Every <shader>.hlsl in a folder (absolute, or from the owner's own; null
// or "": every folder of the owner's taken away): the number of them, or -1
// when it is not a folder.
int overrides_add_folder(int owner, const char* dir);
// A new or changed source is compiled ahead, for every variant of its shader
// met so far, before it takes effect (override_prepare, override_install):
// the draws keep the source they had until then, so a change compiles nothing
// on the thread that draws. One taken away goes at once. True while none
// waits (the tests').
bool overrides_ahead_idle();

// ---- The renderer's ----
// Bumped at each change of what applies; 0 until anything does.
extern std::atomic<std::uint64_t> g_override_gen;
inline std::uint64_t override_gen() { return g_override_gen.load(std::memory_order_acquire); }
// The salt of the overrides that apply to a program (its GX name or footer
// hash, as the renderer names it); 0 for none.
std::uint64_t override_salt(const std::string& program);
// Some override is named by a lift's hash alone: each lift's hash is wanted
// (override_lift_seen), for the program it belongs to.
bool override_fallback_wanted();
void override_lift_seen(std::uint64_t lift_hash, const std::string& program);

// ---- translation_cache.cpp's (the compile) ----
// The source a stage of `stable` (its stage tag, its lift's hash as 16 hex
// digits) compiles with under `salt`: the variant's own, then the shader's,
// then the lift hash's. False when the set names none.
bool override_pick(std::uint64_t salt, const std::string& stable, const std::string& stage, const std::string& hash16,
                   std::shared_ptr<const OverrideSource>& out);
// A file's text as it was listed; false with `why` when it changed since - it
// is listed again then, so its shader is built again at its next draw.
bool override_read(const OverrideSource& s, std::string& text, std::string& why);
// The text DXC compiles: #lines that keep DXC's lines the file's, and
// bbhost.hlsli spliced in where it is included (`#include "bbhost.hlsli"`:
// DXC is given no include handler).
std::string override_dxc_text(const OverrideSource& s, const std::string& text);
// bbhost.hlsli: the helpers an edited shader reads its owner's shader data
// with (BBHOST_DATA, its device address, defined for an owner with data);
// the dump writes it beside the shaders for an editor.
const char* override_include_text();
// What a stage's build made of a source: taken, or refused and why.
void override_outcome(const OverrideSource& s, bool taken, const std::string& why);
// The source for the logs: "shaders/<file>" or "<owner>'s <shader>".
std::string override_label(const OverrideSource& s);
// A new or changed source's compile ahead: true when it is queued for the
// lifts met so far that its name applies to - override_install takes it once
// they are compiled; false when none was met (the caller installs it now).
bool override_prepare(const std::shared_ptr<const OverrideSource>& s);
// A source compiled ahead applies now, unless another took its place.
void override_install(const std::shared_ptr<const OverrideSource>& s);
// A lift met for a program has this hash (a <hash>.<stage> source's compile ahead).
void override_attribute(std::uint64_t lift_hash, const std::string& program);
// The HLSL this build prints for a shader named as the dump names its files
// (BbHostApi::shader_source): "<name>.<stage>", the first variant met this
// run; "<name>.<stage>~<lift hash>" (or "<lift hash>.<stage>"), that one -
// as the dump writes it. 0 with it in `out`; 1 when no stage of it was met
// yet; 2 when edited shaders do not apply.
int hlsl_override_source(const std::string& shader, std::string& out);

// ---- An owner's shader data (BbHostApi::shader_data) ----
// kOverrideDataBytes an owner, 0 at the start: written into a shadow at any
// time; latched where the command processor ends a frame
// (override_data_latch, host_gpu_frame_recorded), so each frame reads one
// copy; uploaded in command order at the next recording's start
// (override_data_take, render.cpp shader_data_upload_locked). Its buffer is
// made at the owner's first compile (override_data_address, through
// render.cpp's maker, the shadow its first contents): the address its
// shaders are compiled with.
int overrides_data_write(int owner, std::uint32_t offset, const void* data, std::uint32_t size);
std::uint64_t override_data_address(int owner);  // 0: none (the player's folder; no buffer)
void overrides_set_data_maker(std::uint64_t (*make)(int owner, const std::uint8_t* bytes));
void override_data_latch();
struct OverrideUpload {
    int owner = 0;
    std::uint32_t offset = 0;
    std::vector<std::uint8_t> bytes;
};
// The latched writes for this recording's start (and the pending ones when
// no frame latched them for 100 ms: a flip by another path); false for none.
bool override_data_take(std::vector<OverrideUpload>& out);

// ---- States and reports ----
// What became of an owner's source of a name: as BbHostApi::shader_state's
// BB_SHADER_* (none, waiting, taken, refused, shadowed, off), and why.
enum OverrideState : int { kOverrideNone = 0, kOverrideWaiting, kOverrideTaken, kOverrideRefused, kOverrideShadowed, kOverrideOff };
int overrides_state(int owner, const std::string& name, std::string* why);
// A module compiled (`us` of DXC's time), a stage built with an edited
// shader's module (translation_cache.cpp), a pipeline made with a salted
// stage (render.cpp, gpu.cpp): the window's and the exit's counts.
void override_note_compile(std::uint64_t us);
void override_note_stage();
void override_note_pipeline();
// The exit report's lines: every edited shader, who gave it and what became
// of it (empty without any); and the 300-flip report's since the last call.
std::vector<std::string> overrides_report();
std::string overrides_window();

// ---- The watch (BBHOST_SHADER_WATCH=1, or a period in milliseconds) ----
// Every folder of edited shaders - the player's and the other owners' - is
// listed again every half second on bb-shwatch (from overrides_start): a new
// or changed file is compiled ahead and then applies, one gone goes. One
// listing of them all; true when it found any change (the tests call it).
bool overrides_watch_scan();
// A name split as a file's: false when it is not a shader's.
bool override_parse(const std::string& name, std::string& stable, std::string& stage, std::string& hash16);

}  // namespace gpu
