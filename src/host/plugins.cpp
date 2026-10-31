#include "host/plugins.h"
#include "host/plugin_address.h"

#include "bbhost_plugin.h"
#include "core/config.h"
#include "core/elf.h"
#include "core/imports.h"
#include "core/memory.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "engine/frame_rate.h"
#include "engine/graphics_patch.h"
#include "engine/gx_state.h"
#include "engine/image_text.h"
#include "decomp/events/flag_store.h"
#include "decomp/gx/render_queue.h"
#include "gcn/container.h"
#include "engine/event_flags.h"
#include "engine/player_data.h"
#include "engine/world_chr.h"
#include "engine/lua_events.h"
#include "engine/loading.h"
#include "engine/paramdef.h"
#include "engine/params.h"
#include "engine/patch_manifest.h"
#include "bbhost/engine/symbols.hpp"
#include "bbhost/sdk.hpp"
#include "core/thunk.h"
#include "guest_abi.h"
#include "hle/fs.h"
#include "hle/modules.h"
#include "host/bindings.h"
#include "host/ruleset.h"
#include "host/updater.h"
#include "host/window.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

struct Plugin {
    std::string name, path;
    void* handle = nullptr;
    int (*init)(const BbHostApi*) = nullptr;
    int (*image)(const BbHostApi*) = nullptr;
    const BbPluginInfo* info = nullptr;
    const BbOption* options = nullptr;  // bb_plugin_options, ended by a zero type
    bool official = false;  // signed with the release key
    bool visitor = false;   // off, but loaded to play other players' worlds (BB_PLUGIN_ADOPTS_RULES)
};
std::vector<Plugin> g_plugins;
ElfImage* g_image = nullptr;
bool g_early = true;  // register_hle allowed
bool g_in_image = false;  // the plugins' image phase: replace_text allowed
std::string g_config_value;

struct Hook {
    BbHookFn fn;
    BbHookFn2 fn2;  // version 8: hook2
    void* user;
};
std::vector<Hook> g_hooks;  // by id, from 1

void* sym(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

void* open_lib(const std::string& path, std::string* err) {
#if defined(_WIN32)
    HMODULE h = LoadLibraryA(path.c_str());
    if (!h) *err = "error " + std::to_string(GetLastError());
    return h;
#else
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) *err = dlerror();
    return h;
#endif
}

void close_lib(void* handle) {
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

// --- the API -----------------------------------------------------------

void api_log(const char* fmt, ...) {
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    host_log("plugin: %s", line);
}

const char* api_config(const char* key) {
    g_config_value = key ? config_value(key) : "";
    return g_config_value.c_str();
}

const char* api_data_dir() { return hle_fs_data_root(); }
const char* api_mods_dir() { return hle_fs_mods_root(); }

int api_register_hle(const char* name, void* fn) {
    if (!g_early || !name || !fn) return 1;
    register_hle_fn(name, fn);
    host_log("plugin: %s replaced by a plugin", name);
    return 0;
}

const char* api_eboot_sha256() { return g_image ? g_image->sha256.c_str() : ""; }
int api_eboot_is_109() { return g_image && g_image->sha256 == kEboot109Sha256; }

// A Binary Ninja address inside the image, else (above the image's own
// range) a runtime address as given.
std::uint64_t runtime_of(std::uint64_t bn, std::size_t n) {
    if (!g_image) return 0;
    const GuestMemory& mem = g_image->mem;
    if (bn >= kPreferredGuestSlide && bn - kPreferredGuestSlide + n <= mem.size) return mem.slide + (bn - kPreferredGuestSlide);
    return 0;
}

std::uint64_t api_guest_addr(std::uint64_t bn) { return api_eboot_is_109() ? runtime_of(bn, 1) : 0; }

// A Binary Ninja address inside the image, else a runtime address: above the
// image, or anywhere the guest has mapped - under Windows its heaps lie below
// the slid image too (host/plugin_address.h).
int api_read(std::uint64_t bn, void* out, std::size_t n) {
    if (!g_image) return 1;
    const std::uint64_t at = plugin_address::read_address(bn, n, g_image->mem.slide, g_image->mem.size,
                                                          [](std::uint64_t a, std::size_t len) { return hle_kernel_va_mapped(a, len); });
    if (!at) return 1;
    return host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), out, n) ? 0 : 1;
}

int api_write(std::uint64_t bn, const void* data, std::size_t n) {
    const std::uint64_t at = runtime_of(bn, n);
    if (!at || !api_eboot_is_109()) return 1;
    GuestMemory& mem = g_image->mem;
    const std::uint64_t lo = at & ~0xfffull, hi = (at + n + 0xfff) & ~0xfffull;
    if (!guest_protect_rw(&mem, lo, hi - lo)) return 2;
    std::memcpy(guest_ptr(mem, at), data, n);
    return 0;
}

// Version 11 (engine/image_text.h).
int api_replace_text(const BbText* texts, std::size_t count) {
    if (!g_in_image || !g_image || !api_eboot_is_109() || (!texts && count)) return -1;
    std::vector<ImageText> items;
    items.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (texts[i].bn_addr < kPreferredGuestSlide || !texts[i].text) continue;
        ImageText t;
        t.at = texts[i].bn_addr - kPreferredGuestSlide;
        for (const std::uint16_t* c = texts[i].text; *c && t.text.size() < 4096; ++c) t.text.push_back(static_cast<char16_t>(*c));
        items.push_back(std::move(t));
    }
    const ImageTextResult r = image_replace_text(g_image, items.data(), items.size());
    host_log("plugin: text: %zu strings replaced (%zu instructions, %zu pointers); %zu named by nothing found, %zu refused",
             r.strings, r.leas, r.pointers, r.unreferenced, r.rejected + (count - items.size()));
    return static_cast<int>(r.leas + r.pointers);
}

int api_patch(std::uint64_t bn, const void* expect, const void* bytes, std::size_t n) {
    const std::uint64_t at = runtime_of(bn, n);
    if (!at || !api_eboot_is_109()) return 1;
    GuestMemory& mem = g_image->mem;
    auto* p = static_cast<std::uint8_t*>(guest_ptr(mem, at));
    if (std::memcmp(p, expect, n) != 0) return std::memcmp(p, bytes, n) == 0 ? 0 : 1;
    const std::uint64_t lo = at & ~0xfffull, hi = (at + n + 0xfff) & ~0xfffull;
    if (!guest_protect_rwx(&mem, lo, hi - lo)) return 2;
    std::memcpy(p, bytes, n);
    if (!guest_protect_rx(&mem, lo, hi - lo)) {
        host_log("plugin: cannot restore the protection of 0x%llx", static_cast<unsigned long long>(bn));
        std::abort();
    }
    return 0;
}

// Every plugin hook's stub calls this with its id; the saved registers are
// r9, r8, rcx, rdx, rsi, rdi.
GUEST_ABI std::int64_t hook_dispatch(std::uint64_t id, const std::uint64_t* saved) {
    if (id == 0 || id > g_hooks.size()) return 0;
    const Hook& h = g_hooks[id - 1];
    if (h.fn2) {
        // The stub's frame (core/thunk.h): saved[0..5] = r9 r8 rcx rdx rsi
        // rdi, saved[-1] the return slot, xmm0-7 16 bytes apart 136 bytes
        // below saved.
        auto* regs = const_cast<std::uint64_t*>(saved);
        auto* xmm = reinterpret_cast<std::uint8_t*>(regs) - 136;
        BbHookCtx c{};
        for (int i = 0; i < 6; ++i) c.arg[i] = regs[5 - i];
        for (int i = 0; i < 8; ++i) std::memcpy(&c.xmm[i], xmm + 16 * i, 8);
        const int skip = h.fn2(&c, h.user);
        for (int i = 0; i < 6; ++i) regs[5 - i] = c.arg[i];
        for (int i = 0; i < 8; ++i) std::memcpy(xmm + 16 * i, &c.xmm[i], 8);
        if (skip) regs[-1] = c.ret;
        return skip ? 1 : 0;
    }
    const BbHookFrame f{saved[5], saved[4], saved[3], saved[2], saved[1], saved[0]};
    return h.fn(&f, h.user) ? 1 : 0;
}

int api_hook(std::uint64_t bn, const void* prologue, std::size_t n, BbHookFn fn, void* user) {
    if (!fn || n < 14 || !api_eboot_is_109()) return 1;
    const std::uint64_t at = runtime_of(bn, n);
    if (!at) return 1;
    std::uint8_t have[64];
    if (n > sizeof(have)) return 1;
    std::memcpy(have, guest_ptr(g_image->mem, at), n);
    if (std::memcmp(have, prologue, n) != 0) return 1;
    g_hooks.push_back({fn, nullptr, user});
    const std::uint64_t id = g_hooks.size();
    if (!engine_prologue_hook(g_image, at, static_cast<const std::uint8_t*>(prologue), n,
                              reinterpret_cast<void*>(&hook_dispatch), id)) {
        g_hooks.pop_back();
        return 2;
    }
    return 0;
}

// Version 2: the engine's services.
struct FrameCallback {
    void (*fn)(void*);
    void* user;
};
std::mutex g_frame_mu;
std::vector<FrameCallback> g_frame_callbacks;  // under g_frame_mu

int api_on_frame(void (*fn)(void*), void* user) {
    if (!fn) return 1;
    std::lock_guard<std::mutex> lk(g_frame_mu);
    g_frame_callbacks.push_back({fn, user});
    return 0;
}

void* api_param_row(const char* table, std::uint32_t id, std::size_t* bytes) {
    if (!api_eboot_is_109() || !table) return nullptr;
    return params_row(table, id, bytes);
}

// "name" or "name[i]" of the table's definition, and the row it lives in.
const ParamField* param_field(const char* table, std::uint32_t id, const char* field, std::uint8_t** row, std::uint32_t* index) {
    if (!api_eboot_is_109() || !table || !field) return nullptr;
    const ParamDefinition* def = params_definition(table);
    if (!def) return nullptr;
    std::string name = field;
    *index = 0;
    if (const std::size_t br = name.find('['); br != std::string::npos) {
        *index = static_cast<std::uint32_t>(std::strtoul(name.c_str() + br + 1, nullptr, 10));
        name.resize(br);
    }
    const ParamField* f = def->field(name);
    *row = static_cast<std::uint8_t*>(params_row(table, id, nullptr));
    return f && *row ? f : nullptr;
}

int api_param_get(const char* table, std::uint32_t id, const char* field, char* out, std::size_t n) {
    std::uint8_t* row = nullptr;
    std::uint32_t index = 0;
    const ParamField* f = param_field(table, id, field, &row, &index);
    if (!f || !out || !n) return 1;
    const std::string v = paramdef_read(*f, row, index);
    if (v.empty()) return 1;
    std::snprintf(out, n, "%s", v.c_str());
    return 0;
}

int api_param_set(const char* table, std::uint32_t id, const char* field, const char* value) {
    std::uint8_t* row = nullptr;
    std::uint32_t index = 0;
    const ParamField* f = param_field(table, id, field, &row, &index);
    return f && value && paramdef_write(*f, row, value, index) ? 0 : 1;
}

int api_lua_event(const char* name) { return lua_event_queue(name) ? 0 : 1; }

int api_event_flag_get(std::uint32_t id, int* value) {
    bool v = false;
    if (!api_eboot_is_109() || !event_flag_get(id, &v)) return 1;
    if (value) *value = v ? 1 : 0;
    return 0;
}

int api_event_flag_set(std::uint32_t id, int value) { return api_eboot_is_109() && event_flag_set(id, value != 0) ? 0 : 1; }

int api_player_stat_get(const char* name, std::int64_t* value) { return api_eboot_is_109() && player_stat_get(name, value) ? 0 : 1; }
int api_player_stat_set(const char* name, std::int64_t value) { return api_eboot_is_109() && player_stat_set(name, value) ? 0 : 1; }

// Version 4: flag changes. The callbacks are only read on the main thread
// after registration (plugins register from bb_plugin_image, before the game
// runs), under the frame mutex all the same.
struct FlagCallback {
    void (*fn)(std::uint32_t, int, void*);
    void* user;
};
std::vector<FlagCallback> g_flag_callbacks;  // under g_frame_mu

int api_player_position(float xyz[3], float* yaw) { return api_eboot_is_109() && world_player_position(xyz, yaw) ? 0 : 1; }
int api_camera_get(float pos[3], float focus[3]) { return api_eboot_is_109() && world_camera(pos, focus) ? 0 : 1; }
int api_player_block(std::uint32_t* block) { return api_eboot_is_109() && world_player_block(block) ? 0 : 1; }
int api_lamp_warp(std::int32_t return_point) { return api_eboot_is_109() && world_lamp_warp_queue(return_point) ? 0 : 1; }
int api_player_warp(std::uint32_t block, const float xyz[3], float yaw) { return api_eboot_is_109() && world_player_warp_queue(block, xyz, yaw) ? 0 : 1; }
// Version 12: the loading screen, as the host's own test last saw it (engine/loading.h).
int api_loading_screen() { return loading_screen_up() ? 1 : 0; }

int api_on_event_flag(void (*fn)(std::uint32_t, int, void*), void* user) {
    if (!fn || !api_eboot_is_109()) return 1;
    {
        std::lock_guard<std::mutex> lk(g_frame_mu);
        g_flag_callbacks.push_back({fn, user});
    }
    event_flag_watch(true);
    return 0;
}

// Version 8.
std::uint64_t api_symbol(const char* name) {
    if (!api_eboot_is_109() || !name) return 0;
    const bb::NamedSymbol* s = bb::find_symbol(name);
    return s ? s->addr.bn() : 0;
}

int api_call_guest(std::uint64_t bn, BbCallRegs* regs) {
    const std::uint64_t at = api_eboot_is_109() ? runtime_of(bn, 1) : 0;
    if (!at || !regs) return 1;
    GuestCallRegs r{};
    r.fn = reinterpret_cast<void*>(static_cast<std::uintptr_t>(at));
    std::memcpy(r.ints, regs->arg, sizeof(r.ints));
    std::memcpy(r.xmm, regs->xmm, sizeof(r.xmm));
    std::memcpy(r.stack, regs->stack, sizeof(r.stack));
    hle_call_guest_regs(&r);
    regs->ret = r.ret;
    regs->ret_xmm0 = r.ret_xmm0;
    return 0;
}

int api_hook2(std::uint64_t bn, const void* prologue, std::size_t n, BbHookFn2 fn, void* user) {
    if (!fn || n < 14 || n > 64 || !api_eboot_is_109()) return 1;
    const std::uint64_t at = runtime_of(bn, n);
    if (!at) return 1;
    std::uint8_t have[64];
    std::memcpy(have, guest_ptr(g_image->mem, at), n);
    if (prologue && std::memcmp(have, prologue, n) != 0) return 1;
    g_hooks.push_back({nullptr, fn, user});
    const std::uint64_t id = g_hooks.size();
    if (!engine_prologue_hook(g_image, at, have, n, reinterpret_cast<void*>(&hook_dispatch), id)) {
        g_hooks.pop_back();
        return 2;
    }
    return 0;
}

int api_param_ids(const char* table, std::uint32_t* ids, std::size_t max, std::size_t* count) {
    return api_eboot_is_109() && table && params_ids(table, ids, max, count) ? 0 : 1;
}

struct WorldLoadCallback {
    void (*fn)(std::uint32_t, void*);
    void* user;
};
std::vector<WorldLoadCallback> g_world_callbacks;  // under g_frame_mu
std::uint64_t g_last_player = 0;
std::uint32_t g_last_block = 0;

int api_on_world_load(void (*fn)(std::uint32_t, void*), void* user) {
    if (!fn) return 1;
    std::lock_guard<std::mutex> lk(g_frame_mu);
    g_world_callbacks.push_back({fn, user});
    return 0;
}

// The world's characters, read through the layouts in include/bbhost/engine
// (sprj/world_chr_man.hpp, chr_ins.hpp, chr_module.hpp) by offset, with every
// pointer read checked: a set can change under a load.
template <typename T>
bool rd(std::uint64_t at, T* out) {
    return at && host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), out, sizeof(T));
}
std::uint64_t rp(std::uint64_t at) {
    std::uint64_t v = 0;
    return rd(at, &v) ? v : 0;
}
std::uint64_t world_chr_man() {
    return g_image && api_eboot_is_109() ? rp(runtime_of(bb::WORLD_CHR_MAN_SINGLETON_PTR.bn(), 8)) : 0;
}

bool fill_chr(std::uint64_t ins, std::uint64_t player_vtable, BbChr* c) {
    std::uint64_t vt = 0;
    if (!rd(ins, &vt) || !vt) return false;
    *c = BbChr{};
    c->ins = reinterpret_cast<void*>(static_cast<std::uintptr_t>(ins));
    rd(ins + 0x08, &c->handle);
    rd(ins + 0x78, &c->chr_type);
    rd(ins + 0x88, &c->team_type);
    const bool player = vt == player_vtable;
    c->npc_param = -1;
    if (!player) {
        // NpcIns+0x3c8: the NpcParam reference its constructor fills (an id,
        // then the row at +8; 0x1ce4870).
        const std::uint64_t ref = rp(ins + 0x3c8);
        std::int32_t id = -1;
        if (ref && rd(ref, &id)) c->npc_param = id;
    }
    const std::uint64_t modules = rp(ins + 0x3b0);
    const std::uint64_t data = modules ? rp(modules + 0x20) : 0;
    const std::uint64_t phys = modules ? rp(modules + 0x68) : 0;
    if (data) {
        rd(data + 0xf8, &c->hp);
        rd(data + 0xfc, &c->max_hp);
        // The character id ("c2560"): a small-string wide string at +0x158.
        std::uint16_t inl[8] = {};
        std::uint64_t len = 0, cap = 0;
        if (rd(data + 0x158, &inl) && rd(data + 0x168, &len) && rd(data + 0x170, &cap) && len && len <= 16) {
            std::uint16_t w[16] = {};
            bool ok = true;
            if (cap <= 7) std::memcpy(w, inl, sizeof(inl));
            else ok = host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(rp(data + 0x158))), w, len * 2);
            if (ok && w[0] == 'c') {
                for (std::uint64_t i = 1; i < len && w[i] >= '0' && w[i] <= '9'; ++i) c->model = c->model * 10 + (w[i] - '0');
            }
        }
    }
    if (phys) rd(phys + 0x1e0, &c->pos);
    return true;
}

int api_chr_list(BbChr* out, std::size_t max, std::size_t* count) {
    const std::uint64_t man = world_chr_man();
    const std::uint64_t main = man ? rp(man + 0x60) : 0;
    if (!main) return 1;
    const std::uint64_t player_vt = rp(main);
    std::vector<std::uint64_t> seen{main};
    std::size_t n = 0;
    const auto add = [&](std::uint64_t ins, bool is_main) {
        BbChr c;
        if (!fill_chr(ins, player_vt, &c)) return;
        c.is_player = is_main ? 1 : 0;
        if (out && n < max) out[n] = c;
        ++n;
    };
    add(main, true);
    // The handle-indexed sets (WorldChrMan+0x850, 0x3e ChrSet pointers):
    // every set a character can be looked up through, players' included.
    for (std::size_t i = 0; i < 0x3e; ++i) {
        const std::uint64_t set = rp(man + 0x850 + i * 8);
        std::uint32_t cap = 0;
        const std::uint64_t entries = set ? rp(set + 0x08) : 0;
        if (!entries || !rd(set, &cap) || cap > 4096) continue;
        for (std::uint32_t k = 0; k < cap; ++k) {
            const std::uint64_t ins = rp(entries + k * 0x38);
            if (!ins) continue;
            bool dup = false;
            for (std::uint64_t s2 : seen) dup = dup || s2 == ins;
            if (dup) continue;
            seen.push_back(ins);
            add(ins, false);
        }
    }
    if (count) *count = n;
    return 0;
}

std::uint64_t chr_or_player(void* chr) {
    if (chr) return reinterpret_cast<std::uint64_t>(chr);
    const std::uint64_t man = world_chr_man();
    return man ? rp(man + 0x60) : 0;
}

int api_sp_effect_apply(void* chr, std::int32_t id) {
    const std::uint64_t ins = chr_or_player(chr);
    const std::uint64_t vt = ins ? rp(ins) : 0;
    const std::uint64_t fn = vt ? rp(vt + 0x3f0) : 0;
    if (!fn) return 1;
    // As Lua_MultiDoping (0x178bc70) calls it: target, effect, source = the
    // target, then zeros, five 1.0 multipliers in xmm0-4, a zero on the stack.
    GuestCallRegs r{};
    r.fn = reinterpret_cast<void*>(static_cast<std::uintptr_t>(fn));
    r.ints[0] = ins;
    r.ints[1] = static_cast<std::uint32_t>(id);
    r.ints[2] = ins;
    for (int i = 0; i < 5; ++i) r.xmm[i] = 0x3f800000u;
    hle_call_guest_regs(&r);
    return 0;
}

int api_sp_effect_has(void* chr, std::int32_t id) {
    const std::uint64_t ins = chr_or_player(chr);
    const std::uint64_t list = ins ? rp(ins + 0x1c8) : 0;  // CHR_INS_SP_EFFECT_OFFSET
    std::uint64_t node = list ? rp(list + 0x08) : 0;       // SPEFFECT_LIST_HEAD_OFFSET
    for (int guard = 0; node && guard < 1024; ++guard) {
        std::int32_t have = 0;
        if (rd(node + 0x40, &have) && have == id) return 1;  // SPEFFECT_NODE_ID_OFFSET
        node = rp(node + 0x58);                              // SPEFFECT_NODE_NEXT_OFFSET
    }
    return 0;
}

// BBHOST_SPEFFECT_TEST=<id>[,<seconds>]: how a SpEffect ticks at the frame
// rate in use. <seconds> (20) into the world the effect is put on the player,
// again each half second until its node is on the list (a status effect's
// buildup fills first); then each change of the player's HP and each tick
// (the node's timer reloaded) is logged with the time and frames since, and
// the totals as the node goes. Once a session.
void speffect_test_tick() {
    static const struct Test {
        std::int32_t id = 0;
        double delay = 20.0;
    } test = [] {
        Test t;
        const char* e = std::getenv("BBHOST_SPEFFECT_TEST");
        if (!e || !e[0]) return t;
        t.id = std::atoi(e);
        if (const char* comma = std::strchr(e, ',')) t.delay = std::atof(comma + 1);
        return t;
    }();
    if (!test.id) return;
    using Clock = std::chrono::steady_clock;
    static int stage = 0;  // 0 waiting, 1 applying, 2 on, 3 done
    static Clock::time_point world_since{}, last_apply{}, on_since{};
    static std::uint64_t on_frames = 0;
    static int attempts = 0, hp_changes = 0, ticks = 0, hp_at_on = 0, last_hp = 0;
    static float last_timer = 0.0f;
    if (stage == 3) return;
    const std::uint64_t man = world_chr_man();
    const std::uint64_t player = man ? rp(man + 0x60) : 0;
    if (!player || loading_screen_up()) {
        if (stage == 0) world_since = Clock::time_point{};
        return;
    }
    const auto now = Clock::now();
    if (world_since == Clock::time_point{}) world_since = now;
    const auto secs = [&](Clock::time_point a) { return std::chrono::duration<double>(now - a).count(); };
    if (stage == 0 && secs(world_since) < test.delay) return;
    const std::uint64_t list = rp(player + 0x1c8);  // CHR_INS_SP_EFFECT_OFFSET
    std::uint64_t node = list ? rp(list + 0x08) : 0;
    for (int guard = 0; node && guard < 1024; ++guard) {
        std::int32_t have = 0;
        if (rd(node + 0x40, &have) && have == test.id) break;
        node = rp(node + 0x58);
    }
    const std::uint64_t modules = rp(player + 0x3b0);
    const std::uint64_t data = modules ? rp(modules + 0x20) : 0;
    std::int32_t hp = 0, max_hp = 0;
    if (data) {
        rd(data + 0xf8, &hp);
        rd(data + 0xfc, &max_hp);
    }
    float remaining = 0.0f, timer = 0.0f;
    std::uint32_t flags = 0;
    if (node) {
        rd(node + 0x00, &remaining);
        rd(node + 0x08, &timer);
        rd(node + 0x1c, &flags);
    }
    if (stage < 2) {
        if (node) {
            stage = 2;
            on_since = now;
            on_frames = 0;
            hp_at_on = last_hp = hp;
            last_timer = timer;
            const std::uint64_t row = rp(node + 0x48);
            float interval = 0.0f;
            if (row) rd(row + 0x0c, &interval);
            host_log("speffect test: %d on after %d application(s): remaining %.3f s, timer %.3f s, interval %.3f s, "
                     "flags 0x%x, HP %d/%d", test.id, attempts, static_cast<double>(remaining), static_cast<double>(timer),
                     static_cast<double>(interval), flags, hp, max_hp);
        } else if (attempts < 60 && (stage == 0 || secs(last_apply) >= 0.5)) {
            stage = 1;
            last_apply = now;
            ++attempts;
            api_sp_effect_apply(nullptr, test.id);
        }
        return;
    }
    ++on_frames;
    if (!node) {
        const double t = secs(on_since);
        host_log("speffect test: %d off after %.3f s, %llu frames (%.1f fps): %d ticks (%.2f a second), %d HP changes, "
                 "HP %d -> %d", test.id, t, static_cast<unsigned long long>(on_frames),
                 t > 0 ? static_cast<double>(on_frames) / t : 0.0, ticks, t > 0 ? ticks / t : 0.0, hp_changes, hp_at_on, hp);
        stage = 3;
        return;
    }
    const bool tick = timer > last_timer + 1e-4f;  // reloaded from the interval
    last_timer = timer;
    if (tick) ++ticks;
    if (hp != last_hp || tick) {
        if (hp != last_hp) ++hp_changes;
        host_log("speffect test: %d +%.3f s frame %llu: %s HP %d -> %d (%+d), remaining %.3f, timer %.3f, flags 0x%x",
                 test.id, secs(on_since), static_cast<unsigned long long>(on_frames), tick ? "tick" : "    ", last_hp, hp,
                 hp - last_hp, static_cast<double>(remaining), static_cast<double>(timer), flags);
        last_hp = hp;
    }
}

// BBHOST_CHR_WATCH=<model>[,god][,<SpEffect id>...]: a fight left to run by
// itself, measured. The first character of that model (2510: Ebrietas) is
// found in the world; each SpEffect listed is logged as it goes on and off
// it, and once a second its HP, the SpEffects on it and the bullets: those
// alive in the bullet manager's pool (the 0x80 entries off its free list)
// and those made in that second. "god" keeps the player's HP full; "near=<m>"
// puts the player that far from it, facing it, 3 s after it is found (the
// game's own warp: BBHOST_TEST_WARP=1 or an opt-in gameplay plugin on).
void chr_watch_tick() {
    static const struct Watch {
        std::uint32_t model = 0;
        bool god = false;
        float approach = 0.0f;
        std::vector<std::int32_t> ids;
    } watch = [] {
        Watch w;
        const char* e = std::getenv("BBHOST_CHR_WATCH");
        if (!e || !e[0]) return w;
        std::string s(e);
        std::size_t at = 0;
        for (int k = 0; at <= s.size(); ++k) {
            const std::size_t comma = std::min(s.find(',', at), s.size());
            const std::string part = s.substr(at, comma - at);
            if (k == 0) w.model = static_cast<std::uint32_t>(std::atoi(part.c_str()));
            else if (part == "god") w.god = true;
            else if (part.rfind("near=", 0) == 0) w.approach = static_cast<float>(std::atof(part.c_str() + 5));
            else if (!part.empty()) w.ids.push_back(std::atoi(part.c_str()));
            at = comma + 1;
        }
        return w;
    }();
    if (!watch.model) return;
    using Clock = std::chrono::steady_clock;
    static Clock::time_point since{}, last_report{};
    static std::uint64_t ins = 0, frames = 0;
    static std::vector<std::int32_t> on;  // the listed ids on it last frame
    static std::vector<std::uint64_t> live;  // bullet entries alive last frame
    static int made = 0;
    static std::int64_t stamina_min = INT64_MAX;
    const std::uint64_t man = world_chr_man();
    const std::uint64_t player = man ? rp(man + 0x60) : 0;
    if (!player || loading_screen_up()) {
        ins = 0;
        return;
    }
    if (watch.god) {
        const std::uint64_t modules = rp(player + 0x3b0);
        const std::uint64_t data = modules ? rp(modules + 0x20) : 0;
        std::int32_t hp = 0, max_hp = 0;
        if (data && rd(data + 0xf8, &hp) && rd(data + 0xfc, &max_hp) && max_hp > 0 && hp > 0 && hp < max_hp) {
            *reinterpret_cast<std::int32_t*>(static_cast<std::uintptr_t>(data + 0xf8)) = max_hp;
        }
    }
    const auto now = Clock::now();
    ++frames;
    if (!ins && frames % 30 == 0) {
        static BbChr chrs[512];
        std::size_t n = 0;
        api_chr_list(chrs, 512, &n);
        for (std::size_t i = 0; i < std::min<std::size_t>(n, 512); ++i) {
            if (chrs[i].model != watch.model) continue;
            ins = reinterpret_cast<std::uint64_t>(chrs[i].ins);
            since = last_report = now;
            host_log("chr watch: c%04u found (npc %d, hp %d/%d) at %.1f %.1f %.1f", watch.model, chrs[i].npc_param,
                     chrs[i].hp, chrs[i].max_hp, static_cast<double>(chrs[i].pos[0]), static_cast<double>(chrs[i].pos[1]),
                     static_cast<double>(chrs[i].pos[2]));
            break;
        }
    }
    if (!ins) return;
    const double t = std::chrono::duration<double>(now - since).count();
    if (std::int64_t st = 0; player_stat_get("stamina", &st)) stamina_min = std::min(stamina_min, st);
    static bool warped = false;
    if (watch.approach > 0.0f && !warped && t >= 3.0) {
        warped = true;
        const std::uint64_t modules = rp(ins + 0x3b0);
        const std::uint64_t phys = modules ? rp(modules + 0x68) : 0;
        float at[3] = {}, me[3] = {}, yaw = 0.0f;
        if (phys && rd(phys + 0x1e0, &at) && world_player_position(me, &yaw)) {
            const float dx = me[0] - at[0], dz = me[2] - at[2];
            const float len = std::sqrt(dx * dx + dz * dz);
            if (len > 0.01f) {
                const float to[3] = {at[0] + dx / len * watch.approach, at[1] + 0.2f, at[2] + dz / len * watch.approach};
                const float face = std::atan2(dx / len, dz / len);  // a thing at yaw y faces (-sin y, -cos y)
                const bool ok = world_player_warp_queue(0, to, face);
                host_log("chr watch: player to %.1f %.1f %.1f (%.1f m from it), %s", static_cast<double>(to[0]),
                         static_cast<double>(to[1]), static_cast<double>(to[2]), static_cast<double>(watch.approach),
                         ok ? "queued" : "warp is off");
            }
        }
    }
    // Its SpEffects this frame.
    std::vector<std::int32_t> ids;
    const std::uint64_t list = rp(ins + 0x1c8);
    std::uint64_t node = list ? rp(list + 0x08) : 0;
    for (int guard = 0; node && guard < 256; ++guard) {
        std::int32_t id = 0;
        if (rd(node + 0x40, &id)) ids.push_back(id);
        node = rp(node + 0x58);
    }
    std::vector<std::int32_t> now_on;
    for (std::int32_t id : watch.ids) {
        if (std::find(ids.begin(), ids.end(), id) != ids.end()) now_on.push_back(id);
    }
    for (std::int32_t id : now_on) {
        if (std::find(on.begin(), on.end(), id) == on.end()) host_log("chr watch: +%.3f s: %d on", t, id);
    }
    for (std::int32_t id : on) {
        if (std::find(now_on.begin(), now_on.end(), id) == now_on.end()) host_log("chr watch: +%.3f s: %d off", t, id);
    }
    on.swap(now_on);
    // The bullets: the pool's entries not on its free list.
    const std::uint64_t bm = g_image && api_eboot_is_109() ? rp(runtime_of(bb::SPRJ_BULLET_MANAGER_SINGLETON_PTR.bn(), 8)) : 0;
    const std::uint64_t pool = bm ? rp(bm + 0x00) : 0;
    std::vector<std::uint64_t> free_list;
    for (std::uint64_t e = bm ? rp(bm + 0x10) : 0; e && free_list.size() < 0x80; e = rp(e + 0x868)) free_list.push_back(e);
    std::vector<std::uint64_t> alive;
    if (pool) {
        for (std::uint64_t k = 0; k < 0x80; ++k) {
            const std::uint64_t e = pool + k * 0x890;
            if (std::find(free_list.begin(), free_list.end(), e) == free_list.end()) alive.push_back(e);
        }
    }
    for (std::uint64_t e : alive) {
        if (std::find(live.begin(), live.end(), e) == live.end()) ++made;
    }
    live.swap(alive);
    if (std::chrono::duration<double>(now - last_report).count() >= 1.0) {
        last_report = now;
        const std::uint64_t modules = rp(ins + 0x3b0);
        const std::uint64_t data = modules ? rp(modules + 0x20) : 0;
        std::int32_t hp = 0;
        if (data) rd(data + 0xf8, &hp);
        std::sort(ids.begin(), ids.end());
        std::string s;
        for (std::int32_t id : ids) s += " " + std::to_string(id);
        float me[3] = {};
        world_player_position(me, nullptr);
        host_log("chr watch: +%.1f s: hp %d, bullets %zu alive, %d made; SpEffects%s; player at %.1f %.1f %.1f, stamina down to %lld",
                 t, hp, live.size(), made, s.c_str(), static_cast<double>(me[0]), static_cast<double>(me[1]),
                 static_cast<double>(me[2]), static_cast<long long>(stamina_min == INT64_MAX ? -1 : stamina_min));
        made = 0;
        stamina_min = INT64_MAX;
    }
}

std::string g_plugin_dir;
const char* api_plugin_dir(const char* name) {
    const char* data = hle_fs_data_root();
    if (!name || !name[0] || !data || !data[0] || std::strpbrk(name, "/\\.:")) return "";
    g_plugin_dir = std::string(data) + "/plugins/" + name;
    std::error_code ec;
    std::filesystem::create_directories(g_plugin_dir, ec);
    return g_plugin_dir.c_str();
}

int api_show_message(const char* text, float seconds) {
    if (!text) return 1;
    host_log("plugin: on screen: %s", text);
    host_message_show(text, seconds);
    return 0;
}

int api_show_text(const char* text, float seconds, float x, float y, float scale, std::uint32_t align) {
    if (!text) return 1;
    host_text_show(text, seconds, x, y, scale, align);
    return 0;
}

int api_set_time_scale(float scale) {
    frame_rate_set_time_scale(scale);
    return 0;
}

// Version 9: the plugin menus' settings and actions, and plugin overlays.
struct OptionCallback {
    std::string name;
    void (*fn)(const char*, const char*, void*);
    void* user;
};
struct ActionCallback {
    std::string name;
    void (*fn)(const char*, void*);
    void* user;
};
std::vector<OptionCallback> g_option_callbacks;  // under g_frame_mu
std::vector<ActionCallback> g_action_callbacks;  // under g_frame_mu
// What the menus asked for, run on the main thread at the next frame.
struct MenuRequest {
    bool action;
    std::string name, key, value;
};
std::vector<MenuRequest> g_menu_requests;  // under g_frame_mu

int api_on_option(const char* name, void (*fn)(const char*, const char*, void*), void* user) {
    if (!name || !fn) return 1;
    std::lock_guard<std::mutex> lk(g_frame_mu);
    g_option_callbacks.push_back({name, fn, user});
    return 0;
}

int api_on_action(const char* name, void (*fn)(const char*, void*), void* user) {
    if (!name || !fn) return 1;
    std::lock_guard<std::mutex> lk(g_frame_mu);
    g_action_callbacks.push_back({name, fn, user});
    return 0;
}

// Per thread: a world change calls plugins on the network and game threads
// while another plugin may read on the main one. Good until that thread's next call.
thread_local std::vector<std::uint8_t> g_game_file;
int api_game_file(const char* path, const void** data, std::size_t* size) {
    if (!path || path[0] != '/' || !data || !size) return 1;
    const std::string host = hle_fs_map_path_base(("/app0" + std::string(path)).c_str());
    std::vector<std::uint8_t> raw;
    if (host.empty() || !gcn::read_file(host, raw)) return 1;
    std::string err;
    g_game_file = gcn::dcx_decompress(raw, &err);
    if (g_game_file.empty() && !raw.empty()) return 1;
    *data = g_game_file.data();
    *size = g_game_file.size();
    return 0;
}

// A map layout a plugin rewrote is read again at that map's next load, the
// map the player stands in too. The game asks for a layout through
// 0x18235e0 (SprjFile, path, key): a file holder already in its table under
// the key - the world being left, while the next one loads - is taken as it
// is, and a co-op summons or a warp within one map never reads the file.
// Each rewrite of a layout gives it a new generation, and the key a request
// carries becomes "<key>#<file>.<generation>": a holder made before the
// rewrite no longer matches, and the file is read.
std::mutex g_layout_gen_mu;
std::map<std::string, unsigned> g_layout_gen;        // layout file name (m27_00_00_01) -> rewrites, under g_layout_gen_mu
std::map<std::string, std::uint64_t> g_layout_hash;  // overlay path -> its content's hash, under g_layout_gen_mu

// "/dvdroot_ps4/map/mapstudio/m27_00_00_01.msb.dcx" -> "m27_00_00_01"; "" for any other file.
std::string layout_name(std::string path) {
    for (char& c : path) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    static const std::string dir = "/map/mapstudio/", ext = ".msb.dcx";
    const std::size_t at = path.rfind(dir);
    if (at == std::string::npos || path.size() < ext.size() || path.compare(path.size() - ext.size(), ext.size(), ext) != 0) return {};
    return path.substr(at + dir.size(), path.size() - ext.size() - at - dir.size());
}

void layout_written(const std::string& path, const std::vector<std::uint8_t>& raw) {
    const std::string name = layout_name(path);
    if (name.empty()) return;
    std::uint64_t h = 1469598103934665603ull;
    for (const std::uint8_t b : raw) h = (h ^ b) * 1099511628211ull;
    std::lock_guard<std::mutex> lk(g_layout_gen_mu);
    auto [it, fresh] = g_layout_hash.emplace(path, h);
    if (!fresh && it->second == h) return;  // the same layout again: what the game holds is right
    it->second = h;
    ++g_layout_gen[name];
}

thread_local char16_t t_layout_key[160];
int layout_request_hook(BbHookCtx* c, void*) {
    const auto* path = reinterpret_cast<const char16_t*>(c->arg[1]);
    const auto* key = reinterpret_cast<const char16_t*>(c->arg[2] ? c->arg[2] : c->arg[1]);
    if (!path || !key) return 0;
    // "mapstudio:/m27_00_00_01.msb": the file's name after the last '/'.
    std::string file;
    for (const char16_t* p = path; *p && file.size() < 64; ++p) {
        if (*p == u'/' || *p == u'\\' || *p == u':') file.clear();
        else file += static_cast<char>(*p < 0x80 ? std::tolower(static_cast<int>(*p)) : '?');
    }
    if (file.size() > 4 && file.compare(file.size() - 4, 4, ".msb") == 0) file.resize(file.size() - 4);
    unsigned gen = 0;
    {
        std::lock_guard<std::mutex> lk(g_layout_gen_mu);
        if (auto it = g_layout_gen.find(file); it != g_layout_gen.end()) gen = it->second;
    }
    if (!gen) return 0;
    const std::string suffix = "#" + file + "." + std::to_string(gen);
    std::size_t n = 0;
    for (; key[n] && n + suffix.size() + 1 < std::size(t_layout_key); ++n) t_layout_key[n] = key[n];
    if (key[n]) return 0;  // a key this long is not a layout's
    for (const char ch : suffix) t_layout_key[n++] = static_cast<char16_t>(ch);
    t_layout_key[n] = 0;
    c->arg[2] = reinterpret_cast<std::uint64_t>(t_layout_key);
    return 0;
}

// A DL wide string in a game object: the characters inline at `at` up to 7,
// else a pointer there; the capacity 0x18 on.
std::string guest_wstring(std::uint64_t at, std::size_t max = 128) {
    std::uint64_t cap = 0, chars = at;
    if (!rd(at + 0x18, &cap)) return {};
    if (cap >= 8) chars = rp(at);
    std::string out;
    for (std::uint16_t ch = 0; chars && out.size() < max && rd(chars + 2 * out.size(), &ch) && ch;)
        out += static_cast<char>(ch < 0x80 ? std::tolower(ch) : '?');
    return out;
}

// One hold of the file is not enough: the parsed layout (MsbResCap) sits in
// MsbRepository under the map's name ("m24_01_00_00", the key's text after its
// last '/' up to the first '.'), held by whatever uses it. A summons into the
// map the player stands in keeps that map resident and rebuilds its
// characters from the parse already there - even after the file was read
// again, MsbFileCap's load (0x18094e0) finds the name taken and takes the
// old parse. So when a layout read under a new key ('#', above) finishes,
// the old parse leaves the repository's table first; its holders keep it,
// and its last release frees it (the release unlinks only what it finds).
int layout_loaded_hook(BbHookCtx* c, void*) {
    const std::uint64_t cap = c->arg[0];
    const std::string key = cap ? guest_wstring(cap + 0x18) : std::string();
    if (key.find('#') == std::string::npos) return 0;
    std::size_t from = key.find_last_of("/\\");
    from = from == std::string::npos ? 0 : from + 1;
    const std::string name = key.substr(from, key.find('.', from) - from);
    const std::uint64_t repo = rp(runtime_of(0x59402f0, 8));
    std::uint32_t buckets = 0;
    const std::uint64_t table = repo ? repo + 0x68 : 0;
    const std::uint64_t heads = table ? rp(table + 0x20) : 0;
    if (!heads || !rd(table + 0x1c, &buckets) || buckets > 1u << 20) return 0;
    for (std::uint32_t b = 0; b < buckets; ++b) {
        std::uint64_t prev = 0;
        for (std::uint64_t node = rp(heads + 8ull * b), hops = 0; node && hops < 4096; ++hops) {
            const std::uint64_t next = rp(node + 0x58);
            if (guest_wstring(node + 0x18) == name) {
                *reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(prev ? prev + 0x58 : heads + 8ull * b)) = next;
                *reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(node + 0x58)) = 0;
                *reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(node + 0x50)) = 0;
                host_log("plugin: layout %s rewritten: the next parse replaces the one held", name.c_str());
                return 0;
            }
            prev = node;
            node = next;
        }
    }
    return 0;
}

// <data>/bbhost/plugin-overlays/<name>: cleared at every start (plugins_load),
// made and mounted on a plugin's first write.
std::string overlay_root() {
    const char* data = hle_fs_data_root();
    return data && data[0] ? std::string(data) + "/bbhost/plugin-overlays" : std::string();
}

int api_overlay_file(const char* name, const char* path, const void* data, std::size_t size) {
    const std::string root = overlay_root();
    if (root.empty() || !name || !name[0] || std::strpbrk(name, "/\\.:") || !path || path[0] != '/' ||
        std::strstr(path, "..") || (!data && size))
        return 1;
    const std::string dir = root + "/" + name;
    const std::filesystem::path file = std::filesystem::path(dir + path);
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    std::vector<std::uint8_t> bytes(static_cast<const std::uint8_t*>(data), static_cast<const std::uint8_t*>(data) + size);
    const std::string p = path;
    const std::vector<std::uint8_t> raw = bytes;
    // Level 1: written at run time too (a re-seed between maps); the game
    // reads any level, and 22 layouts take ~0.3 s instead of ~3.
    if (p.size() > 4 && p.compare(p.size() - 4, 4, ".dcx") == 0) bytes = gcn::dcx_compress(bytes, 1);
    if (bytes.empty() && size) return 1;
    // Written beside it and renamed over it: a map load may read the file while
    // it is rewritten (a co-op join, a re-seed), and must see one whole layout.
    std::filesystem::path tmp = file;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out || !out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
            out.close();
            std::filesystem::remove(tmp, ec);
            return 1;
        }
    }
    std::filesystem::rename(tmp, file, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return 1;
    }
    hle_fs_add_plugin_overlay(dir.c_str());
    layout_written(p, raw);
    return 0;
}

// Version 10.
// Play other players' modded worlds by their rules when summoned -
// the visitor, X-BBHost-Adopt. On unless plugins.join_modded_worlds = false
// (the account setting on the site turns it off server side too).
bool join_modded_worlds() { return config_value("plugins.join_modded_worlds") != "false"; }

std::mutex g_rules_mu;
std::map<std::string, std::string> g_rules;  // a plugin's own entry (set_rules), under g_rules_mu
std::string g_visiting;                       // the host's rules while a guest plays by them, under g_rules_mu
struct WorldRulesCallback {
    std::string name;
    void (*fn)(const char*, void*);
    void* user;
};
std::vector<WorldRulesCallback> g_world_rules_callbacks;  // under g_rules_mu

int api_visitor(const char* name) {
    for (const Plugin& p : g_plugins) {
        if (name && p.name == name) return p.visitor ? 1 : 0;
    }
    return 0;
}

int api_set_rules(const char* name, const char* rules) {
    if (!name || !rules) return 1;
    for (const Plugin& p : g_plugins) {
        if (p.name == name && p.visitor) return 1;  // a visitor plays no rules of its own
    }
    std::lock_guard<std::mutex> lk(g_rules_mu);
    g_rules[name] = rules;
    return 0;
}

int api_on_world_rules(const char* name, void (*fn)(const char*, void*), void* user) {
    if (!name || !fn) return 1;
    std::lock_guard<std::mutex> lk(g_rules_mu);
    g_world_rules_callbacks.push_back({name, fn, user});
    return 0;
}

// Version 13: controls and a view of a plugin's own.
int api_input(BbInput* out) {
    if (!out) return 1;
    PluginInput in;
    const bool ok = host_plugin_input(in);
    std::memcpy(out->key, in.key, sizeof(out->key));
    out->buttons = in.buttons;
    out->pressed = in.pressed;
    out->dx = in.dx;
    out->dy = in.dy;
    out->wheel = in.wheel;
    out->playing = in.playing ? 1 : 0;
    return ok ? 0 : 1;
}

int api_input_take(const std::uint16_t* scancodes, std::size_t n, std::uint32_t buttons, int wheel, int motion) {
    if (n && !scancodes) return 1;
    host_plugin_input_take(scancodes, n, buttons, wheel != 0, motion != 0);
    return 0;
}

// The camera. The game state's update (sub_1d3ac10) runs the camera
// manager's blend (sub_18368b0: the follow camera's update and the blend of
// the cameras the manager owns into its own, at +0x10), then copies the
// manager's camera into the scene's - which a scripted camera then replaces
// while one plays. Between the two, at 0x1d3be5a, the manager is still in rdi
// (sub_18377d0, called in between, is a leaf that leaves it there): a hook
// there hands the plugins its camera, which BbCamera lays out. The other game
// state's update (sub_1d39bb0) runs the blend at 0x1d39c2d, and its hook finds
// the manager from the camera root.
struct CameraCallback {
    void (*fn)(BbCamera*, void*);
    void* user;
};
std::vector<CameraCallback> g_camera_callbacks;  // under g_frame_mu
bool g_camera_hooked = false;

void camera_callbacks(std::uint64_t manager) {
    if (!manager) return;
    std::vector<CameraCallback> now;
    {
        std::lock_guard<std::mutex> lk(g_frame_mu);
        now = g_camera_callbacks;
    }
    auto* cam = reinterpret_cast<BbCamera*>(static_cast<std::uintptr_t>(manager + 0x10));
    for (const CameraCallback& c : now) c.fn(cam, c.user);
}

GUEST_ABI std::int64_t camera_hook(std::uint64_t, const std::uint64_t* saved) {
    camera_callbacks(saved[5]);  // rdi
    return 0;
}

GUEST_ABI std::int64_t camera_hook_other(std::uint64_t, const std::uint64_t*) {
    camera_callbacks(world_camera_owner());
    return 0;
}

int api_on_camera(void (*fn)(BbCamera*, void*), void* user) {
    if (!fn || !g_image || !api_eboot_is_109()) return 1;
    {
        std::lock_guard<std::mutex> lk(g_frame_mu);
        g_camera_callbacks.push_back({fn, user});
    }
    if (g_camera_hooked) return 0;
    g_camera_hooked = true;
    // mov rax, [r13+0x128]; vmovss xmm0, [rax+0x50]; vmovss [r15+0x50], xmm0 - the copy's start
    static const std::uint8_t copy[18] = {0x49, 0x8b, 0x85, 0x28, 0x01, 0x00, 0x00, 0xc5, 0xfa,
                                          0x10, 0x40, 0x50, 0xc4, 0xc1, 0x7a, 0x11, 0x47, 0x50};
    // mov rax, [r14+0x118]; mov rdi, [rax+0x68]; mov esi, [rax+0x20] - after the other update's blend
    static const std::uint8_t other[14] = {0x49, 0x8b, 0x86, 0x18, 0x01, 0x00, 0x00, 0x48, 0x8b, 0x78, 0x68, 0x8b, 0x70, 0x20};
    if (!engine_site_hook(g_image, runtime_of(0x1d3be5a, sizeof copy), copy, sizeof copy,
                          reinterpret_cast<void*>(&camera_hook))) {
        host_log("plugin: 0x1d3be5a is not the scene camera's copy; on_camera is not called");
    }
    if (!engine_site_hook(g_image, runtime_of(0x1d39c32, sizeof other), other, sizeof other,
                          reinterpret_cast<void*>(&camera_hook_other))) {
        host_log("plugin: 0x1d39c32 is not the other game state's blend return; on_camera is not called there");
    }
    return 0;
}

int api_key_bound(std::uint16_t scancode) {
    if (scancode == 0) return 0;
    for (int a = 0; a < kBindCount; ++a) {
        if (host_binding_key(a) == scancode) return 1;
    }
    return 0;
}

void api_draw_style(int style) { gx_draw_style(style & (kGxDrawLines | kGxDrawBlack | kGxDrawSubtract)); }

int api_on_scene_category(std::int32_t pass, void (*fn)(std::uint64_t, void*), void* user) {
    if (!g_image || !api_eboot_is_109()) return 1;
    render_queue_scene_insert(pass, fn, user);
    return 0;
}

const BbHostApi g_api = {
    BB_PLUGIN_API_VERSION, sizeof(BbHostApi), &api_log, &api_config, &api_data_dir, &api_mods_dir, &api_register_hle,
    &api_eboot_sha256, &api_eboot_is_109, &api_guest_addr, &api_read, &api_write, &api_patch, &api_hook,
    &api_on_frame, &api_param_row, &api_param_get, &api_param_set, &api_lua_event, &api_event_flag_get, &api_event_flag_set,
    &api_player_stat_get, &api_player_stat_set, &api_on_event_flag, &api_player_position, &api_camera_get,
    &api_player_block, &api_player_warp,
    &api_lamp_warp,
    &api_symbol, &api_call_guest, &api_hook2, &api_param_ids, &api_on_world_load, &api_chr_list,
    &api_sp_effect_apply, &api_sp_effect_has, &api_plugin_dir, &api_show_message, &api_set_time_scale,
    &api_on_option, &api_on_action, &api_game_file, &api_overlay_file,
    &api_set_rules, &api_on_world_rules, &api_visitor,
    &api_replace_text,
    &api_loading_screen,
    &api_input, &api_input_take, &api_on_camera, &api_key_bound,
    &api_show_text,
    &api_on_scene_category,
    &api_draw_style,
};

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f ? std::string(std::istreambuf_iterator<char>(f), {}) : std::string();
}

// `plugins.<name>` in the config: "true"/"1" enables, "false"/"0" disables,
// "" leaves the plugin's own default (an opt-in plugin is off).
int config_switch(const std::string& name) {
    const std::string v = config_value(("plugins." + name).c_str());
    if (v == "true" || v == "1") return 1;
    if (v == "false" || v == "0") return 0;
    return -1;
}

// A setting as the config file holds it: true/false and numbers bare,
// anything else a quoted string.
std::string toml_value(const std::string& v) {
    if (v == "true" || v == "false") return v;
    char* end = nullptr;
    if (!v.empty()) {
        std::strtod(v.c_str(), &end);
        if (end && *end == '\0') return v;
    }
    std::string q = "\"";
    for (char c : v) {
        if (c == '"' || c == '\\') q += '\\';
        q += c;
    }
    return q + "\"";
}

PluginEntry entry_of(const std::string& file_name, const std::string& path, const BbPluginInfo* info, const BbOption* options) {
    PluginEntry e;
    e.file = file_name;
    e.path = path;
    e.name = file_name.substr(0, file_name.rfind('.'));
    if (info) {
        if (info->name && info->name[0]) e.name = info->name;
        e.flags = info->flags;
        e.title = info->title ? info->title : "";
        e.version = info->version ? info->version : "";
        e.author = info->author ? info->author : "";
        e.description = info->description ? info->description : "";
    }
    if (e.title.empty()) e.title = e.name;
    for (const BbOption* o = options; o && o->type != BB_OPT_END; ++o) {
        PluginOptionDesc d;
        d.type = o->type & 0xffu;
        d.live = (o->type & BB_OPT_LIVE) != 0;
        d.key = o->key ? o->key : "";
        d.label = o->label ? o->label : d.key;
        d.help = o->help ? o->help : "";
        d.def = o->def ? o->def : "";
        d.choices = o->choices ? o->choices : "";
        d.min = o->min;
        d.max = o->max;
        e.options.push_back(std::move(d));
    }
    return e;
}

bool is_plugin_file(const std::string& n) {
#if defined(_WIN32)
    return n.size() > 4 && n.compare(n.size() - 4, 4, ".dll") == 0;
#else
    return n.size() > 3 && n.compare(n.size() - 3, 3, ".so") == 0;
#endif
}

}  // namespace

void plugins_load() {
    if (const char* e = std::getenv("BBHOST_PLUGINS"); e && e[0] == '0') {
        host_log("plugin: none loaded (BBHOST_PLUGINS=0)");
        return;
    }
    // Last run's plugin overlays go: a plugin writes what this run needs.
    if (const std::string root = overlay_root(); !root.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
    const std::vector<std::string> dirs = plugins_dirs();
    // Official plugins fetched while their old copy was loaded wait as .new.
    for (const std::string& dir : dirs) {
        std::vector<std::string> names;
        if (!host_list_dir(dir.c_str(), &names)) continue;
        for (const std::string& n : names) {
            if (n.size() < 5 || n.compare(n.size() - 4, 4, ".new") != 0) continue;
            const std::string base = dir + "/" + n.substr(0, n.size() - 4);
            std::error_code ec;
            std::filesystem::rename(dir + "/" + n, base, ec);
            if (!ec) {
                std::filesystem::rename(dir + "/" + n + ".sig", base + ".sig", ec);
                host_log("plugin: %s updated from its fetched copy", base.c_str());
            }
        }
    }
    std::map<std::string, std::string> files;
    for (const std::string& dir : dirs) {
        std::vector<std::string> names;
        if (!host_list_dir(dir.c_str(), &names)) continue;
        for (const std::string& n : names) {
            if (is_plugin_file(n)) files[n] = dir + "/" + n;
        }
    }
    // Official plugins carry <file>.sig: the release key's Ed25519 signature
    // over the file, made by the release workflow. A plugin with a signature
    // that does not verify was changed after signing and is not loaded;
    // plugins.signed_only = true loads official plugins alone.
    const bool signed_only = config_value("plugins.signed_only") == "true";
    for (const auto& [name, path] : files) {
        bool official = false;
        if (const std::string sig = read_file(path + ".sig"); !sig.empty()) {
            if (!updater::signature_ok(read_file(path), sig, updater::release_key())) {
                host_log("plugin: %s: its signature does not match the file; not loaded", path.c_str());
                continue;
            }
            official = true;
        } else if (signed_only) {
            host_log("plugin: %s: not signed, and plugins.signed_only is set; not loaded", path.c_str());
            continue;
        }
        std::string err;
        void* h = open_lib(path, &err);
        if (!h) {
            host_log("plugin: %s: cannot load: %s", path.c_str(), err.c_str());
            continue;
        }
        Plugin p;
        p.name = name.substr(0, name.rfind('.'));
        p.path = path;
        p.handle = h;
        p.init = reinterpret_cast<int (*)(const BbHostApi*)>(sym(h, "bb_plugin_init"));
        p.image = reinterpret_cast<int (*)(const BbHostApi*)>(sym(h, "bb_plugin_image"));
        p.info = static_cast<const BbPluginInfo*>(sym(h, "bb_plugin_info"));
        p.options = static_cast<const BbOption*>(sym(h, "bb_plugin_options"));
        if (!p.init && !p.image) {
            host_log("plugin: %s: no bb_plugin_init or bb_plugin_image; not a plugin", path.c_str());
            close_lib(h);
            continue;
        }
        if (p.info && p.info->name && p.info->name[0]) p.name = p.info->name;
        const std::uint32_t flags = p.info ? p.info->flags : 0;
        const int sw = config_switch(p.name);
        const bool off = sw == 0 || (sw < 0 && (flags & BB_PLUGIN_OPT_IN));
        // A plugin that can play by another player's rules is loaded even
        // when off - as a visitor, changing nothing of the player's own -
        // so a vanilla player summoned into a randomizer world plays that
        // world and comes home to vanilla. plugins.join_modded_worlds = false
        // turns it off.
        if (off && (flags & BB_PLUGIN_ADOPTS_RULES) && join_modded_worlds()) {
            p.visitor = true;
        } else if (off) {
            if (sw == 0) host_log("plugin: %s: off (plugins.%s = false)", p.name.c_str(), p.name.c_str());
            else host_log("plugin: %s: off until turned on ([plugins] %s = true)", p.name.c_str(), p.name.c_str());
            close_lib(h);
            continue;
        }

        if (p.init) {
            const int r = p.init(&g_api);
            if (r != 0) {
                host_log("plugin: %s: init returned %d; unloaded", p.name.c_str(), r);
                close_lib(h);
                continue;
            }
        }
        p.official = official;
        // A plugin the player turned on that says it changes the game (a boss
        // rush) may move the player; the test switch is the other way.
        if ((flags & BB_PLUGIN_OPT_IN) && (flags & BB_PLUGIN_GAMEPLAY) && sw == 1 && !p.visitor) world_warp_allow();
        if (p.info) {
            host_log("plugin: %s %s loaded%s%s (%s)", p.name.c_str(), p.info->version ? p.info->version : "",
                     official ? ", official" : "",
                     p.visitor ? ", off: only to play other players' worlds" : (flags & BB_PLUGIN_GAMEPLAY) ? ", changes the game" : "",
                     path.c_str());
        } else {
            host_log("plugin: %s loaded%s (%s)", p.name.c_str(), official ? ", official" : "", path.c_str());
        }
        g_plugins.push_back(p);
    }
    g_early = false;
}

void plugins_image(ElfImage* image) {
    g_image = image;
    if (!g_plugins.empty() && api_eboot_is_109()) {
        static const std::uint8_t prologue[17] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                  0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x28};
        if (api_hook2(0x18235e0, prologue, sizeof(prologue), &layout_request_hook, nullptr) != 0)
            host_log("plugin: the layout request hook did not go in: a rewritten layout waits for another map's load");
        static const std::uint8_t loaded[20] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                                0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xd8, 0x00, 0x00, 0x00};
        if (api_hook2(0x18094e0, loaded, sizeof(loaded), &layout_loaded_hook, nullptr) != 0)
            host_log("plugin: the layout parse hook did not go in: a summons keeps the resident map's enemies");
    }
    g_in_image = true;
    for (std::size_t i = 0; i < g_plugins.size();) {
        Plugin& p = g_plugins[i];
        if (p.image) {
            const int r = p.image(&g_api);
            if (r != 0) {
                host_log("plugin: %s: image phase returned %d; unloaded", p.name.c_str(), r);
                close_lib(p.handle);
                g_plugins.erase(g_plugins.begin() + static_cast<std::ptrdiff_t>(i));
                continue;
            }
        }
        ++i;
    }
    g_in_image = false;
}

void plugins_frame() {
    speffect_test_tick();  // BBHOST_SPEFFECT_TEST
    chr_watch_tick();      // BBHOST_CHR_WATCH
    std::vector<FrameCallback> now;
    std::vector<WorldLoadCallback> loads;
    std::vector<MenuRequest> requests;
    std::vector<OptionCallback> options;
    std::vector<ActionCallback> actions;
    {
        std::lock_guard<std::mutex> lk(g_frame_mu);
        requests.swap(g_menu_requests);
        if (!requests.empty()) {
            options = g_option_callbacks;
            actions = g_action_callbacks;
        }
        if (g_frame_callbacks.empty() && g_world_callbacks.empty() && requests.empty()) return;
        now = g_frame_callbacks;
        loads = g_world_callbacks;
    }
    for (const MenuRequest& r : requests) {
        if (r.action) {
            host_log("plugin: %s: action %s", r.name.c_str(), r.key.c_str());
            for (const ActionCallback& c : actions) {
                if (c.name == r.name) c.fn(r.key.c_str(), c.user);
            }
            continue;
        }
        const std::string raw = toml_value(r.value);
        config_value_set(r.name + "." + r.key, raw);
        config_set_values(config_user_file(), {{r.name, r.key, raw}});
        for (const OptionCallback& c : options) {
            if (c.name == r.name) c.fn(r.key.c_str(), r.value.c_str(), c.user);
        }
    }
    if (!loads.empty()) {
        // A world load is a new main player (every load builds one) or the
        // player in another block (a warp that keeps it).
        const std::uint64_t man = world_chr_man();
        const std::uint64_t player = man ? rp(man + 0x60) : 0;
        std::uint32_t block = 0;
        if (!player || !world_player_block(&block)) {
            g_last_player = 0;
        } else if (player != g_last_player || block != g_last_block) {
            g_last_player = player;
            g_last_block = block;
            for (const WorldLoadCallback& c : loads) c.fn(block, c.user);
        }
    }
    for (const FrameCallback& c : now) c.fn(c.user);
}

void plugins_event_flag(unsigned id, bool value) {
    std::vector<FlagCallback> now;
    {
        std::lock_guard<std::mutex> lk(g_frame_mu);
        if (g_flag_callbacks.empty()) return;
        now = g_flag_callbacks;
    }
    for (const FlagCallback& c : now) c.fn(id, value ? 1 : 0, c.user);
}

// --- the plugin manager -----------------------------------------------------

std::vector<std::string> plugins_dirs() {
    std::vector<std::string> dirs = {config_exe_dir() + "/plugins", config_user_dir() + "/plugins"};
    if (const char* d = hle_fs_data_root(); d && d[0]) dirs.push_back(std::string(d) + "/plugins");
    return dirs;
}

std::vector<PluginEntry> plugins_scan() {
    std::map<std::string, std::string> files;
    for (const std::string& dir : plugins_dirs()) {
        std::vector<std::string> names;
        if (!host_list_dir(dir.c_str(), &names)) continue;
        for (const std::string& n : names) {
            if (is_plugin_file(n)) files[n] = dir + "/" + n;
        }
    }
    std::vector<PluginEntry> out;
    for (const auto& [name, path] : files) {
        PluginEntry e;
        bool bad = false, official = false;
        if (const std::string sig = read_file(path + ".sig"); !sig.empty()) {
            official = updater::signature_ok(read_file(path), sig, updater::release_key());
            bad = !official;
        }
        std::string err;
        void* h = bad ? nullptr : open_lib(path, &err);
        if (h) {
            e = entry_of(name, path, static_cast<const BbPluginInfo*>(sym(h, "bb_plugin_info")),
                         static_cast<const BbOption*>(sym(h, "bb_plugin_options")));
            close_lib(h);
        } else {
            e = entry_of(name, path, nullptr, nullptr);
        }
        e.official = official;
        e.bad_signature = bad;
        const int sw = config_switch(e.name);
        e.enabled = !bad && (sw == 1 || (sw < 0 && !(e.flags & BB_PLUGIN_OPT_IN)));
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<PluginEntry> plugins_catalog() {
    std::vector<PluginEntry> out;
    for (const Plugin& p : g_plugins) {
        PluginEntry e = entry_of(p.path.substr(p.path.find_last_of("/\\") + 1), p.path, p.info, p.options);
        e.name = p.name;
        e.official = p.official;
        e.loaded = true;
        e.visitor = p.visitor;
        e.enabled = !p.visitor;
        out.push_back(std::move(e));
    }
    return out;
}

void plugins_set_enabled(const std::string& name, bool on) {
    config_set_values(config_user_file(), {{"plugins", name, on ? "true" : "false"}});
    config_value_set("plugins." + name, on ? "true" : "false");
}

std::string plugins_option_value(const PluginEntry& p, const PluginOptionDesc& o) {
    const std::string v = config_value(p.name + "." + o.key);
    return v.empty() ? o.def : v;
}

void plugins_set_option(const std::string& name, const std::string& key, const std::string& value) {
    if (g_plugins.empty()) {
        // The launcher: nothing runs yet, so the file and the loaded values are all.
        const std::string raw = toml_value(value);
        config_set_values(config_user_file(), {{name, key, raw}});
        config_value_set(name + "." + key, raw);
        return;
    }
    std::lock_guard<std::mutex> lk(g_frame_mu);
    g_menu_requests.push_back({false, name, key, value});
}

void plugins_run_action(const std::string& name, const std::string& action) {
    std::lock_guard<std::mutex> lk(g_frame_mu);
    g_menu_requests.push_back({true, name, action, ""});
}

namespace {
bool has_seed_option(const Plugin& p) {
    for (const BbOption* o = p.options; o && o->type; ++o) {
        if (o->key && std::strcmp(o->key, "seed") == 0) return true;
    }
    return false;
}

// The seed a gameplay plugin plays: its "seed" setting, else the one it made
// and kept in <data>/plugins/<name>/seed.txt (the randomizer's way).
std::string plugin_seed(const Plugin& p) {
    std::string seed = config_value((p.name + ".seed").c_str());
    if (seed.empty()) {
        const char* data = hle_fs_data_root();
        if (data && data[0]) {
            std::ifstream in(std::string(data) + "/plugins/" + p.name + "/seed.txt");
            if (in) std::getline(in, seed);
        }
    }
    while (!seed.empty() && (seed.back() == '\r' || seed.back() == ' ')) seed.pop_back();
    return seed;
}

// A seed as the server takes it ([a-z0-9_.-], up to 40): kept when it is
// that already, else a hash of it, so "My Run" and "my run" stay two worlds.
std::string ruleset_seed(const std::string& seed) { return bb::seed_token(seed); }
}  // namespace

namespace {
// The rules this session's own world is played by (not a host's).
std::string own_ruleset() {
    static std::mutex mu;
    static std::string cached;
    std::lock_guard<std::mutex> lk(mu);
    std::map<std::string, std::string> own;
    {
        std::lock_guard<std::mutex> rk(g_rules_mu);
        own = g_rules;
    }
    if (!cached.empty() && own.empty()) return cached;
    std::vector<std::string> parts;
    bool unseeded = false;
    for (const Plugin& p : g_plugins) {
        if (!p.info || !(p.info->flags & BB_PLUGIN_GAMEPLAY) || p.visitor) continue;
        if (auto it = own.find(p.name); it != own.end()) {
            parts.push_back(it->second);  // what the plugin says it plays (set_rules)
            continue;
        }
        std::string token;
        for (const char ch : p.name) token += (ch >= 'A' && ch <= 'Z') ? static_cast<char>(ch - 'A' + 'a') : ch;
        if (has_seed_option(p)) {
            const std::string seed = plugin_seed(p);
            if (seed.empty()) unseeded = true;  // not made yet: ask again next request
            else token += ";seed=" + ruleset_seed(seed);
        }
        parts.push_back(token);
    }
    // The patch manifests that change the game (debug dash, the rally window,
    // bigger weapons, ...: engine/patch_manifest.h), one entry each. Final once
    // the manifests have applied, so the rules are kept only from then on.
    for (const std::string& name : patch_manifests_gameplay()) parts.push_back(patch_rule_token(name));
    const std::string out = ruleset_join(parts);
    if (!unseeded && patch_manifests_done()) cached = out;
    static std::string logged;
    if (out != logged) {
        logged = out;
        host_log("plugin: this session's online rules (X-BBHost-Ruleset): %s", out.c_str());
    }
    return out;
}

// "boss_rush,randomizer;seed=x" -> its entries without the named plugins'.
std::vector<std::string> rules_without(const std::string& rules, const std::vector<std::string>& names) {
    std::vector<std::string> out;
    std::size_t at = 0;
    while (at <= rules.size()) {
        const std::size_t end = std::min(rules.find(',', at), rules.size());
        const std::string part = rules.substr(at, end - at);
        at = end + 1;
        if (part.empty() || part == "vanilla") continue;
        const std::string name = part.substr(0, part.find(';'));
        if (std::find(names.begin(), names.end(), name) == names.end()) out.push_back(part);
    }
    std::sort(out.begin(), out.end());
    return out;
}
}  // namespace

bool plugins_active(const std::string& name) {
    for (const Plugin& p : g_plugins) {
        if (p.name == name && !p.visitor) return true;
    }
    return false;
}

std::string plugins_ruleset() {
    {
        std::lock_guard<std::mutex> rk(g_rules_mu);
        // A guest playing by its host's rules says so (plugins_enter_world).
        if (!g_visiting.empty()) return g_visiting;
    }
    return own_ruleset();
}

std::string plugins_adopts() {
    // As the visitor (plugins_load): with plugins.join_modded_worlds = false
    // the server matches this player only within its own rules.
    if (!join_modded_worlds()) return {};
    std::string out;
    for (const Plugin& p : g_plugins) {
        if (p.info && (p.info->flags & BB_PLUGIN_ADOPTS_RULES)) out += (out.empty() ? "" : ",") + p.name;
    }
    return out;
}

void plugins_enter_world(const std::string& host_rules) {
    if (!host_rules.empty()) {
        // From the server: what bbhost would send itself ([a-z0-9_.;=,-], short),
        // and different from this session's own rules only in plugins it can
        // adopt - else it is not a world this client can play.
        bool ok = host_rules.size() <= 256;
        for (const char ch : host_rules) {
            if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == ';' ||
                  ch == '=' || ch == ',' || ch == '-'))
                ok = false;
        }
        std::vector<std::string> adopt;
        for (const Plugin& p : g_plugins) {
            if (p.info && (p.info->flags & BB_PLUGIN_ADOPTS_RULES)) adopt.push_back(p.name);
        }
        if (ok && rules_without(host_rules, adopt) != rules_without(own_ruleset(), adopt)) ok = false;
        if (!ok) {
            host_log("plugin: the host's rules (%zu bytes) are not ones this session can play by; keeping its own",
                     host_rules.size());
            return;
        }
    }
    std::vector<WorldRulesCallback> calls;
    {
        std::lock_guard<std::mutex> lk(g_rules_mu);
        if (host_rules == g_visiting) return;
        g_visiting = host_rules;
        calls = g_world_rules_callbacks;
    }
    host_log("plugin: %s", host_rules.empty() ? "back to the player's own rules" : ("playing by the host's rules: " + host_rules).c_str());
    for (const WorldRulesCallback& c : calls) {
        if (host_rules.empty()) {
            c.fn(nullptr, c.user);
            continue;
        }
        // The host's entry for this plugin: "randomizer;seed=x" in "boss_rush,randomizer;seed=x".
        std::string entry;
        std::size_t at = 0;
        while (at <= host_rules.size()) {
            const std::size_t end = std::min(host_rules.find(',', at), host_rules.size());
            const std::string part = host_rules.substr(at, end - at);
            if (part == c.name || part.rfind(c.name + ";", 0) == 0) entry = part;
            at = end + 1;
        }
        c.fn(entry.c_str(), c.user);  // "" when the host plays without this plugin
    }
}
