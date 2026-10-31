// The C++ side of the plugin API: bbhost_plugin.h's C table made pleasant,
// the game's params as typed rows (bbhost/params.hpp), and the engine's
// layouts and symbols (bbhost/engine/engine.hpp, included on demand).
//
//   #include "bbhost/sdk.hpp"
//   static bb::Plugin g;
//   BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
//       if (!g.attach(api, 8)) return 1;
//       if (auto* w = g.param<bb::params::EquipParamWeapon>(1000000)) w->weight = 0;
//       g.every_frame([] { ... });
//       return 0;
//   }
//
// Everything here runs in the plugin; the host sees only the C table.
#pragma once

#include "bbhost_plugin.h"
#include "bbhost/params.hpp"

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace bb {

// A small, fast, seedable generator (splitmix64): the same seed gives the
// same sequence on every platform, which std:: distributions do not promise.
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    // [0, n)
    std::uint32_t below(std::uint32_t n) { return n ? static_cast<std::uint32_t>(next() % n) : 0; }
    // [lo, hi]
    int range(int lo, int hi) { return lo + static_cast<int>(below(static_cast<std::uint32_t>(hi - lo + 1))); }
    float unit() { return static_cast<float>(next() >> 40) / static_cast<float>(1ull << 24); }
    template <typename T>
    void shuffle(std::vector<T>& v) {
        for (std::size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[below(static_cast<std::uint32_t>(i))]);
    }
};

// A seed as the server and other players see it (the X-BBHost-Ruleset
// header's ";seed="): the text itself when it is [a-z0-9_.-] and up to 40
// long, else "h" and its FNV-1a 64 hash in hex. A world made from the token
// can be made again by anyone who has the token - a guest joining it.
inline std::string seed_token(const std::string& text) {
    bool plain = !text.empty() && text.size() <= 40;
    for (const char ch : text) {
        if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == '-')) plain = false;
    }
    if (plain) return text;
    std::uint64_t h = 1469598103934665603ull;
    for (const unsigned char ch : text) {
        h ^= ch;
        h *= 1099511628211ull;
    }
    char out[24];
    std::snprintf(out, sizeof(out), "h%016llx", static_cast<unsigned long long>(h));
    return out;
}

// A text seed ("my run 1") as a number, stably (FNV-1a).
inline std::uint64_t seed_of(const std::string& text) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : text) h = (h ^ c) * 0x100000001b3ull;
    return h;
}

class Plugin {
public:
    const BbHostApi* api = nullptr;

    // Keeps the table when the host is at least `version`. False (and a
    // log line) otherwise: return nonzero from the entry point then.
    bool attach(const BbHostApi* a, std::uint32_t version) {
        if (!a || a->version < version) {
            if (a && a->log) a->log("needs plugin API %u, the host has %u", version, a->version);
            return false;
        }
        api = a;
        return true;
    }

    void log(const char* fmt, ...) const {
        char line[1024];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(line, sizeof(line), fmt, ap);
        va_end(ap);
        api->log("%s", line);
    }

    // --- config: [section] key = value in bbhost.toml -----------------------
    std::string config(const char* key, const char* fallback = "") const {
        const char* v = api->config(key);
        return v && v[0] ? std::string(v) : std::string(fallback);
    }
    bool config_bool(const char* key, bool fallback) const {
        const std::string v = config(key);
        if (v.empty()) return fallback;
        return v == "true" || v == "1" || v == "on";
    }
    double config_number(const char* key, double fallback) const {
        const std::string v = config(key);
        return v.empty() ? fallback : std::strtod(v.c_str(), nullptr);
    }

    // --- params --------------------------------------------------------------
    // The live row of a gameparam table, typed: bb::params::<Table>::Row*.
    // nullptr before the table loads (they load during the boot: from
    // every_frame or on_world_load they are there) or for a missing id.
    template <typename Table>
    typename Table::Row* param(std::uint32_t id) const {
        std::size_t bytes = 0;
        void* row = api->param_row(Table::name, id, &bytes);
        if (!row || (bytes && bytes != sizeof(typename Table::Row))) return nullptr;
        return static_cast<typename Table::Row*>(row);
    }
    // Every row id of a table, in order.
    template <typename Table>
    std::vector<std::uint32_t> param_ids() const {
        std::size_t n = 0;
        if (api->param_ids(Table::name, nullptr, 0, &n) != 0) return {};
        std::vector<std::uint32_t> ids(n);
        if (api->param_ids(Table::name, ids.data(), n, &n) != 0) return {};
        ids.resize(n);
        return ids;
    }
    // fn(id, row&) for every row.
    template <typename Table, typename Fn>
    void each_row(Fn&& fn) const {
        for (std::uint32_t id : param_ids<Table>()) {
            if (auto* r = param<Table>(id)) fn(id, *r);
        }
    }

    // --- callbacks -----------------------------------------------------------
    // fn() once a game frame on the main thread.
    void every_frame(std::function<void()> fn) {
        frames_.push_back(std::move(fn));
        if (frames_.size() == 1) api->on_frame(&Plugin::frame_thunk, this);
    }
    // fn(block) each time the player is placed in a map.
    void on_world_load(std::function<void(std::uint32_t)> fn) {
        loads_.push_back(std::move(fn));
        if (loads_.size() == 1) api->on_world_load(&Plugin::load_thunk, this);
    }
    // fn(id, value) for every event flag that changes.
    void on_event_flag(std::function<void(std::uint32_t, bool)> fn) {
        flags_.push_back(std::move(fn));
        if (flags_.size() == 1) api->on_event_flag(&Plugin::flag_thunk, this);
    }
    // fn(camera) each frame once the game has placed the player's camera,
    // before anything is drawn from it (plugin API 13). False on an older host.
    bool on_camera(std::function<void(BbCamera&)> fn) {
        if (api->version < 13 || api->size < offsetof(BbHostApi, on_camera) + sizeof(api->on_camera)) return false;
        cameras_.push_back(std::move(fn));
        return cameras_.size() > 1 || api->on_camera(&Plugin::camera_thunk, this) == 0;
    }

    // --- the world -----------------------------------------------------------
    std::vector<BbChr> characters() const {
        std::size_t n = 0;
        if (api->chr_list(nullptr, 0, &n) != 0) return {};
        std::vector<BbChr> v(n + 16);
        if (api->chr_list(v.data(), v.size(), &n) != 0) return {};
        v.resize(n < v.size() ? n : v.size());
        return v;
    }
    bool flag(std::uint32_t id) const {
        int v = 0;
        return api->event_flag_get(id, &v) == 0 && v;
    }
    void set_flag(std::uint32_t id, bool on) const { api->event_flag_set(id, on ? 1 : 0); }
    std::int64_t stat(const char* name, std::int64_t fallback = 0) const {
        std::int64_t v = 0;
        return api->player_stat_get(name, &v) == 0 ? v : fallback;
    }
    void set_stat(const char* name, std::int64_t v) const { api->player_stat_set(name, v); }
    void message(const std::string& text, float seconds = 4.0f) const { api->show_message(text.c_str(), seconds); }
    // A line placed by the plugin (plugin API 14): its box's corner at (x, y),
    // fractions of the window, `align` BB_TEXT_*. False on an older host, which
    // shows the text as message's banner instead.
    bool text_at(const std::string& text, float seconds, float x, float y, float scale, std::uint32_t align) const {
        if (api->version < 14 || api->size < offsetof(BbHostApi, show_text) + sizeof(api->show_text)) {
            message(text, seconds);
            return false;
        }
        return api->show_text(text.c_str(), seconds, x, y, scale, align) == 0;
    }
    // A loading screen is up (plugin API 12: the host's own answer, the one
    // its log's "loading:" lines come from). on_world_load comes with a
    // load's screen still up: a plugin that moves the player waits for this to
    // be false. On an older host the same test is made here, from the game:
    // its NowLoading request (a byte at 0x596286b) with no in-game frame run
    // since (the CSNowLoadingHelper at *0x593e8b8, +0x50 or +0x51), on the
    // 1.09 eboot those addresses are for; false on another.
    bool loading_screen() const {
        if (api->version >= 12 && api->size >= offsetof(BbHostApi, loading_screen) + sizeof(api->loading_screen) &&
            api->loading_screen)
            return api->loading_screen() != 0;
        if (!api->eboot_is_109()) return false;
        std::uint8_t requested = 0;
        std::uint64_t helper = 0;
        if (api->read(0x596286b, &requested, 1) != 0 || !requested) return false;
        if (api->read(0x593e8b8, &helper, 8) != 0 || !helper) return true;
        const auto* h = reinterpret_cast<const volatile std::uint8_t*>(static_cast<std::uintptr_t>(helper));
        return !h[0x50] && !h[0x51];
    }

    // The physics world's gravity (x, y, z; y up), which the player, the
    // enemies, ragdolls and loose objects all fall by: {0, -9.8, 0} as the
    // game makes it. Write it to change gravity for everything. nullptr
    // before the world exists. (The physics system's slot 0x593d700 holds a
    // wrapper at +0x28 whose +8 is the Havok world, made with this vector at
    // +0x5b0 by 0x1bfa260; checked here before it is handed out.)
    float* gravity() const {
        std::uint64_t sys = 0, wrap = 0, world = 0;
        // The slot by its Binary Ninja address: read() moves an image address
        // by the image's slide itself, and refuses a runtime address inside
        // the slid image (Windows).
        if (!api->eboot_is_109() || api->read(0x593d700, &sys, 8) || !sys || api->read(sys + 0x28, &wrap, 8) || !wrap ||
            api->read(wrap + 8, &world, 8) || !world)
            return nullptr;
        float v[3];
        if (api->read(world + 0x5b0, v, sizeof(v)) || v[0] != 0.0f || v[2] != 0.0f || !(v[1] < 0.0f && v[1] > -1000.0f))
            return nullptr;
        return reinterpret_cast<float*>(world + 0x5b0);
    }
    void sp_effect(std::int32_t id, void* chr = nullptr) const { api->sp_effect_apply(chr, id); }

    // --- game code -----------------------------------------------------------
    // Calls a game function by symbol name with integer/pointer arguments.
    template <typename... A>
    std::uint64_t call(const char* symbol, A... a) const {
        BbCallRegs r{};
        const std::uint64_t args[] = {0, to_u64(a)...};
        for (std::size_t i = 1; i < sizeof(args) / sizeof(args[0]) && i <= 6; ++i) r.arg[i - 1] = args[i];
        const std::uint64_t at = api->symbol(symbol);
        if (!at || api->call_guest(at, &r) != 0) return 0;
        return r.ret;
    }

private:
    template <typename T>
    static std::uint64_t to_u64(T v) {
        if constexpr (std::is_pointer_v<T>) return reinterpret_cast<std::uint64_t>(v);
        else return static_cast<std::uint64_t>(v);
    }
    static void frame_thunk(void* self) {
        for (auto& f : static_cast<Plugin*>(self)->frames_) f();
    }
    static void load_thunk(std::uint32_t block, void* self) {
        for (auto& f : static_cast<Plugin*>(self)->loads_) f(block);
    }
    static void flag_thunk(std::uint32_t id, int value, void* self) {
        for (auto& f : static_cast<Plugin*>(self)->flags_) f(id, value != 0);
    }
    static void camera_thunk(BbCamera* cam, void* self) {
        for (auto& f : static_cast<Plugin*>(self)->cameras_) f(*cam);
    }
    std::vector<std::function<void()>> frames_;
    std::vector<std::function<void(std::uint32_t)>> loads_;
    std::vector<std::function<void(std::uint32_t, bool)>> flags_;
    std::vector<std::function<void(BbCamera&)>> cameras_;
};

// The packed map block (player_block) as its parts: m24_01_00_00 = {24, 1, 0, 0}.
struct Block {
    std::uint8_t area, block, sub1, sub2;
    static Block of(std::uint32_t b) {
        return {static_cast<std::uint8_t>(b >> 24), static_cast<std::uint8_t>(b >> 16), static_cast<std::uint8_t>(b >> 8),
                static_cast<std::uint8_t>(b)};
    }
};

}  // namespace bb
