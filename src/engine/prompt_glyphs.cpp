// The start-up half of engine/prompt_glyphs.h: choosing the style and
// mounting the patched menu/common.tpf.dcx.
#include "engine/prompt_glyphs.h"

#include "hle/fs.h"
#include "host/options.h"
#include "host/settings.h"
#include "host/window.h"
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>

namespace {

namespace fs = std::filesystem;

// Bumped whenever the glyphs or the way they are written change, so an
// install makes its copy again instead of keeping the last one.
constexpr int kGeneratorVersion = 1;
const char kCommonTpf[] = "dvdroot_ps4/menu/common.tpf.dcx";

std::optional<GlyphBytes> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::nullopt;
    return GlyphBytes(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

bool write_file(const fs::path& p, const GlyphBytes& b) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    const fs::path tmp = p.string() + ".part";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
        if (!f) return false;
    }
    fs::rename(tmp, p, ec);
    return !ec;
}

std::uint64_t fnv1a(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ull) {
    const auto* c = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
    return h;
}

const char* folder_of(PromptStyle s) {
    switch (s) {
        case PromptStyle::Xbox: return "xbox";
        case PromptStyle::SteamDeck: return "steamdeck";
        default: return "steamcontroller";
    }
}

}  // namespace

void prompt_glyphs_install() {
    if (const char* e = std::getenv("BBHOST_PROMPT_GLYPHS"); e && e[0] == '0') {
        host_log("prompt glyphs: off (BBHOST_PROMPT_GLYPHS=0)");
        return;
    }
    const int setting = host_settings_ref().controller_prompts;
    const int family = host_pad_family();
    const PromptStyle style = prompt_style_choose(setting, family, host_steam_deck());
    host_log("prompt glyphs: %s (%s; the first pad is %s%s)", prompt_style_name(style), setting ? "chosen" : "Auto",
             host_pad_family_name(family), host_steam_deck() ? ", on a Steam Deck" : "");
    if (style == PromptStyle::PlayStation) return;
    const std::string data = hle_fs_data_root();
    if (data.empty()) return;
    const auto src = read_file(hle_fs_game_file(kCommonTpf));
    if (!src) {
        host_log("prompt glyphs: the dump has no %s; the game's own glyphs stay", kCommonTpf);
        return;
    }
    // One folder a style, each with its own stamp: switching back and forth
    // reuses what was made before.
    const fs::path out = fs::path(data) / "bbhost" / "prompt-glyphs" / folder_of(style);
    std::uint64_t h = fnv1a(&kGeneratorVersion, sizeof(kGeneratorVersion));
    h = fnv1a(&style, sizeof(style), h);
    h = fnv1a(src->data(), src->size(), h);
    char stamp[32];
    std::snprintf(stamp, sizeof(stamp), "%016llx", static_cast<unsigned long long>(h));
    const auto have = read_file(out / "stamp");
    if (!(have && std::string(have->begin(), have->end()) == stamp && fs::exists(out / kCommonTpf))) {
        std::string why;
        const GlyphBytes made = prompt_glyphs_apply_dcx(*src, style, &why);
        if (made.empty()) {
            host_log("prompt glyphs: %s not made: %s; the game's own glyphs stay", kCommonTpf, why.c_str());
            return;
        }
        if (!write_file(out / kCommonTpf, made) || !write_file(out / "stamp", GlyphBytes(stamp, stamp + std::strlen(stamp)))) {
            host_log("prompt glyphs: cannot write %s", out.string().c_str());
            return;
        }
        host_log("prompt glyphs: made %s (%zu bytes) in %s", kCommonTpf, made.size(), out.string().c_str());
    }
    hle_fs_add_generated_root(out.string().c_str());
}
