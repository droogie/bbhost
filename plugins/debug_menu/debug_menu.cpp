// Debug Menu: an official bbhost plugin (docs/plugins.md) - the developers'
// debug menu, which the game still carries but leaves off.
//
//   [plugins]
//   debug_menu = true
//
// With it on, bbhost restores the menu at the next start (engine/debug_menu.h)
// and the Debug Menu key (`) opens and closes it. The menu draws with a debug
// font the game does not ship; this plugin makes it (debug_font.hpp) from the
// public-domain X11 fonts k14 and 7x14 and hands it to the game through its
// file overlay, so nothing has to be downloaded. A DbgFont14h.ccm and .tpf in
// the player's mods folder still win.
//
// It is a gameplay plugin: the menu can hand out items and change the rules,
// so its sessions announce a ruleset of their own and the server keeps their
// messages, bloodstains and statistics apart from other players'.
//
// This Japanese-only release keeps the developers' original Japanese text.
#include "bbhost/sdk.hpp"

#include "debug_font.hpp"

#include <cstdint>
#include <string>

namespace {

// font14.bin (tools/make_debug_font.py), built in (CMakeLists.txt).
const std::uint8_t kFont14[] = {
#include "debug_font14.inc"
};

bb::Plugin g;

}  // namespace

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN | BB_PLUGIN_GAMEPLAY,
    "debug_menu",
    "デバッグメニュー",
    "1.0.0",
    "bbhost",
    "ゲームに残る開発用メニューです。次回起動以降、` キーで開閉します。対戦相手への不正使用はアカウント停止の対象です。友人との試用は許可されています。",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, nullptr, "オンラインでの使用",
     "他プレイヤーのワールド・侵入・ランキングで不正に有利になる使い方はアカウント停止の対象です。友人との試用は許可されています。"},
    {BB_OPT_HEADING, nullptr, "使い方",
     "` キーで開閉します。キー設定で変更できます。方向キーで選択、Enter で開く、Backspace で戻ります。有効なセッションの手記・血痕・統計は他のプレイヤーと分離されます。"},
    {BB_OPT_END}};

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (!g.attach(api, 9)) return 1;
    if (!api->eboot_is_109()) return 1;
    debug_font::Files font;
    std::string err;
    if (!debug_font::build(kFont14, sizeof(kFont14), &font, &err)) {
        g.log("debug_menu: %s", err.c_str());
        return 1;
    }
    if (api->overlay_file("debug_menu", "/dvdroot_ps4/adhoc/font/DbgFont14h.ccm", font.ccm.data(), font.ccm.size()) != 0 ||
        api->overlay_file("debug_menu", "/dvdroot_ps4/adhoc/font/DbgFont14h.tpf", font.tpf.data(), font.tpf.size()) != 0) {
        g.log("debug_menu: the debug font could not be written to the overlay");
        return 1;
    }
    g.log("debug_menu: the debug font made (%d glyphs on %d pages); bbhost restores the menu", font.glyphs, font.pages);
    return 0;
}
