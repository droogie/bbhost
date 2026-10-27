#include "host/japanese_labels.h"
#include "host/options.h"
#include "host/updater.h"
#include "host/settings.h"
#include "host/settings_store.h"

#include "host/bindings.h"
#include "host/overlay.h"
#include "host/window.h"
#include "core/config.h"
#include "engine/live_resolution.h"
#include "hle/modules.h"
#include "hle/np.h"
#include "net/account.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if defined(BBHOST_HAVE_SDL3)
#include <SDL3/SDL.h>
#endif

namespace {

struct Setting;
bool reads_on(const Setting& s, int index);  // below
int fps_from_label(const char* label);

// One setting. Everything here is a choice between a few named values: a
// toggle is a choice of two, which keeps drawing and input to one case.
struct Setting {
    const char* key;                 // what it is saved as
    const char* label;
    std::vector<const char*> values;
    int index = 0;
    const char* note = nullptr;      // shown under the row when it is selected
    bool restart = false;            // takes effect on the next run
};

enum SettingId {
    kWindowMode,
    kVsync,
    kFrameCap,
    kFpsCounter,
    kMemoryCounter,
    kFov,
    kCameraDistance,
    kCameraHeight,
    kCameraTuning,
    kMouseMenu,
    kMouseCamera,
    kMouseSens,
    kMouseInvertX,
    kMouseInvertY,
    kMouseAutoRotation,
    kDrawCursor,
    kButtonPrompts,
    kSsao,
    kMotionBlur,
    kAntiAlias,
    kDepthOfField,
    kChromaticAberration,
    kBloom,
    kVignette,
    kShadowDistance,
    kAoStrength,
    kSaturation,
    kFog,
    kModelDetail,
    kResolution,
    kShaders,
    kSkipLogos,
    kDebugMenu,
    kDebugCamera,
    kChangeAppearance,
    kRebirth,
    kFivePlayers,
    kSettingCount,
};

// A row on screen: a section header, a setting, or the close button.
struct Row {
    // Text: a field the player types (field 0 the account name, 1 a recovery
    // code). Action: a button (field 0 link with Discord, 1 create an account
    // on this PC, 2 recover, 3 website code, 4 sign out). Info: a line the
    // host writes (field 0 who is signed in, 1 the code in play, if any).
    // Fields 10 and up are the UPDATES rows (host/updater.h): Info 10 the
    // version and what the check found, 11 why it failed or the newer
    // release's page; Action 10 check, 11 open that page in the browser.
    enum Kind { Header, Option, Close, Text, Action, Info } kind;
    const char* text;
    int setting = -1;
    int field = -1;
};

Setting g_set[kSettingCount] = {
    {"window_mode", "画面モード", {"Windowed", "Fullscreen"}, 0,
     "全画面はデスクトップの解像度を使うボーダーレス表示です。", false},
    {"vsync", "垂直同期", {"On", "Off"}, 0,
     "無効では完成したフレームをすぐに表示します。画面が裂けて見える場合があります。", false},
    // The game's own pace (engine/frame_rate.h), the same five the game's Frame
    // Rate pick list offers (engine/option_menu.cpp). The key is the old Frame
    // cap's, so a file keeps its 30 or 60; a file older than version 3 holds a
    // cap on presentation, whose 90, 120, 144 and Off ran the game at 60, and
    // reads as 60 (host_options_load).
    {"frame_cap", "フレームレート", {"30", "60", "90", "120", "Uncapped"}, 0,
     "30 fps は原作の速度です。60・90・120 fps は各速度で動作します。無制限では実時間に合わせ、垂直同期時は画面の更新頻度、無効時はワールド内で最大 240 fps です。次回起動時に反映します。",
     false},
    // The game's frames a second, counted as they start (host/window.cpp,
    // host_fps_frame_start), with how busy the GPU is beside it.
    {"fps_counter", "FPS 表示", {"Off", "On"}, 0,
     "画面右上に毎秒の描画フレーム数を表示します。", false},
    // What bbhost holds of video memory and of the memory Windows counts as
    // shared GPU memory (host/window.cpp, host_gpu_memory_glance).
    {"memory_counter", "メモリー表示", {"Off", "On"}, 0,
     "ビデオメモリー使用量、解放可能なキャッシュ、GPU が参照するゲームメモリーを表示します。Windows では後者が共有 GPU メモリーに相当します。", false},
    // The follow camera's vertical field of view, LockCamParam's camFovY
    // (43 degrees in most areas) widened 5% a step (engine/camera.h).
    {"fov", "視野角", {"+0%", "+5%", "+10%", "+15%", "+20%", "+25%", "+30%", "+35%", "+40%", "+45%", "+50%"}, 0,
     "カメラの視野の広さです。+0% は原作、+50% は水平方向に約 97 度です。", false},
    // The same row's camDistTarget and chrOrgOffset_Y: 5 the game's own, 10%
    // a step either way.
    {"camera_distance", "カメラの距離", {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}, 5,
     "キャラクターからカメラまでの距離です。5 が原作の設定です。", false},
    {"camera_height", "カメラの高さ", {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}, 5,
     "キャラクターに対するカメラの高さです。5 が原作の設定です。", false},
    // The follow camera above 30 fps (engine/frame_rate.cpp camera_original):
    // the game's own values, the same camera at every frame rate, whatever is
    // saved here - the menus no longer offer it, and only
    // BBHOST_CAMERA_TUNING=fps++ runs the retune from Kyo's 60 FPS++ list. So
    // the default is Original, and a save records the camera that runs.
    {"camera_tuning", "カメラの調整", {"FPS++ (Kyo)", "Original"}, 1,
     "再起動が必要です。30 fps 超では FPS++ が Kyo の調整値、原作は 30 fps 時の値を使用します。",
     true},
    // On, as in DS3: a click only reaches the menus while one is open
    // (engine/menu_pointer.h), so it cannot swing a weapon in the world.
    {"mouse_menu", "メニューでマウスを使用", {"Off", "On"}, 1,
     "左クリックで決定、右クリックで戻る、ホイールで選択を移動します。", false},
    // On by default, and it only *disables* the automatic behaviour: the
    // mouse turns the camera whenever a menu is not open, which is what DS3
    // does and what a PC player expects. Off hands the camera back to the pad.
    {"mouse_camera", "マウスでカメラ操作", {"On", "Off"}, 0,
     "メニュー外でマウスによるカメラ操作を許可します。無効時はパッドで操作します。", false},
    // 0..10, the range of the game's own Camera Sensitivity slider, which is
    // the widget the PC Controls screen draws it with, and DS3's: the same
    // angle a count at each step (engine/mouse_camera.h), 5 its default.
    {"mouse_sens", "マウス感度", {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}, 5,
     "マウスの移動量に対するカメラの回転量です。", false},
    {"mouse_invert_x", "マウスの横方向", {"Off", "On"}, 0,
     "マウスの左右のカメラ操作を反転します。", false},
    {"mouse_invert_y", "マウスの縦方向", {"Off", "On"}, 0,
     "マウスの上下のカメラ操作を反転します。", false},
    // Off by default: with the mouse the camera stays where it is put, as
    // the community's "Disable Camera Auto Rotation" patch has it - but only
    // while the keyboard and mouse were used last; a controller gets the
    // game's own camera back (engine/mouse_camera.h).
    {"mouse_auto_rotation", "マウス使用時の自動回転", {"Off", "On"}, 0,
     "無効時は移動中もカメラの向きを維持します。コントローラー使用時は原作のカメラ動作になります。", false},
    {"draw_cursor", "ポインターを表示", {"On", "Off"}, 0,
     "ゲームにはマウスポインターがないため、bbhost が描画します。", false},
    // The game's key guide and its tutorial notes name a pad button as an
    // image tag; the port rewrites the tag into the key bound to it
    // (engine/key_prompts.h). Auto follows the device last used (host_input_device).
    {"button_prompts", "ボタン表示", {"Auto", "Keyboard", "Controller"}, 0,
     "ゲーム内の操作表示です。自動では最後に使ったキーボードまたはコントローラーに合わせます。", false},
    // The post-processing switches. They follow their settings live
    // (engine/graphics_patch.cpp, host/shader_patch.h).
    {"ssao", "アンビエントオクルージョン", {"On", "Off"}, 0,
     "画面空間の接触陰影です。無効にすると軽くなります。", false},
    // Not a byte patch: the blur shader passes every pixel through
    // (host/shader_patch.h).
    {"motion_blur", "モーションブラー", {"On", "Off"}, 0,
     "カメラやキャラクターの速い動きをぼかします。", false},
    {"anti_alias", "アンチエイリアス", {"On", "Off"}, 0,
     "原作の輪郭補正処理です。", false},
    {"depth_of_field", "被写界深度", {"On", "Off"}, 0,
     "遠景をぼかします。屋内では効果が小さい場合があります。", false},
    {"chromatic_aberration", "色収差", {"On", "Off"}, 0,
     "画面端の色ずれです。", false},
    // Two of the area's draw parameters (param/drawparam), scaled or cleared
    // as the game hands them to YEBIS each frame - engine/graphics_patch.cpp.
    // 10 is the area's own glare; 0 is none.
    {"bloom", "ブルーム", {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}, 10,
     "明るい光のにじみです。10 が原作の設定で、下げると弱まります。", false},
    {"vignette", "ビネット", {"On", "Off"}, 0,
     "一部のエリアで画面の周辺を暗くする効果です。", false},
    // The area's shadow cascades, stretched past the first split: 1 + 0.2 a
    // step, so the game's slider row (0..10) is the same setting.
    {"shadow_distance", "影の描画距離",
     {"x1.0", "x1.2", "x1.4", "x1.6", "x1.8", "x2.0", "x2.2", "x2.4", "x2.6", "x2.8", "x3.0"}, 0,
     "日光と月光の影を描く距離です。広げると影の解像感が下がります。", false},
    // Two more of the area's numbers, scaled: 5 is the area's own, 0 none, 10
    // twice it.
    {"ao_strength", "陰影の濃さ", {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}, 5,
     "隅や接触部分の陰影の濃さです。5 が原作の設定です。", false},
    {"saturation", "彩度", {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}, 5,
     "色の鮮やかさです。5 が原作、0 は白黒です。", false},
    // The fog table's opacities (engine/graphics_patch.cpp): 10 the area's own.
    {"fog", "霧", {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}, 10,
     "遠景と高さによる霧の濃さです。10 が原作、0 で除去します。", false},
    // The game's level-of-detail bias (engine/graphics_patch.cpp): Full keeps
    // every model at its finest mesh, Low and Lowest move each to a coarser
    // one sooner. Lowest is the community "model LOD 2" (-8% GPU, models
    // visibly simpler up close).
    {"model_detail", "モデルの精細度", {"Full", "Normal", "Low", "Lowest"}, 1,
     "標準は原作の設定です。低・最低では簡略モデルへの切り替えを早め、最高では切り替えません。", false},
    // The render resolution. res_width and res_height are two int32s the eboot
    // never writes, so this is a data patch and every pass scales together -
    // see engine/graphics_patch.cpp. The game's pick list shows six entries
    // and scrolls through more. In order of size within each shape - 16:9,
    // then 16:10, 4:3, 21:9 and 32:9 - and saved by name, not by place, so a
    // saved entry survives the order changing (engine/option_menu.cpp's
    // kResolutions and host/launcher.cpp's labels follow this list). The
    // other shapes are engine/live_resolution.cpp's. 960x600 and 1024x640 are
    // the Deck's screen at 75% and 80%: it holds 60 fps there (pixel-bound at
    // 1280x800, ~45), and the presenter upscales by FSR 1.
    {"resolution", "解像度", {"640x360", "960x540", "1280x720", "1440x810", "1600x900",
                                  "1920x1080", "2560x1440", "2880x1620", "3200x1800", "3840x2160",
                                  "960x600", "1024x640", "1280x800", "1920x1200", "2560x1600",
                                  "1280x960", "1440x1080", "2560x1080", "3440x1440", "5120x2160",
                                  "3840x1080", "5120x1440"}, 5,
     "ゲームの描画解像度です。原作は 1920x1080。横長と 16:10 画面にも対応し、HUD は 16:9 を維持します。大きな解像度はメモリー確保のため再起動が必要な場合があります。",
     true},
    // The shaders' form on the GPU (host/translation_cache.h): HLSL is the
    // lifted shaders printed as HLSL and compiled by DXC on this machine,
    // taken from the cache while the rest compile behind the game; Lifted is
    // the lifter's own SPIR-V alone. The same pictures either way.
    {"shaders", "シェーダー", {"HLSL", "Lifted"}, 0,
     "再起動が必要です。HLSL はこの PC の DXC でバックグラウンド処理し、次回起動時に使用します。変換済み SPIR-V は独自変換のみです。",
     true},
    {"skip_logos", "企業ロゴをスキップ", {"Off", "On"}, 0,
     "タイトル前の３つのロゴと、前回「ゲーム終了」を使わなかった場合の警告を省略します。", true},
    // The old switch for the developers' debug menu, which the Debug Menu
    // plugin replaced (engine/debug_menu.h): no row shows it, and an "On" an
    // older bbhost saved turns the plugin on once (main.cpp).
    {"debug_menu", "デバッグメニュー", {"Off", "On"}, 0, "デバッグメニュープラグインに置き換わりました。", true},
    // Lance McDonald's "Restore Debug Camera" (patches/debug-camera.toml). Its
    // code takes the place of a debug-only load-test step, so the debug menu's
    // LOAD TEST and DUNGEON MOVEMAP TEST crash while it is on.
    {"debug_camera", "デバッグカメラ", {"Off", "On"}, 0,
     "再起動が必要です。調べる (E) を押しながら L3 (左 Ctrl) で自由カメラのモードを切り替えます。", true},
    // The PC enhancements: additions to the game itself, each read once at
    // start (main.cpp hands them to config_set_enhancements), all on unless
    // bbhost.toml's [world] says otherwise. Off, all three, is the game as it
    // shipped. (The Old Hunters, a fourth in v0.2.9, is the patch old-hunters.)
    {"change_appearance", "狩人の夢の鏡", {"On", "Off"}, 0,
     "再起動が必要です。狩人の夢の鏡で未使用の容姿編集画面を開きます。", true},
    {"rebirth", "祭壇で生まれ変わる", {"On", "Off"}, 0,
     "再起動が必要です。ヤーナムの石があれば、嘆きの祭壇でレベルと能力値を振り直せます。", true},
    {"five_players", "５人プレイ", {"On", "Off"}, 0,
     "再起動が必要です。メンシス・高楼・辺境・DLCで侵入者を２人に増やします。全参加者の有効化が必要です。", true},
};

const Row g_rows[] = {
    {Row::Header, "画面"},
    {Row::Option, nullptr, kWindowMode},
    {Row::Option, nullptr, kVsync},
    {Row::Option, nullptr, kFrameCap},
    {Row::Option, nullptr, kFpsCounter},
    {Row::Option, nullptr, kMemoryCounter},
    {Row::Option, nullptr, kFov},
    {Row::Option, nullptr, kCameraDistance},
    {Row::Option, nullptr, kCameraHeight},
    {Row::Header, "入力"},
    {Row::Option, nullptr, kMouseMenu},
    {Row::Option, nullptr, kMouseCamera},
    {Row::Option, nullptr, kMouseSens},
    {Row::Option, nullptr, kMouseInvertX},
    {Row::Option, nullptr, kMouseInvertY},
    {Row::Option, nullptr, kMouseAutoRotation},
    {Row::Option, nullptr, kDrawCursor},
    {Row::Option, nullptr, kButtonPrompts},
    {Row::Header, "グラフィックス"},
    {Row::Option, nullptr, kSsao},
    {Row::Option, nullptr, kMotionBlur},
    {Row::Option, nullptr, kAntiAlias},
    {Row::Option, nullptr, kDepthOfField},
    {Row::Option, nullptr, kChromaticAberration},
    {Row::Option, nullptr, kBloom},
    {Row::Option, nullptr, kVignette},
    {Row::Option, nullptr, kShadowDistance},
    {Row::Option, nullptr, kAoStrength},
    {Row::Option, nullptr, kSaturation},
    {Row::Option, nullptr, kFog},
    {Row::Option, nullptr, kModelDetail},
    {Row::Option, nullptr, kResolution},
    {Row::Option, nullptr, kShaders},
    {Row::Header, "PC 追加機能"},
    {Row::Option, nullptr, kChangeAppearance},
    {Row::Option, nullptr, kRebirth},
    {Row::Option, nullptr, kFivePlayers},
    {Row::Header, "起動"},
    {Row::Option, nullptr, kSkipLogos},
    {Row::Option, nullptr, kDebugCamera},
    // The private server's account. No passwords: link this PC to
    // the account signed into on the website with Discord, or make one held
    // by this PC with a recovery code. The token is kept in the options file.
    {Row::Header, "アカウント"},
    {Row::Info, nullptr, -1, 0},
    {Row::Info, nullptr, -1, 1},
    {Row::Action, "Discord と連携", -1, 0},
    {Row::Text, "名前", -1, 0},
    {Row::Action, "この PC でアカウントを作成", -1, 1},
    {Row::Text, "復旧コード", -1, 1},
    {Row::Action, "アカウントを復旧", -1, 2},
    {Row::Action, "ウェブサイトのログインコード", -1, 3},
    {Row::Action, "ログアウト", -1, 4},
    {Row::Header, "更新"},
    {Row::Info, nullptr, -1, 10},
    {Row::Info, nullptr, -1, 11},
    {Row::Action, "更新を確認", -1, 10},
    {Row::Action, "リリースページを開く", -1, 11},
    {Row::Close, "閉じる"},
};
constexpr int kRowCount = static_cast<int>(sizeof(g_rows) / sizeof(g_rows[0]));

std::atomic<bool> g_open{false};
int g_sel = 1;  // the first selectable row
// Where each row was drawn last frame, so the pointer hit-tests what it sees
// rather than what the layout would be at some other display size.
float g_row_y[kRowCount] = {};
float g_row_h = 0.0f, g_row_x = 0.0f, g_row_w = 0.0f;
// And where an option row's value was: from just left of its "<" (g_row_vl)
// to its middle (g_row_vm) is the side a click steps back on.
float g_row_vl[kRowCount] = {}, g_row_vm[kRowCount] = {};
bool g_laid_out = false;
bool g_dirty = false;  // something changed since the last save

// What every setting starts with: the table's own indices, taken before
// anything is loaded over them, then whatever bbhost.toml sets for it - the
// frame cap, the logos, and a package's resolution and window mode (the
// Steam Deck kit's) - so the menus' Defaults go back to those.
std::array<int, kSettingCount> g_default_index = [] {
    std::array<int, kSettingCount> a{};
    for (int id = 0; id < kSettingCount; ++id) a[static_cast<std::size_t>(id)] = g_set[id].index;
    return a;
}();

// A resolution change made while the game runs (engine/live_resolution.h)
// asks to be kept, the way Dark Souls III's ApplyScreenSettingJob does: the
// new size at once, then the game's own yes/no dialog, ten seconds, or it
// goes back. While it waits, g_res_prev is the entry it goes back to and the
// file keeps.
int g_res_prev = -1;
bool g_res_reverting = false;  // the change back itself asks nothing

bool selectable(int r) {
    return r >= 0 && r < kRowCount && g_rows[r].kind != Row::Header && g_rows[r].kind != Row::Info;
}

// Whether the resolution entry `index` changes while the game runs. One past
// what this run's render-target heap was grown for at its start (5120x2160
// after a smaller start) applies on the next run like any restart setting.
bool res_live(int index) {
    unsigned w = 0, h = 0;
    return live_resolution_available() && std::sscanf(g_set[kResolution].values[static_cast<std::size_t>(index)], "%ux%u", &w, &h) == 2 &&
           live_resolution_can(w, h);
}

// The account rows. The fields are typed through the window's text entry
// (host_text_entry_*), one at a time; the calls run on their own thread so
// the screen keeps drawing, and their outcome is the two status lines.
std::string g_field[2];       // name, recovery code (never saved)
int g_edit_field = -1;        // which field the text entry is on, -1 none
std::mutex g_acct_mu;
std::string g_acct_status;    // the last outcome, for the first status line
std::string g_acct_detail;    // a code the player needs (link, recovery, website)
std::atomic<bool> g_acct_busy{false};
std::atomic<bool> g_acct_save{false};    // signed in or out: the token wants saving
std::atomic<bool> g_link_cancel{false};  // stops a link that is waiting for approval

void set_account_lines(const std::string& status, const std::string& detail) {
    std::lock_guard<std::mutex> lk(g_acct_mu);
    g_acct_status = status;
    g_acct_detail = detail;
}

std::string signed_in_line() {
    // The game read its NpId when it went online, so a sign-in after that
    // names the account to the server but not to the game until the next run.
    if (net::account_refused())
        return "ログイン中: " + net::account_name() + " - 認証が失効または更新されました。この PC を再連携してください";
    return "ログイン中: " + net::account_name() +
           (hle_np_context_started() ? " （次回起動で完全に反映）" : "");
}

// Link with Discord: a code the player approves on the website's /link page,
// signed in there with Discord; this thread polls until it is approved,
// denied or expired.
void link_thread() {
    net::DeviceLink link;
    std::string err;
    if (!net::account_link_start(link, err)) {
        set_account_lines("連携に失敗: " + err, "");
        g_acct_busy.store(false);
        return;
    }
    const std::string uri = link.verification_uri + "?code=" + link.user_code;
    set_account_lines("ウェブサイトでの承認を待っています...",
                      "コード " + link.user_code + "  -  承認先: " + link.verification_uri);
    SDL_OpenURL(uri.c_str());
    int interval = link.interval;
    for (int waited = 0; waited < link.expires_in && !g_link_cancel.load();) {
        SDL_Delay(static_cast<Uint32>(interval) * 1000u);
        waited += interval;
        const std::string status = net::account_link_poll(link, err);
        if (status == "pending") continue;
        if (status == "slow_down") {
            ++interval;
            continue;
        }
        if (status == "approved") {
            set_account_lines(signed_in_line(), "");
            g_acct_save.store(true);
        } else {
            set_account_lines(status == "denied" ? "ウェブサイトで連携が拒否されました"
                              : status == "expired" ? "コードが期限切れです。もう一度連携してください"
                                                    : "連携に失敗: " + err, "");
        }
        g_acct_busy.store(false);
        return;
    }
    set_account_lines(g_link_cancel.load() ? "連携をキャンセルしました" : "コードが期限切れです。もう一度連携してください", "");
    g_acct_busy.store(false);
}

// The UPDATES buttons, and whether each can do anything right now.
void update_action(int which) {
    if (which == 10) updater::check_now();
    if (which == 11) updater::open_release_page();
}

bool action_idle(int which) {
    if (which < 10) return !g_acct_busy.load();
    if (updater::busy()) return false;
    if (which == 11) return updater::update_available();
    return true;
}

void account_action(int which, const std::string& name, const std::string& code) {
    if (which >= 10) {
        update_action(which);
        return;
    }
    if (which == 4) {  // sign out, also cancelling a link in progress
        g_link_cancel.store(true);
        std::thread([] {
            net::account_logout();
            set_account_lines("ログアウトしました", "");
            g_acct_save.store(true);
        }).detach();
        return;
    }
    if (g_acct_busy.load()) return;
    if ((which == 1 || which == 2) && name.empty()) {
        set_account_lines("先に名前を入力してください", "");
        return;
    }
    if (which == 2 && code.empty()) {
        set_account_lines("先に復旧コードを入力してください", "");
        return;
    }
    if (which == 3 && !net::account_logged_in()) {
        set_account_lines("先にログインしてください", "");
        return;
    }
    g_acct_busy.store(true);
    g_link_cancel.store(false);
    if (which == 0) {
        set_account_lines("サーバーにコードを要求中...", "");
        std::thread(link_thread).detach();
        return;
    }
    set_account_lines(which == 1 ? "アカウントを作成中..." : which == 2 ? "復旧中..." : "コードを要求中...", "");
    std::thread([which, name, code] {
        std::string err, out;
        bool ok = false;
        if (which == 1) {
            ok = net::account_create(name, out, err);
        } else if (which == 2) {
            ok = net::account_recover(name, code, out, err);
        } else {
            ok = net::account_web_code(out, err);
        }
        if (!ok) {
            // A server without accounts says so itself; anything else is the
            // server's refusal (a name taken, a daily limit) in its words.
            set_account_lines(err.find("アカウントサービスがありません") != std::string::npos ? err : "拒否されました: " + err, "");
        } else if (which == 3) {
            set_account_lines(signed_in_line(), "ウェブ用コード " + out + "  -  10分以内にアカウントページで使用してください");
        } else {
            // The recovery code is shown here once and never written anywhere.
            set_account_lines(signed_in_line(), "復旧コード " + out + "  -  一度しか表示されません。今すぐ控えてください");
            g_field[1].clear();
            g_acct_save.store(true);
        }
        g_acct_busy.store(false);
    }).detach();
}

std::string account_status() {
    std::lock_guard<std::mutex> lk(g_acct_mu);
    if (!g_acct_status.empty()) return g_acct_status;
    return net::account_logged_in() ? "ログイン中: " + net::account_name() : "未ログイン";
}

std::string account_detail() {
    std::lock_guard<std::mutex> lk(g_acct_mu);
    return g_acct_detail;
}

void edit_begin(int field) {
    // The window has one text entry. Taken from the game's own box while that
    // is open, it ended the box with this field's text - or, closed with the
    // screen, left the game waiting on a box nobody could type into again.
    if (g_edit_field < 0 && host_text_entry_open()) {
        set_account_lines("先にゲームの文字入力を終了してください (Enter または Escape)", "");
        return;
    }
    g_edit_field = field;
    host_text_entry_begin(g_field[field].c_str(), field == 0 ? 16 : 48, false, field == 0 ? "アカウント名" : "復旧コード");
}

// Called every frame the screen draws: takes the typed text, ends the entry
// when the player did.
void edit_poll() {
    if (g_edit_field < 0) return;
    std::string text;
    const int r = host_text_entry_poll(text);
    if (r == 0) {
        g_field[g_edit_field] = text;
        return;
    }
    if (r == 1) g_field[g_edit_field] = text;
    host_text_entry_end();
    g_edit_field = -1;
}

// The frame-rate values are the labels themselves: "Uncapped" is 0, anything
// else is the number.
int fps_from_label(const char* v) { return std::atoi(v); }

int fps_index(int fps) {
    for (std::size_t v = 0; v < g_set[kFrameCap].values.size(); ++v) {
        if (fps_from_label(g_set[kFrameCap].values[v]) == fps) {
            return static_cast<int>(v);
        }
    }
    return -1;
}

std::string options_path() {
    // BBHOST_OPTIONS_PATH: a file of its own, so a scripted test that changes
    // settings (tools/menu_drive.sh) never writes the player's.
    if (const char* e = std::getenv("BBHOST_OPTIONS_PATH"); e && e[0]) {
        return e;
    }
    // The per-user folder (config_user_dir), beside the per-user config: the
    // F10 settings and the account outlive any copy of bbhost. A --config is a
    // profile with a file of its own (bbhost-options-<name>.toml), so a dev
    // server's account never mixes with the live one's. The file from before
    // (beside that config, or the config a package or tree used, or in the
    // working directory) is copied over the first time: nobody is signed out.
    static const std::string path = [] {
        const HostConfig& c = config();
        const std::string mine = config_user_dir() + "/" +
                                 (c.profile.empty() ? std::string("bbhost-options.toml") : "bbhost-options-" + c.profile + ".toml");
        std::ifstream have(mine);
        if (have) return mine;
        const auto beside = [](const std::string& f) {
            const std::size_t slash = f.find_last_of("/\\");
            return (slash == std::string::npos ? std::string() : f.substr(0, slash + 1)) + "bbhost-options.toml";
        };
        std::vector<std::string> old;
        if (!c.profile.empty()) {
            old.push_back(beside(c.profile_config));
        } else {
            for (const char* f : {"bbhost.toml"}) old.push_back(beside(f));
            old.push_back(beside(config_exe_dir() + "/bbhost.toml"));
        }
        // A folder of its own (BBHOST_CONFIG_DIR: every test harness) takes the
        // settings but not the [account]: one old file beside several test
        // configs signed each of their profiles in as that player, with a
        // token long replaced, and the server refused them all day.
        const char* own_dir = std::getenv("BBHOST_CONFIG_DIR");
        const bool keep_account = !(own_dir && own_dir[0]);
        for (const std::string& f : old) {
            if (f == mine) continue;
            std::ifstream in(f, std::ios::binary);
            if (!in) continue;
            std::ofstream out(mine, std::ios::binary);
            bool in_account = false, dropped = false;
            for (std::string line; std::getline(in, line);) {
                if (!line.empty() && line[0] == '[') in_account = line.rfind("[account]", 0) == 0;
                if (in_account && !keep_account) {
                    dropped = true;
                    continue;
                }
                out << line << '\n';
            }
            if (out) host_log("options: copied %s to %s (the per-user settings every copy of bbhost reads)%s", f.c_str(),
                              mine.c_str(), dropped ? ", not its account (a folder of its own)" : "");
            break;
        }
        return mine;
    }();
    return path;
}

// Applied every time one changes, and once at load, so there is one path to
// the live state instead of a copy in each caller.
void apply(int id) {
    const Setting& s = g_set[id];
    switch (id) {
        case kWindowMode: host_window_set_fullscreen(s.index == 1); break;
        case kVsync: host_window_set_vsync(s.index == 0); break;
        case kFrameCap:
            hle_video_set_fps_cap(fps_from_label(s.values[static_cast<std::size_t>(s.index)]));
            break;
        case kMouseMenu:
        case kMouseCamera:
        case kMouseSens:
        case kMouseInvertX:
        case kMouseInvertY:
        case kMouseAutoRotation:
        case kDrawCursor:
        case kButtonPrompts:
            break;  // the settings object carries them (host/settings.h)
        case kSsao:
        case kMotionBlur:
        case kAntiAlias:
        case kDepthOfField:
        case kChromaticAberration:
        case kResolution: {
            unsigned w = 0, h = 0;
            // A change the player made asks to be kept (g_res_prev set by
            // set_index); going back, and the start, ask nothing. BBHOST_RES
            // wins here too, or applying the settings at the start undid it.
            const char* forced = std::getenv("BBHOST_RES");
            if ((forced && forced[0] && std::sscanf(forced, "%ux%u", &w, &h) == 2) ||
                std::sscanf(s.values[static_cast<std::size_t>(s.index)], "%ux%u", &w, &h) == 2)
                live_resolution_request(w, h, g_res_prev >= 0 && !g_res_reverting);
            break;
        }
        case kSkipLogos:
        case kDebugMenu:
        case kDebugCamera:
        case kChangeAppearance:
        case kRebirth:
        case kFivePlayers:
            break;  // read by the loader before any of this exists
        default: break;
    }
}

// The file's format. 2: Mouse in menus defaults to On. A file without a
// version was written by a build where it defaulted to Off, and every save
// writes every setting - so its "Off" is that old default far more often
// than a choice, and reads as On (once: the next save writes version 2).
// 3: frame_cap is the game's frame rate (30, 60, 90, Uncapped; 120 since
// 2026-10-10, a new value, so the version stays); an older file's 90, 120,
// 144 and Off were caps on presentation that ran the game at 60, and read as
// 60. camera_tuning (2026-10-10) is a new key whose default, FPS++, is what
// every older build ran, so a file without it reads as before and the
// version stays.
constexpr int kOptionsVersion = 3;

void save() {
    if (!g_dirty) {
        return;
    }
    const std::string path = options_path();
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        host_log("options: cannot write %s", path.c_str());
        return;
    }
    std::fprintf(f, "# bbhost PC port options. Written by the options screen (F10).\n");
    std::fprintf(f, "[options]\n");
    std::fprintf(f, "version = \"%d\"\n", kOptionsVersion);
    for (const Setting& s : g_set) {
        // A resolution still waiting for its keep-or-revert answer is not
        // saved: the file keeps the one it would go back to.
        const int index = &s == &g_set[kResolution] && g_res_prev >= 0 ? g_res_prev : s.index;
        std::fprintf(f, "%s = \"%s\"\n", s.key, s.values[static_cast<std::size_t>(index)]);
    }
    host_bindings_save(f);
    // The account: its name and token.
    if (net::account_logged_in()) {
        std::fprintf(f, "[account]\nname = \"%s\"\ntoken = \"%s\"\n", net::account_name().c_str(),
                     net::account_token().c_str());
    }
    std::fclose(f);
    g_dirty = false;
    host_log("options: saved %s", path.c_str());
}

// The settings object (host/settings.h): every field from the table, scaled
// the way its consumer wants it, published again whenever a setting changes
// (its serial is host_opt_serial) and told to the listeners. Made on first
// use, so a listener may register from anywhere at any time, and never
// destroyed: threads still reading it while the process exits read memory
// that is still there.
HostSettingsStore& settings_store() {
    static HostSettingsStore& store = *new HostSettingsStore;
    return store;
}

bool on_of(int id) { return reads_on(g_set[id], g_set[id].index); }
int index_of(int id) { return g_set[id].index; }

void rebuild_settings() {
    HostSettings h;
    h.fullscreen = index_of(kWindowMode) == 1;
    h.vsync = index_of(kVsync) == 0;
    h.frame_cap = fps_from_label(g_set[kFrameCap].values[static_cast<std::size_t>(index_of(kFrameCap))]);
    h.fps_counter = on_of(kFpsCounter);
    h.memory_counter = on_of(kMemoryCounter);
    {
        // BBHOST_RES=WxH still wins, so an A/B run started with one behaves
        // the way its command line says.
        unsigned w = 0, hh = 0;
        const char* e = std::getenv("BBHOST_RES");
        if (!(e && e[0] && std::sscanf(e, "%ux%u", &w, &hh) == 2)) {
            const Setting& r = g_set[kResolution];
            if (std::sscanf(r.values[static_cast<std::size_t>(r.index)], "%ux%u", &w, &hh) != 2) w = hh = 0;
        }
        if (w && hh) {
            h.res_width = static_cast<int>(w);
            h.res_height = static_cast<int>(hh);
        }
    }
    // Distance and height: 5 is the game's own, 10% a step either way. Field
    // of view: 0 is the game's own, 5% wider a step (BBHOST_FOV overrides).
    const auto around = [](int id) { return 0.5f + 0.1f * static_cast<float>(index_of(id)); };
    h.camera_distance_scale = around(kCameraDistance);
    h.camera_height_scale = around(kCameraHeight);
    h.camera_original = index_of(kCameraTuning) == 1;
    {
        const char* e = std::getenv("BBHOST_FOV");
        const float env = e && *e ? std::strtof(e, nullptr) : 0.0f;
        const int fov = index_of(kFov);
        h.fov_scale = env > 0.0f ? env : fov <= 0 ? 1.0f : 1.0f + 0.05f * static_cast<float>(fov);
    }
    h.mouse_menu = on_of(kMouseMenu);
    h.mouse_camera = on_of(kMouseCamera);
    // DS3's mouse sensitivity (engine/mouse_camera.h): the same 0..10 and the
    // same default, and the same angle a count at each step.
    h.mouse_sens = index_of(kMouseSens) < 0 || index_of(kMouseSens) > 10 ? 5 : index_of(kMouseSens);
    h.mouse_invert_x = on_of(kMouseInvertX);
    h.mouse_invert_y = on_of(kMouseInvertY);
    h.mouse_auto_rotation = on_of(kMouseAutoRotation);
    h.draw_cursor = on_of(kDrawCursor);
    h.button_prompts = index_of(kButtonPrompts) < 0 ? 0 : index_of(kButtonPrompts);
    h.ssao = on_of(kSsao);
    h.motion_blur = on_of(kMotionBlur);
    h.anti_alias = on_of(kAntiAlias);
    h.depth_of_field = on_of(kDepthOfField);
    h.chromatic_aberration = on_of(kChromaticAberration);
    h.vignette = on_of(kVignette);
    h.bloom = static_cast<float>(index_of(kBloom)) / 10.0f;
    h.saturation = static_cast<float>(index_of(kSaturation)) / 5.0f;  // 5 the area's own
    h.fog = static_cast<float>(index_of(kFog)) / 10.0f;
    h.ao_strength = static_cast<float>(index_of(kAoStrength)) / 5.0f;
    h.shadow_scale = 1.0f + 0.2f * static_cast<float>(index_of(kShadowDistance));
    {
        static constexpr int kBias[] = {-2, 0, 1, 2};
        const int i = index_of(kModelDetail);
        h.lod_bias = i >= 0 && i < 4 ? kBias[i] : 0;
    }
    h.skip_logos = on_of(kSkipLogos);
    h.debug_menu = on_of(kDebugMenu);
    h.debug_camera = on_of(kDebugCamera);
    h.change_appearance = on_of(kChangeAppearance);
    h.rebirth = on_of(kRebirth);
    h.five_players = on_of(kFivePlayers);
    settings_store().publish(h);
}

void set_index(int id, int index, bool from_file) {
    Setting& s = g_set[id];
    const int n = static_cast<int>(s.values.size());
    index = ((index % n) + n) % n;
    if (index == s.index && from_file) {
        return;
    }
    if (id == kResolution && !from_file && !g_res_reverting && index != s.index && res_live(index)) {
        if (g_res_prev < 0) g_res_prev = s.index;  // a second change while one waits goes back to the first's start
        if (index == g_res_prev) g_res_prev = -1;  // back where it started: nothing to ask
    }
    s.index = index;
    rebuild_settings();
    if (!from_file) {
        g_dirty = true;
    }
    apply(id);
}

void res_keep() {
    if (g_res_prev < 0) return;
    host_log("options: resolution %s kept", g_set[kResolution].values[static_cast<std::size_t>(g_set[kResolution].index)]);
    g_res_prev = -1;
    g_dirty = true;
    save();
}

void res_revert() {
    if (g_res_prev < 0) return;
    const int prev = g_res_prev;
    g_res_prev = -1;
    host_log("options: resolution back to %s", g_set[kResolution].values[static_cast<std::size_t>(prev)]);
    g_res_reverting = true;
    set_index(kResolution, prev, false);
    g_res_reverting = false;
    // An entry waiting for the next start (res_live) cannot be gone back to
    // now: the picture goes back to the size it had before the change.
    if (!res_live(prev)) live_resolution_go_back();
    save();
}

}  // namespace



void host_options_apply() {
    for (int id = 0; id < kSettingCount; ++id) {
        apply(id);
    }
}

void host_options_load() {
    // Once: the setup window (host/launcher.h) loads them before it shows them.
    static bool loaded = false;
    if (loaded) return;
    loaded = true;
    // bbhost.toml still sets the defaults; the options file is what the player
    // changed, and the environment switches still win over both.
    host_bindings_load_defaults();
    g_set[kSkipLogos].index = config().skip_intro ? 1 : 0;
    // The PC enhancements seed from bbhost.toml's [world] (all on unless it
    // says otherwise); the options file then holds the player's choice, and a
    // BBHOST_* switch still wins over both (config_set_enhancements).
    g_set[kChangeAppearance].index = config().change_appearance ? 0 : 1;
    g_set[kRebirth].index = config().rebirth ? 0 : 1;
    g_set[kFivePlayers].index = config().five_players ? 0 : 1;
    // video.fps_cap is the frame rate's default: 30, 60, 90, 120 or 0
    // (uncapped); another number takes the nearest rate below it.
    {
        const int cap = config().fps_cap;
        const int mode = cap <= 0 ? 0 : cap >= 120 ? 120 : cap >= 90 ? 90 : cap >= 60 ? 60 : 30;
        if (mode != cap) host_log("options: video.fps_cap = %d is not a frame rate; %d", cap, mode);
        g_set[kFrameCap].index = std::max(0, fps_index(mode));
    }
    // A package's own (video.resolution, video.window_mode: the Steam Deck
    // kit's 1280x800, fullscreen; video.model_detail), by the entry's name.
    const auto seed = [](int id, const std::string& value) {
        for (std::size_t k = 0; !value.empty() && k < g_set[id].values.size(); ++k)
            if (value == g_set[id].values[k]) g_set[id].index = static_cast<int>(k);
    };
    seed(kResolution, config().resolution);
    seed(kWindowMode, config().window_mode);
    seed(kModelDetail, config().model_detail);
    seed(kShaders, config().shaders);
    for (const int id : {static_cast<int>(kSkipLogos), static_cast<int>(kFrameCap), static_cast<int>(kResolution), static_cast<int>(kWindowMode),
                         static_cast<int>(kModelDetail), static_cast<int>(kShaders), static_cast<int>(kChangeAppearance),
                         static_cast<int>(kRebirth), static_cast<int>(kFivePlayers)})
        g_default_index[static_cast<std::size_t>(id)] = g_set[id].index;
    const std::string path = options_path();
    FILE* f = std::fopen(path.c_str(), "r");
    if (f) {
        char line[512];
        bool keys = false;  // in [keys]: bindings, not settings
        bool account = false;  // in [account]: the name and token
        std::string acct_name, acct_token;
        int file_version = 1;
        bool old_hunters_off = false;  // v0.2.9's PC enhancement, turned off
        std::string frame_cap_val;
        while (std::fgets(line, sizeof(line), f)) {
            if (line[0] == '[') {
                keys = std::strncmp(line, "[keys]", 6) == 0;
                account = std::strncmp(line, "[account]", 9) == 0;
                continue;
            }
            char* eq = std::strchr(line, '=');
            if (!eq || line[0] == '#') {
                continue;
            }
            *eq = 0;
            std::string key(line), val(eq + 1);
            auto trim = [](std::string& v) {
                while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ' ||
                                      v.back() == '\t' || v.back() == '"')) {
                    v.pop_back();
                }
                std::size_t i = 0;
                while (i < v.size() && (v[i] == ' ' || v[i] == '\t' || v[i] == '"')) {
                    ++i;
                }
                v.erase(0, i);
            };
            trim(key);
            trim(val);
            if (keys) {
                host_bindings_parse(key.c_str(), val.c_str());
                continue;
            }
            if (account) {
                if (key == "name") acct_name = val;
                if (key == "token") acct_token = val;
                continue;
            }
            if (key == "version") {
                file_version = std::atoi(val.c_str());
                continue;
            }
            if (key == "old_hunters") {
                old_hunters_off = val == "無効";
                continue;
            }
            // Looked at again once the file's version is known (below).
            if (key == "frame_cap") frame_cap_val = val;
            // The four named steps mouse_sens used to have, on the new scale.
            if (key == "mouse_sens") {
                const char* old[] = {"Low", "Medium", "High", "Very high"};
                const int to[] = {2, 5, 7, 9};
                for (int k = 0; k < 4; ++k) {
                    if (val == old[k]) val = std::to_string(to[k]);
                }
            }
            for (int id = 0; id < kSettingCount; ++id) {
                if (key != g_set[id].key) {
                    continue;
                }
                for (std::size_t v = 0; v < g_set[id].values.size(); ++v) {
                    if (val == g_set[id].values[v]) {
                        g_set[id].index = static_cast<int>(v);
                    }
                }
            }
        }
        std::fclose(f);
        // Before version 3 the frame_cap was a cap on presentation, and its 90,
        // 120, 144 and Off ran the game at 60: the frame rate that ran is 60
        // (before any save below writes version 3).
        if (file_version < 3 &&
            (frame_cap_val == "90" || frame_cap_val == "120" || frame_cap_val == "144" || frame_cap_val == "Off")) {
            g_set[kFrameCap].index = std::max(0, fps_index(60));
            g_dirty = true;
            host_log("options: frame_cap \"%s\" (a cap that ran the game at 60) reads as the frame rate 60",
                     frame_cap_val.c_str());
        }
        if (!acct_name.empty() && !acct_token.empty()) {
            net::account_set(acct_name, acct_token);
            g_field[0] = acct_name;
            host_log("options: account %s from %s", acct_name.c_str(), path.c_str());
        }
        // After the account: save() writes everything it holds, and saving
        // before the account was set dropped the [account] section - a
        // dev-server tester was signed out (2026-10-02).
        if (file_version < 2 && g_set[kMouseMenu].index == 0) {
            g_set[kMouseMenu].index = 1;
            g_dirty = true;
            host_log("options: Mouse in menus read as On - this file predates it defaulting to On");
            save();  // as version 2, so a later Off is the player's and stays
        }
        // The Old Hunters was a PC enhancement in v0.2.9 and is the patch
        // old-hunters since. A player who turned it off keeps it off: the
        // per-user bbhost.toml's [patches] says so from now on, and the next
        // save drops the old key (unless [patches] already says something).
        if (old_hunters_off) {
            bool carried = !config_value("patches.old-hunters").empty();
            if (!carried) {
                const std::string user = config_user_file();
                carried = config_set_values(user, {{"patches", "old-hunters", "false"}});
                if (carried) {
                    config_value_set("patches.old-hunters", "false");
                    host_log("options: The Old Hunters was off here; it is the patch old-hunters now, off under [patches] in %s",
                             user.c_str());
                }
            }
            if (carried) {
                g_dirty = true;
                save();
            }
        }
        host_log("options: read %s", path.c_str());
    }
    // BBHOST_ACCOUNT=name:token, for a scripted run that must not touch the file.
    if (const char* e = std::getenv("BBHOST_ACCOUNT"); e && e[0]) {
        const std::string v = e;
        if (const std::size_t c = v.find(':'); c != std::string::npos) {
            net::account_set(v.substr(0, c), v.substr(c + 1));
            g_field[0] = v.substr(0, c);
        }
    }
    // The environment switches the track was brought up with still win, so a
    // run started with one behaves the way it says on the command line.
    if (const char* e = std::getenv("BBHOST_MOUSE_MENU"); e && e[0]) {
        g_set[kMouseMenu].index = e[0] == '0' ? 0 : 1;
    }
    // The post-processing switches, for A/B runs: BBHOST_SSAO=0 and friends.
    for (const auto& [env, id] : {std::pair<const char*, int>{"BBHOST_SSAO", kSsao},
                                  {"BBHOST_MOTION_BLUR", kMotionBlur},
                                  {"BBHOST_ANTI_ALIAS", kAntiAlias},
                                  {"BBHOST_DEPTH_OF_FIELD", kDepthOfField},
                                  {"BBHOST_CHROMATIC_ABERRATION", kChromaticAberration}}) {
        if (const char* e = std::getenv(env); e && e[0]) {
            g_set[id].index = e[0] == '0' ? 1 : 0;  // these are {"On","Off"}
        }
    }
    if (const char* e = std::getenv("BBHOST_MOUSE_CAMERA"); e && e[0]) {
        g_set[kMouseCamera].index = e[0] == '0' ? 0 : 1;
    }
    if (const char* e = std::getenv("BBHOST_MOUSE_SENS"); e && e[0]) {
        const int v = std::atoi(e);
        g_set[kMouseSens].index = v < 0 ? 0 : v > 10 ? 10 : v;
    }
    if (const char* e = std::getenv("BBHOST_MOUSE_CURSOR"); e && e[0]) {
        g_set[kDrawCursor].index = e[0] == '0' ? 1 : 0;
    }
    host_options_apply();
    rebuild_settings();
}

bool host_options_open() { return g_open.load(std::memory_order_relaxed); }
bool host_options_editing() { return g_edit_field >= 0; }

void host_options_set_open(bool open) {
    if (open == g_open.load(std::memory_order_relaxed)) {
        return;
    }
    g_open.store(open, std::memory_order_relaxed);
    if (open) {
        // Nothing has consumed the pointer while the screen was closed, so a
        // click from minutes ago is still latched. Drop it: the first click
        // the screen sees must be one aimed at it.
        host_mouse_state();
        g_laid_out = false;
    } else {
        if (g_edit_field >= 0) {
            host_text_entry_end();
            g_edit_field = -1;
        }
        save();
    }
    host_log("options: %s", open ? "open" : "closed");
}

bool host_options_key(int scancode, bool repeat) {
#if defined(BBHOST_HAVE_SDL3)
    if (scancode == SDL_SCANCODE_F10) {
        if (!repeat) {
            host_options_set_open(!host_options_open());
        }
        return true;
    }
    if (!host_options_open()) {
        return false;
    }
    // A field being typed: the window's text entry takes the keys (Enter
    // accepts, Escape cancels); the poll in the frame picks the result up.
    if (g_edit_field >= 0) {
        return false;
    }
    const int step = scancode == SDL_SCANCODE_UP ? -1 : scancode == SDL_SCANCODE_DOWN ? 1 : 0;
    if (step) {
        for (int i = 0; i < kRowCount; ++i) {
            g_sel = (g_sel + step + kRowCount) % kRowCount;
            if (selectable(g_sel)) {
                break;
            }
        }
        return true;
    }
    const int id = g_rows[g_sel].setting;
    if (scancode == SDL_SCANCODE_LEFT && id >= 0) {
        set_index(id, g_set[id].index - 1, false);
        return true;
    }
    if (scancode == SDL_SCANCODE_RIGHT && id >= 0) {
        set_index(id, g_set[id].index + 1, false);
        return true;
    }
    if (scancode == SDL_SCANCODE_RETURN || scancode == SDL_SCANCODE_KP_ENTER ||
        scancode == SDL_SCANCODE_SPACE) {
        if (g_rows[g_sel].kind == Row::Close) {
            host_options_set_open(false);
        } else if (g_rows[g_sel].kind == Row::Text) {
            edit_begin(g_rows[g_sel].field);
        } else if (g_rows[g_sel].kind == Row::Action) {
            account_action(g_rows[g_sel].field, g_field[0], g_field[1]);
        } else if (id >= 0) {
            set_index(id, g_set[id].index + 1, false);
        }
        return true;
    }
    if (scancode == SDL_SCANCODE_ESCAPE) {
        host_options_set_open(false);
        return true;
    }
    // Modal: while it is open it eats every key, so nothing leaks to the pad.
    return true;
#else
    (void)scancode;
    (void)repeat;
    return false;
#endif
}

namespace {
void options_panel_frame(float display_w, float display_h);
}

void host_options_frame(float display_w, float display_h) {
    if (g_acct_save.exchange(false)) {
        g_dirty = true;
        save();
    }
    // BBHOST_RES_PROMPT_TEST=<flip>:<entry>: the Resolution setting changed
    // to that entry at that flip, the way the menus change it (tests).
    static long test_flip = [] {
        const char* e = std::getenv("BBHOST_RES_PROMPT_TEST");
        return e && *e ? std::atol(e) : -1L;
    }();
    if (test_flip >= 0 && static_cast<long>(hle_video_flip_count()) >= test_flip) {
        test_flip = -1;
        const char* e = std::strchr(std::getenv("BBHOST_RES_PROMPT_TEST"), ':');
        if (e) host_opt_set_resolution_index(std::atoi(e + 1));
    }
    // The game's keep-or-revert dialog answered (engine/live_resolution.h).
    if (const int a = live_resolution_take_answer()) {
        if (a == 1) res_keep(); else res_revert();
    }
    if (host_options_open()) options_panel_frame(display_w, display_h);
}

namespace {
void options_panel_frame(float display_w, float display_h) {
    edit_poll();
    // Tall enough for every row: headers take 32 px, rows 40, and the note and
    // key guide under them 100. A fixed 620 had been outgrown by the rows, and
    // then so had 1080 lines - so when the display is shorter than that the
    // pitch closes up to fit (the text keeps its size), rather than the last
    // rows running under the key guide and off the panel.
    // The rows outgrew every display once the account section came, so
    // rather than closing the pitch up until the lines overlap, the
    // list scrolls: the rows that fit are drawn, the selection is kept in
    // view, and a hint says there is more above or below.
    float rows_h = 0.0f;
    for (const Row& r : g_rows) rows_h += r.kind == Row::Header ? 32.0f : 40.0f;
    const float room = display_h - 20.0f - 178.0f;
    const float fit = 1.0f;
    const bool scrolls = rows_h > room && room > 0.0f;
    const float pw = 860.0f, ph = std::fmin(78.0f + rows_h + 100.0f, display_h - 20.0f);
    const float px = (display_w - pw) * 0.5f, py = (display_h - ph) * 0.5f;
    static int g_scroll = 0;  // the first row drawn
    if (!scrolls) {
        g_scroll = 0;
    } else {
        auto height = [](int r) { return g_rows[r].kind == Row::Header ? 32.0f : 40.0f; };
        const float avail = ph - 78.0f - 100.0f - 24.0f;  // less the two hints
        // A header just above the selection comes along, so a section never
        // starts with its title out of view.
        int top = g_sel;
        while (top > 0 && (g_rows[top - 1].kind == Row::Header || g_rows[top - 1].kind == Row::Info)) --top;
        if (g_scroll > top) g_scroll = top;
        for (;;) {
            float h = 0.0f;
            for (int r = g_scroll; r <= g_sel; ++r) h += height(r);
            if (h <= avail || g_scroll >= g_sel) break;
            ++g_scroll;
        }
    }

    // The pointer, before anything is drawn, against last frame's rows - the
    // layout does not move between frames unless the display size does, and
    // hit-testing what was actually drawn is the rule the overlay is for.
    //
    // Hover selects only when the pointer **moves** (or clicks). Taken every
    // frame, a pointer resting on the panel held the selection on its row, and
    // the arrow keys could not move it.
    const MouseState m = host_mouse_state();
    static float last_x = -1.0f, last_y = -1.0f;
    const bool moved = m.x != last_x || m.y != last_y;
    last_x = m.x;
    last_y = m.y;
    if (g_laid_out && m.in_window && (moved || (m.pressed & 3u))) {
        for (int r = 0; r < kRowCount; ++r) {
            if (!selectable(r) || m.x < g_row_x || m.x > g_row_x + g_row_w) {
                continue;
            }
            if (m.y >= g_row_y[r] && m.y < g_row_y[r] + g_row_h) {
                g_sel = r;
                if (m.pressed & 1u) {
                    if (g_rows[r].kind == Row::Close) {
                        host_options_set_open(false);
                        return;
                    }
                    if (g_rows[r].kind == Row::Text) {
                        if (g_edit_field != g_rows[r].field) edit_begin(g_rows[r].field);
                    } else if (g_rows[r].kind == Row::Action) {
                        account_action(g_rows[r].field, g_field[0], g_field[1]);
                    } else if (g_rows[r].setting >= 0) {
                        // By where it lands, as the arrows are drawn: on the
                        // value's "<" half it steps back, on its ">" half
                        // forward. The label steps forward, as Enter does.
                        const bool back = m.x >= g_row_vl[r] && m.x < g_row_vm[r];
                        set_index(g_rows[r].setting, g_set[g_rows[r].setting].index + (back ? -1 : 1), false);
                    }
                } else if (m.pressed & 2u) {
                    if (g_rows[r].kind == Row::Option) {
                        set_index(g_rows[r].setting, g_set[g_rows[r].setting].index - 1, false);
                    }
                }
                break;
            }
        }
    }
    if (m.wheel != 0.0f) {
        const int step = m.wheel > 0.0f ? -1 : 1;
        for (int i = 0; i < kRowCount; ++i) {
            g_sel = (g_sel + step + kRowCount) % kRowCount;
            if (selectable(g_sel)) {
                break;
            }
        }
    }

    // Panel. The game's gold on near-black, the same as the text box, so the
    // port's own surfaces read as one thing rather than three.
    host_overlay_rect(0.0f, 0.0f, display_w, display_h, 0x00000080u);
    host_overlay_rect(px - 3.0f, py - 3.0f, pw + 6.0f, ph + 6.0f, 0xc8a05affu);
    host_overlay_rect(px, py, pw, ph, 0x0a0a0af2u);
    host_overlay_text(px + 28.0f, py + 22.0f, 1.0f, 0xc8a05affu, "BLOODBORNE PC 設定");
    // The online state, always in view whatever the list is scrolled to:
    // who this is signed in as, and the room when there is one.
    {
        const std::string online = hle_np_online_status();
        const float ow = host_overlay_text_width(0.6f, online.c_str());
        host_overlay_text(px + pw - 28.0f - ow, py + 30.0f, 0.6f, 0x9a8a64ffu, online.c_str());
    }
    host_overlay_rect(px + 28.0f, py + 56.0f, pw - 56.0f, 1.0f, 0x6a5a34ffu);

    float y = py + 78.0f;
    g_row_h = 34.0f * fit;
    const float text_dy = (g_row_h - 22.0f) * 0.5f;  // 0.8 text is about 22 px: centred in the row
    g_row_x = px + 16.0f;
    g_row_w = pw - 32.0f;
    const float y_end = py + ph - 100.0f;
    if (scrolls && g_scroll > 0) {
        host_overlay_text(px + 40.0f, y, 0.55f, 0x7a6a44ffu, "↑ 上に項目があります");
        y += 24.0f;
    }
    bool more_below = false;
    for (int r = 0; r < kRowCount; ++r) {
        const Row& row = g_rows[r];
        const float h = (row.kind == Row::Header ? 32.0f : 40.0f) * fit;
        if (r < g_scroll || y + h > y_end - (scrolls ? 24.0f : 0.0f)) {
            g_row_y[r] = -10000.0f;  // not drawn: the pointer cannot land on it
            if (r >= g_scroll) more_below = true;
            continue;
        }
        g_row_y[r] = y;
        if (row.kind == Row::Header) {
            host_overlay_text(px + 28.0f, y + 10.0f * fit, 0.62f, 0x8a7a54ffu, row.text);
            y += 32.0f * fit;
            continue;
        }
        const bool sel = r == g_sel;
        if (sel) {
            host_overlay_rect(g_row_x, y, g_row_w, g_row_h, 0x3a3020ffu);
            host_overlay_rect(g_row_x, y, 3.0f, g_row_h, 0xc8a05affu);
        }
        if (row.kind == Row::Close || row.kind == Row::Action) {
            const bool busy = row.kind == Row::Action && !action_idle(row.field);
            host_overlay_text(px + 40.0f, y + text_dy, 0.8f, busy ? 0x666666ffu : sel ? 0xffffffffu : 0xbbbbbbffu,
                              row.text);
            y += 40.0f * fit;
            continue;
        }
        if (row.kind == Row::Info) {
            const std::string line = row.field == 11  ? updater::detail()
                                     : row.field == 10 ? updater::status()
                                     : row.field == 1  ? account_detail()
                                                       : account_status();
            host_overlay_text(px + 40.0f, y + text_dy + 2.0f, 0.66f, 0x9a9a9affu, line.c_str());
            y += 40.0f * fit;
            continue;
        }
        if (row.kind == Row::Text) {
            host_overlay_text(px + 40.0f, y + text_dy, 0.8f, sel ? 0xffffffffu : 0xbbbbbbffu, row.text);
            std::string shown = g_field[row.field];
            if (g_edit_field == row.field) shown += "_";
            if (shown.empty()) shown = "（ここに入力）";
            const float vw = host_overlay_text_width(0.8f, shown.c_str());
            host_overlay_text(px + pw - 60.0f - vw, y + text_dy, 0.8f,
                              g_edit_field == row.field ? 0xffd88cffu : sel ? 0xffd88cffu : 0x9a9a9affu, shown.c_str());
            y += 40.0f * fit;
            continue;
        }
        const Setting& s = g_set[row.setting];
        host_overlay_text(px + 40.0f, y + text_dy, 0.8f, sel ? 0xffffffffu : 0xbbbbbbffu, s.label);
        const char* v = host_japanese_value(s.values[static_cast<std::size_t>(s.index)]);
        const float vw = host_overlay_text_width(0.8f, v);
        const float vx = px + pw - 60.0f - vw;
        g_row_vl[r] = vx - 30.0f;
        g_row_vm[r] = vx + vw * 0.5f;
        host_overlay_text(vx, y + text_dy, 0.8f, sel ? 0xffd88cffu : 0x9a9a9affu, v);
        if (s.index > 0) {
            host_overlay_text(vx - 22.0f, y + text_dy, 0.8f, sel ? 0xc8a05affu : 0x555555ffu, "<");
        }
        if (s.index + 1 < static_cast<int>(s.values.size())) {
            host_overlay_text(px + pw - 50.0f, y + text_dy, 0.8f, sel ? 0xc8a05affu : 0x555555ffu, ">");
        }
        if (s.restart && !(row.setting == kResolution && res_live(s.index))) {
            host_overlay_text(px + 40.0f + host_overlay_text_width(0.8f, s.label) + 14.0f, y + text_dy + 4.0f,
                              0.52f, 0x7a6a44ffu, "（再起動が必要）");
        }
        y += 40.0f * fit;
    }
    if (more_below) {
        host_overlay_text(px + 40.0f, y, 0.55f, 0x7a6a44ffu, "↓ 下に項目があります");
    }
    g_laid_out = true;

    // The note for whatever is selected, in a fixed place, so the rows above
    // never move as the selection does.
    const int id = g_rows[g_sel].setting;
    const char* note = id >= 0 ? g_set[id].note : "閉じると保存します。";
    if (g_rows[g_sel].kind == Row::Text) note = "Enter またはクリックで入力。Enter で確定、Escape で取り消します。";
    if (g_rows[g_sel].kind == Row::Action) {
        static const char* const kNotes[] = {
            "コードを表示してサイトを開きます。Discord でログインし、承認してください。",
            "上の名前でこの PC のアカウントを作成します。復旧コードを控えてください。",
            "名前と復旧コードで復旧します。他の PC はログアウトされます。",
            "この PC のアカウントでサイトにログインするための使い切りコードです。",
            "認証情報を削除します。認証必須のサーバーではゲームからログアウトします。",
        };
        static const char* const kUpdateNotes[] = {
            "GitHub で最新リリースを確認します。",
            "新しいリリースのダウンロードページをブラウザーで開きます。",
        };
        const int f = g_rows[g_sel].field;
        note = f >= 10 ? kUpdateNotes[f - 10] : kNotes[f];
    }
    host_overlay_rect(px + 28.0f, py + ph - 100.0f, pw - 56.0f, 1.0f, 0x6a5a34ffu);
    if (note) {
        host_overlay_text_wrapped(px + 28.0f, py + ph - 90.0f, 0.62f, 0x9a9a9affu, pw - 56.0f, note);
    }
    host_overlay_text(px + 28.0f, py + ph - 34.0f, 0.62f, 0x7a7a7affu,
                      "方向キー: 選択・変更   Enter: 切替   クリック: 設定   Esc / F10: 閉じる");
}
}  // namespace

HostSettings host_settings() { return settings_store().now(); }
const HostSettings& host_settings_ref() { return settings_store().now(); }
void host_settings_listen(HostSettingsListener fn) {
    if (!settings_store().listen(fn)) host_log("options: no room for another settings listener; one consumer will not follow changes");
}

namespace {
// Which way round a two-value setting reads comes from the values themselves,
// not a list of ids kept in step by hand: a new {"On","Off"} setting was
// silently inverted by that list, which is the sort of bug that only shows up
// as "the option does the opposite".
bool reads_on(const Setting& s, int index) {
    const std::size_t i = static_cast<std::size_t>(index);
    if (i < s.values.size()) {
        if (std::strcmp(s.values[i], "On") == 0) return true;
        if (std::strcmp(s.values[i], "Off") == 0) return false;
    }
    return index != 0;  // anything else: the first value is the "no"
}

int setting_id(const char* key) {
    for (int id = 0; key && id < kSettingCount; ++id) {
        if (std::strcmp(key, g_set[id].key) == 0) return id;
    }
    return -1;
}

}  // namespace

std::uint64_t host_opt_serial() { return settings_store().now().serial; }

bool host_opt_default_get(const char* key) {
    const int id = setting_id(key);
    return id >= 0 && reads_on(g_set[id], g_default_index[static_cast<std::size_t>(id)]);
}
int host_opt_default_index(const char* key) {
    const int id = setting_id(key);
    return id < 0 ? -1 : g_default_index[static_cast<std::size_t>(id)];
}
int host_opt_default_frame_cap() {
    return fps_from_label(g_set[kFrameCap].values[static_cast<std::size_t>(g_default_index[kFrameCap])]);
}
int host_opt_default_resolution_index() { return g_default_index[kResolution]; }

bool host_opt_get(const char* key) {
    const int id = setting_id(key);
    return id >= 0 && reads_on(g_set[id], g_set[id].index);
}

bool host_opt_resolution(int* w, int* h) {
    const HostSettings hs = host_settings();
    if (hs.res_width <= 0 || hs.res_height <= 0) return false;
    *w = hs.res_width;
    *h = hs.res_height;
    return true;
}

int host_opt_resolution_index() { return g_set[kResolution].index; }

void host_opt_set_resolution_index(int i) {
    if (i < 0 || static_cast<std::size_t>(i) >= g_set[kResolution].values.size()) {
        return;
    }
    if (i == g_set[kResolution].index) return;
    set_index(kResolution, i, false);
    save();
}

int host_opt_frame_cap() {
    return fps_from_label(g_set[kFrameCap].values[static_cast<std::size_t>(g_set[kFrameCap].index)]);
}

void host_opt_set_frame_cap(int fps) {
    const int i = fps_index(fps);
    if (i < 0) {
        return;
    }
    if (i == g_set[kFrameCap].index) return;
    set_index(kFrameCap, i, false);
    save();
}

void host_opt_set(const char* key, bool on) {
    if (!key) {
        return;
    }
    for (int id = 0; id < kSettingCount; ++id) {
        if (std::strcmp(key, g_set[id].key) != 0) {
            continue;
        }
        // Which index is "on" comes from the values, as in host_opt_get: a
        // hand-kept list of the {"On","Off"} settings missed Mouse camera and
        // the four graphics switches, so turning one of those on from the
        // game's menu saved it off. Two values that are not On/Off
        // (Windowed/Fullscreen) read the second as on.
        const Setting& s = g_set[id];
        int want = on ? 1 : 0;
        for (std::size_t v = 0; v < s.values.size(); ++v) {
            if (std::strcmp(s.values[v], on ? "On" : "Off") == 0) {
                want = static_cast<int>(v);
            }
        }
        if (want == s.index) return;  // the setup window sets every one; only a change is written
        set_index(id, want, false);
        // The F10 screen saves when it closes; a row in the game's menu has no
        // such moment, so a change made there is written now or not at all.
        save();
        return;
    }
}

int host_opt_index(const char* key) {
    for (int id = 0; key && id < kSettingCount; ++id) {
        if (std::strcmp(key, g_set[id].key) == 0) return g_set[id].index;
    }
    return -1;
}

void host_opt_set_index(const char* key, int index) {
    for (int id = 0; key && id < kSettingCount; ++id) {
        if (std::strcmp(key, g_set[id].key) != 0) continue;
        if (index < 0 || static_cast<std::size_t>(index) >= g_set[id].values.size()) return;
        if (index == g_set[id].index) return;
        set_index(id, index, false);
        save();
        return;
    }
}

void host_options_save_now() {
    g_dirty = true;
    save();
}

bool host_steam_deck() {
    static const bool deck = [] {
        if (const char* e = std::getenv("BBHOST_STEAM_DECK"); e && e[0]) return e[0] == '1';
        // The firmware's vendor and product: Valve's Jupiter (the LCD Deck) or
        // Galileo (the OLED one). Not any Valve machine - another has another
        // GPU, and these two configurations are this one's.
        std::string vendor, product;
#if defined(_WIN32)
        const auto bios = [](const char* name) {
            char buf[128] = {};
            DWORD size = sizeof(buf);
            if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\BIOS", name, RRF_RT_REG_SZ, nullptr, buf, &size) !=
                ERROR_SUCCESS)
                return std::string();
            return std::string(buf);
        };
        vendor = bios("SystemManufacturer");
        product = bios("SystemProductName");
#else
        const auto line = [](const char* path) {
            std::ifstream f(path);
            std::string s;
            std::getline(f, s);
            return s;
        };
        vendor = line("/sys/devices/virtual/dmi/id/sys_vendor");
        product = line("/sys/devices/virtual/dmi/id/product_name");
#endif
        return vendor == "Valve" && (product == "Jupiter" || product == "Galileo");
    }();
    return deck;
}

namespace {
// The Deck's configurations, by the frame rate: 60 or more (or uncapped) is 60 fps.
int deck_profile_of(int fps) { return fps == 0 || fps >= 60 ? 1 : 0; }
}  // namespace

int host_deck_profile() { return deck_profile_of(host_opt_frame_cap()); }
int host_deck_default_profile() { return deck_profile_of(host_opt_default_frame_cap()); }

void host_set_deck_profile(int profile) {
    const bool sixty = profile == 1;
    if (const int i = fps_index(sixty ? 60 : 30); i >= 0 && i != g_set[kFrameCap].index) set_index(kFrameCap, i, false);
    if (const int m = sixty ? 2 : 1; m != g_set[kModelDetail].index) set_index(kModelDetail, m, false);  // Low, Normal
    const char* const size = sixty ? "960x600" : "1280x800";
    for (std::size_t v = 0; v < g_set[kResolution].values.size(); ++v) {
        if (std::strcmp(g_set[kResolution].values[v], size) != 0 || static_cast<int>(v) == g_set[kResolution].index) continue;
        // Both sizes are known to fit the Deck: no keep-or-revert question,
        // and one still waiting is settled by this.
        g_res_prev = -1;
        g_res_reverting = true;
        set_index(kResolution, static_cast<int>(v), false);
        g_res_reverting = false;
    }
    host_log("options: Steam Deck at %s", sixty ? "60 fps (960x600 upscaled, Model detail Low)" : "30 fps (1280x800)");
    save();
}

// The ACCOUNT actions for the setup window (host/options.h).
void host_account_action(int which, const std::string& name, const std::string& code) {
    if (which >= 0 && which <= 4) account_action(which, name, code);
}
bool host_account_busy() { return g_acct_busy.load(); }
std::string host_account_signed_in() {
    return net::account_logged_in() ? "ログイン中: " + net::account_name() : "未ログイン";
}
std::string host_account_outcome() {
    std::lock_guard<std::mutex> lk(g_acct_mu);
    return g_acct_status;
}
std::string host_account_detail() { return account_detail(); }
bool host_account_take_save() { return g_acct_save.exchange(false); }
