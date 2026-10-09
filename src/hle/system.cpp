#include "hle/common.h"
#include "hle/np.h"
#include "hle/fs.h"
#include "hle/platform.h"
#if !defined(_WIN32)
#include <dirent.h>
#endif
#include "hle/hle.h"
#include "hle/modules.h"
#include "core/thunk.h"
#include "core/write_watch.h"
#include "engine/debug_menu.h"
#include "engine/menu_pointer.h"
#include "engine/mouse_camera.h"
#include "host/bindings.h"
#include "host/ingame_menu.h"
#include "host/options.h"
#include "host/settings.h"
#include "host/window.h"
#include "net/account.h"
#include "net/session.h"
#include "core/config.h"
#include "core/sfo.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <map>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <csetjmp>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if defined(_WIN32)
#include "core/win_vm.h"
#endif
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

std::mutex g_sysmod_mu;
std::unordered_map<int, int> g_sysmods;

GUEST_ABI int hle_sysmodule_load(int id) {
    std::lock_guard<std::mutex> lock(g_sysmod_mu);
    g_sysmods[id] = 1;
    host_log("sceSysmoduleLoadModule 0x%x", id);
    return 0;
}
GUEST_ABI int hle_sysmodule_unload(int id) {
    std::lock_guard<std::mutex> lock(g_sysmod_mu);
    g_sysmods.erase(id);
    return 0;
}
GUEST_ABI int hle_sysmodule_loaded(int id) {
    std::lock_guard<std::mutex> lock(g_sysmod_mu);
    return g_sysmods.count(id) ? 0 : static_cast<int>(0x805A0010);
}

GUEST_ABI int hle_appcontent_init(void*, void* boot) {
    if (boot) {
        std::memset(boot, 0, 32);
    }
    host_log("sceAppContentInitialize");
    return 0;
}
// The title's own param.sfo (/app0/sce_sys/param.sfo): its USER_DEFINED_PARAM_1..4
// are what sceAppContentAppParamGetInt(1..4) return on a console. Bloodborne
// makes its region out of the first (get_app_content, then CSLocalize), and the
// region picks the language list, the network region it reports and the button
// convention. Answering 0 had made the US disc (whose value is 2) Japanese:
// Circle decided in the menus and Cross dodged in play.
int sfo_user_param(int n) {
    static int vals[5] = {};
    static std::once_flag once;
    std::call_once(once, [] {
        const std::string path = hle_fs_map_path("/app0/sce_sys/param.sfo");
        std::map<std::string, SfoValue> sfo;
        std::string err;
        if (!sfo_read(path, &sfo, &err)) {
            host_log("param.sfo: %s (%s); USER_DEFINED_PARAM_1..4 read as 0", err.c_str(), path.c_str());
            return;
        }
        for (int which = 1; which <= 4; ++which) {
            const auto it = sfo.find("USER_DEFINED_PARAM_" + std::to_string(which));
            if (it == sfo.end() || !it->second.is_int) continue;
            vals[which] = static_cast<int>(it->second.num);
            host_log("param.sfo: USER_DEFINED_PARAM_%d = %d", which, vals[which]);
        }
    });
    return n >= 1 && n <= 4 ? vals[n] : 0;
}

GUEST_ABI int hle_appcontent_param_int(int id, int* value) {
    int v = 0;
    if (id == 0) {
        v = 3;  // SCE_APP_CONTENT_APPPARAM_ID_SKU_FLAG: full
    } else if (id >= 1 && id <= 4) {
        v = sfo_user_param(id);  // USER_DEFINED_PARAM_1..4
    }
    if (value) {
        *value = v;
    }
    host_log("sceAppContentAppParamGetInt %d -> %d", id, v);
    return 0;
}
GUEST_ABI int hle_appcontent_addcont(void*, int, int*) { return 0; }

// A console delivers a LOGIN event for the initial user after
// initialisation; games tie the pad's "decide" button to that user.
std::atomic<int> g_user_login_events{0};
GUEST_ABI int hle_user_init(void*) {
    host_log("sceUserServiceInitialize");
    g_user_login_events.store(1);
    return 0;
}
GUEST_ABI int hle_user_term() { return 0; }
GUEST_ABI int hle_user_initial(int* user) {
    if (user) {
        *user = 1;
    }
    return 0;
}
GUEST_ABI int hle_user_list(int* ids) {
    if (ids) {
        ids[0] = 1;
        ids[1] = -1;
        ids[2] = -1;
        ids[3] = -1;
    }
    return 0;
}
GUEST_ABI int hle_user_name(int, char* buf, std::uint64_t n) {
    const char* name = "Player";
    if (buf && n) {
        std::snprintf(buf, static_cast<std::size_t>(n), "%s", name);
    }
    return 0;
}
// int sceUserServiceGetEvent(SceUserServiceEvent* {int32 eventType; int32 userId})
GUEST_ABI int hle_user_event(std::int32_t* ev) {
    if (ev && g_user_login_events.load() > 0) {
        g_user_login_events.fetch_sub(1);
        ev[0] = 0;  // SCE_USER_SERVICE_EVENT_TYPE_LOGIN
        ev[1] = 1;  // user id
        host_log("sceUserServiceGetEvent -> LOGIN user 1");
        return 0;
    }
    return static_cast<int>(0x80960005);  // SCE_USER_SERVICE_ERROR_NO_EVENT
}

GUEST_ABI int hle_sysparam_int(int id, int* value) {
    int v = 0;
    if (id == 1) {
        // SCE_SYSTEM_SERVICE_PARAM_ID_LANG: English (US); BBHOST_SYSTEM_LANG=N
        // another (2 French, 4 German, 18 English UK, ...) - for testing
        // the other languages' message bundles.
        static const int lang = [] {
            const char* e = std::getenv("BBHOST_SYSTEM_LANG");
            return e && *e ? std::atoi(e) : 1;
        }();
        v = lang;
    } else if (id == 1000) {
        v = 1;      // SCE_SYSTEM_SERVICE_PARAM_ID_ENTER_BUTTON_ASSIGN: Cross
    }
    if (value) {
        *value = v;
    }
    host_log("sceSystemServiceParamGetInt %d -> %d", id, v);
    return 0;
}
GUEST_ABI int hle_sys_status(void* st) {
    if (st) {
        std::memset(st, 0, 16);
    }
    return 0;
}
GUEST_ABI int hle_sys_event(void*) { return static_cast<int>(0x80A10003); }
GUEST_ABI int hle_sys_hide_splash() { return 0; }
GUEST_ABI int hle_common_dialog_init() { return 0; }
GUEST_ABI int hle_np_restriction(int) { return 0; }
GUEST_ABI int hle_np_title(const void*, const void*) { return 0; }

GUEST_ABI int hle_playgo_init(const void*) {
    host_log("scePlayGoInitialize");
    return 0;
}
GUEST_ABI int hle_playgo_open(int* handle, const void*) {
    if (!handle) {
        return static_cast<int>(0x80B2000E);
    }
    *handle = 1;
    host_log("scePlayGoOpen -> 1");
    return 0;
}
GUEST_ABI int hle_playgo_chunk_id(int, std::uint16_t* ids, unsigned n, unsigned* out_n) {
    if (!out_n) {
        return static_cast<int>(0x80B2000E);
    }
    if (!ids) {
        *out_n = 1;
        return 0;
    }
    const unsigned wrote = n ? 1 : 0;
    if (wrote) {
        ids[0] = 0;
    }
    *out_n = wrote;
    return 0;
}
GUEST_ABI int hle_playgo_install_speed(int, int) { return 0; }
GUEST_ABI int hle_playgo_locus(int, const std::uint16_t*, unsigned n, int* locus) {
    if (!locus) {
        return static_cast<int>(0x80B2000E);
    }
    for (unsigned i = 0; i < n; ++i) {
        locus[i] = 3;
    }
    return 0;
}

GUEST_ABI int hle_rtc_dow(int year, int month, int day) {
    static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    if (month < 1 || month > 12 || day < 1) {
        return 0;
    }
    if (month < 3) {
        year -= 1;
    }
    return (year + year / 4 - year / 100 + year / 400 + t[month - 1] + day) % 7;
}

constexpr std::uint64_t kRtcUnixEpochTicks = 0xdcbffeff2bc000ull;

struct RtcDateTime {
    std::uint16_t year;
    std::uint16_t month;
    std::uint16_t day;
    std::uint16_t hour;
    std::uint16_t minute;
    std::uint16_t second;
    std::uint32_t microsecond;
};
struct RtcTick {
    std::uint64_t tick;
};

void rtc_from_tm(RtcDateTime* out, const std::tm& t, std::uint32_t us) {
    out->year = static_cast<std::uint16_t>(t.tm_year + 1900);
    out->month = static_cast<std::uint16_t>(t.tm_mon + 1);
    out->day = static_cast<std::uint16_t>(t.tm_mday);
    out->hour = static_cast<std::uint16_t>(t.tm_hour);
    out->minute = static_cast<std::uint16_t>(t.tm_min);
    out->second = static_cast<std::uint16_t>(t.tm_sec);
    out->microsecond = us;
}

GUEST_ABI int hle_rtc_local(RtcDateTime* t) {
    if (!t) {
        return sce_err(EINVAL);
    }
    timespec ts{};
#if defined(_WIN32)
    const std::time_t sec = std::time(nullptr);
    std::tm local{};
    localtime_s(&local, &sec);
    rtc_from_tm(t, local, 0);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
    std::tm local{};
    localtime_r(&ts.tv_sec, &local);
    rtc_from_tm(t, local, static_cast<std::uint32_t>(ts.tv_nsec / 1000));
#endif
    return 0;
}
GUEST_ABI int hle_rtc_get_tick(const RtcDateTime* t, RtcTick* tick) {
    if (!t || !tick) {
        return sce_err(EINVAL);
    }
    std::tm tm{};
    tm.tm_year = static_cast<int>(t->year) - 1900;
    tm.tm_mon = static_cast<int>(t->month) - 1;
    tm.tm_mday = t->day;
    tm.tm_hour = t->hour;
    tm.tm_min = t->minute;
    tm.tm_sec = t->second;
    tm.tm_isdst = -1;
#if defined(_WIN32)
    const std::time_t sec = _mkgmtime(&tm);
#else
    const std::time_t sec = timegm(&tm);
#endif
    if (sec == static_cast<std::time_t>(-1)) {
        return sce_err(EINVAL);
    }
    tick->tick = static_cast<std::uint64_t>(sec) * 1000000ull + t->microsecond + kRtcUnixEpochTicks;
    return 0;
}
GUEST_ABI int hle_rtc_set_tick(RtcDateTime* t, const RtcTick* tick) {
    if (!t || !tick) {
        return sce_err(EINVAL);
    }
    std::uint64_t v = tick->tick;
    if (v >= kRtcUnixEpochTicks) {
        v -= kRtcUnixEpochTicks;
    }
    const std::time_t sec = static_cast<std::time_t>(v / 1000000ull);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &sec);
#else
    gmtime_r(&sec, &utc);
#endif
    rtc_from_tm(t, utc, static_cast<std::uint32_t>(v % 1000000ull));
    return 0;
}
GUEST_ABI int hle_rtc_tick_res() { return 1000000; }

std::int64_t host_utc_offset_us() {
#if defined(_WIN32)
    long bias = 0;
    _get_timezone(&bias);
    return -static_cast<std::int64_t>(bias) * 1000000ll;
#else
    time_t now = time(nullptr);
    struct tm lt{};
    localtime_r(&now, &lt);
    return static_cast<std::int64_t>(lt.tm_gmtoff) * 1000000ll;
#endif
}
GUEST_ABI int hle_rtc_utc_to_local(const RtcTick* utc, RtcTick* local) {
    if (!utc || !local) {
        return static_cast<int>(0x80010016u);
    }
    local->tick = utc->tick + static_cast<std::uint64_t>(host_utc_offset_us());
    return 0;
}
GUEST_ABI int hle_rtc_local_to_utc(const RtcTick* local, RtcTick* utc) {
    if (!utc || !local) {
        return static_cast<int>(0x80010016u);
    }
    utc->tick = local->tick - static_cast<std::uint64_t>(host_utc_offset_us());
    return 0;
}
// sceRtcFormatRFC2822LocalTime(char* out, const SceRtcTick* utc, int minutes_offset)
GUEST_ABI int hle_rtc_format_rfc2822(char* out, const RtcTick* utc, int minutes) {
    if (!out || !utc) {
        return static_cast<int>(0x80010016u);
    }
    const std::int64_t unix_us = static_cast<std::int64_t>(utc->tick) - kRtcUnixEpochTicks;
    time_t t = static_cast<time_t>(unix_us / 1000000) + static_cast<time_t>(minutes) * 60;
    struct tm g{};
#if defined(_WIN32)
    gmtime_s(&g, &t);
#else
    gmtime_r(&t, &g);
#endif
    static const char* days[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
    static const char* mons[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    const int sign = minutes < 0 ? -1 : 1;
    const int abs_min = minutes * sign;
    std::snprintf(out, 32, "%s, %02d %s %04d %02d:%02d:%02d %c%02d%02d", days[g.tm_wday & 7], g.tm_mday % 100,
                  mons[g.tm_mon % 12], (g.tm_year + 1900) % 10000, g.tm_hour % 100, g.tm_min % 100, g.tm_sec % 100,
                  sign < 0 ? '-' : '+', (abs_min / 60) % 100, abs_min % 60);
    return 0;
}
GUEST_ABI int hle_rtc_network_tick(RtcTick* tick) {
    if (!tick) {
        return sce_err(EINVAL);
    }
    timespec ts{};
#if defined(_WIN32)
    const std::time_t sec = std::time(nullptr);
    tick->tick = static_cast<std::uint64_t>(sec) * 1000000ull + kRtcUnixEpochTicks;
#else
    clock_gettime(CLOCK_REALTIME, &ts);
    tick->tick = static_cast<std::uint64_t>(ts.tv_sec) * 1000000ull +
                 static_cast<std::uint64_t>(ts.tv_nsec) / 1000ull + kRtcUnixEpochTicks;
#endif
    return 0;
}
GUEST_ABI int hle_rtc_set_time_t(RtcDateTime* t, std::int64_t unix_sec) {
    if (!t) {
        return sce_err(EINVAL);
    }
    const std::time_t sec = static_cast<std::time_t>(unix_sec);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &sec);
#else
    gmtime_r(&sec, &utc);
#endif
    rtc_from_tm(t, utc, 0);
    return 0;
}

using NpStateCb = void(GUEST_ABI*)(int, int, void*, void*);
using NpPresenceCb = void(GUEST_ABI*)(void*, int, void*);
NpStateCb g_np_state_cb;
void* g_np_state_ud;
NpPresenceCb g_np_pres_cb;
void* g_np_pres_ud;
bool g_np_state_fired;
bool g_np_pres_fired;
std::uint8_t g_np_id[36];

GUEST_ABI int hle_np_reg_state(NpStateCb cb, void* ud) {
    g_np_state_cb = cb;
    g_np_state_ud = ud;
    g_np_state_fired = false;
    host_log("sceNpRegisterStateCallback");
    return 0;
}
GUEST_ABI int hle_np_reg_presence(NpPresenceCb cb, void* ud) {
    g_np_pres_cb = cb;
    g_np_pres_ud = ud;
    g_np_pres_fired = false;
    host_log("sceNpRegisterGamePresenceCallback");
    return 0;
}
GUEST_ABI int hle_np_get_state(int, int* state) {
    if (!state) {
        return static_cast<int>(0x80550003);
    }
    // BBHOST_NP_SIGNED_OUT=1 reports SCE_NP_STATE_SIGNED_OUT (1) instead of
    // SIGNED_IN (2), to test whether the online boot path derails the menu
    // preload (shadPS4 runs Np-stubbed = signed out and loads the full title).
    static const int forced = [] {
        const char* e = std::getenv("BBHOST_NP_SIGNED_OUT");
        return (e && e[0] == '1') ? 1 : 2;
    }();
    // online.require_account: signed out until the F10 screen has logged an
    // account in, so the game plays offline rather than as a name
    // taken from a file.
    *state = (forced == 2 && config().online_require_account && !net::account_logged_in()) ? 1 : forced;
    return 0;
}
std::string g_online_id = "Player";
GUEST_ABI int hle_np_online_id(int, void* id) {
    if (!id) {
        return static_cast<int>(0x80550003);
    }
    // Read live: the account logged in on the F10 screen is the
    // name the game must carry in its signs and rooms, as the session
    // service already does through net::online_id().
    const std::string cur = net::online_id();
    std::memset(id, 0, 20);
    std::memcpy(id, cur.c_str(), std::min<std::size_t>(cur.size(), 16));
    return 0;
}
GUEST_ABI int hle_np_get_npid(int, void* id) {
    if (!id) {
        return static_cast<int>(0x80550003);
    }
    // The same handle as sceNpGetOnlineId: the session service matches peers
    // by it, so the game's NpId and the server's online id must agree.
    hle_np_fill_npid(id, net::online_id().c_str());
    return 0;
}
GUEST_ABI int hle_np_presence_status(int, int* st) {
    if (st) {
        *st = 1;
    }
    return 0;
}
GUEST_ABI int hle_np_cmp_online(const void* a, const void* b) {
    if (!a || !b) {
        return static_cast<int>(0x80550003);
    }
    return std::memcmp(a, b, 16);
}
GUEST_ABI int hle_np_cmp_npid(const void* a, const void* b) {
    if (!a || !b) {
        return static_cast<int>(0x80550003);
    }
    return std::memcmp(a, b, 36);
}
GUEST_ABI int hle_np_unreg_state(void*) {
    g_np_state_cb = nullptr;
    g_np_state_ud = nullptr;
    return 0;
}
GUEST_ABI int hle_score_title_ctx(const void*, int) {
    static int id = 1;
    return id++;
}
// A no-op, as on shadPS4: the console kernel's sceNpCheckCallback also
// runs NP state management that latches FrpgNetMan's error flag; firing
// the callbacks is not needed because sceNpGetState already reports
// signed in.
GUEST_ABI int hle_np_check_cb_noop() { return 0; }

GUEST_ABI int hle_np_check_cb() {
    if (!g_np_state_fired && g_np_state_cb) {
        g_np_state_fired = true;
        std::memset(g_np_id, 0, sizeof(g_np_id));
        std::memcpy(g_np_id, "Player", 6);
        host_log("sceNpCheckCallback -> SignedIn");
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(g_np_state_cb), 1, 2, g_np_id, g_np_state_ud);
    }
    if (!g_np_pres_fired && g_np_pres_cb) {
        g_np_pres_fired = true;
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(g_np_pres_cb), g_np_id, 1, g_np_pres_ud);
    }
    return 0;
}

// ---- Save data: directory-backed under <data>/saves/<dirName>.
// SceSaveDataMount { int32 userId; pad; const char* titleId; const char* dirName;
//   const char* fingerprint; uint64 blocks; uint32 mountMode; reserved[32] }
// SceSaveDataMountResult { char mountPoint[16]; uint64 requiredBlocks; uint32 unused;
//   uint32 mountStatus; reserved[28]; }
struct SaveMountReq {
    std::int32_t user_id;
    std::int32_t pad;
    const char* title_id;
    const char* dir_name;
    const char* fingerprint;
    std::uint64_t blocks;
    std::uint32_t mount_mode;
    std::uint8_t reserved[32];
};
struct SaveMountResult {
    char mount_point[16];
    std::uint64_t required_blocks;
    std::uint32_t unused;
    std::uint32_t mount_status;
    std::uint8_t reserved[28];
};
constexpr int kSaveErrParam = static_cast<int>(0x809F0000u);
constexpr int kSaveErrNotFound = static_cast<int>(0x809F0008u);
constexpr int kSaveErrExists = static_cast<int>(0x809F0007u);
constexpr int kSaveErrBusy = static_cast<int>(0x809F000Du);
constexpr int kSaveErrMountFull = static_cast<int>(0x809F000Fu);
constexpr std::uint32_t kMountRdonly = 1;
constexpr std::uint32_t kMountRdwr = 2;
constexpr std::uint32_t kMountCreate = 4;
constexpr std::uint32_t kMountCreate2 = 0x40;
constexpr std::uint64_t kSaveBlockSize = 32768;

std::mutex g_save_mu;
struct SaveMount {
    std::string dir_name;
    bool used = false;
    bool writable = false;  // mounted to write: the unmount may back it up
};
SaveMount g_save_mounts[16];

std::string save_dir_host(const std::string& dir_name) {
    return hle_fs_map_path(("/data/saves/" + dir_name).c_str());
}

bool save_dir_exists(const std::string& dir_name) {
    std::string h = save_dir_host(dir_name);
    if (h.empty()) {
        return false;
    }
    struct stat st{};
    return ::stat(h.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool valid_dir_name(const char* n) {
    if (!n || !n[0]) {
        return false;
    }
    for (const char* p = n; *p; ++p) {
        if (!(std::isalnum(static_cast<unsigned char>(*p)) || *p == '_' || *p == '-')) {
            return false;
        }
        if (p - n >= 31) {
            return false;
        }
    }
    return true;
}

// Save backups, a port's safety net (docs/running.md): when a mount opened to
// write is unmounted - the game has finished writing it - the directory is
// copied to <data>/save-backups/<dir>/<time>/, at most every
// BBHOST_SAVE_BACKUP_MINUTES (5) per save, the newest BBHOST_SAVE_BACKUPS (8)
// kept (0 turns it off). Outside saves/ so the game's own search never sees
// them; a backup is restored by copying its files back into saves/<dir>.
// Synchronous, so a copy is never half of a save: the game unmounts ~20
// times in a few minutes and a save is ~3 MB, so this is a few ms every
// five minutes.
std::map<std::string, std::chrono::steady_clock::time_point> g_last_backup;

void back_up_save(const std::string& dir) {
    static const int keep = [] {
        const char* e = std::getenv("BBHOST_SAVE_BACKUPS");
        return e && *e ? std::atoi(e) : 8;
    }();
    static const int minutes = [] {
        const char* e = std::getenv("BBHOST_SAVE_BACKUP_MINUTES");
        return e && *e ? std::max(0, std::atoi(e)) : 5;
    }();
    if (keep <= 0) return;
    const auto now = std::chrono::steady_clock::now();
    auto last = g_last_backup.find(dir);
    if (last != g_last_backup.end() && now - last->second < std::chrono::minutes(minutes)) return;
    namespace fs = std::filesystem;
    const std::string src = save_dir_host(dir);
    const std::string root = hle_fs_map_path(("/data/save-backups/" + dir).c_str());
    if (src.empty() || root.empty()) return;
    char stamp[32];
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm);
    std::error_code ec;
    fs::create_directories(fs::path(root) / stamp, ec);
    if (!ec) fs::copy(src, fs::path(root) / stamp, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    if (ec) {
        static int said = 0;
        if (said++ < 4) host_log("save backup: %s could not be copied (%s)", dir.c_str(), ec.message().c_str());
        return;
    }
    g_last_backup[dir] = now;
    // The newest `keep`: the names sort by time.
    std::vector<fs::path> backups;
    for (const auto& e : fs::directory_iterator(root, ec)) {
        if (e.is_directory()) backups.push_back(e.path());
    }
    std::sort(backups.begin(), backups.end());
    for (std::size_t i = 0; i + static_cast<std::size_t>(keep) < backups.size(); ++i) fs::remove_all(backups[i], ec);
    host_log("save backup: %s -> save-backups/%s/%s (%zu kept)", dir.c_str(), dir.c_str(), stamp,
             std::min(backups.size(), static_cast<std::size_t>(keep)));
}

GUEST_ABI int hle_save_init(void*) {
    host_log("sceSaveDataInitialize");
    hle_fs_mount("/data/saves", "saves");  // ensures <data>/saves exists
    return 0;
}
GUEST_ABI int hle_save_term() { return 0; }

GUEST_ABI int hle_save_mount(const SaveMountReq* req, SaveMountResult* result) {
    if (!req || !result || !valid_dir_name(req->dir_name)) {
        return kSaveErrParam;
    }
    std::memset(result, 0, sizeof(*result));
    const std::string dir = req->dir_name;
#if defined(_WIN32)
    if (const char* e = std::getenv("BBHOST_SAVE_DUMP"); e && e[0] == '1') win_vm_dump_views();
#endif
    const bool exists = save_dir_exists(dir);
    const bool create = (req->mount_mode & (kMountCreate | kMountCreate2)) != 0;
    if (!exists && !create) {
        host_log("sceSaveDataMount %s: not found", dir.c_str());
        return kSaveErrNotFound;
    }
    if (exists && (req->mount_mode & kMountCreate) && !(req->mount_mode & kMountCreate2) &&
        !(req->mount_mode & (kMountRdwr | kMountRdonly))) {
        return kSaveErrExists;
    }
    std::lock_guard<std::mutex> lock(g_save_mu);
    int slot = -1;
    for (int i = 0; i < 16; ++i) {
        if (g_save_mounts[i].used && g_save_mounts[i].dir_name == dir) {
            return kSaveErrBusy;
        }
        if (slot < 0 && !g_save_mounts[i].used) {
            slot = i;
        }
    }
    if (slot < 0) {
        return kSaveErrMountFull;
    }
    char mp[16] = {};
    std::snprintf(mp, sizeof(mp), "/savedata%d", slot & 15);
    if (!hle_fs_mount(mp, ("saves/" + dir).c_str())) {
        return kSaveErrParam;
    }
    g_save_mounts[slot].used = true;
    g_save_mounts[slot].dir_name = dir;
    g_save_mounts[slot].writable = (req->mount_mode & (kMountRdwr | kMountCreate | kMountCreate2)) != 0;
    std::memcpy(result->mount_point, mp, sizeof(mp));
    result->required_blocks = 0;
    result->mount_status = exists ? 0 : 1;  // 1 = SCE_SAVE_DATA_MOUNT_STATUS_CREATED
    host_log("sceSaveDataMount %s mode=0x%x blocks=%llu -> %s%s", dir.c_str(), req->mount_mode,
             static_cast<unsigned long long>(req->blocks), mp, exists ? "" : " (created)");
    return 0;
}

int save_slot_of(const char* mount_point) {
    int slot = -1;
    if (mount_point && std::sscanf(mount_point, "/savedata%d", &slot) == 1 && slot >= 0 && slot < 16 &&
        g_save_mounts[slot].used) {
        return slot;
    }
    return -1;
}

GUEST_ABI int hle_save_umount(const char* mount_point) {
    std::lock_guard<std::mutex> lock(g_save_mu);
    int slot = save_slot_of(mount_point);
    if (slot < 0) {
        return kSaveErrParam;
    }
    hle_fs_umount(mount_point);
    g_save_mounts[slot].used = false;
    host_log("sceSaveDataUmount %s", mount_point);
    if (g_save_mounts[slot].writable) back_up_save(g_save_mounts[slot].dir_name);
    return 0;
}

// SceSaveDataDelete { int32 userId; pad; const char* titleId; const char* dirName; reserved[32] }
GUEST_ABI int hle_save_delete(const SaveMountReq* req) {
    if (!req || !valid_dir_name(req->dir_name)) {
        return kSaveErrParam;
    }
    std::string h = save_dir_host(req->dir_name);
    if (h.empty() || !save_dir_exists(req->dir_name)) {
        return kSaveErrNotFound;
    }
    // Remove contents then the directory.
    std::vector<std::string> names;
#if !defined(_WIN32)
    if (DIR* d = ::opendir(h.c_str())) {
        while (dirent* e = ::readdir(d)) {
            if (std::strcmp(e->d_name, ".") && std::strcmp(e->d_name, "..")) {
                names.push_back(e->d_name);
            }
        }
        ::closedir(d);
    }
    for (const auto& n : names) {
        ::unlink((h + "/" + n).c_str());
    }
    ::rmdir(h.c_str());
#else
    _rmdir(h.c_str());
#endif
    host_log("sceSaveDataDelete %s", req->dir_name);
    return 0;
}

// SceSaveDataDirNameSearchCond { int32 userId; pad; const char* titleId; const char* dirName;
//   int32 key; int32 order; reserved[16] }
// SceSaveDataDirNameSearchResult { uint32 hitNum; pad; SceSaveDataDirName* dirNames;
//   uint32 dirNamesNum; uint32 setNum; SceSaveDataParam* params; SceSaveDataSearchInfo* infos;
//   reserved[12] }
struct SaveSearchCond {
    std::int32_t user_id;
    std::int32_t pad;
    const char* title_id;
    const char* dir_name;
    std::int32_t key;
    std::int32_t order;
    std::uint8_t reserved[16];
};
struct SaveParam {
    char title[128];
    char sub_title[128];
    char detail[1024];
    std::uint32_t user_param;
    std::int32_t pad;
    std::int64_t mtime;
    std::uint8_t reserved[32];
};
static_assert(sizeof(SaveParam) == 1328, "SceSaveDataParam");
struct SaveSearchInfo {
    std::uint64_t blocks;
    std::uint64_t free_blocks;
    std::uint8_t reserved[32];
};
struct SaveSearchResult {
    std::uint32_t hit_num;
    std::int32_t pad;
    char (*dir_names)[32];
    std::uint32_t dir_names_num;
    std::uint32_t set_num;
    SaveParam* params;
    SaveSearchInfo* infos;
    std::uint8_t reserved[12];
};

bool load_param(const std::string& dir_name, SaveParam* p) {
    std::memset(p, 0, sizeof(*p));
    std::string h = save_dir_host(dir_name);
    if (h.empty()) {
        return false;
    }
    std::FILE* f = std::fopen((h + "/sce_param.bin").c_str(), "rb");
    if (!f) {
        struct stat st{};
        if (::stat(h.c_str(), &st) == 0) {
            p->mtime = static_cast<std::int64_t>(st.st_mtime);
        }
        return true;
    }
    std::size_t n = std::fread(p, 1, sizeof(*p), f);
    std::fclose(f);
    return n == sizeof(*p);
}

bool store_param(const std::string& dir_name, const SaveParam* p) {
    std::string h = save_dir_host(dir_name);
    if (h.empty()) {
        return false;
    }
    std::FILE* f = std::fopen((h + "/sce_param.bin").c_str(), "wb");
    if (!f) {
        return false;
    }
    std::fwrite(p, 1, sizeof(*p), f);
    std::fclose(f);
    return true;
}

bool glob_match(const char* pat, const char* s) {
    if (!pat || !*pat) {
        return true;
    }
    while (*pat) {
        if (*pat == '%' || *pat == '*') {
            ++pat;
            if (!*pat) {
                return true;
            }
            for (const char* t = s; *t; ++t) {
                if (glob_match(pat, t)) {
                    return true;
                }
            }
            return false;
        }
        if (*pat != *s) {
            return false;
        }
        ++pat;
        ++s;
    }
    return *s == 0;
}

GUEST_ABI int hle_save_dirname_search(const SaveSearchCond* cond, SaveSearchResult* res) {
    if (!cond || !res) {
        return kSaveErrParam;
    }
    std::vector<std::string> found;
    std::string root = hle_fs_map_path("/data/saves");
#if !defined(_WIN32)
    if (DIR* d = ::opendir(root.c_str())) {
        while (dirent* e = ::readdir(d)) {
            if (e->d_name[0] == '.') {
                continue;
            }
            if (glob_match(cond->dir_name, e->d_name) && save_dir_exists(e->d_name)) {
                found.push_back(e->d_name);
            }
        }
        ::closedir(d);
    }
#endif
    std::sort(found.begin(), found.end());
    if (cond->order == 1) {  // SCE_SAVE_DATA_SORT_ORDER_DESCENT
        std::reverse(found.begin(), found.end());
    }
    res->hit_num = static_cast<std::uint32_t>(found.size());
    std::uint32_t n = 0;
    for (const auto& name : found) {
        if (n >= res->dir_names_num) {
            break;
        }
        if (res->dir_names) {
            std::memset(res->dir_names[n], 0, 32);
            std::strncpy(res->dir_names[n], name.c_str(), 31);
        }
        if (res->params) {
            load_param(name, &res->params[n]);
        }
        if (res->infos) {
            std::memset(&res->infos[n], 0, sizeof(SaveSearchInfo));
            res->infos[n].blocks = 4096;
            res->infos[n].free_blocks = 4000;
        }
        ++n;
    }
    res->set_num = n;
    host_log("sceSaveDataDirNameSearch pattern=%s -> %u hit", cond->dir_name ? cond->dir_name : "*", res->hit_num);
    return 0;
}

// sceSaveDataSetParam(mountPoint, type, buf, size): type 0 = whole SceSaveDataParam,
// 1 title, 2 subtitle, 3 detail, 4 userParam, 5 mtime.
GUEST_ABI int hle_save_set_param(const char* mount_point, int type, const void* buf, std::uint64_t size) {
    std::lock_guard<std::mutex> lock(g_save_mu);
    int slot = save_slot_of(mount_point);
    if (slot < 0 || !buf) {
        return kSaveErrParam;
    }
    SaveParam p;
    load_param(g_save_mounts[slot].dir_name, &p);
    auto copy = [&](char* dst, std::size_t cap) {
        std::memset(dst, 0, cap);
        std::memcpy(dst, buf, static_cast<std::size_t>(std::min<std::uint64_t>(size, cap - 1)));
    };
    switch (type) {
        case 0:
            if (size >= sizeof(SaveParam)) {
                std::memcpy(&p, buf, sizeof(SaveParam));
            }
            break;
        case 1: copy(p.title, sizeof(p.title)); break;
        case 2: copy(p.sub_title, sizeof(p.sub_title)); break;
        case 3: copy(p.detail, sizeof(p.detail)); break;
        case 4:
            if (size >= 4) {
                std::memcpy(&p.user_param, buf, 4);
            }
            break;
        case 5:
            if (size >= 8) {
                std::memcpy(&p.mtime, buf, 8);
            }
            break;
        default:
            return kSaveErrParam;
    }
    if (type != 5) {
        p.mtime = static_cast<std::int64_t>(std::time(nullptr));
    }
    return store_param(g_save_mounts[slot].dir_name, &p) ? 0 : kSaveErrParam;
}

// SceSaveDataIcon { void* buf; size_t bufSize; size_t dataSize; reserved[32] }
struct SaveIcon {
    const void* buf;
    std::uint64_t buf_size;
    std::uint64_t data_size;
    std::uint8_t reserved[32];
};
GUEST_ABI int hle_save_save_icon(const char* mount_point, const SaveIcon* icon) {
    std::lock_guard<std::mutex> lock(g_save_mu);
    int slot = save_slot_of(mount_point);
    if (slot < 0 || !icon || !icon->buf) {
        return kSaveErrParam;
    }
    std::string h = save_dir_host(g_save_mounts[slot].dir_name);
    std::FILE* f = h.empty() ? nullptr : std::fopen((h + "/icon0.png").c_str(), "wb");
    if (!f) {
        return kSaveErrParam;
    }
    std::fwrite(icon->buf, 1, static_cast<std::size_t>(icon->data_size), f);
    std::fclose(f);
    return 0;
}

// Save data memory: one fixed-size file per user.
std::string save_memory_path(int user_id) {
    char name[64];
    std::snprintf(name, sizeof(name), "/data/saves/memory_u%d.bin", user_id);
    return hle_fs_map_path(name);
}
GUEST_ABI int hle_save_setup_memory(int user_id, std::uint64_t size, const void*) {
    std::string h = save_memory_path(user_id);
    if (h.empty()) {
        return kSaveErrParam;
    }
    std::FILE* f = std::fopen(h.c_str(), "rb");
    if (f) {
        std::fclose(f);
        host_log("sceSaveDataSetupSaveDataMemory user=%d size=%llu (exists)", user_id,
                 static_cast<unsigned long long>(size));
        return kSaveErrExists;
    }
    f = std::fopen(h.c_str(), "wb");
    if (!f) {
        return kSaveErrParam;
    }
    std::vector<char> zero(static_cast<std::size_t>(size), 0);
    std::fwrite(zero.data(), 1, zero.size(), f);
    std::fclose(f);
    host_log("sceSaveDataSetupSaveDataMemory user=%d size=%llu", user_id, static_cast<unsigned long long>(size));
    return 0;
}
GUEST_ABI int hle_save_get_memory(int user_id, void* buf, std::uint64_t size, std::int64_t off) {
    std::string h = save_memory_path(user_id);
    std::FILE* f = h.empty() ? nullptr : std::fopen(h.c_str(), "rb");
    if (!f) {
        return kSaveErrNotFound;
    }
    std::fseek(f, static_cast<long>(off), SEEK_SET);
    std::size_t n = write_watch_fread(buf, 1, static_cast<std::size_t>(size), f);
    std::fclose(f);
    if (n < size) {
        std::memset(static_cast<char*>(buf) + n, 0, static_cast<std::size_t>(size) - n);
    }
    return 0;
}
GUEST_ABI int hle_save_set_memory(int user_id, const void* buf, std::uint64_t size, std::int64_t off) {
    std::string h = save_memory_path(user_id);
    std::FILE* f = h.empty() ? nullptr : std::fopen(h.c_str(), "r+b");
    if (!f) {
        return kSaveErrNotFound;
    }
    std::fseek(f, static_cast<long>(off), SEEK_SET);
    std::fwrite(buf, 1, static_cast<std::size_t>(size), f);
    std::fclose(f);
    return 0;
}

// Save data dialog: there is no UI; every dialog finishes immediately.
GUEST_ABI int hle_save_dialog_init() { return 0; }
GUEST_ABI int hle_save_dialog_term() { return 0; }
GUEST_ABI int hle_save_dialog_open(const void*) {
    host_log("HLE fake: sceSaveDataDialogOpen (auto-finished)");
    return 0;
}
GUEST_ABI int hle_save_dialog_update() { return 3; }  // SCE_COMMON_DIALOG_STATUS_FINISHED

int g_trophy_ctx = 1;
int g_trophy_handle = 1;
GUEST_ABI int hle_trophy_ctx(int* ctx, int, unsigned, std::uint64_t) {
    if (!ctx) {
        return static_cast<int>(0x805D0002);
    }
    *ctx = g_trophy_ctx++;
    host_log("sceNpTrophyCreateContext -> %d", *ctx);
    return 0;
}
GUEST_ABI int hle_trophy_handle(int* h) {
    if (!h) {
        return static_cast<int>(0x805D0002);
    }
    *h = g_trophy_handle++;
    host_log("sceNpTrophyCreateHandle -> %d", *h);
    return 0;
}
// Trophies persist as one line per id in <data>/trophies.txt.
std::mutex g_trophy_mu;
std::string trophy_file() { return hle_fs_map_path("/data/trophies.txt"); }
bool trophy_unlocked(int id, std::int64_t* when) {
    std::FILE* f = std::fopen(trophy_file().c_str(), "r");
    if (!f) {
        return false;
    }
    int tid;
    long long ts;
    bool found = false;
    while (std::fscanf(f, "%d %lld", &tid, &ts) == 2) {
        if (tid == id) {
            found = true;
            if (when) {
                *when = ts;
            }
        }
    }
    std::fclose(f);
    return found;
}
// sceNpTrophyUnlockTrophy(ctx, handle, trophyId, SceNpTrophyId* platinumId)
GUEST_ABI int hle_trophy_unlock(int ctx, int handle, int id, int* platinum_id) {
    std::lock_guard<std::mutex> lock(g_trophy_mu);
    if (platinum_id) {
        *platinum_id = -1;  // SCE_NP_TROPHY_INVALID_TROPHY_ID: platinum not (yet) awarded
    }
    if (trophy_unlocked(id, nullptr)) {
        return static_cast<int>(0x80551513u);  // SCE_NP_TROPHY_ERROR_ALREADY_UNLOCKED
    }
    std::FILE* f = std::fopen(trophy_file().c_str(), "a");
    if (f) {
        std::fprintf(f, "%d %lld\n", id, static_cast<long long>(std::time(nullptr)));
        std::fclose(f);
    }
    host_log("sceNpTrophyUnlockTrophy ctx=%d handle=%d id=%d", ctx, handle, id);
    return 0;
}
// SceNpTrophyGameDetails { size, numGroups, numTrophies, numPlatinum, numGold, numSilver, numBronze, title[128], description[1024] }
// SceNpTrophyGameData { size, unlockedTrophies, unlockedPlatinum, unlockedGold, unlockedSilver, unlockedBronze, progressPercentage }
GUEST_ABI int hle_trophy_game_info(int, int, std::uint8_t* details, std::uint8_t* data) {
    constexpr std::uint32_t kTrophies = 40;
    if (details) {
        std::memset(details + 8, 0, 0x480 - 8);
        std::uint32_t v[7] = {0x488, 1, kTrophies, 1, 3, 8, 28};
        std::memcpy(details + 8, v + 1, 24);
        std::strncpy(reinterpret_cast<char*>(details + 32), "Bloodborne", 127);
    }
    if (data) {
        std::uint32_t unlocked = 0;
        std::lock_guard<std::mutex> lock(g_trophy_mu);
        for (int id = 0; id < static_cast<int>(kTrophies); ++id) {
            if (trophy_unlocked(id, nullptr)) {
                ++unlocked;
            }
        }
        std::memset(data + 8, 0, 24);
        std::uint32_t u[6] = {unlocked, 0, 0, 0, unlocked, unlocked * 100 / kTrophies};
        std::memcpy(data + 8, u, 24);
    }
    return 0;
}
// SceNpTrophyDetails { size, trophyId, trophyGrade, groupId, hidden, reserved[3], name[128], description[1024] }
// SceNpTrophyData { size, trophyId, unlocked, reserved[3], timestamp }
GUEST_ABI int hle_trophy_info(int, int, int id, std::uint8_t* details, std::uint8_t* data) {
    if (details) {
        std::memset(details + 8, 0, 0x490 - 8);
        std::int32_t* d = reinterpret_cast<std::int32_t*>(details + 8);
        d[0] = id;
        d[1] = 4;  // bronze
        d[2] = 0;
        std::snprintf(reinterpret_cast<char*>(details + 24), 127, "Trophy %d", id);
    }
    if (data) {
        std::int64_t when = 0;
        bool unlocked;
        {
            std::lock_guard<std::mutex> lock(g_trophy_mu);
            unlocked = trophy_unlocked(id, &when);
        }
        std::int32_t* d = reinterpret_cast<std::int32_t*>(data + 8);
        d[0] = id;
        d[1] = unlocked ? 1 : 0;
        d[2] = d[3] = d[4] = 0;
        std::uint64_t ts = unlocked ? static_cast<std::uint64_t>(when) * 1000000ull + kRtcUnixEpochTicks : 0;
        std::memcpy(data + 32, &ts, 8);
    }
    return 0;
}

// Leaderboards: no server yet. Contexts and requests are real handles; the
// synchronous queries report the community server as unreachable.
constexpr int kScoreErrOffline = static_cast<int>(0x80550605u);  // SCE_NP_COMMUNITY_ERROR_UNKNOWN? treated as failure
std::atomic<int> g_score_req{1};
GUEST_ABI int hle_score_delete_title_ctx(int) { return 0; }
GUEST_ABI int hle_score_create_request(int) { return g_score_req.fetch_add(1); }
GUEST_ABI int hle_score_delete_request(int) { return 0; }
GUEST_ABI int hle_score_abort_request(int) { return 0; }
GUEST_ABI int hle_score_set_pc_id(int, int) { return 0; }
GUEST_ABI int hle_score_offline() {
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 4) {
        host_log("HLE fake: sceNpScore* query (no leaderboard server)");
    }
    return kScoreErrOffline;
}
GUEST_ABI int hle_score_comment_ok(int, const char*) { return 0; }

// Mouse: no device yet. Open hands out a handle; Read reports no data.
GUEST_ABI int hle_mouse_init() { return 0; }
GUEST_ABI int hle_mouse_open(int, int, int, const void*) { return 1; }
GUEST_ABI int hle_mouse_close(int) { return 0; }
GUEST_ABI int hle_mouse_read(int, void*, int) { return 0; }

GUEST_ABI int hle_launch_web_browser(const char* url, const void*) {
    host_log("HLE fake: sceSystemServiceLaunchWebBrowser %s", url ? url : "");
    return static_cast<int>(0x80A10003u);
}

// ---- NP auth / availability / Plus / parental: the private server does not
// use PSN, so these succeed locally with the values the game expects.
std::atomic<int> g_np_req{1};
GUEST_ABI int hle_np_create_async_request(const void*) { return g_np_req.fetch_add(1); }
GUEST_ABI int hle_np_delete_request(int) { return 0; }
GUEST_ABI int hle_np_abort_request(int) { return 0; }
GUEST_ABI int hle_np_poll_async(int, int* result) {
    if (result) {
        *result = 0;
    }
    return 0;  // finished
}
GUEST_ABI int hle_np_check_availability(int, const void*, void*) { return 0; }
// sceNpGetParentalControlInfo(reqId, const SceNpOnlineId*, int8* age, SceNpParentalControlInfo* {bool content, chat, ugc})
GUEST_ABI int hle_np_parental(int, const void*, std::int8_t* age, std::uint8_t* info) {
    if (age) {
        *age = 25;
    }
    if (info) {
        std::memset(info, 0, 3);
    }
    return 0;
}
// sceNpCheckPlus(reqId, const SceNpCheckPlusParameter*, SceNpCheckPlusResult* {bool authorized})
GUEST_ABI int hle_np_check_plus(int, const void*, std::uint8_t* result) {
    if (result) {
        result[0] = 1;
    }
    return 0;
}
GUEST_ABI int hle_np_plus_cb(const void*, void*) { return 0; }
GUEST_ABI int hle_np_plus_notify(int, std::uint64_t) { return 0; }
GUEST_ABI int hle_npauth_create_async_request(const void*) { return g_np_req.fetch_add(1); }
GUEST_ABI int hle_npauth_delete_request(int) { return 0; }
GUEST_ABI int hle_npauth_poll_async(int, int* result) {
    if (result) {
        *result = 0;
    }
    return 0;
}
// sceNpAuthGetAuthorizationCode(reqId, const SceNpAuthGetAuthorizationCodeParameter*, SceNpAuthorizationCode* {char code[128]}, int* issuerId)
GUEST_ABI int hle_npauth_get_code(int, const void*, char* code, int* issuer) {
    if (code) {
        std::memset(code, 0, 128);
        std::strncpy(code, "DUMMY", 127);
    }
    if (issuer) {
        *issuer = 10;
    }
    return 0;
}

GUEST_ABI int hle_trophy_register(int ctx, int, std::uint64_t, void*) {
    host_log("sceNpTrophyRegisterContext ctx=%d", ctx);
    return 0;
}

constexpr int kWebApiInvalid = static_cast<int>(0x80552902);
constexpr int kWebApiNoReq = static_cast<int>(0x80552906);
constexpr char kWebApiJsonType[] = "application/json; charset=utf-8";

struct WebApiReq {
    int ctx = 0;
    int status = 200;
    std::string body;
    std::size_t read_off = 0;
};

std::mutex g_web_mu;
int g_web_lib = 1;
int g_web_ctx = 1;
int g_web_filter = 1;
int g_web_cb = 1;
std::int64_t g_web_req = 1;
std::unordered_map<std::int64_t, WebApiReq> g_web_reqs;

std::string webapi_body_for_path(const char* path) {
    if (path && std::strstr(path, "friendList")) {
        return "{\"totalResults\":0,\"friendList\":[]}";
    }
    if (path && std::strstr(path, "blockList")) {
        return "{\"totalResults\":0,\"blockList\":[]}";
    }
    return "{\"totalResults\":0}";
}


GUEST_ABI int hle_webapi_init(int http_ctx, std::uint64_t pool) {
    std::lock_guard<std::mutex> lock(g_web_mu);
    const int id = g_web_lib++;
    host_log("sceNpWebApiInitialize http=%d pool=%llu -> %d", http_ctx,
             static_cast<unsigned long long>(pool), id);
    return id;
}
GUEST_ABI int hle_webapi_term(int) { return 0; }
GUEST_ABI int hle_webapi_create_ctx(int lib, const void*) {
    (void)lib;
    std::lock_guard<std::mutex> lock(g_web_mu);
    const int id = g_web_ctx++;
    host_log("sceNpWebApiCreateContext lib=%d -> %d", lib, id);
    return id;
}
GUEST_ABI int hle_webapi_delete_ctx(int) { return 0; }
GUEST_ABI int hle_webapi_create_filter(int lib, const void*, std::uint64_t n) {
    std::lock_guard<std::mutex> lock(g_web_mu);
    const int id = g_web_filter++;
    host_log("sceNpWebApiCreatePushEventFilter lib=%d n=%llu -> %d", lib,
             static_cast<unsigned long long>(n), id);
    return id;
}
GUEST_ABI int hle_webapi_delete_filter(int, int) { return 0; }
GUEST_ABI int hle_webapi_reg_push(int ctx, int filter, void*, void*) {
    std::lock_guard<std::mutex> lock(g_web_mu);
    const int id = g_web_cb++;
    host_log("sceNpWebApiRegisterPushEventCallback ctx=%d filter=%d -> %d", ctx, filter, id);
    return id;
}
GUEST_ABI int hle_webapi_unreg_push(int, int) { return 0; }
GUEST_ABI int hle_webapi_create_req(int ctx, const char* group, const char* path, int method,
                                   const void*, std::int64_t* req_id) {
    if (!path || !req_id) {
        return kWebApiInvalid;
    }
    std::lock_guard<std::mutex> lock(g_web_mu);
    const std::int64_t id = g_web_req++;
    WebApiReq r{};
    r.ctx = ctx;
    r.body = webapi_body_for_path(path);
    g_web_reqs[id] = std::move(r);
    *req_id = id;
    static int logs;
    if (logs < 8) {
        host_log("sceNpWebApiCreateRequest ctx=%d group=%s path=%s method=%d -> %lld", ctx,
                 group ? group : "", path, method, static_cast<long long>(id));
        ++logs;
    }
    return 0;
}
GUEST_ABI int hle_webapi_delete_req(std::int64_t req) {
    std::lock_guard<std::mutex> lock(g_web_mu);
    g_web_reqs.erase(req);
    return 0;
}
GUEST_ABI int hle_webapi_abort_req(std::int64_t req) { return hle_webapi_delete_req(req); }
GUEST_ABI int hle_webapi_send(std::int64_t req, const void*, std::uint64_t) {
    std::lock_guard<std::mutex> lock(g_web_mu);
    return g_web_reqs.count(req) ? 0 : kWebApiNoReq;
}
GUEST_ABI int hle_webapi_status(std::int64_t req, int* code) {
    if (!code) {
        return kWebApiInvalid;
    }
    std::lock_guard<std::mutex> lock(g_web_mu);
    auto it = g_web_reqs.find(req);
    if (it == g_web_reqs.end()) {
        return kWebApiNoReq;
    }
    *code = it->second.status;
    return 0;
}
GUEST_ABI int hle_webapi_read(std::int64_t req, void* data, std::uint64_t size) {
    std::lock_guard<std::mutex> lock(g_web_mu);
    auto it = g_web_reqs.find(req);
    if (it == g_web_reqs.end()) {
        return kWebApiNoReq;
    }
    WebApiReq& r = it->second;
    if (!data || size == 0 || r.read_off >= r.body.size()) {
        return 0;
    }
    std::size_t n = static_cast<std::size_t>(size);
    const std::size_t left = r.body.size() - r.read_off;
    if (n > left) {
        n = left;
    }
    std::memcpy(data, r.body.data() + r.read_off, n);
    r.read_off += n;
    return static_cast<int>(n);
}
GUEST_ABI int hle_webapi_hdr_len(std::int64_t req, const char* name, std::uint64_t* len) {
    if (!name || !len) {
        return kWebApiInvalid;
    }
    std::lock_guard<std::mutex> lock(g_web_mu);
    if (!g_web_reqs.count(req)) {
        return kWebApiNoReq;
    }
    if (std::strcmp(name, "Content-Type") == 0 || std::strcmp(name, "content-type") == 0) {
        *len = sizeof(kWebApiJsonType);
        return 0;
    }
    return kWebApiInvalid;
}
GUEST_ABI int hle_webapi_hdr_val(std::int64_t req, const char* name, char* buf, std::uint64_t n) {
    if (!name || !buf || n == 0) {
        return kWebApiInvalid;
    }
    std::lock_guard<std::mutex> lock(g_web_mu);
    if (!g_web_reqs.count(req)) {
        return kWebApiNoReq;
    }
    if (std::strcmp(name, "Content-Type") != 0 && std::strcmp(name, "content-type") != 0) {
        return kWebApiInvalid;
    }
    std::snprintf(buf, static_cast<std::size_t>(n), "%s", kWebApiJsonType);
    return 0;
}
GUEST_ABI int hle_webapi_parse_npid(const char* json, void* npid) {
    if (!json || !npid) {
        return kWebApiInvalid;
    }
    const char* s = json;
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') {
        ++s;
    }
    char online[17]{};
    auto take_quoted = [&](const char* p) -> bool {
        const char* q = std::strchr(p, '"');
        if (!q) {
            return false;
        }
        ++q;
        const char* e = std::strchr(q, '"');
        if (!e) {
            return false;
        }
        const std::size_t n = static_cast<std::size_t>(e - q);
        if (n == 0 || n >= sizeof(online)) {
            return false;
        }
        std::memcpy(online, q, n);
        online[n] = 0;
        return true;
    };
    if (*s == '{') {
        const char* key = std::strstr(s, "onlineId");
        if (!key) {
            key = std::strstr(s, "\"data\"");
        }
        if (!key || !take_quoted(key)) {
            return kWebApiInvalid;
        }
    } else if (*s == '"') {
        if (!take_quoted(s)) {
            return kWebApiInvalid;
        }
    } else {
        std::snprintf(online, sizeof(online), "%s", s);
    }
    hle_np_fill_npid(npid, online);
    return 0;
}


std::mutex g_lookup_mu;
int g_lookup_ctx = 1;
int g_lookup_req = 1;
GUEST_ABI int hle_lookup_title_ctx(const void*) {
    std::lock_guard<std::mutex> lock(g_lookup_mu);
    const int id = g_lookup_ctx++;
    host_log("sceNpLookupCreateTitleCtx -> %d", id);
    return id;
}
GUEST_ABI int hle_lookup_delete_ctx(int) { return 0; }
GUEST_ABI int hle_lookup_async(int, const void*) {
    std::lock_guard<std::mutex> lock(g_lookup_mu);
    return g_lookup_req++;
}
GUEST_ABI int hle_lookup_delete_req(int) { return 0; }
GUEST_ABI int hle_lookup_abort(int) { return 0; }
GUEST_ABI int hle_lookup_poll(int, int* result) {
    if (result) {
        *result = 0;
    }
    return 0;
}
GUEST_ABI int hle_lookup_npid(int, const void* online, void* npid, void*) {
    if (!online || !npid) {
        return static_cast<int>(0x80550003);
    }
    hle_np_fill_npid(npid, static_cast<const char*>(online));
    return 0;
}

// The buttons the last scePadRead handed the game, for the port's own menu
// rows, which act on a press the game also sees (engine/option_menu.cpp).
std::atomic<std::uint32_t> g_pad_delivered{0};

GUEST_ABI int hle_pad_init() { return 0; }
GUEST_ABI int hle_pad_open(int user, int type, int index, const void*) {
    host_log("scePadOpen user=%d type=%d index=%d -> 1", user, type, index);
    return 1;
}
GUEST_ABI int hle_pad_close(int) { return 0; }

// ScePadData (120 bytes): buttons u32 @0; leftStick u8 x,y @4; rightStick @6;
// analogButtons l2,r2 @8 + pad; orientation f32x4 @12; acceleration f32x3 @28;
// angularVelocity f32x3 @40; touchData @52 (touchNum u8, res[3], res1 u32,
// touch[2]{x u16,y u16,id u8,res[3]}); connected bool @76; timestamp u64 @80;
// extensionUnitData[12] @88; connectedCount u8 @100; reserve[2]; deviceUniqueDataLen @103;
// deviceUniqueData[12] @104.
}  // namespace

std::uint32_t hle_pad_delivered_buttons() { return g_pad_delivered.load(std::memory_order_relaxed); }

namespace {

struct PadTap {
    std::uint32_t button;
    std::chrono::steady_clock::time_point from, until;
};
std::mutex g_taps_mu;
std::vector<PadTap> g_taps;  // under g_taps_mu

// The wheel's notches as presses of mouse inputs 6 (up) and 7 (down): bits
// 0x20 and 0x40 of the buttons the bindings read. The notches are counted
// whole (SDL's high-resolution wheels report fractions) and a few at most are
// kept, so a long spin does not go on pressing after the wheel stops.
std::uint32_t wheel_presses(float notches) {
    using Clock = std::chrono::steady_clock;
    constexpr auto kPress = std::chrono::milliseconds(70), kGap = std::chrono::milliseconds(40);
    static std::mutex mu;
    static float part = 0.0f;
    static int up = 0, down = 0;
    static std::uint32_t bit = 0;
    static Clock::time_point until, gap_until;
    std::lock_guard<std::mutex> lock(mu);
    part += notches;
    while (part >= 1.0f) {
        part -= 1.0f;
        up = std::min(up + 1, 4);
    }
    while (part <= -1.0f) {
        part += 1.0f;
        down = std::min(down + 1, 4);
    }
    const Clock::time_point now = Clock::now();
    if (bit && now >= until) {
        bit = 0;
        gap_until = now + kGap;
    }
    if (!bit && now >= gap_until && (up || down)) {
        bit = up ? 0x20u : 0x40u;
        --(up ? up : down);
        until = now + kPress;
        static int logs = 0;
        if (logs < 6) {
            ++logs;
            host_log("mouse: the wheel %s, a press of what is bound to it", bit == 0x20u ? "up" : "down");
        }
    }
    return bit;
}

GUEST_ABI int hle_pad_read(int handle, std::uint8_t* st) {
    if (!st || handle != 1) {
        return static_cast<int>(0x80920002u);  // SCE_PAD_ERROR_INVALID_HANDLE
    }
    std::memset(st, 0, 120);
    PadState p = host_pad_state();
    const std::uint32_t device_buttons = p.buttons;  // pad and keyboard, before the mouse
    // BBHOST_MOUSE_MENU=1: the pointer drives the menus through the pad, which
    // is the only channel the game reads - a click is Cross, right-click is
    // Circle, the wheel is Up and Down. This is navigation, not a cursor:
    // driving the *selection* from a position needs the menu's item rectangles,
    // and this is what works without them.
    //
    // On by default (since 2026-10-02): the press below only goes out while a
    // menu is open (menu_pointer_in_menu), so a click in the world swings the
    // weapon, never a menu. The options screen owns the setting
    // (BBHOST_MOUSE_MENU still overrides it), and while that screen is open it
    // is the pointer's only consumer - two readers would eat each other's edges.
    // The mouse turns the camera. DS3 does this with CSMouseMan: it reads the
    // OS cursor, feeds the movement to the camera and SetCursorPos-es back to
    // the middle of the client rect every frame so the pointer never leaves.
    // SDL's relative mode is the same thing without the round trip, so the
    // host only has to decide *when* - and it releases while the options
    // screen is open, which is the guaranteed way back to a pointer.
    {
        // **Automatic**, the way DS3 does it. CSMouseMan does not ask the
        // player whether the mouse should turn the camera: it turns the camera
        // whenever a menu is not up, and gives the pointer back when one is.
        // A "mouse camera" setting defaulting to off meant the cursor sat on
        // screen in gameplay doing nothing, which is not a mode any PC game
        // offers. The setting now only *disables* the behaviour for someone
        // who wants the pad to own the camera.
        const HostSettings hs = host_settings();
        const bool want = hs.mouse_camera && !host_options_open() && !ingame_menu_open() &&
                          !menu_pointer_in_menu() && !host_text_entry_open();
        // What the follow camera turns by, and whether its own turns as the
        // character moves are held: the keyboard and mouse used last, the
        // mouse turning the camera, and Mouse auto-rotation off. A controller
        // picked up gives the game's camera back (engine/mouse_camera.h).
        mouse_camera_publish(hs.mouse_sens, hs.mouse_invert_x, hs.mouse_invert_y,
                             !hs.mouse_auto_rotation && hs.mouse_camera &&
                                 host_input_device() == InputDevice::KeyboardMouse);
        if (want != host_mouse_relative()) {
            host_mouse_set_relative(want);
        }
        // On **change**, not on the first few polls. Capped-at-N logging
        // reported "camera mode off" three times during boot - when a menu
        // really was open - and read as the feature being broken while it was
        // working. The same trap this project hit with held buttons.
        static int cam_was = -1;
        if (cam_was != static_cast<int>(want)) {
            cam_was = static_cast<int>(want);
            host_log("mouse: camera %s (in a menu: %s)", want ? "on, pointer hidden" : "off",
                     menu_pointer_in_menu() ? "yes" : "no");
        }
        if (want) {
            const MouseState m = host_mouse_state();
            const auto axis = [](float v) {
                const float c = static_cast<float>(kStickRest) + v;
                return static_cast<std::uint8_t>(c < kStickLow ? kStickLow : c > kStickHigh ? kStickHigh : c);
            };
            // The mouse turns the camera itself, as DS3's does: an angle a
            // count added in the follow camera's update, not a stick
            // deflection here (engine/mouse_camera.h). The stick only carries
            // the one thing DS3's mouse does with a target locked: a flick
            // that switches it. Where that hook could not be placed (not the
            // 1.09 eboot), the stick is the mouse's way in, as before.
            float fx = 0.0f, fy = 0.0f;
            if (mouse_camera_flick(fx, fy)) {
                p.rx = axis(fx * 127.0f);
                p.ry = axis(fy * 127.0f);
                static int flogs = 0;
                if (flogs < 8) {
                    ++flogs;
                    host_log("mouse: the lock-on flick on the right stick: %u,%u", p.rx, p.ry);
                }
            } else if (!mouse_camera_installed() && (m.dx != 0.0f || m.dy != 0.0f)) {
                const float sx = hs.mouse_invert_x ? -1.0f : 1.0f;
                const float sy = hs.mouse_invert_y ? -1.0f : 1.0f;
                p.rx = axis(static_cast<float>(p.rx) - kStickRest + sx * m.dx * 1.1f);
                p.ry = axis(static_cast<float>(p.ry) - kStickRest + sy * m.dy * 1.1f);
            }
            // And the buttons, because a mouse that turns the camera and does
            // nothing when clicked is half a mouse. Each is bound to an action
            // (host/bindings.h): by default left attacks, right is the firearm -
            // which is also the parry - and the middle button transforms the
            // weapon, Bloodborne having no shield.
            //
            // These follow camera mode rather than having a setting of their
            // own: if the mouse is driving the camera it is being used as a
            // gameplay device, and if it is not, the menu mapping owns the
            // buttons instead. The two are mutually exclusive by construction.
            // Shift is the strong-attack modifier, as it is in DS3: one key
            // rather than four more bindings.
            bool held[kBindCount];
            // `pressed` as well: a click that starts and ends between two reads
            // is still a press, for one read, rather than nothing. And the
            // wheel, which has no held state of its own: each notch is a
            // press of whatever is bound to it (host/bindings.h), long enough
            // for the game to see at 30 fps and with a gap before the next, so
            // two notches are two presses.
            host_bindings_mouse_held(m.buttons | m.pressed | wheel_presses(m.wheel), held);
            host_bindings_apply(held, host_key_strong(), p);
            // On a change only: a held button is seen on every pad read, and
            // logging each one spent the whole budget on the first press.
            static std::uint32_t last_buttons = 0;
            static int blogs = 0;
            if (m.buttons != last_buttons && blogs < 12) {
                last_buttons = m.buttons;
                ++blogs;
                host_log("mouse: camera buttons %x -> pad %04x", m.buttons, p.buttons);
            }
        }
    }
    // The keys that always work a menu (host/window.h), only while one is -
    // the debug menu included, which is up over the world.
    if (p.menu_buttons && (menu_pointer_in_menu() || debug_menu_open())) {
        p.buttons |= p.menu_buttons;
    }
    // A text box over the game (the IME dialog's, host/window.h) holds the
    // pointer as it holds the pad: a click or a wheel notch there is the
    // box's, so it is taken and dropped rather than kept latched for the menu
    // behind to act on once the box has closed. F10's screen, when it is up
    // as well, is the pointer's reader instead.
    const bool typing = host_text_entry_open();
    // The pointer drives the menus only while one is open - otherwise a click
    // in the world would send Circle at whatever the last menu was.
    if (host_settings().mouse_menu && !host_options_open() && !host_mouse_relative() &&
        menu_pointer_in_menu()) {
        // Held for a few polls each: a load hitch that skips polls would
        // otherwise drop the press, the same reason the autopress taps hold.
        static std::uint32_t held = 0;
        static int hold_polls = 0;
        static std::uint64_t hold_until_flip = 0;
        static std::uint32_t repeat = 0;
        static float rx = 0.0f, ry = 0.0f;
        const MouseState m = host_mouse_state();
        if (typing) {
            // So does a press still held from the click that opened the box:
            // its hold only counts down here, and kept through the box it
            // reached the menu the moment the box closed - the game opened
            // the name box again over the name just typed.
            held = repeat = 0;
            hold_polls = 0;
            hold_until_flip = 0;
            menu_pointer_take_press();
        } else {
            std::uint32_t want = 0;
            // Which button decides is the game's region's convention - Cross in
            // the US build, Circle in the Japanese region this host reported by
            // mistake until 2026-09-18 - so the pointer asks the game
            // (menu_confirm_button) rather than assuming either.
            //
            // A click is a decide **only on an item**, as in DS3: it is handed to
            // the menu's list update (engine/menu_pointer.h), which hit-tests it,
            // moves the cursor there, and answers here a frame later. A click on
            // empty space does nothing.
            if (m.pressed & 1u) {
                menu_pointer_click();
            }
            // DS3's arrow selector steps on its click action while it is *held*
            // (sub_140ac8860 reads action 0x13 through the held/repeat query), so
            // holding the button on an arrow keeps stepping. The same here: a
            // click the menu resolved to Left or Right stays pressed while the
            // button does and the pointer stays put, and the game's own
            // auto-repeat on a held direction does the stepping.
            const std::uint32_t pressed_now = menu_pointer_take_press();
            want |= pressed_now;
            if (pressed_now & 0xa0u) {
                repeat = pressed_now & 0xa0u;
                rx = m.x;
                ry = m.y;
            }
            if (repeat && (!(m.buttons & 1u) || std::fabs(m.x - rx) > 24.0f || std::fabs(m.y - ry) > 24.0f)) {
                repeat = 0;
            }
            if (m.pressed & 2u) want |= menu_back_button();  // right returns
            if (m.wheel > 0.0f) want |= 0x10u;     // wheel up -> Up
            if (m.wheel < 0.0f) want |= 0x40u;     // wheel down -> Down
            if (want) {
                held = want;
                hold_polls = 8;
                // The menus sample the pad once a frame, so a press must outlive
                // a frame: 8 polls (~34 ms) is two frames at 60 fps and none at
                // all in a paired run at 9 fps, where a click on Use did nothing.
                hold_until_flip = hle_video_flip_count() + 2;
                static int logs = 0;
                if (logs < 12) {
                    ++logs;
                    host_log("mouse: menu press %04x from the pointer at %.0f,%.0f", want, m.x, m.y);
                }
            }
            if (hold_polls > 0 || hle_video_flip_count() < hold_until_flip) {
                if (hold_polls > 0) --hold_polls;
                p.buttons |= held;
            }
            p.buttons |= repeat;
            // Hover is not here: it happens inside the menu's own list update, where
            // the game moves its cursor (engine/menu_pointer.h).
        }
    } else if (typing && !host_options_open()) {
        host_mouse_state();  // the box's, outside a menu as well
    }
    // BBHOST_AUTOPRESS=1: headless test aid; after 25 s tap Cross, Circle,
    // Options and Down in turn every 1.5 s so title/menu screens advance.
    // BBHOST_AUTOPRESS="25:circle,29:down,30.5:circle,...": scripted taps
    // (seconds since start; names cross circle square triangle options up
    // down left right l1 r1 l2 r2 l3 r3 touch, and the sticks lup ldown
    // lleft lright, rup rdown rleft rright). Each is held for 500 ms so a load
    // hitch that skips a couple of pad polls does not drop the press.
    // "f330:circle" taps at flip 330 instead and holds 15 flips: runs whose
    // loads take different wall time then press at the same point in the game.
    struct Tap {
        std::uint32_t ms;
        std::uint32_t button;
        bool by_flip;
        std::uint64_t flip;
        std::uint32_t hold_ms;  // 0 = the default
    };
    static const std::vector<Tap> script = [] {
        std::vector<Tap> v;
        const char* e = std::getenv("BBHOST_AUTOPRESS");
        if (!e || !std::strchr(e, ':')) {
            return v;
        }
        std::string spec(e);
        std::size_t pos = 0;
        while (pos < spec.size()) {
            std::size_t end = spec.find(',', pos);
            if (end == std::string::npos) {
                end = spec.size();
            }
            std::string item = spec.substr(pos, end - pos);
            pos = end + 1;
            std::size_t colon = item.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            // "f<flip>:<name>" taps at a flip instead of a time, so runs whose
            // loads take different wall time press at the same point in the game.
            Tap t{0, 0, item[0] == 'f', 0, 0};
            if (t.by_flip) {
                t.flip = std::strtoull(item.c_str() + 1, nullptr, 10);
            } else {
                t.ms = static_cast<std::uint32_t>(std::strtod(item.c_str(), nullptr) * 1000.0);
            }
            // "t:button" or "t:button:hold_ms". A tap's hold has to vary: long
            // enough that a boot or load transition cannot drop it, short
            // enough that a menu cursor does not auto-repeat and move several
            // rows. No single value is both, and picking wrong costs a whole
            // run, so each tap carries its own.
            std::string name = item.substr(colon + 1);
            const std::size_t colon2 = name.find(':');
            if (colon2 != std::string::npos) {
                t.hold_ms = static_cast<std::uint32_t>(std::strtoul(name.c_str() + colon2 + 1, nullptr, 10));
                name = name.substr(0, colon2);
            }
            t.button = hle_pad_button_named(name);
            if (t.button) {
                v.push_back(t);
            }
        }
        return v;
    }();
    static const bool autopress = [] {
        const char* e = std::getenv("BBHOST_AUTOPRESS");
        return e && e[0] != 0;
    }();
    // How long a tap is held. 500 ms is right for gameplay, where a load hitch
    // can skip several pad polls and a short press would be dropped. It is
    // wrong for menus: a cursor auto-repeats while held, so one tap moves an
    // unpredictable number of rows and a scripted walk lands somewhere
    // different every run. BBHOST_AUTOPRESS_HOLD_MS=120 makes menu navigation
    // deterministic; the flip form scales with it.
    static const long hold_ms = [] {
        const char* e = std::getenv("BBHOST_AUTOPRESS_HOLD_MS");
        const long v = e ? std::strtol(e, nullptr, 10) : 0;
        return v > 0 ? v : 500;
    }();

    // Taps the port itself asks for (hle_pad_tap): a press the game reads as
    // the player's, for a few polls.
    {
        std::lock_guard<std::mutex> lk(g_taps_mu);
        const auto now = std::chrono::steady_clock::now();
        for (auto it = g_taps.begin(); it != g_taps.end();) {
            if (now >= it->until) {
                it = g_taps.erase(it);
                continue;
            }
            if (now >= it->from) p.buttons |= it->button;
            ++it;
        }
    }
    if (autopress) {
        static const auto t0 = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        p.connected = true;
        if (!script.empty()) {
            static std::size_t logged = 0;
            for (std::size_t i = 0; i < script.size(); ++i) {
                const Tap& t = script[i];
                const std::uint64_t flip = hle_video_flip_count();
                // Held 500 ms, or 15 flips (500 ms at the game's 30 fps).
                const long hold = t.hold_ms ? static_cast<long>(t.hold_ms) : hold_ms;
                const std::uint64_t holdf = static_cast<std::uint64_t>(hold * 30 / 1000) + 1;
                if (t.by_flip ? flip >= t.flip && flip < t.flip + holdf
                              : ms >= t.ms && ms < t.ms + hold) {
                    // The stick names (bits above the pad's) deflect a stick; walking wants a
                    // hold of seconds ("f3000:lup:5000"), turning the camera the right stick's
                    // ("rright:3000") - what a mouse soak's camera turns are on a pad.
                    if (t.button & 0x1ef0000u) {
                        if (t.button & 0x10000u) p.ly = 0;
                        if (t.button & 0x20000u) p.ly = 255;
                        if (t.button & 0x40000u) p.lx = 0;
                        if (t.button & 0x80000u) p.lx = 255;
                        if (t.button & 0x200000u) p.ry = 0;
                        if (t.button & 0x400000u) p.ry = 255;
                        if (t.button & 0x800000u) p.rx = 0;
                        if (t.button & 0x1000000u) p.rx = 255;
                    } else {
                        p.buttons |= t.button;
                    }
                    if (i >= logged) {
                        logged = i + 1;
                        host_log("autopress: t=%.1fs button 0x%x at flip %llu", ms / 1000.0, t.button,
                                 static_cast<unsigned long long>(flip));
                    }
                }
            }
        } else if (ms > 25000 && (ms % 1500) < 120) {
            static const std::uint32_t seq[4] = {0x4000u, 0x2000u, 0x8u, 0x40u};
            p.buttons |= seq[(ms / 1500) % 4];
        }
        p.timestamp = static_cast<std::uint64_t>(ms) * 1000;
    }
    // BBHOST_TRACE_PAD=1: every read that sees a button change, with the
    // flip count and the reads-per-flip rate (double-press investigations).
    static const bool trace_pad = [] {
        const char* e = std::getenv("BBHOST_TRACE_PAD");
        return e && e[0] == '1';
    }();
    if (trace_pad) {
        static std::uint32_t last_buttons = 0;
        static std::uint64_t reads = 0, last_flip = 0, reads_at_flip = 0;
        ++reads;
        const std::uint64_t flip = hle_video_flip_count();
        if (flip != last_flip) {
            reads_at_flip = reads;
            last_flip = flip;
        }
        static std::uint8_t last_touches = 0;
        if (p.touch_count != last_touches) {
            last_touches = p.touch_count;
            if (p.touch_count) {
                host_log("pad: %u touch(es), first at %u,%u id %u", p.touch_count, p.touch[0].x,
                         p.touch[0].y, p.touch[0].id);
            } else {
                host_log("pad: no touches");
            }
        }
        if (p.buttons != last_buttons) {
            host_log("pad: buttons 0x%05x -> 0x%05x at flip %llu (read %llu, %llu reads since that flip, ts %llu us)", last_buttons, p.buttons,
                     static_cast<unsigned long long>(flip), static_cast<unsigned long long>(reads),
                     static_cast<unsigned long long>(reads - reads_at_flip), static_cast<unsigned long long>(p.timestamp));
            last_buttons = p.buttons;
        }
    }
    // BBHOST_PAD_LOG=1: every change in the buttons the game receives, with a
    // millisecond clock - for finding which press closed or opened something.
    {
        static const bool pad_log = [] {
            const char* e = std::getenv("BBHOST_PAD_LOG");
            return e && e[0] == '1';
        }();
        static std::uint32_t last = 0;
        if (pad_log && p.buttons != last) {
            static const auto t0 = std::chrono::steady_clock::now();
            host_log("pad: %lld ms buttons %05x (was %05x), %05x of it from the pad or keyboard",
                     static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                std::chrono::steady_clock::now() - t0)
                                                .count()),
                     p.buttons, last, device_buttons);
            last = p.buttons;
        }
    }
    // A key binding being chosen (host/bindings.h): the game sees an idle pad,
    // whatever the mouse or the pointer would have added, until the key that
    // ended it is released.
    if (host_bind_capture_blocking()) {
        const bool connected = p.connected;
        const std::uint64_t ts = p.timestamp;
        p = PadState{};
        p.connected = connected;
        p.timestamp = ts;
    }
    g_pad_delivered.store(p.buttons, std::memory_order_relaxed);
    // The Debug Menu key, on the thread that reads the pad (engine/debug_menu.h).
    // BBHOST_DEBUG_MENU_AT=N (a test aid) presses it once at flip N; the pad
    // then works the open menu, so BBHOST_AUTOPRESS can walk its pages.
    static const std::uint64_t debug_at = [] {
        const char* e = std::getenv("BBHOST_DEBUG_MENU_AT");
        return e && e[0] ? std::strtoull(e, nullptr, 0) : 0ull;
    }();
    static bool debug_pressed = false;
    const bool scripted = debug_at && !debug_pressed && hle_video_flip_count() >= debug_at;
    if (scripted) debug_pressed = true;
    if (host_bind_take_debug_toggle() || scripted) {
        debug_menu_toggle();
    }
    std::memcpy(st + 0, &p.buttons, 4);
    st[4] = p.lx;
    st[5] = p.ly;
    st[6] = p.rx;
    st[7] = p.ry;
    st[8] = p.l2;
    st[9] = p.r2;
    const float orient[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    std::memcpy(st + 12, orient, 16);
    const float accel[3] = {0.0f, -1.0f, 0.0f};
    std::memcpy(st + 28, accel, 12);
    // touchData @52: touchNum u8, reserve[3], reserve1 u32, then two
    // ScePadTouch { u16 x, u16 y, u8 id, u8 reserve[3] } at 60 and 68. The
    // coordinates are the touchpad surface's own pixels, the resolution
    // scePadGetControllerInformation reports, so the two agree by
    // construction (kPadTouchW/kPadTouchH).
    st[52] = p.touch_count;
    for (std::uint8_t i = 0; i < p.touch_count && i < 2; ++i) {
        std::uint8_t* t = st + 60 + i * 8;
        std::memcpy(t + 0, &p.touch[i].x, 2);
        std::memcpy(t + 2, &p.touch[i].y, 2);
        t[4] = p.touch[i].id;
    }
    st[76] = p.connected ? 1 : 0;
    std::memcpy(st + 80, &p.timestamp, 8);
    st[100] = p.connected ? 1 : 0;
    // BBHOST_PAD_DUMP=1: the whole record but its timestamp, on every change.
    {
        static const bool dump = [] {
            const char* e = std::getenv("BBHOST_PAD_DUMP");
            return e && e[0] == '1';
        }();
        static std::uint8_t last[120];
        static bool any = false;
        std::uint8_t now[120];
        std::memcpy(now, st, 120);
        std::memset(now + 80, 0, 8);
        if (dump && (!any || std::memcmp(now, last, 120) != 0)) {
            any = true;
            std::memcpy(last, now, 120);
            char hex[241];
            for (int i = 0; i < 120; ++i) std::snprintf(hex + i * 2, 3, "%02x", now[i]);
            const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            host_log("paddump: t=%.4f %s", t, hex);
        }
    }
    return 0;
}
// ScePadVibrationParam { u8 largeMotor; u8 smallMotor; }
GUEST_ABI int hle_pad_vibrate(int, const std::uint8_t* param) {
    if (param) {
        host_pad_rumble(param[1], param[0]);
    }
    return 0;
}
// ScePadControllerInformation { touchPadInfo { f32 pixelDensity; u16 resX, resY }; stickInfo { u8 deadZoneLeft,
//   deadZoneRight }; u8 connectionType; u8 connectedCount; bool connected; u32 deviceClass; u8 reserved[8] }
GUEST_ABI int hle_pad_info(int, std::uint8_t* info) {
    if (!info) {
        return static_cast<int>(0x80920001u);
    }
    std::memset(info, 0, 28);
    const float density = 44.86f;
    std::memcpy(info + 0, &density, 4);
    const std::uint16_t res[2] = {kPadTouchW, kPadTouchH};
    std::memcpy(info + 4, res, 4);
    info[8] = 13;   // left dead zone
    info[9] = 13;   // right dead zone
    info[10] = 0;   // SCE_PAD_CONNECTION_TYPE_LOCAL
    info[11] = 1;   // connectedCount
    info[12] = host_pad_state().connected ? 1 : 0;
    return 0;
}

}  // namespace

void hle_np_set_online_id(const char* id) {
    if (id && id[0]) g_online_id = id;
}

void hle_register_system() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
    REG("sceSysmoduleLoadModule", hle_sysmodule_load);
    REG("sceSysmoduleUnloadModule", hle_sysmodule_unload);
    REG("sceSysmoduleIsLoaded", hle_sysmodule_loaded);
    REG("sceAppContentInitialize", hle_appcontent_init);
    REG("sceAppContentAppParamGetInt", hle_appcontent_param_int);
    REG("sceAppContentGetAddcontInfoList", hle_appcontent_addcont);
    REG("sceUserServiceInitialize", hle_user_init);
    REG("sceUserServiceTerminate", hle_user_term);
    REG("sceUserServiceGetInitialUser", hle_user_initial);
    REG("sceUserServiceGetLoginUserIdList", hle_user_list);
    REG("sceUserServiceGetUserName", hle_user_name);
    REG("sceUserServiceGetEvent", hle_user_event);
    REG("sceSystemServiceParamGetInt", hle_sysparam_int);
    REG("sceSystemServiceGetStatus", hle_sys_status);
    REG("sceSystemServiceReceiveEvent", hle_sys_event);
    REG("sceSystemServiceHideSplashScreen", hle_sys_hide_splash);
    REG("sceCommonDialogInitialize", hle_common_dialog_init);
    REG("sceNpSetContentRestriction", hle_np_restriction);
    REG("sceNpSetNpTitleId", hle_np_title);
    REG("scePlayGoInitialize", hle_playgo_init);
    REG("scePlayGoOpen", hle_playgo_open);
    REG("scePlayGoGetChunkId", hle_playgo_chunk_id);
    REG("scePlayGoSetInstallSpeed", hle_playgo_install_speed);
    REG("scePlayGoGetLocus", hle_playgo_locus);
    REG("sceRtcGetDayOfWeek", hle_rtc_dow);
    REG("sceRtcGetCurrentClockLocalTime", hle_rtc_local);
    REG("sceRtcGetTick", hle_rtc_get_tick);
    REG("sceRtcSetTick", hle_rtc_set_tick);
    REG("sceRtcGetTickResolution", hle_rtc_tick_res);
    REG("sceRtcGetCurrentNetworkTick", hle_rtc_network_tick);
    REG("sceRtcSetTime_t", hle_rtc_set_time_t);
    REG("sceRtcConvertUtcToLocalTime", hle_rtc_utc_to_local);
    REG("sceRtcConvertLocalTimeToUtc", hle_rtc_local_to_utc);
    REG("sceRtcFormatRFC2822LocalTime", hle_rtc_format_rfc2822);
    REG("sceNpRegisterStateCallback", hle_np_reg_state);
    REG("sceNpUnregisterStateCallback", hle_np_unreg_state);
    REG("sceNpRegisterGamePresenceCallback", hle_np_reg_presence);
    REG("sceNpGetState", hle_np_get_state);
    REG("sceNpCheckCallback", hle_np_check_cb_noop);
    REG("sceNpCreateAsyncRequest", hle_np_create_async_request);
    REG("sceNpDeleteRequest", hle_np_delete_request);
    REG("sceNpAbortRequest", hle_np_abort_request);
    REG("sceNpPollAsync", hle_np_poll_async);
    REG("sceNpCheckNpAvailability", hle_np_check_availability);
    REG("sceNpGetParentalControlInfo", hle_np_parental);
    REG("sceNpCheckPlus", hle_np_check_plus);
    REG("sceNpRegisterPlusEventCallback", hle_np_plus_cb);
    REG("sceNpUnregisterPlusEventCallback", hle_np_plus_cb);
    REG("sceNpNotifyPlusFeature", hle_np_plus_notify);
    REG("sceNpAuthCreateAsyncRequest", hle_npauth_create_async_request);
    REG("sceNpAuthDeleteRequest", hle_npauth_delete_request);
    REG("sceNpAuthPollAsync", hle_npauth_poll_async);
    REG("sceNpAuthGetAuthorizationCode", hle_npauth_get_code);
    REG("sceNpGetOnlineId", hle_np_online_id);
    REG("sceNpGetNpId", hle_np_get_npid);
    REG("sceNpGetGamePresenceStatus", hle_np_presence_status);
    REG("sceNpCmpOnlineId", hle_np_cmp_online);
    REG("sceNpCmpNpId", hle_np_cmp_npid);
    REG("sceNpScoreCreateNpTitleCtx", hle_score_title_ctx);
    REG("sceNpWebApiInitialize", hle_webapi_init);
    REG("sceNpWebApiTerminate", hle_webapi_term);
    REG("sceNpWebApiCreateContext", hle_webapi_create_ctx);
    REG("sceNpWebApiDeleteContext", hle_webapi_delete_ctx);
    REG("sceNpWebApiCreatePushEventFilter", hle_webapi_create_filter);
    REG("sceNpWebApiDeletePushEventFilter", hle_webapi_delete_filter);
    REG("sceNpWebApiRegisterPushEventCallback", hle_webapi_reg_push);
    REG("sceNpWebApiUnregisterPushEventCallback", hle_webapi_unreg_push);
    REG("sceNpWebApiCreateRequest", hle_webapi_create_req);
    REG("sceNpWebApiDeleteRequest", hle_webapi_delete_req);
    REG("sceNpWebApiAbortRequest", hle_webapi_abort_req);
    REG("sceNpWebApiSendRequest", hle_webapi_send);
    REG("sceNpWebApiGetHttpStatusCode", hle_webapi_status);
    REG("sceNpWebApiReadData", hle_webapi_read);
    REG("sceNpWebApiGetHttpResponseHeaderValueLength", hle_webapi_hdr_len);
    REG("sceNpWebApiGetHttpResponseHeaderValue", hle_webapi_hdr_val);
    REG("sceNpWebApiUtilityParseNpId", hle_webapi_parse_npid);
    REG("sceNpLookupCreateTitleCtx", hle_lookup_title_ctx);
    REG("sceNpLookupDeleteTitleCtx", hle_lookup_delete_ctx);
    REG("sceNpLookupCreateAsyncRequest", hle_lookup_async);
    REG("sceNpLookupDeleteRequest", hle_lookup_delete_req);
    REG("sceNpLookupAbortRequest", hle_lookup_abort);
    REG("sceNpLookupPollAsync", hle_lookup_poll);
    REG("sceNpLookupNpId", hle_lookup_npid);
    REG("sceSaveDataInitialize", hle_save_init);
    REG("sceSaveDataTerminate", hle_save_term);
    REG("sceSaveDataMount", hle_save_mount);
    REG("sceSaveDataUmount", hle_save_umount);
    REG("sceSaveDataDelete", hle_save_delete);
    REG("sceSaveDataDirNameSearch", hle_save_dirname_search);
    REG("sceSaveDataSetParam", hle_save_set_param);
    REG("sceSaveDataSaveIcon", hle_save_save_icon);
    REG("sceSaveDataSetupSaveDataMemory", hle_save_setup_memory);
    REG("sceSaveDataGetSaveDataMemory", hle_save_get_memory);
    REG("sceSaveDataSetSaveDataMemory", hle_save_set_memory);
    REG("sceSaveDataDialogInitialize", hle_save_dialog_init);
    REG("sceSaveDataDialogTerminate", hle_save_dialog_term);
    REG("sceSaveDataDialogOpen", hle_save_dialog_open);
    REG("sceSaveDataDialogUpdateStatus", hle_save_dialog_update);
    REG("sceNpTrophyCreateContext", hle_trophy_ctx);
    REG("sceNpTrophyCreateHandle", hle_trophy_handle);
    REG("sceNpTrophyRegisterContext", hle_trophy_register);
    REG("sceNpTrophyUnlockTrophy", hle_trophy_unlock);
    REG("sceNpTrophyGetGameInfo", hle_trophy_game_info);
    REG("sceNpTrophyGetTrophyInfo", hle_trophy_info);
    REG("sceNpScoreDeleteNpTitleCtx", hle_score_delete_title_ctx);
    REG("sceNpScoreCreateRequest", hle_score_create_request);
    REG("sceNpScoreDeleteRequest", hle_score_delete_request);
    REG("sceNpScoreAbortRequest", hle_score_abort_request);
    REG("sceNpScoreSetPlayerCharacterId", hle_score_set_pc_id);
    REG("sceNpScoreGetRankingByNpIdPcId", hle_score_offline);
    REG("sceNpScoreGetRankingByRange", hle_score_offline);
    REG("sceNpScoreGetGameData", hle_score_offline);
    REG("sceNpScoreRecordScore", hle_score_offline);
    REG("sceNpScoreGetBoardInfo", hle_score_offline);
    REG("sceNpScoreRecordGameData", hle_score_offline);
    REG("sceNpScoreCensorComment", hle_score_comment_ok);
    REG("sceNpScoreSanitizeComment", hle_score_comment_ok);
    REG("sceMouseInit", hle_mouse_init);
    REG("sceMouseOpen", hle_mouse_open);
    REG("sceMouseClose", hle_mouse_close);
    REG("sceMouseRead", hle_mouse_read);
    REG("sceSystemServiceLaunchWebBrowser", hle_launch_web_browser);
    REG("scePadInit", hle_pad_init);
    REG("scePadOpen", hle_pad_open);
    REG("scePadClose", hle_pad_close);
    REG("scePadReadState", hle_pad_read);
    REG("scePadGetControllerInformation", hle_pad_info);
    REG("scePadResetOrientation", hle_ok);
    REG("scePadSetAngularVelocityDeadbandState", hle_ok);
    REG("scePadSetTiltCorrectionState", hle_ok);
    REG("scePadSetVibration", hle_pad_vibrate);
#undef REG
}

std::uint32_t hle_pad_button_named(const std::string& name) {
    static const struct {
        const char* name;
        std::uint32_t bit;
    } names[] = {{"lup", 0x10000u}, {"ldown", 0x20000u}, {"lleft", 0x40000u}, {"lright", 0x80000u},  // left stick, full
                 {"rup", 0x200000u}, {"rdown", 0x400000u}, {"rleft", 0x800000u}, {"rright", 0x1000000u},  // right stick, full
                 {"cross", 0x4000u}, {"circle", 0x2000u}, {"square", 0x8000u}, {"triangle", 0x1000u},
                 {"options", 0x8u}, {"up", 0x10u},        {"down", 0x40u},     {"left", 0x80u},
                 {"right", 0x20u},  {"l1", 0x400u},       {"r1", 0x800u},      {"l2", 0x100u},
                 {"r2", 0x200u},    {"l3", 0x2u},         {"r3", 0x4u},        {"touch", 0x100000u}};
    for (const auto& n : names) {
        if (name == n.name) return n.bit;
    }
    return 0;
}

void hle_pad_tap(std::uint32_t button, int delay_ms, int hold_ms) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_taps_mu);
    g_taps.push_back(PadTap{button, now + std::chrono::milliseconds(delay_ms), now + std::chrono::milliseconds(delay_ms + hold_ms)});
}

// Saves mounted for writing now: a run that has to end waits for none (a
// device-loss exit, which would otherwise cut a save off mid-write).
int hle_save_writable_mounts() {
    std::lock_guard<std::mutex> lock(g_save_mu);
    int n = 0;
    for (const auto& m : g_save_mounts) n += m.used && m.writable;
    return n;
}
