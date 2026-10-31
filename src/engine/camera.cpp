#include "engine/camera.h"

#include "core/portable.h"

#include "core/elf.h"
#include "core/memory.h"
#include "engine/addr.h"
#include "hle/modules.h"
#include "host/options.h"
#include "host/settings.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

// The table is param/gameparam's LockCamParam.param, which the game keeps in
// memory as the file it read: a 0x40-byte header whose +0xc is the type name
// "LOCK_CAM_PARAM_ST" and whose +0xa is the row count, then 0x18-byte row
// records (id, data offset at +8), each row 0x20 bytes with camFovY, in
// degrees, at +0x14. Nothing in the eboot points at it directly - the param
// repository reaches it through the name table at 0x5755ad0 (LockCamParam is
// index 0x21). engine/params.cpp walks that repository and names the buffer
// as the game loads it (camera_params_changed); before that - or if it never
// does - it is found by its type name in the guest's CPU-readable mappings,
// off the render thread.
//
// It turns up twice: direct memory mapped at two addresses. Both are one
// table, so the copies are told apart by their physical address - scaling
// "both" compounds the factor, which is what a first version did (the field of
// view ran up to the ceiling and stayed there when the setting came back).
//
// The reader is the follow camera's update, sub_183ac60, every frame: it
// takes the row's camFovY, clamps it to 38..48 degrees, and eases the
// camera's field of view (its +0x50, radians) toward it. The upper clamp is
// `cmovbe rax, rsi` at 0x183af4e choosing between the row's value and a 48
// on the stack; camera_install makes that a plain `mov rax, rsi`, so the
// row's value is taken whenever it is at least 38 - which the shipped rows,
// at 43 and 48, always were.
namespace {

constexpr std::uint64_t kFovClamp = 0x183af4e;
const std::uint8_t kFovClampBytes[] = {0x48, 0x0f, 0x46, 0xc6};  // cmovbe rax, rsi
const std::uint8_t kFovNoClamp[] = {0x48, 0x89, 0xf0, 0x90};      // mov rax, rsi; nop

constexpr char kTypeName[] = "LOCK_CAM_PARAM_ST";
constexpr std::size_t kTypeAt = 0xc, kRowCountAt = 0xa, kRowsAt = 0x40, kRowRecord = 0x18, kRowSize = 0x20;
// The row's fields the settings scale - all three read by the camera's update
// each frame and eased toward: camDistTarget (+0x0, 4 m in most rows),
// chrOrgOffset_Y (+0xc, the point it looks at, 1.42 m above the character)
// and camFovY (+0x14).
constexpr std::size_t kFovAt = 0x14;
constexpr std::size_t kFieldAt[3] = {0x0, 0xc, kFovAt};
using Scales = std::array<float, 3>;

struct Table {
    std::uint8_t* base;
    std::uint32_t rows;
    std::vector<Scales> orig, written;
};

std::mutex g_mu;
std::vector<Table> g_tables;
std::atomic<int> g_state{0};  // 0 not looked, 1 looking, 2 done
const Scales kGame{1.0f, 1.0f, 1.0f};
Scales g_applied = kGame;  // under g_mu: what the rows were last given

// What the settings ask for, kept by their listener (settings_changed), and
// whether a view has work to do. Set by a change of the three settings, by
// the search finishing and by the param repository (re)loading the table
// (camera_params_changed); a view with nothing to do reads this one flag.
std::mutex g_want_mu;
Scales g_want = kGame;  // under g_want_mu
// The wheel's zoom (camera_zoom): a multiplier on the Camera distance
// setting's distance, 1 until the wheel turns. Under g_want_mu.
float g_zoom = 1.0f;
// Each notch 20% further or closer, and the distance the setting and the
// zoom make together kept from half the game's own to six times it (the
// setting alone reaches 0.5 to 1.5).
constexpr float kZoomStep = 1.2f, kZoomNear = 0.5f, kZoomFar = 6.0f;
std::atomic<bool> g_pending{false};
// LockCamParam's file as the param repository holds it - the buffer the
// camera's update reads - from engine/params.cpp's walk, which sees it as the
// game loads its tables at boot; 0 until then (the search below stands in).
std::atomic<std::uint64_t> g_param_file{0};

Table table_at(std::uint8_t* base) {
    Table t{base, 0, {}, {}};
    std::uint16_t rows = 0;
    std::memcpy(&rows, base + kRowCountAt, 2);
    t.rows = rows;
    t.orig.assign(rows, Scales{0.0f, 0.0f, 0.0f});
    t.written.assign(rows, Scales{-1.0f, -1.0f, -1.0f});
    return t;
}

// The param repository's buffer is the file, its rows where the records say:
// checked the way plausible() checks a found one, by reads that cannot fault.
bool repository_file_ok(const std::uint8_t* base) {
    std::uint8_t head[kRowsAt];
    if (!host_read_safe(base, head, sizeof(head)) || std::memcmp(head + kTypeAt, kTypeName, sizeof(kTypeName)) != 0)
        return false;
    std::uint16_t rows = 0;
    std::memcpy(&rows, head + kRowCountAt, 2);
    if (rows == 0 || rows > 4096) return false;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::uint64_t off = 0;
        std::uint8_t row[kRowSize];
        if (!host_read_safe(base + kRowsAt + r * kRowRecord + 8, &off, 8) || off < kRowsAt + rows * kRowRecord ||
            off > (1u << 20) || !host_read_safe(base + off, row, sizeof(row)))
            return false;
        float fov = 0;
        std::memcpy(&fov, row + kFovAt, 4);
        if (!(fov > 1.0f && fov < 179.0f)) return false;
    }
    return true;
}

// A header that is the file's, not the name turning up somewhere else.
bool plausible(const std::uint8_t* base, std::size_t room) {
    std::uint16_t rows = 0;
    std::memcpy(&rows, base + kRowCountAt, 2);
    if (rows == 0 || rows > 4096 || kRowsAt + rows * kRowRecord > room) return false;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::uint64_t off = 0;
        std::memcpy(&off, base + kRowsAt + r * kRowRecord + 8, 8);
        if (off < kRowsAt + rows * kRowRecord || off + kRowSize > room) return false;
        float fov = 0;
        std::memcpy(&fov, base + off + kFovAt, 4);
        if (!(fov > 1.0f && fov < 179.0f)) return false;
    }
    return true;
}

void find_tables() {
    std::vector<GuestMapInfo> maps;
    hle_kernel_snapshot_maps(maps);
    std::vector<Table> found;
    std::vector<std::int64_t> phys_seen;
    std::uint64_t scanned = 0;
    for (const GuestMapInfo& m : maps) {
        if (!(m.prot & 1) || m.len < sizeof(kTypeName)) continue;  // CPU read
        auto* p = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(m.va));
        const std::uint8_t* end = p + m.len;
        scanned += m.len;
        for (const std::uint8_t* q = p; q < end;) {
            q = static_cast<const std::uint8_t*>(host_memmem(q, static_cast<std::size_t>(end - q), kTypeName, sizeof(kTypeName)));
            if (!q) break;
            if (q - p >= static_cast<std::ptrdiff_t>(kTypeAt)) {
                auto* base = const_cast<std::uint8_t*>(q - kTypeAt);
                const std::int64_t phys =
                    m.dmem && m.phys >= 0 ? m.phys + (reinterpret_cast<std::uint64_t>(base) - m.va) : -1;
                bool alias = false;
                for (std::int64_t ph : phys_seen) alias |= phys >= 0 && ph == phys;
                if (!alias && plausible(base, static_cast<std::size_t>(end - base))) {
                    phys_seen.push_back(phys);
                    found.push_back(table_at(base));
                }
            }
            q += sizeof(kTypeName);
        }
    }
    host_log("camera: LockCamParam found %zu time(s) (aliases of one memory counted once) in %llu MiB of guest "
             "memory",
             found.size(), static_cast<unsigned long long>(scanned >> 20));
    std::lock_guard<std::mutex> lock(g_mu);
    g_tables = std::move(found);
    g_state.store(2, std::memory_order_release);
    g_pending.store(true, std::memory_order_release);  // the next view writes the rows
}

// The settings' listener: the multipliers the settings object carries
// (host/settings.h) - distance and height 10% a step around the game's own,
// field of view 5% wider a step, BBHOST_FOV overriding - kept for the next
// view, which has work only when they moved.
void settings_changed(const HostSettings& hs) {
    const Scales k{hs.camera_distance_scale, hs.camera_height_scale, hs.fov_scale};
    std::lock_guard<std::mutex> lock(g_want_mu);
    if (k == g_want) return;  // another setting changed
    g_want = k;
    g_pending.store(true, std::memory_order_release);
}

Scales wanted_scales() {
    std::lock_guard<std::mutex> lock(g_want_mu);
    Scales k = g_want;
    if (g_zoom != 1.0f) k[0] = std::clamp(k[0] * g_zoom, std::min(k[0], kZoomNear), std::max(k[0], kZoomFar));
    return k;
}

}  // namespace

// BBHOST_FOV_WATCH=1: once the table is found, watch reads of its rows and
// report where they come from - to find the camera code that takes camFovY.
bool fov_watching() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_FOV_WATCH");
        return e && (e[0] == '1' || e[0] == '2');
    }();
    return on;
}

void fov_watch() {
    static int ticks = -1;
    static std::chrono::steady_clock::time_point armed;
    if (!fov_watching() || g_state.load(std::memory_order_acquire) != 2) return;
    if (ticks < 0) {
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_tables.empty()) return;
        // BBHOST_FOV_WATCH=1 the first copy, 2 the second.
        const char* e = std::getenv("BBHOST_FOV_WATCH");
        const std::size_t which = e[0] == '2' && g_tables.size() > 1 ? 1 : 0;
        const Table& t = g_tables[which];
        for (const Table& c : g_tables) host_log("camera: LockCamParam copy at %p", static_cast<void*>(c.base));
        std::uint64_t first = 0, last = 0;
        std::memcpy(&first, t.base + kRowsAt + 8, 8);
        std::memcpy(&last, t.base + kRowsAt + (t.rows - 1) * kRowRecord + 8, 8);
        // The rows cross a page boundary; watch the page most of them are on.
        auto lo = reinterpret_cast<std::uint64_t>(t.base) + first;
        const std::uint64_t next_page = (lo + 0x1000) & ~0xfffull;
        if (e[1] != 'a' && next_page - lo < (reinterpret_cast<std::uint64_t>(t.base) + last + kRowSize) - next_page)
            lo = next_page;  // "1a": the first page, rows 0 and 1
        hle_watch_arm_reads(lo, reinterpret_cast<std::uint64_t>(t.base) + last + kRowSize);
        ticks = 0;
        armed = std::chrono::steady_clock::now();
    } else if (ticks == 0 && std::chrono::steady_clock::now() - armed > std::chrono::seconds(10)) {
        ticks = 1;
        hle_watch_report();
    }
}

// The clamp at kFovClamp is taken out by patches/fov-uncap.toml;
// this only says whether it happened, for the log's sake.
void camera_install(ElfImage* image) {
    const std::uint64_t at = image->mem.slide + (kFovClamp - kPreferredGuestSlide);
    const auto* p = static_cast<const std::uint8_t*>(guest_ptr(image->mem, at));
    if (std::memcmp(p, kFovNoClamp, sizeof(kFovNoClamp)) != 0) {
        host_log("camera: the field of view stays capped at 48 degrees (patches/fov-uncap.toml not applied)");
    }
    host_settings_listen(&settings_changed);
}

void camera_zoom(int steps, bool reset) {
    std::lock_guard<std::mutex> lock(g_want_mu);
    const float was = g_zoom;
    if (reset) {
        g_zoom = 1.0f;
    } else {
        // Kept to what the clamp lets through (wanted_scales), so turning
        // back after a long spin past the end moves at once.
        const float base = std::max(g_want[0], 0.01f);
        g_zoom = std::clamp(g_zoom * std::pow(kZoomStep, static_cast<float>(steps)), kZoomNear / base, kZoomFar / base);
    }
    if (g_zoom == was) return;
    g_pending.store(true, std::memory_order_release);
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 8) host_log("camera: zoom x%.2f (the Camera distance setting's x%.2f)", g_zoom, g_want[0]);
}

void camera_params_changed(std::uint64_t file) {
    // Loaded, or reloaded (perhaps into a new buffer, while the old one still
    // reads as the file it was): the next view takes this buffer and writes
    // its rows. No lock: the param walk calls this under its own.
    g_param_file.store(file, std::memory_order_release);
    g_pending.store(true, std::memory_order_release);
}

void camera_tick() {
    if (fov_watching()) {
        // Find the table and watch it; touch nothing.
        int state = g_state.load(std::memory_order_acquire);
        if (state == 0 && g_state.compare_exchange_strong(state, 1)) std::thread(find_tables).detach();
        fov_watch();
        return;
    }
    // Nothing changed since the rows were written: one load, no lock.
    if (!g_pending.load(std::memory_order_acquire)) return;
    std::unique_lock<std::mutex> lock(g_mu, std::try_to_lock);
    if (!lock.owns_lock()) return;  // another view is at it; the flag stays for it
    // Taken before the wish is read: a change made from here on sets it again.
    if (!g_pending.exchange(false, std::memory_order_acq_rel)) return;
    const Scales k = wanted_scales();
    if (k == kGame && g_applied == kGame) return;  // nothing wanted, nothing of ours written
    // The param repository's buffer when it has named one: no search through
    // guest memory (~7 GiB, seconds of a core). What was written carries over
    // to it, so a buffer that is memory already written seen at another
    // address is not scaled twice, while a reload has the game's own values,
    // which differ from what was written and are taken as the new originals.
    if (const std::uint64_t file = g_param_file.load(std::memory_order_acquire);
        file && (g_tables.size() != 1 || reinterpret_cast<std::uint64_t>(g_tables[0].base) != file)) {
        auto* base = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(file));
        if (repository_file_ok(base)) {
            Table t = table_at(base);
            if (g_tables.size() == 1 && g_tables[0].rows == t.rows) {
                t.orig = g_tables[0].orig;
                t.written = g_tables[0].written;
            }
            host_log("camera: LockCamParam at 0x%llx, from the param repository (%u rows)",
                     static_cast<unsigned long long>(file), t.rows);
            g_tables.clear();
            g_tables.push_back(std::move(t));
            g_state.store(2, std::memory_order_release);
        }
    }
    int state = g_state.load(std::memory_order_acquire);
    if (state == 0 && g_state.compare_exchange_strong(state, 1)) {
        std::thread(find_tables).detach();  // sets the flag again when it is done
        return;
    }
    if (state != 2) return;
    if (g_tables.empty()) {
        // Not loaded when it was looked for: look again at the next change
        // or when the param repository loads it (camera_params_changed).
        g_state.store(0, std::memory_order_release);
        return;
    }
    for (const Table& t : g_tables) {
        // Still the table it was: a param reload would move it, and writing
        // through the old address would write into whatever took its place.
        if (std::memcmp(t.base + kTypeAt, kTypeName, sizeof(kTypeName)) != 0) {
            host_log("camera: LockCamParam moved; looking again");
            g_tables.clear();
            g_state.store(0, std::memory_order_release);
            g_pending.store(true, std::memory_order_release);
            return;
        }
    }
    for (Table& t : g_tables) {
        for (std::uint32_t r = 0; r < t.rows; ++r) {
            std::uint64_t off = 0;
            std::memcpy(&off, t.base + kRowsAt + r * kRowRecord + 8, 8);
            for (int f = 0; f < 3; ++f) {
                auto* v = reinterpret_cast<float*>(t.base + off + kFieldAt[f]);
                if (*v != t.written[r][f]) t.orig[r][f] = *v;  // the game's own, first seen or reloaded
                const float want = t.orig[r][f] * k[f];
                if (*v != want) *v = want;
                t.written[r][f] = want;
            }
        }
    }
    if (k != g_applied) {
        host_log("camera: distance x%g, height x%g, field of view x%g (%zu table(s))", k[0], k[1], k[2],
                 g_tables.size());
        g_applied = k;
    }
}
