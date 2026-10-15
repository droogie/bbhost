#include "engine/camera.h"

#include "core/portable.h"

#include "core/elf.h"
#include "core/memory.h"
#include "engine/addr.h"
#include "engine/params.h"
#include "hle/modules.h"
#include "host/options.h"
#include "host/settings.h"
#include "log.h"

#include <array>
#include <atomic>
#include <chrono>
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
// index 0x21) - so it is found once by its type name, in the guest's
// CPU-readable mappings, off the render thread.
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
                    Table t{base, 0, {}, {}};
                    std::uint16_t rows = 0;
                    std::memcpy(&rows, base + kRowCountAt, 2);
                    t.rows = rows;
                    t.orig.assign(rows, Scales{0.0f, 0.0f, 0.0f});
                    t.written.assign(rows, Scales{-1.0f, -1.0f, -1.0f});
                    found.push_back(std::move(t));
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
}

// The multipliers the settings object carries (host/settings.h): distance
// and height 10% a step around the game's own, field of view 5% wider a
// step, BBHOST_FOV overriding.
Scales wanted_scales() {
    const HostSettings hs = host_settings();
    return Scales{hs.camera_distance_scale, hs.camera_height_scale, hs.fov_scale};
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
}

void camera_debug_tick() {
    if (!fov_watching()) return;
    // Find the table and watch it; touch nothing.
    int state = g_state.load(std::memory_order_acquire);
    if (state == 0 && g_state.compare_exchange_strong(state, 1)) std::thread(find_tables).detach();
    fov_watch();
}

namespace {

// The repository's LockCamParam as last seen: its rows' own values (read when
// the table is loaded, before anything of ours), what was last written, and
// the scales that was for. Its rows are the file the camera's update reads
// (engine/params.h), so one copy; the scan above found it twice, mapped at
// two addresses.
std::mutex g_apply_mu;
std::vector<void*> g_rows;
std::vector<Scales> g_orig;
Scales g_applied{1.0f, 1.0f, 1.0f};

// Writes orig * k into each row; under g_apply_mu.
void write_rows_locked(const Scales& k) {
    for (std::size_t r = 0; r < g_rows.size(); ++r) {
        auto* row = static_cast<std::uint8_t*>(g_rows[r]);
        for (int f = 0; f < 3; ++f) {
            const float want = g_orig[r][f] * k[f];
            float now = 0.0f;
            std::memcpy(&now, row + kFieldAt[f], 4);
            if (now != want) std::memcpy(row + kFieldAt[f], &want, 4);
        }
    }
    if (k != g_applied) {
        host_log("camera: distance x%g, height x%g, field of view x%g (%zu LockCamParam rows, written once)", k[0], k[1], k[2],
                 g_rows.size());
        g_applied = k;
    }
}

}  // namespace

void camera_table_loaded() {
    if (fov_watching()) return;
    std::uint32_t ids[512];
    std::size_t count = 0;
    if (!params_ids("LockCamParam", ids, sizeof(ids) / sizeof(ids[0]), &count) || count > sizeof(ids) / sizeof(ids[0])) return;
    std::lock_guard<std::mutex> lock(g_apply_mu);
    g_rows.clear();
    g_orig.clear();
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t bytes = 0;
        void* row = params_row("LockCamParam", ids[i], &bytes);
        if (!row || (bytes && bytes < kRowSize)) continue;
        Scales o{};
        for (int f = 0; f < 3; ++f) std::memcpy(&o[f], static_cast<const std::uint8_t*>(row) + kFieldAt[f], 4);
        g_rows.push_back(row);
        g_orig.push_back(o);
    }
    // A loaded table holds the game's own values: ours go in only when the
    // settings are not the game's.
    g_applied = Scales{1.0f, 1.0f, 1.0f};
    const Scales k = wanted_scales();
    if (k != g_applied) write_rows_locked(k);
}

void camera_settings_changed() {
    if (fov_watching()) return;
    const Scales k = wanted_scales();
    std::lock_guard<std::mutex> lock(g_apply_mu);
    if (k == g_applied || g_rows.empty()) return;
    write_rows_locked(k);
}
