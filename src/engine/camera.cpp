#include "engine/camera.h"

#include "core/portable.h"

#include "core/elf.h"
#include "core/memory.h"
#include "engine/addr.h"
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
std::atomic<std::uint64_t> g_revision{1};  // discovery and parameter maintenance
std::atomic<bool> g_modified{false};  // another view thread may have applied an override
std::atomic<std::uint64_t> g_applied_serial{0}, g_seen_revision{0}, g_seen_maps{0};
std::atomic<std::uint64_t> g_scanned_revision{0}, g_scanned_maps{0};
const Scales kGame{1.0f, 1.0f, 1.0f};
Scales g_applied = kGame;  // protected by g_mu; all view threads share the table

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
    if (!g_tables.empty()) g_revision.fetch_add(1, std::memory_order_release);
    g_state.store(2, std::memory_order_release);
}

// The multipliers the settings object carries (host/settings.h): distance
// and height 10% a step around the game's own, field of view 5% wider a
// step, BBHOST_FOV overriding.
Scales wanted_scales(const HostSettings& hs) {
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

void camera_refresh() {
    // Parameter reloads can reuse an address and write the same type name.
    // The repository's existing maintenance pass requests an integrity check.
    g_revision.fetch_add(1, std::memory_order_release);
}

void camera_tick() {
    if (fov_watching()) {
        // Find the table and watch it; touch nothing.
        int state = g_state.load(std::memory_order_acquire);
        if (state == 0 && g_state.compare_exchange_strong(state, 1)) std::thread(find_tables).detach();
        fov_watch();
        return;
    }
    HostSettings hs = host_settings();
    Scales k = wanted_scales(hs);
    const bool restore = k == kGame && g_modified.load(std::memory_order_acquire);
    if (k == kGame && !restore) return;
    const std::uint64_t revision = g_revision.load(std::memory_order_acquire);
    const std::uint64_t maps = hle_kernel_maps_generation();
    int state = g_state.load(std::memory_order_acquire);
    if (state == 0) {
        // Do not rescan all guest mappings on every view if the parameters
        // have not loaded yet. Retry on new mappings or parameter maintenance.
        if (g_scanned_revision.load(std::memory_order_acquire) == revision &&
            g_scanned_maps.load(std::memory_order_acquire) == maps) return;
        if (g_state.compare_exchange_strong(state, 1)) {
            g_scanned_revision.store(revision, std::memory_order_release);
            g_scanned_maps.store(maps, std::memory_order_release);
            std::thread(find_tables).detach();
        }
        return;
    }
    if (state != 2) return;
    if (g_applied_serial.load(std::memory_order_acquire) == hs.serial &&
        g_seen_revision.load(std::memory_order_acquire) == revision &&
        g_seen_maps.load(std::memory_order_acquire) == maps) return;
    std::unique_lock<std::mutex> lock(g_mu, std::try_to_lock);
    if (!lock.owns_lock()) return;
    // Re-read after taking the table lock: another view thread may have
    // consumed a newer menu change while this call was waiting.
    hs = host_settings();
    k = wanted_scales(hs);
    if (g_tables.empty()) {
        // A completed discovery with no tables is pending, not permanently
        // applied. A later load must get another chance to find the table.
        // Retain the generations from when discovery started. A mapping or
        // maintenance change during that scan still needs its own retry.
        g_state.store(0, std::memory_order_release);
        return;
    }
    const bool refresh = k != g_applied || g_seen_revision.load(std::memory_order_relaxed) != revision;
    if (!refresh && g_seen_maps.load(std::memory_order_relaxed) == maps) {
        // A different option changed; the camera multipliers did not.
        g_applied_serial.store(hs.serial, std::memory_order_release);
        return;
    }
    for (const Table& t : g_tables) {
        // Still the table it was: a param reload would move it, and writing
        // through the old address would write into whatever took its place.
        std::array<std::uint8_t, kRowsAt> header{};
        std::uint16_t rows = 0;
        const bool readable = host_read_safe(t.base, header.data(), header.size());
        if (readable) std::memcpy(&rows, header.data() + kRowCountAt, sizeof(rows));
        if (!readable || rows != t.rows ||
            std::memcmp(header.data() + kTypeAt, kTypeName, sizeof(kTypeName)) != 0) {
            host_log("camera: LockCamParam moved; looking again");
            g_tables.clear();
            g_scanned_revision.store(0, std::memory_order_release);
            g_state.store(0, std::memory_order_release);
            return;
        }
    }
    g_seen_maps.store(maps, std::memory_order_release);
    // Mapping churn only requires validating the cached header. The row walk
    // runs for a camera change, completed discovery or maintenance request.
    if (!refresh) {
        g_applied_serial.store(hs.serial, std::memory_order_release);
        return;
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
    }
    g_applied = k;
    g_modified.store(k != kGame, std::memory_order_release);
    g_seen_revision.store(revision, std::memory_order_release);
    g_applied_serial.store(hs.serial, std::memory_order_release);
}
