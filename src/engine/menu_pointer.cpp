#include "engine/menu_pointer.h"

#include "core/portable.h"

#include "engine/debug_menu.h"
#include "engine/option_menu.h"

#include "core/elf.h"
#include "core/memory.h"
#include "core/thunk.h"
#include "engine/addr.h"
#include "guest_abi.h"
#include "hle/hle.h"
#include "host/ingame_menu.h"
#include "host/options.h"
#include "host/settings.h"
#include "host/window.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>


// ---------------------------------------------------------------------------
// The list component (vtable 0x5799fb0, constructor sub_1ece690)
//
//   +0x30   listener: a std::function-like object, slot 0x10 takes &index.
//           The update calls it after the cursor moves.
//   +0x60   "may the cursor go here" predicate, slot 0x10 takes &index
//   +0x70   item count
//   +0x74   cursor index
//   +0x78   cursor override (used instead of +0x74 when >= 0)
//   +0x7c   cells per line
//   +0x80   visible lines; 1 means a single line that holds every item
//   +0x84   scroll along the line
//   +0x88   the component's root clip, as a value wrapper
//   +0x24c  scroll in lines
//   +0x258  item provider: slot 0x18(out, self, &{cell, line}) builds a
//           wrapper for "Item_<cell>_<line>" under the root
//   +0x2b0  wraps around
//
// A dialog embeds one at a fixed offset - the System list and the pause grid
// at +0xa70, which is why their cursor read as dialog+0xae4 and their scroll
// as +0xcbc in the earlier work - and calls its update with a pointer to one
// byte, non-zero when the component has input focus.
//
// sub_1ecf6b0(component, index) is SetCursor: it scrolls if it must, asks the
// predicate, hides every item's "Cursor" clip and shows the new one's, stores
// the index and returns 1 (2 when the predicate refused, 0 when out of range).
// It does **not** call the listener or post the sound - the update does that
// once it sees the index changed - so a hover has to as well.
//
// Value wrappers (vtable 0x579a070, built by sub_1edb220 and the provider)
// hold a GFx::Value at +0x30: interface +0x40, type +0x48, data +0x50. Type
// & 0x8f == 10 is a display object, and its native DisplayObject is
// *(data + 0x88) - the same offset DS3 reads, and the one sub_1ecff30 itself
// reads when it checks its root is visible.
//
// DisplayObject: parent +0x38, flags byte +0x6b (0x40 visible), GetBounds at
// vtable +0x1f0 (sret: out rect, this, matrix). sub_402040(obj, &m) is the
// world matrix - parent first, then the object's own (vtable +0x18). These are
// DS3's slots plus eight bytes, because a PS4 build has two destructor slots
// where MSVC has one.
// ---------------------------------------------------------------------------

namespace {

constexpr std::uint64_t kListUpdate = 0x1ecff30;
constexpr std::uint64_t kListVTable = 0x5799fb0;
constexpr std::size_t kListUpdateSlot = 3;
// The callers that do not go through the vtable: dialogs that embed the
// component and forward their own update to it - four with a call, and the
// inventory's item list (sub_1f108c0), which is one instruction and reaches it
// with a tail jump. Missing that jump is what left the item list dead to the
// pointer while its tab strip worked.
struct Site {
    std::uint64_t at;
    std::uint8_t op;  // 0xe8 call, 0xe9 jmp
};
constexpr Site kListUpdateCalls[] = {
    {0x1f25226, 0xe8}, {0x1f26066, 0xe8}, {0x1f26f28, 0xe8}, {0x1f72c80, 0xe8}, {0x1f108c7, 0xe9}};
// Fourteen bytes of padding before sub_1ec1e10, for a jump to the host.
constexpr std::uint64_t kPad = 0x1ec1e02;
constexpr std::uint8_t kPadBytes[] = {0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f,
                                      0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};

constexpr std::uint64_t kSetCursor = 0x1ecf6b0;
constexpr std::uint64_t kScrollBy = 0x1ed0dd0;  // (component, along the line, lines), clamped
constexpr std::uint64_t kHideCursors = 0x1ecf0e0;  // hides every visible row's Cursor
constexpr std::uint64_t kChild = 0x1edb220;       // wrapper = parent.GetMember(name)
constexpr std::uint64_t kCursorName = 0x4d34044;  // "Cursor"
constexpr std::uint64_t kWorldMatrix = 0x402040;
// The two global blockers sub_1edfec0 checks before it reads any input.
// The front end's menu flow object (sub_1fb2cc0 drives it); it is also the
// first of the two input blockers.
constexpr std::uint64_t kMenuFlow = 0x5980e38;     // ptr; blocked if *(u64*)(+0xa78)
// CSLocalize's region (0..5), which the game computes at boot from
// sceAppContentAppParamGetInt(1) - param.sfo's USER_DEFINED_PARAM_1 - modulo
// ten through the table at 0x4b36690, and which picks, among much else, its
// button convention: region 5, what an answer of 0 gives, decides with Circle
// and returns with Cross, Japan's way; the US disc's 2 gives region 1, which
// decides with Cross.
constexpr std::uint64_t kRegion = 0x5528948;
constexpr std::size_t kFlowStarted = 0xb11;        // 0 on the title, 1 once in the game
// The open menu, if any. The flow's update (sub_1faf8c0) branches on it: zero
// and it polls the buttons that open a menu, non-zero and it runs that menu's
// update through the functor at +0xab0. This is what makes a screen with no
// list - Stats - count as a menu.
constexpr std::size_t kFlowMenu = 0xa88;
constexpr std::uint64_t kInputBlockB = 0x5962878;  // ptr; blocked if *(u32*)(+0x2e4)
// The menu sound request: {int32 id, int32 priority} and a debug label.
// sub_1ee9bc0 plays the id and resets it to -1 once a frame; id 0 is the
// cursor move, posted by the update only while nothing louder is pending.
constexpr std::uint64_t kMenuSe = 0x597fc60;
constexpr std::uint64_t kMenuSeLabel = 0x597fc68;
constexpr std::uint64_t kCursorMoveLabel = 0x4d344e4;  // "カーソル移動"

constexpr std::size_t kListener = 0x30;
constexpr std::size_t kCount = 0x70;
constexpr std::size_t kCursor = 0x74;
constexpr std::size_t kCursorOverride = 0x78;
constexpr std::size_t kLineLen = 0x7c;
constexpr std::size_t kLines = 0x80;
constexpr std::size_t kLineScroll = 0x84;
constexpr std::size_t kScroll = 0x24c;
constexpr std::size_t kProvider = 0x258;
// The root clip's value, inside the wrapper at +0x88 (vtable 0x5799f90, whose
// GFx::Value sits eight bytes earlier than in 0x579a070's).
constexpr std::size_t kRootType = 0xc8;
constexpr std::size_t kRootData = 0xd0;
// The scroll bar, bound by sub_1edbd20 at +0xe0: its ScrollBarV value's type.
constexpr std::size_t kScrollBarType = 0x120;
constexpr std::size_t kWraps = 0x2b0;

constexpr std::size_t kWrapperValue = 0x30;
constexpr std::size_t kWrapperInterface = 0x40;
constexpr std::size_t kWrapperType = 0x48;
constexpr std::size_t kWrapperData = 0x50;
constexpr std::size_t kNative = 0x88;
constexpr std::size_t kParent = 0x38;
constexpr std::size_t kFlags = 0x6b;
constexpr std::uint8_t kVisible = 0x40;
constexpr std::size_t kGetBounds = 0x1f0;

std::uint64_t g_slide = 0;
bool g_log = false;
// BBHOST_POINTER=0: the hook stays in and still reports focus (the camera
// needs it) but never moves a cursor - for telling the game's own behaviour
// from the pointer's.
bool g_passive = false;

std::atomic<std::uint64_t> g_tick{0};
std::atomic<std::int64_t> g_focus_ms{0};
std::atomic<bool> g_click{false};
std::atomic<std::int64_t> g_click_ms{0};
std::atomic<std::uint32_t> g_press{0};

std::uint64_t guest(std::uint64_t bn) { return g_slide + (bn - kPreferredGuestSlide); }
void* guest_fn(std::uint64_t bn) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest(bn)));
}
void* ptr(std::uint64_t v) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(v)); }

// Everything below reads pointers out of live guest objects. The kernel's
// check, not a heuristic: process_vm_readv fails on an unmapped page where a
// plain read would take the game down.
bool safe_read(std::uint64_t addr, void* out, std::size_t len) {
    if (addr < 0x1000) {
        return false;
    }
    return host_read_safe(ptr(addr), out, len);
}

// A function in the eboot: inside its one executable segment, which also holds
// read-only data, and opening with the prologue every function here has.
constexpr std::uint64_t kExecEnd = 0x54d96dc;
bool guest_code(std::uint64_t addr) {
    if (addr < guest(0x400000) || addr >= guest(kExecEnd)) {
        return false;
    }
    std::uint8_t p[4] = {};
    return safe_read(addr, p, sizeof(p)) && p[0] == 0x55 && p[1] == 0x48 && p[2] == 0x89 && p[3] == 0xe5;
}

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <typename T>
T field(const std::uint8_t* base, std::size_t off) {
    T v{};
    std::memcpy(&v, base + off, sizeof(v));
    return v;
}

// A wrapper the guest builds into our buffer. The GFx::Value inside links
// itself into its movie's list of managed values, so it must be released
// before the buffer goes out of scope - which the destructor does, the way
// every stack wrapper in the game is torn down.
struct Wrapper {
    alignas(16) std::uint8_t b[0x80]{};

    Wrapper() = default;
    Wrapper(const Wrapper&) = delete;
    Wrapper& operator=(const Wrapper&) = delete;
    ~Wrapper() {
        if (!(field<std::uint32_t>(b, kWrapperType) & 0x40)) {
            return;
        }
        const auto oi = field<std::uint64_t>(b, kWrapperInterface);
        std::uint64_t vt = 0, release = 0;
        if (oi && safe_read(oi, &vt, sizeof(vt)) && safe_read(vt + 0x18, &release, sizeof(release)) &&
            guest_code(release)) {
            hle_call_guest(ptr(release), ptr(oi), b + kWrapperValue,
                           field<std::uint64_t>(b, kWrapperData));
        }
    }
    // The native display object, or 0 when this is not one.
    std::uint64_t display() const {
        if ((field<std::uint32_t>(b, kWrapperType) & 0x8f) != 10) {
            return 0;
        }
        std::uint64_t d = 0;
        const auto data = field<std::uint64_t>(b, kWrapperData);
        return data && safe_read(data + kNative, &d, sizeof(d)) ? d : 0;
    }
};

// Visible, and so is everything above it. A list keeps clips for slots it has
// no item for, hidden; a screen fading out hides its root.
bool shown(std::uint64_t obj) {
    for (int depth = 0; obj && depth < 32; ++depth) {
        std::uint8_t flags = 0;
        if (!safe_read(obj + kFlags, &flags, sizeof(flags)) || !(flags & kVisible)) {
            return false;
        }
        if (!safe_read(obj + kParent, &obj, sizeof(obj))) {
            return false;
        }
    }
    return true;
}

struct Rect {
    float x1, y1, x2, y2;
};

// DS3's SprjScaleformValue::HitTest (sub_140e63aa0), on Bloodborne's slots:
// local bounds, world matrix, inverse, point in local space. `world` gets the
// object's placement on the stage in pixels, for the log.
bool hit(std::uint64_t obj, float sx, float sy, Rect* world, float grow_px = 0.0f) {
    std::uint64_t vt = 0, get_bounds = 0;
    if (!safe_read(obj, &vt, sizeof(vt)) || !safe_read(vt + kGetBounds, &get_bounds, sizeof(get_bounds)) ||
        !guest_code(get_bounds)) {
        return false;
    }
    alignas(16) float identity[8] = {1.0f, 0, 0, 0, 0, 1.0f, 0, 0};
    alignas(16) float r[4] = {};
    hle_call_guest(ptr(get_bounds), r, ptr(obj), identity);
    if (!(r[2] > r[0]) || !(r[3] > r[1])) {
        return false;  // nothing drawn
    }
    alignas(16) float m[8] = {1.0f, 0, 0, 0, 0, 1.0f, 0, 0};
    hle_call_guest(guest_fn(kWorldMatrix), ptr(obj), m);
    if (world) {
        // Corners through the matrix, in pixels. Axis-aligned is enough for
        // a log line; menus do not rotate.
        const float xs[2] = {r[0], r[2]}, ys[2] = {r[1], r[3]};
        world->x1 = world->y1 = 1e9f;
        world->x2 = world->y2 = -1e9f;
        for (float x : xs) {
            for (float y : ys) {
                const float wx = (m[0] * x + m[1] * y + m[3]) / 20.0f;
                const float wy = (m[4] * x + m[5] * y + m[7]) / 20.0f;
                world->x1 = std::fmin(world->x1, wx);
                world->x2 = std::fmax(world->x2, wx);
                world->y1 = std::fmin(world->y1, wy);
                world->y2 = std::fmax(world->y2, wy);
            }
        }
    }
    const float det = m[0] * m[5] - m[1] * m[4];
    if (std::fabs(det) < 1e-12f) {
        return false;
    }
    // Stage pixels to twips, then into the object's own space.
    const float px = sx * 20.0f - m[3], py = sy * 20.0f - m[7];
    const float lx = (m[5] * px - m[1] * py) / det;
    const float ly = (-m[4] * px + m[0] * py) / det;
    // A margin in stage pixels, in the object's own units.
    const float g = grow_px * 20.0f / std::sqrt(std::fabs(det));
    return lx >= r[0] - g && lx <= r[2] + g && ly >= r[1] - g && ly <= r[3] + g;
}

int cursor_of(const std::uint8_t* c) {
    const auto o = field<std::int32_t>(c, kCursorOverride);
    return o >= 0 ? o : field<std::int32_t>(c, kCursor);
}

// Which item is under the pointer: the absolute index, or -1.
//
// The cells walked are exactly the ones sub_1ecf6b0 can address - `lines`
// lines of `cells` each, starting at the scroll - and an item's absolute
// index is computed the way it computes it, including the single-line case
// where a line holds the whole list.
int item_at(std::uint8_t* c, float sx, float sy, bool log, std::uint64_t* packed_out = nullptr) {
    const int count = field<std::int32_t>(c, kCount);
    const int cells = field<std::int32_t>(c, kLineLen);
    const int lines = field<std::int32_t>(c, kLines);
    const int cell_scroll = field<std::int32_t>(c, kLineScroll);
    const int line_scroll = field<std::int32_t>(c, kScroll);
    const bool wraps = field<std::uint8_t>(c, kWraps) != 0;
    if (count <= 0 || cells <= 0 || lines <= 0 || cells > 64 || lines > 64) {
        return -1;
    }
    const int stride = lines != 1 ? std::max(cells, 1) : std::max(std::max(cells, count), 1);
    const auto provider = field<std::uint64_t>(c, kProvider);
    std::uint64_t pvt = 0, make = 0;
    if (!provider || !safe_read(provider, &pvt, sizeof(pvt)) || !safe_read(pvt + 0x18, &make, sizeof(make)) ||
        !guest_code(make)) {
        return -1;
    }
    int best = -1;
    float best_d = 0.0f;
    std::uint64_t best_packed = 0;
    for (int line = 0; line < lines; ++line) {
        for (int cell = 0; cell < cells; ++cell) {
            int index = (line_scroll + line) * stride + cell_scroll + cell;
            if (wraps && count) {
                index %= count;
            }
            if (index < 0 || index >= count) {
                continue;
            }
            const std::uint64_t packed = static_cast<std::uint32_t>(cell) |
                                         (static_cast<std::uint64_t>(static_cast<std::uint32_t>(line)) << 32);
            Wrapper item;
            hle_call_guest(ptr(make), item.b, ptr(provider), &packed);
            const std::uint64_t obj = item.display();
            if (!obj || !shown(obj)) {
                continue;
            }
            // DS3 tests the item's HitArea, else its Cursor, else the item.
            // Bloodborne's movies have no HitArea; every row has a Cursor -
            // the highlight bar, which spans the row. Hidden on the rows the
            // cursor is not on, but its geometry is there.
            Wrapper region;
            hle_call_guest(guest_fn(kChild), region.b, item.b, guest_fn(kCursorName), 0);
            std::uint64_t target = region.display();
            Rect w{};
            bool in = target && hit(target, sx, sy, &w);
            if (!target) {
                target = obj;
                in = hit(target, sx, sy, &w);
            }
            // An option row's glow covers its label, not its value; the value
            // widget beside it is part of the row too, as DS3's whole-row
            // HitArea is. Only the widget that is showing - a row carries all
            // four and hides three.
            for (const char* part : {"Widgets/ComboBox", "Widgets/Slider"}) {
                if (in) {
                    break;
                }
                Wrapper widget;
                hle_call_guest(guest_fn(kChild), widget.b, item.b, part, 0);
                const std::uint64_t wobj = widget.display();
                Rect ww{};
                if (wobj && shown(wobj) && hit(wobj, sx, sy, &ww)) {
                    in = true;
                    w = ww;
                }
            }
            if (log) {
                host_log("pointer:   item %d (cell %d line %d) %s at %.0f,%.0f-%.0f,%.0f%s", index, cell,
                         line, target == obj ? "self" : "Cursor", static_cast<double>(w.x1),
                         static_cast<double>(w.y1), static_cast<double>(w.x2), static_cast<double>(w.y2),
                         in ? "  <- pointer" : "");
            }
            // Bloodborne's Cursor is a glow, taller than the row it lights -
            // the System list's are 130px on a 68px pitch - so neighbours
            // overlap and the first hit would be the row above half the
            // time. Of the rows that contain the point, the one whose centre
            // is nearest is the one the player is pointing at.
            if (in) {
                const float dx = (w.x1 + w.x2) * 0.5f - sx, dy = (w.y1 + w.y2) * 0.5f - sy;
                const float d = dx * dx + dy * dy;
                if (best < 0 || d < best_d) {
                    best = index;
                    best_d = d;
                    best_packed = packed;
                }
            }
        }
    }
    if (packed_out) {
        *packed_out = best_packed;
    }
    return best;
}

// What a click on an item means. DS3's arrow selector (sub_140ac8860) hit-tests
// its two arrow sprites and steps the value by the one it hit; Bloodborne's
// option rows carry the same thing as `Widgets/Slider/LeftArrow` and
// `RightArrow`. Its other value widget, the ComboBox, draws its arrows into
// the value's own art rather than as sprites, so there the side of the value
// that was clicked picks the direction - measured on the live clip, like
// everything else here. Anything else is a decide.
std::uint32_t click_button(std::uint8_t* c, int index, std::uint64_t packed, float sx, float sy) {
    const std::uint32_t kCircle = menu_confirm_button();  // the decide, whichever button it is
    constexpr std::uint32_t kLeft = 0x80u, kRight = 0x20u;
    // A row the port draws itself may want the click as a decide wherever it
    // lands - a key binding is changed by choosing it, not by stepping it.
    if (option_menu_click_decides(c, index)) {
        return kCircle;
    }
    const auto provider = field<std::uint64_t>(c, kProvider);
    std::uint64_t pvt = 0, make = 0;
    if (!provider || !safe_read(provider, &pvt, sizeof(pvt)) || !safe_read(pvt + 0x18, &make, sizeof(make)) ||
        !guest_code(make)) {
        return kCircle;
    }
    Wrapper item;
    hle_call_guest(ptr(make), item.b, ptr(provider), &packed);
    if (!item.display()) {
        return kCircle;
    }
    const auto shown_hit = [&](const char* path, Rect* w, float grow) {
        Wrapper part;
        hle_call_guest(guest_fn(kChild), part.b, item.b, path, 0);
        const std::uint64_t obj = part.display();
        const bool vis = obj && shown(obj);
        const bool in = vis && hit(obj, sx, sy, w, grow);
        if (g_log && obj) {
            host_log("pointer:   %s %s at %.0f,%.0f-%.0f,%.0f%s", path, vis ? "shown" : "hidden",
                     static_cast<double>(w->x1), static_cast<double>(w->y1), static_cast<double>(w->x2),
                     static_cast<double>(w->y2), in ? "  <- pointer" : "");
        }
        return in;
    };
    // The slider's arrows are eleven pixels wide; a margin makes them a
    // target a hand can hit without changing which arrow wins.
    Rect w{};
    if (shown_hit("Widgets/Slider/LeftArrow", &w, 14.0f)) {
        return kLeft;
    }
    if (shown_hit("Widgets/Slider/RightArrow", &w, 14.0f)) {
        return kRight;
    }
    if (shown_hit("Widgets/ComboBox", &w, 0.0f)) {
        return sx < (w.x1 + w.x2) * 0.5f ? kLeft : kRight;
    }
    return kCircle;
}

// Shows the Cursor of the row at a visible cell, the last step of
// sub_1ecf6b0: value wrapper slot 1 is SetVisible (sub_1ecb760).
void show_cursor(std::uint8_t* c, std::uint64_t packed) {
    const auto provider = field<std::uint64_t>(c, kProvider);
    std::uint64_t pvt = 0, make = 0;
    if (!provider || !safe_read(provider, &pvt, sizeof(pvt)) || !safe_read(pvt + 0x18, &make, sizeof(make)) ||
        !guest_code(make)) {
        return;
    }
    Wrapper item;
    hle_call_guest(ptr(make), item.b, ptr(provider), &packed);
    Wrapper cursor;
    hle_call_guest(guest_fn(kChild), cursor.b, item.b, guest_fn(kCursorName), 0);
    const auto vt = field<std::uint64_t>(cursor.b, 0);
    std::uint64_t set_visible = 0;
    if (cursor.display() && vt && safe_read(vt + 8, &set_visible, sizeof(set_visible)) &&
        guest_code(set_visible)) {
        hle_call_guest(ptr(set_visible), cursor.b, 1);
    }
}

bool input_blocked() {
    std::uint64_t a = 0, b = 0;
    if (safe_read(guest(kMenuFlow), &a, sizeof(a)) && a) {
        std::uint64_t v = 0;
        if (safe_read(a + 0xa78, &v, sizeof(v)) && v) {
            return true;
        }
    }
    if (safe_read(guest(kInputBlockB), &b, sizeof(b)) && b) {
        std::uint32_t v = 0;
        if (safe_read(b + 0x2e4, &v, sizeof(v)) && v) {
            return true;
        }
    }
    return false;
}

// The front end - the title screen and every menu before the world loads.
// sub_1fb2c60 is the game's own test: the menu flow object exists and its
// +0xb11 is clear. sub_1fb2cc0 sets that byte when the game proper starts
// (Continue, Load, New Game), and it reads 0 on the title before then.
bool front_end() {
    std::uint64_t flow = 0;
    std::uint8_t started = 1;
    return safe_read(guest(kMenuFlow), &flow, sizeof(flow)) && flow &&
           safe_read(flow + kFlowStarted, &started, sizeof(started)) && !started;
}

// A menu is open: the flow is running one (see kFlowMenu).
bool menu_open() {
    std::uint64_t flow = 0, menu = 0;
    return safe_read(guest(kMenuFlow), &flow, sizeof(flow)) && flow &&
           safe_read(flow + kFlowMenu, &menu, sizeof(menu)) && menu;
}

bool direction_held() {
    const PadState p = host_pad_state();
    if (p.buttons & 0xf0u) {
        return true;
    }
    const auto off = [](std::uint8_t v) { return v < 64 || v > 192; };
    return off(p.lx) || off(p.ly);
}

// Where each component last saw the pointer. DS3's "moved" is a per-frame
// flag on CSMouseMan; per component is the same thing here, since a list
// updates once a frame, and it lets two lists on one screen (a tab strip and
// its grid) both see the same movement.
struct Seen {
    std::uint8_t* comp = nullptr;
    float x = 0.0f, y = 0.0f;
    std::int64_t ms = 0;
    bool logged = false;
};
Seen g_seen[32];
int g_seen_next = 0;

Seen& seen(std::uint8_t* c, bool* fresh) {
    const std::int64_t now = now_ms();
    for (Seen& s : g_seen) {
        if (s.comp == c) {
            // A component not updated for a while is a new screen that reused
            // the address; it has not seen the pointer move.
            *fresh = now - s.ms > 250;
            s.ms = now;
            if (*fresh) {
                s.logged = false;
            }
            return s;
        }
    }
    Seen& s = g_seen[g_seen_next++ % 32];
    s = Seen{};
    s.comp = c;
    s.ms = now;
    *fresh = true;
    return s;
}

void pointer_step(std::uint8_t* c, const std::uint8_t* input) {
    if (g_log && input && !*input) {
        static std::uint8_t* told[64] = {};
        static int told_next = 0;
        bool known = false;
        for (auto* t : told) {
            known = known || t == c;
        }
        if (!known) {
            told[told_next++ % 64] = c;
            host_log("pointer: list %p (no focus) count %d cells %d lines %d cursor %d", static_cast<void*>(c),
                     field<std::int32_t>(c, kCount), field<std::int32_t>(c, kLineLen),
                     field<std::int32_t>(c, kLines), cursor_of(c));
        }
    }
    if (!input || !*input || input_blocked()) {
        return;
    }
    // A list with focus is a menu the player is in, pointer or not - this is
    // what hands the mouse back from the camera.
    g_tick.fetch_add(1, std::memory_order_relaxed);
    g_focus_ms.store(now_ms(), std::memory_order_relaxed);
    // Nor while a controller is the device in use: its cursor is hidden, and
    // the mouse moving again is what hands it back (host_input_device).
    if (!host_settings_ref().mouse_menu || host_options_open() || ingame_menu_open() || host_mouse_relative() || g_passive ||
        host_input_device() == InputDevice::Pad) {
        return;
    }
    float sx = 0.0f, sy = 0.0f;
    if (!host_mouse_stage_position(sx, sy)) {
        return;
    }
    bool fresh = false;
    Seen& s = seen(c, &fresh);
    const bool moved = !fresh && (sx != s.x || sy != s.y);
    s.x = sx;
    s.y = sy;
    // Nor under a text box (the IME dialog's): the pointer passing over the
    // list behind it moved the game's cursor while the box was being typed
    // in. Where it went is still noted, so that is not a movement to hover on
    // once the box has closed either.
    if (host_text_entry_open()) {
        return;
    }
    if (g_log && !s.logged) {
        s.logged = true;
        // Whether sub_1edbd20 found a ScrollBarV under the list's root: a list
        // that holds more than it shows and has none gives no sign it scrolls.
        const bool bar = (field<std::uint32_t>(c, kScrollBarType) & 0x8f) == 10;
        const int shown_items = field<std::int32_t>(c, kLineLen) * field<std::int32_t>(c, kLines);
        host_log("pointer: list %p count %d cells %d lines %d cursor %d scroll %d/%d scrollbar %s%s",
                 static_cast<void*>(c), field<std::int32_t>(c, kCount), field<std::int32_t>(c, kLineLen),
                 field<std::int32_t>(c, kLines), cursor_of(c), field<std::int32_t>(c, kScroll),
                 field<std::int32_t>(c, kLineScroll), bar ? "yes" : "no",
                 !bar && field<std::int32_t>(c, kLines) > 1 && field<std::int32_t>(c, kCount) > shown_items
                     ? "  <- scrolls with nothing to show it"
                     : "");
        item_at(c, sx, sy, true);
    }
    bool click = false;
    if (g_click.load(std::memory_order_relaxed)) {
        if (now_ms() - g_click_ms.load(std::memory_order_relaxed) > 250) {
            g_click.store(false, std::memory_order_relaxed);  // no list claimed it
        } else {
            click = true;
        }
    }
    // A list of one cell on one line is a spinner - the value of an option
    // row, which Left and Right step. It is DS3's arrow selector, and a click
    // on it steps toward the side it landed on. The component's own root clip
    // is what is drawn, arrows included.
    if (field<std::int32_t>(c, kLines) == 1 && field<std::int32_t>(c, kLineLen) == 1 &&
        field<std::int32_t>(c, kCount) > 1) {
        if (!click) {
            return;
        }
        std::uint64_t root = 0;
        const auto data = field<std::uint64_t>(c, kRootData);
        if ((field<std::uint32_t>(c, kRootType) & 0x8f) != 10 || !data ||
            !safe_read(data + kNative, &root, sizeof(root))) {
            return;
        }
        Rect w{};
        const bool in = root && shown(root) && hit(root, sx, sy, &w);
        if (g_log) {
            host_log("pointer: spinner %p at %.0f,%.0f-%.0f,%.0f%s", static_cast<void*>(c),
                     static_cast<double>(w.x1), static_cast<double>(w.y1), static_cast<double>(w.x2),
                     static_cast<double>(w.y2), in ? "  <- click" : "");
        }
        if (in && g_click.exchange(false)) {
            g_press.store(sx < (w.x1 + w.x2) * 0.5f ? 0x80u : 0x20u, std::memory_order_relaxed);
        }
        return;
    }
    // DS3: hover only on movement, and never against a direction the player
    // is holding - a resting mouse must not fight the pad.
    // With the log on, why a movement did not hover, twice a second at most:
    // a pad reading as held, or a point on no item.
    const auto refused = [&](const char* why) {
        static std::int64_t last_ms = 0;
        static int logs = 0;
        if (!g_log || logs >= 60 || now_ms() - last_ms < 500) return false;
        last_ms = now_ms();
        ++logs;
        const PadState p = host_pad_state();
        char where[160];
        host_mouse_describe(where, sizeof(where));
        host_log("pointer: list %p at %.0f,%.0f not hovered: %s (pad buttons %04x stick %u,%u; %s)", static_cast<void*>(c),
                 static_cast<double>(sx), static_cast<double>(sy), why, p.buttons, p.lx, p.ly, where);
        return true;
    };
    if (!moved && !click) {
        return;
    }
    if (direction_held()) {
        refused("a direction is held");
        return;
    }
    std::uint64_t packed = 0;
    const int want = item_at(c, sx, sy, false, &packed);
    if (want < 0) {
        refused("no item there");  // the list's rectangles were logged when it was first seen
        return;  // a miss changes nothing, and a click on nothing does nothing
    }
    if (click) {
        g_click.store(false, std::memory_order_relaxed);
    }
    const int was = cursor_of(c);
    int now = was;
    if (want != was) {
        // sub_1ecf6b0 keeps a line of margin: landing on the first or last
        // visible line scrolls the list by one. From the pad that is what
        // brings the next row into view; from the pointer it moves the list
        // under a pointer that has not moved, the next twitch selects the row
        // after, and the list creeps to its end. The pointer only targets
        // rows already on screen, so the scroll is put back with the same
        // function (sub_1ed0dd0 sets the offsets and redraws - no tween). The
        // wheel is what scrolls, as in DS3.
        const int s1 = field<std::int32_t>(c, kLineScroll), s2 = field<std::int32_t>(c, kScroll);
        const int r = static_cast<std::int32_t>(hle_call_guest(guest_fn(kSetCursor), c, want));
        const int d1 = s1 - field<std::int32_t>(c, kLineScroll), d2 = s2 - field<std::int32_t>(c, kScroll);
        if (r == 1 && (d1 || d2)) {
            hle_call_guest(guest_fn(kScrollBy), c, d1, d2);
            // sub_1ecf6b0 lit the row's glow at the slot it had scrolled it
            // to; light it where it is now. Its own two steps: hide every
            // row's Cursor, show this one's.
            hle_call_guest(guest_fn(kHideCursors), c);
            show_cursor(c, packed);
        }
        now = cursor_of(c);
        static int logs = 0;
        if (g_log && logs < 40) {
            ++logs;
            host_log("pointer: list %p hover %d -> %d (set %d, now %d)", static_cast<void*>(c), was, want, r,
                     now);
        }
    }
    // A click acts on the item it landed on, and only once the cursor is
    // there: a row the game refuses (sub_1ecf6b0 answering 2) is not chosen.
    if (click && now == want) {
        const std::uint32_t button = click_button(c, want, packed, sx, sy);
        g_press.store(button, std::memory_order_relaxed);
        if (g_log) {
            host_log("pointer: list %p click on item %d -> pad %04x", static_cast<void*>(c), want, button);
        }
    }
    if (now == was) {
        return;
    }
    // What sub_1ecff30 does once it sees the index change.
    const auto listener = field<std::uint64_t>(c, kListener);
    std::uint64_t lvt = 0, call = 0;
    if (listener && safe_read(listener, &lvt, sizeof(lvt)) && safe_read(lvt + 0x10, &call, sizeof(call)) &&
        guest_code(call)) {
        std::int32_t index = now;
        hle_call_guest(ptr(call), ptr(listener), &index);
    }
    auto* se = static_cast<std::int32_t*>(guest_fn(kMenuSe));
    if (se[1] <= 0) {
        se[0] = 0;
        se[1] = 0;
        *static_cast<std::uint64_t*>(guest_fn(kMenuSeLabel)) = guest(kCursorMoveLabel);
    }
}

GUEST_ABI void pc_list_update(std::uint8_t* comp, std::uint8_t* input) {
    // Before the pad, as DS3 does it: a direction the pad reads this frame
    // then moves on from wherever the pointer put the cursor.
    pointer_step(comp, input);
    option_menu_list_update(comp, input && *input);
    const int before = cursor_of(comp);
    hle_call_guest(guest_fn(kListUpdate), comp, input);
    // BBHOST_POINTER_LOG=1: a cursor the game's own update moved (the pad,
    // the wheel's Down, a key), so a scripted run can count rows per press.
    if (g_log) {
        const int after = cursor_of(comp);
        static int logs = 0;
        if (after != before && logs < 2000000) {
            ++logs;
            host_log("pointer: list %p pad cursor %d -> %d", static_cast<void*>(comp), before, after);
        }
    }
}

}  // namespace

bool menu_pointer_front_end() { return front_end(); }

void menu_pointer_install(ElfImage* image) {
    g_slide = image->mem.slide;
    g_log = [] {
        const char* e = std::getenv("BBHOST_POINTER_LOG");
        return e && e[0] == '1';
    }();
    g_passive = [] {
        const char* e = std::getenv("BBHOST_POINTER");
        return e && e[0] == '0';
    }();
    const std::uint64_t slot = guest(kListVTable) + kListUpdateSlot * 8;
    auto* sp = static_cast<std::uint64_t*>(guest_ptr(image->mem, slot));
    auto* pad = static_cast<std::uint8_t*>(guest_ptr(image->mem, guest(kPad)));
    if (*sp != guest(kListUpdate) || std::memcmp(pad, kPadBytes, sizeof(kPadBytes)) != 0) {
        host_log("pointer: refused - the list vtable or the padding at 0x%llx is not as expected",
                 static_cast<unsigned long long>(kPad));
        return;
    }
    for (const Site& site : kListUpdateCalls) {
        const auto* cp = static_cast<const std::uint8_t*>(guest_ptr(image->mem, guest(site.at)));
        std::int32_t rel = 0;
        std::memcpy(&rel, cp + 1, sizeof(rel));
        if (cp[0] != site.op ||
            guest(site.at) + 5 + static_cast<std::uint64_t>(static_cast<std::int64_t>(rel)) != guest(kListUpdate)) {
            host_log("pointer: refused - 0x%llx does not reach sub_1ecff30", static_cast<unsigned long long>(site.at));
            return;
        }
    }
    const auto hook = reinterpret_cast<std::uint64_t>(hle_wrap_fn(reinterpret_cast<void*>(&pc_list_update)));
    if (!guest_protect_rw(&image->mem, slot & ~0xfffull, 0x1000)) {
        host_log("pointer: cannot unprotect the list vtable");
        return;
    }
    *sp = hook;
    // The direct callers reach the host through the padding: mov r11, hook;
    // jmp r11.
    const std::uint64_t pva = guest(kPad);
    if (guest_protect_rwx(&image->mem, pva & ~0xfffull, 0x1000)) {
        pad[0] = 0x49;
        pad[1] = 0xbb;
        std::memcpy(pad + 2, &hook, 8);
        pad[10] = 0x41;
        pad[11] = 0xff;
        pad[12] = 0xe3;
        guest_protect_rx(&image->mem, pva & ~0xfffull, 0x1000);
        for (const Site& site : kListUpdateCalls) {
            const std::uint64_t call = guest(site.at);
            if (!guest_protect_rwx(&image->mem, call & ~0xfffull, 0x1000)) {
                continue;
            }
            auto* cp = static_cast<std::uint8_t*>(guest_ptr(image->mem, call));
            const auto rel = static_cast<std::int32_t>(static_cast<std::int64_t>(pva) -
                                                       static_cast<std::int64_t>(call + 5));
            std::memcpy(cp + 1, &rel, sizeof(rel));
            guest_protect_rx(&image->mem, call & ~0xfffull, 0x1000);
        }
    }
    host_log("pointer: the menus' list update is hooked (vtable and %zu direct callers)",
             sizeof(kListUpdateCalls) / sizeof(kListUpdateCalls[0]));

}

std::uint32_t menu_region_confirm_button() {
    std::int32_t region = -1;
    if (!g_slide || !safe_read(guest(kRegion), &region, sizeof(region))) {
        return 0x4000u;
    }
    return region == 5 ? 0x2000u : 0x4000u;
}

std::uint32_t menu_confirm_button() {
    // The developers' debug menu decides with Circle in every region.
    if (debug_menu_open()) {
        return 0x2000u;
    }
    return menu_region_confirm_button();
}

std::uint32_t menu_back_button() { return menu_confirm_button() == 0x2000u ? 0x4000u : 0x2000u; }

void menu_pointer_click() {
    g_click_ms.store(now_ms(), std::memory_order_relaxed);
    g_click.store(true, std::memory_order_relaxed);
}

std::uint32_t menu_pointer_take_press() { return g_press.exchange(0); }

std::uint64_t menu_pointer_tick() { return g_tick.load(std::memory_order_relaxed); }

bool menu_pointer_in_menu() {
    // DS3 keeps the pointer while any UI state owns the screen. Here that is
    // the front end as a whole - between its menus no list has focus for a
    // second or so, and taking the mouse for the camera then hid the pointer
    // for nothing - and in the world, a list with focus.
    if (front_end()) {
        return true;
    }
    // In the world, the flow's own record of an open menu - the equivalent of
    // DS3's UI-state flags, and the only signal that covers a screen with no
    // list (Stats). A list with focus as well, for anything the flow does not
    // own; a list updates every frame it has focus, so a gap longer than a
    // few frames means none has.
    return menu_open() || now_ms() - g_focus_ms.load(std::memory_order_relaxed) < 400;
}
