#include "engine/plugin_dialog.h"

#include "core/thunk.h"
#include "engine/addr.h"
#include "engine/menu_assets.h"
#include "engine/menu_pointer.h"
#include "engine/option_rows.h"
#include "guest_abi.h"
#include "hle/fs.h"
#include "hle/hle.h"
#include "hle/modules.h"
#include "host/plugin_panel.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/syscall.h>
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// The menu flow (engine/menu_pointer.cpp's kMenuFlow): the object the game
// opens its menus on. Its update (sub_1faf8c0) runs the open menu (+0xa88)
// and, with none open, the buttons that open one; the game's menu step
// (sub_1b692f0) pushes a screen on it with two calls, and so does the host:
//
//     sub_1fb4c70(&screen, flow + 0x48, &opener, 0)    the screen, from an opener
//     sub_1fb27a0(&out, flow, &screen, &callback)       pushed: +0xa88 is it now
//
// There is no stack: a push replaces the open menu, which sub_1fb27a0 puts
// on the list at +0xac0 (count +0xb08) that the next update empties
// (sub_1faf4d0). So a page pushed over a menu of the game's in the world takes
// its place, and closing it leaves the player in the world. Keeping that menu
// alive under the page to push it again would put one dialog more on the
// eight the menus' context holds (its +0x50, filled by sub_201d420) than the
// game ever has: a loading screen opening over both overflowed it (DL_PANIC).
// sub_1fb27a0 counts its pushes at +0xb18 and sub_1fb26b0 its closes, which
// tells a refused push, and a page the game took: sub_1fb26b0 closes whatever
// menu is open for the game's own reasons as well - a load (sub_1fb3140), the
// world's prompts when the action they offer changes (sub_1c68f30), and frame
// after frame while the hunter is dying (at 0 HP, until the death's load) -
// which a menu that finishes by itself (Circle) does not count. A corner page
// the game took comes back once the flow has been quiet a moment; the plugin
// is not told. Where the flow closes its menus frame after frame, no page of
// the game's dialog can stay: the host draws it instead (the fallback below),
// as it does when the flow still refuses the page a second after the take or
// takes it a third time within 10 s, and the game's dialog has it again the
// next time it is shown.
//
// sub_1fb27a0 refuses (pushes nothing) while MenuMan's +0x2fc is set. Setting
// +0xb12 asks the flow to close its open menu at its next update (sub_1fb26b0).
// The section opener is the PC sections' (engine/option_rows.h): the title
// written into the bundle's message first, then sub_1f20900 with PlugPage or
// PlugCorner and the handler below.
// ---------------------------------------------------------------------------

namespace plugin_dialog {
namespace {

// A value wrapper's child by name (engine/menu_pointer.cpp's kChild): the
// wrappers sub_1edb220 builds (vtable 0x579a070) hold their GFx::Value's
// interface at +0x40, type at +0x48 and data at +0x50; a list's root wrapper
// (+0x88 in the list, vtable 0x5799f90) holds the same eight bytes earlier,
// so it is passed eight bytes back. ObjectInterface slot 0x138 is SetText
// (pdata, text, html), as the rows' providers call it.
constexpr std::uint64_t kChild = 0x1edb220;
constexpr std::size_t kListRoot = 0x88, kValueIface = 0x40, kValueType = 0x48, kValueData = 0x50, kSetText = 0x138;
constexpr std::uint64_t kMenuFlowPtr = 0x5980e38;
constexpr std::uint64_t kMenuManPtr = 0x5962878;
constexpr std::uint64_t kPushScreen = 0x1fb27a0;
constexpr std::size_t kFlowCtx = 0x48, kFlowMenu = 0xa88, kFlowBlocked = 0xa78, kFlowClose = 0xb12;
constexpr std::size_t kFlowChanges = 0xb18;
constexpr std::size_t kMenuManNoPush = 0x2fc;
// The port's messages for these pages (tools/pc_option_messages.tsv): eight
// row placeholders, the question the host writes and the System list's row
// (the title's placeholder, 126020, is the movie's: menu_assets.cpp).
constexpr std::uint32_t kRowId = 126000, kQuestionId = 920002, kSystemRowId = 110015;
const char kMovie[] = "/dvdroot_ps4/menu/optionsetting.gfx";
const char kPageName[] = "PlugPage";
const char kCornerName[] = "PlugCorner";
constexpr int kSlots = 8;          // the sections' rows
constexpr int kMaxRows = 96;       // a page's rows the dialog takes
constexpr std::size_t kText = 160;  // a row's label, help or value, in UTF-16 units with the markup

// The game's slider (sub_1f2c500 builds it at a slider row's +0xa0): its
// number's field, its range, value and step (the game builds 0..10 in ones),
// and its setter (sub_1edcaa0: the value clamped, the number and the bar - a
// 100-frame clip at the value's share of the range - drawn again when it
// changed, the row told when asked, a sound when given one).
constexpr std::size_t kSliderText = 0xad0, kSliderMin = 0xbf0, kSliderMax = 0xbf4, kSliderValue = 0xbf8, kSliderStep = 0xbfc;
constexpr std::uint64_t kSliderSet = 0x1edcaa0;
constexpr std::int64_t kNoSound = 0xffffffffll;  // its sound argument: id -1, none

// The pad bits (ScePad).
constexpr std::uint32_t kUp = 0x10, kRight = 0x20, kDown = 0x40, kLeft = 0x80, kL1 = 0x400, kR1 = 0x800;
constexpr std::uint32_t kCross = 0x4000, kCircle = 0x2000;
// What a corner page leaves the game: the buttons the dialog moves by.
constexpr std::uint32_t kDialogButtons = kUp | kRight | kDown | kLeft | kL1 | kR1 | kCross | kCircle;

bool g_available = false;

std::uint64_t rd64(std::uint64_t a) {
    std::uint64_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), sizeof(v));
    return v;
}
std::uint32_t rd32(std::uint64_t a) {
    std::uint32_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), sizeof(v));
    return v;
}
std::uint64_t flow() { return rd64(option_rows::addr(kMenuFlowPtr)); }

long thread_id() {
#if defined(_WIN32)
    return 0;
#else
    return static_cast<long>(::syscall(SYS_gettid));
#endif
}

// ---- UTF-8 into the rows' UTF-16, as HTML text (the rows' fields are) ----------
void put_html(char16_t* out, std::size_t room, const std::string& s, bool grey) {
    std::size_t k = 0;
    const auto put = [&](const char* t) {
        for (; *t && k + 1 < room; ++t) out[k++] = static_cast<char16_t>(static_cast<unsigned char>(*t));
    };
    if (grey) put("<font color='#77746a'>");
    const std::size_t close = grey ? 7 : 0;  // "</font>"
    for (std::size_t i = 0; i < s.size() && k + 1 + close < room;) {
        std::uint32_t c = static_cast<unsigned char>(s[i]);
        const int extra = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : c >= 0xc0 ? 1 : 0;
        c &= extra == 3 ? 0x07 : extra == 2 ? 0x0f : extra == 1 ? 0x1f : 0x7f;
        ++i;
        for (int e = 0; e < extra && i < s.size(); ++e, ++i) c = (c << 6) | (static_cast<unsigned char>(s[i]) & 0x3f);
        if (c == '&') put("&amp;");
        else if (c == '<') put("&lt;");
        else if (c == '>') put("&gt;");
        else if (c < 0x10000 && (c >= 0x20 || c == '\n')) out[k++] = static_cast<char16_t>(c);
    }
    if (grey) put("</font>");
    out[k] = 0;
}

// A row's label or value, set smaller when longer than its field holds at the
// field's own size: the rows' fields wrap, and a second line is out of view.
// About 24 characters fit a label, 18 a value; a fifth more a step smaller.
constexpr std::size_t kLabelFits = 24, kValueFits = 18;
void put_fit(char16_t* out, std::size_t room, const std::string& s, bool grey, std::size_t fits) {
    std::size_t chars = 0;
    for (const char c : s) chars += (static_cast<unsigned char>(c) & 0xc0) != 0x80;
    const char* size = chars <= fits ? nullptr : chars * 5 <= fits * 6 ? "20" : chars * 5 <= fits * 7 ? "17" : "15";
    if (!size) {
        put_html(out, room, s, grey);
        return;
    }
    std::size_t k = 0;
    const auto put = [&](const char* t) {
        for (; *t && k + 1 < room; ++t) out[k++] = static_cast<char16_t>(*t);
    };
    put("<font size='");
    put(size);
    put("'>");
    const std::size_t close = 7;  // "</font>"
    if (k + close + 1 < room) put_html(out + k, room - k - close, s, grey);
    while (out[k]) ++k;
    put("</font>");
    out[k] = 0;
}

// ---- the page as built (the menu thread's) ------------------------------------
struct Built {
    std::uint8_t* dialog = nullptr;  // our section's dialog while it is open
    std::uint64_t vtable = 0;
    Page page;                       // what the host gave
    int records = 0;                 // rows built: min(rows, slots)
    int first = 0;                   // the page's row the first slot shows (a long page scrolls)
    bool scrolls = false;            // more rows than slots: every row a text row, moved by the host
    std::uint8_t value[kSlots]{};    // each slot's byte
    std::uint8_t last[kSlots]{};
    std::int32_t position[kSlots]{}; // a host's slider's position as last set
    char16_t label[kSlots][kText]{}, help[kSlots][kText]{}, text[kSlots][kText]{}, pair[kSlots][2][kText]{};
    std::uint32_t prev = 0;          // the raw buttons at the last update
    int held_dir = 0;                // Left -1 / Right +1 / Up -2 / Down +2 held, for the repeat
    std::chrono::steady_clock::time_point held_since, last_repeat;
    std::uint64_t popup = 0;         // a question we asked, while it is up
    int popup_row = -1;
    std::chrono::steady_clock::time_point opened_at;
};
Built g_b;
std::mutex g_mu;          // the page waiting to be shown, and the flags below
Page g_next;              // under g_mu: what the host wants up
bool g_have_next = false;
std::atomic<bool> g_want_open{false}, g_want_close{false}, g_rows_changed{false}, g_values_changed{false};
std::atomic<bool> g_yes{false};
// A change the player made goes to the config at the plugins' next frame,
// after this hook: the values are not read back until the frame after that.
std::atomic<int> g_hold_refresh{0};
std::atomic<int> g_state{0};  // 0 none, 1 pushed (opening), 2 up
std::atomic<bool> g_corner{false};
std::atomic<bool> g_on_subpage{false};      // Circle goes back (kept from the game)
std::atomic<bool> g_popup_up{false};        // our question is up: every button is its
std::atomic<bool> g_mask_down{false}, g_mask_up{false};  // the scroll's ends (kept from the game)
std::atomic<bool> g_mask_sides{false};      // Left and Right on a host's slider (kept from the game)
std::atomic<bool> g_hold_pressed{false};    // the page went: what is held stays out of the game until let go
// Until when a page of ours that went counts as none of the game's menus: the
// flow takes a moment to let its menu go (photo mode, whose page Circle hides,
// would end itself as if a game's menu had opened).
std::atomic<std::int64_t> g_alone_until_ms{0};
std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::atomic<std::uint32_t> g_raw{0};        // the player's buttons, before pad_filter
std::atomic<std::uint8_t> g_raw_lx{128}, g_raw_ly{128};
std::atomic<int> g_sound{-1};               // a menu sound for the game's thread to post
std::uint64_t g_pushed = 0;                 // the flow's open menu right after our push (game thread)
bool g_ours_on_flow = false;                // we pushed it (else it is inside the System list)
bool g_close_sent = false;                  // the flow was asked to take our page away
std::uint64_t g_pushed_changes = 0;         // the flow's +0xb18 right after our push
// A corner page the game's menu flow took, to come back once the flow has
// been quiet a moment (game thread).
bool g_reopen = false;
std::int64_t g_reopen_since_ms = 0;
std::atomic<bool> g_reopen_pending{false};  // for pad_filter
int g_reopens = 0;
// When the flow took the corner page the last two times (0: not since it was
// last shown by its plugin).
std::atomic<std::int64_t> g_take_ms[2] = {0, 0};
// The flow's closes and pushes that were not ours (game thread): when the
// last two came, and the count last seen.
std::uint64_t g_flow_seen = 0;
std::int64_t g_flow_change_ms[2] = {0, 0};
const char kBusy[] = "the game's menu flow closes its menus frame after frame here";
const char kTaken[] = "the game's menu flow took the corner page three times in 10 s";

void post_sound(int id) { g_sound.store(id, std::memory_order_relaxed); }

// Whether g_b.dialog is still our section: its vtable, and its first slot's
// widget still writing through our byte (a freed dialog's memory is reused).
bool alive() {
    if (!g_b.dialog || g_b.records < 1) return false;
    if (rd64(reinterpret_cast<std::uint64_t>(g_b.dialog)) != g_b.vtable) return false;
    if (option_rows::row_count(g_b.dialog) != g_b.records) return false;
    std::uint8_t* w = option_rows::row_widget(g_b.dialog, 0);
    if (!w) return false;
    const Widget kind = g_b.scrolls ? Widget::Text : g_b.page.rows[0].widget;
    if (kind == Widget::Slider) return true;  // a slider keeps its pointer elsewhere
    return option_rows::value_target(w) == reinterpret_cast<std::uint64_t>(&g_b.value[0]);
}

// The page's row a slot shows.
int row_of(int slot) { return g_b.first + slot; }

// The slot's texts from the page's row (a long page's rows are all text rows).
void fill_slot(int slot) {
    const int r = row_of(slot);
    if (r < 0 || r >= static_cast<int>(g_b.page.rows.size())) {
        g_b.label[slot][0] = g_b.help[slot][0] = g_b.text[slot][0] = 0;
        return;
    }
    const Row& row = g_b.page.rows[static_cast<std::size_t>(r)];
    put_fit(g_b.label[slot], kText, row.label, row.grey, kLabelFits);
    put_html(g_b.help[slot], kText, row.help, false);
    // (A host's slider's number is sized by own_slider, for its box.)
    if (!g_b.scrolls && row.widget == Widget::Slider && row.positions > 0) put_html(g_b.text[slot], kText, row.value, row.grey);
    else put_fit(g_b.text[slot], kText, row.value, row.grey, kValueFits);
    put_html(g_b.pair[slot][0], kText, row.pair[0], row.grey);
    put_html(g_b.pair[slot][1], kText, row.pair[1], row.grey);
}

void own_slider(std::uint8_t* dialog, int slot);

// The rows built into the open dialog, from g_b.page (the handler's, or a
// page changed in place).
void build_rows(void* dialog) {
    const int n = static_cast<int>(std::min<std::size_t>(g_b.page.rows.size(), kMaxRows));
    g_b.scrolls = n > kSlots;
    g_b.records = std::min(n, kSlots);
    g_b.first = 0;
    if (g_b.page.cursor >= g_b.records) g_b.first = std::min(g_b.page.cursor - g_b.records + 1, n - g_b.records);
    for (int s = 0; s < g_b.records; ++s) {
        fill_slot(s);
        const Row& row = g_b.page.rows[static_cast<std::size_t>(row_of(s))];
        const Widget kind = g_b.scrolls ? Widget::Text : row.widget;
        const bool own = kind == Widget::Slider && row.positions > 0;
        g_b.value[s] = g_b.last[s] = kind == Widget::Text || own ? 0 : static_cast<std::uint8_t>(row.on);
        const std::uint32_t id = kRowId + static_cast<std::uint32_t>(s % 8);
        switch (kind) {
        case Widget::Toggle:
            option_rows::add_toggle_row(dialog, &g_b.value[s], id, g_b.value[s], g_b.label[s], g_b.help[s]);
            break;
        case Widget::Pair:
            option_rows::add_pair_row(dialog, &g_b.value[s], id, g_b.pair[s][0], g_b.pair[s][1], g_b.value[s], g_b.label[s],
                                      g_b.help[s]);
            break;
        case Widget::Slider:
            option_rows::add_slider_row(dialog, &g_b.value[s], id, g_b.value[s], g_b.label[s], g_b.help[s]);
            if (own) own_slider(static_cast<std::uint8_t*>(dialog), s);
            break;
        case Widget::Text:
            option_rows::add_text_row(dialog, &g_b.value[s], id, g_b.label[s], g_b.help[s], g_b.text[s]);
            break;
        }
    }
}

// A value wrapper the guest builds into our buffer, released as the game's
// own stack wrappers are (engine/menu_pointer.cpp's).
struct Wrapper {
    alignas(16) std::uint8_t b[0x80]{};
    Wrapper() = default;
    Wrapper(const Wrapper&) = delete;
    Wrapper& operator=(const Wrapper&) = delete;
    std::uint32_t type() const {
        std::uint32_t t = 0;
        std::memcpy(&t, b + kValueType, sizeof(t));
        return t;
    }
    std::uint64_t iface() const {
        std::uint64_t v = 0;
        std::memcpy(&v, b + kValueIface, sizeof(v));
        return v;
    }
    std::uint64_t data() const {
        std::uint64_t v = 0;
        std::memcpy(&v, b + kValueData, sizeof(v));
        return v;
    }
    ~Wrapper() {
        if (!(type() & 0x40) || !iface()) return;
        const std::uint64_t release = rd64(rd64(iface()) + 0x18);
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(release)), iface(), b + 0x30, data());
    }
};

// A row's widget shown. The row builders (sub_1f2a100 a choice, sub_1f2ac00
// the slider) find their row's clip as the section's Item_<row>_0 (sub_1f2ba20,
// under the section's value at +0xdf0), hide the widgets they do not use and
// bind their own without showing it - and the Defaults row sub_1f20900 adds
// after the handler's rows (taken off again, option_rows::open_section) hides
// its clip's whole Widgets (sub_1f2bb00). A new section's are all shown, so a
// row built in place on a slot that held another kind, or that slot, needs
// its own shown again, and the Widgets over it.
constexpr std::size_t kSectionValue = 0xdf0;
constexpr std::size_t kSetVisible = 0x08;  // a value wrapper's slot 1 (sub_1ecb760)
void show_widget(std::uint8_t* dialog, int slot, Widget kind) {
    char item[16];
    std::snprintf(item, sizeof(item), "Item_%d_0", slot);
    Wrapper row;
    hle_call_guest<std::int64_t>(option_rows::fn(kChild), row.b, dialog + kSectionValue, item, 0);
    if (!(row.type() & 0x8f)) return;
    for (const char* path : {"Widgets", kind == Widget::Slider ? "Widgets/Slider" : "Widgets/ComboBox"}) {
        Wrapper part;
        hle_call_guest<std::int64_t>(option_rows::fn(kChild), part.b, row.b, path, 0);
        if (!(part.type() & 0x8f)) return;
        const std::uint64_t set_visible = rd64(rd64(reinterpret_cast<std::uint64_t>(part.b)) + kSetVisible);
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(set_visible)), part.b, 1);
    }
}

// ---- a menu of the game's under ours -----------------------------------------
// The menu flow stops the menu it stacks ours on, but not every menu the game
// has up is the flow's: the title's list goes on reading the pad under a page
// the host pushes over it (Down moved it, Cross would pick its item). So the
// lists that had the input just before such a push are held while the page is
// up: no input and no pointer (menu_pointer.cpp's list update asks holds()),
// and hidden, as the game hides a menu it opens another from.
constexpr std::uint64_t kListVTable = 0x5799fb0, kListRootVTable = 0x5799f90;
struct Seen {
    void* comp = nullptr;
    std::chrono::steady_clock::time_point at;
};
Seen g_seen[4];        // the game thread's: lists that had the input lately
void* g_under[4]{};    // held under our page

// Still a list (its memory not freed and reused for something else).
bool list_alive(void* comp) {
    const auto c = reinterpret_cast<std::uint64_t>(comp);
    const std::uint64_t vt = rd64(c), root_vt = rd64(c + kListRoot);
    const std::uint64_t list = option_rows::addr(kListVTable), root = option_rows::addr(kListRootVTable);
    return (vt == list || vt == list + 0x10) && (root_vt == root || root_vt == root + 0x10);
}

void show_list(void* comp, bool shown) {
    std::uint8_t* root = static_cast<std::uint8_t*>(comp) + kListRoot;
    const std::uint64_t set_visible = rd64(rd64(reinterpret_cast<std::uint64_t>(root)) + kSetVisible);
    hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(set_visible)), root, shown ? 1 : 0);
}

void note_list(void* comp) {
    const auto now = std::chrono::steady_clock::now();
    Seen* slot = &g_seen[0];
    for (Seen& s : g_seen) {
        if (s.comp == comp) {
            slot = &s;
            break;
        }
        if (s.at < slot->at) slot = &s;
    }
    slot->comp = comp;
    slot->at = now;
}

void hold_under() {
    const auto now = std::chrono::steady_clock::now();
    int n = 0;
    for (Seen& s : g_seen) {
        if (!s.comp || now - s.at > std::chrono::milliseconds(300) || !list_alive(s.comp)) continue;
        g_under[n++] = s.comp;
        show_list(s.comp, false);
    }
    for (Seen& s : g_seen) s = Seen{};
    if (n) host_log("plugin-dialog: over a menu of the game's: %d of its lists held until the page is gone", n);
}

void release_under() {
    for (void*& c : g_under) {
        if (c && list_alive(c)) show_list(c, true);
        c = nullptr;
    }
}

// A text field's text, as the rows' providers set theirs (HTML): `field` a
// value wrapper sub_1edb220 built.
void set_field_text(const std::uint8_t* field, const char16_t* text) {
    std::uint32_t type = 0;
    std::memcpy(&type, field + kValueType, sizeof(type));
    const std::uint64_t iface = rd64(reinterpret_cast<std::uint64_t>(field + kValueIface));
    if (!(type & 0x8f) || !iface) return;
    const std::uint64_t data = rd64(reinterpret_cast<std::uint64_t>(field + kValueData));
    const std::uint64_t settext = rd64(rd64(iface) + kSetText);
    hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(settext)), iface, data, text, 1);
}

// A slider of the host's (Row::positions): the game's slider given the range
// 0..positions in ones, put at the row's position, its number the host's.
// Left and Right are the host's on it (pad_filter keeps them from the game),
// so the game moves it only as asked here.
void own_slider(std::uint8_t* dialog, int slot) {
    const Row& row = g_b.page.rows[static_cast<std::size_t>(row_of(slot))];
    std::uint8_t* w = option_rows::row_widget(dialog, slot);
    if (!w) return;
    std::uint8_t* c = option_rows::value_list(w);
    const std::int32_t zero = 0, one = 1, top = row.positions;
    std::memcpy(c + kSliderMin, &zero, sizeof(zero));
    std::memcpy(c + kSliderMax, &top, sizeof(top));
    std::memcpy(c + kSliderStep, &one, sizeof(one));
    std::int32_t at = 0;
    std::memcpy(&at, c + kSliderValue, sizeof(at));
    if (at != row.on) hle_call_guest<std::int64_t>(option_rows::fn(kSliderSet), c, row.on, 0, kNoSound, 0);
    g_b.position[slot] = row.on;
    // The number's box holds two digits at the field's size (the game's are
    // 0 to 10): a longer number ("100", "0.25", "-100") is set smaller to fit,
    // a point or a sign counted half a digit.
    std::size_t len = 0, halves = 0;
    for (; g_b.text[slot][len]; ++len) halves += g_b.text[slot][len] == u'.' || g_b.text[slot][len] == u'-' ? 1 : 2;
    const char* size = halves <= 4 ? nullptr : halves <= 7 ? "15" : halves <= 8 ? "13" : "11";
    char16_t html[kText + 32];
    std::size_t k = 0;
    const auto put = [&](const char* t) {
        for (; *t && k + 1 < sizeof(html) / sizeof(html[0]); ++t) html[k++] = static_cast<char16_t>(*t);
    };
    if (size) {
        put("<font size='");
        put(size);
        put("'>");
    }
    for (std::size_t i = 0; i < len && k + 1 < sizeof(html) / sizeof(html[0]); ++i) html[k++] = g_b.text[slot][i];
    if (size) put("</font>");
    html[k] = 0;
    set_field_text(c + kSliderText, html);
}

// The game's own position of a host's slider (it moves it itself on buttons
// the dialog does not keep from it), or -1.
int slider_at(int slot) {
    std::uint8_t* w = option_rows::row_widget(g_b.dialog, slot);
    if (!w) return -1;
    std::int32_t at = 0;
    std::memcpy(&at, option_rows::value_list(w) + kSliderValue, sizeof(at));
    return at;
}

// The title, set on the section's title field (PlugTitle.StaticText_126020,
// the generated movie's) as the rows' providers set theirs. False when the
// movie's fields are not there.
bool set_title(std::uint8_t* dialog, const std::string& title) {
    static char16_t text[kText];
    put_html(text, kText, title, false);
    std::uint8_t* root = option_rows::rows_list(dialog) + kListRoot;
    Wrapper holder, field;
    hle_call_guest<std::int64_t>(option_rows::fn(kChild), holder.b, root - 8, "PlugTitle", 0);
    if (!(holder.type() & 0x8f)) return false;
    hle_call_guest<std::int64_t>(option_rows::fn(kChild), field.b, holder.b, "StaticText_126020", 0);
    if (!(field.type() & 0x8f) || !field.iface()) return false;
    const std::uint64_t settext = rd64(rd64(field.iface()) + kSetText);
    hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(settext)), field.iface(), field.data(), text, 1);
    return true;
}

// BBHOST_PLUGIN_DIALOG_LOG=1: the rows as built, for a scripted run.
bool tracing() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_PLUGIN_DIALOG_LOG");
        return e && e[0] == '1';
    }();
    return on;
}
void trace_rows() {
    if (!tracing()) return;
    for (int s = 0; s < g_b.records; ++s) {
        const Row& row = g_b.page.rows[static_cast<std::size_t>(row_of(s))];
        host_log("plugin-dialog:   slot %d: %d \"%s\" = \"%s\"%s", s, static_cast<int>(g_b.scrolls ? Widget::Text : row.widget),
                 row.label.c_str(), row.value.c_str(), row.positions ? " (the host's slider)" : "");
    }
}

GUEST_ABI void section_handler(void* dialog, void* params) {
    (void)params;
    option_rows::handler_begin();
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_b.page = g_next;
    }
    g_b.dialog = nullptr;
    build_rows(dialog);
    g_b.dialog = static_cast<std::uint8_t*>(dialog);
    std::memcpy(&g_b.vtable, dialog, sizeof(g_b.vtable));
    g_b.prev = g_raw.load(std::memory_order_relaxed);  // the Cross that opened it is not a press here
    g_b.popup = 0;
    g_popup_up.store(false);
    g_b.opened_at = std::chrono::steady_clock::now();
    static bool told = false;
    if (!told) {
        told = true;
        host_log("plugin-dialog: the handler runs on thread %ld", thread_id());
    }
    host_log("plugin-dialog: %s up, \"%s\", %zu rows in %d slots%s", g_b.page.corner ? kCornerName : kPageName,
             g_b.page.title.c_str(), g_b.page.rows.size(), g_b.records, g_b.scrolls ? ", scrolling" : "");
    trace_rows();
}

GUEST_ABI std::int64_t section_opener(void* root, void* params) {
    static void* handler = hle_wrap_fn(reinterpret_cast<void*>(&section_handler));
    bool corner = false;
    std::string title;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        corner = g_next.corner;
        title = g_next.title;
    }
    g_corner.store(corner);
    const std::int64_t dialog = option_rows::open_section(root, params, corner ? kCornerName : kPageName, handler);
    if (dialog && g_b.dialog) {
        // sub_1f20900's own Defaults row is off again: the rows are the page's.
        if (g_b.page.cursor > 0 && g_b.page.cursor < g_b.records + g_b.first)
            option_rows::set_cursor(option_rows::rows_list(g_b.dialog), g_b.page.cursor - g_b.first);
        if (!set_title(g_b.dialog, title)) {
            static bool told = false;
            if (!told) host_log("plugin-dialog: the section's title field is not there; its title stays the movie's");
            told = true;
        }
        g_state.store(2);
    }
    return dialog;
}

// The System list's Plugins row: the plugins' menus, listed as a page.
GUEST_ABI void* system_row_open(void* out, void* ctx) {
    plugin_panel_native_list();
    Page p;
    if (!plugin_panel_native_page(p)) return out;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_next = p;
        g_have_next = true;
    }
    g_ours_on_flow = false;
    g_state.store(1);
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&section_opener));
    return option_rows::section_screen(out, ctx, opener);
}

// Pushed on the flow from the game's thread.
bool push(std::uint64_t f) {
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&section_opener));
    std::uint64_t screen = 0;
    option_rows::section_screen(&screen, reinterpret_cast<void*>(static_cast<std::uintptr_t>(f + kFlowCtx)), opener);
    if (!screen) return false;
    const std::uint64_t before = rd64(f + kFlowChanges);
    alignas(16) std::uint8_t callback[0x40]{};  // an empty std::function: no callback
    std::uint64_t out = 0;
    hle_call_guest<std::int64_t>(option_rows::fn(kPushScreen), &out, f, &screen, &callback);
    option_rows::release(out);
    option_rows::release(screen);
    g_close_sent = false;
    // Refused, the menu that was open (if one was) still is.
    g_pushed_changes = rd64(f + kFlowChanges);
    g_pushed = g_pushed_changes != before ? rd64(f + kFlowMenu) : 0;
    return g_pushed != 0;
}

// The game's menu flow closed or replaced the page we pushed (not Circle, which
// it does not count, and not as we asked).
bool taken_by_game(std::uint64_t f) { return g_ours_on_flow && f && rd64(f + kFlowChanges) != g_pushed_changes; }

// The wait for a corner page the flow took is over; `why` the game's dialog
// will not have it (the host draws it until it is next shown), or nullptr.
void end_reopen(const char* why) {
    g_reopen = false;
    g_reopen_pending.store(false);
    if (tracing()) host_log("plugin-dialog: the corner page's buttons go where they went before it was taken");
    if (!why) return;
    static bool told_busy = false, told_taken = false, told_refused = false;
    bool& told = why == kBusy ? told_busy : why == kTaken ? told_taken : told_refused;
    if (!told) {
        told = true;
        host_log("plugin-dialog: the corner page is drawn by the host: %s", why);
    }
    plugin_panel_native_failed(why);
}

// A page the game's flow took: a corner page waits to come back, a centred one
// is closed, as by the player but said to be the game's.
void game_took(bool came_up) {
    const std::int64_t ms = now_ms();
    g_flow_change_ms[1] = g_flow_change_ms[0];
    g_flow_change_ms[0] = ms;
    if (g_corner.load() && plugin_panel_native_corner_wanted()) {
        // Taken a third time within 10 s: it would only flicker back and go.
        const std::int64_t before_last = g_take_ms[1].load();
        g_take_ms[1].store(g_take_ms[0].load());
        g_take_ms[0].store(ms);
        if (before_last && ms - before_last <= 10000) {
            end_reopen(kTaken);
            return;
        }
        if (g_reopens++ < 8)
            host_log("plugin-dialog: the game's menu flow took the corner page%s; it comes back once the flow is quiet",
                     came_up ? "" : " before it came up");
        g_reopen = true;
        g_reopen_pending.store(true);
        g_reopen_since_ms = ms;
        if (tracing()) host_log("plugin-dialog: the corner page's buttons kept from the game until it is back");
        return;
    }
    host_log("plugin-dialog: down (taken by the game's menu flow%s)", came_up ? "" : " before it came up");
    plugin_panel_native_gone(true, "the game's menu flow");
}

// Why the flow will not take a push now, or nullptr.
const char* refused(std::uint64_t f) {
    if (!f) return "no menu flow";
    const std::uint64_t mm = rd64(option_rows::addr(kMenuManPtr));
    if (mm && rd32(mm + kMenuManNoPush)) return "the menus are locked";
    if (rd64(f + kFlowBlocked)) return "the flow is blocked";
    return nullptr;
}

// ---- the rows' work, on the menu thread --------------------------------------

// The host's page again: its values into the slots (or, changed shape, the
// rows rebuilt).
bool refresh() {
    Page p;
    if (!plugin_panel_native_page(p)) return false;
    bool same = p.rows.size() == g_b.page.rows.size() && p.corner == g_b.page.corner && p.title == g_b.page.title;
    for (std::size_t i = 0; same && i < p.rows.size(); ++i)
        same = p.rows[i].widget == g_b.page.rows[i].widget && p.rows[i].positions == g_b.page.rows[i].positions;
    if (!same) return false;
    g_b.page.rows = std::move(p.rows);
    for (int s = 0; s < g_b.records; ++s) {
        char16_t was[kText];
        std::memcpy(was, g_b.text[s], sizeof(was));
        fill_slot(s);
        const Row& row = g_b.page.rows[static_cast<std::size_t>(row_of(s))];
        if (!g_b.scrolls && row.widget == Widget::Slider && row.positions > 0) {
            // A host's slider: its position and number, when either moved.
            if (g_b.position[s] != row.on || slider_at(s) != row.on || std::memcmp(was, g_b.text[s], sizeof(was)) != 0)
                own_slider(g_b.dialog, s);
            continue;
        }
        if (!g_b.scrolls && row.widget != Widget::Text && g_b.value[s] != static_cast<std::uint8_t>(row.on)) {
            g_b.value[s] = g_b.last[s] = static_cast<std::uint8_t>(row.on);
        }
        std::uint8_t* w = option_rows::row_widget(g_b.dialog, s);
        if (w && (g_b.scrolls || row.widget != Widget::Slider)) option_rows::redraw(option_rows::value_list(w));
    }
    option_rows::redraw(option_rows::rows_list(g_b.dialog));
    return true;
}

// A page of another shape: the rows taken off and built again in the open
// dialog, the cursor on the page's.
// Asked from the menu's update: done from the frame hook, outside the
// dialog's update, which goes on with the row it started on.
std::atomic<bool> g_rebuild{false};
void rebuild() { g_rebuild.store(true); }

void rebuild_now() {
    Page p;
    if (!plugin_panel_native_page(p)) return;
    g_b.page = std::move(p);
    option_rows::clear_rows(g_b.dialog);
    build_rows(g_b.dialog);
    for (int s = 0; s < g_b.records; ++s)
        show_widget(g_b.dialog, s, g_b.scrolls ? Widget::Text : g_b.page.rows[static_cast<std::size_t>(row_of(s))].widget);
    option_rows::layout(g_b.dialog);
    option_rows::set_cursor(option_rows::rows_list(g_b.dialog), std::max(0, g_b.page.cursor - g_b.first));
    set_title(g_b.dialog, g_b.page.title);
    host_log("plugin-dialog: \"%s\" in place, %zu rows", g_b.page.title.c_str(), g_b.page.rows.size());
    trace_rows();
}

GUEST_ABI std::int64_t question_yes(std::int64_t, std::int64_t) {
    g_yes.store(true);
    return 0;
}

void press(int row) {
    std::string question;
    const int r = plugin_panel_native_press(row, &question);
    if (r == 3 && !question.empty()) {
        // The question on the section's own stack, its text written into the
        // bundle's message first (room for kPluginQuestionRoom).
        const char16_t* t = option_rows::dialog_message(kQuestionId);
        std::size_t room = 0;
        while (t && t[room] && room < kPluginQuestionRoom + 8) ++room;
        if (room >= 8) {
            char16_t buf[kPluginQuestionRoom + 1];
            put_html(buf, std::min<std::size_t>(room + 1, sizeof(buf) / sizeof(buf[0])), question, false);
            char16_t* w = const_cast<char16_t*>(t);
            std::size_t k = 0;
            for (; buf[k] && k < room; ++k) w[k] = buf[k];
            for (; k < room; ++k) w[k] = u' ';
            static void* yes = hle_wrap_fn(reinterpret_cast<void*>(&question_yes));
            g_yes.store(false);
            g_b.popup = option_rows::ask(g_b.dialog, kQuestionId, yes);
            g_b.popup_row = row;
            g_popup_up.store(g_b.popup != 0);
        }
        return;
    }
    if (r == 2) rebuild();
    else if (r == 1) {
        g_hold_refresh.store(1);
        g_values_changed.store(true);
    }
}

void step(int slot, int dir) {
    const int row = row_of(slot);
    if (row < 0 || row >= static_cast<int>(g_b.page.rows.size())) return;
    const Row& rw = g_b.page.rows[static_cast<std::size_t>(row)];
    if (rw.grey || !rw.steps) return;
    if (plugin_panel_native_step(row, dir)) {
        post_sound(kMenuSoundMove);
        g_hold_refresh.store(1);
        g_values_changed.store(true);
    }
}

// The window moved one row (a long page), the cursor kept on its slot.
void scroll(int by) {
    const int n = static_cast<int>(g_b.page.rows.size());
    const int to = std::clamp(g_b.first + by, 0, std::max(0, n - g_b.records));
    if (to == g_b.first) return;
    g_b.first = to;
    for (int s = 0; s < g_b.records; ++s) {
        fill_slot(s);
        std::uint8_t* w = option_rows::row_widget(g_b.dialog, s);
        if (w) option_rows::redraw(option_rows::value_list(w));
    }
    option_rows::redraw(option_rows::rows_list(g_b.dialog));
    post_sound(kMenuSoundMove);
}

void rows_update() {
    const auto now = std::chrono::steady_clock::now();
    std::uint8_t* list = option_rows::rows_list(g_b.dialog);
    const int slot = option_rows::cursor_of(list);
    // A question we asked: YES runs the row; NO (or Circle) pops it.
    if (g_b.popup) {
        if (g_yes.exchange(false)) {
            if (plugin_panel_native_confirm(g_b.popup_row)) rebuild();
            else {
                g_hold_refresh.store(1);
                g_values_changed.store(true);
            }
        }
        if (option_rows::refs(g_b.popup) <= 1) {
            option_rows::release(g_b.popup);
            g_b.popup = 0;
            g_popup_up.store(false);
        }
        g_b.prev = g_raw.load(std::memory_order_relaxed);
        return;
    }
    // The game's own widgets: a byte the player changed - or, on a host's
    // slider, a position the game moved it to itself.
    for (int s = 0; !g_b.scrolls && s < g_b.records; ++s) {
        const Row& row = g_b.page.rows[static_cast<std::size_t>(row_of(s))];
        if (row.widget == Widget::Slider && row.positions > 0) {
            const int at = slider_at(s);
            if (at < 0 || at == g_b.position[s]) continue;
            g_b.position[s] = at;
            if (plugin_panel_native_set(row_of(s), at)) {
                g_hold_refresh.store(1);
                g_values_changed.store(true);
            }
            continue;
        }
        if (g_b.value[s] == g_b.last[s]) continue;
        g_b.last[s] = g_b.value[s];
        if (plugin_panel_native_set(row_of(s), g_b.value[s])) {
            g_hold_refresh.store(1);
            g_values_changed.store(true);
        }
    }
    const std::uint32_t raw = g_raw.load(std::memory_order_relaxed);
    const std::uint8_t lx = g_raw_lx.load(std::memory_order_relaxed), ly = g_raw_ly.load(std::memory_order_relaxed);
    std::uint32_t down = raw;
    if (lx < 64) down |= kLeft;
    if (lx > 192) down |= kRight;
    if (ly < 64) down |= kUp;
    if (ly > 192) down |= kDown;
    const std::uint32_t edge = down & ~g_b.prev;
    g_b.prev = down;
    // Too soon after it opened: what opened it is still down.
    if (now - g_b.opened_at < std::chrono::milliseconds(150)) return;
    // Held Left / Right (and a long page's Up / Down at its ends) repeat.
    int dir = 0;
    if (down & kLeft) dir = -1;
    else if (down & kRight) dir = 1;
    else if (down & kUp) dir = -2;
    else if (down & kDown) dir = 2;
    bool fire = false;
    if (dir && (edge & (kLeft | kRight | kUp | kDown))) {
        g_b.held_dir = dir;
        g_b.held_since = g_b.last_repeat = now;
        fire = true;
    } else if (dir && dir == g_b.held_dir && now - g_b.held_since > std::chrono::milliseconds(380) &&
               now - g_b.last_repeat > std::chrono::milliseconds(70)) {
        g_b.last_repeat = now;
        fire = true;
    }
    if (!dir) g_b.held_dir = 0;
    if (tracing() && (edge || fire)) host_log("plugin-dialog: input edge %05x down %05x slot %d fire %d dir %d", edge, down, slot, fire ? 1 : 0, dir);
    if (fire && slot >= 0 && slot < g_b.records) {
        if (dir == -1 || dir == 1) {
            const Row& row = g_b.page.rows[static_cast<std::size_t>(row_of(slot))];
            if (g_b.scrolls || row.widget == Widget::Text || row.positions > 0) step(slot, dir);
        } else if (g_b.scrolls && dir == 2 && slot == g_b.records - 1) {
            scroll(1);
        } else if (g_b.scrolls && dir == -2 && slot == 0) {
            scroll(-1);
        }
    }
    // Cross on a text row (the game's own rows take theirs).
    if ((edge & menu_confirm_button()) && slot >= 0 && slot < g_b.records) {
        const Row& row = g_b.page.rows[static_cast<std::size_t>(row_of(slot))];
        if ((g_b.scrolls || row.widget == Widget::Text) && row.presses && !row.grey) {
            post_sound(kMenuSoundDecide);
            press(row_of(slot));
        }
    }
    // L1 / R1: the menu's pages.
    if (edge & (kL1 | kR1)) {
        if (plugin_panel_native_turn((edge & kR1) ? 1 : -1)) {
            post_sound(kMenuSoundMove);
            rebuild();
        }
    }
    // Circle on a page opened on top (the game does not see it): back.
    if ((edge & menu_back_button()) && g_on_subpage.load()) {
        if (plugin_panel_native_back()) {
            post_sound(kMenuSoundCancel);
            rebuild();
        }
    }
    // The ends of a long page's window: Up and Down there are ours.
    const int now_slot = option_rows::cursor_of(list);
    g_mask_down.store(g_b.scrolls && now_slot == g_b.records - 1 && g_b.first + g_b.records < static_cast<int>(g_b.page.rows.size()));
    g_mask_up.store(g_b.scrolls && now_slot == 0 && g_b.first > 0);
    g_mask_sides.store(!g_b.scrolls && now_slot >= 0 && now_slot < g_b.records &&
                       g_b.page.rows[static_cast<std::size_t>(row_of(now_slot))].positions > 0);
}

}  // namespace

bool available() { return g_available && option_rows::ready(); }

// The panel's page to the game's dialog: in place when it is up, else pushed at
// the next frame.
static bool open_page() {
    if (!available()) return false;
    Page p;
    if (!plugin_panel_native_page(p)) return false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_next = p;
        g_have_next = true;
    }
    if (g_state.load() == 2) {
        g_rows_changed.store(true);  // up already: the page in place
        return true;
    }
    g_want_close.store(false);
    g_want_open.store(true);
    return true;
}

bool open() {
    // Shown by its plugin: the flow's takes are counted afresh.
    if (g_state.load() != 2) {
        g_take_ms[0].store(0);
        g_take_ms[1].store(0);
    }
    return open_page();
}

void close() {
    g_want_open.store(false);
    if (g_state.load() != 0) g_want_close.store(true);
}

void changed(bool rows) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (rows) g_rows_changed.store(true);
    else g_values_changed.store(true);
}

bool up() { return g_state.load() == 2; }

bool alone() { return (g_state.load() != 0 && g_ours_on_flow && !g_under[0]) || now_ms() < g_alone_until_ms.load(); }

bool holds(void* comp) {
    if (!comp || g_state.load() == 0) return false;
    for (void* c : g_under)
        if (c == comp) return true;
    return false;
}
bool corner_up() { return up() && g_corner.load(); }

void frame() {
    if (!available()) return;
    plugin_panel_native_frame();
    if (const int s = g_sound.exchange(-1); s >= 0) menu_sound_post(s, s == kMenuSoundDecide || s == kMenuSoundCancel ? 1 : 0);
    static bool told = false;
    if (!told) {
        told = true;
        host_log("plugin-dialog: the frame hook runs on thread %ld", thread_id());
    }
    const std::uint64_t f = flow();
    const int state = g_state.load();
    const std::int64_t ms = now_ms();
    if (f) {
        // The game's own closes and pushes while no page of ours is on the flow
        // (ours are counted in as they are made).
        const std::uint64_t c = rd64(f + kFlowChanges);
        if (c != g_flow_seen && state == 0) {
            g_flow_change_ms[1] = g_flow_change_ms[0];
            g_flow_change_ms[0] = ms;
        }
        g_flow_seen = c;
    }
    const bool busy = ms - g_flow_change_ms[1] < 500;  // two in half a second
    if (state == 1 && !alive() && taken_by_game(f) && rd64(f + kFlowMenu) != g_pushed) {
        // The flow closed or replaced it before the page came up.
        g_state.store(0);
        g_b.dialog = nullptr;
        release_under();
        game_took(false);
        return;
    }
    if (state == 1 && !alive() && std::chrono::steady_clock::now() - g_b.opened_at > std::chrono::seconds(3)) {
        // Pushed, but the page never came up: the host draws it instead.
        g_state.store(0);
        g_b.dialog = nullptr;
        release_under();
        host_log("plugin-dialog: the page did not open");
        plugin_panel_native_failed("the game did not open its dialog");
        return;
    }
    if (state == 2 || state == 1) {
        // Ours while the dialog lives (inside the System list, or pushed by us).
        if (state == 2 && !alive()) {
            g_state.store(0);
            g_b.dialog = nullptr;
            g_b.popup = 0;
            g_popup_up.store(false);
            g_on_subpage.store(false);
            g_mask_down.store(false);
            g_mask_up.store(false);
            g_mask_sides.store(false);
            if (g_ours_on_flow && !g_under[0]) g_alone_until_ms.store(now_ms() + 1000);
            release_under();
            g_close_sent = false;
            // The Circle that closed it must not also go back in the menu of the
            // game's it uncovers (that menu never saw it go down), nor roll.
            g_hold_pressed.store(true);
            const bool asked = g_want_close.exchange(false);
            if (!asked && taken_by_game(f)) {
                game_took(true);
                return;
            }
            host_log("plugin-dialog: down (%s)", asked ? "asked" : "closed");
            plugin_panel_native_gone(!asked);
            return;
        }
        if (state == 2 && !g_want_close.load() && (g_rebuild.exchange(false) || g_rows_changed.exchange(false))) {
            rebuild_now();
        } else if (state == 2 && !g_want_close.load() && !g_b.popup) {
            // The values again a few times a second: a change made through the
            // config reaches it a frame later, and the plugin or F10 may change
            // one too.
            static std::chrono::steady_clock::time_point last;
            const auto now = std::chrono::steady_clock::now();
            if (g_hold_refresh.load() > 0) {
                g_hold_refresh.fetch_sub(1);
            } else if (g_values_changed.exchange(false) || now - last > std::chrono::milliseconds(150)) {
                last = now;
                if (!refresh()) rebuild_now();
            }
        }
        if (g_want_close.load() && state == 2 && !g_close_sent) {
            if (!g_ours_on_flow) {
                hle_pad_tap(menu_back_button(), 0, 80);  // inside the System list: back out of it
            } else if (f && rd64(f + kFlowMenu) == g_pushed) {
                *reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(f + kFlowClose)) = 1;
                g_close_sent = true;
            }
        }
        return;
    }
    if (g_reopen) {
        if (g_want_open.load() || !plugin_panel_native_corner_wanted()) {
            // Shown again or taken down by its plugin meanwhile: as it asks.
            end_reopen(nullptr);
        } else {
            // Back once the flow has neither closed nor pushed a menu for a
            // moment and takes a push. A second after the take, drawn by the
            // host instead where the flow keeps closing menus or will not take
            // one, as the open below does.
            const char* why = refused(f);
            if (busy || why || ms - g_flow_change_ms[0] < 300) {
                if ((busy || why) && ms - g_reopen_since_ms > 1000) end_reopen(busy ? kBusy : why);
                return;
            }
            end_reopen(nullptr);
            if (!open_page()) return;
        }
    }
    static std::chrono::steady_clock::time_point asked_at;
    static bool waiting = false;
    if (!g_want_open.load()) {
        waiting = false;
        return;
    }
    if (busy) {
        g_want_open.store(false);
        static int told = 0;
        if (told++ < 4) host_log("plugin-dialog: not opened: %s", kBusy);
        plugin_panel_native_failed(kBusy);
        return;
    }
    if (const char* why = refused(f)) {
        // A moment of it (a menu fading) is waited out; longer, the host draws the page.
        const auto now = std::chrono::steady_clock::now();
        if (!waiting) {
            waiting = true;
            asked_at = now;
        } else if (now - asked_at > std::chrono::seconds(1)) {
            waiting = false;
            g_want_open.store(false);
            host_log("plugin-dialog: not opened: %s", why);
            plugin_panel_native_failed(why);
        }
        return;
    }
    waiting = false;
    g_want_open.store(false);
    g_ours_on_flow = true;
    g_b.opened_at = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_corner.store(g_next.corner);
    }
    g_state.store(1);
    hold_under();
    const bool pushed = push(f);
    g_flow_seen = rd64(f + kFlowChanges);
    if (!pushed) {
        g_state.store(0);
        release_under();
        host_log("plugin-dialog: the flow did not take the page");
        plugin_panel_native_failed("the menu flow did not take it");
    }
}

void list_update(void* comp, bool focused) {
    if (focused && g_state.load() == 0) note_list(comp);
    if (!g_b.dialog) return;
    if (comp != option_rows::rows_list(g_b.dialog)) return;
    if (!alive()) {
        g_b.dialog = nullptr;
        return;
    }
    g_on_subpage.store(plugin_panel_native_on_subpage());
    if (focused) rows_update();
}

bool click_decides(void* list, int index) {
    if (!g_b.dialog || list != option_rows::rows_list(g_b.dialog) || index < 0 || index >= g_b.records) return false;
    const Row& row = g_b.page.rows[static_cast<std::size_t>(row_of(index))];
    return row.presses && !row.steps && !row.grey;
}

void system_row(void* builder) {
    if (!available() || !plugin_panel_native_menus_any()) return;
    static void* open = hle_wrap_fn(reinterpret_cast<void*>(&system_row_open));
    option_rows::add_command_row(builder, kSystemRowId, open);
    static int told = 0;
    if (told++ < 2) host_log("plugin-dialog: System gains Plugins");
}

void pad_filter(std::uint32_t& buttons, std::uint8_t& lx, std::uint8_t& ly, std::uint8_t& rx, std::uint8_t& ry, std::uint8_t& l2,
                std::uint8_t& r2) {
    g_raw.store(buttons, std::memory_order_relaxed);
    g_raw_lx.store(lx, std::memory_order_relaxed);
    g_raw_ly.store(ly, std::memory_order_relaxed);
    // A button kept from the game stays kept until let go: the Circle that
    // went back from a page on top must not then close the page under it.
    static std::uint32_t held_mask = 0;
    if (g_hold_pressed.exchange(false, std::memory_order_relaxed)) held_mask |= buttons;
    if (g_state.load(std::memory_order_relaxed) != 2) {
        held_mask &= buttons;
        buttons &= ~held_mask;
        // A corner page on its way back keeps its buttons from the game.
        if (g_reopen_pending.load(std::memory_order_relaxed)) buttons &= ~kDialogButtons;
        return;
    }
    // L1 and R1 turn our pages, Circle goes back from a page on top - but
    // with our question up every button is the question's.
    std::uint32_t mask = 0;
    if (!g_popup_up.load(std::memory_order_relaxed)) {
        mask |= kL1 | kR1;
        if (g_on_subpage.load(std::memory_order_relaxed)) mask |= menu_back_button();
    }
    if (g_mask_down.load(std::memory_order_relaxed)) {
        mask |= kDown;
        if (ly > 128) ly = 128;
    }
    if (g_mask_up.load(std::memory_order_relaxed)) {
        mask |= kUp;
        if (ly < 128) ly = 128;
    }
    if (g_mask_sides.load(std::memory_order_relaxed)) {
        mask |= kLeft | kRight;
        lx = 128;
    }
    held_mask = (held_mask | (buttons & mask)) & buttons;
    buttons &= ~(mask | held_mask);
    if (g_corner.load(std::memory_order_relaxed)) {
        // Photo mode's camera flies with the rest.
        buttons &= kDialogButtons;
        lx = ly = rx = ry = 128;
        l2 = r2 = 0;
    }
}

}  // namespace plugin_dialog

// The install: the generated movie has the sections (engine/menu_assets.cpp
// made them) - checked once, as option_menu checks its own.
void plugin_dialog_install() {
    // The movie the game will read: the generated one, or a player's own in
    // the mods folder, which wins and may be older than the pages' sections.
    const std::string movie = hle_fs_overlay_file(plugin_dialog::kMovie);
    const char* why = nullptr;
    if (movie.empty()) {
        why = "no generated options movie";
    } else {
        std::ifstream f(movie, std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (bytes.find(plugin_dialog::kPageName) == std::string::npos || bytes.find(plugin_dialog::kCornerName) == std::string::npos)
            why = "the options movie read has no plugin pages (one in the mods folder from an older bbhost?)";
    }
    if (why) {
        host_log("plugin-dialog: %s; plugins' menus are drawn by the host", why);
        return;
    }
    plugin_dialog::g_available = true;
}
