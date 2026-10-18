#include "guest_abi.h"
#include "engine/key_prompts.h"

#include "core/elf.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "engine/menu_pointer.h"
#include "host/bindings.h"
#include "host/options.h"
#include "host/settings.h"
#include "host/window.h"
#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// The wide-string formatter every button glyph is built with, and the first 18
// bytes of it: push rbp; mov rbp, rsp; push r15; push r14; push r12; push rbx;
// sub rsp, 0x120. It is variadic - al carries how many vector registers hold
// arguments - so the hook keeps rax (engine/graphics_patch.h).
constexpr std::uint64_t kFormatter = 0x1ee7270;
constexpr std::uint8_t kFormatterPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                              0x54, 0x53, 0x48, 0x81, 0xec, 0x20, 0x01, 0x00, 0x00};

// What the game asks for, and what the port formats instead. The brackets are
// the port's: a key name has to read as a key where a glyph used to sit.
const char16_t kImgFormat[] = u"<img src='%s' vspace='-8'>";
const char16_t kTextFormat[] = u"[%s]";

// One `KG_` tag and the actions it stands for. A tag naming a direction pair
// or a whole stick takes several, and they print in this order. Two of them
// are not actions but pad buttons the region decides (kCrossTag below).
struct Tag {
    const char* name;  // after "KG_"
    int action[4];     // -1 unused
};
constexpr int kNone = -1;
// `KG_OK` and `KG_Cancel` are not "confirm" and "back": they are the movie's
// Circle and Cross images, named the Japanese way round, and the game picks
// which of them the confirm row gets from the region. The US disc's confirm
// row draws KG_Cancel - the cross - which is how this was settled. So the two
// stand for pad buttons, and which key each names depends on the region
// (engine/menu_pointer.h).
constexpr int kCrossTag = -2, kCircleTag = -3;
const Tag kTags[] = {
    {"OK", {kCircleTag, kNone, kNone, kNone}},
    {"Cancel", {kCrossTag, kNone, kNone, kNone}},
    // The shoulders and triggers, and the sticks pressed in.
    {"R1", {kBindAttack, kNone, kNone, kNone}},
    {"R2", {kBindStrong, kNone, kNone, kNone}},
    {"L1", {kBindTransform, kNone, kNone, kNone}},
    {"L2", {kBindFirearm, kNone, kNone, kNone}},
    {"L3", {kBindL3, kNone, kNone, kNone}},
    {"R3", {kBindLockOn, kNone, kNone, kNone}},
    // The face buttons: the game's own help text reads "KG_R_U: Restore HP
    // (Uses Blood Vial)" and "KG_R_L: Use item", so `R_` is that cluster and
    // up is Triangle, left Square.
    {"R_U", {kBindBloodVial, kNone, kNone, kNone}},
    {"R_L", {kBindUseItem, kNone, kNone, kNone}},
    // The d-pad, singly and in pairs, and all four at once.
    {"L_U", {kBindDpadUp, kNone, kNone, kNone}},
    {"L_D", {kBindSwitchItem, kNone, kNone, kNone}},
    {"L_L", {kBindSwitchLeft, kNone, kNone, kNone}},
    {"L_R", {kBindSwitchRight, kNone, kNone, kNone}},
    {"L_UD", {kBindDpadUp, kBindSwitchItem, kNone, kNone}},
    {"L_LR", {kBindSwitchLeft, kBindSwitchRight, kNone, kNone}},
    {"L_UDLR", {kBindDpadUp, kBindSwitchItem, kBindSwitchLeft, kBindSwitchRight}},
    // The sticks themselves: the movement keys, and the camera keys - or the
    // mouse, when the mouse is what turns the camera.
    {"LS", {kBindMoveF, kBindMoveL, kBindMoveB, kBindMoveR}},
    {"RS", {kBindLookU, kBindLookD, kBindLookL, kBindLookR}},
    // The touchpad's two sides, which are two actions on one button.
    {"TP_L", {kBindGestures, kNone, kNone, kNone}},
    {"TP_R", {kBindEffects, kNone, kNone, kNone}},
};

std::atomic<std::uint64_t> g_rewritten{0}, g_unknown{0};
// The tags seen without an action, logged once each so a missing one shows up
// in a run rather than in a reading of this file.
constexpr int kSeenMax = 32;
char g_seen[kSeenMax][24];
std::atomic<int> g_seen_n{0};

bool wide_equals(const char16_t* a, const char16_t* b) {
    for (int i = 0; i < 256; ++i) {
        if (a[i] != b[i]) return false;
        if (!a[i]) return true;
    }
    return false;
}

// A `KG_` tag is ASCII; anything else is not one of ours.
bool tag_ascii(const char16_t* w, char* out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 1 < n && w[i]; ++i) {
        if (w[i] < 0x20 || w[i] > 0x7e) return false;
        out[i] = static_cast<char>(w[i]);
    }
    if (w[i]) return false;  // longer than a tag can be
    out[i] = '\0';
    return true;
}

// UTF-8 to UTF-16. Key names are ASCII, but a keyboard layout SDL names in its
// own language would not be, and the game's text is UTF-16 throughout.
void widen(const char* s, char16_t* out, std::size_t n) {
    std::size_t o = 0;
    for (const auto* p = reinterpret_cast<const unsigned char*>(s); *p && o + 1 < n; ++p) {
        std::uint32_t c = *p;
        if (c < 0x80) {
            // as is
        } else if ((c & 0xe0) == 0xc0 && p[1]) {
            c = ((c & 0x1f) << 6) | (p[1] & 0x3f);
            ++p;
        } else if ((c & 0xf0) == 0xe0 && p[1] && p[2]) {
            c = ((c & 0x0f) << 12) | ((p[1] & 0x3f) << 6) | (p[2] & 0x3f);
            p += 2;
        } else {
            c = '?';  // outside the BMP, or malformed: nothing a key is named
        }
        out[o++] = static_cast<char16_t>(c);
    }
    out[o] = 0;
}

bool same_set(const char names[4][24], int n, const char* const want[4]) {
    if (n != 4) return false;
    bool used[4] = {false, false, false, false};
    for (int i = 0; i < 4; ++i) {
        bool found = false;
        for (int j = 0; j < 4 && !found; ++j) {
            if (!used[j] && std::strcmp(names[i], want[j]) == 0) {
                used[j] = true;
                found = true;
            }
        }
        if (!found) return false;
    }
    return true;
}

// Cross and Circle in a menu: whichever of confirm and back the region gives
// them. In play they are Interact and Dodge, but a placeholder named for that
// use carries the meaning already (`<?CancelIcon?>: Backstep`), so only the
// menu sense is needed here.
int resolve(int action) {
    if (action != kCrossTag && action != kCircleTag) return action;
    const bool cross_confirms = menu_region_confirm_button() == 0x4000u;
    if (action == kCrossTag) return cross_confirms ? kBindConfirm : kBindBack;
    return cross_confirms ? kBindBack : kBindConfirm;
}

// The label for a tag, in UTF-8. Empty when nothing is bound to it, which
// leaves the game's own glyph in place.
void label_for(const Tag& t, bool mouse_camera, char* out, std::size_t n) {
    out[0] = '\0';
    char names[4][24];
    int count = 0;
    for (int i = 0; i < 4; ++i) {
        if (t.action[i] == kNone) continue;
        char one[24];
        host_binding_prompt(resolve(t.action[i]), one, sizeof(one));
        if (!one[0]) continue;
        std::snprintf(names[count], sizeof(names[count]), "%s", one);
        ++count;
    }
    if (!count) return;
    // A stick the mouse stands in for says so, rather than listing four keys
    // nobody uses for the camera.
    if (mouse_camera && t.action[0] == kBindLookU) {
        std::snprintf(out, n, "Mouse");
        return;
    }
    static const char* const kWasd[4] = {"W", "A", "S", "D"};
    static const char* const kArrows[4] = {"Up", "Down", "Left", "Right"};
    if (same_set(names, count, kWasd)) {
        std::snprintf(out, n, "WASD");
        return;
    }
    if (same_set(names, count, kArrows)) {
        std::snprintf(out, n, "Arrow Keys");
        return;
    }
    std::size_t o = 0;
    for (int i = 0; i < count && o + 1 < n; ++i) {
        o += static_cast<std::size_t>(std::snprintf(out + o, n - o, "%s%s", i ? "/" : "", names[i]));
    }
}

// The two settings the prompts follow, kept by the settings' listener
// (key_prompts_install registers it), which is told each change: a prompt
// reads a value, not the settings. They were asked on every call, each call
// an atomic exchange on one shared word - from the game's worker threads at
// once, ~1% of each of three EzWorkPool threads in a world profile.
std::atomic<int> g_prompt_mode{0};
std::atomic<bool> g_mouse_camera{true};

void settings_changed(const HostSettings& hs) {
    g_prompt_mode.store(hs.button_prompts, std::memory_order_relaxed);
    g_mouse_camera.store(hs.mouse_camera, std::memory_order_relaxed);
}

bool prompts_wanted() {
    switch (g_prompt_mode.load(std::memory_order_relaxed)) {
        case 1: return true;                    // Keyboard
        case 2: return false;                   // Controller
        default: return host_input_device() == InputDevice::KeyboardMouse;  // Auto: the device last used
    }
}

bool mouse_is_camera() { return g_mouse_camera.load(std::memory_order_relaxed); }

void note_unknown(const char* tag) {
    const int n = g_seen_n.load(std::memory_order_relaxed);
    for (int i = 0; i < n && i < kSeenMax; ++i) {
        if (std::strcmp(g_seen[i], tag) == 0) return;
    }
    if (n < kSeenMax) {
        std::snprintf(g_seen[n], sizeof(g_seen[n]), "%s", tag);
        g_seen_n.store(n + 1, std::memory_order_relaxed);
        host_log("key prompts: no action for the tag %s; its glyph stays", tag);
    }
    g_unknown.fetch_add(1, std::memory_order_relaxed);
}

// BBHOST_KEY_PROMPTS_TRACE=1: every format this formatter is handed, with its
// first argument, so a prompt that does not come through here shows itself.
std::atomic<int> g_trace{0};
bool tracing_messages() {
    static const bool on = [] {
        const char* e = std::getenv("BBHOST_KEY_PROMPTS_TRACE");
        return e && e[0] == '1';
    }();
    return on;
}
// UTF-16 to a printable ASCII line, for the log only.
void narrow(const char16_t* w, char* out, std::size_t n) {
    std::size_t i = 0;
    for (; i + 1 < n && w[i]; ++i) out[i] = w[i] < 0x20 || w[i] > 0x7e ? '.' : static_cast<char>(w[i]);
    out[i] = '\0';
}
std::uint64_t g_eboot_delta = 0;  // guest address - the eboot's own (Binary Ninja's), for the trace
void trace(const char16_t* fmt, std::uint64_t arg, std::uint64_t caller) {
    // Image tags only past the first 400, so a menu opened later is still traced.
    const bool img = wide_equals(fmt, kImgFormat);
    if (g_trace.fetch_add(1, std::memory_order_relaxed) >= (img ? 6000 : 400)) return;
    char f[80], a[80] = "(not a string)";
    narrow(fmt, f, sizeof(f));
    // Only an image tag's argument is known to be a string: the same formatter
    // takes "%d" and "%s(%d)", where reading the argument as a pointer faults.
    if (img && arg) {
        narrow(reinterpret_cast<const char16_t*>(static_cast<std::uintptr_t>(arg)), a, sizeof(a));
    }
    host_log("key prompts trace: \"%s\" <- \"%s\" from %#llx", f, a, static_cast<unsigned long long>(caller - g_eboot_delta));
}

// Switching device with a screen open (host_input_device). A screen formats
// its key guide's entries once, when it opens - only the list's own "Select"
// entry is formatted again each frame - and joins them into one HTML line it
// hands to its text field every frame (sub_1ed6370). So the line is mended on
// its way in, at the text field's SetText (GFx::Value::ObjectInterface's
// slot 0x138, sub_6f6f20, found from a live value's interface): with the keys
// in use a glyph tag becomes the key's label, and with a controller a label
// this hook made becomes the glyph tag again. Each label remembers the tag it
// was made for; one made for two tags is left alone.
constexpr std::uint64_t kSetText = 0x6f6f20;
constexpr std::uint8_t kSetTextPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                             0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x38};
std::mutex g_labels_mu;
std::unordered_map<std::u16string, std::u16string> g_label_tags;  // "[E]" -> "KG_R_U", "" when made for two
std::atomic<std::uint64_t> g_mended{0};
void note_label(const char16_t* label, const char16_t* tag) {
    std::u16string l = u"[" + std::u16string(label) + u"]", t(tag);
    std::lock_guard<std::mutex> lock(g_labels_mu);
    auto [it, fresh] = g_label_tags.emplace(l, t);
    if (!fresh && it->second != t) it->second.clear();
}
const std::u16string kImgOpen = u"<img src='";
const std::u16string kImgClose = u"' vspace='-8'>";
// The line with every glyph tag named as a key (keys) or every label of ours
// put back as its glyph (!keys); false when nothing in it changes.
bool mend_line(const char16_t* text, bool keys, std::u16string& out) {
    const std::u16string in(text);
    out.clear();
    bool changed = false;
    std::size_t at = 0;
    while (at < in.size()) {
        if (keys) {
            const std::size_t open = in.find(kImgOpen + u"KG_", at);
            if (open == std::u16string::npos) break;
            const std::size_t close = in.find(kImgClose, open);
            if (close == std::u16string::npos) break;
            const std::u16string tag = in.substr(open + kImgOpen.size(), close - open - kImgOpen.size());
            char ascii[24];
            const Tag* found = nullptr;
            if (tag_ascii(tag.c_str(), ascii, sizeof(ascii))) {
                for (const Tag& t : kTags) {
                    if (std::strcmp(t.name, ascii + 3) == 0) found = &t;
                }
            }
            char utf8[96] = "";
            if (found) label_for(*found, mouse_is_camera(), utf8, sizeof(utf8));
            out.append(in, at, open - at);
            if (utf8[0]) {
                char16_t w[96];
                widen(utf8, w, sizeof(w) / sizeof(w[0]));
                out += u"[" + std::u16string(w) + u"]";
                changed = true;
            } else {
                out.append(in, open, close + kImgClose.size() - open);
            }
            at = close + kImgClose.size();
        } else {
            const std::size_t open = in.find(u'[', at);
            if (open == std::u16string::npos) break;
            const std::size_t close = in.find(u']', open);
            if (close == std::u16string::npos) break;
            const std::u16string label = in.substr(open, close + 1 - open);
            std::u16string tag;
            {
                std::lock_guard<std::mutex> lock(g_labels_mu);
                if (auto it = g_label_tags.find(label); it != g_label_tags.end()) tag = it->second;
            }
            out.append(in, at, open - at);
            if (!tag.empty()) {
                out += kImgOpen + tag + kImgClose;
                changed = true;
            } else {
                out += label;
            }
            at = close + 1;
        }
    }
    if (!changed) return false;
    out.append(in, at, std::u16string::npos);
    return true;
}
// saved[] is r9, r8, rcx, rdx, rsi, rdi: the text is rdx, isHtml cl.
GUEST_ABI std::int64_t settext_hook(std::uint64_t, std::uint64_t* saved) {
    const auto* text = reinterpret_cast<const char16_t*>(static_cast<std::uintptr_t>(saved[3]));
    if (!text || !(saved[2] & 0xff)) return 0;  // rcx: isHtml - a tag in plain text would show as markup
    const bool keys = prompts_wanted();
    // The cheap test first: most text holds neither a tag nor a bracket.
    bool any = false;
    for (const char16_t* c = text; *c && !any; ++c) any = keys ? (c[0] == u'<' && c[1] == u'i') : c[0] == u'[';
    if (!any) return 0;
    if (!keys) {
        std::lock_guard<std::mutex> lock(g_labels_mu);
        if (g_label_tags.empty()) return 0;
    }
    static thread_local std::u16string mended;
    if (!mend_line(text, keys, mended)) return 0;
    saved[3] = reinterpret_cast<std::uint64_t>(mended.c_str());  // SetText copies it before it returns
    if (g_mended.fetch_add(1, std::memory_order_relaxed) == 0) host_log("key prompts: a line already on screen was mended for the device in use");
    return 0;
}

// saved[] is r9, r8, rcx, rdx, rsi, rdi: the format is rsi, the one argument
// rdx. Writing to them is what the guest then reads (core/thunk.h).
GUEST_ABI std::int64_t formatter_hook(std::uint64_t, std::uint64_t* saved) {
    const auto* fmt = reinterpret_cast<const char16_t*>(static_cast<std::uintptr_t>(saved[4]));
    static const bool tracing = [] {
        const char* e = std::getenv("BBHOST_KEY_PROMPTS_TRACE");
        return e && e[0] == '1';
    }();
    if (tracing && fmt) trace(fmt, saved[3], saved[6]);  // saved[6]: the return address, above the pushes
    // The cheap test first: this formatter builds every wide string the menus
    // put together, and only the image tags are ours.
    if (!fmt || fmt[0] != u'<' || fmt[1] != u'i' || !wide_equals(fmt, kImgFormat)) return 0;
    const auto* arg = reinterpret_cast<const char16_t*>(static_cast<std::uintptr_t>(saved[3]));
    if (!arg) return 0;
    char tag[24];
    if (!tag_ascii(arg, tag, sizeof(tag)) || std::strncmp(tag, "KG_", 3) != 0) return 0;
    if (!prompts_wanted()) return 0;
    const Tag* found = nullptr;
    for (const Tag& t : kTags) {
        if (std::strcmp(t.name, tag + 3) == 0) {
            found = &t;
            break;
        }
    }
    if (!found) {
        note_unknown(tag);
        return 0;
    }
    char utf8[96];
    label_for(*found, mouse_is_camera(), utf8, sizeof(utf8));
    if (!utf8[0]) return 0;  // nothing bound: the glyph says it better than "[]"
    // The guest reads both while it formats, and the formatter copies the
    // result before it returns, so one buffer per thread is enough.
    static thread_local char16_t label[96];
    widen(utf8, label, sizeof(label) / sizeof(label[0]));
    saved[4] = reinterpret_cast<std::uint64_t>(kTextFormat);
    saved[3] = reinterpret_cast<std::uint64_t>(label);
    note_label(label, arg);
    const std::uint64_t n = g_rewritten.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1) host_log("key prompts: %s is now [%s]", tag, utf8);
    return 0;  // and the game formats it
}

// --- the message text ----------------------------------------------------
//
// A glyph the game builds goes through the formatter above, which is how the
// menus' key guides change. The other half never reaches it, because the text
// already holds the tag. The world's tutorial notes are written that way
// ("<img src=\"KG_R1\" vspace='-8'>: Attack (Right hand weapon)", group 2),
// and the key guide group carries named placeholders instead
// ("<?selectUD?>:Select <?conclusion?>:Enter <?cancel?>:Close", group 79).
//
// Both are message text, so both are rewritten where the text comes from:
// `MsgRepositoryImp::LookupEntry`, guest 0x136f340. The hook does not change
// the return value - it cannot, being an entry hook - it changes what the
// lookup is about to find. An FMG category block keeps its strings as 8-byte
// offsets from the block, and the lookup returns `block + offsets[index]`, so
// writing a rewritten string's address into that slot as `host - block` makes
// the game's own search hand out the port's string. It is done once per
// message, and undone when a binding changes.
constexpr std::uint64_t kLookupEntry = 0x136f340;
constexpr std::uint8_t kLookupPrologue[] = {0x89, 0xf0, 0x48, 0x8b, 0x77, 0x08, 0x48, 0x8b,
                                            0x34, 0xc6, 0x31, 0xc0, 0x48, 0x85, 0xf6};

// A placeholder the game fills with a button, and what the port puts there.
// Actions are as above; the four menu directions are not bindings at all -
// the arrow keys work every menu list whatever the player bound
// (host/window.cpp) - so they are named outright.
struct Placeholder {
    const char* name;
    const char* fixed;  // the text, when it is not an action
    int action[4];
};
const Placeholder kPlaceholders[] = {
    {"conclusion", nullptr, {kBindConfirm, kNone, kNone, kNone}},
    {"cancel", nullptr, {kBindBack, kNone, kNone, kNone}},
    // These two are named for the button, not the row, and only the world's
    // notes use them: "<?CancelIcon?>: Backstep" is the dodge. The region
    // swaps which image each one picks *and* what that button does in play,
    // so the two cancel out and no region test is needed.
    {"OKIcon", nullptr, {kBindInteract, kNone, kNone, kNone}},
    {"CancelIcon", nullptr, {kBindRoll, kNone, kNone, kNone}},
    {"selectU", "Up", {kNone, kNone, kNone, kNone}},
    {"selectD", "Down", {kNone, kNone, kNone, kNone}},
    {"selectL", "Left", {kNone, kNone, kNone, kNone}},
    {"selectR", "Right", {kNone, kNone, kNone, kNone}},
    {"selectUD", "Up/Down", {kNone, kNone, kNone, kNone}},
    {"selectLR", "Left/Right", {kNone, kNone, kNone, kNone}},
    {"selectAll", "Arrow Keys", {kNone, kNone, kNone, kNone}},
    // The rows the guide gives a face button: Toggle Display is Triangle and
    // Remove / Delete is Square, which is the same pair as KG_R_U / KG_R_L.
    {"viewChange", nullptr, {kBindBloodVial, kNone, kNone, kNone}},
    {"commando", nullptr, {kBindUseItem, kNone, kNone, kNone}},
    // Rotate Face is the right stick, which the mouse stands in for.
    {"spin", nullptr, {kBindLookU, kBindLookD, kBindLookL, kBindLookR}},
    {"spinReset", nullptr, {kBindLockOn, kNone, kNone, kNone}},
    {"move", nullptr, {kBindMoveF, kBindMoveL, kBindMoveB, kBindMoveR}},
    // The shoulders: category change is L1/R1, the box shortcuts L2/R2.
    {"categoryChangeL", nullptr, {kBindTransform, kNone, kNone, kNone}},
    {"categoryChangeR", nullptr, {kBindAttack, kNone, kNone, kNone}},
    {"shortCutL", nullptr, {kBindFirearm, kNone, kNone, kNone}},
    {"shortCutR", nullptr, {kBindStrong, kNone, kNone, kNone}},
    {"pageUp", nullptr, {kBindTransform, kNone, kNone, kNone}},
    {"pageDown", nullptr, {kBindAttack, kNone, kNone, kNone}},
    {"startMenuSwitch", nullptr, {kBindMenu, kNone, kNone, kNone}},
    {"slectMenuSwitch", nullptr, {kBindGestures, kNone, kNone, kNone}},
};

template <typename T>
T rd(std::uint64_t va) {
    T v{};
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va)), sizeof(v));
    return v;
}

std::mutex g_msg_mu;
// The slots rewritten, with what they held, so a rebinding can put the game's
// own text back and have the next lookup build it again.
struct Patched {
    std::uint64_t slot;
    std::uint64_t was;
};
std::vector<Patched> g_patched;
std::atomic<std::size_t> g_patched_n{0};  // g_patched.size(), read without the lock
std::vector<std::u16string*> g_strings;  // kept alive: the game holds the pointer
std::unordered_map<std::uint64_t, char> g_examined;  // slot -> seen
std::uint64_t g_msg_serial = 0;
std::atomic<std::uint64_t> g_messages{0}, g_lookups{0}, g_slots{0}, g_examined_n{0};

// The label for a placeholder or a tag, bracketed. Empty when nothing is bound.
std::u16string bracketed(const char* fixed, const int action[4]) {
    char utf8[96] = {0};
    if (fixed) {
        std::snprintf(utf8, sizeof(utf8), "%s", fixed);
    } else {
        Tag t{"", {action[0], action[1], action[2], action[3]}};
        label_for(t, mouse_is_camera(), utf8, sizeof(utf8));
    }
    if (!utf8[0]) return std::u16string();
    char16_t w[96];
    widen(utf8, w, sizeof(w) / sizeof(w[0]));
    std::u16string out = u"[";
    out += w;
    out += u"]";
    return out;
}

bool ascii_name(const char16_t* p, std::size_t n, char* out, std::size_t m) {
    if (n + 1 > m) return false;
    for (std::size_t i = 0; i < n; ++i) {
        const char16_t c = p[i];
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        if (!ok) return false;
        out[i] = static_cast<char>(c);
    }
    out[n] = '\0';
    return true;
}

// Rewrites one message. False when it holds no button at all, which is nearly
// every message and costs one scan.
bool rewrite(const char16_t* in, std::u16string& out) {
    bool any = false;
    out.clear();
    for (std::size_t i = 0; in[i];) {
        if (in[i] != u'<') {
            out += in[i++];
            continue;
        }
        // <?name?>
        if (in[i + 1] == u'?') {
            std::size_t j = i + 2;
            while (in[j] && in[j] != u'?' && in[j] != u'>' && j - i < 40) ++j;
            if (in[j] == u'?' && in[j + 1] == u'>') {
                char name[40];
                if (ascii_name(in + i + 2, j - i - 2, name, sizeof(name))) {
                    bool took = false;
                    for (const Placeholder& p : kPlaceholders) {
                        if (std::strcmp(p.name, name) != 0) continue;
                        const std::u16string label = bracketed(p.fixed, p.action);
                        if (!label.empty()) {
                            out += label;
                            i = j + 2;
                            took = any = true;
                        }
                        break;
                    }
                    if (took) continue;
                }
            }
            out += in[i++];
            continue;
        }
        // <img src='KG_XX' ...>
        static const char16_t kImg[] = u"<img src=";
        std::size_t k = 0;
        while (kImg[k] && in[i + k] == kImg[k]) ++k;
        if (kImg[k] == 0 && (in[i + k] == u'\'' || in[i + k] == u'"')) {
            const char16_t quote = in[i + k];
            std::size_t j = i + k + 1;
            while (in[j] && in[j] != quote && j - i < 60) ++j;
            char name[40];
            if (in[j] == quote && ascii_name(in + i + k + 1, j - i - k - 1, name, sizeof(name)) &&
                std::strncmp(name, "KG_", 3) == 0) {
                std::size_t e = j;
                while (in[e] && in[e] != u'>' && e - i < 80) ++e;
                if (in[e] == u'>') {
                    bool took = false;
                    for (const Tag& t : kTags) {
                        if (std::strcmp(t.name, name + 3) != 0) continue;
                        const std::u16string label = bracketed(nullptr, t.action);
                        if (!label.empty()) {
                            out += label;
                            i = e + 1;
                            took = any = true;
                        }
                        break;
                    }
                    if (took) continue;
                }
            }
        }
        out += in[i++];
    }
    return any;
}

// The lookup's own search, so the hook knows which slot the game is about to
// read. Any disagreement ends in 0 and the prompt keeps its glyph.
std::uint64_t entry_slot(std::uint64_t repo, std::uint32_t version, std::uint32_t category, std::uint32_t entry,
                         std::uint64_t* block) {
    const std::uint64_t versions = rd<std::uint64_t>(repo + 8);
    if (!versions) return 0;
    const std::uint64_t cats = rd<std::uint64_t>(versions + std::uint64_t(version) * 8);
    if (!cats) return 0;
    const std::uint64_t b = rd<std::uint64_t>(cats + std::uint64_t(category) * 8);
    if (!b) return 0;
    const std::uint32_t ranges = rd<std::uint32_t>(b + 0x0c);
    if (!ranges || ranges > (1u << 20)) return 0;
    if (rd<std::uint32_t>(b + 0x2c) > entry) return 0;
    std::uint32_t lo = 0, hi = ranges - 1;
    if (rd<std::uint32_t>(b + std::uint64_t(hi) * 16 + 0x30) < entry) return 0;
    for (int guard = 0; guard < 64 && lo <= hi; ++guard) {
        const std::uint32_t mid = lo + (hi - lo) / 2;
        const std::uint64_t r = b + std::uint64_t(mid) * 16;
        if (rd<std::uint32_t>(r + 0x30) < entry) {
            lo = mid + 1;
            continue;
        }
        const std::uint32_t first = rd<std::uint32_t>(r + 0x2c);
        if (first > entry) {
            if (mid == 0) return 0;
            hi = mid - 1;
            continue;
        }
        const std::uint64_t offsets = rd<std::uint64_t>(b + 0x18);
        if (!offsets) return 0;
        const std::uint32_t index = entry - first + rd<std::uint32_t>(r + 0x28);
        *block = b;
        return offsets + std::uint64_t(index) * 8;
    }
    return 0;
}

GUEST_ABI std::int64_t lookup_hook(std::uint64_t, const std::uint64_t* saved) {
    g_lookups.fetch_add(1, std::memory_order_relaxed);
    const bool wanted = prompts_wanted();
    if (!wanted && g_patched_n.load(std::memory_order_relaxed) == 0) return 0;
    std::uint64_t block = 0;
    const std::uint64_t slot = entry_slot(saved[5], static_cast<std::uint32_t>(saved[4]),
                                          static_cast<std::uint32_t>(saved[3]),
                                          static_cast<std::uint32_t>(saved[2]), &block);
    if (!slot) return 0;
    g_slots.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_msg_mu);
    // A binding changed, or the player picked up the other device: the game's
    // own text goes back, and whatever is looked up next is built again - with
    // the new key, or left as the pad's glyph.
    const std::uint64_t serial = host_bindings_serial() * 2 + (wanted ? 1 : 0);
    if (serial != g_msg_serial) {
        for (const Patched& p : g_patched) {
            std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(p.slot)), &p.was, 8);
        }
        g_patched.clear();
        g_patched_n.store(0, std::memory_order_relaxed);
        g_examined.clear();
        g_msg_serial = serial;
    }
    if (!wanted) return 0;
    if (!g_examined.emplace(slot, 1).second) return 0;
    const std::uint64_t seen = g_examined_n.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t off = rd<std::uint64_t>(slot);
    if (!off) return 0;
    const auto* text = reinterpret_cast<const char16_t*>(static_cast<std::uintptr_t>(block + off));
    if (tracing_messages() && seen < 400) {
        char was[140];
        narrow(text, was, sizeof(was));
        host_log("key prompts seen: category %u entry %u \"%s\"", static_cast<unsigned>(saved[3]),
                 static_cast<unsigned>(saved[2]), was);
    }
    std::u16string built;
    if (!rewrite(text, built)) return 0;
    auto* kept = new std::u16string(built);
    g_strings.push_back(kept);
    const std::uint64_t at = reinterpret_cast<std::uint64_t>(kept->c_str());
    const std::uint64_t rel = at - block;
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(slot)), &rel, 8);
    g_patched.push_back({slot, off});
    g_patched_n.store(g_patched.size(), std::memory_order_relaxed);
    const std::uint64_t n = g_messages.fetch_add(1, std::memory_order_relaxed);
    if (n == 0 || tracing_messages()) {
        char was[120], now[120];
        narrow(text, was, sizeof(was));
        narrow(built.c_str(), now, sizeof(now));
        host_log("key prompts: category %u entry %u: \"%s\" -> \"%s\"", static_cast<unsigned>(saved[3]),
                 static_cast<unsigned>(saved[2]), was, now);
    }
    return 0;
}

// BBHOST_KEY_PROMPTS_TRACE=1: the rewrite over two lines of the game's own
// text, so a run says what it would do without having to reach the screen
// that shows them.
void self_check() {
    static const char16_t* const kSamples[] = {
        u"<img src=\"KG_R1\" vspace='-8'>: Attack (Right hand weapon)",
        u"<img src=\"KG_LS\" vspace='-8'>+<?CancelIcon?>: Roll/Quickstep",
        u"<?selectUD?>:Select <?conclusion?>:Enter <?cancel?>:Close",
    };
    for (const char16_t* sample : kSamples) {
        std::u16string built;
        const bool any = rewrite(sample, built);
        char was[160], now[160];
        narrow(sample, was, sizeof(was));
        narrow(any ? built.c_str() : sample, now, sizeof(now));
        host_log("key prompts check: \"%s\" -> \"%s\"", was, now);
    }
}

}  // namespace

void key_prompts_install(ElfImage* image) {
    host_settings_listen(&settings_changed);
    if (const char* e = std::getenv("BBHOST_KEY_PROMPTS"); e && e[0] == '0') {
        host_log("key prompts: off, the pad's glyphs stay");
        return;
    }
    const std::uint64_t at = image->mem.slide + (kFormatter - kPreferredGuestSlide);
    g_eboot_delta = at - kFormatter;
    const bool ok = engine_prologue_hook_variadic(image, at, kFormatterPrologue, sizeof(kFormatterPrologue),
                                                  reinterpret_cast<void*>(&formatter_hook));
    const std::uint64_t lookup = image->mem.slide + (kLookupEntry - kPreferredGuestSlide);
    const bool msg = engine_prologue_hook(image, lookup, kLookupPrologue, sizeof(kLookupPrologue),
                                          reinterpret_cast<void*>(&lookup_hook));
    const std::uint64_t set_text = image->mem.slide + (kSetText - kPreferredGuestSlide);
    const bool st = engine_prologue_hook(image, set_text, kSetTextPrologue, sizeof(kSetTextPrologue),
                                         reinterpret_cast<void*>(&settext_hook));
    if (!st) host_log("key prompts: did NOT hook the text field's SetText (0x%llx); a device switch shows on the next screen", static_cast<unsigned long long>(kSetText));
    host_log("key prompts: %s the wide formatter (0x%llx) and %s the message lookup (0x%llx); "
             "%d tags and %d placeholders carry the player's keys",
             ok ? "hooked" : "did NOT hook (unexpected bytes)",
             static_cast<unsigned long long>(kFormatter),
             msg ? "hooked" : "did NOT hook (unexpected bytes)",
             static_cast<unsigned long long>(kLookupEntry), static_cast<int>(sizeof(kTags) / sizeof(kTags[0])),
             static_cast<int>(sizeof(kPlaceholders) / sizeof(kPlaceholders[0])));
    if (tracing_messages()) self_check();
}

void key_prompts_report() {
    const std::uint64_t n = g_rewritten.load(std::memory_order_relaxed);
    const std::uint64_t m = g_messages.load(std::memory_order_relaxed);
    if (!n && !m && !g_lookups.load(std::memory_order_relaxed)) return;
    host_log("key prompts: %llu messages and %llu formatted glyphs named a key, %llu tags had no action "
             "(%llu lookups, %llu found, %llu distinct)",
             static_cast<unsigned long long>(m), static_cast<unsigned long long>(n),
             static_cast<unsigned long long>(g_unknown.load()),
             static_cast<unsigned long long>(g_lookups.load()),
             static_cast<unsigned long long>(g_slots.load()),
             static_cast<unsigned long long>(g_examined_n.load()));
}
