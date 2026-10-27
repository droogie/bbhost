#include "engine/option_menu.h"
#include "engine/graphics_patch.h"
#include "host/window.h"

#include "core/elf.h"
#include "core/thunk.h"
#include "engine/addr.h"
#include "engine/debug_menu.h"
#include "guest_abi.h"
#include "hle/fs.h"
#include "hle/hle.h"
#include "engine/menu_assets.h"
#include "engine/menu_pointer.h"
#include "engine/yebis.h"
#include "hle/modules.h"
#include "host/bindings.h"
#include "core/utf8.h"
#include "host/options.h"
#include "log.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>


// ---------------------------------------------------------------------------
// The System menu, as the engine actually builds it
//
// `sub_1faf8c0` is the menu-path dispatcher: it takes a wide string and opens
// the screen it names. `u"System"` runs `sub_1fb3ad0`; `u"System/Language"`,
// `u"System/Network"` and the rest run `sub_1fb4c70` with the matching opener.
//
// `sub_1fb3ad0` is a **command list**, built one row at a time into a local
// builder and finalised by `sub_1fea630`. A row is
//
//     sub_1f4e2a0(builder, captions, std::function<Dlg*(Dlg*, Ctx)>, &flag)
//
// and it is removed again by popping 0x180 bytes off the vector at
// builder+0x60. The six rows and what each one opens:
//
//     msg 0x1adb0  Controls     sub_1fb4e00 -> sub_1fdb960
//     msg 0x1adb1  Environment  sub_1fb4e90 -> sub_1fdb980
//     msg 0x1adb2  Brightness   sub_1fb4f20 -> sub_1fdb9a0
//     msg 0x1adb3  Network      sub_1fb4fb0 -> sub_1fdb9c0
//     msg 0x1adb4  Exit Game    sub_200c9f0            (added by sub_1f4c9a0)
//     msg 0x1adb5  Language     sub_1fb4be0 -> sub_1fdb940
//
// `sub_1fb3ad0`'s third argument decides which of the last two survives: with
// it set Language is popped, with it clear Exit Game is. Both callers the
// title flow uses pass 1.
//
// An **opener** is five instructions and nothing else:
//
//     sub_1fdb9x0(root, params) -> sub_1f20900(root, params, "<Name>Setting",
//                                              handler, 0, 0)
//
// and `sub_1f20900` does the rest: it walks the six-name table at 0x57398b0,
// binds the sprite whose name matches, allocates a 0xf20 dialog, and calls
// `handler(dialog, dialog + 0xeb0)`. The handler builds the rows; the engine
// owns the cursor, the widgets, the key guide and the fade.
//
// So the port does not need any of the game's five. It writes its **own**
// opener - a host function that calls `sub_1f20900` with its own sprite name
// and its own handler - and adds a **seventh row** to the command list that
// runs it. Language keeps `sub_1fdb940`, Environment keeps `sub_1fdb980`, and
// every section but the port's is untouched code.
//
// That leaves two guest patches, both small and both name-checked:
//
//   1. the section-name table gets a seventh slot, so `sub_1f20900` will bind
//      the port's sprites (the overlay movie adds them). The slot holds a
//      pointer to a name the port writes before each open, so one slot serves
//      every section the port has;
//   2. the one `call sub_1fea630` that finalises the System builder is
//      redirected, so the port can append its row before the dialog is made.
//
// Everything else - the movie section, the rows, the values - is the port's
// own.
// ---------------------------------------------------------------------------

namespace {

// Binary Ninja addresses (the eboot at its preferred 0x400000 base), so they
// read the same here as in the database. Every one is gated on the 1.09 hash.
constexpr std::uint64_t kOpenSection = 0x1f20900;         // sub_1f20900
constexpr std::uint64_t kScratchInit = 0x208af20;         // the handler's scratch block
constexpr std::uint64_t kAddSliderRow = 0x1f2ac00;        // a slider row
constexpr std::uint64_t kAddChoiceRow = 0x1f2a100;        // a choice row
constexpr std::uint64_t kBuildOnOff = 0x1f2b3b0;          // its two-entry On/Off list
// The pick-list row - what Language's "Text" uses, opening a sub-menu to
// choose from. Its list is a different, larger container than the On/Off one
// (that one panics above two entries), appended one entry at a time, and its
// entries are keyed by an int32 rather than a byte. That is what a frame cap
// needs.
constexpr std::uint64_t kAddListRow = 0x1f29370;
constexpr std::uint64_t kListAppend = 0x1f1c0c0;
constexpr std::uint64_t kWstrCpy = 0x2d67040;             // builds a wstring in place
// The game's own two-entry Normal / Reversed list, as Controls' Camera X-Axis
// builds it: message 0x9d76 with the value 1, 0x9d77 with 0.
constexpr std::uint64_t kBuildNormalReversed = 0x1f29ee0;
// Appends one 0x48-byte entry to a choice row's list (the On/Off container,
// two entries at most, count at +0x98).
constexpr std::uint64_t kChoiceAppend = 0x1f2c200;
// A list component's redraw: the visible range, through its item provider.
// The providers draw each item from the raw UTF-16 pointer it was built with,
// so text the port owns can be changed and shown again with this alone.
constexpr std::uint64_t kListRedraw = 0x1ed06e0;
constexpr std::uint64_t kListVTable = 0x5799fb0;
// The dialog's row layout: the rows list's count from the record vector, its
// providers, the cursor. sub_1f20900 ends with it, after appending Defaults.
constexpr std::uint64_t kLayoutRows = 0x1f28de0;
constexpr std::uint32_t kDefaultsRowId = 0x9d8a;
// sub_1f20900 will only open a sprite whose name is in this six-entry table,
// and slots [6] and [7] after it are unreferenced zero padding before a
// vtable. The loop over it binds each name's sprite and keeps the one that
// matches the name it was asked for, so the table is really the list of
// names it accepts. Slot [6] therefore holds a pointer to **g_open_name**,
// which every port opener fills in just before its call, and the loop's
// `cmp rdx, 6` becomes a 7 - one byte. Four sections, one slot, and [7] stays
// the zero it was.
constexpr std::uint64_t kSectionNameTable = 0x57398b0;
constexpr std::uint64_t kSectionCountImm = 0x1f20e21;     // the 6 in cmp rdx, 6
char g_open_name[32] = "PCSetting";
// The sections the generated movie adds (engine/menu_assets.h), each a
// clone of ControllSetting with as many widget slots as it has rows.
const char kOurSectionName[] = "PCSetting";
// The second one. A section does not scroll, so more settings than its slots
// means another section - which is how the game does it (Environment,
// Brightness, Network and Language are four sections, not one long list). The
// name table had room: slots [6] and [7] were both unreferenced padding.
//
// This clone keeps only as many slots as it has rows. sub_1f20900 appends a
// "Defaults" row of its own (message 0x9d8a) after the handler returns, and it
// lands in the first slot the handler did not fill - which is how a fourth row
// reading "Defaults" appeared under three settings. With no spare slot it has
// nowhere to draw, which is also why PCSetting, whose six rows fill all six of
// its slots, never showed one.
const char kOurGraphicsName[] = "PCGraphics";
// The mouse: camera, sensitivity, the two axes, the pointer in menus.
const char kOurControlsName[] = "PCControls";
// The key bindings: a page row and seven actions, relabelled in place when the
// page changes (see "The Key Bindings screen" below).
const char kOurKeysName[] = "PCKeys";
// The post-processing the player sees rather than pays for: motion blur, depth
// of field, chromatic aberration, bloom, vignette. Split from PCGraphics when
// the two together came to nine rows, two more than a section's slots.
const char kOurEffectsName[] = "PCEffects";
// The follow camera: field of view, distance, height.
const char kOurCameraName[] = "PCCamera";
// The Steam Deck's frame rate and model detail - on a Deck only (host_steam_deck).
const char kOurDeckName[] = "PCDeck";
// The PC enhancements: the port's additions to the game itself - The Old
// Hunters, the mirror, the rebirth, five players. Each is read once at start,
// which its line help says.
const char kOurEnhanceName[] = "PCEnhance";

// The overlay asset that has to supply that sprite. Opening a section whose
// sprite the movie does not define is not something to find out at runtime, so
// the install refuses when this file is not in the overlay.
const char kOurMovie[] = "/dvdroot_ps4/menu/optionsetting.gfx";
// And the bundle carrying the port's message ids. Without it every row would
// ask for text that is not there.
// The port's message ids are in every language's menu bundle the overlay
// has (engine/menu_assets.cpp); the game reads the one its region and
// language pick, so any of them will do for the check.
const char* const kMessageLanguages[] = {"engus", "enggb", "frafr", "deude", "itait", "spaes", "spaar", "porbr", "porpt", "nldnl",
                                         "rusru", "polpl", "dandk", "finfi", "norno", "swese", "turtr", "jpnjp", "japanese", "korkr",
                                         "zhocn", "zhotw"};

constexpr std::uint64_t kMsgRepository = 0x1ee8cc0;       // call_msg_repository
// The two message groups a row reads, which are also the bundle entry ids:
// group 200 is SP_メニューテキスト.fmg and 201 its line help.
constexpr std::uint32_t kMsgMenuText = 0xc8, kMsgLineHelp = 0xc9;

// The System command list: the row adder, the call that finalises the builder,
// and the alignment padding the redirect lands in.
constexpr std::uint64_t kAddCommandRow = 0x1f4e2a0;       // sub_1f4e2a0
// "Exit Game" is built by a *different* adder - three arguments, no flag - and
// is the one row the port has to move rather than follow. sub_1fb3ad0 adds it
// fifth of six and pops it again unless its third argument is set, so it
// survives in game and not at the title; either way the port's rows are
// appended after it, which left Exit Game in the middle of the list.
constexpr std::uint64_t kAddExitRow = 0x1f4c9a0;          // sub_1f4c9a0
constexpr std::uint64_t kExitGameOpener = 0x200c9f0;      // sub_200c9f0
constexpr std::uint32_t kExitGameRowId = 0x1adb4;
// The command builder's row vector: begin at +0x68, end at +0x70, 0x180 bytes
// a row. sub_1fb3ad0 pops a row by calling slot 1 of its vtable and dropping
// the end pointer, and that is exactly what pop_last_row does here.
constexpr std::size_t kBuilderBegin = 0x68, kBuilderEnd = 0x70;
constexpr std::size_t kBuilderRowStride = 0x180;
constexpr std::uint64_t kSystemFinalize = 0x1fea630;      // sub_1fea630
constexpr std::uint64_t kSystemFinalizeCall = 0x1fb4978;  // the one call to it
// 16 bytes of alignment nops between sub_1fb4be0's ret and sub_1fb4c70. A
// call cannot reach a host address in 5 bytes, so the redirect points here and
// this holds `movabs r11, host; jmp r11`. Both addresses are on one page.
constexpr std::uint64_t kTrampolinePad = 0x1fb4c60;
constexpr std::uint8_t kPadBytes[] = {0x90, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x2e,
                                      0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00};

// Opening a section is a two-step dance through std::function, and the port
// has to build one of each: `sub_1fb4c70(out, ctx, f)` starts the menu flow
// that loads the OptionSetting movie and then calls `f(root, params)`.
constexpr std::uint64_t kOpenSectionFlow = 0x1fb4c70;     // sub_1fb4c70
constexpr std::uint64_t kRowFunctorVTable = 0x573d120;    // Dlg*(Dlg*, Ctx)
constexpr std::uint64_t kOpenerFunctorVTable = 0x5743880; // int64(root, params)

// The port's own message ids. Every option screen owns a thousand-id block -
// 110000 System, 111000 Environment, ... 115000 Language - where N is a row
// and N+20 is the screen title. The port takes 116000, and 110006 for the
// System list row that opens it; both are unused in the shipped bundle. The
// strings themselves live in tools/pc_option_messages.tsv.
constexpr std::uint32_t kSystemRowId = 110006;
constexpr std::uint32_t kSystemGraphicsRowId = 110007;
constexpr std::uint32_t kSystemControlsRowId = 110008;
constexpr std::uint32_t kSystemKeysRowId = 110009;
constexpr std::uint32_t kSystemEffectsRowId = 110010;
constexpr std::uint32_t kSystemCameraRowId = 110011;
constexpr std::uint32_t kSystemDeckRowId = 110012;
constexpr std::uint32_t kSystemEnhanceRowId = 110013;
// PC Enhancements, 124000: three toggles, each for the next start (124000
// was The Old Hunters, a patch of its own since: the setup window's Patches tab).
constexpr std::uint32_t kEnhMirror = 124001, kEnhRebirth = 124002, kEnhFivePlayers = 124003;
// Steam Deck, 123000: its frame rate - the two configurations,
// host_set_deck_profile - and the model detail.
constexpr std::uint32_t kDeckFrameRate = 123000, kDeckModelDetail = 123001;
// PC Controls, 118000; the two pointer rows keep the ids they had in PC
// Settings.
constexpr std::uint32_t kCtlCamera = 118000, kCtlSensitivity = 118001, kCtlInvertX = 118002, kCtlInvertY = 118003;
constexpr std::uint32_t kCtlFov = 118004;  // shown in PC Camera
constexpr std::uint32_t kCamDistance = 121000, kCamHeight = 121001, kCamMouseAutoRotation = 121002;
constexpr std::uint32_t kCamTuning = 121003;  // its two entries: 121030 FPS++ (Kyo), 121031 Original
constexpr std::uint32_t kCtlMouseMenu = 116002, kCtlDrawCursor = 116003;
// Key Bindings, 119000: the page row, seven placeholder rows whose captions
// the port replaces with its own text, and the page names.
constexpr std::uint32_t kKeysPageRow = 119000, kKeysFirstRow = 119001;
// The graphics section's own block. All apply live - the resolution too,
// with the host's keep-or-revert box (host/options.h)
// (engine/graphics_patch.cpp, host/shader_patch.h). Motion blur, depth of
// field and chromatic aberration keep their 117000 ids in PC Effects, whose
// own block is 120000.
constexpr std::uint32_t kGfxSsao = 117000, kGfxAntiAlias = 117001, kGfxDof = 117002;
constexpr std::uint32_t kGfxChromaticAberration = 117003;
constexpr std::uint32_t kGfxResolution = 117004;
constexpr std::uint32_t kGfxMotionBlur = 117005;
constexpr std::uint32_t kGfxShadowDistance = 117006, kGfxAoStrength = 117007, kGfxFog = 117008;
// HLSL Shaders (the Shaders setting: HLSL or Lifted), read at the next start.
constexpr std::uint32_t kGfxShaders = 117009;
constexpr std::uint32_t kFxBloom = 120000, kFxVignette = 120001, kFxSaturation = 120002;
constexpr std::uint32_t kRowFrameCap = 116005;
// The section title, StaticText_116020 in the generated movie.
constexpr std::uint32_t kSectionTitleId = 116020;

std::uint64_t g_slide = 0;
bool g_installed = false;

std::uint64_t guest(std::uint64_t bn) { return g_slide + (bn - kPreferredGuestSlide); }
void* guest_fn(std::uint64_t bn) { return reinterpret_cast<void*>(static_cast<std::uintptr_t>(guest(bn))); }

// A std::function the guest can call, built in place. The engine's is MSVC's
// shape: 0x20 bytes of inline storage holding the callable, then a pointer to
// it at +0x20. The callable is `_Func_impl_no_alloc<fn-pointer>` - its vtable
// at +0x00 and the wrapped function pointer at +0x08 - which is why storing a
// host thunk there works exactly as the game storing `sub_1fdb960` does.
// Both consumers copy it before keeping it, so a stack one is safe to pass.
struct GuestFunction {
    alignas(16) std::uint8_t buf[0x28]{};

    GuestFunction(std::uint64_t vtable_bn, void* fn) {
        const std::uint64_t vt = guest(vtable_bn);
        void* self = &buf[0];
        std::memcpy(&buf[0x00], &vt, sizeof(vt));
        std::memcpy(&buf[0x08], &fn, sizeof(fn));
        std::memcpy(&buf[0x20], &self, sizeof(self));
    }
};

// A caption pair: two objects 0x40 apart, the label in the first and the line
// help in the second, which is the shape every row builder reads. Within each,
// +0x00 is the raw UTF-16 pointer the row draws, +0x08 a wstring the command
// row copies, +0x38 a flag.
//
// The port used to build one of these from an arbitrary message id and then
// store a pointer to its own string literal over the text. It drew, but the
// flags beside it were the placeholder's, and a section title - which the
// movie carries as a `StaticText_<id>` placement, the instance name being the
// id - could not be done that way at all. So the port has message ids of its
// own now (tools/pc_option_messages.tsv, written into the overlay's
// menu.msgbnd.dcx by engine/menu_assets.cpp) and asks for them exactly as the game
// does.
struct Captions {
    alignas(16) std::uint8_t buf[0x80]{};

    explicit Captions(std::uint32_t id) {
        hle_call_guest<std::int64_t>(guest_fn(kMsgRepository), &buf[0], kMsgMenuText, id);
        hle_call_guest<std::int64_t>(guest_fn(kMsgRepository), &buf[0x40], kMsgLineHelp, id);
    }

    // The repository's text is copied into a wstring beside the pointer, and a
    // caption long enough to leave the SSO buffer put it on the heap. Every
    // builder that makes one of these frees it the same way, so this does too:
    // one leak per menu open is still a leak.
    ~Captions() {
        free_wstring(&buf[0x08]);
        free_wstring(&buf[0x48]);
    }

    Captions(const Captions&) = delete;
    Captions& operator=(const Captions&) = delete;

    // MSVC's _String_val: the union at +0x00 is either eight wchars or the
    // heap pointer, the length is at +0x10, the capacity at +0x18 and the DL
    // allocator at +0x20. Capacity below 8 is the small-string buffer.
    static void free_wstring(void* ws) {
        auto* p = static_cast<std::uint8_t*>(ws);
        std::uint64_t cap = 0, ptr = 0, alloc = 0;
        std::memcpy(&cap, p + 0x18, sizeof(cap));
        if (cap < 8) {
            return;
        }
        std::memcpy(&ptr, p, sizeof(ptr));
        std::memcpy(&alloc, p + 0x20, sizeof(alloc));
        if (!ptr || !alloc) {
            return;
        }
        std::uint64_t vtable = 0;
        std::memcpy(&vtable, reinterpret_cast<void*>(static_cast<std::uintptr_t>(alloc)),
                    sizeof(vtable));
        std::uint64_t fn = 0;
        std::memcpy(&fn, reinterpret_cast<void*>(static_cast<std::uintptr_t>(vtable + 0x70)),
                    sizeof(fn));
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(fn)),
                                     alloc, ptr);
    }
};

// The repository's own text for a message id: the caption object's +0x00,
// which the repository owns and which outlives anything the port holds.
const char16_t* message_text(std::uint32_t id) {
    Captions c(id);
    const char16_t* t = nullptr;
    std::memcpy(&t, &c.buf[0], sizeof(t));
    return t;
}

// A list entry, the 0x48 bytes every list the rows take is made of: the
// value at +0x00 (a byte in a choice row's list, an int32 in a pick list's),
// the raw UTF-16 pointer that is drawn at +0x08, a wstring copy of it at
// +0x10 and a flag at +0x40. The containers deep-copy an entry, so the
// caller's copy, and later the container's, own a heap buffer for any text
// of eight characters or more, and free it the way the game's own handlers
// do: the buffer pointer at +0x18, its capacity at +0x30, the allocator at
// +0x38 - which is Captions::free_wstring's shape from +0x18.
void make_entry(std::uint8_t (&entry)[0x48], std::int32_t value, const char16_t* text) {
    // A message the bundle does not have (a player's own older menu.msgbnd in
    // paths.mods wins over the one made at start) gives no text, and the
    // game's wstring copy would read through null: an empty entry instead.
    if (!text) text = u"";
    std::memset(entry, 0, sizeof(entry));
    std::memcpy(&entry[0], &value, sizeof(value));
    std::memcpy(&entry[8], &text, sizeof(text));
    hle_call_guest<std::int64_t>(guest_fn(kWstrCpy), &entry[0x10], text);
    entry[0x40] = 1;
}

void free_entries(std::uint8_t* first, std::uint64_t count) {
    for (std::uint64_t k = 0; k < count; ++k) {
        Captions::free_wstring(first + k * 0x48 + 0x18);
    }
}

// Every builder takes the value its row's **Defaults** restores. The widgets
// copy it when they are built - a choice row keeps the byte at +0x368 beside
// its value pointer, a pick list the int32 at +0x3d4 - and the game's own
// Defaults row (sub_1f24850: "restore defaults?", then sub_1f1a2d0 calls every
// widget's slot +0x20) writes it back through the value pointer. So each row
// is built with what its setting ships with (host_opt_default_*), and the poll
// sees the change like any other.

// The game's slider: a byte from 0 to 10 in steps of 1 (sub_1f2ac00 builds it
// with those bounds), as Controls' Camera Sensitivity uses it.
void add_slider_row(void* dialog, std::uint8_t* value, std::uint32_t id, std::uint8_t def) {
    Captions c(id);
    hle_call_guest<std::int64_t>(guest_fn(kAddSliderRow), dialog, &c.buf[0], value, &def);
}

// An On/Off row. sub_1f2b3b0 builds the two-entry list from the game's own
// "on" and "off" messages, so the port does not have to construct a choice
// entry - the one structure here it would otherwise have to guess.
void add_toggle_row(void* dialog, void* value, std::uint32_t id, std::uint8_t def) {
    Captions c(id);
    alignas(16) std::uint8_t list[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kBuildOnOff), &list);
    hle_call_guest<std::int64_t>(guest_fn(kAddChoiceRow), dialog, &c.buf[0], value, &list, &def);
    std::uint64_t n = 0;
    std::memcpy(&n, &list[0x98], sizeof(n));
    free_entries(list, n);
}

// Normal / Reversed, from the game's own list: 1 is Normal, 0 Reversed.
void add_axis_row(void* dialog, std::uint8_t* value, std::uint32_t id, std::uint8_t def) {
    Captions c(id);
    alignas(16) std::uint8_t list[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kBuildNormalReversed), &list);
    hle_call_guest<std::int64_t>(guest_fn(kAddChoiceRow), dialog, &c.buf[0], value, &list, &def);
    std::uint64_t n = 0;
    std::memcpy(&n, &list[0x98], sizeof(n));
    free_entries(list, n);
}

// A row that shows text the port owns and changes: a choice row whose list is
// one entry, drawn from `text`, with the caption and the line help pointed at
// the port's own buffers too. With one entry Left and Right have nothing to
// step to; what the row does is up to the port (a key binding: Circle starts
// a capture). The captions come from `id` for their flags, and the three
// pointers are replaced - every provider draws the raw pointer when there is
// one (sub_1f263b0 for a choice list's items, the dialog's for its rows).
void add_text_row(void* dialog, std::uint8_t* value, std::uint32_t id, const char16_t* label, const char16_t* help,
                  const char16_t* text) {
    Captions c(id);
    std::memcpy(&c.buf[0x00], &label, sizeof(label));
    std::memcpy(&c.buf[0x40], &help, sizeof(help));
    alignas(16) std::uint8_t list[0x100]{};
    alignas(16) std::uint8_t entry[0x48];
    make_entry(entry, 0, text);
    hle_call_guest<std::int64_t>(guest_fn(kChoiceAppend), &list, &entry);
    free_entries(entry, 1);
    *value = 0;
    std::uint8_t def = 0;  // its only entry; the Key Bindings screen has no Defaults row
    hle_call_guest<std::int64_t>(guest_fn(kAddChoiceRow), dialog, &c.buf[0], value, &list, &def);
    free_entries(list, 1);
}

// One choice in a pick-list: the value the row stores and the message id whose
// text it shows.
struct Choice {
    std::int32_t value;
    std::uint32_t id;
};

// A pick-list row. Its entries are the 0x48-byte shape above with an int32
// value, which is the same shape the game builds for Language.
void add_list_row(void* dialog, void* value, std::uint32_t id, const Choice* choices, int count, std::int32_t def) {
    Captions c(id);
    // The container is an inline array plus a count at +0x908; zeroing it is
    // the whole of its initialisation.
    alignas(16) std::uint8_t list[0x1000]{};
    for (int i = 0; i < count; ++i) {
        alignas(16) std::uint8_t entry[0x48];
        make_entry(entry, choices[i].value, message_text(choices[i].id));
        hle_call_guest<std::int64_t>(guest_fn(kListAppend), &list, &entry);
        free_entries(entry, 1);
    }
    hle_call_guest<std::int64_t>(guest_fn(kAddListRow), dialog, &c.buf[0], value, &list, &def);
    free_entries(list, static_cast<std::uint64_t>(count));
}

// A left/right row of two entries the port names: the On/Off rows' container
// (two entries at most), filled with its own entries - a byte value and a
// message each - instead of the game's On and Off.
void add_pair_row(void* dialog, std::uint8_t* value, std::uint32_t id, const Choice (&choices)[2], std::uint8_t def) {
    Captions c(id);
    alignas(16) std::uint8_t list[0x100]{};
    for (const Choice& ch : choices) {
        alignas(16) std::uint8_t entry[0x48];
        make_entry(entry, ch.value, message_text(ch.id));
        hle_call_guest<std::int64_t>(guest_fn(kChoiceAppend), &list, &entry);
        free_entries(entry, 1);
    }
    hle_call_guest<std::int64_t>(guest_fn(kAddChoiceRow), dialog, &c.buf[0], value, &list, &def);
    std::uint64_t n = 0;
    std::memcpy(&n, &list[0x98], sizeof(n));
    free_entries(list, n);
}

// The rows' values live **here**, not in the dialog's own settings block. A
// row writes a byte through the pointer it is given and does not care where it
// points, and the game applies its own block on exit - so pointing these at
// the game's would make the port's settings change the game's. They are the
// port's settings; they belong to the port.
// One entry per setting the rows expose. The byte is what a row writes; `key`
// is the same name the F10 screen and bbhost-options.toml use, so the two front
// ends drive one set of settings and one file. F10 is scaffolding - this is
// where these belong.
struct Toggle {
    const char* key;
    std::uint32_t id;   // its label and line help, in the port's own block
    std::uint8_t value;
    std::uint8_t last;  // what was pushed, so a poll acts only on a change
    // The row's On is the setting's first value rather than its "on" - HLSL
    // Shaders' On is the Shaders setting's HLSL, whose second value, Lifted,
    // is what host_opt_get reads as on.
    bool invert = false;
};
// PC Settings. The two pointer rows it had moved to PC Controls, beside the
// rest of the mouse.
Toggle g_toggles[] = {
    {"vsync", 116000, 1, 0xff},
    {"window_mode", 116001, 0, 0xff},
    {"skip_logos", 116004, 0, 0xff},
    // The frames per second the presenter counts (host/window.cpp).
    {"fps_counter", 116007, 0, 0xff},
};
constexpr int kToggleCount = static_cast<int>(sizeof(g_toggles) / sizeof(g_toggles[0]));

// A 0..10 slider row's byte, and what was pushed.
struct Slider {
    const char* key;  // a setting whose values are its eleven steps
    std::uint32_t id;
    std::uint8_t value;
    std::uint8_t last;
};

// PC Graphics: the scene - anti-aliasing, ambient occlusion and its strength
// (engine/graphics_patch.h), the shaders' form (HLSL Shaders, read at the next
// start: host/translation_cache.h), how far shadows reach, how thick the fog
// is, and the resolution.
Toggle g_graphics[] = {
    {"anti_alias", kGfxAntiAlias, 1, 0xff},
    {"ssao", kGfxSsao, 1, 0xff},
    {"shaders", kGfxShaders, 1, 0xff, true},
};
constexpr int kGraphicsCount = static_cast<int>(sizeof(g_graphics) / sizeof(g_graphics[0]));
// 5 is the area's own occlusion, 0 none, 10 twice as dark.
Slider g_gfx_ao{"ao_strength", kGfxAoStrength, 5, 0xff};
// 0 is the area's own shadow distance, each step 20% farther.
Slider g_gfx_shadow{"shadow_distance", kGfxShadowDistance, 0, 0xff};
// 10 is the area's own fog, 0 none.
Slider g_gfx_fog{"fog", kGfxFog, 10, 0xff};
// Plus the three sliders and the resolution pick list below them.
constexpr int kGraphicsRows = kGraphicsCount + 4;
// The PC Graphics dialog as last opened, for the questions asked over it
// (confirm_push, below). Menu thread.
std::uint8_t* g_gfx_dialog = nullptr;           // PC Graphics while it may be open (menu thread)
std::uint64_t g_gfx_vtable = 0;
std::uint64_t g_gfx_base_depth = 0;  // its popup stack with nothing open on it


// PC Effects: motion blur and depth of field (shader patches,
// host/shader_patch.h), chromatic aberration and vignette, and two sliders -
// bloom (10 the area's own glare, 0 none) and saturation (5 the area's own,
// 0 black and white).
Toggle g_effects[] = {
    {"motion_blur", kGfxMotionBlur, 1, 0xff},
    {"depth_of_field", kGfxDof, 1, 0xff},
    {"chromatic_aberration", kGfxChromaticAberration, 1, 0xff},
    {"vignette", kFxVignette, 1, 0xff},
};
constexpr int kEffectsCount = static_cast<int>(sizeof(g_effects) / sizeof(g_effects[0]));
Slider g_fx_bloom{"bloom", kFxBloom, 10, 0xff};
Slider g_fx_saturation{"saturation", kFxSaturation, 5, 0xff};
constexpr int kEffectsRows = kEffectsCount + 2;

// The frame rate (engine/frame_rate.h), as a pick-list. The row stores the
// chosen entry's int32, so the values are the rates themselves and 0 is
// uncapped. Five entries of the widget's six slots (the old Frame cap's
// 30/60/90/120/144/Off had all six; 116034 was its 144).
const Choice kFrameCaps[] = {
    {30, 116030}, {60, 116031}, {90, 116032}, {120, 116033}, {0, 116035},
};
std::int32_t g_frame_cap = 30;

// The render resolution, as the index into the setting's own list of names -
// the row stores an int32 and the host owns what each one means, so the two
// front ends cannot disagree about the order. The setting's order (host/
// options.cpp): by size within each shape, 16:9, 16:10, 4:3, 21:9, 32:9 (the
// pick list scrolls past six). Each entry's text is its size's message, which
// keeps its id whatever its place.
const Choice kResolutions[] = {
    {0, 117038},  {1, 117039},  {2, 117010},  {3, 117040},  {4, 117011},  {5, 117012},   // 640x360 .. 1920x1080
    {6, 117013},  {7, 117041},  {8, 117014},  {9, 117015},                              // 2560x1440 .. 3840x2160
    {10, 117036}, {11, 117037}, {12, 117035}, {13, 117042}, {14, 117043},               // 16:10
    {15, 117044}, {16, 117045},                                                         // 4:3
    {17, 117030}, {18, 117031}, {19, 117032},                                           // 21:9
    {20, 117033}, {21, 117034},                                                         // 32:9
};
std::int32_t g_resolution = 5;
std::int32_t g_last_resolution = -1;
std::int32_t g_last_frame_cap = -1;
bool g_seeded = false;

// PC Controls: the mouse, which DS3 PC gives a screen of its own. Camera
// Sensitivity is the game's own slider (0..10, as Controls' Camera
// Sensitivity), and the two axes the game's own Normal / Reversed list, whose
// entries store 1 for Normal and 0 for Reversed.
Toggle g_ctl_camera{"mouse_camera", kCtlCamera, 1, 0xff};
std::uint8_t g_ctl_sens = 5, g_ctl_sens_last = 0xff;
const char* const kAxisKeys[2] = {"mouse_invert_x", "mouse_invert_y"};
const std::uint32_t kAxisIds[2] = {kCtlInvertX, kCtlInvertY};
std::uint8_t g_ctl_axis[2] = {1, 1}, g_ctl_axis_last[2] = {0xff, 0xff};
Toggle g_ctl_pointer[] = {
    {"mouse_menu", kCtlMouseMenu, 1, 0xff},
    {"draw_cursor", kCtlDrawCursor, 1, 0xff},
};
constexpr int kControlsRows = 4 + static_cast<int>(sizeof(g_ctl_pointer) / sizeof(g_ctl_pointer[0]));

// PC Camera: the follow camera's field of view (0 the game's own, 5% wider a
// step), its distance and the height it looks at (5 the game's own, 10% a
// step) - LockCamParam's fields, engine/camera.h - and whether it turns by
// itself as the character moves while the mouse turns it (engine/mouse_camera.h).
Slider g_cam_fov{"fov", kCtlFov, 0, 0xff};
Slider g_cam_distance{"camera_distance", kCamDistance, 5, 0xff};
Slider g_cam_height{"camera_height", kCamHeight, 5, 0xff};
Toggle g_cam_mouse_auto_rotation{"mouse_auto_rotation", kCamMouseAutoRotation, 0, 0xff};
// And its tuning above 30 fps (engine/frame_rate.cpp): a left/right choice of
// two the port names, as the setting's index - 0 FPS++, 1 Original - read
// when the game starts, so a change shows at once and applies then.
const Choice kCameraTunings[2] = {{0, 121030}, {1, 121031}};
std::uint8_t g_cam_tuning = 0, g_cam_tuning_last = 0xff;
constexpr int kCameraRows = 4;

// Steam Deck (on one only, host_steam_deck): the frame rate as a left/right
// choice of the two configurations - 0 is 30 fps at the screen's 1280x800, 1
// is 60 fps rendering at 960x600 with Model detail Low - which the host sets
// as one (host_set_deck_profile), and the model detail as a pick list, which
// no other section has. The pick list last, as everywhere.
const Choice kDeckRates[2] = {{0, 123030}, {1, 123031}};
const Choice kModelDetails[] = {{0, 123040}, {1, 123041}, {2, 123042}, {3, 123043}};
std::uint8_t g_deck_rate = 0, g_deck_rate_last = 0xff;
std::int32_t g_deck_model = 1, g_deck_model_last = -1;
std::uint8_t* g_deck_dialog = nullptr;  // the section while it may be open (menu thread)
std::uint64_t g_deck_vtable = 0;
std::atomic<bool> g_deck_redraw{false};  // the model detail row shows a value the frame rate set
constexpr int kDeckRows = 2;

// PC Enhancements: three toggles, read at the next start (main.cpp,
// config_set_enhancements), so a change here shows at once and applies then.
Toggle g_enhance[] = {
    {"change_appearance", kEnhMirror, 1, 0xff},
    {"rebirth", kEnhRebirth, 1, 0xff},
    {"five_players", kEnhFivePlayers, 1, 0xff},
};
constexpr int kEnhanceRows = static_cast<int>(sizeof(g_enhance) / sizeof(g_enhance[0]));

void seed_camera_tuning() {
    const int t = host_opt_index("camera_tuning");
    g_cam_tuning = g_cam_tuning_last = static_cast<std::uint8_t>(t == 1 ? 1 : 0);
}

void seed_deck() {
    g_deck_rate = g_deck_rate_last = static_cast<std::uint8_t>(host_deck_profile());
    const int m = host_opt_index("model_detail");
    g_deck_model = g_deck_model_last = m < 0 ? 1 : m;
}

// Every row opens showing what the setting is now - the F10 screen may have
// changed it since - so each section seeds its values when it opens, and the
// poll then pushes only what the player changes.
void seed(Toggle& t) {
    t.value = host_opt_get(t.key) != t.invert ? 1 : 0;
    t.last = t.value;
}

// What the row's Defaults restores: the setting as it ships.
std::uint8_t default_of(const Toggle& t) { return host_opt_default_get(t.key) != t.invert ? 1 : 0; }

void seed(Slider& s) {
    const int i = host_opt_index(s.key);
    s.value = static_cast<std::uint8_t>(i < 0 ? 0 : i);
    s.last = s.value;
}
std::uint8_t default_of(const Slider& s) {
    const int i = host_opt_default_index(s.key);
    return static_cast<std::uint8_t>(i < 0 ? 0 : i);
}
// A slider moved: its step is the setting's index.
void push(Slider& s) {
    if (s.value == s.last) return;
    s.last = s.value;
    host_opt_set_index(s.key, s.value);
    host_log("pc-options: %s %u from the System menu", s.key, s.value);
}
void push(Toggle& t) {
    if (t.value == t.last) return;
    t.last = t.value;
    host_opt_set(t.key, (t.value != 0) != t.invert);
    host_log("pc-options: %s %s from the System menu", t.key, t.value ? "on" : "off");
}

void seed_controls() {
    seed(g_ctl_camera);
    const int sens = host_opt_index("mouse_sens");
    g_ctl_sens = static_cast<std::uint8_t>(sens < 0 ? 5 : sens);
    g_ctl_sens_last = g_ctl_sens;
    for (int i = 0; i < 2; ++i) {
        g_ctl_axis[i] = host_opt_get(kAxisKeys[i]) ? 0 : 1;
        g_ctl_axis_last[i] = g_ctl_axis[i];
    }
    for (auto& t : g_ctl_pointer) seed(t);
}

// How many widget slots each section sprite has. ControllSetting places
// Item_0_0..Item_5_0 on char 97, so six, and a row the port builds past that
// has nowhere to draw - which is what happened when the donor was Environment
// and its five slots took six rows. **Option sections do not scroll**: the
// movie gives them a fixed set of slots and no scrollbar. The generator
// (engine/menu_assets.cpp) drops the slots a section does not use -
// sub_1f20900 appends a "Defaults" row that lands in the first free one - and
// adds the two more the Key Bindings screen needs, which is as many as fit
// above the key guide. The Defaults row has a slot of its own besides these:
// the donor's label slot (char 100), right after the section's last row.
constexpr int kSectionSlots = 6, kKeysSlots = 8;
// PC Graphics has one slot more (its seventh row, HLSL Shaders): the generator
// adds it as it adds Key Bindings' two, and moves the Defaults label below it.
constexpr int kGraphicsSlots = 7;

// The port's rows: its toggles plus the frame-cap pick list, and in the
// graphics section its toggles plus the resolution list.
constexpr int kRowCount = kToggleCount + 1;
static_assert(kRowCount <= kSectionSlots && kGraphicsRows <= kGraphicsSlots && kControlsRows <= kSectionSlots &&
                  kCameraRows <= kSectionSlots && kDeckRows <= kSectionSlots && kEnhanceRows <= kSectionSlots &&
                  kEffectsRows <= kSectionSlots && 1 + kBindPerPage <= kKeysSlots,
              "more rows than the section sprite has slots: the extra ones have nowhere to draw. Add slots in engine/menu_assets.cpp (build_option_movie), or split into a second section.");

// The dialog's row vector, found by reading the row builder: sub_1f2a100 ends
// with sub_1f2bf60(dialog + 0xe50, &row), and that is a push_back of 0x90-byte
// records - begin at +0xe58, end at +0xe60, each holding the row's caption
// pointer at +0x00, its line help's at +0x40 and its widget at +0x80. Reading
// it back is how the port knows what the engine actually took.
constexpr std::size_t kItemsBegin = 0x0e58, kItemsEnd = 0x0e60, kItemStride = 0x90, kItemWidget = 0x80;
// The dialog's own list component, whose items are the rows (their captions
// and cursor), and within a choice widget (sub_1f2a100, 0x410 bytes) the list
// that draws its value and the pointer the value is written through.
constexpr std::size_t kDialogRows = 0x0a70, kChoiceList = 0xa0, kChoiceValue = 0x360;

int dialog_row_count(void* dialog) {
    std::uint64_t begin = 0, end = 0;
    std::memcpy(&begin, static_cast<std::uint8_t*>(dialog) + kItemsBegin, sizeof(begin));
    std::memcpy(&end, static_cast<std::uint8_t*>(dialog) + kItemsEnd, sizeof(end));
    if (!begin || end < begin || (end - begin) % kItemStride != 0) {
        return -1;
    }
    return static_cast<int>((end - begin) / kItemStride);
}

std::uint8_t* dialog_widget(void* dialog, int row) {
    std::uint64_t begin = 0, w = 0;
    std::memcpy(&begin, static_cast<std::uint8_t*>(dialog) + kItemsBegin, sizeof(begin));
    std::memcpy(&w, reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(begin)) + row * kItemStride + kItemWidget,
                sizeof(w));
    return reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(w));
}

void log_rows(const char* name, void* dialog, int built, int slots) {
    // Read back from the engine's own vector, so a row built and then lost is
    // loud rather than silent. sub_1f20900 appends its "Defaults" row after
    // the handler returns, so it is not counted here.
    const int took = dialog_row_count(dialog);
    host_log("pc-options: %s: %d rows built, %d in the dialog, %d slots%s", name, built, took, slots,
             took != built || took > slots ? "  <- not every row will draw" : "");
}

// The port's section handler. sub_1f20900 has already made the dialog and
// bound it to the PCSetting sprite; `params` is the block the row builders
// write through. One scratch block, then one builder call per row.
GUEST_ABI void pc_section_handler(void* dialog, void* params) {
    host_log("pc-options: PCSetting opened, dialog=%p params=%p", dialog, params);
    alignas(16) std::uint8_t scratch[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kScratchInit), &scratch);
    for (auto& t : g_toggles) seed(t);
    g_frame_cap = host_opt_frame_cap();
    g_last_frame_cap = g_frame_cap;
    // The toggles first and the pick list **last**. The engine's pick list is
    // translucent and opens directly under the row that owns it, so with the
    // list on row 0 its four entries sat over the five rows below and their
    // values read straight through the panel. On the last row it opens over
    // the empty space under the section instead. Language gets away with the
    // list first because it only has one row behind it.
    for (auto& t : g_toggles) {
        add_toggle_row(dialog, &t.value, t.id, default_of(t));
    }
    add_list_row(dialog, &g_frame_cap, kRowFrameCap, kFrameCaps,
                 static_cast<int>(sizeof(kFrameCaps) / sizeof(kFrameCaps[0])), host_opt_default_frame_cap());
    log_rows("PCSetting", dialog, kRowCount, kRowCount);
}

// The graphics section's handler. Same shape as the one above: a scratch
// block, then one row builder per row.
GUEST_ABI void pc_graphics_handler(void* dialog, void* params) {
    host_log("pc-options: PCGraphics opened, dialog=%p params=%p", dialog, params);
    g_gfx_dialog = static_cast<std::uint8_t*>(dialog);
    std::memcpy(&g_gfx_vtable, dialog, sizeof(g_gfx_vtable));
    std::memcpy(&g_gfx_base_depth, g_gfx_dialog + 0x40, sizeof(g_gfx_base_depth));
    alignas(16) std::uint8_t scratch[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kScratchInit), &scratch);
    for (auto& t : g_graphics) seed(t);
    seed(g_gfx_ao);
    seed(g_gfx_shadow);
    seed(g_gfx_fog);
    g_resolution = host_opt_resolution_index();
    g_last_resolution = g_resolution;
    for (auto& t : g_graphics) {
        add_toggle_row(dialog, &t.value, t.id, default_of(t));
    }
    add_slider_row(dialog, &g_gfx_ao.value, g_gfx_ao.id, default_of(g_gfx_ao));
    add_slider_row(dialog, &g_gfx_shadow.value, g_gfx_shadow.id, default_of(g_gfx_shadow));
    add_slider_row(dialog, &g_gfx_fog.value, g_gfx_fog.id, default_of(g_gfx_fog));
    // The pick list **last**, for the same reason the frame cap is last in the
    // other section: it is translucent and opens directly under the row that
    // owns it, so anywhere else its entries sit over the rows below and their
    // values read through the panel.
    add_list_row(dialog, &g_resolution, kGfxResolution, kResolutions,
                 static_cast<int>(sizeof(kResolutions) / sizeof(kResolutions[0])), host_opt_default_resolution_index());
    log_rows("PCGraphics", dialog, kGraphicsRows, kGraphicsRows);
}

// PC Effects: four toggles and two sliders. No pick list, so nothing opens
// over the rows.
GUEST_ABI void pc_effects_handler(void* dialog, void* params) {
    host_log("pc-options: PCEffects opened, dialog=%p params=%p", dialog, params);
    alignas(16) std::uint8_t scratch[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kScratchInit), &scratch);
    for (auto& t : g_effects) seed(t);
    seed(g_fx_bloom);
    seed(g_fx_saturation);
    for (auto& t : g_effects) {
        add_toggle_row(dialog, &t.value, t.id, default_of(t));
    }
    add_slider_row(dialog, &g_fx_bloom.value, g_fx_bloom.id, default_of(g_fx_bloom));
    add_slider_row(dialog, &g_fx_saturation.value, g_fx_saturation.id, default_of(g_fx_saturation));
    log_rows("PCEffects", dialog, kEffectsRows, kEffectsRows);
}

// PC Controls: the mouse camera, how fast it turns and which way, and the
// pointer in menus.
GUEST_ABI void pc_controls_handler(void* dialog, void* params) {
    host_log("pc-options: PCControls opened, dialog=%p params=%p", dialog, params);
    alignas(16) std::uint8_t scratch[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kScratchInit), &scratch);
    seed_controls();
    add_toggle_row(dialog, &g_ctl_camera.value, g_ctl_camera.id, default_of(g_ctl_camera));
    const int sens = host_opt_default_index("mouse_sens");
    add_slider_row(dialog, &g_ctl_sens, kCtlSensitivity, static_cast<std::uint8_t>(sens < 0 ? 5 : sens));
    for (int i = 0; i < 2; ++i) {
        // The game's list: 1 is Normal, 0 Reversed, and the setting is "invert".
        add_axis_row(dialog, &g_ctl_axis[i], kAxisIds[i], host_opt_default_get(kAxisKeys[i]) ? 0 : 1);
    }
    for (auto& t : g_ctl_pointer) add_toggle_row(dialog, &t.value, t.id, default_of(t));
    log_rows("PCControls", dialog, kControlsRows, kControlsRows);
}

// PC Camera: three sliders, the auto-rotation toggle and the tuning's pair.
GUEST_ABI void pc_camera_handler(void* dialog, void* params) {
    host_log("pc-options: PCCamera opened, dialog=%p params=%p", dialog, params);
    alignas(16) std::uint8_t scratch[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kScratchInit), &scratch);
    for (Slider* c : {&g_cam_fov, &g_cam_distance, &g_cam_height}) {
        seed(*c);
        add_slider_row(dialog, &c->value, c->id, default_of(*c));
    }
    seed(g_cam_mouse_auto_rotation);
    add_toggle_row(dialog, &g_cam_mouse_auto_rotation.value, g_cam_mouse_auto_rotation.id,
                   default_of(g_cam_mouse_auto_rotation));
    log_rows("PCCamera", dialog, kCameraRows, kCameraRows);
}

// PC Enhancements: four toggles, nothing that opens over the rows.
GUEST_ABI void pc_enhance_handler(void* dialog, void* params) {
    host_log("pc-options: PCEnhance opened, dialog=%p params=%p", dialog, params);
    alignas(16) std::uint8_t scratch[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kScratchInit), &scratch);
    for (auto& t : g_enhance) seed(t);
    for (auto& t : g_enhance) add_toggle_row(dialog, &t.value, t.id, default_of(t));
    log_rows("PCEnhance", dialog, kEnhanceRows, kEnhanceRows);
}

// Steam Deck: the frame rate's pair, then the model detail's pick list.
GUEST_ABI void pc_deck_handler(void* dialog, void* params) {
    host_log("pc-options: PCDeck opened, dialog=%p params=%p", dialog, params);
    g_deck_dialog = static_cast<std::uint8_t*>(dialog);
    std::memcpy(&g_deck_vtable, dialog, sizeof(g_deck_vtable));
    alignas(16) std::uint8_t scratch[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kScratchInit), &scratch);
    seed_deck();
    add_pair_row(dialog, &g_deck_rate, kDeckFrameRate, kDeckRates, static_cast<std::uint8_t>(host_deck_default_profile()));
    const int model_default = host_opt_default_index("model_detail");
    add_list_row(dialog, &g_deck_model, kDeckModelDetail, kModelDetails, static_cast<int>(sizeof(kModelDetails) / sizeof(kModelDetails[0])),
                 model_default < 0 ? 1 : model_default);
    log_rows("PCDeck", dialog, kDeckRows, kDeckRows);
}

// ---------------------------------------------------------------------------
// The Key Bindings screen
//
// DS3 PC's key configuration is a scrolling table, a tab per group of actions,
// and "press the key" to change one. Bloodborne's sections do not scroll and
// their rows only ever write a value - but every list these rows use draws its
// items from raw UTF-16 pointers, through a provider that runs again whenever
// the list is redrawn (sub_1ed06e0; sub_1f263b0 for a choice row's value). So
// the port points a row's caption, line help and value at buffers of its own,
// and changing what the screen says is writing those buffers and redrawing.
// That gives DS3's shape in the game's own widgets:
//
//   - row 0 names the page - DS3's tabs. It is a text row like the others,
//     and Left / Right (or Circle, or a click) step it: the game's pick list
//     would have opened translucent over the seven rows below it;
//   - rows 1-7 are the page's actions, each a one-entry choice row showing its
//     binding ("X  /  Right Button"). Choosing another page rewrites all seven
//     in place;
//   - Circle on an action (Enter, or a click) starts a capture
//     (host/bindings.h): the row reads "Press a key", the game sees an idle
//     pad, and the next key or mouse button becomes the binding. Escape
//     cancels, Delete unbinds, and a key another action had is taken from it.
//   - Delete on a selected action clears it, key and mouse button both, so an
//     action a capture gave a second binding can be brought back to one.
//
// All of it runs in the menu's own list update (option_menu_list_update), on
// the thread that owns these objects, and only while the objects are provably
// this screen's: each action row's value pointer is one of the port's.
// ---------------------------------------------------------------------------
int g_keys_page = 0;  // kept across opens, as DS3 keeps its tab
// Row 0 is the page, rows 1..7 the page's actions: the arrays below are
// indexed by row, and an action row r is action page * 7 + (r - 1).
constexpr int kKeysRows = 1 + kBindPerPage;
std::uint8_t g_keys_value[kKeysRows];
char16_t g_keys_label[kKeysRows][48];
char16_t g_keys_help[kKeysRows][160];
char16_t g_keys_text[kKeysRows][64];
struct KeysScreen {
    std::uint8_t* dialog = nullptr;
    std::uint8_t* rows = nullptr;          // the dialog's own list
    std::uint8_t* widget[kKeysRows] = {};  // each row's choice widget
    int page = -1;                         // the page the rows show
    std::uint32_t pad_was = 0;             // for Circle's and the arrows' edges
};
KeysScreen g_keys;
// A clear made on the menu thread, for the poll to save on the window thread
// with every other settings write.
std::atomic<bool> g_keys_save{false};

// Decode UTF-8; never split a surrogate pair at the fixed buffer boundary.
template <std::size_t N>
void to_utf16(char16_t (&out)[N], const char* s) {
    std::size_t i = 0;
    while (s && *s) {
        auto cp = utf8_next(s);
        if (i + (cp > 0xffff ? 2 : 1) >= N) break;
        if (cp > 0xffff) {
            cp -= 0x10000;
            out[i++] = static_cast<char16_t>(0xd800 + (cp >> 10));
            out[i++] = static_cast<char16_t>(0xdc00 + (cp & 0x3ff));
        } else out[i++] = static_cast<char16_t>(cp);
    }
    out[i] = 0;
}

// The page names, as the tabs read. Plain ASCII: the line help and values
// are drawn as HTML text, where an ampersand is not a character.
const char* const kKeyPageNames[kBindPages] = {"移動", "戦闘", "アイテムとジェスチャー", "カメラとメニュー",
                                               "その他"};

// The actions listed: the Debug Menu key only when the menu is there (the
// Debug Menu plugin, engine/debug_menu.h); it is the last.
static_assert(kBindDebugMenu == kBindCount - 1, "the Debug Menu key is the last action");
int keys_listed() { return debug_menu_active() ? kBindCount : kBindDebugMenu; }

// The row's action; keys_listed() or more for a row the last, short page
// leaves blank, which shows nothing and takes no capture.
int keys_action(int row) { return g_keys.page * kBindPerPage + row - 1; }
bool keys_row_bound(int row) { return row > 0 && keys_action(row) < keys_listed(); }

// Restore Defaults: the row right after the last action, on the last page.
// The section has no room for the game's own Defaults row (its eight rows fill
// the space above the key guide), and a key binding reset is DS3's too. It
// asks twice - the first press says what the second will do - so a stray
// Enter cannot wipe a set of bindings. Moving off it, or four seconds, and
// it asks again from the start.
bool keys_row_reset(int row) { return row > 0 && keys_action(row) == keys_listed(); }
enum class KeysReset { Idle, Armed, Done };
KeysReset g_keys_reset = KeysReset::Idle;
std::chrono::steady_clock::time_point g_keys_reset_at;

void keys_fill_reset_text(int row) {
    // Short: the value column cut "Press again to restore" to "Press again to".
    to_utf16(g_keys_text[row], g_keys_reset == KeysReset::Armed  ? "もう一度押す"
                               : g_keys_reset == KeysReset::Done ? "初期化しました"
                                                                 : "すべての操作");
}

void keys_fill_value(int row) {
    if (keys_row_reset(row)) {
        keys_fill_reset_text(row);
        return;
    }
    if (!keys_row_bound(row)) {
        g_keys_text[row][0] = 0;
        return;
    }
    char d[64];
    host_binding_describe(keys_action(row), d, sizeof(d));
    to_utf16(g_keys_text[row], d);
}

void keys_fill(int page) {
    g_keys.page = (page % kBindPages + kBindPages) % kBindPages;
    g_keys_page = g_keys.page;
    to_utf16(g_keys_label[0], "ページ");
    to_utf16(g_keys_help[0], "左右で操作のグループを切り替えます。");
    to_utf16(g_keys_text[0], kKeyPageNames[g_keys.page]);
    for (int r = 1; r < kKeysRows; ++r) {
        if (keys_row_reset(r)) {
            to_utf16(g_keys_label[r], "初期設定に戻す");
            to_utf16(g_keys_help[r], "すべての操作を初期設定に戻します。Enter またはクリックで選び、もう一度押して確定します。");
            keys_fill_reset_text(r);
            continue;
        }
        if (!keys_row_bound(r)) {
            g_keys_label[r][0] = g_keys_help[r][0] = g_keys_text[r][0] = 0;
            continue;
        }
        const BindingInfo& b = host_binding_info(keys_action(r));
        to_utf16(g_keys_label[r], host_binding_label(b.label));
        char help[512];
        std::snprintf(help, sizeof(help), "%s Enter / クリックで変更、Delete で解除。", b.help);
        to_utf16(g_keys_help[r], help);
        keys_fill_value(r);
    }
}

void redraw(std::uint8_t* list) { hle_call_guest<std::int64_t>(guest_fn(kListRedraw), list); }

// Whether the remembered objects are still this screen's. The dialog is freed
// when it closes and its memory reused, so identity is checked where only the
// port could have put it: every action row writes through one of its bytes.
bool keys_alive() {
    if (!g_keys.dialog || !g_keys.rows) return false;
    std::uint64_t vt = 0;
    std::memcpy(&vt, g_keys.rows, sizeof(vt));
    if (vt != guest(kListVTable) + 0x10 && vt != guest(kListVTable)) return false;
    for (int r = 0; r < kKeysRows; ++r) {
        if (!g_keys.widget[r]) return false;
        std::uint64_t v = 0;
        std::memcpy(&v, g_keys.widget[r] + kChoiceValue, sizeof(v));
        if (v != reinterpret_cast<std::uint64_t>(&g_keys_value[r])) return false;
    }
    return true;
}

GUEST_ABI void pc_keys_handler(void* dialog, void* params) {
    host_log("pc-options: PCKeys opened, dialog=%p params=%p", dialog, params);
    alignas(16) std::uint8_t scratch[0x100]{};
    hle_call_guest<std::int64_t>(guest_fn(kScratchInit), &scratch);
    g_keys = KeysScreen{};
    // The Circle that opened the screen may still be down when the page row
    // first has focus; it is not a press on that row.
    g_keys.pad_was = hle_pad_delivered_buttons();
    keys_fill(g_keys_page);
    add_text_row(dialog, &g_keys_value[0], kKeysPageRow, g_keys_label[0], g_keys_help[0], g_keys_text[0]);
    for (int r = 1; r < kKeysRows; ++r) {
        add_text_row(dialog, &g_keys_value[r], kKeysFirstRow + r - 1, g_keys_label[r], g_keys_help[r], g_keys_text[r]);
    }
    log_rows("PCKeys", dialog, kKeysRows, kKeysRows);
    if (dialog_row_count(dialog) < kKeysRows) return;
    auto* d = static_cast<std::uint8_t*>(dialog);
    g_keys.rows = d + kDialogRows;
    for (int r = 0; r < kKeysRows; ++r) g_keys.widget[r] = dialog_widget(dialog, r);
    g_keys.dialog = d;
    if (!keys_alive()) {
        host_log("pc-options: PCKeys: the dialog is not laid out as expected; bindings show but cannot change here");
        g_keys = KeysScreen{};
    }
}

// The screen's per-frame work, on the menu thread. `row` is which of the
// screen's lists is updating: -2 the dialog's own, 0 the page row's value,
// 1..7 an action row's.
void keys_update(int row, bool focused) {
    bool rows_changed = false, values_changed = false;
    // Restore Defaults forgets a pending press when the cursor leaves it or
    // time runs out, and "Restored" goes back to what the row does.
    if (g_keys_reset != KeysReset::Idle &&
        ((focused && row >= 0 && !keys_row_reset(row)) ||
         std::chrono::steady_clock::now() - g_keys_reset_at > std::chrono::seconds(4))) {
        g_keys_reset = KeysReset::Idle;
        for (int r = 1; r < kKeysRows; ++r) {
            if (keys_row_reset(r)) {
                keys_fill_reset_text(r);
                redraw(g_keys.widget[r] + kChoiceList);
            }
        }
    }
    if (const int done = host_bind_capture_take_done(); done >= 0) {
        // Every row, not just the one captured: a key taken from another
        // action on this page changes that row too.
        for (int r = 1; r < kKeysRows; ++r) keys_fill_value(r);
        values_changed = true;
    }
    if (focused && row >= 0 && host_bind_capturing() < 0) {
        const std::uint32_t pad = hle_pad_delivered_buttons();
        const std::uint32_t edge = pad & ~g_keys.pad_was;
        g_keys.pad_was = pad;
        const std::uint32_t decide = menu_confirm_button();
        if (row == 0 && (edge & (decide | 0x20u | 0x80u))) {
            // Right, or Circle / a click, is the next page; Left the previous.
            keys_fill(g_keys.page + ((edge & 0x80u) ? -1 : 1));
            rows_changed = values_changed = true;
        } else if (keys_row_bound(row) && host_bind_take_clear_key()) {
            host_binding_clear(keys_action(row));
            keys_fill_value(row);
            redraw(g_keys.widget[row] + kChoiceList);
            g_keys_save.store(true, std::memory_order_relaxed);
        } else if (keys_row_bound(row) && (edge & decide)) {
            to_utf16(g_keys_text[row], "キーを押してください...");
            redraw(g_keys.widget[row] + kChoiceList);
            host_bind_capture_begin(keys_action(row));
        } else if (keys_row_reset(row) && (edge & decide)) {
            if (g_keys_reset == KeysReset::Armed) {
                host_bindings_load_defaults();
                g_keys_save.store(true, std::memory_order_relaxed);
                g_keys_reset = KeysReset::Done;
                for (int r = 1; r < kKeysRows; ++r) keys_fill_value(r);
                host_log("pc-options: key bindings restored to their defaults");
            } else {
                g_keys_reset = KeysReset::Armed;
            }
            g_keys_reset_at = std::chrono::steady_clock::now();
            keys_fill_reset_text(row);
            values_changed = true;
        }
    }
    if (rows_changed) redraw(g_keys.rows);
    if (values_changed) {
        for (int r = 0; r < kKeysRows; ++r) redraw(g_keys.widget[r] + kChoiceList);
    }
}

// The "Defaults" row sub_1f20900 appends after the handler returns, in a
// section that has no slot for it - Key Bindings, whose eight rows are all
// that fit above the key guide. The other three sections keep it: their
// movies carry the label slot right after the last row
// (engine/menu_assets.cpp), and their rows are built with real defaults.
// Without a slot it never drew - but the rows list still counted it, and
// pressing Down past the last row scrolled the captions up by one while the
// values, which are separate lists, stayed where they were: every caption
// beside the wrong value. So there it comes off again, the way a vector
// element is destroyed -
// its two caption copies freed, its widget unreferenced - and the rows are
// laid out again from the vector, as sub_1f20900 itself last did.
void drop_defaults_row(std::uint8_t* dialog) {
    std::uint64_t begin = 0, end = 0;
    std::memcpy(&begin, dialog + kItemsBegin, sizeof(begin));
    std::memcpy(&end, dialog + kItemsEnd, sizeof(end));
    if (!begin || end < begin + kItemStride) return;
    auto* rec = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(end - kItemStride));
    const char16_t* label = nullptr;
    std::memcpy(&label, rec, sizeof(label));
    if (label != message_text(kDefaultsRowId)) {
        host_log("pc-options: the last row is not Defaults; left alone");
        return;
    }
    Captions::free_wstring(rec + 0x08);
    Captions::free_wstring(rec + 0x48);
    std::uint64_t widget = 0;
    std::memcpy(&widget, rec + kItemWidget, sizeof(widget));
    if (widget) {
        // DLReferenceCountObject: the count at +8, and the last reference
        // calls the deleting destructor, slot 0.
        auto* w = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(widget));
        std::int32_t refs = 0;
        std::memcpy(&refs, w + 8, sizeof(refs));
        const std::int32_t left = refs - 1;
        std::memcpy(w + 8, &left, sizeof(left));
        if (refs == 1) {
            std::uint64_t vt = 0, dtor = 0;
            std::memcpy(&vt, w, sizeof(vt));
            std::memcpy(&dtor, reinterpret_cast<void*>(static_cast<std::uintptr_t>(vt)), sizeof(dtor));
            hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dtor)), w);
        }
    }
    end -= kItemStride;
    std::memcpy(dialog + kItemsEnd, &end, sizeof(end));
    hle_call_guest<std::int64_t>(guest_fn(kLayoutRows), dialog, 0);
}

// A section that keeps the Defaults row, while it is open. The engine's pick
// list is translucent and opens right under the row that owns it - the port's
// is always the last row, so what lies under it is exactly the Defaults
// caption, which read straight through the list over its first entry. So that
// caption is the port's own copy of the game's text, and it is emptied while
// the list is open: the row providers draw from the raw pointer each record
// carries, which the rows' layout took when the section opened.
constexpr std::size_t kPickItems = 0x100;  // a pick-list widget's drop-down list (sub_1f29370: "ItemList")
struct DefaultsView {
    std::uint8_t* dialog = nullptr;
    std::uint8_t* pick = nullptr;  // the section's pick-list widget, or none
    bool hidden = false;
    char16_t first = 0;  // the caption's first character while it is emptied
};
DefaultsView g_defaults_view;
char16_t g_defaults_caption[32];

void keep_defaults_row(std::uint8_t* dialog, int pick_row) {
    g_defaults_view = DefaultsView{};
    const int rows = dialog_row_count(dialog);
    if (rows < 1) return;
    std::uint64_t begin = 0;
    std::memcpy(&begin, dialog + kItemsBegin, sizeof(begin));
    auto* rec = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(begin)) + (rows - 1) * kItemStride;
    const char16_t* label = nullptr;
    std::memcpy(&label, rec, sizeof(label));
    if (!label || label != message_text(kDefaultsRowId)) {
        host_log("pc-options: the last row is not Defaults; left alone");
        return;
    }
    std::size_t n = 0;
    while (label[n] && n + 1 < sizeof(g_defaults_caption) / sizeof(g_defaults_caption[0])) {
        g_defaults_caption[n] = label[n];
        ++n;
    }
    g_defaults_caption[n] = 0;
    const char16_t* ours = g_defaults_caption;
    std::memcpy(rec, &ours, sizeof(ours));
    hle_call_guest<std::int64_t>(guest_fn(kLayoutRows), dialog, 0);
    g_defaults_view.dialog = dialog;
    g_defaults_view.pick = pick_row >= 0 && pick_row < rows - 1 ? dialog_widget(dialog, pick_row) : nullptr;
}

// From the list update: the pick list took focus, or the rows got it back.
void defaults_view_update(std::uint8_t* comp, bool focused) {
    DefaultsView& v = g_defaults_view;
    if (!v.dialog || !v.pick || !focused) return;
    if (comp == v.pick + kPickItems && !v.hidden) {
        v.hidden = true;
        v.first = g_defaults_caption[0];
        g_defaults_caption[0] = 0;
        redraw(v.dialog + kDialogRows);
    } else if (comp == v.dialog + kDialogRows && v.hidden) {
        v.hidden = false;
        g_defaults_caption[0] = v.first;
        redraw(v.dialog + kDialogRows);
    }
}

// Opening a section: name it, hand sub_1f20900 the handler. The name goes
// through g_open_name, the table slot the port owns. `pick_row` is the row
// of the section's pick list, or -1; -2 means the section has no Defaults
// row (no slot for it in its movie), so the engine's is taken off again.
std::int64_t open_named(void* root, void* params, const char* name, void* handler, int pick_row) {
    std::snprintf(g_open_name, sizeof(g_open_name), "%s", name);
    g_defaults_view = DefaultsView{};
    const std::int64_t dialog =
        hle_call_guest<std::int64_t>(guest_fn(kOpenSection), root, params, g_open_name, handler, 0, 0);
    auto* d = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(dialog));
    if (d && pick_row == -2) {
        drop_defaults_row(d);
    } else if (d) {
        keep_defaults_row(d, pick_row);
    }
    return dialog;
}

// ---- Quit Game on the title menu ---------------------------------------------------
//
// A PC game quits from its title menu. The title's command list
// (sub_1f4a3d0: Continue, Load Game, New Game, System, Log In) is finalised
// by one call, sub_1fea820 at 0x1f4adb3; it goes through a stub that adds a
// "Quit Game" command row first (sub_1f4c9a0, the way System's Exit Game is
// added). The row's opener is built like the game's own Exit Game opener
// (sub_200c9f0): a message box from a group-204 caption made by the menu's
// factory (sub_201ed10), a YES command around an action (sub_1d5db60), and
// the popup made of the two (sub_201a070) handed back for the menu to open -
// so the question is the menu's own, modal, with YES and NO.
constexpr std::uint64_t kTitleFinalize = 0x1fea820, kTitleFinalizeCall = 0x1f4adb3;
constexpr std::uint32_t kQuitRowId = 122000, kQuitQuestion = 920001;

GUEST_ABI std::int64_t quit_yes_action(std::int64_t, std::int64_t) {
    host_log("pc-options: Quit Game");
    host_window_request_quit();
    return 0;
}

void release_engine_ref(std::uint64_t obj) {
    if (!obj) return;
    auto* rc = reinterpret_cast<std::int32_t*>(static_cast<std::uintptr_t>(obj + 8));
    if ((*rc)-- == 1) {
        const std::uint64_t vt = *reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(obj));
        hle_call_guest<std::int64_t>(
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(*reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(vt)))), obj);
    }
}

GUEST_ABI void* quit_opener(void* out, void* factory) {
    alignas(16) std::uint8_t caption[0x40]{};
    hle_call_guest<std::int64_t>(guest_fn(kMsgRepository), &caption[0], 0xcc, kQuitQuestion);
    std::uint64_t box = 0, command = 0;
    hle_call_guest<std::int64_t>(guest_fn(0x201ed10), &box, factory, &caption[0]);
    static void* yes = hle_wrap_fn(reinterpret_cast<void*>(&quit_yes_action));
    GuestFunction action(kOpenerFunctorVTable, yes);
    hle_call_guest<std::int64_t>(guest_fn(0x1d5db60), &command, &action.buf[0], 2);
    hle_call_guest<std::int64_t>(guest_fn(0x201a070), out, &box, &command);
    release_engine_ref(command);
    release_engine_ref(box);
    Captions::free_wstring(&caption[0x08]);
    return out;
}

GUEST_ABI std::int64_t title_finalize_hook(std::uint64_t, const std::uint64_t* saved) {
    void* builder = reinterpret_cast<void*>(static_cast<std::uintptr_t>(saved[4]));  // rsi
    {
        Captions c(kQuitRowId);
        static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&quit_opener));
        GuestFunction f(kRowFunctorVTable, opener);
        hle_call_guest<std::int64_t>(guest_fn(kAddExitRow), builder, c.buf, f.buf);
    }
    hle_call_guest6(guest_fn(kTitleFinalize), static_cast<std::int64_t>(saved[5]), static_cast<std::int64_t>(saved[4]),
                    static_cast<std::int64_t>(saved[3]), 0, 0, 0);
    return 1;  // the call was made here
}

// ---- A yes/no question over PC Graphics ------------------------------------------
//
// The options screens ask their own questions on a stack of their own - the
// Defaults row's "restore defaults?" (sub_1f24850): the section dialog's
// message-box factory (+0x48, sub_201ed10) makes a box from a group-204
// caption, sub_1d5db60 wraps the YES action (a std::function) in a command,
// sub_201a070 makes the popup from the two, and it is pushed on the dialog's
// popup stack (+0x18, the count at +0x40, room for four). Pushed there, the
// question is modal over the screen that asked - the standalone dialog
// (MenuMan's) is not: the rows under it went on taking the keys. YES runs
// the action; NO or Circle pops the popup without it.
constexpr std::uint64_t kBoxFromCaption = 0x201ed10, kCommandFromAction = 0x1d5db60, kPopupFromBox = 0x201a070;
constexpr std::uint32_t kMsgDialogText = 0xcc;  // group 204
std::atomic<std::uint32_t> g_confirm_ask{0};  // message id, 0 for none
std::atomic<bool> g_resolution_redraw{false};
constexpr std::uint64_t kDropdownVTable = 0x5739e90, kDropdownRedraw = 0x1f2b7c0;  // its text from the bound value
std::atomic<int> g_confirm_answer{0};         // 1 yes, 2 no
std::atomic<bool> g_confirm_yes{false};
std::atomic<bool> g_confirm_timed_out{false};  // whatever closes it now counts as no
bool g_confirm_open = false;  // menu thread
std::uint64_t g_confirm_popup = 0;  // ours, on the stack

// PC Graphics still open: the dialog keeps its vtable and its first row
// still writes the port's byte (a freed dialog's memory is reused).
bool gfx_alive() {
    if (!g_gfx_dialog) return false;
    std::uint64_t vt = 0;
    std::memcpy(&vt, g_gfx_dialog, sizeof(vt));
    if (vt != g_gfx_vtable || dialog_row_count(g_gfx_dialog) < 1) return false;
    std::uint64_t v = 0;
    std::memcpy(&v, dialog_widget(g_gfx_dialog, 0) + kChoiceValue, sizeof(v));
    return v == reinterpret_cast<std::uint64_t>(&g_graphics[0].value);
}

GUEST_ABI std::int64_t confirm_yes_action(std::int64_t, std::int64_t) {
    g_confirm_yes.store(true);
    return 0;
}

// A refcounted engine object's release (+8 the count, slot 0 deletes).
void release_ref(std::uint64_t obj) {
    if (!obj) return;
    auto* rc = reinterpret_cast<std::int32_t*>(static_cast<std::uintptr_t>(obj + 8));
    if ((*rc)-- == 1) {
        const std::uint64_t vt = *reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(obj));
        hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(*reinterpret_cast<std::uint64_t*>(
                                         static_cast<std::uintptr_t>(vt)))),
                                     obj);
    }
}

std::uint64_t popup_slot(std::uint8_t* dialog, std::uint64_t i) {
    // The stack's inline storage is aligned up to 8 from +0x18.
    const std::uint64_t base = reinterpret_cast<std::uint64_t>(dialog) + 0x18;
    return base + ((0 - base) & 7) + i * 8;
}


bool confirm_push(std::uint8_t* dialog, std::uint32_t msg) {
    std::uint64_t depth = 0;
    std::memcpy(&depth, dialog + 0x40, sizeof(depth));
    if (depth + 1 >= 5) return false;  // the stack is full
    alignas(16) std::uint8_t caption[0x40]{};
    hle_call_guest<std::int64_t>(guest_fn(kMsgRepository), &caption[0], kMsgDialogText, msg);
    std::uint64_t box = 0, command = 0, popup = 0;
    hle_call_guest<std::int64_t>(guest_fn(kBoxFromCaption), &box, dialog + 0x48, &caption[0]);
    static void* yes = hle_wrap_fn(reinterpret_cast<void*>(&confirm_yes_action));
    GuestFunction action(kOpenerFunctorVTable, yes);
    hle_call_guest<std::int64_t>(guest_fn(kCommandFromAction), &command, &action.buf[0], 2);
    hle_call_guest<std::int64_t>(guest_fn(kPopupFromBox), &popup, &box, &command);
    bool pushed = false;
    if (popup) {
        const std::uint64_t slot = popup_slot(dialog, depth);
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(slot)), &popup, sizeof(popup));  // the stack keeps our reference
        g_confirm_popup = popup;
        // One more reference, ours: the stack the game moves it to holds the
        // other, and when that is dropped the popup has closed.
        ++*reinterpret_cast<std::int32_t*>(static_cast<std::uintptr_t>(popup + 8));
        depth += 1;
        std::memcpy(dialog + 0x40, &depth, sizeof(depth));
        pushed = true;
    }
    release_ref(command);
    release_ref(box);
    Captions::free_wstring(&caption[0x08]);
    return pushed;
}

// The Resolution row's text after its value was changed from outside: the
// dialog's row records (+0xe58, 0x90 each) hold the row object at +0x80; a
// dropdown's sub_1f2b7c0 redraws it from the value it is bound to (+0x3c8).
void dropdown_redraw(std::uint8_t* dialog, const void* bound) {
    std::uint64_t first = 0, end = 0;
    std::memcpy(&first, dialog + 0xe58, sizeof(first));
    std::memcpy(&end, dialog + 0xe60, sizeof(end));
    for (std::uint64_t r = first; r + 0x90 <= end; r += 0x90) {
        const std::uint64_t row = *reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(r + 0x80));
        if (!row || *reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(row)) != guest(kDropdownVTable)) continue;
        if (*reinterpret_cast<std::uint64_t*>(static_cast<std::uintptr_t>(row + 0x3c8)) != reinterpret_cast<std::uint64_t>(bound)) continue;
        hle_call_guest<std::int64_t>(guest_fn(kDropdownRedraw), row);
        return;
    }
}
void resolution_row_redraw() {
    if (!g_resolution_redraw.load() || !gfx_alive()) return;
    g_resolution_redraw.store(false);
    dropdown_redraw(g_gfx_dialog, &g_resolution);
}

// The Steam Deck section still open, as gfx_alive tells PC Graphics.
bool deck_alive() {
    if (!g_deck_dialog) return false;
    std::uint64_t vt = 0;
    std::memcpy(&vt, g_deck_dialog, sizeof(vt));
    if (vt != g_deck_vtable || dialog_row_count(g_deck_dialog) < 1) return false;
    std::uint64_t v = 0;
    std::memcpy(&v, dialog_widget(g_deck_dialog, 0) + kChoiceValue, sizeof(v));
    return v == reinterpret_cast<std::uint64_t>(&g_deck_rate);
}
void deck_row_redraw() {
    if (!g_deck_redraw.load() || !deck_alive()) return;
    g_deck_redraw.store(false);
    dropdown_redraw(g_deck_dialog, &g_deck_model);
}

void confirm_update() {
    resolution_row_redraw();
    deck_row_redraw();
    if (g_confirm_open) {
        const bool alive = gfx_alive();
        if (g_confirm_yes.exchange(false)) {
            g_confirm_open = false;
            g_confirm_answer.store(g_confirm_timed_out.load() ? 2 : 1);
        } else if (!alive || *reinterpret_cast<std::int32_t*>(static_cast<std::uintptr_t>(g_confirm_popup + 8)) <= 1) {
            g_confirm_open = false;
            g_confirm_answer.store(2);
        }
        if (!g_confirm_open) {
            release_ref(g_confirm_popup);
            g_confirm_popup = 0;
        }
        return;
    }
    const std::uint32_t msg = g_confirm_ask.load();
    if (!msg || !gfx_alive()) return;
    // Not over the pick list the change was made in: it closes after its
    // choice is taken, and the question waits for it.
    std::uint64_t depth = 0;
    std::memcpy(&depth, g_gfx_dialog + 0x40, sizeof(depth));
    if (depth > g_gfx_base_depth) return;
    g_confirm_ask.store(0);
    g_confirm_yes.store(false);
    g_confirm_timed_out.store(false);
    if (confirm_push(g_gfx_dialog, msg)) {
        g_confirm_open = true;
        host_log("pc-options: asking over PC Graphics (message %u)", msg);
    } else {
        g_confirm_answer.store(2);
    }
}

// The port's **own** openers, which is all any of the game's five are: name
// the sprite, name the handler, let sub_1f20900 do the rest. Nothing of the
// game's is redirected to get here.
GUEST_ABI std::int64_t pc_open_section(void* root, void* params) {
    static void* handler = hle_wrap_fn(reinterpret_cast<void*>(&pc_section_handler));
    return open_named(root, params, kOurSectionName, handler, kToggleCount);  // the frame cap
}
GUEST_ABI std::int64_t pc_open_graphics(void* root, void* params) {
    static void* handler = hle_wrap_fn(reinterpret_cast<void*>(&pc_graphics_handler));
    return open_named(root, params, kOurGraphicsName, handler, kGraphicsRows - 1);  // the resolution
}
GUEST_ABI std::int64_t pc_open_effects(void* root, void* params) {
    static void* handler = hle_wrap_fn(reinterpret_cast<void*>(&pc_effects_handler));
    return open_named(root, params, kOurEffectsName, handler, -1);
}
GUEST_ABI std::int64_t pc_open_controls(void* root, void* params) {
    static void* handler = hle_wrap_fn(reinterpret_cast<void*>(&pc_controls_handler));
    return open_named(root, params, kOurControlsName, handler, -1);
}
GUEST_ABI std::int64_t pc_open_camera(void* root, void* params) {
    static void* handler = hle_wrap_fn(reinterpret_cast<void*>(&pc_camera_handler));
    return open_named(root, params, kOurCameraName, handler, -1);
}
GUEST_ABI std::int64_t pc_open_deck(void* root, void* params) {
    static void* handler = hle_wrap_fn(reinterpret_cast<void*>(&pc_deck_handler));
    return open_named(root, params, kOurDeckName, handler, kDeckRows - 1);  // the model detail
}
GUEST_ABI std::int64_t pc_open_enhance(void* root, void* params) {
    static void* handler = hle_wrap_fn(reinterpret_cast<void*>(&pc_enhance_handler));
    return open_named(root, params, kOurEnhanceName, handler, -1);
}
GUEST_ABI std::int64_t pc_open_keys(void* root, void* params) {
    static void* handler = hle_wrap_fn(reinterpret_cast<void*>(&pc_keys_handler));
    return open_named(root, params, kOurKeysName, handler, -2);
}

// What a command row runs when the player picks it - the same two lines
// sub_1fb4be0 runs for Language, with the port's opener in place of
// sub_1fdb940.
void* row_open(void* out, void* ctx, void* opener) {
    GuestFunction f(kOpenerFunctorVTable, opener);
    hle_call_guest<std::int64_t>(guest_fn(kOpenSectionFlow), out, ctx, f.buf);
    return out;
}
GUEST_ABI void* pc_row_open(void* out, void* ctx) {
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&pc_open_section));
    return row_open(out, ctx, opener);
}
GUEST_ABI void* pc_graphics_row_open(void* out, void* ctx) {
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&pc_open_graphics));
    return row_open(out, ctx, opener);
}
GUEST_ABI void* pc_effects_row_open(void* out, void* ctx) {
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&pc_open_effects));
    return row_open(out, ctx, opener);
}
GUEST_ABI void* pc_controls_row_open(void* out, void* ctx) {
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&pc_open_controls));
    return row_open(out, ctx, opener);
}
GUEST_ABI void* pc_camera_row_open(void* out, void* ctx) {
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&pc_open_camera));
    return row_open(out, ctx, opener);
}
GUEST_ABI void* pc_deck_row_open(void* out, void* ctx) {
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&pc_open_deck));
    return row_open(out, ctx, opener);
}
GUEST_ABI void* pc_enhance_row_open(void* out, void* ctx) {
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&pc_open_enhance));
    return row_open(out, ctx, opener);
}
GUEST_ABI void* pc_keys_row_open(void* out, void* ctx) {
    static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&pc_open_keys));
    return row_open(out, ctx, opener);
}

// The seventh row, appended to the System command list just before the builder
// becomes a dialog. The game's own six have already been added (and Language
// or Exit Game already popped), so this lands at the bottom of whichever list
// the player is looking at.
// Is this row Exit Game? Its std::function holds sub_200c9f0, and a row is
// 0x180 bytes, so the question is whether that pointer is anywhere in it.
// Reading the id back out is not an option - a row keeps the caption's text,
// not the number it came from.
bool row_is_exit_game(const std::uint8_t* row) {
    const std::uint64_t want = guest(kExitGameOpener);
    for (std::size_t i = 0; i + sizeof(std::uint64_t) <= kBuilderRowStride;
         i += sizeof(std::uint64_t)) {
        std::uint64_t v = 0;
        std::memcpy(&v, row + i, sizeof(v));
        if (v == want) {
            return true;
        }
    }
    return false;
}

std::uint8_t* builder_last_row(void* builder) {
    std::uint64_t begin = 0, end = 0;
    auto* b = static_cast<std::uint8_t*>(builder);
    std::memcpy(&begin, b + kBuilderBegin, sizeof(begin));
    std::memcpy(&end, b + kBuilderEnd, sizeof(end));
    if (!begin || end <= begin || (end - begin) % kBuilderRowStride != 0) {
        return nullptr;
    }
    return reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(end - kBuilderRowStride));
}

// The game's own pop, replayed: slot 1 of the row's vtable, then drop the end
// pointer by one row.
void pop_last_row(void* builder) {
    std::uint8_t* row = builder_last_row(builder);
    if (!row) {
        return;
    }
    std::uint64_t vt = 0, dtor = 0;
    std::memcpy(&vt, row, sizeof(vt));
    std::memcpy(&dtor, reinterpret_cast<void*>(static_cast<std::uintptr_t>(vt + 8)), sizeof(dtor));
    hle_call_guest<std::int64_t>(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dtor)), row);
    std::uint64_t end = 0;
    auto* b = static_cast<std::uint8_t*>(builder);
    std::memcpy(&end, b + kBuilderEnd, sizeof(end));
    end -= kBuilderRowStride;
    std::memcpy(b + kBuilderEnd, &end, sizeof(end));
}

GUEST_ABI void* pc_system_finalize(void* out, void* builder, void* arg3) {
    static void* const rows[][2] = {
        {reinterpret_cast<void*>(static_cast<std::uintptr_t>(kSystemEnhanceRowId)),
         hle_wrap_fn(reinterpret_cast<void*>(&pc_enhance_row_open))},
        {reinterpret_cast<void*>(static_cast<std::uintptr_t>(kSystemRowId)),
         hle_wrap_fn(reinterpret_cast<void*>(&pc_row_open))},
        {reinterpret_cast<void*>(static_cast<std::uintptr_t>(kSystemGraphicsRowId)),
         hle_wrap_fn(reinterpret_cast<void*>(&pc_graphics_row_open))},
        {reinterpret_cast<void*>(static_cast<std::uintptr_t>(kSystemEffectsRowId)),
         hle_wrap_fn(reinterpret_cast<void*>(&pc_effects_row_open))},
        {reinterpret_cast<void*>(static_cast<std::uintptr_t>(kSystemControlsRowId)),
         hle_wrap_fn(reinterpret_cast<void*>(&pc_controls_row_open))},
        {reinterpret_cast<void*>(static_cast<std::uintptr_t>(kSystemCameraRowId)),
         hle_wrap_fn(reinterpret_cast<void*>(&pc_camera_row_open))},
        {reinterpret_cast<void*>(static_cast<std::uintptr_t>(kSystemKeysRowId)),
         hle_wrap_fn(reinterpret_cast<void*>(&pc_keys_row_open))},
    };
    std::int64_t flag = 0;
    // Exit Game, if it is there, is last at this point - sub_1fb3ad0 adds it
    // fifth and then adds and pops Language above it. Appending after it would
    // bury it in the middle of the in-game list, so it comes off first and
    // goes back on at the end. Popping and re-adding is what the builder's own
    // code does with these rows; nothing here moves a row's bytes, which would
    // break the std::function's pointer to its own inline storage.
    std::uint8_t* last = builder_last_row(builder);
    const bool move_exit = last && row_is_exit_game(last);
    if (move_exit) {
        pop_last_row(builder);
    }
    // On a Steam Deck its own section first of the port's: the frame rate is
    // what a Deck player comes to System for.
    if (host_steam_deck()) {
        Captions c(kSystemDeckRowId);
        static void* opener = hle_wrap_fn(reinterpret_cast<void*>(&pc_deck_row_open));
        GuestFunction f(kRowFunctorVTable, opener);
        hle_call_guest<std::int64_t>(guest_fn(kAddCommandRow), builder, c.buf, f.buf, &flag);
    }
    // PC Enhancements, PC Settings, PC Graphics, PC Effects, PC Controls, PC Camera, Key Bindings. The list has five
    // lines and **scrolls**, with the scroll bar the port's movie gives it, so
    // rows past the fifth need no slot of their own.
    for (const auto& r : rows) {
        Captions c(static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(r[0])));
        GuestFunction f(kRowFunctorVTable, r[1]);
        hle_call_guest<std::int64_t>(guest_fn(kAddCommandRow), builder, c.buf, f.buf, &flag);
    }
    if (move_exit) {
        // Three arguments and no flag: Exit Game is not a section opener, it
        // is a command, and sub_1f4c9a0 is what builds one.
        Captions c(kExitGameRowId);
        GuestFunction f(kRowFunctorVTable, guest_fn(kExitGameOpener));
        hle_call_guest<std::int64_t>(guest_fn(kAddExitRow), builder, c.buf, f.buf);
    }
    hle_call_guest<std::int64_t>(guest_fn(kSystemFinalize), out, builder, arg3);
    static int logged = 0;
    if (logged < 4) {
        ++logged;
        host_log("pc-options: System list built, Exit Game %s",
                 move_exit ? "moved to the bottom" : "not in this list");
    }
    return out;
}

// Adds the port's name slot to sub_1f20900's table and widens the loop that
// walks it. Without this the name is simply never matched and no port section
// can be opened.
bool patch_section_table(ElfImage* image) {
    const std::uint64_t slot = guest(kSectionNameTable) + 6 * 8;
    const std::uint64_t imm = guest(kSectionCountImm);
    auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, slot));
    auto* c = static_cast<std::uint8_t*>(guest_ptr(image->mem, imm));
    std::uint64_t existing = 0, existing2 = 0;
    std::memcpy(&existing, p, 8);
    std::memcpy(&existing2, p + 8, 8);
    if (existing != 0 || existing2 != 0 || *c != 6) {
        host_log("pc-options: refused, section table slot 0x%llx is %llu and the bound is %u",
                 static_cast<unsigned long long>(slot), static_cast<unsigned long long>(existing),
                 *c);
        return false;
    }
    if (!guest_protect_rw(&image->mem, slot & ~0xfffull, 0x1000)) {
        host_log("pc-options: cannot unprotect the section table");
        return false;
    }
    const std::uint64_t name = reinterpret_cast<std::uint64_t>(g_open_name);
    std::memcpy(p, &name, 8);
    if (!guest_protect_rwx(&image->mem, imm & ~0xfffull, 0x1000)) {
        host_log("pc-options: cannot unprotect the section loop");
        return false;
    }
    *c = 7;  // one more name in the table, so the loop walks seven
    guest_protect_rx(&image->mem, imm & ~0xfffull, 0x1000);
    return true;
}

// Redirects the one call that turns the System builder into a dialog, so the
// port can append its row first.
//
// A `call rel32` cannot reach a host address, and the thunk page is wherever
// mmap put it, so the call is pointed at the 16 bytes of alignment padding
// between sub_1fb4be0 and sub_1fb4c70 and those hold an absolute jump. Both
// the call and the padding are on one page, and both are checked byte for byte
// before anything is written.
bool patch_system_row(ElfImage* image) {
    const std::uint64_t call = guest(kSystemFinalizeCall);
    const std::uint64_t pad = guest(kTrampolinePad);
    auto* cp = static_cast<std::uint8_t*>(guest_ptr(image->mem, call));
    auto* pp = static_cast<std::uint8_t*>(guest_ptr(image->mem, pad));
    std::int32_t rel = 0;
    std::memcpy(&rel, cp + 1, sizeof(rel));
    if (cp[0] != 0xe8 || call + 5 + static_cast<std::uint64_t>(rel) != guest(kSystemFinalize)) {
        host_log("pc-options: refused, 0x%llx is not the call to sub_1fea630",
                 static_cast<unsigned long long>(kSystemFinalizeCall));
        return false;
    }
    if (std::memcmp(pp, kPadBytes, sizeof(kPadBytes)) != 0) {
        host_log("pc-options: refused, 0x%llx is not the alignment padding",
                 static_cast<unsigned long long>(kTrampolinePad));
        return false;
    }
    if (!guest_protect_rwx(&image->mem, call & ~0xfffull, 0x1000)) {
        host_log("pc-options: cannot unprotect the System command list");
        return false;
    }
    const std::uint64_t target =
        reinterpret_cast<std::uint64_t>(hle_wrap_fn(reinterpret_cast<void*>(&pc_system_finalize)));
    std::uint8_t* t = pp;
    *t++ = 0x49;
    *t++ = 0xbb;  // movabs r11, target
    std::memcpy(t, &target, 8);
    t += 8;
    *t++ = 0x41;
    *t++ = 0xff;
    *t++ = 0xe3;  // jmp r11
    const std::int32_t to_pad = static_cast<std::int32_t>(static_cast<std::int64_t>(pad) -
                                                         static_cast<std::int64_t>(call + 5));
    std::memcpy(cp + 1, &to_pad, sizeof(to_pad));
    guest_protect_rx(&image->mem, call & ~0xfffull, 0x1000);
    return true;
}

// The overlay has to define the PCSetting sprite and carry the port's message
// ids. Without the sprite sub_1f20900 binds a name the movie does not have;
// without the ids every row draws blank. Neither is a thing to discover in
// front of the player.
// Either overlay serves: the player's own (paths.mods) or the one made from
// the dump at start (engine/menu_assets.h).
bool overlay_has(const char* rel) {
    if (!hle_fs_overlay_file(rel).empty()) {
        return true;
    }
    host_log("pc-options: no overlay has %s", rel);
    return false;
}

// Game list updates so far (option_menu_list_update: every list on screen,
// every frame): the rows' bytes below can only change while some list updates.
std::atomic<std::uint64_t> g_list_updates{0};

bool overlay_has_messages() {
    for (const char* lang : kMessageLanguages) {
        const std::string rel = std::string("/dvdroot_ps4/msg/") + lang + "/menu.msgbnd.dcx";
        if (!hle_fs_overlay_file(rel.c_str()).empty()) return true;
    }
    host_log("pc-options: no overlay has a menu message bundle");
    return false;
}

}  // namespace

void option_menu_poll() {
    if (!g_installed) {
        return;
    }
    // Resolve the YEBIS settings once the game has built them. It is looked
    // for from here because this is the one thing that ticks every frame and
    // is already gated on the port being installed.
    yebis_poll();
    if (!g_seeded) {
        // The rows must open showing what the settings actually are, not a
        // default, so the first poll takes the host's values rather than
        // giving them.
        g_seeded = true;
        for (auto& t : g_toggles) seed(t);
        for (auto& t : g_graphics) seed(t);
        for (auto& t : g_effects) seed(t);
        for (auto& t : g_enhance) seed(t);
        seed(g_gfx_ao);
        seed(g_gfx_shadow);
        seed(g_gfx_fog);
        seed(g_fx_bloom);
        seed(g_fx_saturation);
        seed(g_cam_fov);
        seed(g_cam_distance);
        seed(g_cam_height);
        seed(g_cam_mouse_auto_rotation);
        seed_camera_tuning();
        seed_controls();
        seed_deck();
        g_resolution = host_opt_resolution_index();
        g_last_resolution = g_resolution;
        g_frame_cap = host_opt_frame_cap();
        g_last_frame_cap = g_frame_cap;
        return;
    }
    // A row's byte is written by the game's list update as the player moves
    // in a menu, so the rows are looked at while a menu is up - its lists
    // update every frame - and for two seconds after the last update, which
    // takes a change made as the menu closed; a game being played without a
    // menu is not asked every pump whether a row moved.
    {
        static std::uint64_t seen = ~0ull;
        static std::chrono::steady_clock::time_point until{};
        const auto now = std::chrono::steady_clock::now();
        const std::uint64_t updates = g_list_updates.load(std::memory_order_relaxed);
        if (updates != seen) {
            seen = updates;
            until = now + std::chrono::seconds(2);
        } else if (now > until) {
            return;
        }
    }
    if (g_resolution != g_last_resolution) {
        g_last_resolution = g_resolution;
        host_opt_set_resolution_index(g_resolution);
        host_log("pc-options: resolution entry %d from the System menu", g_resolution);
    } else if (host_opt_resolution_index() != g_last_resolution) {
        // Changed elsewhere - F10, or the keep-or-revert question going
        // back - and the row shows it on the next menu update.
        g_resolution = g_last_resolution = host_opt_resolution_index();
        g_resolution_redraw.store(true);
    }
    for (auto& t : g_graphics) push(t);
    for (auto& t : g_effects) push(t);
    push(g_gfx_ao);
    push(g_gfx_shadow);
    push(g_gfx_fog);
    push(g_fx_bloom);
    push(g_fx_saturation);
    push(g_cam_fov);
    push(g_cam_distance);
    push(g_cam_height);
    push(g_cam_mouse_auto_rotation);
    if (g_cam_tuning != g_cam_tuning_last) {
        g_cam_tuning_last = g_cam_tuning;
        host_opt_set_index("camera_tuning", g_cam_tuning);
        host_log("pc-options: camera tuning %s from the System menu (applies on the next run)",
                 g_cam_tuning ? "Original" : "FPS++ (Kyo)");
    }
    for (auto& t : g_toggles) push(t);
    for (auto& t : g_enhance) push(t);
    // Steam Deck: the frame rate sets the frame cap, the resolution and the
    // model detail at once, and the model detail row then shows what it set -
    // unless that row changed too (Defaults puts back both), when it has the
    // last word.
    const bool deck_rate = g_deck_rate != g_deck_rate_last, deck_model = g_deck_model != g_deck_model_last;
    if (deck_rate) {
        g_deck_rate_last = g_deck_rate;
        host_set_deck_profile(g_deck_rate);
        g_frame_cap = g_last_frame_cap = host_opt_frame_cap();
        host_log("pc-options: Steam Deck frame rate %s from the System menu", g_deck_rate ? "60" : "30");
    }
    if (deck_model) {
        g_deck_model_last = g_deck_model;
        host_opt_set_index("model_detail", g_deck_model);
        host_log("pc-options: model detail %d from the System menu", g_deck_model);
    } else if (deck_rate) {
        const int m = host_opt_index("model_detail");
        g_deck_model = g_deck_model_last = m < 0 ? 1 : m;
        g_deck_redraw.store(true);
    }
    if (g_frame_cap != g_last_frame_cap) {
        g_last_frame_cap = g_frame_cap;
        host_opt_set_frame_cap(g_frame_cap);
        host_log("pc-options: frame rate %d from the System menu (0 is uncapped; applies on the next run)", g_frame_cap);
    }
    if (g_keys_save.exchange(false)) {
        host_options_save_now();
    }
    // PC Controls.
    for (Toggle* t : {&g_ctl_camera, &g_ctl_pointer[0], &g_ctl_pointer[1]}) {
        if (t->value != t->last) {
            t->last = t->value;
            host_opt_set(t->key, t->value != 0);
            host_log("pc-options: %s %s from the System menu", t->key, t->value ? "on" : "off");
        }
    }
    if (g_ctl_sens != g_ctl_sens_last) {
        g_ctl_sens_last = g_ctl_sens;
        host_opt_set_index("mouse_sens", g_ctl_sens);
        host_log("pc-options: mouse sensitivity %u from the System menu", g_ctl_sens);
    }
    for (int i = 0; i < 2; ++i) {
        if (g_ctl_axis[i] != g_ctl_axis_last[i]) {
            g_ctl_axis_last[i] = g_ctl_axis[i];
            host_opt_set(kAxisKeys[i], g_ctl_axis[i] == 0);  // the game's list: 0 is Reversed
            host_log("pc-options: %s %s from the System menu", kAxisKeys[i], g_ctl_axis[i] ? "normal" : "reversed");
        }
    }
}

bool option_menu_click_decides(void* list, int index) {
    return g_installed && g_keys.dialog && list == g_keys.rows && index >= 1 && index < kKeysRows &&
           (keys_row_bound(index) || keys_row_reset(index)) && keys_alive();
}

void option_menu_list_update(void* comp, bool focused) {
    if (!g_installed) {
        return;
    }
    g_list_updates.fetch_add(1, std::memory_order_relaxed);  // a menu is up (option_menu_poll)
    confirm_update();
    auto* c = static_cast<std::uint8_t*>(comp);
    defaults_view_update(c, focused);
    if (!g_keys.dialog) {
        return;
    }
    int row = -1;
    if (c == g_keys.rows) {
        row = -2;
    } else {
        for (int r = 0; r < kKeysRows && row == -1; ++r) {
            if (c == g_keys.widget[r] + kChoiceList) row = r;
        }
    }
    if (row == -1) {
        return;
    }
    if (!keys_alive()) {
        g_keys = KeysScreen{};
        return;
    }
    keys_update(row, focused);
}

void option_menu_install(ElfImage* image) {
    // On by default: the PC rows are part of the port, not an experiment.
    // BBHOST_PC_OPTIONS=0 leaves System exactly as the game builds it.
    const char* e = std::getenv("BBHOST_PC_OPTIONS");
    if (e && e[0] == '0') {
        host_log("pc-options: off (BBHOST_PC_OPTIONS=0)");
        return;
    }
    g_slide = image->mem.slide;
    menu_assets_ensure();
    if (!overlay_has(kOurMovie) || !overlay_has_messages()) {
        host_log("pc-options: off - the overlay is missing the PCSetting sprite or the port's message ids");
        return;
    }
    if (!patch_section_table(image)) {
        return;
    }
    if (!patch_system_row(image)) {
        return;
    }
    if (!engine_call_site_hook(image, guest(kTitleFinalizeCall), guest(kTitleFinalize), reinterpret_cast<void*>(&title_finalize_hook)))
        host_log("pc-options: the title menu's Quit Game did not go in");
    g_installed = true;
    host_log("pc-options: System gains %sPC Enhancements, PC Settings, PC Graphics, PC Effects, PC Controls, PC Camera and Key Bindings",
             host_steam_deck() ? "Steam Deck (a Steam Deck), " : "");
}

bool option_menu_confirm_possible() { return g_installed && gfx_alive(); }

void option_menu_confirm_ask(std::uint32_t message) {
    g_confirm_answer.store(0);
    g_confirm_ask.store(message);
}

int option_menu_confirm_take() { return g_confirm_answer.exchange(0); }

// The question has no Circle and no close of its own to call, so when time
// runs out it is closed the way a player would close it - the menu's decide
// button on whichever answer is highlighted - and counted as no either way.
void option_menu_confirm_close() {
    if (!g_confirm_open) return;
    g_confirm_timed_out.store(true);
    hle_pad_tap(menu_confirm_button(), 0, 120);
}
