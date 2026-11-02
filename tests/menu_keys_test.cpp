// The menus' own keys (host/bindings.cpp): what a menu reads from the
// shoulders and two face buttons - its pages, its two shortcuts, its display
// and status switches - comes from the menu actions' keys, and the play-only
// actions that share those pad buttons in play (the attacks, the firearm,
// Transform, the Blood Vial, the quick item) stay out of the menus; a menu
// action and a play-only one may share a key, anything else gives it up; an
// options file from before the menu actions keeps a key the player gave an
// action that works in menus too.
#include "core/config.h"
#include "host/bindings.h"
#include "host/window.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#if defined(BBHOST_HAVE_SDL3)
#include <SDL3/SDL.h>
#endif

// What host/bindings.cpp asks of the rest of the host.
const HostConfig& config() {
    static HostConfig c;
    return c;
}
bool debug_menu_active() { return false; }
std::uint32_t menu_confirm_button() { return 0x4000u; }
std::uint32_t menu_back_button() { return 0x2000u; }

namespace {

int g_fail = 0;
#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++g_fail;                                                  \
        }                                                              \
    } while (0)

constexpr std::uint32_t kL1 = 0x400, kR1 = 0x800, kL2 = 0x100, kR2 = 0x200, kTriangle = 0x1000, kCircle = 0x2000,
                        kCross = 0x4000, kSquare = 0x8000;

// The pad the keyboard gives with these actions held, before the pad read
// decides whether a menu has the input; `pad` is what a controller holds.
PadState keys(std::initializer_list<int> actions, std::uint32_t pad = 0) {
    bool held[kBindCount] = {};
    for (const int a : actions) held[a] = true;
    PadState p;
    p.buttons = pad;
    host_bindings_apply(held, false, p, 0);
    return p;
}
// The same as a menu reads it.
PadState in_menu(PadState p) {
    host_bindings_menu_pad(p);
    return p;
}

// A key or mouse button chosen on the Key Bindings screen.
void bind_key(int action, int scancode) {
    host_bind_capture_begin(action);
    host_bind_capture_key(scancode);
}

}  // namespace

int main() {
    // Where each action works.
    for (const int a : {kBindConfirm, kBindBack, kBindMenuPageL, kBindMenuPageR, kBindMenuShortL, kBindMenuShortR,
                        kBindMenuDisplay, kBindMenuStatus})
        CHECK(host_binding_in_menus(a) == 1);
    for (const int a : {kBindAttack, kBindStrong, kBindStrongMod, kBindTransform, kBindFirearm, kBindBloodVial, kBindUseItem})
        CHECK(host_binding_in_menus(a) == -1);
    for (const int a : {kBindMoveF, kBindRoll, kBindInteract, kBindLockOn, kBindMenu, kBindSwitchItem, kBindGestures, kBindLookU})
        CHECK(host_binding_in_menus(a) == 0);

    // The storage box: Show Storage / Show Inventory, its two boxes, is L2 or
    // R2. In play Strong Attack (V) is R2; in a menu it is nothing, and the
    // menu shortcut is R2.
    {
        const PadState strong = keys({kBindStrong});
        CHECK(strong.buttons == kR2 && strong.r2 == 255 && strong.play_buttons == kR2 && strong.menu_buttons == 0);
        const PadState m = in_menu(strong);
        CHECK(m.buttons == 0 && m.r2 == 0);
        const PadState deposit = keys({kBindMenuShortR});
        CHECK(deposit.buttons == 0 && deposit.menu_buttons == kR2);
        const PadState dm = in_menu(deposit);
        CHECK(dm.buttons == kR2 && dm.r2 == 255);
        // The firearm (X) and the other shortcut: L2 the same way.
        CHECK(in_menu(keys({kBindFirearm})).buttons == 0);
        CHECK(in_menu(keys({kBindMenuShortL})).buttons == kL2);
        // Transform and Attack are L1 and R1 in play only; a menu's pages are its own.
        CHECK(in_menu(keys({kBindTransform, kBindAttack})).buttons == 0);
        CHECK(in_menu(keys({kBindMenuPageL})).buttons == kL1);
        CHECK(in_menu(keys({kBindMenuPageR})).buttons == kR1);
        // The quick item and the Blood Vial: Square and Triangle in play only;
        // Toggle Display and Toggle Status are the menu's.
        CHECK(keys({kBindUseItem, kBindBloodVial}).buttons == (kSquare | kTriangle));
        CHECK(in_menu(keys({kBindUseItem, kBindBloodVial})).buttons == 0);
        CHECK(in_menu(keys({kBindMenuDisplay, kBindMenuStatus})).buttons == (kSquare | kTriangle));
        // In play the menu's keys press nothing.
        CHECK(keys({kBindMenuPageL, kBindMenuPageR, kBindMenuShortL, kBindMenuShortR, kBindMenuDisplay, kBindMenuStatus}).buttons == 0);
    }
    // What works in both still does: Interact and Dodge are Cross and Circle
    // (a menu's confirm and back in the US region), Menu Confirm and Back the
    // menu's alone.
    {
        const PadState both = keys({kBindInteract, kBindRoll});
        CHECK(both.buttons == (kCross | kCircle) && both.play_buttons == 0);
        CHECK(in_menu(both).buttons == (kCross | kCircle));
        const PadState ok = keys({kBindConfirm});
        CHECK(ok.buttons == 0 && ok.menu_buttons == kCross);
        CHECK(in_menu(ok).buttons == kCross);
        // A menu reads the movement keys as its directions: the stick is the menu's too.
        PadState moving = keys({kBindMoveF});
        moving.ly = kStickLow;
        CHECK(moving.play_buttons == 0 && in_menu(moving).ly == kStickLow);
    }
    // A controller's own R2 stays a menu's R2 when the keyboard's Strong
    // Attack is held as well: only what the keyboard alone put there goes.
    {
        const PadState p = keys({kBindStrong}, kR2);
        CHECK(p.play_buttons == 0);
        CHECK(in_menu(p).buttons == kR2);
    }
    // Rebinding: a menu action and a play-only one share a key; a menu action
    // and one that works in menus too do not.
    {
        bind_key(kBindUseItem, 9);
        bind_key(kBindMenuDisplay, 9);
        CHECK(host_binding_key(kBindUseItem) == 9 && host_binding_key(kBindMenuDisplay) == 9);
        bind_key(kBindInteract, 8);
        bind_key(kBindMenuPageR, 8);
        CHECK(host_binding_key(kBindMenuPageR) == 8 && host_binding_key(kBindInteract) == 0);
        bind_key(kBindConfirm, 50);
        bind_key(kBindMenuPageL, 50);
        CHECK(host_binding_key(kBindMenuPageL) == 50 && host_binding_key(kBindConfirm) == 0);
    }
#if defined(BBHOST_HAVE_SDL3)
    // The defaults: the pages on Page Up and Page Down, the shortcuts on Home
    // and End, the switches on F and R beside the quick item's and the Blood
    // Vial's.
    {
        host_bindings_load_defaults();
        CHECK(host_binding_key(kBindMenuPageL) == SDL_SCANCODE_PAGEUP && host_binding_key(kBindMenuPageR) == SDL_SCANCODE_PAGEDOWN);
        CHECK(host_binding_key(kBindMenuShortL) == SDL_SCANCODE_HOME && host_binding_key(kBindMenuShortR) == SDL_SCANCODE_END);
        CHECK(host_binding_key(kBindMenuDisplay) == host_binding_key(kBindUseItem));
        CHECK(host_binding_key(kBindMenuStatus) == host_binding_key(kBindBloodVial));
        // No two actions that work at once share a default key.
        for (int a = 0; a < kBindCount; ++a)
            for (int b = a + 1; b < kBindCount; ++b) {
                const int k = host_binding_key(a);
                if (!k || k != host_binding_key(b)) continue;
                const bool apart = host_binding_in_menus(a) * host_binding_in_menus(b) == -1;
                if (!apart) std::printf("  %s and %s share a key\n", host_binding_info(a).key, host_binding_info(b).key);
                CHECK(apart);
            }
    }
    // An options file from before the menu actions, where the player put Page
    // Up on Interact and Page Down on the quick item: Interact keeps Page Up and
    // the previous page goes without; the quick item and the next page share
    // Page Down.
    {
        host_bindings_load_defaults();
        CHECK(host_bindings_parse("interact", "PageUp"));
        CHECK(host_bindings_parse("use_item", "PageDown"));
        host_bindings_migrate_menu_keys();
        CHECK(host_binding_key(kBindInteract) == SDL_SCANCODE_PAGEUP && host_binding_key(kBindMenuPageL) == 0);
        CHECK(host_binding_key(kBindUseItem) == SDL_SCANCODE_PAGEDOWN && host_binding_key(kBindMenuPageR) == SDL_SCANCODE_PAGEDOWN);
    }
#endif
    if (g_fail) {
        std::printf("menu_keys_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("menu_keys_test: ok\n");
    return 0;
}
