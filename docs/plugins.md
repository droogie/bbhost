# Plugins

A plugin is a shared library (`.so` on Linux, `.dll` on Windows; on macOS a
Mach-O bundle, still named `.so`) that bbhost
loads at start. Because the game runs natively inside bbhost's process, a
plugin can read and write game objects directly, hook game functions, call
them, and use the services bbhost builds on top of the engine (params, event
flags, the player, warps, characters, settings, file overlays).

Like the rest of bbhost, the plugin API (`include/bbhost_plugin.h`,
`include/bbhost/`) is licensed under the GNU GPL, version 3 or later. A
plugin built against it is a derivative work: if you distribute one, it must
be under a GPL-compatible license, with its source.

## Where plugins live

bbhost loads every plugin in `<exe dir>/plugins` and `<data>/plugins`. A file
in a later directory replaces one with the same name in an earlier one.
`BBHOST_PLUGINS=0` loads none.

Plugins are enabled and disabled by name under `[plugins]` in `bbhost.toml`:

```toml
[plugins]
randomizer = true     # an opt-in plugin, turned on
mutators = false      # any plugin can be turned off
signed_only = false   # true: load only signed official plugins
```

The setup window has a Plugins tab with the same switches and each plugin's
settings. In game, F9 opens the plugin menu over the game.

## A first plugin

`plugins/hello/hello.c` is the example. bbhost's own build produces it as
`build/plugins/hello.so` (or `hello.dll`). A plugin needs only the header:

```sh
# Linux
cc -shared -fPIC -I<bbhost>/include -o plugins/mine.so mine.c
# Windows, cross-compiled with Clang
clang --target=x86_64-w64-mingw32 -shared -I<bbhost>/include -o plugins/mine.dll mine.c
# macOS: x86_64, like bbhost
clang -arch x86_64 -bundle -I<bbhost>/include -o plugins/mine.so mine.c
```

The smallest useful plugin counts frames:

```c
#include "bbhost_plugin.h"

static const BbHostApi* g_api;
static unsigned g_frames;

static void on_frame(void* user) {
    (void)user;
    if (++g_frames % 300 == 0) g_api->log("hello: another 300 frames");
}

BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION, 0, "mine", "My plugin", "1.0.0", "me", "Counts frames."};

BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (api->version < 2) return 1;  /* on_frame arrived in version 2 */
    g_api = api;
    api->on_frame(on_frame, 0);
    return 0;
}
```

## Life cycle

| Entry point | Called | Typical use |
|---|---|---|
| `bb_plugin_info` (data, optional) | read before anything else, also by the setup window | name, title, version, flags |
| `bb_plugin_options` (data, optional) | read without starting the plugin | the settings the menus draw |
| `int bb_plugin_init(const BbHostApi*)` | before the eboot is loaded | `log`, `config`, `register_hle` |
| `int bb_plugin_image(const BbHostApi*)` | after the eboot is loaded and bbhost's patches are in | everything else |

Either entry point may be missing. Returning nonzero unloads the plugin with a
log line. Everything a plugin logs is prefixed with `plugin:`.

The API only ever grows. A plugin checks that `api->version` is at least the
version whose calls it makes, and `api->size` for fields a newer host
appended. The current version is 10 (`BB_PLUGIN_API_VERSION`).

## Addresses

Addresses in the API are Binary Ninja addresses of the 1.09 eboot, with the
image at its preferred base `0x400000`. `guest_addr()` converts one to the
runtime address. `read`, `write`, `patch` and `hook` take Binary Ninja
addresses directly and refuse them when the loaded eboot is not the 1.09
build. A runtime address above the image (the game's heap) is passed as is.

The engine SDK expresses addresses as image-relative `bb::Rva` values; `.bn()`
gives the Binary Ninja address.

## The API by area

**Memory and code**

- `read(addr, out, n)`: a safe copy; 0 on success.
- `write(addr, data, n)`: writes data pages.
- `patch(addr, expect, bytes, n)`: writes code bytes only if `expect` is there,
  then makes the page read-only and executable again. A patch manifest
  (see [modding.md](modding.md)) is simpler when the bytes are fixed.

**Hooks and calls**

- `hook(addr, prologue, n, fn, user)`: a prologue hook. `fn` sees the six
  integer argument registers and returns nonzero to skip the function.
  `prologue` is the function's first `n` bytes (at least 14, none of them
  rip-relative).
- `hook2(...)` (v8): the same with a `BbHookCtx`: integer and float argument
  registers, both writable, and the value a skipped call returns.
- `call_guest(addr, regs)` (v8): calls a game function with up to six integer
  arguments, eight float registers and four stack arguments; returns rax and
  xmm0. Call it from the main thread.
- `symbol(name)` (v8): the address of a named engine symbol from
  `bbhost/engine/symbols.hpp`.
- `register_hle(name, fn)`: replaces bbhost's implementation of a PS4 system
  function by its export name. Only from `bb_plugin_init`. The game calls `fn`
  with the SysV ABI, so it must be declared `BB_GUEST_ABI` (Clang or GCC; not
  possible with MSVC).

**The engine's state**

- `on_frame(fn, user)` (v2): once per game frame, on the game's main thread.
  This is where game state may be touched safely.
- `param_row`, `param_get`, `param_set`, `param_ids` (v2, v8): param rows by
  table name and id, and fields by the game's own field names, checked against
  the paramdefs the game ships.
- `lua_event(name)` (v2): raises a named Lua event the way the game's scripts do.
- `event_flag_get`, `event_flag_set` (v3) and `on_event_flag` (v4): the world's
  state as numbered flags (bosses killed, doors opened, NPC progress), and a
  callback for every flag a setter changes.
- `player_stat_get`, `player_stat_set` (v3): level, echoes, insight, the six
  attributes, HP and stamina, and the save's counters, by name.
- `player_position`, `camera_get` (v5), `player_block` (v6): where the player
  and the camera are, and the map block.
- `player_warp` (v6) and `lamp_warp` (v7): move the player within a block or
  travel to a lamp. Only for opt-in gameplay plugins the player has turned on.
- `on_world_load` (v8): called whenever the player is placed in a map.
- `chr_list` (v8): every character the world holds, with its `ChrIns` pointer,
  NpcParam row, model, team, HP and position.
- `sp_effect_apply`, `sp_effect_has` (v8): apply a SpEffectParam row to a
  character the way the game's scripts do.

**Settings, UI and files**

- `bb_plugin_options` (v9): a table of headings, switches, numbers, choices,
  text fields and action buttons. Values are stored under `[<plugin name>]` in
  the per-user `bbhost.toml` and read back with `config("<name>.<key>")`.
- `on_option`, `on_action` (v9): callbacks for changes and button presses made
  in the in-game menu.
- `show_message(text, seconds)` (v8): a banner over the game.
- `plugin_dir(name)` (v8): a data folder of the plugin's own.
- `game_file(path, ...)` (v9): a game file by its `/app0` path, decompressed.
- `overlay_file(name, path, data, size)` (v9): a file the game reads instead
  of its own, compressed again when the path ends in `.dcx`. The randomizer
  rewrites map layouts this way.
- `replace_text(texts, count)` (v11): the game's own text. Every reference to
  each UTF-16 string in the image - the lea instructions that load its address
  and the pointers to it in the image's data - names the new text instead; the
  string itself stays. From `bb_plugin_image` only. The Debug Menu plugin's
  English is made this way. Replace only text the game shows: a string it also
  looks up by name stops matching. The character scripts call the game's
  functions by their Japanese names, and with those names in English every
  character died at load.

**Online rules** (v10)

A gameplay plugin is announced to the server with every request
(`X-BBHost-Ruleset`), so modded runs stay apart from vanilla ones and players
only meet others with the same rules. `set_rules` updates the plugin's entry
(for example when its seed changes); `on_world_rules` and `visitor` let a
plugin flagged `BB_PLUGIN_ADOPTS_RULES` play by a host's rules when summoned
into another player's modded world.

## The C++ SDK

C++ plugins can use `include/bbhost/`:

| Header | Contents |
|---|---|
| `bbhost/sdk.hpp` | `bb::Plugin`: the API with lambdas (`every_frame`, `on_world_load`, `on_event_flag`), typed params, characters, flags, stats, messages, `gravity()`, `call("SYMBOL", args...)`; `bb::Rng` |
| `bbhost/params.hpp` | a struct for every param row type and a tag for every table, generated from the game's own paramdefs, with offsets checked by `static_assert` |
| `bbhost/engine/` | engine layouts and addresses for 1.09: characters, the world, matchmaking and sessions, GX, YEBIS and more |

```cpp
#include "bbhost/sdk.hpp"

static bb::Plugin g;

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (!g.attach(api, 8)) return 1;
    g.every_frame([] {
        // The Saw Cleaver weighs nothing.
        if (auto* w = g.param<bb::params::EquipParamWeapon>(1000000)) w->weight = 0;
    });
    return 0;
}
```

`tests/sdk_headers_test.cpp` compiles every header as part of the test suite.

## Which tool for which change

- A number the game reads from a table (damage, enemy HP, item lots, shops):
  change the param row once the table has loaded. Characters are recomputed
  from their params, so writing a character's own copy does not last.
- Something that should happen now (a buff, a status): apply a SpEffectParam
  row.
- World state (bosses, doors, NPC progress): event flags. A boss fight is a
  family of flags around its defeat flag.
- Moving the player: `player_warp` inside a block, `lamp_warp` between maps.
- Physics and time: `gravity()` and the time scale.
- Anything else: the engine layouts, `hook2` and `call_guest`.

## Official plugins

The plugins in `plugins/` are built with bbhost and shipped in its packages,
each with a `.sig` file: an Ed25519 signature by the release key. bbhost checks
it before loading; a plugin whose signature does not match is refused. The
setup window can download the latest signed plugins from the newest release.

| Plugin | What it does |
|---|---|
| `randomizer` | Shuffles one-time pickups and shop stock from a seed, and optionally enemies: each regular enemy becomes another kind (model, AI and gear together), drawn from its own area or from the whole game, with optional roaming bosses. Writes a spoiler file. |
| `boss_rush` | Fights the 19 story bosses back to back, optionally with the ten chalice-dungeon bosses, starting each round outside the boss's fog wall. Tracks time and deaths. |
| `mutators` | Switchable rules: gravity, game speed, bullet time at low HP, one-hit deaths, blood drain, enemy HP, echo rate, random effects. |
| `debug_menu` | Restores the developers' debug menu (`` ` `` opens it), in English or the developers' Japanese, and makes the debug font it draws with from the public-domain X11 fonts k14 and 7x14, so nothing has to be downloaded. Cheating against other players with it gets an account banned; trying it out online with friends is fine. |

The gameplay plugins - all four - announce themselves in the session's
ruleset, so the server keeps their players' messages, bloodstains and
statistics apart from those of players without them.
