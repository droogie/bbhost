# Running bbhost

## What you need

bbhost contains no game data. You bring:

1. **The game files**: a dump of Bloodborne with the 1.09 update's files copied
   over it - the folder that contains `dvdroot_ps4`. Its
   `sce_sys/param.sfo` says `APP_VER 01.09`. The update can also stay in a
   folder of its own beside the game's, named after it with `-UPDATE` (or
   `-patch`) appended - `CUSA00900` and `CUSA00900-UPDATE`, the way shadPS4
   keeps a game library. bbhost then reads every file the update has from
   there and the rest from the game folder, as the console applies an update;
   the log names the folder (`FS update=`).
2. **The 1.09 eboot**, decrypted to an ELF. Its SHA-256 must be
   `941f887a562aae054fac35af8cc8f27cf075f3d4cc2e029fb5ae2a663aaa5ae7`. Every
   patch, hook and engine layout in bbhost is an address in that build, so any
   other eboot stops at start with its hash and the version its dump reports.
   A dump without the update carries the 1.00 eboot, which will not do. When
   `paths.eboot` is not set, the game's own `eboot.bin` is used if it is
   decrypted already (a dump made for shadPS4), the update folder's first.

Both must come from a copy of the game you own.

And a PC with:

- x86-64 Windows 10/11 or Linux
- a GPU with a Vulkan 1.3 driver (recent NVIDIA, AMD or Intel; the Steam Deck works)
- 16 GB of RAM and 8 GB of video memory are recommended

## First run

Started without a configuration, bbhost opens its setup window: pick the game
folder and the eboot (both are checked as you go), choose the online server
and account if you want to play online, and press Play. The window writes the
per-user `bbhost.toml` and starts the game. `bbhost --setup` opens it again.

Without a display (a headless run) bbhost writes a template config instead and
tells you which paths to fill in.

## Configuration

`bbhost.example.toml` lists every setting with its default. The configuration
is read in layers, each overriding only the keys it sets:

1. the per-user `bbhost.toml`: `%APPDATA%\bbhost` on Windows,
   `$XDG_CONFIG_HOME/bbhost` or `~/.config/bbhost` on Linux
   (`BBHOST_CONFIG_DIR` replaces the folder);
2. `bbhost.toml` in the working directory, or next to the executable;
3. `--config <file>`;
4. the command line: `--app0 <dir>`, `--data <dir>`, `--eboot <file>`, or the
   eboot as the last argument.

The first log line names every file it read. Paths may use either slash; a
relative path is taken from the folder of the file it is in.

The in-game settings (F10, or the PC Settings, PC Graphics and Key Bindings
entries the game's own System menu gains) are saved beside the per-user config
in `bbhost-options.toml`.

## Saves

Saves are in `<data>/saves/SPRJ0005`, where `<data>` defaults to
`%LOCALAPPDATA%\bbhost\data` or `~/.local/share/bbhost/data`. Each time the
game writes a save (at most every five minutes) a copy goes to
`<data>/save-backups/`, and the newest eight are kept. To restore one, quit and
copy its files over `<data>/saves/SPRJ0005`.

The shader caches in `<data>/bbhost` (`translation-cache.bin`,
`vulkan-pipeline-cache.bin`) belong to the shader translator that made them,
which `shader-cache-build.txt` names. An update that leaves the translator
alone keeps them, so it starts as fast as the build before it; one that
changed the translator throws both away (`gpu: shader caches made by
translator ..., this is translator ...: cleared`) and builds them again while
the title and the loading screens are up, from `stage-manifest.bin`, the list
of the game's shaders, which is kept. `BBHOST_KEEP_SHADER_CACHE=1` keeps them
anyway. The files are safe to delete: the next start builds them again.

## Controls

A controller works as on the console. Without one, the keyboard and mouse
take over and the button prompts follow: the mouse turns the camera and works
in the menus, and every action can be rebound (F10 or the game's Key Bindings
menu) - to a key, any of five mouse buttons, or the wheel turned up or down
(each notch a press). Hold Left Alt with the movement keys to walk, and two
opposite keys held together cancel, both as in Dark Souls III. F9 opens the
plugin menu and F10 bbhost's own settings.

The mouse turns the camera the way Dark Souls III's PC version does: by an
angle for each count the mouse reports, 0.25 degrees at the default
sensitivity of 5 (0.1 at 0, 0.4 at 10), with no acceleration, no dead zone
and no top speed, at any frame rate. The right stick turns it as on the
console, and while the stick is moving the mouse waits. With a target locked
on, a quick flick of the mouse switches targets. Mouse sensitivity and
inverting either axis are on the game's PC Controls page; the game's own
Camera settings are the stick's.

With the mouse, the camera also stays where you put it: it does not turn by
itself as the character moves - following the character round, or turning
away from walls. Lock On with no target (Q) still swings it round behind the
character. That follows the device you used last: pick up a controller and
the game's own camera is back; touch the keyboard or mouse and it holds
again. Mouse Auto-Rotation on the PC Camera page (F10: Mouse auto-rotation)
turns it back on for the mouse too. It is the community's "Disable Camera
Auto Rotation via Movement" patch made part of the camera - less its hold on
the lock-on button's recentre; that patch, applied as before, still works as
it always did, with any device.

The character's name, a chalice glyph and the network password are typed in
a box over the game, which stands in for the console's on-screen keyboard:
Enter accepts, Escape cancels, Backspace deletes and Ctrl+V pastes. A
controller's Cross (A) accepts and Circle (B) cancels. The boxes take what the
console's keyboard offers - letters, digits and punctuation from plain ASCII -
and a glyph takes its own 32 characters, lowercased as they are typed. The
name box starts from `[player] name`, which the first key typed replaces;
`[player] ime = "auto"` answers it with that name instead.

## Logs

bbhost logs to standard error. The packages' `run-bbhost.bat` (Windows) and
`run-bbhost.sh` (Linux) write each session's log to `logs/`. A bug report is
most useful with that log attached.

## Troubleshooting

**The eboot is refused.** It is not the 1.09 build. Use the update's
`eboot.bin`, decrypted; the message says what the dump's `param.sfo` reports.

**The game stops at start in `FileTransferTask.cpp(865)`, or the log says
`patch: old-hunters skipped`.** The game folder holds files older than the
1.09 update's: the update was never copied over it, or only its new files were
(Windows' "Skip these files" when it asks about the ones already there). The
log's `change appearance: off - not the 1.09 m21_00_00_00 layout` says the
same. Copy the update's files over the game folder again, replacing the ones
there, or keep the update in a `-UPDATE` folder beside it (above). Until The
Old Hunters' own files are there, bbhost leaves the game as the edition
without it: its title screen would otherwise be a flat green picture.

**The graphics card stops responding.** When the GPU hangs and the system
resets it, nothing can be drawn again in that session; bbhost stops the sound,
says so in a box and exits. The next three starts run with the GPU's progress
markers enabled, so the log of a second hang names the draw that hung - send
that log. Overlays that hook Vulkan are a common cause: OBS's game capture is
kept out automatically on AMD cards (`[video] obs_capture` changes that), and
OBS's Window Capture records bbhost without any hook.

**Wrong picture.** Wrong colours, a green or black screen, missing parts:
press F12 in the game while it shows. bbhost writes that frame, every render
target the last draws wrote and the draw list into a folder of its own beside
the logs, `logs/f12-<date>-<time>/` (`build/` when there is no `logs/`
folder, `BBHOST_CAPTURE_DIR` to choose), and the log names the folder. Zip it
and send it with the log; `tools/f12_check.py <folder> <log>` checks it for
the rendering bugs found so far. The images are PPM, which costs nothing to
write; `BBHOST_F12_PNG=1` writes PNG instead, which any image viewer opens, a
fifth of the size on disk, written over a few seconds in the background.
`BBHOST_F12_TEXTURES=1` adds every texture the frame sampled, and the game
pauses while it writes them.

**Black or broken image in the world.** On cards with little video memory,
check the log for `video memory is full`. bbhost sizes its own buffers to the
card's budget and falls back to system memory rather than failing, but other
programs holding video memory reduce what is left.

**AMD on Linux, no picture.** AMD's kernel driver only imports anonymous
memory, so bbhost passes the game's memory to the GPU through `/dev/udmabuf`.
If that device is not accessible to your user, the log says so; this udev rule
fixes it:

```
# /etc/udev/rules.d/70-udmabuf.rules
KERNEL=="udmabuf", TAG+="uaccess"
```

**AMD on Windows, slower in a window.** Play in fullscreen (Fullscreen in PC
Settings, or Window mode in F10). In a window the desktop composites the
picture, and AMD's driver then never completes the game's occlusion queries -
the same as when the window is not focused - so the game, getting no answers,
stops culling hidden geometry. On a Radeon 8060S at 1920x1080 that is 112 fps
uncapped in a window against 161 in fullscreen, and at 60 fps the GPU 55% busy
against 37%. The log's `occlusion:` line, every 300 flips, then says `0 dumps`
or `left not ready (N queries unavailable)`.

**Steam Deck.** Start bbhost from Steam (as a non-Steam game), in Game Mode or
Desktop Mode; otherwise Steam's desktop controls send keys alongside the pad
and every press acts twice. `tools/steamdeck/STEAMDECK.md` has the setup.

## Display and frame rate

The picture keeps its aspect ratio, with ultrawide resolutions (21:9, 32:9)
available in the settings. The resolution can be changed while playing.
bbhost paces the game to the refresh rate of the display its window is on;
`[video] vblank_hz` overrides a display that reports the wrong rate.

The frame rate is the game's own speed, chosen when it starts (a change
applies on the next run): 30 is the game as it shipped; 60 and 90 run its
logic at that rate with its fixed steps converted (Kyo's 60 and 90 FPS++
lists, `engine/frame_rate.cpp`); Uncapped steps it by the measured frame time
(1/240 to 1/30 s), at the display's refresh rate with V-Sync on and up to 240
fps with it off. At 90 and Uncapped the game paces itself and its frames are
shown as they come, so 90 on a 60 Hz display shows 60 of them a second.
`BBHOST_GAME_FPS=30|60|90|0` overrides the setting for a run (0 is uncapped),
`BBHOST_UNCAPPED_FPS=N` sets the uncapped limit (30-240) whatever V-Sync says,
and `BBHOST_UNCAPPED_LOG=1` logs once a second the frame time uncapped steps
by and the values it wrote. The start of the log says which rate runs and how
many values and code sites were converted (`frame rate:` lines), and names
any site whose bytes were not the expected ones (that rate is then refused:
90 or uncapped falls back to 60).
A frame goes to the screen as soon as the game has finished it; only its flip,
which the game paces itself on, waits for the next tick of that clock. The game
keeps two flips queued, and showing each frame at the tick that completed it
held every frame two refreshes behind (`BBHOST_PRESENT_ON_ARRIVAL=0` brings that
back, to compare). Every 300 flips the log's `pacing:` line says how late the
clock's ticks woke and how long flips took from queued to shown and to
completed.

## Online

bbhost implements the PlayStation Network calls the game makes against a
private server, so co-op, invasions, messages and bloodstains work between
bbhost players. The setup window picks the server and links an account; see
[online.md](online.md).

## The debug menu

The developers' debug menu is still in the game, switched off. The **Debug
Menu** plugin restores it (the community's "Restore Debug Menu" patch by
Whitehawkx and auser1337): turn it on in the setup window's Plugins tab and
start the game again. The `` ` `` key then opens and closes the menu (Key
Bindings can move it). While it is open the character stands still; the arrows
move, Enter opens an entry and Backspace goes back. Its text is in English; the
plugin's **Menu text** setting brings back the developers' Japanese.

The English is `plugins/debug_menu/strings_en.tsv`: each string's address in the
1.09 eboot, a hash of the game's text there, and the English - no game text.
`tools/debug_menu_strings.py list` shows the Japanese beside it from your own
eboot, `missing` what has no English yet, and `check` tests every row (the
hash, the printf conversions, the font). Corrections are welcome as pull
requests. A few strings keep the developers' text because the game looks them
up by name - the names the character scripts call the game's functions by, for
one - and `check` refuses a row for any of them.

The menu draws with a debug font the game does not ship. The plugin makes one
from the public-domain X11 fonts k14 and 7x14, so nothing has to be
downloaded. A `DbgFont14h.ccm` and `.tpf` of your own in the mods folder
(`<mods>/dvdroot_ps4/adhoc/font/`; [modding.md](modding.md)) are used instead -
`tools/build_debug_assets.sh` puts the community "Debug Menu Restoration"
package's original font there.

**Playing online:** the menu can hand out items and change the rules, so
sessions with the plugin on keep their messages, bloodstains and statistics
apart from other players'. Using it to cheat against other players - in their
worlds, in invasions or on the leaderboards - gets an account banned. Trying it
out online with friends is fine.

## Windows crash capture

For opt-in triage and full-memory dumps, see [Windows crash dumps](windows-crash-dumps.md).
