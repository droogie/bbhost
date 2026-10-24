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
- 16 GB of RAM and 8 GB of video memory are recommended; 6 GB NVIDIA cards
  play at 1080p too

## First run

Started without a configuration, bbhost opens its setup window: pick the game
folder and the eboot (both are checked as you go), choose the online server
and account if you want to play online, and press Play. The window writes the
per-user `bbhost.toml` and starts the game. `bbhost --setup` opens it again.

The game turns its shaders into HLSL and compiles them on your PC in the
background, so you can play straight away; the first session uses the
translator's own shaders while that runs, and may stutter a little more than
later ones the first time it meets each place. To compile before playing -
after installing or updating bbhost or your graphics driver - open the setup
window's Shaders tab and press **Compile shaders now**: every shader of the
game's, the pipelines your earlier sessions drew and those on the list a
release ships (`pipeline-list.bin`, below), then the first session
needs no compiling in the background. (The few pipelines a place needs the
first time you ever visit it are still built then.) The tab shows whether the
shader caches are up to date for this bbhost and this driver, and the
compile's progress, time left and any failures; it can be cancelled, and what
was compiled is kept. When the caches are missing or out of date - a new
install, a bbhost update that changed the shader translator, a new graphics
driver - the tab, and a line on the main tab, recommend it with the time it
is likely to take; it is never forced. The same tab has the Shaders setting
(HLSL, or Lifted for the translator's own shaders only; also F10 under
Graphics, and HLSL Shaders in the game's System > PC Graphics).
`bbhost --compile-shaders` does the same from a terminal.

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
the title and the loading screens are up, from the two lists beside them,
which are kept: `stage-manifest.bin`, the shaders the game's draws needed that
it never creates itself, and `pipeline-manifest.bin`, every graphics pipeline
a draw created. At each start bbhost builds every pipeline on that list again
in the background, its shaders, its pipeline libraries and its optimized
link, so what an earlier run has drawn is not compiled again when it comes
into view. The lists hold what the game drew, not what bbhost made of it, so
they outlast translator updates and driver changes: what they build is made
again from them. A list `pipeline-list.bin` beside `bbhost` (or the one
`BBHOST_PIPELINE_LIST` names) adds its pipelines to a player's own, so a first
start has them too; such a list names the game's shaders, the player's own
bundles supply them. `BBHOST_KEEP_SHADER_CACHE=1` keeps the caches anyway,
`BBHOST_PIPELINE_MANIFEST=0` leaves the pipeline list out. The files are safe
to delete: the next start builds them again.

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

The game's own prints go to the console too. A line it prints over and over
(YEBIS's depth-of-field error in parts of Cainhurst comes every frame) is
written once, then as a count every ten seconds while it keeps coming and at
exit: "the game printed this N more times". `BBHOST_GUEST_PRINTS=all` writes
every line as it comes.

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

**A crash just after arriving somewhere, with `fmod ring:` in the log.** FMOD,
the game's sound library, lists every sound it has loaded, and it frees the one
subsound object a stream of many sounds (a voice bank) shares even when that
object refused to be released because an event still plays it - leaving it in
the list. The next sound loaded beside it then writes into memory the game has
used again since. bbhost takes such a sound out of the list first and keeps
it, marked so that nothing plays it again; the log's `fmod ring:` lines say
when it did, and the one at exit how many it kept (a few hundred bytes each
that are never freed). `BBHOST_FMOD_GUARD=0` is the game's own behaviour;
`BBHOST_FMOD_CHECK=1` also logs FMOD's sounds as they come and go and checks
the list as sounds are loaded.

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
pauses while it writes them. `BBHOST_F12_NAN=1` paints every texel of a
float target with a NaN channel magenta, so a black blob that is a NaN shows
where it is; the log counts each float target's NaN and infinite texels
either way.

**Black or broken image in the world, or slow on a card with little video
memory.** bbhost follows the driver's video memory budget as the game runs:
it keeps the last 512 MiB for render targets (less on a card with under
6 GB free: a twelfth of what is free at the start, at least 256 MiB), puts
textures in system memory when the card is down to that, and moves them back
once there is room. Every
five seconds the log says `video memory: headroom ...`; look there, and for
`video memory is full`, before anything else. Other programs holding video
memory (browsers, OBS, a second game) shrink the budget. On a GPU with under
7.5 GB free bbhost also holds less memory ahead (the log says `small video
memory`). `BBHOST_VRAM_RESERVE_MB=0` goes back to placing everything until
an allocation fails, `BBHOST_SMALL_VRAM=0` turns the small mode off.

Every 1,800 flips (`BBHOST_VRAM_REPORT_FLIPS=<n>` changes it) the log also
says where the video memory goes: `gpu: video memory (MiB)`
gives what the driver counts against its budget, what bbhost's own
allocations hold and the driver's own share; `video memory held` lists what
the game needs now (render targets, textures in use, buffers by owner) and
`video memory caches` what bbhost keeps because the card has room (the
vertex data's mirrors, textures not used for a while, free room in the image
heap), which it gives back when memory runs short. At exit the log gives each
owner's highest. Render targets never go in the textures' blocks, so that a
block of textures can empty and be given back; a texture still takes free
room between the targets before a new block is made
(`BBHOST_HEAP_CLASSES=strict` keeps the two apart; `BBHOST_HEAP_CLASSES=0`
and `BBHOST_HEAP_DRAIN=0` turn the split and the draining off);
`BBHOST_TESS_RING=whole` and `BBHOST_SCRATCH_RETURN=0` keep the tessellation
ring and the texture upload scratch at their largest from the start. On a GPU
without sparse binding (AMD, integrated GPUs) the vertex data's mirrors are
made 64 MiB at a time as the game's meshes need them (4 MiB on an
integrated GPU; `BBHOST_SHADOW_WINDOW_MB` sets it), rather than a whole
gigabyte each (`BBHOST_SHADOW_WINDOWS=0` goes back to whole ones). On an
integrated GPU, where those mirrors would be the same memory again, the
meshes in the game's own vertex and index buffers are read where they are,
with no copy (`BBHOST_SHADOW_IN_PLACE=0` copies them as before; `=1` reads
them in place on any GPU), and vertex data left in host memory is not copied
for each draw either (`BBHOST_VERTEX_STAGE=1` copies it as on a discrete GPU).
With window imports (the default on an integrated GPU under Windows;
`BBHOST_IMPORT_WINDOWS` or `import_windows` under `[gpu]` decide it; see
[rendering.md](rendering.md)) the meshes are read in place too, from their
windows, which a thread of their own imports as the game makes its buffers
(`BBHOST_SHADOW_IN_PLACE_WINDOWS=0` copies them into the mirrors instead, as
window imports did before; `BBHOST_IMPORT_PREFETCH_MESHES=0` imports their
windows at their first draw).
An integrated GPU keeps no reserve, so nothing trims what bbhost keeps in
play there: the textures made ahead of their first bind (most are never
bound), textures not used for a while, free room in the image heap and the
upload staging held from a load on. Three switches give them back, on any GPU;
all three are on by default on an integrated GPU under Windows (a 2048 MiB
budget there) and off elsewhere:
- `BBHOST_STAGING_RETURN=1` (or `staging_return = true` under `[gpu]`): the
  upload staging goes back down to 256 MiB in play, 64 MiB every two seconds
  from 30 s after the latest load (a death or a lamp starts the count again).
  The log gives each free's time and flip, and the exit line `given back in
  play` their total.
- `BBHOST_CACHE_BUDGET_MB=<n>` (or `cache_budget_mb = <n>` under `[gpu]`): a
  budget for the video memory bbhost holds. Over it, textures are made at their
  first bind instead of ahead, those not bound for 20 s go, the textures'
  blocks drain sooner and empty blocks go back at once. The exit line `made by
  a bind` counts the textures a bind then had to make itself.
- `BBHOST_HEAP_LEAN=1` (or `heap_lean = true` under `[gpu]`): the image heap
  keeps no spare block in play and drains the textures' blocks sooner.

On other GPUs, unless memory is tight or video memory small, the game's own
immutable vertex and index buffers are bound from copies of their own, made
in video memory as they are first drawn, so the command processor does not
look each draw's bytes up; one that finds no room keeps the mirrors
(`BBHOST_GXVK_BUFFERS=0` goes back to the mirrors for all of them; `=1` makes
the copies on an integrated GPU, with tight memory or small video memory
too).

**Task Manager shows more GPU memory than shadPS4.** Task Manager's GPU
columns for a process count two things: *dedicated GPU memory* is the card's
video memory, *shared GPU memory* is system memory the GPU can reach. bbhost
lets the GPU read the game's own memory where it lies, as the PS4's single
memory does: the game's GPU-visible memory (about 4 GB in the world) is
imported into Vulkan, and Windows counts every imported page as shared GPU
memory, although it is the game's memory, which the process holds either way,
not a second copy. Upload staging (memory the CPU fills for the GPU to copy
from, up to about 0.8 GB) counts there too. shadPS4 copies what the GPU uses
into buffers and images of its own instead, so its shared column stays small.
Measured at the same place (Central Yharnam, 1080p) on one laptop: with its
RTX 3070 Ti Laptop GPU, shadPS4 0.19 showed 2.7 GB dedicated and 0.1 GB
shared, bbhost 3.0 GB dedicated and 4.8 GB shared. On an integrated GPU
Windows counts all of it as shared: on the same laptop's Radeon 680M shadPS4
showed 2.9 GB and bbhost 7.8 GB. To compare the two, read the dedicated
column on a card of its own, or turn on the Memory counter (F10), which shows
bbhost's own count under the FPS counter: the video memory in use, how much of
it is cache that bbhost gives back when memory runs short, and the shared
memory. With window imports (`import_windows = true` under `[gpu]`, the
default on an integrated GPU under Windows) only the memory the GPU uses is
imported: on that laptop the shared column fell by
about 2.7 GB on the RTX and 2.1 GB on the 680M, at the cost of waits when a
place is first drawn (the log's `imports waited for`).

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
applies on the next run): 30 is the game as it shipped; 60, 90 and 120 run its
logic at that rate; Uncapped steps it by the measured frame time (1/240 to 1/30
s), at the display's refresh rate with V-Sync on and up to 240 fps with it off -
in the world and in character creation: at the title and the main menu V-Sync
off sets no limit.
Above 30 the game's step is the frame's measured time (at most 1/30 s), and the
places that stepped a fixed 1/30 s a frame - physics, enemy AI, ragdolls,
grass, the camera's easing, momentum, fades - follow it, so the game runs at
the same speed at every rate (`engine/frame_rate.cpp`, from Kyo's FPS++ lists
and yogesh1239's high-FPS fixes). The play time adds each frame's milliseconds
with the fractions kept (the game drops them: its clock ran 4% slow at 60 and
120). The effects - fire, lamps', candles' and braziers' flicker, blood,
magic, every effect the game's effect manager runs - step in whole 1/30 s
above 30, as many times a second as at 30 (every other frame at 60, every
fourth at 120), the flicker keeping the cadence the game has at 30. The wind
that sways grass and trees moves every frame at 30's pace: the game steps it a
fixed 1/60 s each 1/30 s frame, so above 30 it takes half each frame's step,
and its bend keeps the game's weight of 1/60 (`engine/wind_pace.h`;
`BBHOST_WIND_PACE=0` steps it by the frame's time with a weight of 1/30, twice
30's pace and bend). The frames between whole steps still
run the effects' update, so effects follow what they ride on and are drawn
every frame, but with what the game counts in updates held for the next
whole step - a node's state, its emissions and the definitions its
conditions pick, the controllers, a light's flicker phase, a wobbling
motion's turns and its lag toward what it follows
(`decomp/sfx/step_gate.cpp`; `BBHOST_SFX_GATE=0` lets them count, as when
Ebrietas's attacks left sparks that never ended at 60, and `=1` holds them
with 60 too). The wind's warm-up after a load (the grass and trees settling)
counts whole steps too (`BBHOST_SFX_WIND_WARMUP_HOLD=0` counts frames as
before). By default (`[video] effect_steps = "smooth"`) the effects keep
30's whole steps for what they count and move what they move by time - a
particle's path, a motion's turn, the GPU particles' clock, a ribbon's head
(its points still inserted once a whole step), the wind sources' gusts - by
each frame's own time, in the effects whose every part has been converted
for it; the rest run as at 30. `effect_steps = 30` runs them all on whole
1/30 s steps, as before; `60` steps the effects in whole 1/60 s instead (60
updates a second above 60); `frame` steps them by the frame's time. A
bbhost.toml that sets `effect_steps` keeps its value; one without it gets
smooth. `BBHOST_SFX_STEP=30|smooth|60|frame` overrides the setting, and
`BBHOST_SFX_SMOOTH=0|1` turns the smooth steps off or on for a test run. An
effect's emissions - the particles and children its nodes make - follow a
clock of their own (`[video] effect_emission`, `BBHOST_SFX_EMISSION`): with
`60`, the default, they come as the game's effects make them at 60 updates a
second, every 1/60 s of the node's time at every frame rate, 30 included -
with the smooth steps each on the frame its time falls in (a node's children
on its whole steps' frames), otherwise on the whole step's frame; the rest of
what the effects count keeps 30's steps. Most of the game's emitters (8,421 of its 15,330 CPU particle
records) ask for an emission every 1/60 s, and an update makes at most one,
so at 30 updates a second the game makes half of them - the PS4's look,
which `effect_emission = 30` keeps (`decomp/sfx/node_emit.cpp`). The wakes
characters leave in water (GX's water interaction, one wave step a frame) step
at 30 a second of game time above 30 too, on the effects' whole steps, so they
spread as fast as at 30; each step takes the latest frame's wakes, as at 30,
and the water shows the last step between steps
(`decomp/gx/water_gate.cpp`; `BBHOST_WATER_GATE=0` steps them every frame,
twice as fast at 60). The flicker of the torches, braziers and candles (each
GX light's blink) begins its cycles on the effects' whole steps, holding the
light's high on the frames between, so the lights blink as fast as at 30
(`decomp/gx/blink.cpp`; `BBHOST_BLINK_HOLD=0` begins them on any frame, 5-7%
faster above 30). Above 60 and
Uncapped the game paces itself and its frames are shown as they come, so 90
on a 60 Hz display shows 60 of them a second; a rate the machine cannot hold
runs slower than real time in the few places that count frames.
The follow camera runs the game's own values at every frame rate, converted
to each frame's length, so above 30 it moves as it does at 30.
`BBHOST_CAMERA_TUNING=fps++` runs Kyo's 60 FPS++ camera retune instead.
`BBHOST_GAME_FPS=30|60|90|120|0` overrides the setting for a run (0 is
uncapped), `BBHOST_UNCAPPED_FPS=N` sets the uncapped limit (30-240) whatever
V-Sync says, the menus included, and `BBHOST_FRAME_RATE_LOG=1` logs once a second the frame time the
game steps by and values written from it. `BBHOST_TEST_FPS_SWING` (a test
switch, at 60, 90, 120 or uncapped) moves the target while the game runs:
`square:LO:HI:S` alternates LO and HI fps every S seconds, `random:LO:HI:S` picks
a new rate in that range every S seconds, `jitter:LO:HI` every frame, and
`sine:LO:HI:S` sweeps it - one process at two rates, or at one that never holds
still, as a variable-refresh display or a busy scene makes it.
`BBHOST_FIXED_TICK=<address>:<bytes>[,...]` (a test switch, same rates) runs
a game function as a 30 Hz tick - once every other frame at 60, and so on, with
the remainder carried - for logic that counts frames (`engine/frame_tick.h`). `BBHOST_SFX_SIM_FPS=N`
(a test switch, 30-1000, with effect steps 30 or 60) lays out the effects'
whole steps as at N fps whatever the frame's time - the carry grows 1/N a
frame - so a renderer far below N (lavapipe) runs the cadence N has, 1/30, 0,
1/30, 0 ... at 60 with whole 1/30 s steps; the rest of the game keeps the
frame's time (1/30 s a frame below 30 fps), so the effects run N/30 times
slower than it. `BBHOST_SIM_FRAME_TIME=1` (a test switch, same rates) does
the same for the whole game: each frame counts as its target's length (1/60 at
60, the swing's when `BBHOST_TEST_FPS_SWING` moves it), so every clock the game
steps by - the task step, physics, the effects' schedule, the wind - advances
as at that rate however long the frame took, and a slow renderer runs the
game in slow motion at the rate's steps. The start of the log says which rate
runs and how many values and code sites were converted (`frame rate:` lines),
and names any site whose bytes were not the expected ones (that rate is then
refused: 90, 120 or uncapped falls back to 60).
A frame goes to the screen as soon as the game has finished it; only its flip,
which the game paces itself on, waits for the next tick of that clock. The game
keeps two flips queued, and showing each frame at the tick that completed it
held every frame two refreshes behind (`BBHOST_PRESENT_ON_ARRIVAL=0` brings that
back, to compare). Every 300 flips the log's `pacing:` line says how late the
clock's ticks woke and how long flips took from queued to shown and to
completed.
A flip completes - and what the frame used goes back to the game - only once
the GPU has rendered that frame: on the console a flip is the GPU reaching it,
and the game writes the next frames' data over what a completed frame read. A
loading screen's flips, and those above 60 and Uncapped, do not wait for the
clock, so they complete as soon as their frame has rendered; the log says how
many of a loading screen's flips waited (`video: the loading screen's ...
flips`). `BBHOST_FLIP_AFTER_GPU=0` completes a flip once the frame is recorded,
as before: the game could then run several frames ahead of the GPU after a
loading screen and overwrite what it had not read yet, which could hold the
GPU for minutes.

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
