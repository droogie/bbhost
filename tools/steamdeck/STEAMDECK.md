# bbhost on the Steam Deck

This is the Linux build of bbhost with settings for the Deck: it renders at
the screen's own 1280x800, full screen, at the game's 30 fps
(`bbhost-steamdeck.toml`). It runs natively on SteamOS; nothing needs
installing. You bring the game: the CUSA00900 dump with the 1.09 update's
files over it, and the 1.09 `eboot.bin` decrypted to an ELF (README-linux.txt
says how to check both).

## Set it up (Desktop Mode, once)

1. Switch to Desktop Mode (Steam button, Power, Switch to Desktop).
2. Copy the game dump and the decrypted eboot onto the Deck - the internal
   storage or the SD card (`/run/media/...`) both work.
3. Extract this archive into your home folder, for example `~/Games`
   (right-click it in Dolphin, Extract, Extract archive here).
4. Add it to Steam: open Steam, Games, "Add a Non-Steam Game to My
   Library...", Browse, set the file type to All Files, pick `run-bbhost.sh`
   in the extracted folder, Add Selected Programs. Its Properties can rename
   it (Bloodborne, say).
5. Start it from Steam (Library, Non-Steam) - here in Desktop Mode too. The
   first start opens the setup window (the touchscreen or the right trackpad
   work it): pick the game folder (the one with `dvdroot_ps4` in it) and the
   eboot, then Play. Check that the title screen comes up, then quit.

Always start it from Steam, not by double-clicking `run-bbhost.sh` or from
Konsole. Outside Steam, Steam's desktop controls stay on: A, B and Y also
send Return, Escape and Space, the d-pad the arrow keys, the triggers mouse
clicks - so every button does two things, and the menus open and close on
their own. bbhost says so on screen when it sees it.

## Play (Game Mode)

Switch back to Game Mode and start it from your Library (Non-Steam). The
Deck's controls are the game's pad. If the sticks or buttons act as a mouse
and keyboard instead, pick the Gamepad layout in its controller settings
(Steam button, Controller Settings).

- The game's System menu has a **Steam Deck** section, on a Deck only, and
  the PC sections (PC Graphics: resolution, PC Settings: frame cap and
  more). F10 opens the host's own options screen (account, updates, keys);
  without a keyboard, give F10 to one of the back buttons in the controller
  settings.
- These settings and the account sign-in are kept for this kit alone, so a
  copy of bbhost on a PC keeps its own; the game's paths are shared.
- The first minutes in a new area compile shaders and can stutter; they are
  kept, so the next start is smoother. To compile the game's shaders before
  playing (after an update, say), start bbhost with the setup window
  (`--setup`, or in Desktop Mode), open its Shaders tab and press Compile
  shaders now; the window recommends it when the caches are missing or out
  of date (about 2-3 minutes on the Deck). It can be cancelled; what is
  done is kept. Without it, the game's HLSL shaders are compiled in the
  background only while the title and loading screens are up (on the Deck,
  compiling during play cost ~2 fps), and are used from the start after.
- **30 or 60 fps**: System, Steam Deck, Frame Rate. 30 fps renders at the
  screen's own 1280x800, the game's own pace and the Deck's default here.
  60 fps renders at 960x600 and bbhost scales the picture up to the screen
  with AMD's FSR 1, which keeps edges sharp, with Model Detail at Low (it
  draws distant buildings one step coarser, which is hard to see on this
  screen): the heaviest views found, like the first gate in Central
  Yharnam, hold 60. The resolution and the model detail change at once; the
  frame rate itself on the next start. The Deck's GPU cannot draw the game
  at its full 1280x800 sixty times a second (about 45). The same section's
  Model Detail can be set on its own (Normal at 60 fps holds 60 walking,
  about 58.5 at the heaviest views), and PC Graphics' Resolution offers
  1024x640, a little sharper and a little less steady. In Game Mode set the screen to 60 Hz in the performance
  overlay (... button, Performance, Refresh Rate), so every frame is shown
  for the same time; at 90 Hz 60 fps cannot be paced evenly. Turning off
  Ambient occlusion, Motion blur and Depth of field (PC Effects) gives a
  little more headroom. 30 fps at 1280x800 is the game's own pace and the
  Deck's default here.
- The Deck's memory is shared with its GPU and tight for this game, so
  bbhost keeps less in reserve here: resolutions up to 1920x1080 change at
  once; larger ones (for a big screen on a dock) apply at the next start,
  and may not fit - they need a gigabyte or more on top, and have not been
  tried on a Deck.
- F10 (UPDATES) says when a newer bbhost is out, and Open the release page
  opens its page in the browser. To update, extract the new kit and copy its
  files over this folder.

## If something goes wrong

Every start writes a log to `logs/` in this folder (the newest ten are
kept). A crash leaves a block starting with `SIGSEGV pc=` near its end: send
that log with the build name in README-linux.txt.
