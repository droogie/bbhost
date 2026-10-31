# Bloodborne on PC (bbhost) - group playtest

Thanks for helping test. This build plays Bloodborne online on our **dev
server**, `dev.thehuntersdream.com`, together with other bbhost players.
Everything here is a test: things will break, and telling us how is the whole
point.

## What you need

- **Your own copy of Bloodborne, version 1.09, dumped from your own PS4.**
  Nothing from the game is in this package.
  - The **game folder** (`CUSA00900`) with the 1.09 update's files copied
    over it. It contains a folder called `dvdroot_ps4`, and its
    `sce_sys/param.sfo` says version 01.09.
  - The **1.09 eboot**: the update's `eboot.bin`, decrypted. bbhost only runs
    this exact file; any other version stops at start with a message saying
    so (a dump without the update has the 1.00 eboot, which will not work).
- **A graphics card with a Vulkan 1.3 driver.** Update your GPU driver
  before you start.
- **Windows 10 or 11 (64-bit)**, or **64-bit Linux** as new as Ubuntu 24.04,
  Linux Mint 22, Debian 13 or Fedora 39 (Arch and a current SteamOS are fine).
  `ldd --version` must say 2.38 or higher. Linux also needs these packages,
  which most desktops already have:
  - Ubuntu / Mint / Debian: `libvulkan1 mesa-vulkan-drivers libcurl4t64 zlib1g`
  - Fedora: `vulkan-loader mesa-vulkan-drivers libcurl zlib`
  - Arch / SteamOS: `vulkan-icd-loader curl zlib` and your GPU's Vulkan driver
- A controller, or keyboard and mouse.

## Install

1. Unzip (Windows) or untar (Linux) the package into a folder of its own, for
   example `C:\Games\bbhost` or `~/Games/bbhost`.
2. Start it (next section). The first time, the **bbhost launcher** window
   opens. Click the folder button next to *Game folder* and pick your game
   folder (the one that **contains** `dvdroot_ps4`), then the file button next
   to *Eboot* and pick your decrypted 1.09 eboot. Each gets a green check when
   it is right. Leave *Online* on the playtest server (the kit chooses it
   anyway), set anything else you like, and click **Play**.
3. That's it. To change these later, start bbhost with `--setup` (Windows:
   `run-bbhost.bat --setup`), or turn on *Show this window at every start*
   (its Settings page). The settings are kept in your **per-user config**
   (`%APPDATA%\bbhost\bbhost.toml` on Windows, `~/.config/bbhost/bbhost.toml`
   on Linux), which you can also edit by hand.

You do this **once**: every copy of bbhost reads that file, so a new kit or a
new download needs nothing set again. Your F10 settings
and your account are kept beside it, and your saves in
`%LOCALAPPDATA%\bbhost\data` (Windows) or `~/.local/share/bbhost/data`
(Linux). Coming from an earlier kit: unzip the new one over the old kit's
folder and your account and saves come along by themselves; the two paths
are set once more, in the per-user config, and never again.

## Start the game

- **Windows:** double-click **`run-bbhost.bat`**. A black window opens and
  stays open while you play (it says where the log goes), and the game window
  appears. The first time, Windows may say "Windows protected your PC": click
  *More info* then *Run anyway*. When **Windows Defender Firewall** then asks
  whether bbhost may communicate on networks, tick both *Private* and
  *Public* and click **Allow access**: other players reach you on UDP port
  9307, and a blocked port is the most common reason summons fail. (If you
  clicked Cancel, allow `bbhost.exe` under *Windows Security > Firewall &
  network protection > Allow an app through firewall*.)
- **Linux:** open a terminal in the folder and run **`./run-bbhost.sh`**.

Always start it this way, not by opening `bbhost.exe`/`bbhost` directly: the
launcher is what points bbhost at the playtest server (started on its own,
bbhost plays on the live server instead), and it writes the log file we need
when something goes wrong (`logs/bbhost-<date>-<time>.log`).

The first time, areas load slowly and stutter for a moment while the game
prepares its graphics for your card. That goes away on the next visit.

**F10** opens the PC settings at any time: resolution, window mode, V-Sync,
frame cap, mouse, key bindings and your account.

## Link your account (once)

The game stays **offline until this PC is linked to an account**, and your
account name is your name online. You can do it in the **setup window**
before the game starts (it opens the first time; `run-bbhost.bat --setup`
opens it again): the **Account** part under **Online** has the same choices
as the game's. Or, in the game, press **F10** at the title screen and go down
to **ACCOUNT** (the arrow keys move, Enter picks; the mouse works too). Pick
one:

- **Link with Discord** (recommended). A code appears and your browser opens
  `https://dev.thehuntersdream.com/link`. Sign in there with Discord, check
  that the page names *this* PC, and approve. The F10 screen then says you are
  signed in. (If the browser does not open, go to that address yourself and
  type the code.)
- **Create account on this PC.** Select *Name*, press Enter, type the name
  you want (3 to 16 letters, digits, `_` or `-`) and press Enter again. Then
  choose *Create account on this PC*. A **recovery
  code** appears **once**. **Write it down.** It is the only way to get this
  account back on another PC or after reinstalling (*Recover account* takes
  the name and that code). Only 3 of these can be made per internet
  connection per day, so if you are several people in one house, use Discord.

Your sign-in is kept in `bbhost-options-bbhost-playtest.toml` beside your
per-user config. Do not share that file.

Then close F10 and choose **Play Online** at the title.

### Show up on the map and leaderboards (optional)

Go to <https://dev.thehuntersdream.com/account> (sign in with Discord, or use
*Website sign-in code* in F10 to get a one-time code) and tick **Public
profile**. Your deaths and messages appear on the live map either way, as "a
hunter"; a public profile puts your name on them, lists you on the
leaderboards and lets anyone see your `/hunter/<name>` page. Untick it any
time.

## Playing together

- **Co-op:** the host rings the **Beckoning Bell**; the helper rings the
  **Small Resonant Bell** in the same area. Stay in the same area and give it
  a minute. Being summoned counts too: try both roles.
- **Invasions:** ring the **Sinister Resonant Bell** to invade a host who is
  looking for co-op (they have rung the Beckoning Bell) in that area.
- **Messages and bloodstains:** write messages, rate other people's, touch
  bloodstains to watch how someone died.

Agree on an area with your friends in chat first; it makes finding each other
much quicker.

## What to test

Tick off what you managed, and tell us about anything that failed:

- [ ] Link the account (Discord or on this PC) and go online
- [ ] Summon a friend (Beckoning Bell)
- [ ] Be summoned into someone's world (Small Resonant Bell)
- [ ] Beat a boss together in co-op
- [ ] Invade someone, and be invaded
- [ ] Leave messages, and rate other people's
- [ ] See other players' bloodstains and phantoms
- [ ] Die a few times, then check that your deaths show on the live map
      (<https://dev.thehuntersdream.com/map>) within about 2 minutes
- [ ] Look at <https://dev.thehuntersdream.com/stats>,
      <https://dev.thehuntersdream.com/leaderboards> and your own page
      `https://dev.thehuntersdream.com/hunter/<your name>`

## Known limits

- Chalice Dungeons are not on the map.
- Positions on the map arrive 1-2 minutes late.
- This is the dev server: its data (accounts, messages, stats) may be reset.
- Some home or mobile connections cannot connect to each other directly. If
  summons keep failing with one particular person, forwarding **UDP port
  9307** to your PC on your router often fixes it.
- Two PCs in the same house: set `p2p_addr` in `bbhost-playtest.toml` to each
  PC's local address (for example `192.168.1.23`).

## Reporting a problem

Send it to the playtest organiser, with:

1. **When** it happened (time and your time zone) and your **account name**.
2. **What happened** and what you expected, and who else was involved
   (their account names), if anyone.
3. **The log file** from the `logs` folder for that session (the newest one;
   the launcher printed its name). Please send it privately: it contains your
   internet address.
4. Windows or Linux, and your graphics card.
5. A screenshot or clip if you have one.

If the game crashed, the end of the log usually has a block starting
`SIGSEGV pc=`; that block is what we need most. On Windows, if the log just
stops, also look in Event Viewer (Windows Logs > Application) for an
"Application Error" about bbhost.exe and send that too.
