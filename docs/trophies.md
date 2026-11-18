# Local trophies

bbhost records 40 trophies: 34 for the base game, including platinum, and six for The Old Hunters. When the game awards a trophy, bbhost saves it and shows a notification for six seconds. Notifications queue; repeated awards do not change the original timestamp. Platinum unlocks after all 33 other base-game trophies. DLC trophies are not required.

Open **Trophies** in the setup window or the in-game F10 menu. The list shows names, requirements, grades, progress, and unlock times in UTC. Filter by All, Unlocked, or Locked. Hidden trophies conceal their name and requirements until unlocked unless you select **Reveal hidden trophies**.

## Records and compatibility

Records live in `trophies.txt` in your configured data folder and are shared by characters using that folder. Existing bbhost records work unchanged. Choosing another data folder in setup changes the displayed record.

Before the first update to an existing file, bbhost makes `trophies.txt.pre-trophies.bak` without overwriting an earlier backup. New contents are flushed to a temporary file and replace the record only after the write succeeds. A failed write does not award or announce the trophy; check free space and permissions.

If an existing record already contains all 33 base-game trophies, trophy registration at the next game start reconciles platinum. Earlier trophies cannot be reconstructed from a game save alone.

These are local records, with no PlayStation Network or Steam achievement sync. Importing other ports' trophy formats is outside this change. Game files and saves are not modified.

## Implementation references

The factual catalog and API layout references were [ayozetr/bloodborne_linux](https://github.com/ayozetr/bloodborne_linux) at `804651e418dc1e80b1dafdc83402106bbea954cd`:

- `scripts/trophies.py` and `src/runtime_services.c`: IDs, grades, hidden flags, DLC grouping, and platinum conditions.
- Its shadPS4-derived `np_trophy.h` and `np_error.h`: structure layouts and error values (GPL-2.0-or-later, shadPS4 Emulator Project).

Persistence, HLE handling, UI, and descriptive text are implemented for bbhost. No game images or keys are included; the notification cup is drawn with overlay primitives.
