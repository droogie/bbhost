# Trophies

The game's trophy calls are answered by `src/hle/system.cpp`. An unlocked
trophy is appended to `<data>/trophies.txt` (an id and a time), and a popup
slides in from the top left of the display: the trophy's grade, name and icon
(`src/host/trophy_toast.cpp`, drawn with the host overlay).

## Names, grades and icons

These come from the game's own `sce_sys/trophy/trophy00.trp`
(`src/core/trophy_pack.cpp`). The pack's text is AES-128-CBC encrypted with a
key made from the game's NP communication ID (found in `sce_sys/npbind.dat`)
and a 16-byte trophy key that is not in the game files. bbhost bundles no key:
supply your own with either

- the environment variable `BBHOST_TROPHY_KEY=<32 hex digits>`, or
- a file `trophy_key.txt` holding those digits, next to `trophies.txt`.

Without it the popup says only "Trophy unlocked - See docs for full behavior"
(no grade, name or icon), and the log says why (`trophies: no trophy names
(...)`). The same happens when the pack cannot be read for any other reason,
a wrong key or ID say. The same data fills
`sceNpTrophyGetTrophyInfo`, so the game sees the real name, grade and
description.

The icon (`TROPnnn.PNG`, not encrypted) is drawn as a 32x32 mosaic of the
overlay's own triangles, because the overlay's one pipeline only samples its
font. A frame short of vertex room falls back to 16x16, then to the plain
grade-coloured square (`host_overlay_room`).

## Switches

- `BBHOST_TROPHY_TOAST=0` turns the popups off.
- `BBHOST_TROPHY_TOAST_TEST=1` shows a popup for the first trophy of each
  grade the game's pack has (bronze, silver, gold, platinum), one after the
  other, shortly after start, to see the look without earning a trophy. With
  no readable pack it shows the one-line popup.
