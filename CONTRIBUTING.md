# Contributing

## Ground rules

- **No game files.** No eboot, no game dump, no assets extracted from the
  game, no saves. Everyone brings their own 1.09 dump and decrypted eboot.
- **Your own code.** Other projects (shadPS4 in particular) are useful as a
  reference for facts - enum values, error codes, register layouts, how a
  working run behaves - but bbhost is its own implementation; don't paste
  their code.
- **No secrets.** Tokens, passwords, keys and `bbhost-options.toml` (it holds
  an account token) stay out of commits.

## Setup

1. Build it ([docs/building.md](docs/building.md)).
2. Write a config with your paths: `./build/bbhost --write-config bbhost.toml`
   (ignored by git), then set `app0` and `eboot`.
3. `export BBHOST_APP0=/path/to/CUSA00900` so the tests that read game files
   run instead of skipping.

## Workflow

- `master` always builds. Work on a branch named for the change
  (`render/...`, `perf/...`, `fix/...`) and open a pull request.
- Rebase on `origin/master` before opening it; keep history linear, one
  commit per logical change.
- CI builds Linux and Windows and runs the tests; it must pass. CI has no GPU
  and no game files, so evidence that a change works in game is yours to
  bring.
- A commit message says what changed and why: a short subject
  (`render: the woods' trees are instanced draws`), then the cause and how it
  was checked.

## Releasing

A release is an annotated tag on `master` whose message holds the release
notes:

1. Write the notes for players in a Markdown file: what changed for them
   first, then the details. No title line; the release is named
   `bbhost v0.2.17`.
2. `tools/tag_release.sh v0.2.17 notes.md [commit]` tags the commit (default
   `HEAD`) with the notes, then `git push <remote> v0.2.17`. The script keeps
   the notes verbatim: a plain `git tag -F` would drop every line starting with
   `#`, Markdown headings included.

The `release` workflow builds and signs the packages, publishes the GitHub
release with those notes, and the bot posts them to the Discord's #releases
channel. A tag without notes still releases, with GitHub's generated notes,
but is not announced; write the notes on the release and run *announce
release* from the Actions page.

## Where code goes

- **System library calls** (HLE) go in the matching `src/hle/*.cpp`; a new
  library family gets its own file and an `hle_register_*()` call in
  `src/hle/runtime.cpp`. Return real handles and outputs: a `return 0` stub is
  a bug. An unavoidable fake is logged once with `HLE fake:`.
- Every function the game calls into is `GUEST_ABI` and reached through the
  FS-switching thunk. Anything that must see the guest's own frame
  (`setjmp`, callbacks into the game) needs `hle_call_guest` or an unthunked
  stub.
- **Patches** to the game's code check the eboot's SHA-256 (patch manifests do
  this for you). Prefer a hook or a host-side fix over a byte patch.
- **Game functions rewritten as source** go in `src/decomp/<area>.cpp`, are
  listed in `src/decomp/decomp.h`, and come with a compare mode that checks
  them against the game's own version ([docs/decomp.md](docs/decomp.md)).
- **Engine knowledge** (layouts, addresses) that plugins can use belongs in
  `include/bbhost/engine/`.
- Windows builds use Clang only (`__attribute__((sysv_abi))`).

## Rendering changes

Bring evidence from a run:

- F12 in game dumps the frame, every render target the last draws wrote and
  the draw list into a folder of its own, `build/f12-<date>-<time>/` (the log
  names it); `tools/f12_check.py <folder> <log>` flags lost clears, size
  mismatches and stale uploads.
- Before and after screenshots of the same place, from a copy of a save
  (`tools/menu_drive.sh` drives the game headless to a screenshot).
- No regressions elsewhere: the smoke target, a run with
  `BBHOST_VK_VALIDATE=1` and no new validation messages, and a second area.
- Where it makes sense, an environment switch that turns the change off, for
  A/B comparisons.

## Performance changes

- Pin A/B runs to one NUMA node (`numactl --cpunodebind=0 --membind=0`);
  unpinned runs vary by 10-20%.
- Compare the same place, alternating configurations, at least two rounds
  each, on an otherwise idle machine. Run the same configuration twice first
  to know the noise.
- Prefer GPU profile totals (`BBHOST_GPU_PROFILE=1`) and per-phase timings to
  frame counts: the game paces itself.
- Put the numbers, the GPU and the method in the pull request.
