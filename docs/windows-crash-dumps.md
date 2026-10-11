# Windows crash dumps

The normal text crash report stays unchanged. To collect more evidence for a
hard-to-reproduce Windows crash, set `BBHOST_FULL_DUMP` to a new output filename
before starting bbhost. The parent directory must already exist:

```powershell
$env:BBHOST_FULL_DUMP = 'D:\bbhost-diagnostics\run-001.dmp'
.\bbhost.exe
```

Keep `crash_dump_helper.exe` from the same build beside `bbhost.exe`. The Windows
package includes it. If using the standalone release executable, build or obtain
the matching helper from the complete package. The feature is off when the
variable is unset; it does not upload anything.

## What gets saved

Startup creates a hidden collector process and reserves the `.partial` output
file. Missing helpers, unwritable output directories, existing output files, or
failure to acknowledge readiness stop startup with exit code 78. Use a fresh
filename for every run; existing dumps and partial files are never replaced.

On capture, the helper first writes and flushes `run-001.dmp.triage.dmp`, with
exception context and thread information. It then writes the readable process
memory to `run-001.dmp`. Each is renamed from its own `.partial` name only after
successful writing and flushing. The two results are reported separately: a full
capture failure must not conceal a triage failure, and a completed triage dump
remains usable when the full capture times out. `.partial` files are incomplete
evidence, not successful captures. A clean exit removes the unused reservation.

DbgHelp runs in the helper instead of inside the faulting process. Handles and
shared storage are prepared at startup; the crashing thread copies the original
exception context, signals the helper, and waits. Concurrent failures share one
capture. The unhandled-exception filter and fatal-abort handler request capture
before logging. Reportable guest exceptions reaching the existing last vectored
handler request it before verbose reporting or a downstream filter can exit.
Earlier vectored handlers can still handle expected faults first. A later SEH
handler could recover a first-chance guest exception, so a dump alone is not
proof the process terminated.

## Time, storage, and scope

The production deadline is 600 seconds for both stages together. The helper is
stopped on timeout, and the log identifies failure. A supervising launcher must
allow this time plus shutdown margin: `run-001.dmp.capturing` is created when the
request begins and remains as evidence that capture started. Do not terminate
the process just because a normal gameplay deadline expired. This PR does not
change external launchers or their watchdogs.

Full dumps can be much larger than physical RAM usage because virtual aliases
are included. Prior diagnostic runs produced approximately 24–25 GB dumps and
took 83–296 seconds. Leave ample disk space (48 GiB free was used for those
trials); there is no universal size estimate or disk-space guarantee. A file
shown as zero bytes while exclusively open is not proof capture is stalled.

These are Windows x64 diagnostics. Abrupt external termination, some fail-fast
paths, exhaustion of the crashing thread's stack, or severe corruption can still
prevent capture. Other threads may change memory during collection; the original
exception context is preserved, but the dump is not an atomic all-thread memory
snapshot. Full dumps can contain account tokens, private runtime data, and game
assets: inspect them locally and do not attach them to public issues. These
files are ignored by Git.

## Tests

The dependency-free test project builds the actual collector, helper, and
production Windows crash handlers, with harmless exception fixtures and owned
sentinel memory. No game, GPU, plugins, or corruption reproduction is required.
Run in an MSYS2 CLANG64 environment on Windows:

```sh
cmake -S tests/windows_crash -B build-crash -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build-crash
ctest --test-dir build-crash --output-on-failure
```

The 19 cases cover early guest capture before a terminating downstream filter,
unhandled exceptions, abort, concurrent requests, clean and forced exits,
disabled collection, missing helper/directory, five output collisions, helper
exit during capture, readiness timeout, full-capture timeout after triage,
triage failure with successful full capture, and a final-name collision after
startup. They parse dump directories, exception/thread/context
records and full-memory sentinel bytes. Unicode and spaces in paths are covered.
A separate helper target delays after triage for the timeout test; that delay is
not compiled into the production helper. CI runs these tests on native Windows.

The predecessor diagnostic collector also recovered two real gameplay crashes
and passed a 25.8 GB alias-mapping capture test. Those runs support the design;
they do not replace testing this extracted version. No real dumps, saves, or
private gameplay harnesses are included here.
