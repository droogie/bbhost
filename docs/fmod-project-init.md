# FMOD project language initialization

Set `BBHOST_FMOD_PROJECT_INIT=1` to initialize the 1.09 FMOD EventProjectI
language count to zero before its original constructor runs. The default is
unchanged while this fix receives broader gameplay testing.

The constructor leaves this member untouched. A successful LANG load assigns
the count later, but a failed load can reach cleanup before that assignment.
Initializing the count makes that cleanup see an empty language list. Successful
loads still supply their own count. This does not repair the underlying load
failure or suppress its error.

The hook uses the existing hosted guest/host transition and original-function
trampoline. The decomp installer checks the supported eboot SHA-256 and the
constructor entry bytes. When explicitly requested, startup aborts if the hook
was not installed (including when disabled by `BBHOST_DECOMP` or
`BBHOST_DECOMP_OFF`), rather than silently running without the fix.

## Evidence and verification

Two independent Windows crash dumps from the same modified enemy-placement
scenario showed FMOD failed-load cleanup traversing language entries using an
uninitialized count. Unrelated allocator and rendering data was zeroed. Static
inspection confirmed the missing constructor initialization and the assignment
after fallible loading. Dump contents, game assets and saves are not distributed.

The private diagnostic build with this initialization completed one subsequent
run without a crash, reported by the tester and corroborated by its window-close
log and absence of an exception capture. The run lasted about 101 seconds.
That is initial acceptance evidence, not proof of long-session stability. The
diagnostic build also contained earlier instrumentation and guards; this PR
extracts only the language initialization, with no dependency on those tools.

`fmod_project_init_test` uses owned storage and a constructor stub to check that
exactly the four-byte count is cleared before the original call, the call and
return value are forwarded, reused storage is reset, successful loading can
replace the count, and the fix requires explicit opt-in. Its `missing-hook` and
`missing-trampoline` modes must terminate abnormally, exercising startup refusal.
It does not execute the corrupt cleanup path or require game files.
