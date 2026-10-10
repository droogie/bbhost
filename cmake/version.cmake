# Writes OUT (bbhost_version.h) with the git revision of SRC, plus "+" when
# the tree has uncommitted changes; rewritten only when the text changes so
# nothing rebuilds needlessly. `-c core.excludesFile=/dev/null`: the global
# excludes file is not always readable. BBHOST_GIT_REV in the
# environment wins: a build in a chroot (tools/linux_portable_build.sh) sees
# the tree but not its git directory, so the revision is read outside.
if(NOT "$ENV{BBHOST_GIT_REV}" STREQUAL "")
  set(rev "$ENV{BBHOST_GIT_REV}")
  set(r 0)
  set(from_env 1)
else()
execute_process(COMMAND git -c core.excludesFile=/dev/null rev-parse --short HEAD
  WORKING_DIRECTORY ${SRC} OUTPUT_VARIABLE rev OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE r)
endif()
if(from_env)
elseif(NOT r EQUAL 0 OR rev STREQUAL "")
  set(rev "unknown")
else()
  execute_process(COMMAND git -c core.excludesFile=/dev/null status --porcelain --untracked-files=no
    WORKING_DIRECTORY ${SRC} OUTPUT_VARIABLE dirty ERROR_QUIET)
  if(NOT dirty STREQUAL "")
    set(rev "${rev}+")
  endif()
endif()
# The release version: the newest v* tag at or before HEAD (git describe),
# "v0.1.0" on the tag itself, "v0.1.0-3-gabc1234" three commits past it,
# "v0.0.0-<rev>" with no tag in the history; "+" when the tree has changes.
# BBHOST_VERSION in the environment wins, as BBHOST_GIT_REV does.
if(NOT "$ENV{BBHOST_VERSION}" STREQUAL "")
  set(version "$ENV{BBHOST_VERSION}")
else()
  execute_process(COMMAND git -c core.excludesFile=/dev/null describe --tags --match "v[0-9]*" --abbrev=7
    WORKING_DIRECTORY ${SRC} OUTPUT_VARIABLE version OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE dr)
  if(NOT dr EQUAL 0 OR version STREQUAL "")
    string(REPLACE "+" "" bare "${rev}")
    set(version "v0.0.0-${bare}")
  endif()
  if(rev MATCHES "\\+$" AND NOT version MATCHES "\\+$")
    set(version "${version}+")
  endif()
endif()
# Its numbers, for the Windows exe's version details (res/bbhost.rc).
set(num "0,0,0")
if(version MATCHES "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)")
  set(num "${CMAKE_MATCH_1},${CMAKE_MATCH_2},${CMAKE_MATCH_3}")
endif()
set(content "#pragma once\n#define BBHOST_GIT_REV \"${rev}\"\n#define BBHOST_VERSION \"${version}\"\n#define BBHOST_VERSION_NUMBERS ${num}\n")
set(old "")
if(EXISTS ${OUT})
  file(READ ${OUT} old)
endif()
if(NOT old STREQUAL content)
  file(WRITE ${OUT} "${content}")
endif()
