# Writes OUT (translator_fingerprint.h): a SHA-256 over the translator's
# sources (src/gcn, every .cpp and .h, by name and content) and the
# translation cache's own (src/host/translation_cache.cpp and .h: what a key
# holds, how an entry is written). The translation cache keeps it in its file
# and starts empty when the running build's differs, and the shader caches are
# stamped with it (src/host/gpu.cpp, shader_caches_check_build): a translator
# change, or a change to the key, throws away the translations and pipelines of
# the one before, and any other change keeps them - an update that leaves the
# translator alone starts with every shader built. Rewritten only when the text
# changes, like bbhost_version.h, so nothing rebuilds needlessly.
file(GLOB sources ${SRC}/src/gcn/*.cpp ${SRC}/src/gcn/*.h)
list(APPEND sources ${SRC}/src/host/translation_cache.cpp ${SRC}/src/host/translation_cache.h)
list(SORT sources)
set(all "")
foreach(f ${sources})
  file(SHA256 ${f} h)
  get_filename_component(n ${f} NAME)
  string(APPEND all "${n}:${h};")
endforeach()
string(SHA256 fp "${all}")
string(SUBSTRING "${fp}" 0 32 fp)
set(content "#pragma once\n#define BBHOST_TRANSLATOR_FINGERPRINT \"${fp}\"\n")
set(old "")
if(EXISTS ${OUT})
  file(READ ${OUT} old)
endif()
if(NOT old STREQUAL content)
  file(WRITE ${OUT} "${content}")
endif()
