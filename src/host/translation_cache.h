#pragma once

// The translation cache (<data root>/bbhost/translation-cache.bin, beside the
// pipeline cache and off with it; BBHOST_TRANSLATION_CACHE=0 turns it off
// alone): what gcn::translate made of a program, kept between runs.
//
// Every start used to translate every shader GX creates again - ~1,350 pixel
// and ~1,660 vertex shaders, each twice (once for its resource paths, once for
// its stage) and then lifted, the stage manifest's few hundred and the compute
// shaders: 14 s of worker CPU in a world run on the Radeon 8060S (translate
// and lift 9.9 s + 3.8 s, libraries 0.2 s - the driver's pipeline cache had
// already made the compiles free; cpu.md CPU-13). The translation is a pure
// function of the program, its TranslateOptions and a few process-wide
// switches (gcn::translation_switches), and the lift one of the same and the
// translation, so a start can take an earlier one's results instead: keyed on
// all three, and on the translator's own sources (src/gcn,
// cmake/translator_fingerprint.cmake), so a build with another translator or
// lifter starts the file again. Kyo's preload keeps its SPIR-V the same way.
//
// Entries are compressed (zlib) and read lazily, at a lookup; the file is read
// whole at device start, rewritten whole (to a temporary, then renamed) at
// most once a minute while new translations come in, and at exit.

#include "gcn/lift.h"
#include "gcn/translate.h"

#include <string>

namespace gpu {

// What a translation is for. kPaths: read for its bindings only (paths_for and
// the resource paths a compile at creation offers it), so its entry keeps no
// SPIR-V and the result has none, taken from the cache or not. The lifts'
// entries are kept under uses of their own (lift_cached).
enum class TranslationUse : std::uint8_t { kStage = 0, kPaths = 1, kLiftPixel = 2, kLiftVertex = 3 };

// gcn::translate through the cache.
gcn::TranslateResult translate_cached(const gcn::Program& prog, const gcn::TranslateOptions& o,
                                      TranslationUse use = TranslationUse::kStage);
// gcn::lift_pixel_shader (`vertex` false) or gcn::lift_vertex_shader through
// the cache. `reference` must be translate_cached(prog, o)'s result, as every
// lift's is: the entry is keyed on the program and options alone. A lift taken
// from the cache has its SPIR-V and rejections, not its proof or listing,
// which no caller reads.
gcn::LiftResult lift_cached(bool vertex, const gcn::Program& prog, const gcn::TranslateOptions& o, const gcn::TranslateResult& reference);

// At device start, after the device's features are given to the translator
// (they are part of every key).
void translation_cache_load(const std::string& path);
// The whole file, when translations were added since the last write; `force`
// writes now (the exit), otherwise at most once a minute.
void translation_cache_save(const std::string& path, bool force);
// The same on a thread of its own (the 300-flip report's).
void translation_cache_save_async(const std::string& path);
// The exit report's line.
void translation_cache_report();
// Since the last call: "translations N cached (x ms), M translated (y ms)", or
// empty when there were none (the 300-flip report).
std::string translation_cache_window();
// The translator's fingerprint (cmake/translator_fingerprint.cmake), which the
// shader caches are stamped with: another one starts them all again.
const char* translator_fingerprint();

}  // namespace gpu
