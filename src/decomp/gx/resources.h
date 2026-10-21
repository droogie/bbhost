// GX's resource creators as source (decomp/gx/resources.cpp): what the GX
// resource registry (engine/gx_resources.cpp) and ours share.
#pragma once

#include <cstdint>

namespace decomp_gx {

// The host's entry for a buffer GX made or refused, which ours of
// gx_create_buffer calls last: buffer_made(holder, description, initial
// data, result) - the holder 0 when none was made, initial data 1 when the
// call brought some, the creator's result (0 made; 8 a description it
// refuses; 5 or 9 an allocation that failed). Behind the FS-switching thunk
// (thunk_wrap). Set by the registry once the decomp list is placed; null
// without the registry, and ours is then the game's creator alone.
// g_buffer_made_host is the same function unwrapped, for the compare run's
// stand-in, which is hosted already (the thunk's return would put the game's
// FS back under it).
extern void* g_buffer_made_entry;
extern void* g_buffer_made_host;

}  // namespace decomp_gx
