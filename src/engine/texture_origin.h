#pragma once
// Where a texture the game creates with data came from, by the file its
// texels were loaded out of (engine/texture_origin.cpp): the game decompresses
// a TPF - on its own (map/mXX/*.tpfbdt's entries, *.tpf.dcx) or inside a BND4
// (parts/*.partsbnd, sfx/*.ffxbnd, chr, obj) - and hands GX texels from inside
// it. Texture quality (host/textures.cpp) scales by it.
#include <cstdint>

enum GxTextureOrigin : std::uint8_t {
    kOriginUnknown = 0,  // no TPF found before the texels (made without data, or not from a file)
    kOriginMap,          // map/: the areas' textures (a TPF named in a map tpfbhd, or mNN_...)
    kOriginParts,        // parts/: armour, weapons, hair
    kOriginSfx,          // sfx/: the effects' textures
    kOriginMenu,         // menu/ and MENU_*: the menus and the HUD, never scaled
    kOriginGrading,      // the areas' colour grading tables (mNN_cgrading.tpf), never scaled
    kOriginOther,        // the rest: chr/, obj/, other/ (decals), ...
    kOriginCount
};

// The origin of the texels at `texels` (D3D11_SUBRESOURCE_DATA::pSysMem); on
// the creating thread, guest memory read safely.
GxTextureOrigin texture_origin_of(std::uint64_t texels);
// Reads the map folders' texture names on a thread of its own, so the first texture made does not wait for them.
void texture_origin_prepare();
const char* texture_origin_name(std::uint8_t origin);
// What the look-ups cost so far, for the exit report: textures placed, TPFs found, microseconds in all.
void texture_origin_stats(std::uint64_t* textures, std::uint64_t* packs, std::uint64_t* us);
