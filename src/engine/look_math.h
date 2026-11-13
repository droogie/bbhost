#pragma once

// The arithmetic of a plugin's look (plugin API 29's post_look) that
// engine/graphics_patch.cpp's apply_look writes into YEBIS's post-process
// block, apart from the block so that tests/look_math_test.cpp checks it
// without the game.

namespace look_math {

// The motion blur's sample count a look writes (the block's +0x1ec): 2 to 8,
// 7 where the area interleaves its samples (+0x204). YEBIS's techniques are
// built for 1 to 8 samples and a larger count draws with shaders that are not
// the blur's; interleaving makes the count odd, so 8 would be 9; 1 and below
// are taken as YEBIS's own 4 or 5 (graphics_patch.cpp says where).
constexpr int blur_samples(int wanted, bool interleaved) {
    const int most = interleaved ? 7 : 8;
    return wanted < 2 ? 2 : wanted > most ? most : wanted;
}

}  // namespace look_math
