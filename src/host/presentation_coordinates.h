#pragma once

#include <algorithm>

namespace host {

struct FitRect {
    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
};

// Shared by presentation and pointer conversion, including letterboxing.
inline FitRect fit_picture(float w, float h, float dw, float dh) {
    if (w <= 0.0f || h <= 0.0f || dw <= 0.0f || dh <= 0.0f) return {0.0f, 0.0f, w, h};
    const float s = std::min(w / dw, h / dh);
    const float fw = dw * s, fh = dh * s;
    return {(w - fw) * 0.5f, (h - fh) * 0.5f, fw, fh};
}

inline bool window_to_picture(float wx, float wy, float ww, float wh,
                              float sw, float sh, float dw, float dh,
                              float& px, float& py) {
    if (ww <= 0.0f || wh <= 0.0f || dw <= 0.0f || dh <= 0.0f) return false;
    if (sw <= 0.0f) sw = ww;
    if (sh <= 0.0f) sh = wh;
    const FitRect r = fit_picture(sw, sh, dw, dh);
    px = (wx * sw / ww - r.x) * dw / r.w;
    py = (wy * sh / wh - r.y) * dh / r.h;
    return px >= 0.0f && py >= 0.0f && px < dw && py < dh;
}

// Scaleform uses a centered, show-all 1920x1080 stage inside the picture.
// Keep points outside that stage: ultrawide world markers can live there.
inline bool picture_to_menu_stage(float px, float py, float dw, float dh,
                                  float& sx, float& sy) {
    if (dw <= 0.0f || dh <= 0.0f) return false;
    const FitRect stage = fit_picture(dw, dh, 1920.0f, 1080.0f);
    sx = (px - stage.x) * 1920.0f / stage.w;
    sy = (py - stage.y) * 1080.0f / stage.h;
    return true;
}

} // namespace host
