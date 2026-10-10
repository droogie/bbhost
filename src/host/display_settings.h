// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace host {

// Same order as the reconstruction selector in both UI front ends.
inline constexpr const char* DlssModes[] = {
    "off", "dlaa", "quality", "balanced", "performance", "ultra_performance"};

inline constexpr const char* UpscalerBackendNames[] = {"dlss", "fsr3", "fsr4"};
inline constexpr const char* FgBackendNames[] = {"dlss", "fsr3"};

inline int parse_upscaler_backend(const std::string& name) {
    if (name == "fsr3") return 1;
    if (name == "fsr4") return 2;
    return 0;
}

inline const char* upscaler_backend_name(int backend) {
    if (backend == 1) return "fsr3";
    if (backend == 2) return "fsr4";
    return "dlss";
}

inline int parse_fg_backend(const std::string& name) {
    if (name == "fsr3") return 1;
    return 0;
}

inline const char* fg_backend_name(int backend) {
    if (backend == 1) return "fsr3";
    return "dlss";
}

// Higher factors can have uneven temporal spacing under an external FPS cap.
// Keep them available for pacing research without exposing them by default.
inline bool experimental_mfg_enabled() {
    const char* e = std::getenv("BBHOST_EXPERIMENTAL_MFG");
    return e && e[0] == '1';
}

inline bool valid_resolution(int width, int height) {
    return width >= 256 && width <= 7680 && height >= 144 && height <= 4320;
}

inline bool parse_resolution(const std::string& text, int* width, int* height) {
    unsigned w = 0, h = 0;
    char tail = 0;
    if (std::sscanf(text.c_str(), "%ux%u%c", &w, &h, &tail) != 2 ||
        w > 7680 || h > 4320 || !valid_resolution(static_cast<int>(w), static_cast<int>(h))) return false;
    *width = static_cast<int>(w);
    *height = static_cast<int>(h);
    return true;
}

inline int dlss_mode_index(const std::string& mode) {
    for (int i = 0; i < 6; ++i) if (mode == DlssModes[i]) return i;
    return 0;
}

struct DisplayDimensions {
    int render_width = 1920, render_height = 1080;
    int output_width = 1920, output_height = 1080;
};

inline DisplayDimensions display_dimensions(int width, int height, int mode,
                                           int output_width = 0, int output_height = 0,
                                           int forced_width = 0, int forced_height = 0) {
    DisplayDimensions size;
    if (!valid_resolution(width, height)) return size;
    size.render_width = size.output_width = width;
    size.render_height = size.output_height = height;
    if (mode >= 2 && mode <= 5) {
        if (valid_resolution(output_width, output_height)) {
            size.output_width = output_width;
            size.output_height = output_height;
        }
        constexpr double scale[] = {1.0, 1.0, 2.0 / 3.0, 0.58, 0.5, 1.0 / 3.0};
        // Even dimensions keep half-resolution post-process targets aligned.
        size.render_width = std::max(256, static_cast<int>(std::lround(size.output_width * scale[mode] / 2.0)) * 2);
        size.render_height = std::max(144, static_cast<int>(std::lround(size.output_height * scale[mode] / 2.0)) * 2);
    }
    // Retain the existing developer/launcher input-resolution override.
    if (valid_resolution(forced_width, forced_height)) {
        size.render_width = forced_width;
        size.render_height = forced_height;
        if (mode < 2 || mode > 5) {
            size.output_width = forced_width;
            size.output_height = forced_height;
        }
    }
    return size;
}

} // namespace host
