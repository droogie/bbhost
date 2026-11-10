// Trophy popups: a box that slides in from the top left, stays a few seconds
// and slides out, like the PS4's. Drawn with the host overlay's own rects and
// text, so it shows during play and needs no ImGui. The name, grade and icon
// come from the game's own trophy pack (core/trophy_pack.h) when it could be
// read; without it the popup only says that a trophy was unlocked. The icon is
// drawn as a mosaic of the overlay's own triangles: the overlay's one pipeline
// only samples its font, so a picture of its own would need a second shader.

#include "host/trophy_toast.h"

#include "core/trophy_pack.h"
#include "host/overlay.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::mutex g_mu;
std::deque<int> g_pending;  // trophy ids waiting to be shown (under g_mu)

constexpr float kIn = 0.35f;    // seconds sliding in
constexpr float kHold = 5.0f;   // seconds fully shown
constexpr float kOut = 0.35f;   // seconds sliding out

// BBHOST_TROPHY_TOAST=0 turns the popups off.
bool enabled() {
    static const bool v = [] {
        const char* e = std::getenv("BBHOST_TROPHY_TOAST");
        return !(e && e[0] == '0');
    }();
    return v;
}

// 0xRRGGBBAA with its alpha scaled by `a` (0..1).
std::uint32_t fade(std::uint32_t rgba, float a) {
    const float al = static_cast<float>(rgba & 0xffu) * std::clamp(a, 0.0f, 1.0f);
    return (rgba & 0xffffff00u) | static_cast<std::uint32_t>(al);
}

// The overlay's font has plain ASCII only: anything else becomes a '?' (one
// per character, not per byte), and line breaks become spaces.
std::string ascii_only(const std::string& s) {
    std::string out;
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c >= 0x80) {
            if (c >= 0xc0) out += '?';  // the lead byte of a multi-byte character
        } else if (c < 0x20) {
            out += ' ';
        } else {
            out += ch;
        }
    }
    return out;
}

// What the popup says when the game's trophy pack could not be read (no trophy
// key, say): no name, grade or icon to show, so it points at the docs.
constexpr const char kNoPackText[] = "Trophy unlocked - See docs for full behavior";

// The icon as rows of same-coloured runs in a cells x cells grid, so a flat
// stretch costs one rectangle and not one per cell.
struct IconRun {
    int row, c0, c1;  // cells c0..c1-1 of one row, one colour
    std::uint32_t rgba;
};
struct IconRuns {
    int cells = 0;  // 0: no icon
    std::vector<IconRun> runs;
};

IconRuns runs_of(const std::vector<std::uint8_t>& rgba, int sw, int sh, int cells) {
    IconRuns out;
    const std::vector<std::uint8_t> small = rgba_resize(rgba, sw, sh, cells, cells);
    const auto px = [&](int r, int c) {
        const std::uint8_t* p = small.data() + (static_cast<std::size_t>(r) * static_cast<std::size_t>(cells) + static_cast<std::size_t>(c)) * 4;
        return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
               (static_cast<std::uint32_t>(p[2]) << 8) | p[3];
    };
    for (int r = 0; r < cells; ++r) {
        for (int c = 0; c < cells;) {
            const std::uint32_t v = px(r, c);
            if ((v & 0xffu) < 16) {  // transparent: nothing to draw
                ++c;
                continue;
            }
            int e = c + 1;
            while (e < cells && px(r, e) == v) ++e;
            out.runs.push_back({r, c, e, v});
            c = e;
        }
    }
    out.cells = cells;
    return out;
}

// The popup's icon at 32x32 and 16x16: the finer one is drawn while the frame
// has the vertex room for it.
void icon_mosaics(int id, IconRuns* fine, IconRuns* coarse) {
    *fine = IconRuns{};
    *coarse = IconRuns{};
    std::vector<std::uint8_t> png, rgba;
    int w = 0, h = 0;
    if (!trophy_pack_icon(id, &png) || !png_decode_rgba(png.data(), png.size(), &w, &h, &rgba, nullptr)) return;
    *fine = runs_of(rgba, w, h, 32);
    *coarse = runs_of(rgba, w, h, 16);
}

struct Grade {
    const char* label;
    std::uint32_t colour;
};
Grade grade_of(char g) {
    switch (g) {
        case 'P': return {"Platinum trophy earned", 0xa6d8f2ffu};
        case 'G': return {"Gold trophy earned", 0xe0b43cffu};
        case 'S': return {"Silver trophy earned", 0xc4c8d0ffu};
        default: return {"Bronze trophy earned", 0xb0713affu};
    }
}

}  // namespace

void host_trophy_toast_push(int trophy_id) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lock(g_mu);
    g_pending.push_back(trophy_id);
}

void host_trophy_toast_draw(float display_w, float display_h) {
    if (!enabled()) return;

    // BBHOST_TROPHY_TOAST_TEST=1: a popup for the first trophy of each grade
    // the game's pack has (bronze, silver, gold, platinum), one after the
    // other, shortly after start, to see the look without earning a trophy.
    // With no pack, one popup: what the player sees without a trophy key.
    static bool test_done = false;
    if (!test_done) {
        test_done = true;
        const char* e = std::getenv("BBHOST_TROPHY_TOAST_TEST");
        if (e && e[0] == '1') {
            bool any = false;
            for (const char grade : {'B', 'S', 'G', 'P'}) {
                for (int id = 0; id < 256; ++id) {
                    TrophyInfo info;
                    if (trophy_pack_find(id, &info) && info.grade == grade) {
                        host_trophy_toast_push(id);
                        any = true;
                        break;
                    }
                }
            }
            if (!any) host_trophy_toast_push(0);
        }
    }

    static bool active = false;  // the window pump's thread only
    static Clock::time_point t0;
    static std::string label, name;
    static std::uint32_t colour = 0;
    static bool bare = false;  // no pack: one line of text and no icon
    static IconRuns mosaic_fine, mosaic_coarse;
    const Clock::time_point now = Clock::now();

    if (!active) {
        int id;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            if (g_pending.empty()) return;
            id = g_pending.front();
            g_pending.pop_front();
        }
        TrophyInfo info;
        if (trophy_pack_find(id, &info) && !info.name.empty()) {
            const Grade g = grade_of(info.grade);
            bare = false;
            label = g.label;
            colour = g.colour;
            name = ascii_only(info.name);
            icon_mosaics(id, &mosaic_fine, &mosaic_coarse);  // none without an icon in the pack
        } else {  // no pack (no key, say): only that a trophy was unlocked
            bare = true;
            label.clear();
            name = kNoPackText;
            mosaic_fine = IconRuns{};
            mosaic_coarse = IconRuns{};
        }
        t0 = now;
        active = true;
    }

    const float t = std::chrono::duration<float>(now - t0).count();
    if (t > kIn + kHold + kOut) {
        active = false;
        return;
    }

    // 0 -> 1 sliding in, 1 while held, 1 -> 0 sliding out; eased.
    float s = 1.0f;
    if (t < kIn) {
        s = t / kIn;
    } else if (t > kIn + kHold) {
        s = 1.0f - (t - kIn - kHold) / kOut;
    }
    s = std::clamp(s, 0.0f, 1.0f);
    s = 1.0f - (1.0f - s) * (1.0f - s);

    // Sizes are for a 1080-pixel-high display and scale with it.
    const float k = display_h > 0.0f ? display_h / 1080.0f : 1.0f;
    const float margin = 24.0f * k, h = 76.0f * k, pad = 10.0f * k;
    const float icon = bare ? 0.0f : h - 2.0f * pad;
    const float left = bare ? pad : pad + icon + pad;  // where the text starts, from the box's edge
    const float label_scale = 0.62f * k;
    float name_scale = 0.9f * k;

    // As wide as the name needs, within reason; a very long name shrinks.
    const float label_w = bare ? 0.0f : host_overlay_text_width(label_scale, label.c_str());
    float name_w = host_overlay_text_width(name_scale, name.c_str());
    const float text_w = std::max(label_w, name_w);
    float w = std::max(380.0f * k, left + text_w + pad);
    const float max_w = display_w > 0.0f ? display_w * 0.6f : 1100.0f * k;
    if (w > max_w) {
        w = max_w;
        const float room = w - left - pad;
        if (name_w > room && name_w > 0.0f) name_scale *= room / name_w;
    }

    const float x = margin - (1.0f - s) * (w + margin);  // in from the left edge
    const float y = margin;
    const float a = s;

    host_overlay_rect(x, y, w, h, fade(0x12141aeeu, a));
    const float b = std::max(1.0f, 1.5f * k);  // border width
    const std::uint32_t edge = fade(0xffffff3cu, a);
    host_overlay_rect(x, y, w, b, edge);
    host_overlay_rect(x, y + h - b, w, b, edge);
    host_overlay_rect(x, y, b, h, edge);
    host_overlay_rect(x + w - b, y, b, h, edge);

    // The grade's colour: the icon's frame, or all of it when there is no icon.
    if (!bare) host_overlay_rect(x + pad, y + pad, icon, icon, fade(colour, a));

    // The icon, as fine as this frame has vertex room for (a frame past the
    // overlay's vertex buffer is not drawn at all, so it falls back to the
    // coarser mosaic and then to the plain square).
    const IconRuns* mosaic = nullptr;
    {
        const std::size_t room = host_overlay_room();
        for (const IconRuns* m : {&mosaic_fine, &mosaic_coarse}) {
            if (m->cells && m->runs.size() * 6 + 1500 <= room) {
                mosaic = m;
                break;
            }
        }
    }
    if (mosaic && !bare) {
        const float fr = std::max(2.0f, 3.0f * k);
        const float cell = (icon - 2.0f * fr) / static_cast<float>(mosaic->cells);
        const float ix = x + pad + fr, iy = y + pad + fr;
        for (const IconRun& r : mosaic->runs) {
            // Each corner is computed the same way for every neighbour, so the
            // triangles share their edges exactly and leave no seams.
            const float xa = ix + static_cast<float>(r.c0) * cell, xb = ix + static_cast<float>(r.c1) * cell;
            const float ya = iy + static_cast<float>(r.row) * cell, yb = iy + static_cast<float>(r.row + 1) * cell;
            const std::uint32_t c = fade(r.rgba, a);
            host_overlay_tri(xa, ya, xb, ya, xa, yb, c);
            host_overlay_tri(xb, ya, xb, yb, xa, yb, c);
        }
    }

    const float tx = x + left;
    if (bare) {  // the one line, in the middle of the box
        host_overlay_text(tx, y + h * 0.5f - 9.0f * name_scale, name_scale, fade(0xffffffffu, a), name.c_str());
        return;
    }
    host_overlay_text(tx, y + pad, label_scale, fade(0xbec3cdffu, a), label.c_str());
    host_overlay_text(tx, y + pad + 22.0f * k, name_scale, fade(0xffffffffu, a), name.c_str());
}
