#define IMGUI_DEFINE_MATH_OPERATORS
#include "host/launcher_ui.h"

#include "log.h"

#include "imgui_internal.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iterator>
#include <random>
#include <vector>

namespace launcher_ui {
namespace {

// res/launcher-art/*.png: Kyo's screenshots - his hunter on a terrace under
// the moon first, then the game's streets, halls and fights - their light
// only: 8-bit grey, 960x540, tones lifted toward the ramp's middle. Carried
// whole in the exe (build/gen/launcher_art.inc); one is picked each start and
// the palette colours it at load.
struct ArtPng {
    const unsigned char* begin;
    const unsigned char* end;
};
#include "launcher_art.inc"

// The picture's light, dark to bright, as the palette: ink, indigo, Kyo's
// violet, his blue, ice.
struct Stop {
    float t;
    int r, g, b;
};
constexpr Stop kRamp[] = {{0.00f, 5, 4, 14},    {0.18f, 16, 10, 42},    {0.38f, 46, 20, 108},  {0.55f, 80, 40, 172},
                          {0.72f, 64, 112, 232}, {0.88f, 132, 186, 255}, {1.00f, 238, 244, 255}};

constexpr ImU32 kCardTint = IM_COL32(30, 22, 68, 180);
constexpr ImU32 kSwitchOff = IM_COL32(255, 255, 255, 34);

float g_scale = 1.0f;
SDL_Renderer* g_ren = nullptr;
ImFont* g_body = nullptr;
ImFont* g_strong = nullptr;
ImFont* g_title = nullptr;

struct Art {
    SDL_Texture* sharp = nullptr;  // the painting in the palette
    SDL_Texture* soft = nullptr;   // the same, blurred: what glass shows
    SDL_Texture* mild = nullptr;   // blurred a little: what panes show
    int w = 0, h = 0;
} g_art;

// Where the whole painting lands this frame (past the window's edges where it
// is cropped); glass samples the blurred copy through the same mapping.
ImVec2 g_cover0, g_cover1;

// The motes drifting up through the night, in window fractions.
struct Mote {
    float x, y, r, speed, phase, sway;
    bool pink;
};
std::vector<Mote> g_motes;
std::mt19937 g_rng(7);

// A colour as ImGui would draw it now: faded by the style's alpha (a page
// fading in, a disabled control).
ImU32 c(ImU32 col) { return ImGui::GetColorU32(col); }

ImU32 mix(ImU32 a, ImU32 b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    auto ch = [&](int shift) {
        const float x = static_cast<float>((a >> shift) & 0xFF), y = static_cast<float>((b >> shift) & 0xFF);
        return static_cast<ImU32>(x + (y - x) * t + 0.5f) << shift;
    };
    return ch(IM_COL32_R_SHIFT) | ch(IM_COL32_G_SHIFT) | ch(IM_COL32_B_SHIFT) | ch(IM_COL32_A_SHIFT);
}

ImTextureRef tex(SDL_Texture* t) { return ImTextureRef(static_cast<ImTextureID>(reinterpret_cast<std::intptr_t>(t))); }

// A rounded rectangle shaded from c0 to c1 (left to right, or top to bottom),
// `a` its alpha: ImGui fills it white and the gradient recolours the vertices.
void gradient_rect(ImDrawList* dl, ImVec2 p0, ImVec2 p1, ImU32 c0, ImU32 c1, float rounding, int a = 255, bool vertical = false,
                   ImDrawFlags flags = 0) {
    const int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(p0, p1, c(IM_COL32(255, 255, 255, a)), rounding, flags);
    ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, p0, vertical ? ImVec2(p0.x, p1.y) : ImVec2(p1.x, p0.y), c0, c1);
}

// A four-pointed star (concave), `r` its waist.
void star(ImDrawList* dl, ImVec2 at, float big, float r, ImU32 col) {
    ImVec2 p[8];
    for (int i = 0; i < 8; ++i) {
        const float a = static_cast<float>(i) * IM_PI / 4.0f - IM_PI / 2.0f;
        const float rr = (i % 2) ? r : big;
        p[i] = at + ImVec2(std::cos(a), std::sin(a)) * rr;
    }
    dl->AddConcavePolyFilled(p, 8, col);
}

// The switch's track and knob at `p` (its top-left), `k` how far on.
ImVec2 switch_size() {
    const float h = std::round(ImGui::GetFrameHeight() * 0.74f);
    return ImVec2(std::round(h * 1.85f), h);
}
void draw_switch(ImDrawList* dl, ImVec2 p, float k, bool hovered) {
    const ImVec2 sz = switch_size();
    const float r = sz.y * 0.5f;
    dl->AddRectFilled(p, p + sz, c(hovered ? IM_COL32(255, 255, 255, 48) : kSwitchOff), r);
    if (k > 0.01f) gradient_rect(dl, p, p + sz, kBlue, kViolet, r, static_cast<int>(255 * k));
    const float x = p.x + r + (sz.x - 2 * r) * k;
    const float kr = r - 3.0f * g_scale;
    dl->AddCircleFilled(ImVec2(x, p.y + r + 1.0f * g_scale), kr, c(IM_COL32(0, 0, 0, 70)));
    dl->AddCircleFilled(ImVec2(x, p.y + r), kr, c(mix(kTextDim, IM_COL32(250, 251, 255, 255), 0.4f + 0.6f * k)));
}

// Fonts: the first of `names` that exists - under %WINDIR%\Fonts on
// Windows, whole paths elsewhere.
ImFont* add_font(std::initializer_list<const char*> names) {
    for (const char* n : names) {
#if defined(_WIN32)
        const char* windir = std::getenv("WINDIR");
        const std::string path = std::string(windir && *windir ? windir : "C:/Windows") + "/Fonts/" + n;
#else
        const std::string path = n;
#endif
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) continue;
        // 1.92's atlas adds glyphs as they are drawn: no ranges.
        if (ImFont* f = ImGui::GetIO().Fonts->AddFontFromFileTTF(path.c_str())) return f;
    }
    return nullptr;
}

void load_fonts() {
#if defined(_WIN32)
    g_body = add_font({"segoeui.ttf"});
    if (!g_body) g_body = ImGui::GetIO().Fonts->AddFontDefault();
    g_strong = add_font({"seguisb.ttf", "segoeuib.ttf"});
    g_title = add_font({"pala.ttf", "georgia.ttf", "constan.ttf"});
#else
    g_body = add_font({"/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/noto/NotoSans-Regular.ttf",
                       "/usr/share/fonts/google-noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                       "/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                       "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                       "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                       "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"});
    if (!g_body) g_body = ImGui::GetIO().Fonts->AddFontDefault();
    g_strong = add_font({"/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf", "/usr/share/fonts/noto/NotoSans-SemiBold.ttf",
                         "/usr/share/fonts/google-noto/NotoSans-SemiBold.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
                         "/usr/share/fonts/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                         "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                         "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
                         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                         "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"});
    g_title = add_font({"/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf", "/usr/share/fonts/noto/NotoSerif-Regular.ttf",
                        "/usr/share/fonts/google-noto/NotoSerif-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                        "/usr/share/fonts/TTF/DejaVuSerif.ttf", "/usr/share/fonts/dejavu/DejaVuSerif.ttf",
                        "/usr/share/fonts/dejavu-serif-fonts/DejaVuSerif.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                        "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf"});
#endif
    ImGui::GetIO().FontDefault = g_body;
}

void apply_style(float s) {
    ImGuiStyle& st = ImGui::GetStyle();
    st = ImGuiStyle();
    st.WindowPadding = ImVec2(14, 12);
    st.FramePadding = ImVec2(10, 6);
    st.ItemSpacing = ImVec2(10, 9);
    st.ItemInnerSpacing = ImVec2(8, 6);
    st.CellPadding = ImVec2(6, 4);
    st.IndentSpacing = 18;
    st.ScrollbarSize = 10;
    st.GrabMinSize = 10;
    st.WindowRounding = 0;
    st.ChildRounding = 12;
    st.FrameRounding = 7;
    st.PopupRounding = 10;
    st.ScrollbarRounding = 8;
    st.GrabRounding = 7;
    st.TabRounding = 7;
    st.WindowBorderSize = 0;
    st.ChildBorderSize = 1;
    st.PopupBorderSize = 1;
    // A faint edge on fields, boxes and buttons: an empty one still shows on glass.
    st.FrameBorderSize = 1;
    st.SeparatorTextBorderSize = 1;
    st.DisabledAlpha = 0.45f;
    // Bubbles come up as soon as the pointer rests.
    st.HoverStationaryDelay = 0.06f;
    st.HoverDelayShort = 0.06f;
    st.ScaleAllSizes(s);
    st.FontScaleDpi = s;
    st.FontSizeBase = 16.0f;

    auto v = [](int r, int g, int b, float a = 1.0f) { return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a); };
    ImVec4* col = st.Colors;
    col[ImGuiCol_Text] = v(232, 236, 255);
    col[ImGuiCol_TextDisabled] = v(140, 146, 192);
    col[ImGuiCol_WindowBg] = v(7, 6, 15, 0.0f);
    col[ImGuiCol_ChildBg] = v(0, 0, 0, 0.0f);
    col[ImGuiCol_PopupBg] = v(15, 12, 34, 0.97f);
    col[ImGuiCol_Border] = v(150, 130, 255, 0.26f);
    col[ImGuiCol_BorderShadow] = v(0, 0, 0, 0.0f);
    col[ImGuiCol_FrameBg] = v(34, 28, 80, 0.58f);
    col[ImGuiCol_FrameBgHovered] = v(48, 40, 112, 0.74f);
    col[ImGuiCol_FrameBgActive] = v(62, 50, 140, 0.88f);
    col[ImGuiCol_TitleBg] = v(15, 12, 34);
    col[ImGuiCol_TitleBgActive] = v(26, 20, 60);
    col[ImGuiCol_TitleBgCollapsed] = v(15, 12, 34, 0.7f);
    col[ImGuiCol_MenuBarBg] = v(15, 12, 34);
    col[ImGuiCol_ScrollbarBg] = v(0, 0, 0, 0.0f);
    col[ImGuiCol_ScrollbarGrab] = v(130, 110, 255, 0.30f);
    col[ImGuiCol_ScrollbarGrabHovered] = v(130, 110, 255, 0.48f);
    col[ImGuiCol_ScrollbarGrabActive] = v(130, 110, 255, 0.62f);
    col[ImGuiCol_CheckMark] = v(246, 248, 255);
    col[ImGuiCol_CheckboxSelectedBg] = v(124, 77, 255, 0.95f);
    col[ImGuiCol_SliderGrab] = v(124, 77, 255);
    col[ImGuiCol_SliderGrabActive] = v(152, 112, 255);
    col[ImGuiCol_Button] = v(52, 44, 120, 0.62f);
    col[ImGuiCol_ButtonHovered] = v(80, 64, 174, 0.84f);
    col[ImGuiCol_ButtonActive] = v(100, 76, 212, 0.96f);
    col[ImGuiCol_Header] = v(124, 77, 255, 0.30f);
    col[ImGuiCol_HeaderHovered] = v(124, 77, 255, 0.44f);
    col[ImGuiCol_HeaderActive] = v(124, 77, 255, 0.58f);
    col[ImGuiCol_Separator] = v(130, 110, 255, 0.22f);
    col[ImGuiCol_SeparatorHovered] = v(130, 110, 255, 0.40f);
    col[ImGuiCol_SeparatorActive] = v(130, 110, 255, 0.60f);
    col[ImGuiCol_ResizeGrip] = v(0, 0, 0, 0.0f);
    col[ImGuiCol_ResizeGripHovered] = v(0, 0, 0, 0.0f);
    col[ImGuiCol_ResizeGripActive] = v(0, 0, 0, 0.0f);
    col[ImGuiCol_InputTextCursor] = v(191, 216, 255);
    col[ImGuiCol_Tab] = v(52, 44, 120, 0.50f);
    col[ImGuiCol_TabHovered] = v(80, 64, 174, 0.84f);
    col[ImGuiCol_TabSelected] = v(100, 76, 212, 0.90f);
    col[ImGuiCol_TabSelectedOverline] = v(61, 139, 255);
    col[ImGuiCol_TabDimmed] = v(52, 44, 120, 0.35f);
    col[ImGuiCol_TabDimmedSelected] = v(80, 64, 174, 0.60f);
    col[ImGuiCol_PlotLines] = v(132, 186, 255);
    col[ImGuiCol_PlotHistogram] = v(61, 139, 255);
    col[ImGuiCol_TableHeaderBg] = v(34, 28, 80, 0.60f);
    col[ImGuiCol_TableBorderStrong] = v(130, 110, 255, 0.30f);
    col[ImGuiCol_TableBorderLight] = v(130, 110, 255, 0.16f);
    col[ImGuiCol_TableRowBgAlt] = v(255, 255, 255, 0.03f);
    col[ImGuiCol_TextLink] = v(120, 172, 255);
    col[ImGuiCol_TextSelectedBg] = v(124, 77, 255, 0.42f);
    col[ImGuiCol_NavCursor] = v(124, 77, 255);
    col[ImGuiCol_ModalWindowDimBg] = v(4, 3, 10, 0.55f);
}

// Three passes of a box blur, near enough a Gaussian.
void blur(std::vector<float>& img, int w, int h, int r) {
    std::vector<float> tmp(img.size());
    const float n = static_cast<float>(2 * r + 1);
    for (int pass = 0; pass < 3; ++pass) {
        for (int y = 0; y < h; ++y) {
            const float* row = &img[static_cast<std::size_t>(y) * w];
            float acc = 0;
            for (int x = -r; x <= r; ++x) acc += row[std::clamp(x, 0, w - 1)];
            for (int x = 0; x < w; ++x) {
                tmp[static_cast<std::size_t>(y) * w + x] = acc / n;
                acc += row[std::min(x + r + 1, w - 1)] - row[std::max(x - r, 0)];
            }
        }
        for (int x = 0; x < w; ++x) {
            float acc = 0;
            for (int y = -r; y <= r; ++y) acc += tmp[static_cast<std::size_t>(std::clamp(y, 0, h - 1)) * w + x];
            for (int y = 0; y < h; ++y) {
                img[static_cast<std::size_t>(y) * w + x] = acc / n;
                acc += tmp[static_cast<std::size_t>(std::min(y + r + 1, h - 1)) * w + x] - tmp[static_cast<std::size_t>(std::max(y - r, 0)) * w + x];
            }
        }
    }
}

SDL_Texture* upload(const std::vector<float>& grey, int w, int h) {
    std::uint8_t lut[256][3];
    for (int i = 0; i < 256; ++i) {
        const float t = static_cast<float>(i) / 255.0f;
        for (std::size_t k = 1; k < std::size(kRamp); ++k) {
            if (t > kRamp[k].t && k + 1 < std::size(kRamp)) continue;
            const Stop& a = kRamp[k - 1];
            const Stop& b = kRamp[k];
            const float f = std::clamp((t - a.t) / (b.t - a.t), 0.0f, 1.0f);
            lut[i][0] = static_cast<std::uint8_t>(a.r + (b.r - a.r) * f + 0.5f);
            lut[i][1] = static_cast<std::uint8_t>(a.g + (b.g - a.g) * f + 0.5f);
            lut[i][2] = static_cast<std::uint8_t>(a.b + (b.b - a.b) * f + 0.5f);
            break;
        }
    }
    std::vector<std::uint8_t> rgba(grey.size() * 4);
    for (std::size_t i = 0; i < grey.size(); ++i) {
        const int v = std::clamp(static_cast<int>(grey[i] + 0.5f), 0, 255);
        rgba[i * 4 + 0] = lut[v][0];
        rgba[i * 4 + 1] = lut[v][1];
        rgba[i * 4 + 2] = lut[v][2];
        rgba[i * 4 + 3] = 255;
    }
    SDL_Texture* t = SDL_CreateTexture(g_ren, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STATIC, w, h);
    if (!t) return nullptr;
    SDL_UpdateTexture(t, nullptr, rgba.data(), w * 4);
    SDL_SetTextureScaleMode(t, SDL_SCALEMODE_LINEAR);
    return t;
}

void load_art() {
    // Any of them this run; the first, the moonlit terrace, for
    // BBHOST_SETUP_TEST=shot's pictures, so they stay the same.
    // BBHOST_SETUP_BACKGROUND=n (from 1) picks one.
    const char* test = std::getenv("BBHOST_SETUP_TEST");
    const char* chosen = std::getenv("BBHOST_SETUP_BACKGROUND");
    std::size_t pick = 0;
    if (chosen && *chosen) {
        pick = std::clamp<std::size_t>(std::strtoul(chosen, nullptr, 10), 1, std::size(kArtPngs)) - 1;
    } else if (!(test && !std::strncmp(test, "shot:", 5))) {
        std::random_device rd;
        pick = std::uniform_int_distribution<std::size_t>(0, std::size(kArtPngs) - 1)(rd);
    }
    const ArtPng& art = kArtPngs[pick];
    host_log("setup: background %zu of %zu", pick + 1, std::size(kArtPngs));
    SDL_IOStream* io = SDL_IOFromConstMem(art.begin, static_cast<std::size_t>(art.end - art.begin));
    SDL_Surface* png = io ? SDL_LoadPNG_IO(io, true) : nullptr;
    SDL_Surface* s = png ? SDL_ConvertSurface(png, SDL_PIXELFORMAT_RGBA32) : nullptr;
    if (png) SDL_DestroySurface(png);
    if (!s) {
        host_log("setup: no background (%s)", SDL_GetError());
        return;
    }
    const int w = s->w, h = s->h;
    std::vector<float> grey(static_cast<std::size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* row = static_cast<const std::uint8_t*>(s->pixels) + static_cast<std::size_t>(y) * s->pitch;
        for (int x = 0; x < w; ++x) grey[static_cast<std::size_t>(y) * w + x] = row[x * 4];
    }
    SDL_DestroySurface(s);
    g_art.w = w;
    g_art.h = h;
    g_art.sharp = upload(grey, w, h);
    // Glass's copy: a quarter the size, blurred; the GPU's filtering smooths
    // the rest. The panes' copy: half the size, blurred a little.
    auto shrunk = [&](int k) {
        const int sw = w / k, sh = h / k;
        std::vector<float> small(static_cast<std::size_t>(sw) * sh);
        for (int y = 0; y < sh; ++y) {
            for (int x = 0; x < sw; ++x) {
                float acc = 0;
                for (int j = 0; j < k; ++j) {
                    for (int i = 0; i < k; ++i) acc += grey[static_cast<std::size_t>(y * k + j) * w + x * k + i];
                }
                small[static_cast<std::size_t>(y) * sw + x] = acc / static_cast<float>(k * k);
            }
        }
        return small;
    };
    std::vector<float> small = shrunk(4);
    blur(small, w / 4, h / 4, 4);
    g_art.soft = upload(small, w / 4, h / 4);
    std::vector<float> half = shrunk(2);
    blur(half, w / 2, h / 2, 2);
    g_art.mild = upload(half, w / 2, h / 2);
}

}  // namespace

ImU32 alpha(ImU32 col, int a) {
    return (col & ~IM_COL32_A_MASK) | (static_cast<ImU32>(std::clamp(a, 0, 255)) << IM_COL32_A_SHIFT);
}

void init(SDL_Renderer* ren, float scale) {
    g_ren = ren;
    g_scale = scale;
    load_fonts();
    apply_style(scale);
    load_art();
}

void init_ingame(float scale) {
    g_ren = nullptr;
    g_art = Art{};
    g_scale = scale;
    load_fonts();
    apply_style(scale);
}

void set_scale(float scale) {
    if (std::fabs(scale - g_scale) < 1e-3f) return;
    g_scale = scale;
    apply_style(scale);
}

void shutdown() {
    if (g_art.sharp) SDL_DestroyTexture(g_art.sharp);
    if (g_art.soft) SDL_DestroyTexture(g_art.soft);
    if (g_art.mild) SDL_DestroyTexture(g_art.mild);
    g_art = Art{};
    g_motes.clear();
    g_body = g_strong = g_title = nullptr;
    g_ren = nullptr;
}

float scale() { return g_scale; }
ImFont* font_strong() { return g_strong ? g_strong : ImGui::GetFont(); }
ImFont* font_title() { return g_title ? g_title : font_strong(); }

static bool g_animations = true;
void set_animations(bool on) { g_animations = on; }
bool animations() { return g_animations; }
// Still: a moment when the moon's glow is at its fullest and the Play
// button's shine is off the button.
float anim_time() { return g_animations ? static_cast<float>(ImGui::GetTime()) : 2.0f; }

float ease(ImGuiID id, bool on, float speed) {
    const float target = on ? 1.0f : 0.0f;
    float* v = ImGui::GetStateStorage()->GetFloatRef(id ^ 0x5bd1e995u, target);
    *v += g_animations ? (target - *v) * std::min(1.0f, ImGui::GetIO().DeltaTime * speed) : target - *v;
    if (std::fabs(target - *v) < 0.002f) *v = target;
    return *v;
}

void backdrop(float sidebar_w) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImVec2 o = vp->Pos, size = vp->Size;
    const float t = anim_time();
    dl->AddRectFilled(o, o + size, kInk);
    if (g_art.sharp) {
        const float ia = static_cast<float>(g_art.w) / static_cast<float>(g_art.h), wa = size.x / size.y;
        ImVec2 ext = wa > ia ? ImVec2(size.x, size.x / ia) : ImVec2(size.y * ia, size.y);
        // A slow drift - a few percent, over a minute - so the night is not a still.
        ext *= 1.04f + 0.02f * std::sin(t * 0.09f);
        const ImVec2 mid = o + size * 0.5f + ImVec2(std::sin(t * 0.05f) * size.x * 0.009f, std::cos(t * 0.04f) * size.y * 0.007f);
        g_cover0 = mid - ext * 0.5f;
        g_cover1 = mid + ext * 0.5f;
        dl->AddImage(tex(g_art.sharp), g_cover0, g_cover1);
    }
    // The veils: all over, then deeper toward the sidebar, the bottom and the top.
    const ImU32 clear = alpha(kInk, 0);
    dl->AddRectFilled(o, o + size, alpha(kInk, 64));
    dl->AddRectFilledMultiColor(o, o + ImVec2(sidebar_w * 2.4f, size.y), alpha(kInk, 170), clear, clear, alpha(kInk, 170));
    dl->AddRectFilledMultiColor(o + ImVec2(0, size.y * 0.42f), o + size, clear, clear, alpha(kInk, 215), alpha(kInk, 215));
    dl->AddRectFilledMultiColor(o, o + ImVec2(size.x, size.y * 0.16f), alpha(kInk, 120), alpha(kInk, 120), clear, clear);

    if (!g_animations) return;  // the motes are motion alone
    if (g_motes.empty()) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (int i = 0; i < 46; ++i) {
            g_motes.push_back({u(g_rng), u(g_rng), 0.8f + 1.4f * u(g_rng), 0.006f + 0.016f * u(g_rng), u(g_rng) * 6.28f,
                               0.004f + 0.01f * u(g_rng), u(g_rng) < 0.22f});
        }
    }
    const float dt = std::min(ImGui::GetIO().DeltaTime, 0.1f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (Mote& m : g_motes) {
        m.y -= m.speed * dt;
        m.x += std::sin(t * 0.5f + m.phase) * m.sway * dt;
        if (m.y < -0.03f) {
            m.y = 1.03f;
            m.x = u(g_rng);
        }
        const ImVec2 p = o + ImVec2(m.x * size.x, m.y * size.y);
        const float tw = 0.55f + 0.45f * std::sin(t * (1.1f + m.phase * 0.25f) + m.phase * 5.0f);
        const ImU32 col = m.pink ? kPink : kIce;
        const float r = m.r * g_scale;
        dl->AddCircleFilled(p, r * 4.0f, alpha(col, static_cast<int>(16 * tw)));
        dl->AddCircleFilled(p, r, alpha(col, static_cast<int>(150 * tw)));
    }
}

// A blurred copy of the painting behind a panel, lined up with the painting.
static void underlay(ImDrawList* dl, SDL_Texture* t, ImVec2 a, ImVec2 b, float rounding) {
    if (!t || g_cover1.x <= g_cover0.x) return;
    const ImVec2 span = g_cover1 - g_cover0;
    dl->AddImageRounded(tex(t), a, b, (a - g_cover0) / span, (b - g_cover0) / span, c(IM_COL32_WHITE), rounding);
}

static void tinted(ImDrawList* dl, ImVec2 a, ImVec2 b, float rounding, ImU32 tint, ImU32 edge) {
    dl->AddRectFilled(a, b, c(tint), rounding);
    // The light along the top edge, as glass catches it.
    dl->AddLine(ImVec2(a.x + rounding, a.y + 0.5f), ImVec2(b.x - rounding, a.y + 0.5f), c(alpha(kIce, 34)), 1.0f);
    if (edge) dl->AddRect(a, b, c(edge), rounding, 1.0f);
}

void glass(ImDrawList* dl, ImVec2 a, ImVec2 b, float rounding, ImU32 tint, ImU32 edge) {
    underlay(dl, g_art.soft, a, b, rounding);
    tinted(dl, a, b, rounding, tint, edge);
}

void pane(ImDrawList* dl, ImVec2 a, ImVec2 b, float rounding, ImU32 tint, ImU32 edge) {
    underlay(dl, g_art.mild, a, b, rounding);
    tinted(dl, a, b, rounding, tint, edge);
}

void draw_icon(ImDrawList* dl, Icon icon, ImVec2 at, float size, ImU32 col) {
    const float h = size * 0.5f;
    auto P = [&](float x, float y) { return ImVec2(at.x + x * h, at.y + y * h); };
    const float th = std::max(1.3f, size * 0.09f);
    switch (icon) {
    case Icon::Game:
        dl->AddRect(P(-0.95f, -0.52f), P(0.95f, 0.58f), col, 0.5f * h, th);
        dl->AddLine(P(-0.64f, 0.03f), P(-0.2f, 0.03f), col, th);
        dl->AddLine(P(-0.42f, -0.19f), P(-0.42f, 0.25f), col, th);
        dl->AddCircleFilled(P(0.3f, -0.05f), 0.12f * h, col);
        dl->AddCircleFilled(P(0.56f, 0.19f), 0.12f * h, col);
        break;
    case Icon::Online:
        dl->AddCircle(at, 0.9f * h, col, 0, th);
        dl->AddEllipse(at, ImVec2(0.38f * h, 0.9f * h), col, 0.0f, 0, th);
        dl->AddLine(P(-0.9f, 0.0f), P(0.9f, 0.0f), col, th);
        break;
    case Icon::Enhance:
        star(dl, P(-0.16f, 0.14f), 0.8f * h, 0.2f * h, col);
        star(dl, P(0.62f, -0.6f), 0.34f * h, 0.09f * h, col);
        break;
    case Icon::Patches: {
        const ImVec2 top[] = {P(0.0f, -0.88f), P(0.92f, -0.42f), P(0.0f, 0.04f), P(-0.92f, -0.42f)};
        dl->AddPolyline(top, 4, col, th, ImDrawFlags_Closed);
        const ImVec2 mid[] = {P(-0.92f, 0.04f), P(0.0f, 0.5f), P(0.92f, 0.04f)};
        dl->AddPolyline(mid, 3, col, th);
        const ImVec2 low[] = {P(-0.92f, 0.46f), P(0.0f, 0.92f), P(0.92f, 0.46f)};
        dl->AddPolyline(low, 3, col, th);
        break;
    }
    case Icon::Shaders: {
        const ImVec2 hex[] = {P(0.0f, -0.92f), P(0.82f, -0.46f), P(0.82f, 0.46f), P(0.0f, 0.92f), P(-0.82f, 0.46f), P(-0.82f, -0.46f)};
        dl->AddPolyline(hex, 6, col, th, ImDrawFlags_Closed);
        dl->AddLine(P(-0.82f, -0.46f), P(0.0f, 0.0f), col, th);
        dl->AddLine(P(0.82f, -0.46f), P(0.0f, 0.0f), col, th);
        dl->AddLine(P(0.0f, 0.0f), P(0.0f, 0.92f), col, th);
        break;
    }
    case Icon::Plugins:
        dl->AddRect(P(-0.56f, -0.28f), P(0.56f, 0.42f), col, 0.24f * h, th);
        dl->AddLine(P(-0.25f, -0.28f), P(-0.25f, -0.86f), col, th);
        dl->AddLine(P(0.25f, -0.28f), P(0.25f, -0.86f), col, th);
        dl->AddLine(P(0.0f, 0.42f), P(0.0f, 0.92f), col, th);
        break;
    case Icon::Settings: {
        ImVec2 pts[32];
        int n = 0;
        for (int k = 0; k < 8; ++k) {
            const float a = static_cast<float>(k) * IM_PI / 4.0f;
            const float rad[] = {0.64f, 0.94f, 0.94f, 0.64f};
            const float ang[] = {a - 0.27f, a - 0.15f, a + 0.15f, a + 0.27f};
            for (int j = 0; j < 4; ++j) pts[n++] = P(std::cos(ang[j]) * rad[j], std::sin(ang[j]) * rad[j]);
        }
        dl->AddPolyline(pts, n, col, th, ImDrawFlags_Closed);
        dl->AddCircle(at, 0.3f * h, col, 0, th);
        break;
    }
    case Icon::Folder: {
        const ImVec2 p[] = {P(-0.92f, -0.66f), P(-0.32f, -0.66f), P(-0.14f, -0.44f), P(0.92f, -0.44f), P(0.92f, 0.72f), P(-0.92f, 0.72f)};
        dl->AddPolyline(p, 6, col, th, ImDrawFlags_Closed);
        break;
    }
    case Icon::File: {
        const ImVec2 p[] = {P(-0.62f, -0.92f), P(0.22f, -0.92f), P(0.66f, -0.48f), P(0.66f, 0.92f), P(-0.62f, 0.92f)};
        dl->AddPolyline(p, 5, col, th, ImDrawFlags_Closed);
        const ImVec2 f[] = {P(0.22f, -0.92f), P(0.22f, -0.48f), P(0.66f, -0.48f)};
        dl->AddPolyline(f, 3, col, th);
        break;
    }
    case Icon::Check: {
        dl->AddCircleFilled(at, 0.92f * h, col);
        const ImVec2 p[] = {P(-0.42f, 0.02f), P(-0.12f, 0.32f), P(0.44f, -0.3f)};
        dl->AddPolyline(p, 3, c(kInk), th * 1.25f);
        break;
    }
    case Icon::Warning:
        dl->AddTriangleFilled(P(0.0f, -0.92f), P(0.98f, 0.82f), P(-0.98f, 0.82f), col);
        dl->AddLine(P(0.0f, -0.32f), P(0.0f, 0.26f), c(kInk), th * 1.25f);
        dl->AddCircleFilled(P(0.0f, 0.54f), th * 0.75f, c(kInk));
        break;
    case Icon::Info:
        dl->AddCircle(at, 0.9f * h, col, 0, th);
        dl->AddCircleFilled(P(0.0f, -0.42f), th * 0.85f, col);
        dl->AddLine(P(0.0f, -0.08f), P(0.0f, 0.5f), col, th);
        break;
    case Icon::Play:
        dl->AddTriangleFilled(P(-0.55f, -0.78f), P(0.82f, 0.0f), P(-0.55f, 0.78f), col);
        break;
    case Icon::Moon:
        for (int i = 4; i >= 1; --i) dl->AddCircleFilled(at, h * (0.9f + static_cast<float>(i) * 0.28f), alpha(col, 12));
        dl->AddCircleFilled(at, 0.9f * h, col);
        break;
    case Icon::Search:
        dl->AddCircle(P(-0.15f, -0.15f), 0.58f * h, col, 0, th);
        dl->AddLine(P(0.28f, 0.28f), P(0.86f, 0.86f), col, th * 1.2f);
        break;
    case Icon::Copy: {
        dl->AddRect(P(-0.78f, -0.42f), P(0.38f, 0.88f), col, 0.16f * h, th);
        const ImVec2 p[] = {P(-0.42f, -0.42f), P(-0.42f, -0.88f), P(0.78f, -0.88f), P(0.78f, 0.5f), P(0.38f, 0.5f)};
        dl->AddPolyline(p, 5, col, th);
        break;
    }
    case Icon::User:
        dl->AddCircle(P(0.0f, -0.38f), 0.36f * h, col, 0, th);
        dl->PathArcTo(P(0.0f, 0.98f), 0.78f * h, IM_PI * 1.1f, IM_PI * 1.9f);
        dl->PathStroke(col, th);
        break;
    case Icon::Download: {
        dl->AddLine(P(0.0f, -0.88f), P(0.0f, 0.3f), col, th);
        const ImVec2 head[] = {P(-0.42f, -0.08f), P(0.0f, 0.32f), P(0.42f, -0.08f)};
        dl->AddPolyline(head, 3, col, th);
        const ImVec2 tray[] = {P(-0.88f, 0.36f), P(-0.88f, 0.86f), P(0.88f, 0.86f), P(0.88f, 0.36f)};
        dl->AddPolyline(tray, 4, col, th);
        break;
    }
    case Icon::Refresh: {
        const float a1 = IM_PI * 1.75f;
        dl->PathArcTo(at, 0.72f * h, IM_PI * 0.15f, a1);
        dl->PathStroke(col, th);
        const ImVec2 e = at + ImVec2(std::cos(a1), std::sin(a1)) * (0.72f * h);
        const ImVec2 tan(-std::sin(a1), std::cos(a1)), nor(std::cos(a1), std::sin(a1));
        dl->AddTriangleFilled(e + tan * (0.34f * h), e + nor * (0.3f * h), e - nor * (0.3f * h), col);
        break;
    }
    case Icon::SignIn: {
        const ImVec2 door[] = {P(0.15f, -0.88f), P(0.88f, -0.88f), P(0.88f, 0.88f), P(0.15f, 0.88f)};
        dl->AddPolyline(door, 4, col, th);
        dl->AddLine(P(-0.88f, 0.0f), P(0.4f, 0.0f), col, th);
        const ImVec2 head[] = {P(0.04f, -0.36f), P(0.42f, 0.0f), P(0.04f, 0.36f)};
        dl->AddPolyline(head, 3, col, th);
        break;
    }
    case Icon::Display:
        dl->AddRect(P(-0.92f, -0.72f), P(0.92f, 0.46f), col, 0.14f * h, th);
        dl->AddLine(P(0.0f, 0.46f), P(0.0f, 0.82f), col, th);
        dl->AddLine(P(-0.42f, 0.84f), P(0.42f, 0.84f), col, th);
        break;
    case Icon::Image: {
        dl->AddRect(P(-0.92f, -0.72f), P(0.92f, 0.72f), col, 0.14f * h, th);
        const ImVec2 hills[] = {P(-0.92f, 0.5f), P(-0.3f, -0.12f), P(0.12f, 0.3f), P(0.4f, 0.04f), P(0.92f, 0.5f)};
        dl->AddPolyline(hills, 5, col, th);
        dl->AddCircleFilled(P(0.44f, -0.36f), 0.15f * h, col);
        break;
    }
    }
}

float spaced_width(ImFont* font, float size, const char* text, float tracking) {
    float w = 0;
    int n = 0;
    for (const char* ch = text; *ch; ++ch, ++n) w += font->CalcTextSizeA(size, FLT_MAX, 0.0f, ch, ch + 1).x;
    return n ? w + tracking * static_cast<float>(n - 1) : 0.0f;
}

void spaced_text(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, ImU32 col, const char* text, float tracking) {
    for (const char* ch = text; *ch; ++ch) {
        dl->AddText(font, size, pos, col, ch, ch + 1);
        pos.x += font->CalcTextSizeA(size, FLT_MAX, 0.0f, ch, ch + 1).x + tracking;
    }
}

void logo(const char* name, const char* version, float width, bool compact) {
    const float s = g_scale;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float t = anim_time();
    const float top = compact ? 6 * s : 74 * s;
    if (!compact) {
        // The pale moon, breathing, and a few stars about it.
        const ImVec2 mc(p.x + width * 0.5f, p.y + 36 * s);
        const float mr = 23 * s;
        const float breathe = 0.5f + 0.5f * std::sin(t * 0.8f);
        for (int i = 7; i >= 1; --i) {
            dl->AddCircleFilled(mc, mr + static_cast<float>(i) * 5.5f * s, c(alpha(kViolet, static_cast<int>(9 + 5 * breathe))));
        }
        dl->AddCircleFilled(mc, mr, c(IM_COL32(226, 234, 255, 255)));
        dl->AddCircleFilled(mc + ImVec2(-7, -5) * s, 7.5f * s, c(IM_COL32(196, 210, 246, 150)));
        dl->AddCircleFilled(mc + ImVec2(8, 7) * s, 5.0f * s, c(IM_COL32(196, 210, 246, 120)));
        dl->AddCircleFilled(mc + ImVec2(4, -11) * s, 3.5f * s, c(IM_COL32(196, 210, 246, 110)));
        const ImVec2 stars[] = {ImVec2(-52, -14), ImVec2(46, -22), ImVec2(58, 10), ImVec2(-40, 18)};
        for (int i = 0; i < 4; ++i) {
            const float tw = 0.5f + 0.5f * std::sin(t * (1.4f + 0.3f * static_cast<float>(i)) + static_cast<float>(i) * 1.7f);
            const float r = (i % 2 ? 4.5f : 3.2f) * s * (0.75f + 0.25f * tw);
            star(dl, mc + stars[i] * s, r, r * 0.28f, c(alpha(i == 1 ? kPink : kIce, static_cast<int>(120 + 120 * tw))));
        }
    }
    // The name, spaced, over a violet glow.
    ImFont* f = font_title();
    float fs = 29 * s, tr = 7 * s;
    // Smaller where the column is narrower than the name (a large Menu text size).
    if (const float nw = spaced_width(f, fs, name, tr) + 6 * s; nw > width) {
        fs *= width / nw;
        tr *= width / nw;
    }
    const ImVec2 np(p.x + (width - spaced_width(f, fs, name, tr)) * 0.5f, p.y + top);
    for (int i = 0; i < 8; ++i) {
        const float a = static_cast<float>(i) * IM_PI / 4.0f;
        spaced_text(dl, f, fs, np + ImVec2(std::cos(a), std::sin(a)) * (2.2f * s), c(alpha(kViolet, 46)), name, tr);
    }
    spaced_text(dl, f, fs, np, c(kText), name, tr);
    float vs = 12.5f * s;
    ImFont* vf = g_body ? g_body : ImGui::GetFont();
    ImVec2 vsz = vf->CalcTextSizeA(vs, FLT_MAX, 0.0f, version);
    if (vsz.x > width) {  // smaller where the column is narrower (a large Menu text size)
        vs *= width / vsz.x;
        vsz = vf->CalcTextSizeA(vs, FLT_MAX, 0.0f, version);
    }
    dl->AddText(vf, vs, ImVec2(p.x + (width - vsz.x) * 0.5f, np.y + fs + 8 * s), c(kTextFaint), version);
    ImGui::Dummy(ImVec2(width, top + fs + 8 * s + vsz.y + 6 * s));
}

void card_begin(const char* id, ImVec2 size, const char* cap) {
    const float s = g_scale;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(18 * s, 14 * s));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(0, 0, 0, 0));
    ImGuiChildFlags cf = ImGuiChildFlags_AlwaysUseWindowPadding;
    if (size.y == 0) cf |= ImGuiChildFlags_AutoResizeY;
    ImGui::BeginChild(id, size, cf, size.y == 0 ? ImGuiWindowFlags_NoScrollbar : 0);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    ImGuiWindow* w = ImGui::GetCurrentWindow();
    // The whole card, edges included - but only what its parent shows: a card
    // scrolled partly out of a page is cut at the page's edge.
    ImRect clip(w->Pos, w->Pos + w->Size);
    if (w->ParentWindow) clip.ClipWithFull(w->ParentWindow->ClipRect);
    w->DrawList->PushClipRect(clip.Min, clip.Max, false);
    pane(w->DrawList, w->Pos, w->Pos + w->Size, ImGui::GetStyle().ChildRounding, kCardTint, alpha(kViolet, 58));
    w->DrawList->PopClipRect();
    if (cap) caption(cap);
}

void card_end() { ImGui::EndChild(); }

std::string fit_text(const std::string& text, float width) {
    if (ImGui::CalcTextSize(text.c_str()).x <= width) return text;
    const float dots = ImGui::CalcTextSize("...").x;
    std::size_t n = text.size();
    while (n > 0 && ImGui::CalcTextSize(text.c_str(), text.c_str() + n).x + dots > width) {
        --n;
        while (n > 0 && (static_cast<unsigned char>(text[n]) & 0xC0) == 0x80) --n;  // whole UTF-8 characters
    }
    while (n > 0 && text[n - 1] == ' ') --n;
    return text.substr(0, n) + "...";
}

void child_cards() {
    // A child's own draw list comes after its parent's, whatever the order
    // here, so these land under the panels.
    ImGuiWindow* w = ImGui::GetCurrentWindow();
    for (ImGuiWindow* child : w->DC.ChildWindows) {
        if (child->Active) pane(w->DrawList, child->Pos, child->Pos + child->Size, child->WindowRounding, kCardTint, 0);
    }
}

void caption(const char* text) {
    const float s = g_scale, fs = 11.5f * s, tr = 1.7f * s;
    ImFont* f = font_strong();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    spaced_text(ImGui::GetWindowDrawList(), f, fs, p, c(alpha(kIce, 175)), text, tr);
    ImGui::Dummy(ImVec2(spaced_width(f, fs, text, tr), fs + 4 * s));
}

void row_label(const char* label, float x) {
    const float x0 = ImGui::GetCursorScreenPos().x;
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::SetCursorScreenPos(ImVec2(x0 + x, ImGui::GetCursorScreenPos().y));
}

bool toggle(const char* id, bool* v) {
    const ImVec2 sz = switch_size();
    const float fh = ImGui::GetFrameHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const bool pressed = ImGui::InvisibleButton(id, ImVec2(sz.x, fh));
    if (pressed) *v = !*v;
    const float k = ease(ImGui::GetItemID(), *v, 14.0f);
    draw_switch(ImGui::GetWindowDrawList(), ImVec2(p.x, p.y + (fh - sz.y) * 0.5f), k, ImGui::IsItemHovered());
    return pressed;
}

bool toggle_row(const char* label, const char* sub, bool* v, const char* tip) {
    const float s = g_scale;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    const ImVec2 sw = switch_size();
    const float text_w = w - sw.x - 16 * s;
    ImFont* f = ImGui::GetFont();
    const float fs = ImGui::GetFontSize(), sub_fs = std::round(fs * 0.86f);
    const float lh = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, label).y;
    const float sub_h = sub ? f->CalcTextSizeA(sub_fs, FLT_MAX, text_w, sub).y + 2 * s : 0.0f;
    const float h = std::max(ImGui::GetFrameHeight(), lh + sub_h);
    ImGui::PushID(label);
    const bool pressed = ImGui::InvisibleButton("##row", ImVec2(w, h));
    if (pressed) *v = !*v;
    const bool hovered = ImGui::IsItemHovered();
    if (tip && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) tooltip(label, tip);
    const float k = ease(ImGui::GetItemID(), *v, 14.0f);
    ImGui::PopID();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float ty = p.y + (h - lh - sub_h) * 0.5f;
    dl->AddText(f, fs, ImVec2(p.x, ty), c(kText), label);
    if (sub) dl->AddText(f, sub_fs, ImVec2(p.x, ty + lh + 2 * s), c(kTextDim), sub, nullptr, text_w);
    draw_switch(dl, ImVec2(p.x + w - sw.x, p.y + (h - sw.y) * 0.5f), k, hovered);
    return pressed;
}

bool segmented(const char* id, int* v, const char* const* labels, int n, float width, const char* const* tips, const char* note) {
    const float s = g_scale, h = ImGui::GetFrameHeight();
    if (width <= 0) width = ImGui::GetContentRegionAvail().x;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float r = ImGui::GetStyle().FrameRounding;
    ImGui::PushID(id);
    dl->AddRectFilled(p, p + ImVec2(width, h), ImGui::GetColorU32(ImGuiCol_FrameBg), r);
    const float seg = width / static_cast<float>(n);
    // The lit one slides to the choice.
    float* at = ImGui::GetStateStorage()->GetFloatRef(ImGui::GetID("##at"), static_cast<float>(*v));
    *at += g_animations ? (static_cast<float>(*v) - *at) * std::min(1.0f, ImGui::GetIO().DeltaTime * 14.0f) : static_cast<float>(*v) - *at;
    const ImVec2 la(p.x + *at * seg + 3 * s, p.y + 3 * s);
    gradient_rect(dl, la, ImVec2(la.x + seg - 6 * s, p.y + h - 3 * s), kBlue, kViolet, std::max(0.0f, r - 2 * s));
    bool changed = false;
    for (int i = 0; i < n; ++i) {
        ImGui::SetCursorScreenPos(ImVec2(p.x + static_cast<float>(i) * seg, p.y));
        ImGui::PushID(i);
        if (ImGui::InvisibleButton("##seg", ImVec2(seg, h)) && *v != i) {
            *v = i;
            changed = true;
        }
        const bool hov = ImGui::IsItemHovered();
        if (tips && tips[i]) tooltip_for_item(labels[i], tips[i], note ? note : "");
        ImGui::PopID();
        const ImVec2 ts = ImGui::CalcTextSize(labels[i]);
        dl->AddText(ImVec2(p.x + static_cast<float>(i) * seg + (seg - ts.x) * 0.5f, p.y + (h - ts.y) * 0.5f),
                    c(i == *v || hov ? kText : kTextDim), labels[i]);
    }
    ImGui::PopID();
    ImGui::SetCursorScreenPos(p);
    ImGui::Dummy(ImVec2(width, h));
    return changed;
}

bool combo(const char* id, int* v, const char* const* items, int n, const char* const* tips) {
    // The frame's place, before the popup (when open) becomes the window.
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::CalcItemWidth(), h = ImGui::GetFrameHeight();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    bool changed = false;
    if (ImGui::BeginCombo(id, *v >= 0 && *v < n ? items[*v] : "", ImGuiComboFlags_NoArrowButton | ImGuiComboFlags_HeightLarge)) {
        for (int i = 0; i < n; ++i) {
            const bool chosen = i == *v;
            if (ImGui::Selectable(items[i], chosen) && !chosen) {
                *v = i;
                changed = true;
            }
            if (tips && tips[i]) tooltip_for_item(items[i], tips[i]);
            if (chosen) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    const float s = g_scale, x = p.x + w - h * 0.55f, y = p.y + h * 0.5f;
    const ImVec2 chev[] = {ImVec2(x - 4.5f * s, y - 2.2f * s), ImVec2(x, y + 2.4f * s), ImVec2(x + 4.5f * s, y - 2.2f * s)};
    dl->AddPolyline(chev, 3, c(kIce), 1.6f * s);
    return changed;
}

std::string fit_left(const std::string& text, float width) {
    if (ImGui::CalcTextSize(text.c_str()).x <= width) return text;
    for (std::size_t cut = 1; cut < text.size(); ++cut) {
        const std::string t = "..." + text.substr(cut);
        if (ImGui::CalcTextSize(t.c_str()).x <= width) return t;
    }
    return "...";
}

bool primary_button(const char* label, ImVec2 size, bool enabled) {
    const float s = g_scale;
    const char* end = ImGui::FindRenderedTextEnd(label);
    const ImVec2 ts = ImGui::CalcTextSize(label, end);
    if (size.x == 0) size.x = ts.x + ImGui::GetStyle().FramePadding.x * 3.0f;
    else if (size.x < 0) size.x += ImGui::GetContentRegionAvail().x;
    if (size.y <= 0) size.y = ImGui::GetFrameHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::BeginDisabled(!enabled);
    const bool pressed = ImGui::InvisibleButton(label, size);
    const bool hovered = ImGui::IsItemHovered(), held = ImGui::IsItemActive();
    const float k = ease(ImGui::GetItemID(), hovered, 12.0f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float r = ImGui::GetStyle().FrameRounding;
    const ImVec2 q = p + size;
    if (enabled) {
        for (int i = 3; i >= 1; --i) {
            const float g = static_cast<float>(i) * 1.6f * s;
            dl->AddRectFilled(p - ImVec2(g, g), q + ImVec2(g, g), c(alpha(kViolet, static_cast<int>((12 + 18 * k) / i))), r + g);
        }
    }
    gradient_rect(dl, p, q, enabled ? kBlue : IM_COL32(64, 62, 104, 255), enabled ? kViolet : IM_COL32(74, 64, 118, 255), r);
    if (held) dl->AddRectFilled(p, q, c(IM_COL32(0, 0, 0, 44)), r);
    else if (k > 0.01f) dl->AddRectFilled(p, q, c(IM_COL32(255, 255, 255, static_cast<int>(26 * k))), r);
    dl->AddText(p + (size - ts) * 0.5f, c(kText), label, end);
    ImGui::EndDisabled();
    return pressed && enabled;
}

bool icon_button(const char* id, Icon icon, float size, const char* tip) {
    if (size <= 0) size = ImGui::GetFrameHeight();
    const bool pressed = ImGui::Button(id, ImVec2(size, size));
    const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    draw_icon(ImGui::GetWindowDrawList(), icon, (a + b) * 0.5f, size * 0.5f, c(ImGui::IsItemHovered() ? kText : kIce));
    if (tip && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) tooltip(nullptr, tip);
    return pressed;
}

bool ghost_button(const char* label, ImVec2 size) {
    ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(255, 255, 255, 14));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(255, 255, 255, 32));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, alpha(kViolet, 120));
    const bool pressed = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);
    return pressed;
}

void avatar(const std::string& name, float size) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(size, size));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 mid = p + ImVec2(size, size) * 0.5f;
    const float r = size * 0.5f;
    if (name.empty()) {
        dl->AddCircleFilled(mid, r, c(IM_COL32(255, 255, 255, 20)));
        draw_icon(dl, Icon::User, mid + ImVec2(0, r * 0.08f), size * 0.5f, c(kTextDim));
        return;
    }
    const int v0 = dl->VtxBuffer.Size;
    dl->AddCircleFilled(mid, r, c(IM_COL32_WHITE));
    ImGui::ShadeVertsLinearColorGradientKeepAlpha(dl, v0, dl->VtxBuffer.Size, p, p + ImVec2(size, size), kBlue, kViolet);
    dl->AddCircle(mid, r, c(alpha(kIce, 110)), 0, 1.0f);
    const char initial[2] = {static_cast<char>(std::toupper(static_cast<unsigned char>(name[0]))), 0};
    ImFont* f = font_strong();
    const float fs = size * 0.48f;
    const ImVec2 ts = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, initial);
    dl->AddText(f, fs, mid - ts * 0.5f, c(kText), initial);
}

bool nav_item(const char* label, Icon icon, bool selected, ImU32 dot, const char* badge) {
    const float s = g_scale;
    const float w = ImGui::GetContentRegionAvail().x, h = 40 * s;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const bool pressed = ImGui::InvisibleButton(label, ImVec2(w, h));
    const bool hovered = ImGui::IsItemHovered();
    const ImGuiID id = ImGui::GetItemID();
    const float ks = ease(id ^ 1u, selected, 12.0f), kh = ease(id ^ 2u, hovered, 16.0f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float r = 9 * s;
    if (kh > 0.01f) dl->AddRectFilled(p, p + ImVec2(w, h), c(IM_COL32(255, 255, 255, static_cast<int>(13 * kh))), r);
    if (ks > 0.01f) {
        dl->AddRectFilled(p, p + ImVec2(w, h), c(alpha(kViolet, static_cast<int>(58 * ks))), r);
        dl->AddRect(p, p + ImVec2(w, h), c(alpha(kViolet, static_cast<int>(80 * ks))), r, 1.0f);
        gradient_rect(dl, ImVec2(p.x + 1 * s, p.y + h * 0.24f), ImVec2(p.x + 4.5f * s, p.y + h * 0.76f), kBlue, kViolet, 2 * s,
                      static_cast<int>(255 * ks), true);
    }
    const float lit = std::max(ks, kh * 0.6f);
    draw_icon(dl, icon, ImVec2(p.x + 25 * s, p.y + h * 0.5f), 18 * s, c(mix(kTextDim, kIce, lit)));
    const float fs = ImGui::GetFontSize();
    // Cut to the row (a large Menu text size in a narrow sidebar), clear of the badge or dot.
    // Where not even a few letters fit, the icon alone, the name in its bubble.
    const std::string name(label, ImGui::FindRenderedTextEnd(label));
    const float room = w - 48 * s - ((badge && *badge) || dot ? 34 * s : 8 * s);
    if (room >= ImGui::CalcTextSize("Abc...").x) {
        dl->AddText(ImVec2(p.x + 48 * s, p.y + (h - fs) * 0.5f - 1 * s), c(mix(kTextDim, kText, lit)), fit_text(name, room).c_str());
    } else if (hovered) {
        tooltip(name.c_str(), std::string());
    }
    if (badge && *badge) {
        const float bs = std::round(fs * 0.8f);
        const ImVec2 bsz = font_strong()->CalcTextSizeA(bs, FLT_MAX, 0.0f, badge);
        dl->AddText(font_strong(), bs, ImVec2(p.x + w - 12 * s - bsz.x, p.y + (h - bsz.y) * 0.5f), c(dot ? dot : kIce), badge);
    } else if (dot) {
        dl->AddCircleFilled(ImVec2(p.x + w - 16 * s, p.y + h * 0.5f), 4 * s, c(dot));
        dl->AddCircleFilled(ImVec2(p.x + w - 16 * s, p.y + h * 0.5f), 8 * s, c(alpha(dot, 40)));
    }
    return pressed;
}

bool play_button(const char* label, ImVec2 size, bool enabled) {
    const float s = g_scale;
    const ImVec2 p = ImGui::GetCursorScreenPos(), q = p + size;
    ImGui::BeginDisabled(!enabled);
    const bool pressed = ImGui::InvisibleButton("##play", size);
    const bool hovered = ImGui::IsItemHovered(), held = ImGui::IsItemActive();
    ImGui::EndDisabled();
    const float kh = ease(ImGui::GetItemID(), hovered && enabled, 10.0f);
    const float t = anim_time();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float r = 12 * s;
    if (enabled) {
        const float pulse = 0.5f + 0.5f * std::sin(t * 2.0f);
        for (int i = 6; i >= 1; --i) {
            const float g = static_cast<float>(i) * 2.2f * s;
            dl->AddRectFilled(p - ImVec2(g, g), q + ImVec2(g, g), c(alpha(kViolet, static_cast<int>(6 + 7 * kh + 3 * pulse))), r + g);
        }
    }
    gradient_rect(dl, p, q, enabled ? kBlue : IM_COL32(56, 54, 92, 255), enabled ? kViolet : IM_COL32(68, 60, 110, 255), r);
    // A light sweeping across now and then.
    if (enabled) {
        const float ph = std::fmod(t, 3.6f) / 3.6f;
        const float bw = size.x * 0.16f, x = p.x - bw * 1.5f + ph * (size.x * 2.4f);
        if (x < q.x) {
            dl->PushClipRect(p + ImVec2(r * 0.3f, 0), q - ImVec2(r * 0.3f, 0), true);
            dl->AddQuadFilled(ImVec2(x, q.y), ImVec2(x + bw * 0.6f, p.y), ImVec2(x + bw * 1.2f, p.y), ImVec2(x + bw * 0.6f, q.y),
                              c(IM_COL32(255, 255, 255, 30)));
            dl->PopClipRect();
        }
    }
    if (held && enabled) dl->AddRectFilled(p, q, c(IM_COL32(0, 0, 0, 44)), r);
    else if (kh > 0.01f) dl->AddRectFilled(p, q, c(IM_COL32(255, 255, 255, static_cast<int>(24 * kh))), r);
    dl->AddRect(p, q, c(alpha(kIce, enabled ? 96 : 34)), r, 1.0f);
    ImFont* f = font_title();
    const float fs = 21 * s, tr = 5 * s;
    const float lw = spaced_width(f, fs, label, tr), iw = 13 * s, gap = 12 * s;
    const float x0 = p.x + (size.x - (iw + gap + lw)) * 0.5f, cy = p.y + size.y * 0.5f;
    const ImU32 ink = c(enabled ? kText : kTextFaint);
    draw_icon(dl, Icon::Play, ImVec2(x0 + iw * 0.5f, cy), iw * 1.15f, ink);
    const float th = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, label).y;
    spaced_text(dl, f, fs, ImVec2(x0 + iw + gap, cy - th * 0.5f + fs * 0.04f), ink, label, tr);
    return pressed && enabled;
}

void pill(const char* text, ImU32 fg, ImU32 bg) {
    const float s = g_scale, fs = std::round(ImGui::GetFontSize() * 0.8f);
    ImFont* f = font_strong();
    const ImVec2 ts = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, text);
    const ImVec2 pad(8 * s, 2.5f * s), sz(ts.x + pad.x * 2, ts.y + pad.y * 2);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float fh = ImGui::GetFrameHeight(), y = p.y + (fh - sz.y) * 0.5f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(ImVec2(p.x, y), ImVec2(p.x + sz.x, y + sz.y), c(bg), sz.y * 0.5f);
    dl->AddText(f, fs, ImVec2(p.x + pad.x, y + pad.y), c(fg), text);
    ImGui::Dummy(ImVec2(sz.x, fh));
}

void key_hint(const char* key, const char* what) {
    const float s = g_scale, fs = std::round(ImGui::GetFontSize() * 0.78f);
    ImFont* f = font_strong();
    const ImVec2 ks = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, key);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float cap_w = std::max(ks.x + 12 * s, 30 * s), h = std::round(fs + 8 * s);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, p + ImVec2(cap_w, h), c(IM_COL32(255, 255, 255, 18)), 5 * s);
    dl->AddRect(p, p + ImVec2(cap_w, h), c(alpha(kIce, 60)), 5 * s);
    dl->AddText(f, fs, p + ImVec2((cap_w - ks.x) * 0.5f, (h - ks.y) * 0.5f), c(kIce), key);
    dl->AddText(ImGui::GetFont(), fs, p + ImVec2(cap_w + 9 * s, (h - ks.y) * 0.5f), c(kTextDim), what);
    ImGui::Dummy(ImVec2(cap_w, h));
}

void status_icon(bool ok, const char* tip) {
    const float fh = ImGui::GetFrameHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(fh * 0.72f, fh));
    draw_icon(ImGui::GetWindowDrawList(), ok ? Icon::Check : Icon::Warning, ImVec2(p.x + fh * 0.36f, p.y + fh * 0.5f), fh * 0.6f,
              c(ok ? kGood : kWarn));
    if (tip && ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) tooltip(nullptr, tip);
}

void progress(float frac, const char* overlay, float height) {
    const float w = ImGui::GetContentRegionAvail().x, h = height > 0 ? height : std::round(ImGui::GetFrameHeight() * 0.72f);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(w, h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, p + ImVec2(w, h), c(IM_COL32(255, 255, 255, 22)), h * 0.5f);
    frac = std::clamp(frac, 0.0f, 1.0f);
    if (frac > 0) gradient_rect(dl, p, p + ImVec2(std::max(h, w * frac), h), kBlue, kViolet, h * 0.5f);
    if (overlay && *overlay) {
        const float fs = std::round(ImGui::GetFontSize() * 0.82f);
        ImFont* f = font_strong();
        const ImVec2 ts = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, overlay);
        dl->AddText(f, fs, p + ImVec2((w - ts.x) * 0.5f, (h - ts.y) * 0.5f), c(kText), overlay);
    }
}

void tooltip(const char* title, const std::string& body, const std::string& note) {
    const float s = g_scale;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14 * s, 11 * s));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8 * s, 5 * s));
    if (ImGui::BeginTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 340 * s);
        if (title && *title) {
            ImGui::PushFont(font_strong(), 0.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, kIce);
            ImGui::TextUnformatted(title);
            ImGui::PopStyleColor();
            ImGui::PopFont();
        }
        if (!body.empty()) ImGui::TextUnformatted(body.c_str());
        if (!note.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, kTextDim);
            ImGui::TextUnformatted(note.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
    ImGui::PopStyleVar(2);
}

void tooltip_for_item(const char* title, const std::string& body, const std::string& note) {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) tooltip(title, body, note);
}

void banner(Icon icon, ImU32 col, const std::string& text) {
    const float s = g_scale;
    const float w = ImGui::GetContentRegionAvail().x;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float pad = 13 * s, iw = 17 * s;
    const float tw = w - pad * 3 - iw;
    const ImVec2 ts = ImGui::CalcTextSize(text.c_str(), nullptr, false, tw);
    const float h = std::max(ts.y, iw) + pad * 1.5f;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    pane(dl, p, p + ImVec2(w, h), 10 * s, mix(kCardTint, alpha(col, 182), 0.16f), alpha(col, 110));
    draw_icon(dl, icon, ImVec2(p.x + pad + iw * 0.5f, p.y + h * 0.5f), iw, c(col));
    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(p.x + pad * 2 + iw, p.y + (h - ts.y) * 0.5f), c(mix(col, kText, 0.35f)),
                text.c_str(), nullptr, tw);
    ImGui::Dummy(ImVec2(w, h));
}

}  // namespace launcher_ui
