#include "host/japanese_font.h"
#include "host/ingame_menu.h"

#if defined(BBHOST_HAVE_SDL3)
#include "host/gpu.h"
#include "host/plugin_ui.h"
#include "host/settings.h"
#include "log.h"

#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::atomic<bool> g_open{false};

// One input event, translated on the pump thread and applied on the
// presenter's (ImGui's IO is not thread-safe).
struct Input {
    enum Kind { Move, Button, Wheel, Key, Text, Focus } kind = Move;
    float x = 0, y = 0;
    int button = 0;
    bool down = false;
    ImGuiKey key = ImGuiKey_None;
    std::string text;
};
std::mutex g_mu;
std::vector<Input> g_queue;  // under g_mu

ImGuiContext* g_ctx = nullptr;
bool g_vk_ready = false;
VkFormat g_format = VK_FORMAT_UNDEFINED;
std::uint32_t g_images = 0;
PluginUi g_ui;
std::chrono::steady_clock::time_point g_last;

ImGuiKey key_of(SDL_Keycode k) {
    switch (k) {
    case SDLK_TAB: return ImGuiKey_Tab;
    case SDLK_LEFT: return ImGuiKey_LeftArrow;
    case SDLK_RIGHT: return ImGuiKey_RightArrow;
    case SDLK_UP: return ImGuiKey_UpArrow;
    case SDLK_DOWN: return ImGuiKey_DownArrow;
    case SDLK_PAGEUP: return ImGuiKey_PageUp;
    case SDLK_PAGEDOWN: return ImGuiKey_PageDown;
    case SDLK_HOME: return ImGuiKey_Home;
    case SDLK_END: return ImGuiKey_End;
    case SDLK_DELETE: return ImGuiKey_Delete;
    case SDLK_BACKSPACE: return ImGuiKey_Backspace;
    case SDLK_RETURN: return ImGuiKey_Enter;
    case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
    case SDLK_SPACE: return ImGuiKey_Space;
    case SDLK_LCTRL: return ImGuiKey_LeftCtrl;
    case SDLK_RCTRL: return ImGuiKey_RightCtrl;
    case SDLK_LSHIFT: return ImGuiKey_LeftShift;
    case SDLK_RSHIFT: return ImGuiKey_RightShift;
    case SDLK_A: return ImGuiKey_A;
    case SDLK_C: return ImGuiKey_C;
    case SDLK_V: return ImGuiKey_V;
    case SDLK_X: return ImGuiKey_X;
    default: return ImGuiKey_None;
    }
}

void push(Input in) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_queue.size() < 512) g_queue.push_back(std::move(in));
}

void set_open(bool open) {
    if (g_open.exchange(open) == open) return;
    host_log("plugin menu: %s", open ? "open" : "closed");
    if (open) {
        Input in;
        in.kind = Input::Focus;
        push(in);
    }
}

}  // namespace

bool ingame_menu_open() { return g_open.load(std::memory_order_relaxed); }

bool ingame_menu_event(const SDL_Event& e, float ratio) {
    if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_F9) {
        if (!e.key.repeat) {
            set_open(!ingame_menu_open());
            if (ingame_menu_open()) {
                // Where the pointer is now: no motion may come before the first click.
                Input in;
                SDL_GetMouseState(&in.x, &in.y);
                in.x *= ratio;
                in.y *= ratio;
                push(in);
            }
        }
        return true;
    }
    if (!ingame_menu_open()) return false;
    switch (e.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) {
            set_open(false);
            return true;
        }
        if (const ImGuiKey k = key_of(e.key.key); k != ImGuiKey_None) {
            Input in;
            in.kind = Input::Key;
            in.key = k;
            in.down = e.type == SDL_EVENT_KEY_DOWN;
            push(in);
        }
        return true;
    case SDL_EVENT_TEXT_INPUT:
        if (e.text.text) {
            Input in;
            in.kind = Input::Text;
            in.text = e.text.text;
            push(in);
        }
        return true;
    case SDL_EVENT_MOUSE_MOTION: {
        Input in;
        in.kind = Input::Move;
        in.x = e.motion.x * ratio;
        in.y = e.motion.y * ratio;
        push(in);
        return true;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        Input in;
        in.kind = Input::Button;
        in.x = e.button.x * ratio;
        in.y = e.button.y * ratio;
        in.button = e.button.button == SDL_BUTTON_LEFT ? 0 : e.button.button == SDL_BUTTON_RIGHT ? 1 : 2;
        in.down = e.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
        push(in);
        return true;
    }
    case SDL_EVENT_MOUSE_WHEEL: {
        Input in;
            in.kind = Input::Wheel;
        in.x = e.wheel.x;
        in.y = e.wheel.y;
        push(in);
        return true;
    }
    default:
        return false;
    }
}

bool ingame_menu_init(VkInstance instance, VkPhysicalDevice phys, VkDevice device, std::uint32_t family, VkQueue queue,
                      VkFormat format, std::uint32_t image_count) {
    if (g_vk_ready && format == g_format && image_count == g_images) return true;
    ingame_menu_shutdown();
    IMGUI_CHECKVERSION();
    g_ctx = ImGui::CreateContext();
    ImGui::SetCurrentContext(g_ctx);
    ImGuiIO& io = ImGui::GetIO();
    ImFontConfig jp_font;
    jp_font.FontDataOwnedByAtlas = false;
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(kJapaneseFont),
        static_cast<int>(kJapaneseFontSize), 18.0f, &jp_font);
    io.IniFilename = nullptr;
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
    ImGui::StyleColorsDark();
    ImGuiStyle& st = ImGui::GetStyle();
    st.WindowRounding = 4.0f;
    st.Colors[ImGuiCol_WindowBg].w = 0.94f;
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = instance;
    info.PhysicalDevice = phys;
    info.Device = device;
    info.QueueFamily = family;
    info.Queue = queue;
    info.DescriptorPoolSize = 16;
    info.MinImageCount = image_count < 2 ? 2 : image_count;
    info.ImageCount = info.MinImageCount;
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    g_format = format;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &g_format;
    if (!ImGui_ImplVulkan_Init(&info)) {
        host_log("plugin menu: the Vulkan backend did not start; F9 does nothing");
        ImGui::DestroyContext(g_ctx);
        g_ctx = nullptr;
        return false;
    }
    g_images = image_count;
    g_vk_ready = true;
    g_ui.in_game = true;
    g_last = std::chrono::steady_clock::now();
    return true;
}

void ingame_menu_shutdown() {
    if (!g_ctx) return;
    ImGui::SetCurrentContext(g_ctx);
    if (g_vk_ready) ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext(g_ctx);
    g_ctx = nullptr;
    g_vk_ready = false;
}

namespace {

// ImGui's frame for the menu, built and rendered to draw data; false when it
// is closed (no ImGui frame at all, so a closed menu costs the presenter
// nothing but this check).
bool build_frame(VkExtent2D extent) {
    if (!g_vk_ready || !ingame_menu_open()) {
        g_ui.scanned = false;  // read the files again when it next opens
        return false;
    }
    ImGui::SetCurrentContext(g_ctx);
    ImGuiIO& io = ImGui::GetIO();
    const auto now = std::chrono::steady_clock::now();
    io.DeltaTime = std::max(1e-4f, std::chrono::duration<float>(now - g_last).count());
    g_last = now;
    io.DisplaySize = ImVec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
    const float scale = static_cast<float>(extent.height) / 900.0f;
    io.FontGlobalScale = scale;
    // The pointer over the menu: ImGui draws it last, on top. The host's own
    // (host_overlay_cursor) is drawn under the menu and stands down while it
    // is open; with "Draw the pointer" off the window system's shows instead.
    io.MouseDrawCursor = host_settings_ref().draw_cursor;
    std::vector<Input> q;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        q.swap(g_queue);
    }
    for (const Input& in : q) {
        switch (in.kind) {
        case Input::Move: io.AddMousePosEvent(in.x, in.y); break;
        case Input::Button:
            io.AddMousePosEvent(in.x, in.y);
            io.AddMouseButtonEvent(in.button, in.down);
            break;
        case Input::Wheel: io.AddMouseWheelEvent(in.x, in.y); break;
        case Input::Key:
            io.AddKeyEvent(in.key, in.down);
            // ImGui reads shortcuts (Ctrl+A, Ctrl+V) from its modifier keys.
            if (in.key == ImGuiKey_LeftCtrl || in.key == ImGuiKey_RightCtrl) io.AddKeyEvent(ImGuiMod_Ctrl, in.down);
            if (in.key == ImGuiKey_LeftShift || in.key == ImGuiKey_RightShift) io.AddKeyEvent(ImGuiMod_Shift, in.down);
            break;
        case Input::Text: io.AddInputCharactersUTF8(in.text.c_str()); break;
        case Input::Focus: io.AddFocusEvent(true); break;
        }
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    const ImVec2 size(io.DisplaySize.x * 0.72f, io.DisplaySize.y * 0.78f);
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    bool keep = true;
    if (ImGui::Begin("プラグイン  (F9 / Esc で閉じる)", &keep, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove)) {
        plugin_ui_draw(g_ui, scale);
    }
    ImGui::End();
    if (!keep) g_open.store(false);
    ImGui::Render();
    // ImGui's own texture uploads - the font atlas when the menu first opens,
    // and again whenever 1.92 adds glyphs to it - submit on the renderer's
    // queue and then wait for that queue to go idle. Left to RenderDrawData
    // they ran on this thread with no lock, beside the submission thread's
    // vkQueueSubmit on the same queue, which Vulkan requires the caller to
    // keep apart. Done here under the queue's lock (as the overlay's atlas
    // is, host/overlay.cpp), RenderDrawData then finds nothing to upload. A
    // frame with no texture to change - nearly all of them - takes no lock.
    ImDrawData* dd = ImGui::GetDrawData();
    bool uploads = false;
    if (dd && dd->Textures) {
        for (ImTextureData* tex : *dd->Textures) uploads = uploads || tex->Status != ImTextureStatus_OK;
    }
    if (uploads) {
        host_gpu_queue_lock();
        for (ImTextureData* tex : *dd->Textures) {
            if (tex->Status != ImTextureStatus_OK) ImGui_ImplVulkan_UpdateTexture(tex);
        }
        host_gpu_queue_unlock();
    }
    return true;
}

}  // namespace

bool ingame_menu_record_in_pass(VkCommandBuffer cmd, VkExtent2D extent) {
    if (!build_frame(extent)) return false;
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    return true;
}

void ingame_menu_record(VkCommandBuffer cmd, VkImageView view, VkExtent2D extent) {
    if (!build_frame(extent)) return;
    VkRenderingAttachmentInfo colour{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colour.imageView = view;
    colour.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colour.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colour.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colour;
    vkCmdBeginRendering(cmd, &ri);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

#else  // a build without a window: no menu

bool ingame_menu_open() { return false; }
bool ingame_menu_event(const SDL_Event&, float) { return false; }
bool ingame_menu_init(VkInstance, VkPhysicalDevice, VkDevice, std::uint32_t, VkQueue, VkFormat, std::uint32_t) { return false; }
void ingame_menu_shutdown() {}
void ingame_menu_record(VkCommandBuffer, VkImageView, VkExtent2D) {}
bool ingame_menu_record_in_pass(VkCommandBuffer, VkExtent2D) { return false; }

#endif
