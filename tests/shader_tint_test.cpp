// examples/plugins/shader_tint (plugin API 27's shaders' example) on a host
// of our own: no game, no GPU. Loaded as the host loads plugins, with a config
// whose toggles are a few frames long, and driven a frame at a time. Checked:
// it asks for its shader's own source until the shader is met, then hands back
// that source patched - bbhost.hlsli included, the tint after main's call of
// lifted_main - turns it on and off by turns (NULL off), writes its strength
// into its shader data every frame, within 0 to 1 and changing; it asks for
// nothing on a host older than API 27; and it is opt-in.
//
//   shader_tint_test <shader_tint.so>
#include "bbhost_plugin.h"

#include <dlfcn.h>

#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

namespace {

const char* const kSource =
    "// 0b0acf50\n// lift 0123456789abcdef\nstatic float4 mrt0;\nvoid lifted_main() {\n    mrt0 = 1.0f.xxxx;\n    return;\n}\n\n"
    "float4 main() : SV_Target0 {\n    lifted_main();\n    return mrt0;\n}\n";

struct Host {
    std::map<std::string, std::string> config = {{"shader_tint.shader", "0b0acf50.ps"}, {"shader_tint.toggle_seconds", "0.15"},
                                                 {"shader_tint.pulse_seconds", "0.1"}};
    void (*frame_fn)(void*) = nullptr;
    int sources_asked = 0, met_after = 3;
    std::vector<std::string> overrides;  // the texts handed over ("" for NULL)
    std::vector<float> strengths;
    std::vector<std::string> log;
} H;

void fake_log(const char* fmt, ...) {
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    H.log.push_back(b);
}
const char* fake_config(const char* key) {
    static std::string v;
    const auto it = H.config.find(key);
    v = it == H.config.end() ? "" : it->second;
    return v.c_str();
}
int fake_on_frame(void (*fn)(void*), void*) {
    H.frame_fn = fn;
    return 0;
}
int fake_shader_source(const char* shader, const char** hlsl, std::size_t* size) {
    CHECK(std::strcmp(shader, "0b0acf50.ps") == 0);
    if (++H.sources_asked < H.met_after) return 1;  // not met yet
    *hlsl = kSource;
    *size = std::strlen(kSource);
    return 0;
}
int fake_shader_override(const char* name, const char* shader, const char* hlsl) {
    CHECK(std::strcmp(name, "shader_tint") == 0 && std::strcmp(shader, "0b0acf50.ps") == 0);
    H.overrides.push_back(hlsl ? hlsl : "");
    return 0;
}
int fake_shader_state(const char*, const char*, char* why, std::size_t n) {
    if (why && n) std::snprintf(why, n, "%s", "");
    return BB_SHADER_TAKEN;
}
int fake_shader_data(const char* name, std::uint32_t offset, const void* data, std::uint32_t size) {
    CHECK(std::strcmp(name, "shader_tint") == 0 && offset == 0 && size == 4);
    float f;
    std::memcpy(&f, data, sizeof(f));
    H.strengths.push_back(f);
    return 0;
}

BbHostApi make_api(std::uint32_t version) {
    BbHostApi api{};
    api.version = version;
    api.size = sizeof(BbHostApi);
    if (version < 27) api.size = static_cast<std::uint32_t>(offsetof(BbHostApi, shader_override));
    api.log = &fake_log;
    api.config = &fake_config;
    api.on_frame = &fake_on_frame;
    if (version >= 27) {
        api.shader_override = &fake_shader_override;
        api.shader_state = &fake_shader_state;
        api.shader_source = &fake_shader_source;
        api.shader_data = &fake_shader_data;
    }
    return api;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: shader_tint_test <shader_tint.so>\n");
        return 2;
    }
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "cannot load %s: %s\n", argv[1], dlerror());
        return 1;
    }
    const auto* info = static_cast<const BbPluginInfo*>(dlsym(lib, "bb_plugin_info"));
    const auto image = reinterpret_cast<int (*)(const BbHostApi*)>(dlsym(lib, "bb_plugin_image"));
    CHECK(info && image);
    if (!info || !image) return 1;
    CHECK(std::strcmp(info->name, "shader_tint") == 0 && (info->flags & BB_PLUGIN_OPT_IN));

    // An API 26 host: refused before it asks for anything.
    const BbHostApi old = make_api(26);
    CHECK(image(&old) != 0 && !H.frame_fn);

    const BbHostApi api = make_api(27);
    CHECK(image(&api) == 0 && H.frame_fn);
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(700)) {
        H.frame_fn(nullptr);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // The source asked for until met, then never again.
    CHECK(H.sources_asked == H.met_after);
    // On, off, on...: the patched text and NULL by turns.
    CHECK(H.overrides.size() >= 3);
    for (std::size_t k = 0; k < H.overrides.size(); ++k) CHECK(H.overrides[k].empty() == (k % 2 == 1));
    const std::string& text = H.overrides.empty() ? std::string() : H.overrides[0];
    CHECK(text.rfind("#include \"bbhost.hlsli\"\n", 0) == 0);
    const std::size_t call = text.rfind("    lifted_main();\n");
    CHECK(call != std::string::npos && text.compare(call + 19, 14, "    mrt0.rgb =") == 0);
    CHECK(text.find("bb_data_float(0)") != std::string::npos);
    CHECK(text.find("static float4 mrt0;") != std::string::npos && text.find("return mrt0;") != std::string::npos);
    // Its strength every frame once patched, in [0, 1], and pulsing.
    CHECK(H.strengths.size() > 50);
    float lo = 2.0f, hi = -1.0f;
    for (const float f : H.strengths) {
        lo = f < lo ? f : lo;
        hi = f > hi ? f : hi;
    }
    CHECK(lo >= 0.0f && hi <= 1.0f && hi - lo > 0.5f);
    dlclose(lib);
    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        for (const std::string& l : H.log) std::fprintf(stderr, "  %s\n", l.c_str());
        return 1;
    }
    std::printf("shader tint plugin: ok (%zu overrides set, %zu strengths)\n", H.overrides.size(), H.strengths.size());
    return 0;
}
