// tools/mapval_plugin.c on a host of our own: no game, no GPU. The probe is
// loaded as the host loads plugins and given a tour of the steps
// tools/area_check.py writes - a lamp travel, a view warp, quiet, camera,
// marks, holds, a frame dump and captures - over a pretend player, camera and
// two characters, in memory laid out as the probe reads the game's.
//
//   mapval_probe_test <mapval.so> [tour file]
//
// With a tour file it only checks that every line of it parses.
#include "bbhost_plugin.h"

#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            std::fprintf(stderr, "%s:%d: FAILED %s\n", __FILE__, __LINE__, #c); \
            ++g_failures;                                           \
        }                                                           \
    } while (0)

std::vector<std::string> g_log;
void fake_log(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_log.push_back(buf);
}
bool logged(const std::string& s) {
    for (const std::string& l : g_log) {
        if (l.find(s) != std::string::npos) return true;
    }
    return false;
}
std::size_t log_index(const std::string& s) {
    for (std::size_t i = 0; i < g_log.size(); ++i) {
        if (g_log[i].find(s) != std::string::npos) return i;
    }
    return g_log.size();
}

// The pretend game's memory: where the probe follows pointers, at addresses
// its range check takes (0x1_0000_0000 - 0x8_0000_0000, or the top of the
// user space where mmap puts things).
std::uint8_t* g_mem = nullptr;
constexpr std::uint64_t kWorldChrMan = 0x0000, kPlayer = 0x1000, kModules = 0x2000, kData = 0x3000, kTransform = 0x4000,
                        kCameraRoot = 0x5000, kCameraOwner = 0x8000, kEnemy = 0x9000, kEnemyEntry = 0xa000, kFriend = 0xb000;
std::uint64_t at(std::uint64_t off) { return reinterpret_cast<std::uint64_t>(g_mem + off); }
template <typename T>
void put(std::uint64_t off, T v) {
    std::memcpy(g_mem + off, &v, sizeof v);
}
template <typename T>
T get(std::uint64_t off) {
    T v;
    std::memcpy(&v, g_mem + off, sizeof v);
    return v;
}

struct World {
    float pos[3] = {5.0f, 0.0f, 5.0f}, yaw = 0.0f;
    std::uint32_t block = 0x18010000;
    int travel_in = -1;  // frames until a requested travel lands
    std::uint32_t travel_block = 0;
    std::vector<std::int32_t> travels;
    std::vector<std::uint32_t> flags;  // event flags turned on, in order
    std::vector<std::uint64_t> disabled;
    void (*frame)(void*) = nullptr;
    void* frame_user = nullptr;
} g_world;

int fake_eboot_is_109() { return 1; }
int fake_read(std::uint64_t bn, void* out, std::size_t n) {
    std::uint64_t v = 0;
    if (bn == 0x593e878) v = at(kWorldChrMan);       // WorldChrMan*
    else if (bn == 0x593e860) v = at(kCameraRoot);   // the camera owner's root
    else return 1;
    std::memcpy(out, &v, n < 8 ? n : 8);
    return 0;
}
int fake_on_frame(void (*fn)(void*), void* user) {
    g_world.frame = fn;
    g_world.frame_user = user;
    return 0;
}
int fake_player_position(float xyz[3], float* yaw) {
    if (g_world.travel_in >= 0) return 1;  // loading: no player
    std::memcpy(xyz, g_world.pos, sizeof g_world.pos);
    if (yaw) *yaw = g_world.yaw;
    return 0;
}
int fake_player_block(std::uint32_t* b) {
    if (g_world.travel_in >= 0) return 1;
    *b = g_world.block;
    return 0;
}
int fake_player_warp(std::uint32_t, const float xyz[3], float yaw) {
    std::memcpy(g_world.pos, xyz, sizeof g_world.pos);
    g_world.yaw = yaw;
    return 0;
}
int fake_event_flag_set(std::uint32_t id, int value) {
    if (value) g_world.flags.push_back(id);
    return 0;
}
int fake_lamp_warp(std::int32_t rp) {
    g_world.travels.push_back(rp);
    g_world.flags.push_back(0);  // the travel, among the flags
    g_world.travel_in = 10;
    g_world.travel_block = static_cast<std::uint32_t>(rp / 100000) << 24 | static_cast<std::uint32_t>((rp / 10000) % 10) << 16;
    return 0;
}
int fake_chr_list(BbChr* out, std::size_t max, std::size_t* count) {
    BbChr c[2] = {};
    c[0].ins = reinterpret_cast<void*>(at(kEnemy));
    c[0].npc_param = 100100;
    c[0].team_type = 23;
    c[0].hp = c[0].max_hp = 200;
    c[0].pos[0] = g_world.pos[0] + 10;
    c[0].pos[1] = g_world.pos[1];
    c[0].pos[2] = g_world.pos[2];
    c[1] = c[0];
    c[1].ins = reinterpret_cast<void*>(at(kFriend));
    c[1].team_type = 6;
    for (std::size_t i = 0; i < 2 && i < max; ++i) out[i] = c[i];
    *count = 2;
    return 0;
}
int fake_call_guest(std::uint64_t bn, BbCallRegs* regs) {
    if (bn == 0x1cc6390 && regs->arg[1] == 1) {
        g_world.disabled.push_back(regs->arg[0]);
        put<std::uint8_t>(kEnemyEntry + 0x20, 1);
    }
    return 0;
}

void layout() {
    put<std::uint64_t>(kWorldChrMan + 0x60, at(kPlayer));
    put<std::int32_t>(kWorldChrMan + 0x278, 0);
    put<std::uint64_t>(kPlayer + 0x3b0, at(kModules));
    put<std::uint64_t>(kModules + 0x20, at(kData));
    put<std::uint64_t>(kModules + 0x68, at(kTransform));
    put<std::int32_t>(kData + 0xf8, 100);
    put<std::int32_t>(kData + 0xfc, 100);
    put<std::uint64_t>(kCameraRoot + 0x2830, at(kCameraOwner));
    put<std::uint64_t>(kEnemy + 0x18, at(kEnemyEntry));
    put<std::uint64_t>(kFriend + 0x18, at(kEnemyEntry + 0x100));
}

std::string read_text(const fs::path& p) {
    std::ifstream f(p);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: mapval_probe_test <mapval.so> [tour file]\n");
        return 2;
    }
    // The memory: asked for at 8 GiB, where the probe's range check expects
    // the game's heap; the top of the user space passes it too.
    void* m = mmap(reinterpret_cast<void*>(0x200000000ull), 0x10000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        std::perror("mmap");
        return 2;
    }
    g_mem = static_cast<std::uint8_t*>(m);
    const std::uint64_t base = at(0);
    if (!((base >= 0x100000000ull && base < 0x800000000ull) || (base >= 0x7e0000000000ull && base < 0x7ffffffff000ull))) {
        std::printf("skipped: no memory where the probe's range check takes it (%p)\n", m);
        return 77;
    }
    layout();

    const fs::path dir = fs::temp_directory_path() / ("mapval-probe-test-" + std::to_string(getpid()));
    fs::create_directories(dir / "requests");
    const fs::path tour = argc > 2 ? fs::path(argv[2]) : dir / "tour.txt";
    if (argc <= 2) {
        std::ofstream(tour) << "# a place by its lamp, then one with a view\n"
                               "travel 2412951 18010000 central-yharnam\n"
                               "quiet 40\n"
                               "camera\n"
                               "mark central-yharnam:arrived\n"
                               "hold 0.1\n"
                               "mark central-yharnam:measure\n"
                               "hold 0.2\n"
                               "mark central-yharnam:measured\n"
                               "dump central-yharnam-a\n"
                               "hold 0.05\n"
                               "capture 6 central-yharnam\n"
                               "hold 0.05\n"
                               "capture 0\n"
                               "mark central-yharnam:end\n"
                               "travel 3202950 20000000 moonside-lake 13200040 13200120\n"
                               "warp moonside-lake:view -451.930 -174.980 392.990 1.2217 0.1\n"
                               "mark moonside-lake:arrived\n"
                               "wait 0.05\n";
    }
    setenv("BBHOST_MAPVAL_TOUR", tour.c_str(), 1);
    setenv("BBHOST_MAPVAL_START", "0", 1);
    setenv("BBHOST_TEST_REQUESTS", (dir / "requests").c_str(), 1);
    setenv("BBHOST_MAPVAL_DUMPS", (dir / "frames").c_str(), 1);
    setenv("BBHOST_MAPVAL_CAPTURES", (dir / "captures").c_str(), 1);

    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }
    auto init = reinterpret_cast<int (*)(const BbHostApi*)>(dlsym(h, "bb_plugin_init"));
    auto image = reinterpret_cast<int (*)(const BbHostApi*)>(dlsym(h, "bb_plugin_image"));
    static BbHostApi api;
    std::memset(&api, 0, sizeof api);
    api.version = BB_PLUGIN_API_VERSION;
    api.size = sizeof api;
    api.log = fake_log;
    api.eboot_is_109 = fake_eboot_is_109;
    api.read = fake_read;
    api.on_frame = fake_on_frame;
    api.player_position = fake_player_position;
    api.player_block = fake_player_block;
    api.player_warp = fake_player_warp;
    api.lamp_warp = fake_lamp_warp;
    api.chr_list = fake_chr_list;
    api.call_guest = fake_call_guest;
    api.event_flag_set = fake_event_flag_set;
    CHECK(init && image);
    CHECK(init(&api) == 0);
    CHECK(image(&api) == 0);
    CHECK(!logged("bad tour line"));
    if (argc > 2) {
        for (const std::string& l : g_log) std::printf("%s\n", l.c_str());
        fs::remove_all(dir);
        return g_failures ? 1 : 0;
    }
    CHECK(logged("mapval: tour of 18 steps"));
    CHECK(g_world.frame != nullptr);

    // Frames, 5 ms apart, until the tour is done or 30 s pass.
    const auto t0 = std::chrono::steady_clock::now();
    while (!logged("mapval tour-done") && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30)) {
        if (g_world.travel_in > 0 && --g_world.travel_in == 0) {
            g_world.travel_in = -1;
            g_world.block = g_world.travel_block;
            g_world.pos[0] = -10.0f;
            g_world.pos[2] = 20.0f;
        }
        // Something hurts the player now and then: the holds put it right.
        if (logged("mapval step 4 hold") && !logged("mapval mark central-yharnam:measure")) put<std::int32_t>(kData + 0xf8, 40);
        g_world.frame(g_world.frame_user);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(logged("mapval tour-done"));
    CHECK((g_world.travels == std::vector<std::int32_t>{2412951, 3202950}));
    // The second travel's flags, turned on before it.
    CHECK((g_world.flags == std::vector<std::uint32_t>{0, 13200040, 13200120, 0}));
    CHECK(logged("mapval travel-done central-yharnam"));
    CHECK(logged("mapval travel-done moonside-lake"));
    // The marks, in order, with the block and the position.
    const std::size_t a = log_index("mapval mark central-yharnam:arrived"), b = log_index("mapval mark central-yharnam:measure "),
                      c = log_index("mapval mark central-yharnam:measured"), e = log_index("mapval mark central-yharnam:end"),
                      v = log_index("mapval at moonside-lake:view"), z = log_index("mapval mark moonside-lake:arrived");
    CHECK(a < b && b < c && c < e && e < v && v < z && z < g_log.size());
    CHECK(logged("mapval mark central-yharnam:arrived") && g_log[a].find(" 18010000 -10.000 0.000 20.000 yaw ") != std::string::npos);
    CHECK(g_log[z].find(" 20000000 -451.930 -174.980 392.990 yaw 1.222") != std::string::npos);
    // quiet: the enemy within 40 m off, the other character left alone.
    CHECK(g_world.disabled.size() == 1 && g_world.disabled[0] == at(kEnemy));
    CHECK(logged("1 enemies within 40 m switched off, 1 other characters left"));
    // camera: the reset asked for.
    CHECK(get<std::uint8_t>(kCameraOwner + 0x80) == 1);
    CHECK(logged("mapval camera") && logged(" reset"));
    // hold: the player's HP back to full.
    CHECK(get<std::int32_t>(kData + 0xf8) == 100);
    // The requests, one file each, in order.
    const std::string r0 = read_text(dir / "requests" / "000000.req"), r1 = read_text(dir / "requests" / "000001.req"),
                      r2 = read_text(dir / "requests" / "000002.req");
    CHECK(r0 == "dump " + (dir / "frames" / "central-yharnam-a.ppm").string() + "\n");
    CHECK(r1 == "capture 6 " + (dir / "captures" / "central-yharnam").string() + "\n");
    CHECK(r2 == "capture 0\n");
    CHECK(!fs::exists(dir / "requests" / "000003.req"));
    CHECK(!fs::exists(dir / "requests" / "000000.tmp"));
    if (g_failures) {
        for (const std::string& l : g_log) std::fprintf(stderr, "  %s\n", l.c_str());
    }
    fs::remove_all(dir);
    std::printf("%s (%zu log lines)\n", g_failures ? "FAILED" : "ok", g_log.size());
    return g_failures ? 1 : 0;
}
