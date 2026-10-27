// The setup window's recommendation to compile the shaders now
// (host/shader_compile.h, shader_compile_advice): made when the caches are
// missing, were made by another translator, or were made for another GPU or
// driver - and not while the driver check is still running - with the time a
// compile takes: the last finished compile's, else a guess from the machine.
#include "host/shader_compile.h"

#include <cstdio>

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

int main() {
    ShaderCacheState s;
    // A new install: no caches.
    ShaderCompileAdvice a = shader_compile_advice(s, false, 16);
    CHECK(a.recommend && a.why == "シェーダーキャッシュがありません" && a.time == "この PC で約１～２分");
    CHECK(shader_compile_advice(s, true, 8).time == "Steam Deck で約２～３分");
    CHECK(shader_compile_advice(s, false, 8).time == "この PC で約２～３分");
    CHECK(shader_compile_advice(s, false, 4).time == "この PC で約３～５分");
    // Caches of this translator and this driver: nothing to recommend.
    s.have_caches = true;
    s.translator_current = true;
    s.device_current = 1;
    CHECK(!shader_compile_advice(s, false, 16).recommend);
    // The driver check not done yet: wait for it rather than recommend.
    s.device_current = -1;
    CHECK(!shader_compile_advice(s, false, 16).recommend);
    // Another GPU or driver.
    s.device_current = 0;
    a = shader_compile_advice(s, false, 16);
    CHECK(a.recommend && a.why.find("別の GPU またはドライバー") != std::string::npos);
    // An update with another translator: whatever the driver says.
    s.translator_current = false;
    s.device_current = 1;
    a = shader_compile_advice(s, false, 16);
    CHECK(a.recommend && a.why.find("別バージョンの bbhost") != std::string::npos);
    // The last finished compile's time wins over the guess; a cancelled one's does not.
    s.last = {"result done", "when 2026-10-17 08:57", "seconds 148"};
    CHECK(shader_compile_advice(s, true, 8).time == "約 2-3 分 （前回の所要時間: 148 秒）");
    s.last = {"result done", "seconds 62"};
    CHECK(shader_compile_advice(s, false, 80).time == "約 1-2 分 （前回の所要時間: 62 秒）");
    s.last = {"result cancelled", "seconds 30"};
    CHECK(shader_compile_advice(s, true, 8).time == "Steam Deck で約２～３分");
    // The minutes: a range of two whole minutes, never "0-1".
    CHECK(shader_compile_minutes(30) == "約１分");
    CHECK(shader_compile_minutes(91) == "約 1-2 分");
    CHECK(shader_compile_minutes(200) == "約 3-4 分");
    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::printf("shader compile advice: ok\n");
    return 0;
}
