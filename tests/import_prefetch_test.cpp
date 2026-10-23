// Which GX buffers' memory the import thread imports as GX makes them, with
// window imports (host/import_prefetch.h): what the GPU reads in place at
// once, the meshes the buffer shadow reads in place when they are told, and
// nothing for what goes through staging or a device-local twin.
#include "host/import_prefetch.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

using import_prefetch::Ask;
using namespace import_prefetch;

constexpr std::uint8_t kVb = kBindVertex, kIb = kBindIndex, kUav = kBindUnordered;
constexpr std::uint8_t kCb = 0x4, kSrv = 0x8;  // constant buffer, shader resource

Ask ask_plain(std::uint8_t usage, std::uint8_t bind, bool shadow_on, bool meshes_on) { return ask(usage, bind, shadow_on, 0, meshes_on); }

void test_in_place() {
    // Read in place whatever the shadow does.
    CHECK(ask_plain(kUsageDefault, kUav, true, false) == Ask::kInPlace);
    CHECK(ask_plain(kUsageDefault, kUav | kSrv, true, true) == Ask::kInPlace);
    CHECK(ask_plain(kUsageDynamic, kVb, true, false) == Ask::kInPlace);
    CHECK(ask_plain(kUsageDynamic, kCb, true, true) == Ask::kInPlace);
    CHECK(ask_plain(kUsageDynamic, kIb, false, false) == Ask::kInPlace);
    // A mesh without the shadow: in place.
    CHECK(ask_plain(kUsageImmutable, kVb, false, false) == Ask::kInPlace);
    CHECK(ask_plain(kUsageDefault, kIb, false, true) == Ask::kInPlace);
    CHECK(ask_plain(kUsageDefault, kVb | kIb, false, false) == Ask::kInPlace);
}

void test_meshes() {
    // With the shadow: only when the meshes' prefetch is on, for default and immutable ones.
    CHECK(ask_plain(kUsageImmutable, kVb, true, true) == Ask::kMeshes);
    CHECK(ask_plain(kUsageDefault, kIb, true, true) == Ask::kMeshes);
    CHECK(ask_plain(kUsageDefault, kVb | kSrv, true, true) == Ask::kMeshes);
    CHECK(ask_plain(kUsageImmutable, kVb, true, false) == Ask::kNone);
    CHECK(ask_plain(kUsageDefault, kIb, true, false) == Ask::kNone);
    // Not a mesh: nothing ahead (textures' and constant buffers' memory goes through staging or their bindings).
    CHECK(ask_plain(kUsageDefault, kCb, true, true) == Ask::kNone);
    CHECK(ask_plain(kUsageImmutable, kSrv, true, true) == Ask::kNone);
    CHECK(ask_plain(kUsageDefault, 0, true, true) == Ask::kNone);
    CHECK(ask_plain(kUsageDefault, kCb, false, true) == Ask::kNone);
    // A usage GX does not have: nothing.
    CHECK(ask_plain(5, kVb, true, true) == Ask::kNone);
}

void test_staging() {
    for (int shadow = 0; shadow < 2; ++shadow) {
        for (int meshes = 0; meshes < 2; ++meshes) {
            CHECK(ask_plain(kUsageStaging, kVb, shadow, meshes) == Ask::kNone);
            CHECK(ask_plain(kUsageStaging, kUav, shadow, meshes) == Ask::kNone);
            CHECK(ask_plain(kUsageStaging, kVb | kIb | kUav, shadow, meshes) == Ask::kNone);
        }
    }
}

void test_twins() {
    // A GPU-written default buffer kept as a twin (levels 1 and 2): no import, not even as a mesh.
    for (int level = 1; level <= 2; ++level) {
        CHECK(ask(kUsageDefault, kUav, true, level, true) == Ask::kNone);
        CHECK(ask(kUsageDefault, kUav | kVb, true, level, true) == Ask::kNone);
        CHECK(ask(kUsageDefault, kUav | kVb, false, level, true) == Ask::kNone);
        // Not a twin's: as without twins.
        CHECK(ask(kUsageImmutable, kVb, true, level, true) == Ask::kMeshes);
        CHECK(ask(kUsageDefault, kVb, false, level, false) == Ask::kInPlace);
    }
    // An unordered-access buffer that is not a default one keeps its import (the twins are default buffers').
    CHECK(ask(kUsageDynamic, kUav, true, 1, false) == Ask::kInPlace);
    // A dynamic one: level 2 only.
    CHECK(ask(kUsageDynamic, kVb, true, 1, true) == Ask::kInPlace);
    CHECK(ask(kUsageDynamic, kVb, true, 2, true) == Ask::kNone);
    CHECK(ask(kUsageDynamic, kCb, true, 2, false) == Ask::kNone);
    // Level 0 and other levels: no twins.
    CHECK(ask(kUsageDefault, kUav, true, 0, false) == Ask::kInPlace);
    CHECK(ask(kUsageDefault, kUav, true, 3, false) == Ask::kInPlace);
    CHECK(ask(kUsageDynamic, kVb, true, 3, false) == Ask::kInPlace);
}

}  // namespace

int main() {
    test_in_place();
    test_meshes();
    test_staging();
    test_twins();
    if (g_fail) {
        std::printf("import_prefetch_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("import_prefetch_test: ok\n");
    return 0;
}
