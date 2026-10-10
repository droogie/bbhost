#include "hle/libc_random.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

void check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "libc_random_test: %s\n", message);
        std::exit(1);
    }
}

// HKS math.random(0,100), verified at eboot RVA 0xa4f6d0:
// rand % 0x3fffffff, float * 2^-30, floor(r * 101).
// c0000.hks Gesture_Loop_onUpdate requires this roll >=70 to change
// GestureAnime2 and fire W_Gesture_Start{,_mirror} after its TAE gate.
int gesture_roll(int random) {
    const float r = static_cast<float>(random % 0x3fffffff) * 0x1p-30f;
    return static_cast<int>(std::floor(r * 101.0f));
}

}  // namespace

int main() {
    bb::libc::Random random;
    // Vectors checked by executing the dump's actual 37-byte rand export;
    // these cover default seed, uint64 wrap, zero and high-bit seeds.
    constexpr std::uint32_t seeds[] = {1, 0, 0xffffffff, 0x80000000};
    constexpr int expected[][8] = {
        {408024109, 11635919, 196474438, 117649705, 812669700, 553475508, 445349752, 271145432},
        {0, 408024109, 11635919, 196474438, 117649705, 812669700, 553475508, 445349752},
        {876841727, 830605684, 668093517, 584808819, 871423820, 289592013, 124176948, 489611834},
        {642432918, 421120801, 969154889, 351229262, 842046760, 958404672, 821634262, 917249545},
    };
    check(random.next() == expected[0][0], "default seed must be 1");
    for (int s = 0; s < 4; ++s) {
        for (int repeat = 0; repeat < 2; ++repeat) {
            random.seed(seeds[s]);
            for (int value : expected[s]) check(random.next() == value, "native rand sequence / reseeding");
        }
    }

    // Demonstrate the old Windows binding's failure for EVERY possible
    // result, rather than relying on a probabilistic run of the host CRT.
    for (int value = 0; value <= 32767; ++value)
        check(gesture_roll(value) == 0, "15-bit rand always suppresses the gesture transition");

    random.seed(1);
    int changes = 0;
    bool seen[101] = {};
    constexpr int count = 100000;
    for (int i = 0; i < count; ++i) {
        const int value = random.next();
        check(value >= 0 && value <= 0x3fffffff, "guest RAND_MAX is 30 bits");
        const int roll = gesture_roll(value);
        check(roll >= 0 && roll <= 100, "HKS roll bounds");
        seen[roll] = true;
        changes += roll >= 70;
    }
    for (bool value : seen) check(value, "all gesture rolls must be reachable");
    check(changes > 29000 && changes < 33000, "pose change should pass about 31/101 rolls, not zero");
    std::printf("libc_random_test: native vectors match; %d/%d gesture rolls pass\n", changes, count);
    return 0;
}
