// BBHOST_GXVK_BUFFERS (host/gxvk.h, buffers_mode_for): unset or empty, GX's
// buffers are bound by handle on a discrete GPU with room in its video memory
// and on an integrated GPU with 8 GiB of it or more, and keep today's path on
// a smaller integrated GPU, with tight memory and with small video memory; a value given holds on every device - 0 today's path,
// 1 by handle, 2 compared, "census" (the census alone) today's path, and
// anything else today's path, as before the default changed.
#include "host/gxvk.h"

#include <cstdio>
#include <vector>

int main() {
    using gxvk::Buffers;
    using D = gxvk::BuffersDevice;
    const D kAll[] = {D::kRoomy, D::kIntegrated, D::kTight, D::kSmallVram, D::kIntegratedRoomy};
    const char* const kName[] = {"room in its video memory", "an integrated GPU", "tight memory", "small video memory",
                                 "an integrated GPU with room"};
    struct Case {
        const char* env;
        D device;
        Buffers want;
    };
    std::vector<Case> cases;
    for (const D d : kAll) {
        const Buffers unset = d == D::kRoomy || d == D::kIntegratedRoomy ? Buffers::kOn : Buffers::kOff;
        cases.push_back({nullptr, d, unset});
        cases.push_back({"", d, unset});
        cases.push_back({"0", d, Buffers::kOff});
        cases.push_back({"1", d, Buffers::kOn});
        cases.push_back({"2", d, Buffers::kCompare});
        cases.push_back({"census", d, Buffers::kOff});
        cases.push_back({"off", d, Buffers::kOff});
    }
    int failed = 0;
    for (const Case& c : cases) {
        const Buffers got = gxvk::buffers_mode_for(c.env, c.device);
        if (got != c.want) {
            std::printf("FAIL: BBHOST_GXVK_BUFFERS=%s with %s: mode %d, wanted %d\n", c.env ? c.env : "(unset)", kName[static_cast<int>(c.device)],
                        static_cast<int>(got), static_cast<int>(c.want));
            ++failed;
        }
    }
    std::printf(failed ? "gxvk_buffers_mode_test: %d failed\n" : "gxvk_buffers_mode_test: ok (%d cases)\n",
                failed ? failed : static_cast<int>(cases.size()));
    return failed ? 1 : 0;
}
