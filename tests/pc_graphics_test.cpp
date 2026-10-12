// The startup graphics gate must link, preserve the default and honor the
// environment override. Each case is a process because the gate is cached.
#include "host/settings.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
std::string g_config;
int g_config_reads = 0;
}

std::string config_value(const std::string& key) {
    if (key != "video.pc_graphics") std::abort();
    ++g_config_reads;
    return g_config;
}

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    unsetenv("BBHOST_PC_GRAPHICS");
    const std::string mode = argv[1];
    bool want = false;
    int reads = 1;
    if (mode == "default" || mode == "cache") {
        // Absent configuration leaves PC graphics disabled, as in the original source.
    } else if (mode == "config-on") {
        g_config = "true";
        want = true;
    } else if (mode == "config-off") {
        g_config = "false";
        want = false;
    } else if (mode == "env-on") {
        g_config = "false";
        setenv("BBHOST_PC_GRAPHICS", "1", 1);
        want = true;
        reads = 0;
    } else if (mode == "env-off") {
        g_config = "true";
        setenv("BBHOST_PC_GRAPHICS", "0", 1);
        want = false;
        reads = 0;
    } else if (mode == "empty-env") {
        g_config = "false";
        setenv("BBHOST_PC_GRAPHICS", "", 1);
        want = false;
    } else {
        return 1;
    }
    const bool got = host_pc_graphics();
    // Calls in render hooks must not repeat config lookups or switch policy
    // after startup, when the installed hooks can no longer change with it.
    g_config = want ? "false" : "true";
    setenv("BBHOST_PC_GRAPHICS", want ? "0" : "1", 1);
    for (int i = 0; i < 100; ++i) {
        if (host_pc_graphics() != want) return 1;
    }
    if (got != want || g_config_reads != reads) {
        std::printf("%s: enabled=%d expected=%d, config reads=%d expected=%d\n",
                    mode.c_str(), got, want, g_config_reads, reads);
        return 1;
    }
    return 0;
}
