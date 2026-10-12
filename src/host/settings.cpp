#include "host/settings.h"
#include "core/config.h"

#include <cstdlib>
#include <string>

// The original source's opt-in policy: read once before hooks are installed.
bool host_pc_graphics() {
    static const bool on = [] {
        if (const char* e = std::getenv("BBHOST_PC_GRAPHICS"); e && *e) return e[0] == '1';
        const std::string value = config_value("video.pc_graphics");
        return value == "true" || value == "1";
    }();
    return on;
}
