// docs/plugins.md's In-game menus example is examples/plugins/menu/menu.c,
// whole: the build compiles the file (bb_plugin_menu_example), and this test
// fails when the docs' copy of it is not the file as it is.
//
//   plugin_menu_docs_test <docs/plugins.md> <examples/plugins/menu/menu.c>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

namespace {
std::string read(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return f ? std::string(std::istreambuf_iterator<char>(f), {}) : std::string();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: plugin_menu_docs_test <docs/plugins.md> <menu.c>\n");
        return 2;
    }
    const std::string docs = read(argv[1]), example = read(argv[2]);
    if (docs.empty() || example.empty()) {
        std::printf("FAIL: could not read %s or %s\n", argv[1], argv[2]);
        return 1;
    }
    const std::string block = "```c\n" + example + "```\n";
    if (docs.find("## In-game menus") == std::string::npos || docs.find(block) == std::string::npos) {
        std::printf("FAIL: docs/plugins.md's In-game menus example is not examples/plugins/menu/menu.c as it is\n");
        return 1;
    }
    std::printf("plugin_menu_docs_test: ok\n");
    return 0;
}
