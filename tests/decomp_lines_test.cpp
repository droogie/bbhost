// The decomp's install report in parts (src/decomp/name_lines.h): every part
// within its width (a longer name alone), the parts rejoined give the names in
// their order, each whole and once, and no part could have taken the next
// part's first name. Fixed cases at the width's edges, the report's own shape
// (204 names at 900), and 20,000 generated lists.

#include "decomp/name_lines.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int g_fails = 0;

#define CHECK(c, ...)                                                       \
    do {                                                                    \
        if (!(c)) {                                                         \
            if (++g_fails <= 20) {                                          \
                std::printf("FAIL %s:%d: ", __FILE__, __LINE__);            \
                std::printf(__VA_ARGS__);                                   \
                std::printf("\n");                                          \
            }                                                               \
        }                                                                   \
    } while (0)

static std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (const std::string& x : v) {
        if (!s.empty()) s += ", ";
        s += x;
    }
    return s;
}

// The rules for one list; names never hold ", " (the report's names are identifiers).
static void check(const std::vector<std::string>& names, std::size_t width, const char* what) {
    const std::vector<std::string> lines = decomp_name_lines(names, width);
    CHECK(join(lines) == join(names), "%s: the parts rejoined are not the names", what);
    CHECK(names.empty() == lines.empty(), "%s: %zu names in %zu parts", what, names.size(), lines.size());
    std::size_t count = 0;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const std::string& l = lines[i];
        CHECK(!l.empty(), "%s: part %zu empty", what, i);
        const bool one = l.find(", ") == std::string::npos;
        CHECK(l.size() <= width || one, "%s: part %zu is %zu characters, past %zu", what, i, l.size(), width);
        for (std::size_t at = 0; (at = l.find(", ", at)) != std::string::npos; at += 2) ++count;
        ++count;
        if (i + 1 < lines.size()) {
            const std::string& next = lines[i + 1];
            const std::size_t first = next.find(", ") == std::string::npos ? next.size() : next.find(", ");
            CHECK(l.size() + 2 + first > width, "%s: part %zu (%zu) could have held the next name (%zu) within %zu", what, i,
                  l.size(), first, width);
        }
    }
    CHECK(count == names.size(), "%s: %zu names in the parts, %zu given", what, count, names.size());
}

static std::string name_of(std::size_t len, char c) { return std::string(len, c); }

int main() {
    // None, one, and lists at the width's edges.
    check({}, 900, "no names");
    CHECK(decomp_name_lines({}, 900).empty(), "no names give no parts");
    check({"gx_upload_pool_init"}, 900, "one name");
    {
        // Two names whose join is exactly the width: one part; one character more: two.
        const std::vector<std::string> fit = {name_of(10, 'a'), name_of(8, 'b')};  // 10 + 2 + 8 = 20
        CHECK(decomp_name_lines(fit, 20).size() == 1, "an exact fit is one part");
        CHECK(decomp_name_lines(fit, 19).size() == 2, "one character over is two parts");
        check(fit, 20, "exact fit");
        check(fit, 19, "one over");
    }
    {
        // A name longer than the width has a part of its own, between parts that fill as usual.
        const std::vector<std::string> v = {"a", "b", name_of(40, 'x'), "c", "d"};
        const std::vector<std::string> lines = decomp_name_lines(v, 10);
        CHECK(lines.size() == 3 && lines[0] == "a, b" && lines[1] == name_of(40, 'x') && lines[2] == "c, d",
              "a long name alone: %zu parts", lines.size());
        check(v, 10, "a long name");
    }
    // The report's own shape: 204 names of 8-60 characters at 900.
    std::mt19937 rng(20261022);
    {
        std::vector<std::string> v;
        std::uniform_int_distribution<int> len(8, 60);
        for (int i = 0; i < 204; ++i) v.push_back("fn_" + std::to_string(i) + "_" + name_of(len(rng), 'q'));
        const std::vector<std::string> lines = decomp_name_lines(v, 900);
        check(v, 900, "204 names");
        CHECK(lines.size() >= 2, "204 names fill %zu parts", lines.size());
        for (const std::string& l : lines) CHECK(l.size() + 48 < 1023, "a part with its prefix passes a log line");
    }
    // Generated lists.
    std::uniform_int_distribution<int> count(0, 80), len(1, 120), width(1, 1000);
    for (int c = 0; c < 20000; ++c) {
        std::vector<std::string> v;
        const int n = count(rng);
        for (int i = 0; i < n; ++i) v.push_back(name_of(static_cast<std::size_t>(len(rng)), static_cast<char>('a' + i % 26)));
        check(v, static_cast<std::size_t>(width(rng)), "generated");
    }
    if (g_fails) {
        std::printf("decomp_lines_test: %d failures\n", g_fails);
        return 1;
    }
    std::printf("decomp_lines_test: ok (fixed cases, 204 names at 900, 20000 generated lists)\n");
    return 0;
}
