// save_commit_test: a save written whole or not at all (hle/save_commit.h).
//
// A save's directory A, a session that writes B into the stage and commits.
// Plain cases: a commit leaves B, a session that never commits leaves A, a new
// save is made, a second recovery finds nothing. Linux, crash injection: a
// child process runs the session and is killed (SIGKILL) at every step - mid
// copy, mid write, after the writes, after the flush, after the marker, after
// the exchange or between the two renames, and after the cleanup's steps -
// then the recovery runs and the save must be all of A (killed before the
// marker) or all of B (after it), byte for byte, with nothing left beside it.
// Both swaps: the exchange (renameat2 RENAME_EXCHANGE, where the test
// directory's file system has it) and the two renames (forced, as on exFAT,
// vfat and NFS, which refuse the exchange).
//
// save_commit_test [DIR]: the directories go under DIR ($TMPDIR, else /tmp).
#include "hle/save_commit.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

using Tree = std::map<std::string, std::string>;  // a file's path in the save -> its bytes

std::string pattern(char c, std::size_t n) {
    std::string s(n, c);
    for (std::size_t i = 0; i < n; i += 97) s[i] = static_cast<char>('0' + (i / 97) % 10);
    return s;
}

// The save before: a big file, a small one, the param and an icon in a folder.
Tree save_a() {
    return {{"userdata0000", pattern('A', 300000)},
            {"userdata0001", pattern('a', 5000)},
            {"sce_param.bin", pattern('p', 1328)},
            {"sce_sys/icon0.png", pattern('i', 20000)}};
}
// The save after: the big file rewritten (longer), one gone, one new, the icon changed.
Tree save_b() {
    return {{"userdata0000", pattern('B', 400000)},
            {"userdata0002", pattern('n', 7000)},
            {"sce_param.bin", pattern('q', 1328)},
            {"sce_sys/icon0.png", pattern('j', 18000)}};
}

Tree read_tree(const std::string& dir) {
    Tree t;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return t;
    for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        std::ifstream f(it->path(), std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        t[fs::relative(it->path(), dir, ec).generic_string()] = bytes;
    }
    return t;
}

void write_file(const std::string& path, const std::string& bytes, std::size_t upto = std::string::npos) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(std::min(upto, bytes.size())));
}

void write_tree(const std::string& dir, const Tree& t) {
    for (const auto& [name, bytes] : t) write_file(dir + "/" + name, bytes);
}

// The game's session in the stage: B written over A, file by file, the ones B
// has not removed. `half` stops after half of the first file (a write cut).
bool play_session(const std::string& stage, bool half) {
    const Tree b = save_b();
    for (const auto& [name, bytes] : b) {
        if (half) {
            write_file(stage + "/" + name, bytes, bytes.size() / 2);
            return false;
        }
        write_file(stage + "/" + name, bytes);
    }
    for (const auto& [name, bytes] : save_a()) {
        if (!b.count(name)) fs::remove(stage + "/" + name);
    }
    return true;
}

bool only_the_save(const std::string& live) {
    const save_commit::Paths p = save_commit::paths_of(live);
    std::error_code ec;
    return !fs::exists(p.stage, ec) && !fs::exists(p.old, ec) && !fs::exists(p.marker, ec) && !fs::exists(p.stage + ".unsure", ec);
}

std::string fresh(const std::string& base, const std::string& name) {
    const std::string root = base + "/save_commit_test-" + name;
    fs::remove_all(root);
    fs::create_directories(root + "/saves");
    return root + "/saves/SPRJ0005";
}

// The swap's ways: 'r' the renames only, 'x' the exchange, 'f' the exchange
// tried and refused (as on exFAT, vfat, NFS), the renames after it - and 'n',
// nothing forced, on a file system that really refuses the exchange (NFS
// here, exFAT or vfat on a Steam Deck's card): the refusal comes from it.
void set_mode(char mode) {
    save_commit::set_no_exchange(mode == 'r');
    save_commit::set_exchange_refused(mode == 'f');
}

void plain_cases(const std::string& base, char mode) {
    set_mode(mode);
    const std::string tag(1, mode);
    {  // A commit: B is the save, nothing beside it.
        const std::string live = fresh(base, "commit-" + tag);
        write_tree(live, save_a());
        CHECK(save_commit::recover(live) == live);
        const std::string stage = save_commit::begin(live);
        CHECK(!stage.empty());
        CHECK(read_tree(stage) == save_a());  // today's layout copied as it is
        CHECK(play_session(stage, false));
        std::string why;
        CHECK(save_commit::commit(live, &why));
        CHECK(read_tree(live) == save_b());
        CHECK(only_the_save(live));
        std::string did = "x";
        save_commit::recover(live, &did);
        CHECK(did.empty());  // nothing left to settle
    }
    {  // A session that never commits: A stays.
        const std::string live = fresh(base, "drop-" + tag);
        write_tree(live, save_a());
        const std::string stage = save_commit::begin(live);
        CHECK(!stage.empty());
        CHECK(play_session(stage, false));
        std::string did;
        CHECK(save_commit::recover(live, &did) == live);
        CHECK(!did.empty());
        CHECK(read_tree(live) == save_a());
        CHECK(only_the_save(live));
    }
    {  // A new save (no directory before).
        const std::string live = fresh(base, "new-" + tag);
        const std::string stage = save_commit::begin(live);
        CHECK(!stage.empty());
        CHECK(read_tree(stage).empty());
        CHECK(play_session(stage, false));
        CHECK(save_commit::commit(live));
        CHECK(read_tree(live) == save_b());
        CHECK(only_the_save(live));
    }
}

#if !defined(_WIN32)
const char* g_kill_at = nullptr;
int g_kill_count = 0;
void kill_hook(const char* step) {
    if (g_kill_at && std::strcmp(step, g_kill_at) == 0 && --g_kill_count == 0) ::raise(SIGKILL);
}

// One crash: a child runs the session and dies at `at` (its `nth` time); the
// recovery runs here and the save must be `want`, whole.
void crash_case(const std::string& base, char mode, const char* at, int nth, bool want_b) {
    const std::string name = std::string(1, mode) + "-" + at + "-" + std::to_string(nth);
    const std::string live = fresh(base, name);
    write_tree(live, save_a());
    const pid_t pid = ::fork();
    if (pid == 0) {
        set_mode(mode);
        g_kill_at = std::strcmp(at, "write-half") == 0 ? nullptr : at;
        g_kill_count = nth;
        save_commit::set_test_hook(kill_hook);
        const std::string stage = save_commit::begin(live);
        if (stage.empty()) ::_exit(2);
        if (std::strcmp(at, "write-half") == 0) {
            play_session(stage, true);
            ::raise(SIGKILL);
        }
        play_session(stage, false);
        if (std::strcmp(at, "written") == 0) ::raise(SIGKILL);
        save_commit::commit(live);
        ::_exit(3);  // not killed: the step never came
    }
    int status = 0;
    ::waitpid(pid, &status, 0);
    const bool killed = WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL;
    if (!killed) std::fprintf(stderr, "%s: the child was not killed at %s (status 0x%x)\n", name.c_str(), at, status);
    CHECK(killed);
    set_mode('r');  // the recovery never exchanges
    std::string did;
    const std::string holds = save_commit::recover(live, &did);
    CHECK(holds == live);
    const Tree got = read_tree(live);
    const bool is_a = got == save_a(), is_b = got == save_b();
    if (!(want_b ? is_b : is_a))
        std::fprintf(stderr, "%s: after the recovery the save is %s (wanted %s); recovery: %s\n", name.c_str(),
                     is_a ? "A" : is_b ? "B" : "a MIX or a cut file", want_b ? "B" : "A", did.c_str());
    CHECK(want_b ? is_b : is_a);
    CHECK(only_the_save(live));
    std::printf("  %-28s killed, recovered to %s%s%s\n", name.c_str(), is_a ? "A" : is_b ? "B" : "?", did.empty() ? "" : ": ",
                did.c_str());
}

bool exchange_works(const std::string& base) {
    const std::string a = base + "/save_commit_test-xa", b = base + "/save_commit_test-xb";
    fs::create_directories(a);
    fs::create_directories(b);
    const bool ok = ::syscall(SYS_renameat2, AT_FDCWD, a.c_str(), AT_FDCWD, b.c_str(), 1u << 1) == 0;
    fs::remove_all(a);
    fs::remove_all(b);
    return ok;
}
#endif

int main(int argc, char** argv) {
    std::string base = argc > 1 ? argv[1] : "";
    if (base.empty()) {
        const char* t = std::getenv("TMPDIR");
        base = t && *t ? t : (fs::temp_directory_path().string());
    }
    std::printf("save_commit_test in %s\n", base.c_str());
    bool exchange = false;
#if !defined(_WIN32)
    exchange = exchange_works(base);
#endif
    std::printf("RENAME_EXCHANGE on this file system: %s\n", exchange ? "yes" : "no (the renames only)");
    plain_cases(base, 'r');
    plain_cases(base, 'f');
    plain_cases(base, exchange ? 'x' : 'n');
#if !defined(_WIN32)
    struct Kill {
        const char* at;
        int nth;
        bool want_b;
        bool exchange_only, renames_only;
    };
    const std::vector<Kill> kills = {
        {"copy", 1, false, false, false},          // mid-copy: the stage half made
        {"copy", 3, false, false, false},
        {"write-half", 1, false, false, false},    // mid-write: a file cut in the stage
        {"written", 1, false, false, false},       // the session written, no commit yet
        {"flushed", 1, false, false, false},       // flushed, no marker
        {"marker", 1, true, false, false},         // the marker on the disk
        {"exchanged", 1, true, true, false},       // swapped in one step
        {"live-moved", 1, true, false, true},      // between the two renames
        {"stage-moved", 1, true, false, true},     // both renames done
        {"marker-removed", 1, true, false, false},
        {"old-removed", 1, true, false, false},
    };
    for (const char m : {'r', 'f', exchange ? 'x' : 'n'}) {
        std::printf("crash injection, %s:\n", m == 'x'   ? "the exchange"
                                              : m == 'f' ? "the exchange refused (forced), the renames after it"
                                              : m == 'n' ? "the exchange refused by this file system, the renames after it"
                                                         : "the two renames");
        for (const Kill& k : kills) {
            if ((k.exchange_only && m != 'x') || (k.renames_only && m == 'x')) continue;
            crash_case(base, m, k.at, k.nth, k.want_b);
        }
    }
#endif
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(base, ec)) {
        if (e.path().filename().string().rfind("save_commit_test-", 0) == 0) fs::remove_all(e.path(), ec);
    }
    if (g_failures) {
        std::fprintf(stderr, "%d failures\n", g_failures);
        return 1;
    }
    std::printf("save_commit_test: all passed\n");
    return 0;
}
