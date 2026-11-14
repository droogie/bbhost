// save_commit_test: a save written whole or not at all (hle/save_commit.h).
//
// A save's directory A, a session that writes B into the stage and commits.
// Plain cases: a commit leaves B, a session that never commits leaves A, a new
// save is made, a file the game deleted is gone, folders come and go, a second
// recovery finds nothing; file by file, the save's directory is the same one
// after the commit (a handle open on it, as File Explorer holds one, still
// names it), and a move refused (a file another process holds) leaves the
// stage whole for reading until a later recovery finishes the commit.
//
// Crash injection: a child process runs the session and is killed at every
// step - mid copy, mid write, after the writes, after the flush, with the
// marker written but not in place and in place, after each file's move, after
// the exchange or the renames, and after each step of the cleanup - then the
// recovery runs and the save must be all of A (killed before the marker) or
// all of B (after it), byte for byte, with nothing left beside it. Then
// recoveries killed in turn (a commit killed mid-way, its recovery killed too),
// and a recovery killed while it copies a refused commit's files back. Each
// for every way of committing: file by file (the exchange not tried, as on
// Windows; and tried and refused, as on exFAT, vfat and NFS), the exchange
// (renameat2 RENAME_EXCHANGE, where the test directory's file system has it),
// and v0.2.32-v0.2.34's two renames, whose leftovers this version recovers.
// The children are forked on Linux and the test run again on Windows (where a
// child ends itself with TerminateProcess at the step).
//
// save_commit_test [DIR]: the directories go under DIR ($TMPDIR, else the
// temp directory).
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

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
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

// A file's path in the save -> its bytes; a folder's path with a '/' after it -> "".
using Tree = std::map<std::string, std::string>;

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
            {"sce_sys/", ""},
            {"sce_sys/icon0.png", pattern('i', 20000)}};
}
// The save after: the big file rewritten (longer), one gone, one new, the icon changed.
Tree save_b() {
    return {{"userdata0000", pattern('B', 400000)},
            {"userdata0002", pattern('n', 7000)},
            {"sce_param.bin", pattern('q', 1328)},
            {"sce_sys/", ""},
            {"sce_sys/icon0.png", pattern('j', 18000)}};
}
// A session after B: C, another one, written in place in a stage.
Tree save_c() {
    Tree c = save_b();
    c["userdata0000"] = pattern('C', 350000);
    c["userdata0003"] = pattern('3', 900);
    return c;
}
// Folders: one kept with its file changed, one gone, one new, one new and empty.
Tree dirs_a() {
    return {{"keep/", ""}, {"keep/a.bin", pattern('k', 3000)}, {"gone/", ""}, {"gone/b.bin", pattern('g', 2000)}, {"x.bin", pattern('x', 100)}};
}
Tree dirs_b() {
    return {{"keep/", ""}, {"keep/a.bin", pattern('K', 3500)}, {"new/", ""}, {"new/c.bin", pattern('c', 1500)}, {"empty/", ""},
            {"x.bin", pattern('x', 100)}};
}

bool is_dir(const std::string& k) { return !k.empty() && k.back() == '/'; }

Tree read_tree(const std::string& dir) {
    Tree t;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return t;
    for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string rel = it->path().lexically_relative(dir).generic_string();
        if (it->is_directory(ec)) {
            t[rel + "/"] = "";
        } else if (it->is_regular_file(ec)) {
            std::ifstream f(it->path(), std::ios::binary);
            t[rel] = std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        }
    }
    return t;
}

void write_file(const std::string& path, const std::string& bytes, std::size_t upto = std::string::npos) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(std::min(upto, bytes.size())));
}

void write_tree(const std::string& dir, const Tree& t) {
    fs::create_directories(dir);
    for (const auto& [name, bytes] : t) {
        if (is_dir(name)) fs::create_directories(dir + "/" + name);
        else write_file(dir + "/" + name, bytes);
    }
}

// The game's session in the stage: `to` written over `from`, file by file,
// then what `to` lacks removed. `half` stops after half of the first file (a
// write cut).
bool play(const std::string& stage, const Tree& from, const Tree& to, bool half = false) {
    for (const auto& [name, bytes] : to) {
        if (is_dir(name)) {
            fs::create_directories(stage + "/" + name);
            continue;
        }
        if (half) {
            write_file(stage + "/" + name, bytes, bytes.size() / 2);
            return false;
        }
        write_file(stage + "/" + name, bytes);
    }
    std::error_code ec;
    for (const auto& [name, bytes] : from) {
        if (!to.count(name)) fs::remove_all(stage + "/" + name, ec);
    }
    return true;
}

bool only_the_save(const std::string& live) {
    const save_commit::Paths p = save_commit::paths_of(live);
    std::error_code ec;
    for (const std::string& left : {p.stage, p.old, p.marker, p.marker_new, p.stage + ".unsure"}) {
        if (fs::exists(left, ec)) {
            std::fprintf(stderr, "left beside the save: %s\n", left.c_str());
            return false;
        }
    }
    return true;
}

std::string fresh(const std::string& base, const std::string& name) {
    const std::string root = base + "/save_commit_test-" + name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root + "/saves");
    return root + "/saves/SPRJ0005";
}

// A handle open on the save's directory, as File Explorer holds one on the
// folder it shows: still the save's directory after a commit, or moved (the
// renames) or replaced (the exchange).
struct DirWatch {
#if defined(_WIN32)
    HANDLE h = INVALID_HANDLE_VALUE;
    std::wstring first;
    static std::wstring now(HANDLE h) {
        wchar_t buf[1024];
        const DWORD n = GetFinalPathNameByHandleW(h, buf, 1024, FILE_NAME_NORMALIZED);
        return n > 0 && n < 1024 ? std::wstring(buf, n) : std::wstring();
    }
    explicit DirWatch(const std::string& dir) {
        h = CreateFileW(fs::path(dir).wstring().c_str(), FILE_LIST_DIRECTORY,
                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                        FILE_FLAG_BACKUP_SEMANTICS, nullptr);
        if (h != INVALID_HANDLE_VALUE) first = now(h);
    }
    ~DirWatch() {
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    }
    // Still the save's directory, under its own name.
    bool same() const { return h != INVALID_HANDLE_VALUE && !first.empty() && now(h) == first; }
#else
    std::string dir;
    ino_t ino = 0;
    explicit DirWatch(const std::string& d) : dir(d) {
        struct stat st{};
        if (::stat(d.c_str(), &st) == 0) ino = st.st_ino;
    }
    bool same() const {
        struct stat st{};
        return ino != 0 && ::stat(dir.c_str(), &st) == 0 && st.st_ino == ino;
    }
#endif
    DirWatch(const DirWatch&) = delete;
    DirWatch& operator=(const DirWatch&) = delete;
};

// The ways of committing:
//   'r' file by file, the exchange not tried (as on Windows);
//   'f' file by file after the exchange was tried and refused (forced, as on
//       exFAT, vfat, NFS); 'n' the same with the refusal from the file system
//       itself (an NFS run here, exFAT or vfat on a Steam Deck's card);
//   'x' the exchange (Linux, a file system that has it);
//   'o' v0.2.32-v0.2.34's two renames (the exchange not tried), and 'p' its
//       renames after the exchange was refused - for the recovery of what
//       they leave.
void set_mode(char mode) {
    save_commit::set_no_exchange(mode == 'r' || mode == 'o');
    save_commit::set_exchange_refused(mode == 'f' || mode == 'p');
    save_commit::set_old_renames(mode == 'o' || mode == 'p');
}
bool files_mode(char mode) { return mode == 'r' || mode == 'f' || mode == 'n'; }
bool old_mode(char mode) { return mode == 'o' || mode == 'p'; }

const char* mode_name(char m) {
    switch (m) {
        case 'r': return "file by file";
        case 'f': return "file by file, the exchange refused (forced)";
        case 'n': return "file by file, the exchange refused by this file system";
        case 'x': return "the exchange";
        case 'o': return "v0.2.32-v0.2.34's renames";
        case 'p': return "v0.2.32-v0.2.34's renames, the exchange refused (forced)";
    }
    return "?";
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
        CHECK(play(stage, save_a(), save_b()));
        bool same = false;
        {
            const DirWatch watch(live);
            std::string why;
            CHECK(save_commit::commit(live, &why));
            same = watch.same();
        }
        CHECK(read_tree(live) == save_b());
        CHECK(only_the_save(live));
        // File by file the save's directory is never moved: a window open in
        // it stays there. The renames moved it - the window followed the save
        // to .old, which was then removed (the dialog players saw). (Wine
        // keeps a handle's name through a rename, so the renames' side is
        // checked on Linux, and shown on Windows.)
        std::printf("%s: a handle open on the save's directory through the commit still names it: %s\n", mode_name(mode),
                    same ? "yes" : "no");
        if (files_mode(mode)) CHECK(same);
#if !defined(_WIN32)
        if (old_mode(mode)) CHECK(!same);
#endif
        std::string did = "x";
        save_commit::recover(live, &did);
        CHECK(did.empty());  // nothing left to settle
    }
    {  // A session that never commits: A stays.
        const std::string live = fresh(base, "drop-" + tag);
        write_tree(live, save_a());
        const std::string stage = save_commit::begin(live);
        CHECK(!stage.empty());
        CHECK(play(stage, save_a(), save_b()));
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
        CHECK(play(stage, {}, save_b()));
        CHECK(save_commit::commit(live));
        CHECK(read_tree(live) == save_b());
        CHECK(only_the_save(live));
    }
    {  // A file the game deleted, nothing else changed: gone from the save.
        const std::string live = fresh(base, "deleted-" + tag);
        write_tree(live, save_a());
        const std::string stage = save_commit::begin(live);
        CHECK(!stage.empty());
        Tree less = save_a();
        less.erase("userdata0001");
        CHECK(play(stage, save_a(), less));
        CHECK(save_commit::commit(live));
        CHECK(read_tree(live) == less);
        CHECK(only_the_save(live));
    }
    {  // Folders: one kept, one gone, one new, one new and empty.
        const std::string live = fresh(base, "dirs-" + tag);
        write_tree(live, dirs_a());
        const std::string stage = save_commit::begin(live);
        CHECK(!stage.empty());
        CHECK(play(stage, dirs_a(), dirs_b()));
        CHECK(save_commit::commit(live));
        CHECK(read_tree(live) == dirs_b());
        CHECK(only_the_save(live));
    }
    if (!files_mode(mode)) return;
    {  // A move refused (another process holds the file): the commit stops;
       // a recovery that cannot finish either makes the stage whole and reads
       // the save there; a session then writes there in place; the commit
       // after finishes it.
        const std::string live = fresh(base, "refused-" + tag);
        write_tree(live, save_a());
        const std::string stage = save_commit::begin(live);
        CHECK(play(stage, save_a(), save_b()));
        save_commit::set_refused_move("userdata0002");
        std::string why;
        CHECK(!save_commit::commit(live, &why));
        CHECK(why.find("userdata0002") != std::string::npos);
        std::string did;
        CHECK(save_commit::recover(live, &did) == stage);
        CHECK(did.find("the stage holds the newest save") != std::string::npos);
        CHECK(read_tree(stage) == save_b());  // whole again
        CHECK(save_commit::begin(live).empty());  // the session writes in the stage
        CHECK(play(stage, save_b(), save_c()));
        CHECK(!save_commit::commit(live));  // still refused
        CHECK(save_commit::recover(live) == stage);
        CHECK(read_tree(stage) == save_c());
        save_commit::set_refused_move("");
        CHECK(save_commit::recover(live, &did) == live);
        CHECK(did == "finished a commit");
        CHECK(read_tree(live) == save_c());
        CHECK(only_the_save(live));
    }
    {  // The same, finished by the next commit instead of a recovery.
        const std::string live = fresh(base, "refused2-" + tag);
        write_tree(live, save_a());
        const std::string stage = save_commit::begin(live);
        CHECK(play(stage, save_a(), save_b()));
        save_commit::set_refused_move("sce_param.bin");
        CHECK(!save_commit::commit(live));
        CHECK(save_commit::recover(live) == stage);
        save_commit::set_refused_move("");
        CHECK(play(stage, save_b(), save_c()));
        CHECK(save_commit::commit(live));
        CHECK(read_tree(live) == save_c());
        CHECK(only_the_save(live));
    }
}

// ---- Crash injection: a child process killed at a step.

const char* g_kill_at = nullptr;
int g_kill_count = 0;

[[noreturn]] void die() {
#if defined(_WIN32)
    TerminateProcess(GetCurrentProcess(), 9);
#else
    ::raise(SIGKILL);
#endif
    std::_Exit(9);
}

void kill_hook(const char* step) {
    if (g_kill_at && std::strcmp(step, g_kill_at) == 0 && --g_kill_count == 0) die();
}

// What a child does, killed at `at` (its `nth` time): "session" begins a
// session on `live`, writes B over A (or over nothing) and commits; "recover"
// recovers. `refuse` is a file whose move is refused ("-" for none).
[[noreturn]] void child(const std::string& live, char mode, const std::string& job, const std::string& at, int nth,
                        const std::string& refuse, bool new_save) {
    set_mode(mode);
    save_commit::set_refused_move(refuse == "-" ? "" : refuse);
    g_kill_at = at == "write-half" || at == "written" ? nullptr : at.c_str();
    g_kill_count = nth;
    save_commit::set_test_hook(kill_hook);
    if (job == "recover") {
        save_commit::recover(live);
        std::_Exit(3);  // not killed: the step never came
    }
    const std::string stage = save_commit::begin(live);
    if (stage.empty()) std::_Exit(2);
    const Tree from = new_save ? Tree{} : save_a();
    if (at == "write-half") {
        play(stage, from, save_b(), true);
        die();
    }
    play(stage, from, save_b());
    if (at == "written") die();
    save_commit::commit(live);
    std::_Exit(3);
}

#if defined(_WIN32)
std::wstring g_self;
#endif

// Runs `child` in a process of its own; true when it was killed at the step
// ("none": when it ended by itself).
bool run_child(const std::string& live, char mode, const std::string& job, const std::string& at, int nth,
               const std::string& refuse, bool new_save) {
#if defined(_WIN32)
    std::wstring cmd = L"\"" + g_self + L"\" --child \"" + fs::path(live).wstring() + L"\" " + std::wstring(1, mode) + L" " +
                       fs::path(job).wstring() + L" " + fs::path(at).wstring() + L" " + std::to_wstring(nth) + L" " +
                       fs::path(refuse).wstring() + L" " + (new_save ? L"1" : L"0");
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(g_self.c_str(), cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return false;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    const DWORD want = at == "none" ? 3 : 9;
    if (code != want) std::fprintf(stderr, "the child ended with %lu at %s (wanted %lu)\n", static_cast<unsigned long>(code), at.c_str(), static_cast<unsigned long>(want));
    return code == want;
#else
    std::fflush(nullptr);
    const pid_t pid = ::fork();
    if (pid == 0) child(live, mode, job, at, nth, refuse, new_save);
    int status = 0;
    ::waitpid(pid, &status, 0);
    const bool ok = at == "none" ? WIFEXITED(status) && WEXITSTATUS(status) == 3 : WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL;
    if (!ok) std::fprintf(stderr, "the child ended with status 0x%x at %s\n", status, at.c_str());
    return ok;
#endif
}

struct Kill {
    const char* at;
    int nth;
};

// One crash: a session killed at `k`, then recoveries killed at each of
// `then`; the last recovery runs here and the save must be `want` ('A', 'B',
// '*' either, or '-' for none), whole. `refuse`: a file whose move every child
// refuses. `torn`: the marker cut to that many bytes before the recoveries,
// or (-2) one byte of it changed - as a disk that lost what it was told to keep.
void crash_case(const std::string& base, char mode, Kill k, char want, const std::vector<Kill>& then = {},
                const std::string& refuse = "-", bool new_save = false, long torn = -1) {
    std::string name = std::string(1, mode) + (new_save ? "-new-" : "-") + k.at + "-" + std::to_string(k.nth);
    for (const Kill& t : then) name += std::string("+") + t.at + "-" + std::to_string(t.nth);
    if (refuse != "-") name += "+refused";
    if (torn != -1) name += torn == -2 ? "+marker-changed" : "+marker-cut-" + std::to_string(torn);
    const std::string live = fresh(base, name);
    if (!new_save) write_tree(live, save_a());
    bool killed = run_child(live, mode, "session", k.at, k.nth, refuse, new_save);
    if (torn != -1) {
        const std::string marker = save_commit::paths_of(live).marker;
        std::ifstream f(marker, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();
        CHECK(bytes.size() > 40);
        if (torn == -2) bytes[bytes.size() / 2] ^= 0x20;
        else bytes.resize(std::min<std::size_t>(bytes.size(), static_cast<std::size_t>(torn)));
        write_file(marker, bytes);
    }
    for (const Kill& t : then) killed = run_child(live, mode, "recover", t.at, t.nth, refuse, new_save) && killed;
    if (!killed) std::fprintf(stderr, "%s: a child was not killed\n", name.c_str());
    CHECK(killed);
    set_mode('r');  // the recovery never exchanges
    save_commit::set_refused_move("");
    std::string did;
    const std::string holds = save_commit::recover(live, &did);
    CHECK(holds == live);
    const Tree got = read_tree(live);
    const bool none = !fs::exists(live) || got.empty();
    const bool is_a = got == save_a(), is_b = got == save_b();
    const bool ok = want == 'A' ? is_a : want == 'B' ? is_b : want == '*' ? is_a || is_b : none;
    const char* was = is_a ? "A" : is_b ? "B" : none ? "none" : "a MIX or a cut file";
    if (!ok) std::fprintf(stderr, "%s: after the recovery the save is %s (wanted %c); recovery: %s\n", name.c_str(), was, want, did.c_str());
    CHECK(ok);
    CHECK(only_the_save(live));
    std::printf("  %-44s killed, recovered to %s%s%s\n", name.c_str(), was, did.empty() ? "" : ": ", did.c_str());
}

#if !defined(_WIN32)
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

void crash_cases(const std::string& base, char m) {
    std::printf("crash injection, %s:\n", mode_name(m));
    // Before the marker is in place: A.
    for (const Kill k : {Kill{"copy", 1}, Kill{"copy", 3}, Kill{"write-half", 1}, Kill{"written", 1}, Kill{"flushed", 1},
                         Kill{"marker-written", 1}})
        crash_case(base, m, k, 'A');
    // From the marker on: B.
    crash_case(base, m, {"marker", 1}, 'B');
    if (m == 'x') {
        for (const char* at : {"exchanged", "old-removed", "marker-removed"}) crash_case(base, m, {at, 1}, 'B');
        return;
    }
    if (old_mode(m)) {
        for (const char* at : {"live-moved", "stage-moved", "marker-removed", "old-removed"}) crash_case(base, m, {at, 1}, 'B');
        // Its marker written, killed before it renamed anything; a new save.
        crash_case(base, m, {"flushed", 1}, '-', {}, "-", true);
        crash_case(base, m, {"stage-moved", 1}, 'B', {}, "-", true);
        return;
    }
#if !defined(_WIN32)
    if (m != 'r') {
        // The exchange's marker in place, then the file-by-file one being
        // written over it (Linux: Windows has no inode to name, so no
        // exchange's marker).
        crash_case(base, m, {"marker-written", 2}, 'B');
        crash_case(base, m, {"marker", 2}, 'B');
    }
#endif
    for (int n = 1; n <= 4; ++n) crash_case(base, m, {"moved", n}, 'B');  // B has four files
    for (const char* at : {"extras-removed", "marker-removed", "stage-removed"}) crash_case(base, m, {at, 1}, 'B');
    // Recoveries killed in turn.
    crash_case(base, m, {"moved", 2}, 'B', {{"moved", 1}});
    crash_case(base, m, {"moved", 1}, 'B', {{"moved", 2}, {"extras-removed", 1}});
    crash_case(base, m, {"marker", 1}, 'B', {{"marker-removed", 1}});
    crash_case(base, m, {"moved", 4}, 'B', {{"extras-removed", 1}, {"stage-removed", 1}});
    // A move refused (the commit stops, not killed), the recovery killed
    // while it copies the files already moved back into the stage; the last
    // recovery is allowed the move.
    crash_case(base, m, {"none", 0}, 'B', {{"copied-back", 1}}, "userdata0002");
    crash_case(base, m, {"none", 0}, 'B', {{"copied-back", 2}, {"copied-back", 1}}, "sce_param.bin");
    // A marker not whole, as no kill leaves one (it is put in place whole):
    // cut short, or a byte changed. Nothing was moved under it - A, or B where
    // it reads as v0.2.32-v0.2.34's "rename" (the stage, whole, made the save).
    if (m == 'r') {
        for (const long cut : {5L, 12L, 17L, 60L, 100L, 106L}) crash_case(base, m, {"marker", 1}, '*', {}, "-", false, cut);
        crash_case(base, m, {"marker", 1}, 'A', {}, "-", false, -2);
    }
    // A new save: none before the marker, B after it.
    crash_case(base, m, {"flushed", 1}, '-', {}, "-", true);
    crash_case(base, m, {"marker", 1}, 'B', {}, "-", true);
    crash_case(base, m, {"moved", 1}, 'B', {}, "-", true);
    crash_case(base, m, {"extras-removed", 1}, 'B', {}, "-", true);
}

int main(int argc, char** argv) {
    if (argc == 9 && std::strcmp(argv[1], "--child") == 0)
        child(argv[2], argv[3][0], argv[4], argv[5], std::atoi(argv[6]), argv[7], argv[8][0] == '1');
#if defined(_WIN32)
    {
        wchar_t self[MAX_PATH];
        const DWORD n = GetModuleFileNameW(nullptr, self, MAX_PATH);
        g_self.assign(self, n);
    }
#endif
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
    std::printf("RENAME_EXCHANGE on this file system: %s\n", exchange ? "yes" : "no");
    const char third = exchange ? 'x' : 'n';
    for (const char m : {'r', 'f', third, 'o', 'p'}) plain_cases(base, m);
    for (const char m : {'r', 'f', third, 'o', 'p'}) crash_cases(base, m);
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
