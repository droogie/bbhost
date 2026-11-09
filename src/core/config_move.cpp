#include "core/config_move.h"

#include "core/config.h"
#include "log.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef CONFIG_MOVE_TEST
bool g_config_move_test_copy = false;
int g_config_move_test_fail_copy = 0;
bool g_config_move_test_no_remove = false;
#endif

namespace {

namespace fs = std::filesystem;

// An original left in the old folder (copied through a link, or refused):
// its name there, its size and its modification time, as it was copied.
struct Seen {
    std::string rel;
    long long size = 0, mtime = 0;
};

// kConfigMoveRecord: the moves the setup window has yet to tell, and the
// originals to leave alone.
struct Record {
    std::vector<ConfigMove> moves;
    std::vector<Seen> seen;
};

bool g_started = false;                // config_move_at_start ran
std::string g_new_dir;                 // its `to`: where the record is
std::string g_fallback;                // its `from`, when the move failed
std::vector<ConfigMove> g_unrecorded;  // this start's move when the record does not hold it

// The record's lines are a kind and its fields, separated by tabs.
std::string field(std::string s) {
    for (char& ch : s) {
        if (ch == '\t' || ch == '\n' || ch == '\r') ch = ' ';
    }
    return s;
}

Record read_record(const std::string& dir) {
    Record r;
    std::ifstream in(dir + "/" + kConfigMoveRecord);
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        for (std::size_t a = 0;;) {
            const std::size_t t = line.find('\t', a);
            f.push_back(line.substr(a, t == std::string::npos ? std::string::npos : t - a));
            if (t == std::string::npos) break;
            a = t + 1;
        }
        const auto at = [&f](std::size_t i) { return i < f.size() ? f[i] : std::string(); };
        const std::string& kind = f[0];
        if (kind == "move") {
            ConfigMove m;
            m.when = at(1);
            m.from = at(2);
            m.to = at(3);
            m.link = at(4);
            r.moves.push_back(m);
        } else if (kind == "seen") {
            if (!at(1).empty()) r.seen.push_back({at(1), std::atoll(at(2).c_str()), std::atoll(at(3).c_str())});
        } else if (!r.moves.empty()) {
            ConfigMove& m = r.moves.back();
            if (kind == "moved") m.moved.push_back(at(1));
            else if (kind == "beside") m.beside.push_back({at(1), at(2)});
            else if (kind == "same") m.same.push_back(at(1));
            else if (kind == "kept") m.kept.push_back({at(1), at(2)});
            else if (kind == "left") m.left.push_back(at(1));
            else if (kind == "path") m.paths.push_back({at(1), at(2), at(3) == "1"});
        }
    }
    return r;
}

// Written beside and renamed over it; gone when it holds nothing.
bool write_record(const std::string& dir, const Record& r) {
    const std::string path = dir + "/" + kConfigMoveRecord;
    std::error_code ec;
    if (r.moves.empty() && r.seen.empty()) {
        fs::remove(path, ec);
        return !ec;
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        out << "# bbhost: the settings it moved into this folder from the one it kept them in before. The\n"
               "# setup window tells each move until its notice is dismissed; the \"seen\" lines are\n"
               "# originals it copied but had to leave there, which it leaves alone from now on.\n";
        for (const ConfigMove& m : r.moves) {
            out << "move\t" << field(m.when) << '\t' << field(m.from) << '\t' << field(m.to) << '\t' << field(m.link) << '\n';
            for (const std::string& s : m.moved) out << "moved\t" << field(s) << '\n';
            for (const ConfigMove::Beside& b : m.beside) out << "beside\t" << field(b.name) << '\t' << field(b.as) << '\n';
            for (const std::string& s : m.same) out << "same\t" << field(s) << '\n';
            for (const ConfigMove::Kept& k : m.kept) out << "kept\t" << field(k.name) << '\t' << field(k.why) << '\n';
            for (const std::string& s : m.left) out << "left\t" << field(s) << '\n';
            for (const ConfigMove::Path& p : m.paths)
                out << "path\t" << field(p.key) << '\t' << field(p.value) << '\t' << (p.made_absolute ? 1 : 0) << '\n';
        }
        for (const Seen& s : r.seen) out << "seen\t" << field(s.rel) << '\t' << s.size << '\t' << s.mtime << '\n';
        out.flush();
        if (!out) {
            fs::remove(tmp, ec);
            return false;
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) fs::remove(tmp, ec);
    return !ec;
}

// A path in the home folder as ~/...
std::string tilde(const std::string& p) {
    const char* h = std::getenv("HOME");
    if (!h || !*h) return p;
    std::string home = h;
    while (home.size() > 1 && home.back() == '/') home.pop_back();
    if (p == home) return "~";
    if (home.size() > 1 && p.size() > home.size() && p.compare(0, home.size(), home) == 0 && p[home.size()] == '/')
        return "~" + p.substr(home.size());
    return p;
}

// "a", "a and b", "a, b and c".
std::string join(const std::vector<std::string>& v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += i + 1 == v.size() ? " and " : ", ";
        out += v[i];
    }
    return out;
}

// What a list of names is at the top: plugins/a.so and plugins/b.so are plugins.
std::vector<std::string> tops(const std::vector<std::string>& names) {
    std::vector<std::string> out;
    for (const std::string& n : names) {
        const std::string t = n.substr(0, n.find('/'));
        if (std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
    }
    return out;
}

// Every "$HOME/" in a text as "~/".
std::string tilde_in(std::string s) {
    const char* h = std::getenv("HOME");
    if (!h || !*h) return s;
    std::string home = h;
    while (home.size() > 1 && home.back() == '/') home.pop_back();
    if (home.size() < 2) return s;
    home += '/';
    for (std::size_t at = s.find(home); at != std::string::npos; at = s.find(home, at + 2)) s.replace(at, home.size(), "~/");
    return s;
}

#if !defined(_WIN32)

std::string local_time(const char* format) {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    return std::strftime(buf, sizeof(buf), format, &tm) ? buf : "";
}

// What bbhost itself keeps in the settings folder: the config, the F10
// settings (bbhost-options-<profile>.toml for a --config profile, with the
// account), the official plugins it fetches and the player's own beside them,
// and the player's own patches. Nothing else there is bbhost's.
bool owned(const std::string& name) {
    if (name == "bbhost.toml" || name == "bbhost-options.toml" || name == "plugins" || name == "patches") return true;
    return name.size() > 20 && name.rfind("bbhost-options-", 0) == 0 && name.compare(name.size() - 5, 5, ".toml") == 0;
}

long long mtime_ns(const struct stat& st) {
    return static_cast<long long>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
}

std::vector<std::string> names_in(const std::string& dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) out.push_back(it->path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}

// A rename that never replaces what is there (renameat2's RENAME_NOREPLACE;
// a file system without it, a look first).
int rename_noreplace(const std::string& from, const std::string& to) {
#if defined(SYS_renameat2)
    if (::syscall(SYS_renameat2, AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), 1u /* RENAME_NOREPLACE */) == 0) return 0;
    if (errno != EINVAL && errno != ENOSYS) return errno;
#endif
    struct stat st {};
    if (::lstat(to.c_str(), &st) == 0) return EEXIST;
    return ::rename(from.c_str(), to.c_str()) == 0 ? 0 : errno;
}

// The move's own renames and removals out of the old folder (the tests' hooks).
int move_rename(const std::string& from, const std::string& to) {
#ifdef CONFIG_MOVE_TEST
    if (g_config_move_test_no_remove) return EACCES;
    if (g_config_move_test_copy) return EXDEV;
#endif
    return rename_noreplace(from, to);
}
int remove_file(const std::string& p) {
#ifdef CONFIG_MOVE_TEST
    if (g_config_move_test_no_remove) return EACCES;
#endif
    return ::unlink(p.c_str()) == 0 ? 0 : errno;
}

bool same_bytes(const std::string& a, const std::string& b) {
    std::ifstream x(a, std::ios::binary), y(b, std::ios::binary);
    if (!x || !y) return false;
    std::vector<char> p(1 << 16), q(1 << 16);
    for (;;) {
        x.read(p.data(), static_cast<std::streamsize>(p.size()));
        y.read(q.data(), static_cast<std::streamsize>(q.size()));
        const std::streamsize n = x.gcount();
        if (n != y.gcount() || x.bad() || y.bad() || std::memcmp(p.data(), q.data(), static_cast<std::size_t>(n)) != 0) return false;
        if (n < static_cast<std::streamsize>(p.size())) return true;  // both at their end, after as many bytes
    }
}

// `from` (a link followed) copied to `to`, which is not there: written
// beside it under a hidden name, its mode and times as they were, synced,
// compared byte for byte, then renamed to `to`. 0, or what failed.
int copy_file_checked(const std::string& from, const std::string& to) {
    const std::size_t slash = to.find_last_of('/');
    const std::string tmp = to.substr(0, slash + 1) + "." + to.substr(slash + 1) + ".bbhost-move";
    const int in = ::open(from.c_str(), O_RDONLY | O_CLOEXEC);
    if (in < 0) return errno;
    struct stat st {};
    if (::fstat(in, &st) != 0) {
        const int e = errno;
        ::close(in);
        return e;
    }
    ::unlink(tmp.c_str());  // one a start that stopped half-way left
    const int out = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (out < 0) {
        const int e = errno;
        ::close(in);
        return e;
    }
    int e = 0;
    std::vector<char> buf(1 << 16);
    while (!e) {
        const ssize_t n = ::read(in, buf.data(), buf.size());
        if (n < 0) {
            if (errno != EINTR) e = errno;
            continue;
        }
        if (n == 0) break;
        for (ssize_t off = 0; off < n && !e;) {
            const ssize_t w = ::write(out, buf.data() + off, static_cast<std::size_t>(n - off));
            if (w < 0) {
                if (errno != EINTR) e = errno;
                continue;
            }
            off += w;
        }
    }
    if (!e && ::fchmod(out, st.st_mode & 07777) != 0) e = errno;
    if (!e) {
        const struct timespec times[2] = {st.st_atim, st.st_mtim};
        ::futimens(out, times);
    }
    if (!e && ::fsync(out) != 0) e = errno;
    if (::close(out) != 0 && !e) e = errno;
    ::close(in);
    if (!e && !same_bytes(from, tmp)) e = EIO;
#ifdef CONFIG_MOVE_TEST
    if (!e && g_config_move_test_fail_copy > 0 && --g_config_move_test_fail_copy == 0) e = EIO;
#endif
    if (!e) e = rename_noreplace(tmp, to);
    if (e) ::unlink(tmp.c_str());
    return e;
}

// One thing done in the new folder, to undo when a later one fails or to
// finish (the original removed) once all of them are done.
struct Op {
    enum Kind { Renamed, Copied, Made, Same } kind;
    std::string src, dst, rel;
    bool through_link = false;  // reached through a link: the original is never removed
};

struct Move {
    std::string from, to, date;
    ConfigMove* m = nullptr;
    std::vector<Seen> seen_in, seen_out;  // the record's originals; those still to leave alone
    std::vector<Op> ops;
    std::vector<std::string> dirs;  // the old folder's folders emptied (relative), deepest first
    std::string error;

    // An original kept at an earlier start, as it was then.
    bool seen_before(const std::string& rel, const struct stat& st) {
        for (const Seen& s : seen_in) {
            if (s.rel == rel && s.size == static_cast<long long>(st.st_size) && s.mtime == mtime_ns(st)) {
                seen_out.push_back(s);
                return true;
            }
        }
        return false;
    }
    void keep(const std::string& rel, const std::string& why) {
        struct stat st {};
        if (::stat((from + "/" + rel).c_str(), &st) != 0) return;
        m->kept.push_back({rel, why});
        seen_out.push_back({rel, static_cast<long long>(st.st_size), mtime_ns(st)});
    }

    // The first free name beside `dst` for the old one.
    std::string dated(const std::string& dst) const {
        for (int n = 1; n < 1000; ++n) {
            const std::string p = dst + ".from-config-" + date + (n > 1 ? "-" + std::to_string(n) : std::string());
            struct stat st {};
            if (::lstat(p.c_str(), &st) != 0) return p;
        }
        return dst + ".from-config-" + date + "-" + std::to_string(::getpid());
    }

    // `rel` (in the old folder) to `dst` (in the new one). `listed`: placed,
    // its name goes on the notice's list - not inside a folder placed whole,
    // whose name is there already.
    bool entry(const std::string& rel, const std::string& dst, bool via_link, bool listed) {
        const std::string src = from + "/" + rel;
        struct stat ls {}, st {};
        if (::lstat(src.c_str(), &ls) != 0) return true;  // gone since it was listed
        const bool through = via_link || S_ISLNK(ls.st_mode);
        if (::stat(src.c_str(), &st) != 0 || (!S_ISDIR(st.st_mode) && !S_ISREG(st.st_mode))) {
            // A link to nothing, a pipe, a socket: nothing bbhost keeps, and nothing to copy.
            m->left.push_back(rel);
            return true;
        }
        const bool dir = S_ISDIR(st.st_mode);
        if (!dir && seen_before(rel, st)) return true;
        struct stat dl {}, ds {};
        if (::lstat(dst.c_str(), &dl) != 0) return put(rel, dst, dir, through, st, listed, false);
        const bool dst_dir = ::stat(dst.c_str(), &ds) == 0 && S_ISDIR(ds.st_mode);
        const bool dst_file = ::stat(dst.c_str(), &ds) == 0 && S_ISREG(ds.st_mode);
        if (dir && dst_dir) {
            // Both have the folder: what is in the old one goes in one by one.
            for (const std::string& n : names_in(src)) {
                if (!entry(rel + "/" + n, dst + "/" + n, through, listed)) return false;
            }
            if (!through) dirs.push_back(rel);
            return true;
        }
        if (!dir && dst_file && ds.st_size == st.st_size && same_bytes(src, dst)) {
            // The one there already: taken as it. Nothing is removed through a link.
            if (through) {
                seen_out.push_back({rel, static_cast<long long>(st.st_size), mtime_ns(st)});
                return true;
            }
            ops.push_back({Op::Same, src, dst, rel, false});
            m->same.push_back(rel);
            return true;
        }
        // The one there stays in use; the old one goes beside it.
        return put(rel, dated(dst), dir, through, st, listed, true);
    }

    // `rel` placed at `dst`, which is not there: renamed, else copied.
    bool put(const std::string& rel, const std::string& dst, bool dir, bool through, const struct stat& st, bool listed, bool beside) {
        const auto list = [&] {
            if (beside) m->beside.push_back({rel, dst.substr(to.size() + 1)});
            else if (listed) m->moved.push_back(rel);
        };
        if (!through) {
            // Another file system (EXDEV), or an old folder that will not give
            // it up (EACCES, EPERM, EROFS): copied instead.
            if (move_rename(from + "/" + rel, dst) == 0) {
                ops.push_back({Op::Renamed, from + "/" + rel, dst, rel, false});
                list();
                return true;
            }
        }
        if (!dir) {
            if (const int e = copy_file_checked(from + "/" + rel, dst)) {
                error = "copying " + rel + " failed: " + std::strerror(e);
                return false;
            }
            ops.push_back({Op::Copied, from + "/" + rel, dst, rel, through});
            list();
            return true;
        }
        if (::mkdir(dst.c_str(), (st.st_mode & 07777) | 0700) != 0) {
            error = "making " + dst + " failed: " + std::strerror(errno);
            return false;
        }
        ops.push_back({Op::Made, from + "/" + rel, dst, rel, through});
        list();
        for (const std::string& n : names_in(from + "/" + rel)) {
            if (!entry(rel + "/" + n, dst + "/" + n, through, false)) return false;
        }
        if (!through) dirs.push_back(rel);
        return true;
    }

    // Everything in place: the originals that were copied removed (not
    // through a link; one the folder refuses stays, remembered), then the
    // folders that emptied.
    void finish() {
        for (const Op& o : ops) {
            if (o.kind != Op::Copied && o.kind != Op::Same) continue;
            if (o.through_link) {
                keep(o.rel, "a link");
                continue;
            }
            if (const int e = remove_file(o.src)) {
                if (o.kind == Op::Same) m->same.erase(std::remove(m->same.begin(), m->same.end(), o.rel), m->same.end());
                keep(o.rel, std::strerror(e));
            }
        }
        for (const std::string& d : dirs) {
#ifdef CONFIG_MOVE_TEST
            if (g_config_move_test_no_remove) continue;
#endif
            ::rmdir((from + "/" + d).c_str());  // fails while it holds anything
        }
    }

    // A failure: everything this start did undone, last first. What cannot
    // be put back is named.
    std::vector<std::string> undo() {
        std::vector<std::string> stuck;
        for (auto o = ops.rbegin(); o != ops.rend(); ++o) {
            if (o->kind == Op::Renamed && rename_noreplace(o->dst, o->src) != 0) stuck.push_back(o->dst);
            if (o->kind == Op::Copied) ::unlink(o->dst.c_str());
            if (o->kind == Op::Made) ::rmdir(o->dst.c_str());
        }
        return stuck;
    }
};

bool is_relative(const std::string& v) {
    if (v.empty() || v[0] == '/' || v[0] == '\\') return false;
    return !(v.size() >= 2 && v[1] == ':');  // a drive letter
}

std::string unquote(const std::string& v) {
    if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front()) return v.substr(1, v.size() - 2);
    return v;
}

// The moved bbhost.toml's [paths]: one that points into the old folder still
// does (nothing of the player's moved from there), and one relative to it -
// relative to the file's own folder - is written out in full, so it names the
// same place from the new folder.
void check_paths(ConfigMove& m) {
    if (std::find(m.moved.begin(), m.moved.end(), "bbhost.toml") == m.moved.end()) return;
    const std::string file = m.to + "/bbhost.toml";
    std::map<std::string, std::string> kv;
    std::string err;
    if (!config_parse_toml(file, &kv, &err)) return;
    std::vector<ConfigEdit> edits;
    for (const char* key : {"app0", "data", "tmp", "mods", "eboot"}) {
        const auto it = kv.find(std::string("paths.") + key);
        if (it == kv.end()) continue;
        const std::string v = unquote(it->second);
        if (v.empty() || v.find("/path/to/") != std::string::npos || v.find("PATH-TO-") != std::string::npos) continue;
        if (is_relative(v)) {
            const std::string full = m.from + "/" + (v.rfind("./", 0) == 0 ? v.substr(2) : v);
            edits.push_back({"paths", key, "\"" + full + "\""});
            m.paths.push_back({std::string("paths.") + key, full, true});
        } else {
            for (const std::string& old : {m.from, m.link}) {
                if (!old.empty() && (v == old || v.rfind(old + "/", 0) == 0)) {
                    m.paths.push_back({std::string("paths.") + key, v, false});
                    break;
                }
            }
        }
    }
    if (!edits.empty() && !config_set_values(file, edits)) {
        host_log("config: could not write the relative paths of %s in full; they are read from %s now", file.c_str(), m.to.c_str());
        m.paths.erase(std::remove_if(m.paths.begin(), m.paths.end(), [](const ConfigMove::Path& p) { return p.made_absolute; }), m.paths.end());
    }
}

#endif  // !_WIN32

}  // namespace

ConfigMove config_move_settings(const std::string& from, const std::string& to) {
    ConfigMove m;
    m.from = from;
    m.to = to;
#if defined(_WIN32)
    return m;
#else
    m.when = local_time("%Y-%m-%d %H:%M");
    struct stat ls {}, st {};
    if (from.empty() || to.empty() || ::lstat(from.c_str(), &ls) != 0) return m;  // no old folder
    if (::stat(from.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return m;
    const std::vector<std::string> names = names_in(from);
    if (std::none_of(names.begin(), names.end(), owned)) return m;
    std::error_code ec;
    if (fs::exists(to, ec) && fs::equivalent(from, to, ec)) return m;  // one folder, by a link
    if (S_ISLNK(ls.st_mode)) {
        const fs::path real = fs::canonical(from, ec);
        m.link = ec ? std::string("another folder") : real.string();
    }
    struct stat ts {};
    if (::stat(to.c_str(), &ts) != 0) {
        fs::create_directories(to, ec);
        if (ec) {
            m.failed = "making " + to + " failed: " + ec.message();
            host_log("%s", config_move_log_line(m).c_str());
            return m;
        }
        ::chmod(to.c_str(), (st.st_mode & 07777) | 0700);  // as private as the old one, and writable
    } else if (!S_ISDIR(ts.st_mode)) {
        m.failed = to + " is not a folder";
        host_log("%s", config_move_log_line(m).c_str());
        return m;
    }
    Record rec = read_record(to);
    Move mv;
    mv.from = from;
    mv.to = to;
    mv.date = local_time("%Y-%m-%d");
    mv.m = &m;
    mv.seen_in = rec.seen;
    bool ok = true;
    for (const std::string& n : names) {
        if (!owned(n)) {
            std::error_code dec;
            m.left.push_back(n + (fs::is_directory(from + "/" + n, dec) ? "/" : ""));
            continue;
        }
        if (!mv.entry(n, to + "/" + n, !m.link.empty(), true)) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        const std::vector<std::string> stuck = mv.undo();
        m.failed = mv.error;
        if (!stuck.empty()) m.failed += "; and could not be put back: " + join(stuck);
        m.moved.clear();
        m.beside.clear();
        m.same.clear();
        m.kept.clear();
        host_log("%s", config_move_log_line(m).c_str());
        return m;
    }
    mv.finish();
    if (m.link.empty()) ::rmdir(from.c_str());  // emptied: gone (it stays while it holds anything)
    check_paths(m);
    const bool moved = !m.moved.empty() || !m.beside.empty() || !m.same.empty() || !m.kept.empty();
    const auto key = [](const Seen& s) { return s.rel + '\t' + std::to_string(s.size) + '\t' + std::to_string(s.mtime); };
    std::vector<std::string> before, after;
    for (const Seen& s : rec.seen) before.push_back(key(s));
    for (const Seen& s : mv.seen_out) after.push_back(key(s));
    std::sort(before.begin(), before.end());
    std::sort(after.begin(), after.end());
    if (moved || before != after) {
        rec.seen = mv.seen_out;
        if (moved) rec.moves.push_back(m);
        if (!write_record(to, rec)) {
            host_log("config: could not write %s/%s: this start's setup window tells the move, the next one does not", to.c_str(),
                     kConfigMoveRecord);
            if (moved) g_unrecorded.push_back(m);
        }
    }
    if (moved) host_log("%s", config_move_log_line(m).c_str());
    return m;
#endif
}

void config_move_at_start(const std::string& from, const std::string& to) {
    if (g_started) return;
    g_started = true;
    g_new_dir = to;
    if (from.empty() || to.empty() || from == to) return;
    const ConfigMove m = config_move_settings(from, to);
    if (!m.failed.empty()) {
        g_fallback = from;
        g_unrecorded.push_back(m);
    }
}

std::string config_move_fallback() { return g_fallback; }

std::vector<ConfigMove> config_moves_to_show() {
    std::vector<ConfigMove> out;
    if (!g_new_dir.empty()) out = read_record(g_new_dir).moves;
    out.insert(out.end(), g_unrecorded.begin(), g_unrecorded.end());
    return out;
}

void config_moves_dismiss() {
    g_unrecorded.clear();
    if (g_new_dir.empty()) return;
    Record r = read_record(g_new_dir);
    if (r.moves.empty()) return;
    r.moves.clear();
    if (write_record(g_new_dir, r)) host_log("config: the settings folder's notice dismissed");
    else host_log("config: could not write %s/%s: the notice comes back at the next start", g_new_dir.c_str(), kConfigMoveRecord);
}

std::string config_move_heading(const ConfigMove& m) {
    if (!m.failed.empty()) return "Your settings could not be moved";
    return "Your settings moved to " + tilde(m.to);
}

std::vector<ConfigMoveLine> config_move_lines(const ConfigMove& m) {
    std::vector<ConfigMoveLine> out;
    const std::string from = tilde(m.from), to = tilde(m.to);
    if (!m.failed.empty()) {
        out.push_back({"bbhost keeps its settings in " + to + " now, but could not move yours from " + from + ": " + tilde_in(m.failed) +
                           ". This start uses them where they are; the next one tries again.",
                       true});
        return out;
    }
    out.push_back({"bbhost keeps its settings in " + to + " now, no longer in " + from + "." +
                   (m.moved.empty() ? std::string() : (m.link.empty() ? " Moved from there: " : " Copied from there: ") + join(tops(m.moved)) + ".")});
    for (const ConfigMove::Beside& b : m.beside) {
        out.push_back({to + "/" + b.name + " was already there and stays in use. The one from " + from + " is beside it: " + to + "/" +
                           b.as + ".",
                       true});
    }
    if (!m.same.empty())
        out.push_back({"The same as the ones already in " + to + ", so those are kept: " + join(m.same) + "."});
    if (!m.link.empty()) {
        if (!m.kept.empty())
            out.push_back({from + " is a link to " + tilde(m.link) + ", so nothing was taken out of it: its files were copied and are still there. bbhost uses the copies.",
                           true});
    } else {
        // By why they stayed: links, then each error.
        std::vector<std::string> whys;
        for (const ConfigMove::Kept& k : m.kept) {
            if (std::find(whys.begin(), whys.end(), k.why) == whys.end()) whys.push_back(k.why);
        }
        for (const std::string& why : whys) {
            std::vector<std::string> names;
            for (const ConfigMove::Kept& k : m.kept) {
                if (k.why == why) names.push_back(k.name);
            }
            if (why == "a link")
                out.push_back({"Links in " + from + ", copied and left as they are: " + join(tops(names)) + ". bbhost uses the copies.", true});
            else
                out.push_back({"Copied, but still in " + from + " as well (" + why + "): " + join(tops(names)) +
                                   ". bbhost uses the copies; the ones there can be deleted.",
                               true});
        }
    }
    if (!m.left.empty()) out.push_back({"Left in " + from + ", as they are not bbhost's: " + join(m.left) + "."});
    std::vector<std::string> into;
    for (const ConfigMove::Path& p : m.paths) {
        if (p.made_absolute)
            out.push_back({p.key + " in bbhost.toml was relative to " + from + ", so it now says the same place in full: " + tilde(p.value) + "."});
        else
            into.push_back(p.key);
    }
    if (!into.empty())
        out.push_back({join(into) + " in bbhost.toml " + (into.size() == 1 ? "points" : "point") + " into " + from + " and still " +
                       (into.size() == 1 ? "works" : "work") + ": nothing of yours was moved from there."});
    if (m.link.empty() && m.kept.empty())
        out.push_back({"An older bbhost still looks in " + from + " and does not find them; anything it leaves there is moved here at the next start."});
    else
        out.push_back({"An older bbhost still reads the originals in " + from + ", which this one no longer changes."});
    return out;
}

std::string config_move_log_line(const ConfigMove& m) {
    if (!m.failed.empty())
        return "config: the settings could not be moved from " + m.from + " to " + m.to + " (" + m.failed + "); this start uses " + m.from;
    std::string s = "config: the settings moved from " + m.from + " to " + m.to;
    if (!m.link.empty()) s += " (a link to " + m.link + ": copied, nothing removed there)";
    const auto part = [&s](const char* what, const std::string& list) {
        if (!list.empty()) s += std::string("; ") + what + ": " + list;
    };
    part("moved", join(tops(m.moved)));
    std::vector<std::string> v;
    for (const ConfigMove::Beside& b : m.beside) v.push_back(b.name + " as " + b.as);
    part("beside the ones there", join(v));
    part("the same as the ones there", join(m.same));
    v.clear();
    for (const ConfigMove::Kept& k : m.kept) v.push_back(k.name + " (" + k.why + ")");
    part("copied, still in the old folder", join(v));
    part("left (not bbhost's)", join(m.left));
    v.clear();
    for (const ConfigMove::Path& p : m.paths) v.push_back(p.key + (p.made_absolute ? " (was relative, now " + p.value + ")" : ""));
    part("paths into the old folder", join(v));
    return s + "; the setup window tells it until dismissed";
}

#ifdef CONFIG_MOVE_TEST
void config_move_test_reset() {
    g_started = false;
    g_new_dir.clear();
    g_fallback.clear();
    g_unrecorded.clear();
    g_config_move_test_copy = false;
    g_config_move_test_fail_copy = 0;
    g_config_move_test_no_remove = false;
}
#endif
