// A save written whole or not at all: see hle/save_commit.h.
#include "hle/save_commit.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace save_commit {
namespace {

namespace fs = std::filesystem;

std::atomic<Hook> g_hook{nullptr};
std::atomic<bool> g_no_exchange{false};
std::atomic<bool> g_exchange_refused{false};

void step(const char* name) {
    if (const Hook h = g_hook.load(std::memory_order_relaxed)) h(name);
}

bool exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(fs::path(p), ec);
}

std::string parent_of(const std::string& p) { return fs::path(p).parent_path().string(); }

// A directory's entries on the disk (Linux: its own fsync; Windows writes a
// rename through with MOVEFILE_WRITE_THROUGH, and there is no handle to flush).
void sync_dir(const std::string& dir) {
#if !defined(_WIN32)
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
#else
    (void)dir;
#endif
}

bool flush_file(const fs::path& p) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(p.wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    const bool ok = FlushFileBuffers(h) != FALSE;
    CloseHandle(h);
    return ok;
#else
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
#endif
}

// Every file under `dir` and every directory's entries on the disk.
bool flush_tree(const std::string& dir) {
    std::error_code ec;
    bool ok = true;
    for (fs::recursive_directory_iterator it(fs::path(dir), ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec)) ok = flush_file(it->path()) && ok;
        else if (it->is_directory(ec)) sync_dir(it->path().string());
    }
    sync_dir(dir);
    return ok && !ec;
}

// A file written whole and flushed, its directory entry too.
bool write_marker(const std::string& path, const std::string& text) {
#if defined(_WIN32)
    HANDLE h = CreateFileW(fs::path(path).wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD put = 0;
    const bool ok = WriteFile(h, text.data(), static_cast<DWORD>(text.size()), &put, nullptr) && put == text.size() &&
                    FlushFileBuffers(h);
    CloseHandle(h);
    return ok;
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    const bool ok = ::write(fd, text.data(), text.size()) == static_cast<ssize_t>(text.size()) && ::fsync(fd) == 0;
    ::close(fd);
    sync_dir(parent_of(path));
    return ok;
#endif
}

std::string read_marker(const std::string& path) {
    std::string text;
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
        char buf[128];
        const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        text.assign(buf, n);
    }
    return text;
}

void remove_file(const std::string& path) {
    std::error_code ec;
    fs::remove(fs::path(path), ec);
    sync_dir(parent_of(path));
}

void remove_tree(const std::string& path) {
    std::error_code ec;
    fs::remove_all(fs::path(path), ec);
    sync_dir(parent_of(path));
}

// One directory renamed to a name nothing has. Windows: written through, and
// tried again for ~5 s while another process holds a file in it (an antivirus
// scan, OneDrive) - Windows refuses a rename over an open file's directory.
bool rename_dir(const std::string& from, const std::string& to) {
#if defined(_WIN32)
    const std::wstring a = fs::path(from).wstring(), b = fs::path(to).wstring();
    for (int i = 0;; ++i) {
        if (MoveFileExW(a.c_str(), b.c_str(), MOVEFILE_WRITE_THROUGH)) return true;
        const DWORD e = GetLastError();
        if (i >= 50 || (e != ERROR_ACCESS_DENIED && e != ERROR_SHARING_VIOLATION && e != ERROR_LOCK_VIOLATION)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#else
    if (::rename(from.c_str(), to.c_str()) != 0) return false;
    sync_dir(parent_of(to));
    return true;
#endif
}

// The two directories swapped in one step (renameat2 RENAME_EXCHANGE): 0, or
// the errno - EINVAL/ENOTSUP/ENOSYS where the file system or the kernel has
// no exchange (exFAT, vfat, NFS), and always on Windows.
int exchange_dirs(const std::string& a, const std::string& b) {
#if defined(_WIN32) || !defined(SYS_renameat2)
    (void)a;
    (void)b;
    return EINVAL;
#else
    if (g_no_exchange.load(std::memory_order_relaxed) || g_exchange_refused.load(std::memory_order_relaxed)) return EINVAL;
    constexpr unsigned kRenameExchange = 1u << 1;  // RENAME_EXCHANGE (linux/fs.h)
    if (::syscall(SYS_renameat2, AT_FDCWD, a.c_str(), AT_FDCWD, b.c_str(), kRenameExchange) != 0) return errno;
    sync_dir(parent_of(b));
    return 0;
#endif
}

std::uint64_t inode_of(const std::string& p) {
#if defined(_WIN32)
    (void)p;
    return 0;
#else
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0 ? static_cast<std::uint64_t>(st.st_ino) : 0;
#endif
}

// A directory tree copied file by file (a test may end the process between two).
bool copy_tree(const std::string& from, const std::string& to) {
    std::error_code ec;
    fs::create_directories(fs::path(to), ec);
    if (ec) return false;
    for (fs::recursive_directory_iterator it(fs::path(from), ec), end; !ec && it != end; it.increment(ec)) {
        const fs::path rel = fs::relative(it->path(), fs::path(from), ec);
        if (ec) return false;
        const fs::path dst = fs::path(to) / rel;
        if (it->is_directory(ec)) {
            fs::create_directories(dst, ec);
        } else if (it->is_regular_file(ec)) {
            fs::copy_file(it->path(), dst, fs::copy_options::overwrite_existing, ec);
            step("copy");
        }
        if (ec) return false;
    }
    return !ec;
}

// The two-rename swap's state, from the marker and which directories exist:
// the stage made the save, the old copy and the marker removed. Shared by a
// commit and a recovery; false leaves what is there for the next recovery.
bool finish_by_renames(const Paths& p, std::string* why) {
    if (exists(p.stage)) {
        if (exists(p.live)) {
            if (exists(p.old)) remove_tree(p.old);  // a finished commit's leftover
            if (!rename_dir(p.live, p.old)) {
                if (why) *why = "the save -> .old";
                return false;
            }
            step("live-moved");
        }
        if (!rename_dir(p.stage, p.live)) {
            if (why) *why = "the stage -> the save";
            return false;
        }
        step("stage-moved");
    } else if (!exists(p.live) && exists(p.old)) {
        if (!rename_dir(p.old, p.live)) {  // cannot happen in this protocol: the old save back
            if (why) *why = ".old -> the save";
            return false;
        }
    }
    remove_file(p.marker);
    step("marker-removed");
    if (exists(p.old)) remove_tree(p.old);
    step("old-removed");
    return true;
}

}  // namespace

Paths paths_of(const std::string& live) {
    const fs::path l(live);
    const fs::path dir = l.parent_path();
    const std::string name = l.filename().string();
    return Paths{live, (dir / ("." + name + ".stage")).string(), (dir / ("." + name + ".old")).string(),
                 (dir / ("." + name + ".commit")).string()};
}

std::string recover(const std::string& live, std::string* did) {
    const Paths p = paths_of(live);
    std::string note;
    if (exists(p.marker)) {
        const std::string m = read_marker(p.marker);
        if (m.rfind("exchange ", 0) == 0) {
            const std::uint64_t ino = std::strtoull(m.c_str() + 9, nullptr, 10);
            if (ino && exists(p.live) && inode_of(p.live) == ino) {
                // Swapped: the stage's name holds the old save now (the
                // exchange), or .old does (the renames, where it had none).
                if (exists(p.stage)) remove_tree(p.stage);
                if (exists(p.old)) remove_tree(p.old);
                remove_file(p.marker);
                if (did) *did = "finished a commit (the old save removed)";
                return live;
            }
            if (!(ino && exists(p.stage) && inode_of(p.stage) == ino)) {
                // Neither side is the stage that was written: nothing is
                // dropped - the save stays, the stage is kept aside.
                if (exists(p.stage)) rename_dir(p.stage, p.stage + ".unsure");
                remove_file(p.marker);
                if (did) *did = "a commit's marker matched neither directory: the save kept, the stage moved aside";
                return live;
            }
            // Not swapped yet: the renames finish it.
        }
        std::string why;
        if (!finish_by_renames(p, &why)) {
            if (did) *did = "a commit could not be finished (" + why + "): the stage holds the newest save";
            return exists(p.stage) ? p.stage : live;
        }
        note = "finished a commit";
    } else {
        if (exists(p.stage)) {
            remove_tree(p.stage);
            note = "dropped a session that never committed";
        }
        if (exists(p.old)) {
            if (exists(p.live)) {
                remove_tree(p.old);
            } else if (!rename_dir(p.old, p.live)) {
                if (did) *did = "the old save could not be put back";
                return p.old;
            }
            note += note.empty() ? "cleared an old copy" : ", cleared an old copy";
        }
    }
    if (did) *did = note;
    return live;
}

std::string begin(const std::string& live) {
    const Paths p = paths_of(live);
    if (exists(p.marker)) return {};  // a commit the recovery could not finish: its stage is the save
    if (exists(p.stage)) remove_tree(p.stage);
    std::error_code ec;
    const bool ok = exists(p.live) ? copy_tree(p.live, p.stage) : fs::create_directories(fs::path(p.stage), ec) && !ec;
    if (!ok) {
        remove_tree(p.stage);
        return {};
    }
    return p.stage;
}

bool commit(const std::string& live, std::string* why) {
    const Paths p = paths_of(live);
    if (!exists(p.stage)) {
        if (why) *why = "no stage";
        return false;
    }
    if (!flush_tree(p.stage)) {
        if (why) *why = "flushing the stage";
        return false;
    }
    step("flushed");
    if (exists(p.live)) {
        const std::uint64_t ino = inode_of(p.stage);
        if (ino && !g_no_exchange.load(std::memory_order_relaxed)) {
            if (!write_marker(p.marker, "exchange " + std::to_string(ino) + "\n")) {
                if (why) *why = "the marker";
                return false;
            }
            step("marker");
            const int e = exchange_dirs(p.stage, p.live);
            if (e == 0) {
                step("exchanged");
                remove_tree(p.stage);  // the old save, under the stage's name now
                step("old-removed");
                remove_file(p.marker);
                step("marker-removed");
                return true;
            }
            // No exchange here (exFAT, vfat, NFS): the renames, under the same
            // marker - a recovery reads it the same way, the stage being
            // still the inode it names.
            if (!finish_by_renames(p, why)) return false;
            return true;
        }
    }
    if (!write_marker(p.marker, "rename\n")) {
        if (why) *why = "the marker";
        return false;
    }
    step("marker");
    return finish_by_renames(p, why);
}

void set_test_hook(Hook hook) { g_hook.store(hook); }
void set_no_exchange(bool on) { g_no_exchange.store(on); }
void set_exchange_refused(bool on) { g_exchange_refused.store(on); }

}  // namespace save_commit
