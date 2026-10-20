#include "core/paced_file.h"

#include <algorithm>
#include <filesystem>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

HostPacedFile::HostPacedFile(std::string path, std::string tmp, std::size_t chunk)
    : path_(std::move(path)), tmp_(std::move(tmp)), chunk_(std::max<std::size_t>(chunk, 1)) {
    f_ = std::fopen(tmp_.c_str(), "wb");
    ok_ = f_ != nullptr;
}

HostPacedFile::~HostPacedFile() {
    if (f_) std::fclose(f_);
    if (!committed_) {
        std::error_code ec;
        std::filesystem::remove(tmp_, ec);
    }
}

bool HostPacedFile::sync() {
    if (std::fflush(f_) != 0) return false;
#if defined(_WIN32)
    return _commit(_fileno(f_)) == 0;
#elif defined(__APPLE__)
    // No fdatasync on macOS; F_FULLFSYNC (drive cache flush) is not needed.
    return ::fsync(fileno(f_)) == 0;
#else
    return ::fdatasync(fileno(f_)) == 0;
#endif
}

bool HostPacedFile::write(const void* data, std::size_t size) {
    const char* p = static_cast<const char*>(data);
    while (ok_ && size) {
        const std::size_t n = std::min(size, chunk_ - unsynced_);
        ok_ = std::fwrite(p, 1, n, f_) == n;
        p += n;
        size -= n;
        unsynced_ += n;
        if (ok_ && unsynced_ >= chunk_) {
            ok_ = sync();
            unsynced_ = 0;
        }
    }
    return ok_;
}

bool HostPacedFile::commit() {
    if (ok_ && unsynced_) ok_ = sync();
    if (f_) ok_ = std::fclose(f_) == 0 && ok_;
    f_ = nullptr;
    std::error_code ec;
    if (ok_) std::filesystem::rename(tmp_, path_, ec);
    committed_ = ok_ && !ec;
    return committed_;
}
