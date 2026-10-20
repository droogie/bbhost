#include "hle/libc_stdio.h"
#include "guest_abi.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>

thread_local int t_guest_errno;

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

__attribute__((sysv_abi, noinline)) int guest_seek(FILE* f, std::int64_t offset, int whence) {
    return libc_result(bb::libc::seek(f, offset, whence));
}
__attribute__((sysv_abi, noinline)) std::int64_t guest_tell(FILE* f) {
    const auto offset = bb::libc::tell(f);
    if (offset == -1) set_guest_errno_host(errno);
    return offset;
}
__attribute__((sysv_abi, noinline)) int guest_position(FILE* f, bb::libc::Fpos* p) {
    return libc_result(bb::libc::get_position(f, p));
}

extern "C" GUEST_ABI int stdio_print_body(FILE* file, const char* fmt, SysvVaList* ap) {
    return bb::libc::print_file(file, fmt, ap);
}
extern "C" GUEST_ABI int stdio_scan_body(FILE* file, const char* fmt, SysvVaList* ap) {
    return bb::libc::scan_file(file, fmt, ap);
}
SYSV_VA_ENTRY(stdio_print, stdio_print_body, 2, rdx)
SYSV_VA_ENTRY(stdio_scan, stdio_scan_body, 2, rdx)

int main() {
    using namespace bb::libc;
    FILE* f = std::tmpfile();
    CHECK(f);
    // Move the cursor only: no multi-gigabyte data is allocated or written.
    constexpr std::int64_t large = 0x100000123LL;
    CHECK(guest_seek(f, large, SEEK_SET) == 0);
    CHECK(guest_tell(f) == large);
    CHECK(seek(f, -17, SEEK_CUR) == 0);
    CHECK(tell(f) == large - 17);
    struct Guarded { Fpos pos; unsigned char guard[8]; } out;
    std::memset(&out, 0xa5, sizeof(out));
    CHECK(guest_position(f, &out.pos) == 0);
    CHECK(out.pos.offset == large - 17 && out.pos.state[0] == 0 && out.pos.state[1] == 0);
    for (auto b : out.guard) CHECK(b == 0xa5);
    CHECK(seek(f, 0, SEEK_END) == 0 && tell(f) == 0);
    errno = 0;
    CHECK(guest_seek(f, -1, SEEK_SET) == -1 && errno == EINVAL && t_guest_errno == 22);
    CHECK(std::fclose(f) == 0);
    t_guest_errno = 77;
    CHECK(!open_file("bbhost-libc-errno-missing-directory/file", "rb") && t_guest_errno == 2);
    errno = ERANGE;  // logging or other host work must not replace guest errno
    CHECK(t_guest_errno == 2);
#if defined(_WIN32)
    const char* null_path = "NUL";
#else
    const char* null_path = "/dev/null";
#endif
    t_guest_errno = 63;
    FILE* read_only = open_file(null_path, "rb");
    CHECK(read_only && t_guest_errno == 63);
    errno = EINVAL;
    CHECK(file_result(read_only, [&] { return std::fgetc(read_only); }, read_access) == EOF);
    CHECK(!std::ferror(read_only) && t_guest_errno == 63);
    CHECK(file_result(read_only, [&] { return std::fputc('x', read_only); }, write_access) == EOF);
    CHECK(std::ferror(read_only) && errno != 0 && t_guest_errno == 63);
    std::clearerr(read_only);
    CHECK(stdio_print(read_only, "%d", 42) == -1 && std::ferror(read_only) && t_guest_errno == 63);
    CHECK(close_file(read_only) == 0);
    FILE* write_only = open_file(null_path, "wb");
    CHECK(write_only);
    CHECK(file_result(write_only, [&] { return std::fgetc(write_only); }, read_access) == EOF);
    CHECK(std::ferror(write_only) && t_guest_errno == 63);
    std::clearerr(write_only);
    int scanned = -1;
    CHECK(stdio_scan(write_only, "%d", &scanned) == EOF && std::ferror(write_only) && t_guest_errno == 63);
    CHECK(scanned == -1);
    CHECK(reopen_file(null_path, "rb", write_only) == write_only);
    CHECK(file_result(write_only, [&] { return std::fputc('x', write_only); }, write_access) == EOF);
    CHECK(std::ferror(write_only) && t_guest_errno == 63);
    CHECK(close_file(write_only) == 0);
    FILE* read_write = open_file(null_path, "r+b");
    CHECK(read_write && file_access(read_write) == (read_access | write_access));
    CHECK(close_file(read_write) == 0);
    FILE* messages = std::tmpfile();
    CHECK(messages);
    t_guest_errno = 35;
    CHECK(stdio_print(messages, "%ld %.2f\n", 1234567890123LL, 1.25) == 19 && t_guest_errno == 35);
    CHECK(seek(messages, 0, SEEK_SET) == 0);
    std::int64_t number = 0;
    double fraction = 0;
    CHECK(stdio_scan(messages, "%ld %lf", &number, &fraction) == 2 && t_guest_errno == 35);
    CHECK(number == 1234567890123LL && fraction == 1.25);
    CHECK(stdio_scan(messages, "%ld", &number) == EOF && !std::ferror(messages) && t_guest_errno == 35);
    CHECK(seek(messages, 0, SEEK_SET) == 0);
    errno = EINVAL;
    print_error(messages, "opening");
    print_error(messages, "");
    print_error(messages, nullptr);
    CHECK(t_guest_errno == 35);
    CHECK(seek(messages, 0, SEEK_SET) == 0);
    char line[128];
    CHECK(std::fgets(line, sizeof(line), messages));
    CHECK(std::strcmp(line, "opening: Resource temporarily unavailable\n") == 0);
    for (int i = 0; i < 2; ++i) {
        CHECK(std::fgets(line, sizeof(line), messages));
        CHECK(std::strcmp(line, "Resource temporarily unavailable\n") == 0);
    }
    CHECK(std::fclose(messages) == 0);
#if !defined(_WIN32)
#if defined(__linux__)  // macOS has no /dev/full
    FILE* full = open_file("/dev/full", "wb");
    CHECK(full && std::setvbuf(full, nullptr, _IONBF, 0) == 0);
    CHECK(file_result(full, [&] { return std::fputc('x', full); }, write_access) == EOF);
    CHECK(t_guest_errno == 28 && std::ferror(full));
    std::clearerr(full);
    t_guest_errno = 73;
    CHECK(stdio_print(full, "%s %d", "payload", 42) == -1 && t_guest_errno == 28 && std::ferror(full));
    close_file(full);
#endif
    int descriptors[2];
    CHECK(::pipe(descriptors) == 0);
    FILE* pipe = ::fdopen(descriptors[0], "rb");
    CHECK(pipe);
    CHECK(::close(descriptors[1]) == 0);
    Fpos failed;
    std::memset(&failed, 0xa5, sizeof(failed));
    const Fpos unchanged = failed;
    CHECK(guest_position(pipe, &failed) == -1 && t_guest_errno == 29);
    CHECK(std::memcmp(&failed, &unchanged, sizeof(failed)) == 0);
    CHECK(std::fclose(pipe) == 0);
    CHECK(::pipe(descriptors) == 0);
    FILE* broken = opened_file(::fdopen(descriptors[0], "rb"), "rb");
    CHECK(broken && ::close(descriptors[0]) == 0 && ::close(descriptors[1]) == 0);
    t_guest_errno = 73;
    CHECK(stdio_scan(broken, "%d", &scanned) == EOF && std::ferror(broken) && t_guest_errno == 9);
    close_file(broken);
#endif
    std::puts("libc stdio tests passed");
}
