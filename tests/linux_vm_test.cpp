// The guest windows on Linux (core/linux_vm.h): a window is reserved
// PROT_NONE; a fixed mapping in it takes its part of the reservation, one
// asked not to replace fails where an earlier mapping made there is (alone
// and with threads racing for the same range), an unmapping gives the part
// back as reservation - never a hole - and the pieces merge again; the parts
// of a failed map stay reserved; nothing the kernel places (a hint or none)
// lands in a window; a window over something mapped is refused and its free
// parts can be had; outside the windows the calls are the plain ones. Each
// case works in a window of its own at an address the kernel had free.
#include "core/linux_vm.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

constexpr std::uintptr_t kMiB = 1u << 20;
constexpr std::size_t kPage = 4096;
constexpr int kAnon = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
constexpr int kRW = PROT_READ | PROT_WRITE;

struct Line {
    std::uintptr_t lo = 0, hi = 0;
    std::string perms;
};

std::vector<Line> maps() {
    std::vector<Line> out;
    FILE* f = std::fopen("/proc/self/maps", "r");
    if (!f) return out;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), f)) {
        unsigned long long a = 0, b = 0;
        char perms[8] = {};
        if (std::sscanf(buf, "%llx-%llx %7s", &a, &b, perms) == 3) out.push_back({a, b, perms});
    }
    std::fclose(f);
    return out;
}

// /proc/self/maps' permissions for the byte at `a` ("---p", "rw-p", "rw-s"), "" where nothing is mapped.
std::string perms_at(std::uintptr_t a) {
    for (const Line& l : maps()) {
        if (a >= l.lo && a < l.hi) return l.perms;
    }
    return "";
}

// How many of /proc/self/maps' lines overlap [lo, hi).
int lines_in(std::uintptr_t lo, std::uintptr_t hi) {
    int n = 0;
    for (const Line& l : maps()) n += l.lo < hi && l.hi > lo;
    return n;
}

// `len` bytes the kernel has free, 2 MiB aligned.
std::uintptr_t free_range(std::size_t len) {
    void* p = mmap(nullptr, len + 2 * kMiB, PROT_NONE, kAnon, -1, 0);
    if (p == MAP_FAILED) return 0;
    munmap(p, len + 2 * kMiB);
    return (reinterpret_cast<std::uintptr_t>(p) + 2 * kMiB - 1) & ~(2 * kMiB - 1);
}

void* at(std::uintptr_t a) { return reinterpret_cast<void*>(a); }

void* map_anon(std::uintptr_t a, std::size_t len, LinuxVmFix fix) { return linux_vm_mmap(at(a), len, kRW, kAnon, -1, 0, fix); }

void test_reserve() {
    const std::uintptr_t base = free_range(72 * kMiB);
    CHECK(base != 0);
    const std::uint64_t before = linux_vm_reserved_bytes();
    CHECK(linux_vm_reserve(base, 64 * kMiB));
    CHECK(linux_vm_reserved_bytes() == before + 64 * kMiB);
    CHECK(perms_at(base) == "---p");
    CHECK(perms_at(base + 64 * kMiB - 1) == "---p");
    CHECK(perms_at(base + 64 * kMiB) == "");
    CHECK(linux_vm_owns(base, 64 * kMiB));
    CHECK(linux_vm_owns(base + 4 * kMiB, 4096));
    CHECK(!linux_vm_owns(base, 64 * kMiB + 4096));
    CHECK(!linux_vm_owns(base - 4096, 8192));
    // Again, or over part of it: ours already.
    errno = 0;
    CHECK(!linux_vm_reserve(base, 64 * kMiB) && errno == EEXIST);
    CHECK(!linux_vm_reserve(base + 60 * kMiB, 8 * kMiB));
    CHECK(perms_at(base + 64 * kMiB) == "");  // nothing reserved past its end by the refusal
    CHECK(linux_vm_reserved_bytes() == before + 64 * kMiB);
}

void test_map_in_window() {
    const std::uintptr_t base = free_range(16 * kMiB);
    CHECK(linux_vm_reserve(base, 16 * kMiB));
    const std::uint64_t mapped = linux_vm_mapped_bytes();
    // No-replace on the reservation: there, read/write.
    void* p = map_anon(base + kMiB, 2 * kMiB, LinuxVmFix::NoReplace);
    CHECK(p == at(base + kMiB));
    CHECK(perms_at(base + kMiB) == "rw-p");
    CHECK(perms_at(base) == "---p");
    CHECK(perms_at(base + 3 * kMiB) == "---p");
    std::memset(at(base + kMiB), 0x5a, 2 * kMiB);
    // No-replace over it: refused, and what it holds kept.
    errno = 0;
    CHECK(map_anon(base + 2 * kMiB, 2 * kMiB, LinuxVmFix::NoReplace) == MAP_FAILED && errno == EEXIST);
    CHECK(*static_cast<volatile unsigned char*>(at(base + 2 * kMiB)) == 0x5a);
    CHECK(perms_at(base + 3 * kMiB) == "---p");
    // Beside it: fine.
    CHECK(map_anon(base + 3 * kMiB, kMiB, LinuxVmFix::NoReplace) == at(base + 3 * kMiB));
    // Exact over the first: replaced (fresh memory).
    CHECK(map_anon(base + kMiB, kMiB, LinuxVmFix::Exact) == at(base + kMiB));
    CHECK(*static_cast<volatile unsigned char*>(at(base + kMiB)) == 0);
    CHECK(*static_cast<volatile unsigned char*>(at(base + 2 * kMiB)) == 0x5a);
    CHECK(linux_vm_mapped_bytes() == mapped + 3 * kMiB);
    // A hint is never placed in a window.
    void* h = linux_vm_mmap(at(base + 8 * kMiB), kMiB, kRW, kAnon, -1, 0, LinuxVmFix::None);
    CHECK(h != MAP_FAILED && h != at(base + 8 * kMiB) && !linux_vm_owns(reinterpret_cast<std::uintptr_t>(h), kMiB));
    if (h != MAP_FAILED) linux_vm_munmap(h, kMiB);
    CHECK(perms_at(base + 8 * kMiB) == "---p");
    // Nor a mapping without one, however many are made beside the windows
    // (what places memory "anywhere": a placed direct mapping's room, the
    // mirror, the libraries) - plain mmap, as those calls are.
    std::vector<void*> anywhere;
    for (int i = 0; i < 64; ++i) {
        void* q = mmap(nullptr, 3 * kMiB, PROT_NONE, kAnon, -1, 0);
        CHECK(q != MAP_FAILED);
        if (q == MAP_FAILED) break;
        const auto a = reinterpret_cast<std::uintptr_t>(q);
        CHECK(!linux_vm_owns(a, 4096) && !linux_vm_owns(a + 3 * kMiB - 4096, 4096));
        CHECK(a + 3 * kMiB <= base || a >= base + 16 * kMiB);
        anywhere.push_back(q);
    }
    for (void* q : anywhere) munmap(q, 3 * kMiB);
    CHECK(perms_at(base + 8 * kMiB) == "---p");

    // Unmapped: the reservation again, never a hole, and mappable once more.
    CHECK(linux_vm_munmap(at(base + 2 * kMiB), kMiB) == 0);
    CHECK(perms_at(base + 2 * kMiB) == "---p");
    CHECK(perms_at(base + kMiB) == "rw-p");
    CHECK(perms_at(base + 3 * kMiB) == "rw-p");
    CHECK(linux_vm_mapped_bytes() == mapped + 2 * kMiB);
    CHECK(map_anon(base + 2 * kMiB, kMiB, LinuxVmFix::NoReplace) == at(base + 2 * kMiB));
    // Taken apart piece by piece, the window is one reservation again.
    CHECK(linux_vm_munmap(at(base + 2 * kMiB), kMiB) == 0);
    CHECK(linux_vm_munmap(at(base + kMiB), kMiB) == 0);
    CHECK(linux_vm_munmap(at(base + 3 * kMiB), kMiB) == 0);
    CHECK(linux_vm_mapped_bytes() == mapped);
    CHECK(perms_at(base + kMiB) == "---p" && perms_at(base + 3 * kMiB) == "---p");
    CHECK(lines_in(base, base + 16 * kMiB) == 1);
}

void test_shared() {
    const std::uintptr_t base = free_range(16 * kMiB);
    CHECK(linux_vm_reserve(base, 16 * kMiB));
    const long fd = syscall(SYS_memfd_create, "linux-vm-test", 1 /* MFD_CLOEXEC */);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(ftruncate(static_cast<int>(fd), 8 * kMiB) == 0);
    auto* mirror = static_cast<unsigned char*>(mmap(nullptr, 8 * kMiB, kRW, MAP_SHARED, static_cast<int>(fd), 0));
    CHECK(mirror != MAP_FAILED);
    // Direct memory at its identity address: 4 MiB into the backing.
    void* p = linux_vm_mmap(at(base + 4 * kMiB), 2 * kMiB, kRW, MAP_SHARED, static_cast<int>(fd), 4 * kMiB, LinuxVmFix::NoReplace);
    CHECK(p == at(base + 4 * kMiB));
    CHECK(perms_at(base + 4 * kMiB) == "rw-s");
    static_cast<unsigned char*>(p)[0x1234] = 0xa7;
    CHECK(mirror[4 * kMiB + 0x1234] == 0xa7);
    // The same memory at the same address again (a second mapping of it): refused, as the kernel refused it.
    errno = 0;
    CHECK(linux_vm_mmap(at(base + 4 * kMiB), kMiB, PROT_READ, MAP_SHARED, static_cast<int>(fd), 4 * kMiB, LinuxVmFix::NoReplace) ==
              MAP_FAILED &&
          errno == EEXIST);
    // A partial unmap of it: the middle is reservation, the rest still the memory.
    CHECK(linux_vm_munmap(at(base + 5 * kMiB), 4096) == 0);
    CHECK(perms_at(base + 5 * kMiB) == "---p");
    CHECK(perms_at(base + 5 * kMiB + 4096) == "rw-s");
    CHECK(static_cast<unsigned char*>(p)[0x1234] == 0xa7);
    CHECK(linux_vm_munmap(p, 2 * kMiB) == 0);
    CHECK(perms_at(base + 4 * kMiB) == "---p" && perms_at(base + 6 * kMiB - 1) == "---p");
    CHECK(mirror[4 * kMiB + 0x1234] == 0xa7);  // the memory itself stays
    CHECK(lines_in(base, base + 16 * kMiB) == 1);
    munmap(mirror, 8 * kMiB);
    close(static_cast<int>(fd));
}

void test_outside() {
    // Outside every window the plain calls: no-replace is MAP_FIXED_NOREPLACE,
    // an unmapping leaves nothing.
    const std::uintptr_t base = free_range(8 * kMiB);
    CHECK(map_anon(base, 2 * kMiB, LinuxVmFix::NoReplace) == at(base));
    CHECK(map_anon(base + kMiB, 2 * kMiB, LinuxVmFix::NoReplace) == MAP_FAILED);
    CHECK(map_anon(base + kMiB, 2 * kMiB, LinuxVmFix::Exact) == at(base + kMiB));
    CHECK(linux_vm_munmap(at(base), 3 * kMiB) == 0);
    CHECK(perms_at(base) == "" && perms_at(base + 2 * kMiB) == "");
}

void test_edges() {
    // A window with free space after it.
    const std::uintptr_t base = free_range(24 * kMiB);
    CHECK(linux_vm_reserve(base, 16 * kMiB));
    const std::uintptr_t end = base + 16 * kMiB;
    // Exact across the end: the window's part counts as mapped.
    CHECK(map_anon(end - kMiB, 2 * kMiB, LinuxVmFix::Exact) == at(end - kMiB));
    CHECK(perms_at(end - kMiB) == "rw-p" && perms_at(end) == "rw-p");
    CHECK(map_anon(end - kMiB, 4096, LinuxVmFix::NoReplace) == MAP_FAILED);
    // Unmapped across the end: the window's part reservation, the rest nothing.
    CHECK(linux_vm_munmap(at(end - kMiB), 2 * kMiB) == 0);
    CHECK(perms_at(end - kMiB) == "---p" && perms_at(end - 4096) == "---p");
    CHECK(perms_at(end) == "" && perms_at(end + kMiB - 4096) == "");
    // A no-replace across the edge finds the reservation there: refused.
    CHECK(map_anon(end - kMiB, 2 * kMiB, LinuxVmFix::NoReplace) == MAP_FAILED);
    CHECK(perms_at(end) == "");
    // A map that fails in a window leaves the reservation (no hole).
    const std::uint64_t mapped = linux_vm_mapped_bytes();
    errno = 0;
    CHECK(linux_vm_mmap(at(base + 4 * kMiB), kMiB, kRW, MAP_SHARED, -1, 0, LinuxVmFix::NoReplace) == MAP_FAILED);
    CHECK(errno == EBADF);
    CHECK(perms_at(base + 4 * kMiB) == "---p");
    CHECK(linux_vm_mapped_bytes() == mapped);
    CHECK(lines_in(base, end) == 1);
}

void test_race() {
    // Threads racing a no-replace for one range: exactly one gets it, each round.
    const std::uintptr_t base = free_range(8 * kMiB);
    CHECK(linux_vm_reserve(base, 8 * kMiB));
    constexpr int kThreads = 8;
    for (int round = 0; round < 50; ++round) {
        std::atomic<int> ready{0}, won{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < kThreads; ++t) {
            ts.emplace_back([&, t] {
                ready.fetch_add(1);
                while (ready.load() < kThreads) {
                }
                // Overlapping ranges, each 1 MiB, all over base + 2 MiB.
                const std::uintptr_t a = base + 2 * kMiB - (t % 2) * (kMiB / 2);
                if (map_anon(a, kMiB, LinuxVmFix::NoReplace) == at(a)) won.fetch_add(1);
            });
        }
        for (auto& t : ts) t.join();
        CHECK(won.load() == 1);
        CHECK(linux_vm_munmap(at(base + kMiB), 3 * kMiB) == 0);
    }
    CHECK(lines_in(base, base + 8 * kMiB) == 1);
}

void test_taken() {
    // Something mapped in the middle of a range: the window refused whole
    // (nothing reserved), the occupant named, and the free parts can be had.
    const std::uintptr_t base = free_range(16 * kMiB);
    void* f = mmap(at(base + 6 * kMiB), 2 * kMiB, kRW, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    CHECK(f == at(base + 6 * kMiB));
    std::memset(f, 0x33, 2 * kMiB);
    errno = 0;
    CHECK(!linux_vm_reserve(base, 16 * kMiB) && errno == EEXIST);
    CHECK(perms_at(base) == "" && perms_at(base + 15 * kMiB) == "");
    const std::vector<std::string> who = linux_vm_occupants(base, 16 * kMiB);
    CHECK(who.size() == 1 && who[0].find("rw-p") != std::string::npos);
    CHECK(linux_vm_reserve_free(base, 16 * kMiB) == 14 * kMiB);
    CHECK(perms_at(base) == "---p" && perms_at(base + 6 * kMiB - 1) == "---p");
    CHECK(perms_at(base + 6 * kMiB) == "rw-p" && *static_cast<volatile unsigned char*>(f) == 0x33);
    CHECK(perms_at(base + 8 * kMiB) == "---p" && perms_at(base + 16 * kMiB - 1) == "---p");
    CHECK(linux_vm_owns(base, 6 * kMiB) && linux_vm_owns(base + 8 * kMiB, 8 * kMiB));
    CHECK(!linux_vm_owns(base, 16 * kMiB) && !linux_vm_owns(base + 6 * kMiB, 4096));
    // A mapping in a free part is as in any window.
    CHECK(map_anon(base + 8 * kMiB, kMiB, LinuxVmFix::NoReplace) == at(base + 8 * kMiB));
    CHECK(linux_vm_munmap(at(base + 8 * kMiB), kMiB) == 0 && perms_at(base + 8 * kMiB) == "---p");
    munmap(f, 2 * kMiB);
}

// Pages of [p, p + len) in memory now.
std::size_t resident(void* p, std::size_t len) {
    std::vector<unsigned char> v(len / kPage);
    if (mincore(p, len, v.data()) != 0) {
        return 0;
    }
    std::size_t n = 0;
    for (unsigned char b : v) {
        n += b & 1;
    }
    return n;
}

// A memfd's fresh pages come in; a PROT_NONE range fails.
void test_populate() {
    const std::size_t len = 8 * kMiB;
    const long fd = syscall(SYS_memfd_create, "linux_vm_test", 0);
    CHECK(fd >= 0 && ftruncate(static_cast<int>(fd), static_cast<off_t>(len)) == 0);
    void* p = mmap(nullptr, len, kRW, MAP_SHARED, static_cast<int>(fd), 0);
    CHECK(p != MAP_FAILED);
    CHECK(resident(p, len) == 0);
    CHECK(linux_vm_populate(p, len));
    CHECK(resident(p, len) == len / kPage);
    munmap(p, len);
    close(static_cast<int>(fd));

    void* none = mmap(nullptr, len, PROT_NONE, kAnon, -1, 0);
    CHECK(!linux_vm_populate(none, len));
    munmap(none, len);
}

}  // namespace

int main() {
    test_reserve();
    test_map_in_window();
    test_shared();
    test_outside();
    test_edges();
    test_race();
    test_taken();
    test_populate();
    if (g_fail) {
        std::printf("linux_vm_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("linux_vm_test: ok\n");
    return 0;
}
