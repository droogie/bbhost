// core/mac_vm.cpp: the guest's address windows on macOS.
#include "core/mac_vm.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

int g_bad = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("mac_vm_test: %s\n", what);
        ++g_bad;
    }
}

void* at(std::uint64_t a) { return reinterpret_cast<void*>(a); }
std::uint64_t addr(const void* p) { return reinterpret_cast<std::uint64_t>(p); }

constexpr int kAnon = MAP_PRIVATE | MAP_ANONYMOUS;
constexpr int kRw = PROT_READ | PROT_WRITE;

}  // namespace

int main() {
    // bbhost's high window: free in any new process.
    constexpr std::uint64_t kWin = kMacVmHighBase, kWinLen = 0x100000000ull;
    if (!mac_vm_reserve(kWin, kWinLen)) {
        std::printf("mac_vm_test: cannot reserve [0x%llx, +4 GiB)\n", static_cast<unsigned long long>(kWin));
        return 1;
    }
    expect(!mac_vm_reserve(kWin + 0x10000000, 0x1000), "a second reservation over the window succeeded");

    expect(mac_vm_in_window(kWin, kWinLen), "the whole window is in it");
    expect(mac_vm_in_window(kWin + 0x200000, 0x1000), "a page in the window is in it");
    expect(!mac_vm_in_window(kWin - 0x1000, 0x2000), "a range straddling the window's start is in it");
    expect(!mac_vm_in_window(kWin + kWinLen - 0x1000, 0x2000), "a range straddling the window's end is in it");
    expect(!mac_vm_in_window(kWin + 0x1000, ~0ull), "a range wrapping past the address space is in it");
    expect(!mac_vm_in_window(0x20000000000ull, 0x1000), "a page outside the windows is in one");

    const std::uint64_t a = kWin + 0x200000;
    void* p = mac_vm_map(at(a), 0x20000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace);
    expect(p == at(a), "NoReplace on placeholders");
    if (p == at(a)) std::memset(p, 0x5a, 0x20000);
    expect(!mac_vm_map(at(a + 0x10000), 0x1000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace),
           "NoReplace over a mapping succeeded");
    expect(!mac_vm_map(at(a - 0x1000), 0x2000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace),
           "NoReplace straddling a mapping succeeded");
    void* q = mac_vm_map(at(a + 0x10000), 0x1000, kRw, kAnon, -1, 0, MacVmPlace::Exact);
    expect(q == at(a + 0x10000) && static_cast<std::uint8_t*>(q)[0] == 0, "Exact over a mapping");
    expect(mac_vm_unmap(at(a + 0x8000), 0x8000), "unmapping part of a mapping");
    expect(mac_vm_map(at(a + 0x8000), 0x8000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace) == at(a + 0x8000),
           "NoReplace where a part was unmapped");
    expect(static_cast<std::uint8_t*>(p)[0x7fff] == 0x5a, "the rest of the mapping kept its contents");
    expect(mac_vm_unmap(at(a), 0x20000), "unmapping");
    expect(mac_vm_map(at(a), 0x20000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace) == at(a), "NoReplace after unmapping");

    // Direct memory is a file mapping.
    char name[32];
    std::snprintf(name, sizeof(name), "/bb-macvm-test-%d", static_cast<int>(getpid()));
    shm_unlink(name);
    const int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    shm_unlink(name);
    expect(fd >= 0 && ftruncate(fd, 0x40000) == 0, "a shared memory object");
    const std::uint64_t f = kWin + 0x1000000;
    auto* v1 = static_cast<std::uint8_t*>(mac_vm_map(at(f), 0x10000, kRw, MAP_SHARED, fd, 0x10000, MacVmPlace::NoReplace));
    auto* v2 = static_cast<std::uint8_t*>(mac_vm_map(at(f + 0x100000), 0x10000, kRw, MAP_SHARED, fd, 0x10000, MacVmPlace::NoReplace));
    expect(v1 == at(f) && v2 == at(f + 0x100000), "two views of one object at placeholders");
    if (v1 && v2) {
        v1[0x123] = 0x77;
        expect(v2[0x123] == 0x77, "the views share their pages");
    }

    // Stub pages near a range: above it first, within a rel32 of all of it.
    const std::uint64_t lo = a, hi = a + 0x20000;
    void* s = mac_vm_map_near(lo, hi, 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC);
    expect(s && addr(s) >= hi && addr(s) + 0x2000 - lo < 0x80000000ull, "a stub page above the range");
    // When nothing above is free, below it.
    const std::uint64_t top = kWin + kWinLen - 0x10000;
    expect(mac_vm_map(at(top), 0x10000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace) == at(top), "the window's last pages");
    void* b = mac_vm_map_near(top, top + 0x10000, 0x1000, kRw);
    expect(b && addr(b) + 0x1000 <= top && top + 0x10000 - addr(b) < 0x80000000ull, "a stub page below the range");
    expect(!mac_vm_map_near(lo, hi, 0x80000000ull, kRw), "a stub longer than a rel32 reaches");
    expect(!mac_vm_map_near(lo, hi, ~0ull, kRw), "a stub longer than the address space");
    expect(!mac_vm_map_near(0x20000000000ull, 0x20000001000ull, 0x1000, kRw), "a stub page out of the windows' reach");

    const std::uint64_t free_at = 0x20000000000ull;
    void* g = mac_vm_map(at(free_at), 0x4000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace);
    expect(g == at(free_at), "NoReplace on free space");
    expect(!mac_vm_map(at(free_at), 0x1000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace), "NoReplace over it again");
    expect(mac_vm_unmap(g, 0x4000), "unmapping outside the windows");
    expect(!mac_vm_map_near(free_at, free_at + 0x1000, 0x1000, kRw), "an unmapping outside the windows left placeholders");
    static char own[0x4000];
    const std::uint64_t own_page = (addr(own) + 0xfff) & ~0xfffull;
    expect(!mac_vm_map(at(own_page), 0x1000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace), "NoReplace over the process's data");

    // As the image's reservation is adopted.
    void* r = mmap(nullptr, 0x100000, kRw, kAnon, -1, 0);
    expect(r != MAP_FAILED && mac_vm_adopt(addr(r), addr(r) + 0x100000), "adopting");
    expect(mac_vm_map(static_cast<std::uint8_t*>(r) + 0x4000, 0x1000, kRw, kAnon, -1, 0, MacVmPlace::NoReplace) ==
               static_cast<std::uint8_t*>(r) + 0x4000,
           "NoReplace in an adopted window");

    std::printf("mac_vm_test: %d wrong\n", g_bad);
    return g_bad ? 1 : 0;
}
