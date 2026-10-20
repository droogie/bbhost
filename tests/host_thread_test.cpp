// core/host_thread.h on macOS: an 8 MiB stack (std::thread's is 512 KiB),
// otherwise std::thread's behaviour.
#include "core/host_thread.h"

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>

namespace {

int g_bad = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("host_thread_test: %s\n", what);
        ++g_bad;
    }
}

// Uses most of a 512 KiB stack's worth of frames: crashes on a small one.
__attribute__((noinline)) int deep(int n) {
    volatile char pad[4096];
    pad[0] = static_cast<char>(n);
    return n ? deep(n - 1) + (pad[0] == static_cast<char>(n)) : 0;
}

}  // namespace

int main() {
    std::size_t stack = 0;
    std::string got;
    int depth = -1;
    HostThread t([&](const std::string& s, int n) {
        stack = pthread_get_stacksize_np(pthread_self());
        got = s;
        depth = deep(n);
    }, std::string("arg"), 1000);  // ~4 MiB of frames
    expect(t.joinable(), "joinable after start");
    t.join();
    expect(!t.joinable(), "not joinable after join");
    expect(stack >= HostThread::kStack, "stack under 8 MiB");
    expect(got == "arg", "argument not passed");
    expect(depth == 1000, "deep recursion did not finish");

    // stop_event_thread (hle/avplayer.cpp) compares ids from inside the thread.
    HostThread self_check;
    std::atomic<int> same{-1};
    std::atomic<bool> go{false};
    self_check = HostThread([&] {
        while (!go.load()) {
        }
        same.store(self_check.get_id() == std::this_thread::get_id());
    });
    go.store(true);
    self_check.join();
    expect(same.load() == 1, "get_id differs from the thread's own id");

    std::atomic<bool> ran{false};
    HostThread d([&] { ran.store(true); });
    d.detach();
    expect(!d.joinable(), "joinable after detach");
    for (int i = 0; i < 1000 && !ran.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    expect(ran.load(), "detached thread did not run");

    if (!g_bad) std::printf("host_thread_test: ok\n");
    return g_bad ? 1 : 0;
}
