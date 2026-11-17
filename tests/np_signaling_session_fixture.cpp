// Use the real peer store and dispatcher queue, without starting its worker.
#include "net/session.cpp"
#include "np_signaling_fixture.h"

namespace signaling_test {
void reset_dispatch() {
    net::g_drun = true;
    net::g_due.clear();
}

void dispatch_ready(int horizon_ms) {
    const auto until = net::clock::now() + std::chrono::milliseconds(horizon_ms);
    std::vector<std::function<void()>> ready;
    while (!net::g_due.empty() && net::g_due.front().at <= until) {
        ready.push_back(std::move(net::g_due.front().fn));
        net::g_due.pop_front();
    }
    for (auto& fn : ready) fn();
}
}
