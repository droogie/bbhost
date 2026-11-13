// The connection test as the setup window runs it (net/nat_check.h has what
// it does): on a thread of its own, from the game's P2P port, while the game
// is not running - the setup window's Connection card starts it, and only
// there. One at a time; nothing waits for it. It never takes longer than the
// exchange's 6 s and the server's answer's 8 s, and closing the window ends
// it. When this PC refused every datagram, the server is not asked: nothing
// reached it.
#pragma once

#include "net/nat_check.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace net::nat {

struct JobParams {
    std::string host;                  // the test server: the game's STUN host
    std::vector<std::uint16_t> ports;  // its test ports (online.nat_test_ports)
    int p2p_port = 0;                  // the P2P port (online.p2p_port)
    std::string token;                 // the account's, "" when nobody is signed in: nothing is saved
    std::string save_url;              // "<auth server>/auth/nat_test"
};

class Job {
public:
    struct View {
        bool running = false;
        bool finished = false;  // `outcome` holds the last test's
        unsigned tests = 0;     // tests finished so far (a new one: this moved on)
        const char* step = "";  // what it is doing, while it runs
        Outcome outcome;
    };
    Job() = default;
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;
    ~Job();  // cancels a test still running and waits for it
    // False while one runs.
    bool start(const JobParams& p);
    View view() const;
    void cancel();

private:
    void run(JobParams p);
    mutable std::mutex mu_;
    std::thread thread_;
    std::atomic<bool> cancel_{false};
    View view_;
};

// The latest result this process knows - a test the setup window ran, or the
// account's last one as the sign-in check read it (whoami's NatTest) - for
// F10's line in the game.
struct Remembered {
    bool known = false;
    Verdict verdict = Verdict::None;
    Inbound inbound = Inbound::NotTested;
    double at = 0;       // Unix seconds
    bool saved = false;  // kept for the account
};
void remember(const Remembered& r);
Remembered remembered();

// The test server: online.stun_server's host, or the game server's
// (np_server_base) when it is unset or "off".
std::string test_host();

}  // namespace net::nat
