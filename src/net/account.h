// The player's account on the private server. There are no
// passwords: the options screen links this PC to an account the player signs
// into on the website with Discord (a device code, as TVs do it), or makes an
// account held by this PC alone, with a one-time recovery code. Either way
// the server hands back a token, kept in bbhost-options.toml. With a token,
// net::online_id() is the account's name, and every request to the server
// carries the token in the X-BB-Token header (net/http.cpp for the host's own
// calls, hle/http.cpp for the game's), so the server binds the traffic to the
// account whatever a config file says.
#pragma once

#include <string>

namespace net {

// Both empty when nobody is signed in. Safe from any thread.
std::string account_name();
std::string account_token();
bool account_logged_in();
void account_set(const std::string& name, const std::string& token);
void account_clear();

// The server refused this PC's sign-in (HTTP 401 on its surfaces, or the
// sign-in check below): the token expired or was revoked, or, with no token,
// the online id is an account's name. Kept until the next account_set /
// account_clear or a check the server answers yes; the event poller slows
// down while it holds and the F10 Account section says what to do.
// account_mark_refused logs the reason once per sign-in.
bool account_refused();
void account_mark_refused(const std::string& reason);

// The setup window's check of this PC's sign-in with the server (POST
// /auth/whoami to auth_server_base(), read by net/sign_in_check.h), on a
// thread of its own: nothing waits for it. A refusal marks the sign-in
// refused (above); a yes clears an earlier refusal. An answer that comes back
// after the sign-in changed, or after a newer check began, is dropped. The
// token goes in the request's body and header only; nothing logs it.
enum class AccountCheckState {
    None,       // no check since the sign-in last changed (or nobody is signed in)
    Checking,   // asked, no answer yet
    Valid,      // the server knows the sign-in
    Refused,    // it does not: expired, or this PC was signed out
    Unchecked,  // no verdict: no answer, a busy server, one without the check
};
struct AccountCheck {
    AccountCheckState state = AccountCheckState::None;
    std::string name;  // the account checked
    std::string why;   // Refused: the server's words; Unchecked: what came back instead
    long status = 0;   // the answer's HTTP status, 0 when none came
};
void account_check_start(int timeout_ms = 5000);
AccountCheck account_check();

// All of these talk to the server (blocking, host thread) and fill `error`
// with the server's message when they fail.

// Linking with the website: start gets a code for the player to approve at
// `verification_uri` (signed in with Discord); poll every `interval` seconds
// until it is no longer "pending". On "approved" the account is set.
struct DeviceLink {
    std::string device_code;       // secret, stays here
    std::string user_code;         // what the player types, XXXX-XXXX
    std::string verification_uri;  // the website's /link page
    int interval = 5;
    int expires_in = 600;
};
bool account_link_start(DeviceLink& out, std::string& error);
// "pending", "slow_down", "approved", "denied" or "expired"; "" on an error.
std::string account_link_poll(const DeviceLink& link, std::string& error);

// An account held by this PC: name it, get a token and a recovery code the
// player must write down (shown once; never logged or saved here).
bool account_create(const std::string& name, std::string& recovery_code, std::string& error);
// A lost or signed-out PC: the name and its recovery code give a new token and
// a new recovery code; every other token of the account stops working.
bool account_recover(const std::string& name, const std::string& code, std::string& new_code, std::string& error);
// A one-time code to sign in on the website with this PC's account.
bool account_web_code(std::string& code, std::string& error);

// Tells the server to forget the token and clears it here.
void account_logout();

// "bbhost on <host name>": what the website's /link page shows the player
// before they approve, so a code phished from another machine stands out.
std::string account_device_name();

}  // namespace net
