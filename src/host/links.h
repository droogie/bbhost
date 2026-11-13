// The setup window's links out (host/launcher.cpp, the sidebar's two buttons
// over Play): the community's Discord and its website, opened in the player's
// browser. The addresses live here alone (tests/launcher_links_test.cpp).
#pragma once

#include <cstring>

namespace links {

struct Link {
    const char* label;  // the button's
    const char* title;  // its bubble's title: what pressing it does
    const char* tip;    // ... and where it goes
    const char* url;
};

// The community's invite: the website's about page,
// https://thehuntersdream.com/about, links it as "Join the community on
// Discord". The page's two other Discord links (in its credits) are other
// communities' servers, not this one.
inline constexpr Link kDiscord = {
    "Discord",
    "Join the Discord",
    "Opens the community's Discord invite in your browser: news, help with bbhost, and other hunters to play with.",
    "https://discord.gg/ZTrJ28aX7K",
};

// The website: the community server's own pages - the world map, the
// leaderboards, your account and its connection test.
inline constexpr Link kWebsite = {
    "Website",
    "Website",
    "Opens thehuntersdream.com in your browser: the world map, the leaderboards, and your account.",
    "https://thehuntersdream.com",
};

inline constexpr const Link* kAll[] = {&kDiscord, &kWebsite};

// What pressing one does: open its address in the browser - or, in a test's
// window (BBHOST_SETUP_TEST set), only say which address it would open, so a
// test never starts a browser.
inline bool opens_browser(const char* setup_test) { return !setup_test || !*setup_test; }

// Only the addresses above open from the window.
inline bool known(const char* url) {
    for (const Link* l : kAll) {
        if (url && std::strcmp(url, l->url) == 0) return true;
    }
    return false;
}

}  // namespace links
