// The setup window's links out (host/links.h): the community's Discord invite
// - the one the website's about page gives as "Join the community on
// Discord", not its credits' other servers - and the website, both https;
// only those open; and a test's window (BBHOST_SETUP_TEST) never opens a
// browser.
#include "host/links.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::printf("FAIL: %s\n", what);
    }
}

}  // namespace

int main() {
    check(std::strcmp(links::kDiscord.url, "https://discord.gg/ZTrJ28aX7K") == 0, "the community's invite, from the about page");
    check(std::strcmp(links::kWebsite.url, "https://thehuntersdream.com") == 0, "the website");
    // the about page's credits link two other communities' servers: never these
    check(std::strcmp(links::kDiscord.url, "https://discord.gg/wzMynmW") != 0 &&
              std::strcmp(links::kDiscord.url, "https://discord.com/invite/Ax6YByr") != 0,
          "not another community's server");
    check(std::strcmp(links::kDiscord.label, "Discord") == 0 && std::strcmp(links::kDiscord.title, "Join the Discord") == 0,
          "the Discord button and its bubble's title");
    check(std::strcmp(links::kWebsite.label, "Website") == 0, "the Website button");
    for (const links::Link* l : links::kAll) {
        check(std::strncmp(l->url, "https://", 8) == 0, "every link https");
        check(std::strlen(l->tip) > 20 && std::string(l->tip).find("Opens ") == 0, "every bubble says what pressing it opens");
        check(links::known(l->url), "every link of the table opens");
    }
    check(std::size(links::kAll) == 2, "two links");
    check(!links::known("https://discord.gg/wzMynmW"), "another address does not open");
    check(!links::known("https://thehuntersdream.com/account"), "not even another of the website's pages");
    check(!links::known(nullptr) && !links::known(""), "nothing does not open");
    // a test's window logs the address instead of opening a browser
    check(links::opens_browser(nullptr), "no test: the browser opens");
    check(links::opens_browser(""), "an empty BBHOST_SETUP_TEST is no test");
    check(!links::opens_browser("links:/tmp/x"), "the links hook: logged, no browser");
    check(!links::opens_browser("shot:/tmp/x") && !links::opens_browser("natcard:C:\\bbtest\\shots") && !links::opens_browser("quit"),
          "any test's window: no browser");
    if (g_fail) {
        std::printf("launcher_links_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("launcher_links_test: all passed\n");
    return 0;
}
