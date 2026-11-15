#include "host/plugin_menu.h"

#include "bbhost_plugin.h"

#include <algorithm>

namespace plugin_menu {

const Page* Layout::page(const std::string& key) const {
    for (const Page& p : pages) {
        if (p.key == key) return &p;
    }
    return nullptr;
}

Layout layout_of(const std::vector<PluginOptionDesc>& options) {
    Layout l;
    std::string menu;     // the BB_OPT_MENU heading the pages come under
    Page* page = nullptr;  // the page the rows go on
    for (const PluginOptionDesc& o : options) {
        if (o.type == BB_OPT_HEADING) {
            page = nullptr;
            if (!(o.flags & BB_OPT_PANEL) || o.key.empty()) continue;
            if (o.flags & BB_OPT_MENU) {
                menu = o.key;
                l.menus.push_back({o.key, o.def.empty() ? o.label : o.def, {}});
            }
            Page p;
            p.key = o.key;
            p.title = o.label;
            p.help = o.help;
            p.menu = menu;
            p.subpage = (o.flags & BB_OPT_SUBPAGE) != 0;
            l.pages.push_back(std::move(p));
            page = &l.pages.back();
            if (!page->subpage && !menu.empty()) l.menus.back().pages.push_back(o.key);
            continue;
        }
        if (page) page->rows.push_back({o, false});
    }
    return l;
}

std::vector<std::string> cycle_of(const Layout& l, const std::string& page) {
    const Page* p = l.page(page);
    if (!p) return {};
    if (p->subpage) return {page};
    std::vector<std::string> out;
    for (const Page& q : l.pages) {
        if (q.menu == p->menu && !q.subpage) out.push_back(q.key);
    }
    return out;
}

bool set_rows(const Layout& l, Runtime& rt, const std::string& page, const std::vector<PluginOptionDesc>& rows) {
    if (!l.page(page)) return false;
    const std::string prefix = page + "/";
    for (auto it = rt.values.begin(); it != rt.values.end();) {
        it = it->first.compare(0, prefix.size(), prefix) == 0 ? rt.values.erase(it) : std::next(it);
    }
    if (rows.empty()) {
        rt.rows.erase(page);
        return true;
    }
    std::vector<Row>& out = rt.rows[page];
    out.clear();
    for (const PluginOptionDesc& d : rows) {
        if (d.type == BB_OPT_HEADING || d.type == BB_OPT_END) continue;  // a page holds rows, not pages
        out.push_back({d, true});
    }
    return true;
}

bool set_value(Runtime& rt, const std::string& page, const std::string& key, const std::string& value) {
    const auto it = rt.rows.find(page);
    if (it == rt.rows.end()) return false;
    const bool has = std::any_of(it->second.begin(), it->second.end(), [&](const Row& r) { return r.desc.key == key; });
    if (!has) return false;
    rt.values[page + "/" + key] = value;
    return true;
}

const std::vector<Row>& rows_of(const Page& p, const Runtime& rt) {
    const auto it = rt.rows.find(p.key);
    return it != rt.rows.end() ? it->second : p.rows;
}

std::string value_of(const Runtime& rt, const std::string& page, const Row& row) {
    const auto it = rt.values.find(page + "/" + row.desc.key);
    return it != rt.values.end() ? it->second : row.desc.def;
}

std::string event_key(const std::string& page, const Row& row) { return row.runtime ? page + "/" + row.desc.key : row.desc.key; }

const std::string& Nav::shown() const {
    static const std::string none;
    if (!stack.empty()) return stack.back().page;
    if (cycle.empty()) return none;
    return cycle[static_cast<std::size_t>(at)];
}

bool Nav::turn(int d) {
    if (!stack.empty() || cycle.size() < 2 || d == 0) return false;
    const int n = static_cast<int>(cycle.size());
    at = ((at + d) % n + n) % n;
    return true;
}

void Nav::open(const std::string& page, int from_row) { stack.push_back({page, from_row}); }

int Nav::back() {
    if (stack.empty()) return -1;
    const int row = stack.back().from_row;
    stack.pop_back();
    return row;
}

std::string one_line(const std::string& help, std::size_t room) {
    // Where the first sentence ends.
    std::size_t end = help.size();
    for (std::size_t i = 0; i < help.size(); ++i) {
        const char c = help[i];
        if ((c == '.' || c == '!' || c == '?') && (i + 1 == help.size() || help[i + 1] == ' ' || help[i + 1] == '\n')) {
            end = i + 1;
            break;
        }
    }
    std::string first = help.substr(0, end);
    for (char& c : first) {
        if (c == '\n') c = ' ';
    }
    std::size_t chars = 0;
    for (const char c : first) chars += (static_cast<unsigned char>(c) & 0xc0) != 0x80;
    if (chars <= room) return first;
    // Too long: the words that fit with "..." after them.
    const std::size_t keep = room > 3 ? room - 3 : 0;
    std::size_t bytes = 0, n = 0;
    while (bytes < first.size() && n < keep) {
        ++bytes;
        while (bytes < first.size() && (static_cast<unsigned char>(first[bytes]) & 0xc0) == 0x80) ++bytes;
        ++n;
    }
    std::size_t cut = bytes;
    if (cut < first.size() && first[cut] != ' ') {
        const std::size_t space = first.rfind(' ', cut);
        if (space != std::string::npos && space > 0) cut = space;
    }
    while (cut > 0 && (first[cut - 1] == ' ' || first[cut - 1] == ',' || first[cut - 1] == ';' || first[cut - 1] == ':')) --cut;
    return first.substr(0, cut) + "...";
}

const char* Interruptions::step(const World& now) {
    const World before = was;
    was = now;
    const auto came = [&](std::uint32_t flag) { return (now.flags & flag) && !(before.flags & flag); };
    const auto dead = [](const World& w) { return (w.flags & BB_WORLD_IN) && w.hp == 0; };
    if (came(BB_WORLD_LOADING)) return "a loading screen";
    if (came(BB_WORLD_CUTSCENE)) return "a cutscene";
    if (dead(now) && !dead(before)) return "the character's death";
    return nullptr;
}

}  // namespace plugin_menu
