#include "engine/patch_manifest_file.h"

#include "core/config.h"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace patch_file {

namespace {

std::string unquote(const std::string& v) {
    if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') && v.back() == v.front()) return v.substr(1, v.size() - 2);
    return v;
}

// A TOML array of strings, ["a", "b"] (or a lone string).
std::vector<std::string> parse_string_list(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false;
    char q = 0;
    for (char c : text) {
        if (quoted) {
            if (c == q) {
                quoted = false;
                out.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        } else if (c == '"' || c == '\'') {
            quoted = true;
            q = c;
        }
    }
    if (out.empty() && !text.empty() && text.front() != '[') out.push_back(unquote(text));
    return out;
}

bool parse_hex_bytes(const std::string& text, std::vector<std::uint8_t>* out) {
    out->clear();
    std::string s;
    for (char c : text) {
        if (c == ' ' || c == ',' || c == '\t') continue;
        s.push_back(c);
    }
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
    if (s.empty() || s.size() % 2) return false;
    for (std::size_t i = 0; i < s.size(); i += 2) {
        char* end = nullptr;
        const std::string pair = s.substr(i, 2);
        const long v = std::strtol(pair.c_str(), &end, 16);
        if (!end || *end) return false;
        out->push_back(static_cast<std::uint8_t>(v));
    }
    return true;
}

}  // namespace

bool parse(const std::string& path, Manifest* m, std::string* err) {
    std::map<std::string, std::string> kv;
    if (!config_parse_toml(path, &kv, err)) return false;
    m->path = path;
    auto get = [&](const std::string& k) {
        auto it = kv.find(k);
        return it == kv.end() ? std::string() : unquote(it->second);
    };
    m->name = get("patch.name");
    m->title = get("patch.title");
    m->eboot = get("patch.eboot");
    m->description = get("patch.description");
    m->option = get("patch.option");
    m->group = get("patch.group");
    m->order = std::atoi(get("patch.order").c_str());
    // `default`: the name an earlier port of Kyo's list gave the same switch.
    std::string en = get("patch.enabled");
    if (en.empty()) en = get("patch.default");
    m->enabled = en.empty() || en == "true" || en == "1";
    const std::string gp = get("patch.gameplay");
    m->gameplay = gp == "true" || gp == "1";
    const std::string lv = get("patch.live");
    m->live = !(lv == "false" || lv == "0");
    if (m->name.empty() || m->eboot.empty()) {
        *err = "[patch] needs name and eboot";
        return false;
    }
    std::map<std::string, Site> sites;
    for (const auto& [k, v] : kv) {
        if (k.rfind("site.", 0) != 0) continue;
        const std::size_t dot = k.find('.', 5);
        if (dot == std::string::npos) continue;
        Site& s = sites[k.substr(5, dot - 5)];
        s.id = k.substr(5, dot - 5);
        const std::string field = k.substr(dot + 1), val = unquote(v);
        if (field == "address") {
            s.address = std::strtoull(val.c_str(), nullptr, 16);
        } else if (field == "expect") {
            if (!parse_hex_bytes(val, &s.expect)) {
                *err = "site " + s.id + ": bad expect";
                return false;
            }
        } else if (field == "bytes") {
            if (!parse_hex_bytes(val, &s.bytes)) {
                *err = "site " + s.id + ": bad bytes";
                return false;
            }
        } else if (field == "kind") {
            s.code = val != "data";
        } else if (field == "yields") {
            s.yields = parse_string_list(v);
        }
    }
    for (auto& [id, s] : sites) {
        if (!s.address || s.expect.empty() || s.bytes.size() != s.expect.size()) {
            *err = "site " + id + ": needs address, expect and bytes of the same length";
            return false;
        }
        m->sites.push_back(s);
    }
    if (m->sites.empty()) {
        *err = "no [site.*]";
        return false;
    }
    return true;
}

}  // namespace patch_file
