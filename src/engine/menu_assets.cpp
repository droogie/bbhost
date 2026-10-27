#include "engine/menu_assets.h"

#include "hle/fs.h"
#include "log.h"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

// Two edits. The movie: each PC section is a clone of the dump's
// ControllSetting sprite under a new instance name and character id, with its
// title retitled to the port's message id, its static text dropped, its
// widget rows kept, dropped or added to fit the section, and its placement
// put on the main timeline hidden (the game shows the section it opens); then
// the System list gets the movie's own ScrollBarV, since nine rows scroll in
// five lines. The messages: the port's ids added to the menu text (group 200)
// and line help (201) FMGs of the dump's DCX-compressed BND4.

namespace {

using Bytes = std::vector<std::uint8_t>;

// Bumped whenever a generator or the recipe changes, so installs remake their
// assets instead of keeping the last generation.
constexpr int kGeneratorVersion = 13;  // 12: UTF-16 text end fix; 11: PC Graphics' seventh row, HLSL Shaders; 10: PC Camera back to four rows (the camera tuning row hidden); 9: PC Camera's fifth row, the camera tuning; 8: PC Enhancements' three rows; 7: the PC Enhancements section; 6: event text (group 30) for the altar's rebirth

const char kMessagesTsv[] =
#include "pc_option_messages.inc"
    ;

struct Fail {
    std::string why;
};

std::uint16_t u16(const Bytes& b, std::size_t i) {
    if (i + 2 > b.size()) throw Fail{"truncated"};
    return static_cast<std::uint16_t>(b[i] | (b[i + 1] << 8));
}
std::uint32_t u32(const Bytes& b, std::size_t i) {
    if (i + 4 > b.size()) throw Fail{"truncated"};
    return static_cast<std::uint32_t>(b[i]) | (static_cast<std::uint32_t>(b[i + 1]) << 8) |
           (static_cast<std::uint32_t>(b[i + 2]) << 16) | (static_cast<std::uint32_t>(b[i + 3]) << 24);
}
std::uint64_t u64(const Bytes& b, std::size_t i) { return u32(b, i) | (static_cast<std::uint64_t>(u32(b, i + 4)) << 32); }
std::uint32_t be32(const Bytes& b, std::size_t i) {
    if (i + 4 > b.size()) throw Fail{"truncated"};
    return (static_cast<std::uint32_t>(b[i]) << 24) | (static_cast<std::uint32_t>(b[i + 1]) << 16) |
           (static_cast<std::uint32_t>(b[i + 2]) << 8) | b[i + 3];
}
void put16(Bytes& b, std::size_t i, std::uint32_t v) {
    b[i] = static_cast<std::uint8_t>(v);
    b[i + 1] = static_cast<std::uint8_t>(v >> 8);
}
void put32(Bytes& b, std::size_t i, std::uint32_t v) {
    for (int k = 0; k < 4; ++k) b[i + k] = static_cast<std::uint8_t>(v >> (8 * k));
}
void put64(Bytes& b, std::size_t i, std::uint64_t v) {
    put32(b, i, static_cast<std::uint32_t>(v));
    put32(b, i + 4, static_cast<std::uint32_t>(v >> 32));
}
void putbe32(Bytes& b, std::size_t i, std::uint32_t v) {
    for (int k = 0; k < 4; ++k) b[i + k] = static_cast<std::uint8_t>(v >> (24 - 8 * k));
}
void app16(Bytes& b, std::uint32_t v) {
    b.push_back(static_cast<std::uint8_t>(v));
    b.push_back(static_cast<std::uint8_t>(v >> 8));
}
void app(Bytes& b, const Bytes& src, std::size_t from, std::size_t to) { b.insert(b.end(), src.begin() + from, src.begin() + to); }
void app(Bytes& b, const Bytes& src) { b.insert(b.end(), src.begin(), src.end()); }
void app(Bytes& b, const std::string& s) { b.insert(b.end(), s.begin(), s.end()); }

// ---------------------------------------------------------------- the movie

constexpr int kEnd = 0, kShowFrame = 1, kPlace2 = 26, kDefineSprite = 39, kPlace3 = 70, kSymbolClass = 76;

struct Tag {
    int code;
    std::size_t head, s, e;
};

std::vector<Tag> tags(const Bytes& b, std::size_t i, std::size_t end) {
    std::vector<Tag> out;
    while (i < end) {
        const std::uint16_t th = u16(b, i);
        const std::size_t head = i;
        i += 2;
        const int code = th >> 6;
        std::size_t length = th & 0x3f;
        if (length == 0x3f) {
            length = u32(b, i);
            i += 4;
        }
        if (i + length > b.size()) throw Fail{"a tag runs past the end"};
        out.push_back({code, head, i, i + length});
        if (code == kEnd) break;
        i += length;
    }
    return out;
}

std::size_t body_start(const Bytes& b) {
    const int nbits = b.at(8) >> 3;
    return 8 + (5 + 4 * nbits + 7) / 8 + 4;
}

Bytes tag_bytes(int code, const Bytes& body) {
    Bytes out;
    if (body.size() < 0x3f) {
        app16(out, static_cast<std::uint32_t>((code << 6) | body.size()));
    } else {
        app16(out, static_cast<std::uint32_t>((code << 6) | 0x3f));
        out.resize(out.size() + 4);
        put32(out, out.size() - 4, static_cast<std::uint32_t>(body.size()));
    }
    app(out, body);
    return out;
}

std::string cstr(const Bytes& b, std::size_t i, std::size_t* next) {
    std::size_t j = i;
    while (j < b.size() && b[j]) ++j;
    if (j >= b.size()) throw Fail{"an unterminated name"};
    *next = j + 1;
    return std::string(b.begin() + i, b.begin() + j);
}

struct BitR {
    const Bytes& b;
    std::size_t i;
    int bit = 0;
    std::uint32_t u(int n) {
        std::uint32_t v = 0;
        for (int k = 0; k < n; ++k) {
            v = (v << 1) | ((b.at(i) >> (7 - bit)) & 1);
            if (++bit == 8) {
                bit = 0;
                ++i;
            }
        }
        return v;
    }
    std::int32_t s(int n) {
        const std::uint32_t v = u(n);
        if (n && ((v >> (n - 1)) & 1)) return static_cast<std::int32_t>(static_cast<std::int64_t>(v) - (std::int64_t{1} << n));
        return static_cast<std::int32_t>(v);
    }
    std::size_t align() {
        if (bit) {
            bit = 0;
            ++i;
        }
        return i;
    }
};

struct BitW {
    std::vector<int> bits;
    void u(std::uint32_t v, int n) {
        for (int k = n - 1; k >= 0; --k) bits.push_back((v >> k) & 1);
    }
    Bytes out() {
        while (bits.size() % 8) bits.push_back(0);
        Bytes r;
        for (std::size_t i = 0; i < bits.size(); i += 8) {
            int v = 0;
            for (int k = 0; k < 8; ++k) v = (v << 1) | bits[i + k];
            r.push_back(static_cast<std::uint8_t>(v));
        }
        return r;
    }
};

int sbits(std::int64_t v) {
    int n = 1;
    while (!(-(std::int64_t{1} << (n - 1)) <= v && v < (std::int64_t{1} << (n - 1)))) ++n;
    return n;
}

// A matrix as the section clone keeps it: the scale and rotation
// fields exactly as read, the translation as numbers.
struct Matrix {
    bool has_scale = false, has_rot = false;
    int sn = 0, rn = 0;
    std::uint32_t sa = 0, sc = 0, ra = 0, rc = 0;
    std::int64_t tx = 0, ty = 0;
};

Matrix read_matrix(const Bytes& b, std::size_t i, std::size_t* end) {
    BitR r{b, i};
    Matrix m;
    if (r.u(1)) {
        m.has_scale = true;
        m.sn = static_cast<int>(r.u(5));
        m.sa = r.u(m.sn);
        m.sc = r.u(m.sn);
    }
    if (r.u(1)) {
        m.has_rot = true;
        m.rn = static_cast<int>(r.u(5));
        m.ra = r.u(m.rn);
        m.rc = r.u(m.rn);
    }
    const int n = static_cast<int>(r.u(5));
    m.tx = r.s(n);
    m.ty = r.s(n);
    *end = r.align();
    return m;
}

Bytes write_matrix(const Matrix& m) {
    BitW w;
    if (m.has_scale) {
        w.u(1, 1);
        w.u(static_cast<std::uint32_t>(m.sn), 5);
        w.u(m.sa, m.sn);
        w.u(m.sc, m.sn);
    } else {
        w.u(0, 1);
    }
    if (m.has_rot) {
        w.u(1, 1);
        w.u(static_cast<std::uint32_t>(m.rn), 5);
        w.u(m.ra, m.rn);
        w.u(m.rc, m.rn);
    } else {
        w.u(0, 1);
    }
    const int n = std::max(sbits(m.tx), sbits(m.ty));
    const std::uint64_t mask = (std::uint64_t{1} << n) - 1;
    w.u(static_cast<std::uint32_t>(n), 5);
    w.u(static_cast<std::uint32_t>(static_cast<std::uint64_t>(m.tx) & mask), n);
    w.u(static_cast<std::uint32_t>(static_cast<std::uint64_t>(m.ty) & mask), n);
    return w.out();
}

std::size_t skip_cxform(const Bytes& b, std::size_t i) {
    BitR r{b, i};
    const std::uint32_t has_add = r.u(1), has_mul = r.u(1);
    const int n = static_cast<int>(r.u(4));
    if (has_mul) r.u(n * 4);
    if (has_add) r.u(n * 4);
    return r.align();
}

struct Place2 {
    std::uint8_t flags = 0;
    std::uint16_t depth = 0;
    std::optional<std::uint16_t> cid;
    std::optional<Matrix> mat;
    std::optional<std::string> name;
    Bytes mid, tail;
};

Place2 parse_place2(const Bytes& b, std::size_t s, std::size_t e) {
    Place2 p;
    std::size_t i = s;
    p.flags = b.at(i);
    p.depth = u16(b, i + 1);
    i += 3;
    if (p.flags & 0x02) {
        p.cid = u16(b, i);
        i += 2;
    }
    std::size_t mat_end = i;
    if (p.flags & 0x04) {
        p.mat = read_matrix(b, i, &mat_end);
        i = mat_end;
    }
    if (p.flags & 0x08) i = skip_cxform(b, i);
    if (p.flags & 0x10) i += 2;
    const std::size_t name_start = i;
    if (p.flags & 0x20) p.name = cstr(b, i, &i);
    p.mid.assign(b.begin() + mat_end, b.begin() + name_start);
    p.tail.assign(b.begin() + i, b.begin() + e);
    return p;
}

Bytes build_place2(const Place2& p, std::uint16_t depth, const std::optional<Matrix>& mat, const std::optional<std::string>& name) {
    std::uint8_t flags = p.flags;
    if (!name) flags &= static_cast<std::uint8_t>(~0x20);
    else flags |= 0x20;
    Bytes out;
    out.push_back(flags);
    app16(out, depth);
    if (flags & 0x02) app16(out, p.cid.value_or(0));
    if (flags & 0x04) {
        if (!mat) throw Fail{"a placement with a matrix flag and no matrix"};
        app(out, write_matrix(*mat));
    }
    app(out, p.mid);
    if (flags & 0x20) {
        app(out, *name);
        out.push_back(0);
    }
    app(out, p.tail);
    return out;
}

Bytes build_place3_hidden(std::uint16_t cid, std::uint16_t depth, const Matrix& mat, const std::string& name) {
    Bytes out;
    out.push_back(0x02 | 0x04 | 0x20);
    out.push_back(0x20);
    app16(out, depth);
    app16(out, cid);
    app(out, write_matrix(mat));
    app(out, name);
    out.push_back(0);
    out.push_back(0);  // visible = false
    return out;
}

std::map<std::uint16_t, std::string> symbol_classes(const Bytes& b) {
    std::map<std::uint16_t, std::string> out;
    for (const Tag& t : tags(b, body_start(b), b.size())) {
        if (t.code != kSymbolClass) continue;
        const std::uint16_t n = u16(b, t.s);
        std::size_t i = t.s + 2;
        for (int k = 0; k < n; ++k) {
            const std::uint16_t cid = u16(b, i);
            out[cid] = cstr(b, i + 2, &i);
        }
    }
    return out;
}

std::optional<Bytes> add_symbol_class(const Bytes& b, std::uint16_t cid, const std::string& name) {
    for (const Tag& t : tags(b, body_start(b), b.size())) {
        if (t.code != kSymbolClass) continue;
        const std::uint16_t n = u16(b, t.s);
        Bytes body(b.begin() + t.s, b.begin() + t.e);
        put16(body, 0, n + 1u);
        app16(body, cid);
        app(body, name);
        body.push_back(0);
        Bytes out(b.begin(), b.begin() + t.head);
        app(out, tag_bytes(kSymbolClass, body));
        app(out, b, t.e, b.size());
        return out;
    }
    return std::nullopt;
}

std::optional<std::uint16_t> place_char(const Bytes& b, std::size_t ts, int code) {
    if (code != kPlace2 && code != kPlace3) return std::nullopt;
    const std::size_t off = code == kPlace3 ? 4 : 3;
    if (!(b.at(ts) & 0x02)) return std::nullopt;
    return u16(b, ts + off);
}

struct Sprite {
    std::size_t head, s, e;
};

std::optional<Sprite> find_sprite(const Bytes& b, std::uint16_t cid) {
    for (const Tag& t : tags(b, body_start(b), b.size())) {
        if (t.code == kDefineSprite && u16(b, t.s) == cid) return Sprite{t.head, t.s, t.e};
    }
    return std::nullopt;
}

std::set<std::string> names_in_sprite(const Bytes& b, std::size_t cs, std::size_t ce) {
    std::set<std::string> out;
    for (const Tag& t : tags(b, cs + 4, ce)) {
        if (t.code == kPlace2) {
            const Place2 p = parse_place2(b, t.s, t.e);
            if (p.name && !p.name->empty()) out.insert(*p.name);
        }
    }
    return out;
}

std::uint16_t max_character_id(const Bytes& b) {
    static const std::set<int> kDefines = {2,  6,  20, 21, 22, 32, 33, 34, 36, 37, 39, 46, 48, 60, 75, 83, 84,
                                           1001, 1003, 1005, 1006, 1007, 1008, 1009, 1010};
    std::uint16_t best = 0;
    for (const Tag& t : tags(b, body_start(b), b.size())) {
        if (kDefines.count(t.code) && t.e - t.s >= 2) best = std::max(best, u16(b, t.s));
    }
    return best;
}

struct SectionArgs {
    std::string clone = "ControllSetting", name, top = "Top";
    int section_depth = 383;
    std::optional<int> title_id;
    std::vector<std::string> drop = {"Item_6_0"};
    std::optional<int> defaults_slot;
    int add_rows = 0;
};

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string& s, const char* p) {
    const std::size_t n = std::strlen(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

// One PC section cloned from a donor section into the movie.
Bytes add_option_section(const Bytes& b, const SectionArgs& a) {
    if (b.size() < 16 || std::memcmp(b.data(), "GFX", 3) != 0) throw Fail{"optionsetting.gfx is not an uncompressed GFX"};
    const auto names = symbol_classes(b);
    struct Main {
        Place2 p;
        std::size_t head, s, e;
    };
    std::map<std::string, Main> places;
    for (const Tag& t : tags(b, body_start(b), b.size())) {
        if (t.code != kPlace2) continue;
        Place2 p = parse_place2(b, t.s, t.e);
        if (p.name && !p.name->empty()) places[*p.name] = Main{p, t.head, t.s, t.e};
    }
    for (const std::string& want : {a.clone, a.top}) {
        if (!places.count(want)) throw Fail{"no main-timeline instance named " + want};
    }
    if (places.count(a.name)) throw Fail{"already has an instance named " + a.name};

    const Place2& clone_p = places[a.clone].p;
    const std::uint16_t new_cid = static_cast<std::uint16_t>(max_character_id(b) + 1);
    const auto clone_sprite = find_sprite(b, clone_p.cid.value_or(0));
    if (!clone_sprite) throw Fail{"no DefineSprite for " + a.clone};
    const std::size_t cs = clone_sprite->s, ce = clone_sprite->e;

    std::optional<std::uint16_t> title_cid, title_old;
    Bytes title_sprite_body;
    if (a.title_id) {
        for (const Tag& t : tags(b, cs + 4, ce)) {
            const auto cid = place_char(b, t.s, t.code);
            if (!cid) continue;
            const auto child = find_sprite(b, *cid);
            if (!child) continue;
            for (const Tag& t2 : tags(b, child->s + 4, child->e)) {
                if (t2.code != kPlace2) continue;
                const Place2 q = parse_place2(b, t2.s, t2.e);
                if (!q.name || !starts_with(*q.name, "StaticText_")) continue;
                const std::string old_name = *q.name;
                const std::string new_name = "StaticText_" + std::to_string(*a.title_id);
                if (new_name.size() != old_name.size()) throw Fail{"the title id is not the same length as " + old_name};
                Bytes body(b.begin() + child->s, b.begin() + child->e);
                put16(body, 0, new_cid + 1u);
                auto hit = std::search(body.begin(), body.end(), old_name.begin(), old_name.end());
                if (hit == body.end()) throw Fail{"cannot find " + old_name + " in the title sprite"};
                std::copy(new_name.begin(), new_name.end(), hit);
                title_cid = static_cast<std::uint16_t>(new_cid + 1);
                title_sprite_body = std::move(body);
                title_old = *cid;
                break;
            }
            if (title_cid) break;
        }
        if (!title_cid) throw Fail{"no title StaticText found in the clone"};
    }

    std::set<std::string> drop_names(a.drop.begin(), a.drop.end());

    std::optional<std::uint16_t> label_depth;
    std::int64_t label_dy = 0;
    std::optional<std::string> label_name;
    if (a.defaults_slot) {
        drop_names.erase("Item_6_0");
        std::map<std::string, std::int64_t> ys;
        for (const Tag& t : tags(b, cs + 4, ce)) {
            if (t.code != kPlace2) continue;
            const Place2 p2 = parse_place2(b, t.s, t.e);
            if (p2.name && (*p2.name == "Item_0_0" || *p2.name == "Item_1_0") && p2.mat) ys[*p2.name] = p2.mat->ty;
            if (p2.name && *p2.name == "Item_6_0") label_depth = p2.depth;
        }
        if (!label_depth || ys.size() != 2) throw Fail{"the donor has no Item_6_0 label slot, or no two rows"};
        label_dy = (ys["Item_1_0"] - ys["Item_0_0"]) * (*a.defaults_slot - 6);
        label_name = "Item_" + std::to_string(*a.defaults_slot) + "_0";
        std::set<std::string> rows = names_in_sprite(b, cs, ce);
        rows.erase("Item_6_0");
        if (rows.count(*label_name) && !drop_names.count(*label_name)) throw Fail{*label_name + " is a row; drop it"};
    }

    struct Added {
        std::uint16_t depth;
        std::string name;
        std::int64_t dy;
    };
    std::vector<Added> add;
    std::optional<std::uint16_t> last_depth;
    if (a.add_rows) {
        std::map<int, Place2> rows;
        std::set<int> depths;
        for (const Tag& t : tags(b, cs + 4, ce)) {
            if (t.code == kPlace2) {
                Place2 p2 = parse_place2(b, t.s, t.e);
                depths.insert(p2.depth);
                if (p2.name && starts_with(*p2.name, "Item_") && ends_with(*p2.name, "_0") && p2.mat) {
                    rows[std::atoi(p2.name->substr(5, p2.name->size() - 7).c_str())] = p2;
                }
            } else if (t.code == kPlace3) {
                depths.insert(u16(b, t.s + 2));
            }
        }
        if (!rows.count(0)) throw Fail{"--add-rows: no Item_0_0"};
        std::vector<int> kept;
        for (const auto& [k, p2] : rows) {
            if (!drop_names.count(*p2.name) && p2.cid == rows[0].cid) kept.push_back(k);
        }
        if (kept.size() < 2) throw Fail{"--add-rows needs two widget rows to take a pitch from"};
        const Place2& last = rows[kept.back()];
        const Place2& prev = rows[kept[kept.size() - 2]];
        const std::int64_t pitch = last.mat->ty - prev.mat->ty;
        last_depth = last.depth;
        int d = last.depth + 1;
        for (int k = 1; k <= a.add_rows; ++k) {
            while (depths.count(d)) ++d;
            const std::string name = "Item_" + std::to_string(kept.back() + k) + "_0";
            add.push_back({static_cast<std::uint16_t>(d), name, pitch * k});
            depths.insert(d);
        }
    }

    auto added_rows = [&](const Place2& p2) {
        Bytes out;
        for (const Added& r : add) {
            std::optional<Matrix> m = p2.mat;
            if (m) m->ty += r.dy;
            app(out, tag_bytes(kPlace2, build_place2(p2, r.depth, m, p2.name ? std::optional<std::string>(r.name) : std::nullopt)));
        }
        return out;
    };

    Bytes new_body;
    app16(new_body, new_cid);
    app(new_body, b, cs + 2, cs + 4);
    for (const Tag& t : tags(b, cs + 4, ce)) {
        if (t.code == kPlace2) {
            const Place2 p2 = parse_place2(b, t.s, t.e);
            const bool named = p2.name && !p2.name->empty();
            if (named && starts_with(*p2.name, "StaticText")) continue;
            if (named && drop_names.count(*p2.name)) continue;
            if (!add.empty() && last_depth && p2.depth == *last_depth) {
                app(new_body, b, t.head, t.e);
                app(new_body, added_rows(p2));
                continue;
            }
            if (label_depth && p2.depth == *label_depth && (label_dy || named)) {
                std::optional<Matrix> m = p2.mat;
                if (m) m->ty += label_dy;
                app(new_body, tag_bytes(kPlace2, build_place2(p2, p2.depth, m, named ? label_name : std::nullopt)));
                continue;
            }
        }
        if (title_cid && place_char(b, t.s, t.code) == title_old) {
            Bytes body(b.begin() + t.s, b.begin() + t.e);
            put16(body, t.code == kPlace3 ? 4 : 3, *title_cid);
            app(new_body, tag_bytes(t.code, body));
            continue;
        }
        app(new_body, b, t.head, t.e);
    }

    if (!clone_p.mat) throw Fail{a.clone + " is placed without a matrix"};
    const Bytes new_section = tag_bytes(kPlace3, build_place3_hidden(new_cid, static_cast<std::uint16_t>(a.section_depth), *clone_p.mat, a.name));

    const std::size_t top_s = places[a.top].s;
    Bytes out(b.begin(), b.begin() + body_start(b));
    for (const Tag& t : tags(b, body_start(b), b.size())) {
        app(out, b, t.head, t.e);
        if (t.code == kDefineSprite && t.s == cs) {
            if (title_cid) app(out, tag_bytes(kDefineSprite, title_sprite_body));
            app(out, tag_bytes(kDefineSprite, new_body));
        } else if (t.code == kPlace2 && t.s == top_s) {
            app(out, new_section);
        }
    }
    if (auto it = names.find(clone_p.cid.value_or(0)); it != names.end()) {
        if (auto with_class = add_symbol_class(out, new_cid, it->second)) out = std::move(*with_class);
    }
    put32(out, 4, static_cast<std::uint32_t>(out.size()));
    return out;
}

// The scroll bar: a placement's (depth, character, matrix, name) with
// PlaceObject3's class name skipped.
struct Placement {
    std::uint16_t depth = 0;
    std::optional<std::uint16_t> cid;
    bool has_mat = false;
    double sx = 1.0, sy = 1.0, r0 = 0.0, r1 = 0.0;
    std::int64_t tx = 0, ty = 0;
    std::optional<std::string> name;
};

Placement placement(const Bytes& b, int code, std::size_t s) {
    Placement p;
    const std::uint8_t flags = b.at(s);
    std::size_t i = s + 1;
    std::uint8_t flags3 = 0;
    if (code == kPlace3) flags3 = b.at(i++);
    p.depth = u16(b, i);
    i += 2;
    if (code == kPlace3 && ((flags3 & 0x08) || ((flags3 & 0x10) && (flags & 0x02)))) cstr(b, i, &i);
    if (flags & 0x02) {
        p.cid = u16(b, i);
        i += 2;
    }
    if (flags & 0x04) {
        BitR r{b, i};
        p.has_mat = true;
        if (r.u(1)) {
            const int n = static_cast<int>(r.u(5));
            p.sx = r.s(n) / 65536.0;
            p.sy = r.s(n) / 65536.0;
        }
        if (r.u(1)) {
            const int n = static_cast<int>(r.u(5));
            p.r0 = r.s(n) / 65536.0;
            p.r1 = r.s(n) / 65536.0;
        }
        const int n = static_cast<int>(r.u(5));
        p.tx = r.s(n);
        p.ty = r.s(n);
        i = r.align();
    }
    if (flags & 0x08) i = skip_cxform(b, i);
    if (flags & 0x10) i += 2;
    if (flags & 0x20) p.name = cstr(b, i, &i);
    return p;
}

Bytes translate_matrix(std::int64_t tx, std::int64_t ty) {
    BitW w;
    const int n = std::max(sbits(tx), sbits(ty));
    w.u(0, 2);
    w.u(static_cast<std::uint32_t>(n), 5);
    const std::uint64_t mask = (std::uint64_t{1} << n) - 1;
    w.u(static_cast<std::uint32_t>(static_cast<std::uint64_t>(tx) & mask), n);
    w.u(static_cast<std::uint32_t>(static_cast<std::uint64_t>(ty) & mask), n);
    return w.out();
}

Bytes add_scrollbar(const Bytes& b, const std::string& list, double ax, double ay, const std::string& name = "ScrollBarV") {
    std::map<std::uint16_t, Sprite> sp;
    std::vector<std::uint16_t> order;  // Python's dict keeps the file's order for the donor search
    for (const Tag& t : tags(b, body_start(b), b.size())) {
        if (t.code != kDefineSprite) continue;
        const std::uint16_t cid = u16(b, t.s);
        if (!sp.count(cid)) order.push_back(cid);
        sp[cid] = Sprite{t.head, t.s, t.e};
    }
    std::optional<Placement> target;
    for (const Tag& t : tags(b, body_start(b), b.size())) {
        if (t.code != kPlace2 && t.code != kPlace3) continue;
        Placement p = placement(b, t.code, t.s);
        if (p.name && *p.name == list) {
            target = p;
            break;
        }
    }
    if (!target || !target->cid || !sp.count(*target->cid)) throw Fail{"no main-timeline sprite named " + list};
    if (target->r0 != 0.0 || target->r1 != 0.0) throw Fail{list + " is rotated"};
    const double sx = target->has_mat ? target->sx : 1.0, sy = target->has_mat ? target->sy : 1.0;
    const double ltx = target->has_mat ? static_cast<double>(target->tx) : 0.0;
    const double lty = target->has_mat ? static_cast<double>(target->ty) : 0.0;

    std::optional<std::uint16_t> donor;
    for (std::uint16_t cid : order) {
        const Sprite& s = sp[cid];
        for (const Tag& t : tags(b, s.s + 4, s.e)) {
            if (t.code != kPlace2 && t.code != kPlace3) continue;
            const Placement p = placement(b, t.code, t.s);
            if (p.name && *p.name == name && p.cid) {
                donor = p.cid;
                break;
            }
        }
        if (donor) break;
    }
    if (!donor) throw Fail{"no " + name + " anywhere to reuse"};

    const Sprite& ls = sp[*target->cid];
    std::vector<int> depths;
    std::optional<std::size_t> first_show;
    for (const Tag& t : tags(b, ls.s + 4, ls.e)) {
        if (t.code == kPlace2 || t.code == kPlace3) {
            const Placement p = placement(b, t.code, t.s);
            depths.push_back(p.depth);
            if (p.name && *p.name == name) throw Fail{list + " already has a " + name};
        }
        if (t.code == kShowFrame && !first_show) first_show = t.head;
    }
    if (!first_show) throw Fail{list + " has no frames"};
    int depth = 0;
    for (int d : depths) depth = std::max(depth, d);
    depth += 1;
    // Python's round() rounds half to even, as nearbyint does in the default mode.
    const auto tx = static_cast<std::int64_t>(std::nearbyint((ax * 20 - ltx) / sx));
    const auto ty = static_cast<std::int64_t>(std::nearbyint((ay * 20 - lty) / sy));
    Bytes place;
    place.push_back(0x02 | 0x04 | 0x20);
    app16(place, static_cast<std::uint32_t>(depth));
    app16(place, *donor);
    app(place, translate_matrix(tx, ty));
    app(place, name);
    place.push_back(0);
    Bytes new_sprite(b.begin() + ls.s, b.begin() + *first_show);
    app(new_sprite, tag_bytes(kPlace2, place));
    app(new_sprite, b, *first_show, ls.e);
    Bytes out(b.begin(), b.begin() + ls.head);
    app(out, tag_bytes(kDefineSprite, new_sprite));
    app(out, b, ls.e, b.size());
    put32(out, 4, static_cast<std::uint32_t>(out.size()));
    return out;
}

// One more line in a command list: the sprite placed as `list` (anywhere)
// gets an Item_<n>_0 placed like its last one, one pitch further down, at a
// free depth. The menu counts a list's lines from its Item_*_0 children, so
// the list then has room for another row. (The title's CommandList, sprite
// 32 of title.gfx: Item_0_0..Item_4_0 at depths 21..1, 40 px apart.)
Bytes add_list_slot(const Bytes& b, const std::string& list) {
    std::map<std::uint16_t, Sprite> sp;
    for (const Tag& t : tags(b, body_start(b), b.size()))
        if (t.code == kDefineSprite) sp[u16(b, t.s)] = Sprite{t.head, t.s, t.e};
    std::optional<std::uint16_t> list_cid;
    for (const auto& [cid, spr] : sp) {
        for (const Tag& t : tags(b, spr.s + 4, spr.e)) {
            if (t.code != kPlace2 && t.code != kPlace3) continue;
            const Placement p = placement(b, t.code, t.s);
            if (p.name && *p.name == list && p.cid) list_cid = p.cid;
        }
        if (list_cid) break;
    }
    if (!list_cid || !sp.count(*list_cid)) throw Fail{"no sprite placed as " + list};
    const Sprite& ls = sp[*list_cid];
    std::map<int, Place2> rows;
    std::set<int> depths;
    std::optional<std::size_t> first_show;
    for (const Tag& t : tags(b, ls.s + 4, ls.e)) {
        if (t.code == kPlace2) {
            Place2 p2 = parse_place2(b, t.s, t.e);
            depths.insert(p2.depth);
            if (p2.name && starts_with(*p2.name, "Item_") && ends_with(*p2.name, "_0") && p2.mat)
                rows[std::atoi(p2.name->substr(5, p2.name->size() - 7).c_str())] = p2;
        } else if (t.code == kPlace3) {
            depths.insert(u16(b, t.s + 2));
        }
        if (t.code == kShowFrame && !first_show) first_show = t.head;
    }
    if (rows.size() < 2 || !first_show) throw Fail{list + " has fewer than two Item_*_0 rows"};
    const Place2& last = rows.rbegin()->second;
    const Place2& prev = std::next(rows.rbegin())->second;
    std::optional<Matrix> m = last.mat;
    m->ty += last.mat->ty - prev.mat->ty;
    int depth = *depths.rbegin() + 1;
    const std::string name = "Item_" + std::to_string(rows.rbegin()->first + 1) + "_0";
    Bytes new_sprite(b.begin() + ls.s, b.begin() + *first_show);
    app(new_sprite, tag_bytes(kPlace2, build_place2(last, static_cast<std::uint16_t>(depth), m, name)));
    app(new_sprite, b, *first_show, ls.e);
    Bytes out(b.begin(), b.begin() + ls.head);
    app(out, tag_bytes(kDefineSprite, new_sprite));
    app(out, b, ls.e, b.size());
    put32(out, 4, static_cast<std::uint32_t>(out.size()));
    return out;
}

// The movie: eight sections, then the System list's scroll bar. PCDeck is
// opened on a Steam Deck only (engine/option_menu.cpp), but is always made.
Bytes build_option_movie(const Bytes& src) {
    Bytes m = src;
    auto section = [&](const char* name, int title, int depth, std::optional<int> defaults, int add_rows,
                       std::vector<std::string> drops) {
        SectionArgs a;
        a.name = name;
        a.title_id = title;
        a.section_depth = depth;
        a.defaults_slot = defaults;
        a.add_rows = add_rows;
        for (auto& d : drops) a.drop.push_back(d);
        m = add_option_section(m, a);
    };
    section("PCSetting", 116020, 383, 6, 0, {});
    section("PCGraphics", 117020, 384, 7, 1, {});  // seven rows: one added, the Defaults label below it
    section("PCControls", 118020, 385, 6, 0, {});
    section("PCKeys", 119020, 386, std::nullopt, 2, {});
    section("PCEffects", 120020, 387, 6, 0, {});
    section("PCCamera", 121020, 388, 4, 0, {"Item_4_0", "Item_5_0"});
    section("PCDeck", 123020, 389, 2, 0, {"Item_2_0", "Item_3_0", "Item_4_0", "Item_5_0"});
    section("PCEnhance", 124020, 390, 3, 0, {"Item_3_0", "Item_4_0", "Item_5_0"});
    return add_scrollbar(m, "Top", 1290, 354.5);
}

// ---------------------------------------------------------------- the messages

constexpr std::uint32_t kMenuText = 200, kLineHelp = 0xc9;
// Dialog text: ids from 920000 go to group 78 (MenuMan's generic dialog,
// sub_2022750) and group 204 (the options screens' own questions, sub_1f24850)
// instead of the menu groups.
constexpr std::uint32_t kDialogText = 78, kScreenDialogText = 0xcc;
constexpr std::int32_t kFirstDialogId = 920000, kLastDialogId = 929999;
bool dialog_id(std::int32_t id) { return id >= kFirstDialogId && id <= kLastDialogId; }
// Event text: ids 14000332..14000399 go to group 30 (イベントテキスト), the talk
// menus' entries and the messages their scripts show (engine/rebirth_script.h).
constexpr std::uint32_t kEventText = 30;
constexpr std::int32_t kFirstEventId = 14000332, kLastEventId = 14000399;
bool event_id(std::int32_t id) { return id >= kFirstEventId && id <= kLastEventId; }
int message_kind(std::int32_t id) { return event_id(id) ? 2 : dialog_id(id) ? 1 : 0; }

Bytes inflate_all(const Bytes& in, std::size_t from) {
    z_stream z{};
    if (inflateInit(&z) != Z_OK) throw Fail{"inflateInit"};
    Bytes out;
    std::uint8_t buf[1 << 16];
    z.next_in = const_cast<Bytef*>(in.data() + from);
    z.avail_in = static_cast<uInt>(in.size() - from);
    int r;
    do {
        z.next_out = buf;
        z.avail_out = sizeof(buf);
        r = inflate(&z, Z_NO_FLUSH);
        if (r != Z_OK && r != Z_STREAM_END) {
            inflateEnd(&z);
            throw Fail{"the message bundle does not inflate"};
        }
        out.insert(out.end(), buf, buf + (sizeof(buf) - z.avail_out));
    } while (r != Z_STREAM_END);
    inflateEnd(&z);
    return out;
}

std::size_t find(const Bytes& b, const char* what, std::size_t n, std::size_t from = 0) {
    auto it = std::search(b.begin() + from, b.end(), what, what + n);
    return it == b.end() ? std::string::npos : static_cast<std::size_t>(it - b.begin());
}

// End of a UTF-16LE string: the first 0 code unit at an even offset from `from`.
// A byte search for "\0\0" also matches across two characters (an ASCII one followed
// by one whose low byte is 0, such as U+AC00 or U+C800), which cut Korean text short.
std::size_t utf16_end(const Bytes& b, std::size_t from, std::size_t limit) {
    std::size_t i = from;
    while (i + 1 < limit && !(b[i] == 0 && b[i + 1] == 0)) i += 2;
    return i + 1 < limit ? i : std::string::npos;
}

struct BndFile {
    std::uint64_t flags;
    std::uint32_t id;
    Bytes name;  // as stored, without its terminator
    Bytes data;
};

struct Bnd4 {
    Bytes head;
    std::uint64_t esize = 0;
    bool unicode = false;
    std::vector<BndFile> files;
};

Bnd4 bnd_read(const Bytes& b) {
    if (b.size() < 0x40 || std::memcmp(b.data(), "BND4", 4) != 0) throw Fail{"not a BND4"};
    Bnd4 r;
    r.head.assign(b.begin(), b.begin() + 0x40);
    r.esize = u64(b, 0x20);
    const std::uint32_t n = u32(b, 0x0c);
    r.unicode = b[0x30] != 0;
    for (std::uint32_t k = 0; k < n; ++k) {
        const std::size_t e = 0x40 + k * r.esize;
        BndFile f;
        f.flags = u64(b, e);
        const std::uint64_t size = u64(b, e + 8);
        const std::uint32_t off = u32(b, e + 0x18);
        f.id = u32(b, e + 0x1c);
        const std::uint32_t name_off = u32(b, e + 0x20);
        if (r.unicode) {
            const std::size_t end = std::min<std::size_t>(b.size(), name_off + 4096);
            std::size_t stop = utf16_end(b, name_off, end);
            if (stop == std::string::npos) stop = end - ((end - name_off) % 2);
            f.name.assign(b.begin() + name_off, b.begin() + stop);
        } else {
            std::size_t z = name_off;
            while (z < b.size() && b[z]) ++z;
            f.name.assign(b.begin() + name_off, b.begin() + z);
        }
        if (off + size > b.size()) throw Fail{"a bundle entry runs past the end"};
        f.data.assign(b.begin() + off, b.begin() + off + size);
        r.files.push_back(std::move(f));
    }
    return r;
}

Bytes bnd_pack(const Bnd4& r) {
    Bytes names;
    std::vector<std::uint32_t> name_off;
    const std::size_t base = 0x40 + r.files.size() * r.esize;
    for (const BndFile& f : r.files) {
        name_off.push_back(static_cast<std::uint32_t>(base + names.size()));
        app(names, f.name);
        names.push_back(0);
        if (r.unicode) names.push_back(0);
    }
    const std::size_t data_start = base + names.size();
    Bytes blobs;
    std::vector<std::uint32_t> offs;
    for (const BndFile& f : r.files) {
        blobs.resize(blobs.size() + ((0x10 - blobs.size() % 0x10) % 0x10), 0);
        offs.push_back(static_cast<std::uint32_t>(data_start + blobs.size()));
        app(blobs, f.data);
    }
    Bytes out = r.head;
    put64(out, 0x28, data_start);
    for (std::size_t k = 0; k < r.files.size(); ++k) {
        Bytes e(r.esize, 0);
        put64(e, 0, r.files[k].flags);
        put64(e, 8, r.files[k].data.size());
        put64(e, 16, r.files[k].data.size());
        put32(e, 0x18, offs[k]);
        put32(e, 0x1c, r.files[k].id);
        put32(e, 0x20, name_off[k]);
        app(out, e);
    }
    app(out, names);
    app(out, blobs);
    return out;
}

using Fmg = std::map<std::int32_t, std::optional<Bytes>>;  // id -> UTF-16LE text, nullopt for an id with none

Fmg fmg_read(const Bytes& b) {
    if (b.size() < 0x28 || b[2] != 2) throw Fail{"an FMG that is not version 2"};
    const std::uint32_t gcount = u32(b, 0x0c), scount = u32(b, 0x10);
    const std::uint64_t soff = u64(b, 0x18);
    Fmg ids;
    for (std::uint32_t g = 0; g < gcount; ++g) {
        const auto idx = static_cast<std::int32_t>(u32(b, 0x28 + g * 16));
        const auto first = static_cast<std::int32_t>(u32(b, 0x28 + g * 16 + 4));
        const auto last = static_cast<std::int32_t>(u32(b, 0x28 + g * 16 + 8));
        for (std::int32_t mid = first, k = 0; mid <= last; ++mid, ++k) {
            if (static_cast<std::uint32_t>(idx + k) >= scount) throw Fail{"an FMG group past its strings"};
            const std::uint64_t o = u64(b, soff + static_cast<std::uint64_t>(idx + k) * 8);
            if (!o) {
                ids[mid] = std::nullopt;
                continue;
            }
            const std::size_t e = utf16_end(b, o, b.size());
            if (e == std::string::npos) throw Fail{"an unterminated FMG string"};
            ids[mid] = Bytes(b.begin() + o, b.begin() + e);
        }
    }
    return ids;
}

Bytes fmg_write(const Fmg& ids) {
    std::vector<std::int32_t> order;
    for (const auto& [k, v] : ids) order.push_back(k);
    struct Group {
        std::size_t idx;
        std::int32_t first, last;
    };
    std::vector<Group> groups;
    std::size_t run_start = 0;
    for (std::size_t i = 1; i <= order.size(); ++i) {
        if (i == order.size() || order[i] != order[i - 1] + 1) {
            groups.push_back({run_start, order[run_start], order[i - 1]});
            run_start = i;
        }
    }
    std::map<Bytes, std::size_t> strings;
    Bytes pool;
    for (std::int32_t mid : order) {
        const auto& t = ids.at(mid);
        if (!t || strings.count(*t)) continue;
        strings[*t] = pool.size();
        app(pool, *t);
        pool.push_back(0);
        pool.push_back(0);
    }
    const std::size_t head = 0x28 + groups.size() * 16, soff = head, body = soff + order.size() * 8;
    Bytes out(body + pool.size(), 0);
    out[2] = 2;
    put32(out, 0x08, 1);
    put32(out, 0x0c, static_cast<std::uint32_t>(groups.size()));
    put32(out, 0x10, static_cast<std::uint32_t>(order.size()));
    put32(out, 0x14, 0xff);
    put64(out, 0x18, soff);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        put32(out, 0x28 + g * 16, static_cast<std::uint32_t>(groups[g].idx));
        put32(out, 0x28 + g * 16 + 4, static_cast<std::uint32_t>(groups[g].first));
        put32(out, 0x28 + g * 16 + 8, static_cast<std::uint32_t>(groups[g].last));
    }
    for (std::size_t i = 0; i < order.size(); ++i) {
        const auto& t = ids.at(order[i]);
        put64(out, soff + i * 8, t ? body + strings[*t] : 0);
    }
    std::copy(pool.begin(), pool.end(), out.begin() + body);
    put32(out, 0x04, static_cast<std::uint32_t>(out.size()));
    return out;
}

Bytes utf16(const std::string& s) {
    Bytes out;
    for (std::size_t i = 0; i < s.size();) {
        std::uint32_t c = static_cast<std::uint8_t>(s[i]);
        int extra = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : c >= 0xc0 ? 1 : 0;
        c &= extra == 3 ? 0x07 : extra == 2 ? 0x0f : extra == 1 ? 0x1f : 0x7f;
        ++i;
        for (int k = 0; k < extra && i < s.size(); ++k, ++i) c = (c << 6) | (static_cast<std::uint8_t>(s[i]) & 0x3f);
        if (c >= 0x10000) {
            c -= 0x10000;
            app16(out, 0xd800 + (c >> 10));
            app16(out, 0xdc00 + (c & 0x3ff));
        } else {
            app16(out, c);
        }
    }
    return out;
}

struct Message {
    std::string text, help;
};

std::map<std::int32_t, Message> read_messages() {
    std::map<std::int32_t, Message> rows;
    std::string all(kMessagesTsv);
    std::size_t at = 0;
    while (at < all.size()) {
        std::size_t nl = all.find('\n', at);
        if (nl == std::string::npos) nl = all.size();
        const std::string line = all.substr(at, nl - at);
        at = nl + 1;
        const std::size_t first = line.find_first_not_of(" \t");
        if (line.empty() || (first != std::string::npos && line[first] == '#')) continue;
        std::vector<std::string> parts;
        for (std::size_t p = 0;;) {
            const std::size_t t = line.find('\t', p);
            parts.push_back(line.substr(p, t == std::string::npos ? std::string::npos : t - p));
            if (t == std::string::npos) break;
            p = t + 1;
        }
        if (parts.size() < 2) throw Fail{"pc_option_messages: a row without text: " + line};
        const auto unescape = [](std::string t) {  // "\n" is a line break
            for (std::size_t k = 0; (k = t.find("\\n", k)) != std::string::npos; ++k) t.replace(k, 2, "\n");
            return t;
        };
        rows[std::atoi(parts[0].c_str())] = Message{unescape(parts[1]), parts.size() > 2 ? unescape(parts[2]) : ""};
    }
    return rows;
}

// The message bundle: the port's ids in groups 200 and 201, the dialog groups
// and the event text.
Bytes build_menu_messages(const Bytes& raw) {
    Bytes payload, dcx_head;
    if (raw.size() >= 4 && std::memcmp(raw.data(), "DCX\0", 4) == 0) {
        if (raw.size() < 0x2c || std::memcmp(raw.data() + 0x28, "DFLT", 4) != 0) throw Fail{"the message bundle is not DFLT"};
        const std::size_t i = find(raw, "DCA\0", 4);
        if (i == std::string::npos) throw Fail{"no DCA block"};
        const std::uint32_t dca_size = be32(raw, i + 4);
        dcx_head.assign(raw.begin(), raw.begin() + i + dca_size);
        payload = inflate_all(raw, i + dca_size);
    } else {
        payload = raw;
    }
    Bnd4 bnd = bnd_read(payload);
    const auto rows = read_messages();
    std::set<std::uint32_t> added;
    for (BndFile& f : bnd.files) {
        if (f.id != kMenuText && f.id != kLineHelp && f.id != kDialogText && f.id != kScreenDialogText && f.id != kEventText) continue;
        const int group_kind = f.id == kEventText ? 2 : (f.id == kDialogText || f.id == kScreenDialogText) ? 1 : 0;
        Fmg ids = fmg_read(f.data);
        for (const auto& [mid, m] : rows) {
            if (message_kind(mid) != group_kind) continue;
            if (ids.count(mid)) throw Fail{"message id " + std::to_string(mid) + " is already in group " + std::to_string(f.id)};
        }
        for (const auto& [mid, m] : rows) {
            if (message_kind(mid) != group_kind) continue;
            const std::string& s = f.id == kLineHelp ? m.help : m.text;
            ids[mid] = s.empty() ? std::nullopt : std::optional<Bytes>(utf16(s));
        }
        f.data = fmg_write(ids);
        added.insert(f.id);
    }
    if (!added.count(kMenuText) || !added.count(kLineHelp) || !added.count(kDialogText) || !added.count(kScreenDialogText) ||
        !added.count(kEventText))
        throw Fail{"the message bundle lacks the menu text, line help, dialog or event text group"};
    const Bytes packed = bnd_pack(bnd);
    if (dcx_head.empty()) return packed;
    uLongf clen = compressBound(static_cast<uLong>(packed.size()));
    Bytes comp(clen);
    if (compress2(comp.data(), &clen, packed.data(), static_cast<uLong>(packed.size()), 9) != Z_OK) throw Fail{"compress2"};
    comp.resize(clen);
    Bytes out = dcx_head;
    const std::size_t dcs = find(out, "DCS\0", 4);
    if (dcs == std::string::npos) throw Fail{"no DCS block"};
    putbe32(out, dcs + 4, static_cast<std::uint32_t>(packed.size()));
    putbe32(out, dcs + 8, static_cast<std::uint32_t>(clen));
    app(out, comp);
    return out;
}

// ---------------------------------------------------------------- files

namespace fs = std::filesystem;

std::optional<Bytes> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::nullopt;
    return Bytes(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

bool write_file(const fs::path& p, const Bytes& b) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    const fs::path tmp = p.string() + ".part";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
        if (!f) return false;
    }
    fs::rename(tmp, p, ec);
    return !ec;
}

std::uint64_t fnv1a(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ull) {
    const auto* c = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
    return h;
}

const char kMovie[] = "dvdroot_ps4/menu/optionsetting.gfx";
// The message bundle of every language the dump has. The game picks its
// folder from the disc's region and the system language - the US disc reads
// engus, a European one enggb, deude, frafr, ... - so a bundle written for
// one language only left every other showing the repository's "(110006)"
// placeholders for the port's rows, and opening one then crashed on the
// missing text. The Japanese fork writes Japanese strings into all available bundles.
// The title menu's movies, each with one more line for Quit Game
// (engine/option_menu.cpp); the DLC edition uses the second.
const char* const kTitleMovies[] = {"dvdroot_ps4/menu/title.gfx", "dvdroot_ps4/menu/title_dlc.gfx"};
const char kMessageDir[] = "dvdroot_ps4/msg";
const char kMessageFile[] = "menu.msgbnd.dcx";

}  // namespace

bool menu_assets_ensure() {
    const std::string app0 = hle_fs_app0_root(), data = hle_fs_data_root();
    if (app0.empty() || data.empty()) return false;
    const fs::path out = fs::path(data) / "bbhost" / "menu-assets";
    const auto movie_src = read_file(hle_fs_game_file(kMovie));
    if (!movie_src) {
        host_log("menu assets: the dump has no %s", kMovie);
        return false;
    }
    // Every language folder with a menu bundle, in a stable order: the game
    // folder's and its update's (core/config.h), each bundle the update's
    // when it has one.
    std::vector<std::pair<std::string, Bytes>> langs;  // relative path, source
    {
        std::set<std::string> names;
        for (const std::string root : {app0, std::string(hle_fs_update_root())}) {
            if (root.empty()) continue;
            std::error_code ec;
            for (const auto& e : fs::directory_iterator(fs::path(root) / kMessageDir, ec))
                if (e.is_directory(ec)) names.insert(e.path().filename().string());
        }
        for (const std::string& n : names) {
            const std::string rel = std::string(kMessageDir) + "/" + n + "/" + kMessageFile;
            if (auto b = read_file(hle_fs_game_file(rel))) langs.emplace_back(rel, std::move(*b));
        }
    }
    if (langs.empty()) {
        host_log("menu assets: the dump has no %s/*/%s", kMessageDir, kMessageFile);
        return false;
    }
    // What these were made from: the generator, the strings and every source.
    std::uint64_t h = fnv1a(&kGeneratorVersion, sizeof(kGeneratorVersion));
    h = fnv1a(kMessagesTsv, sizeof(kMessagesTsv), h);
    h = fnv1a(movie_src->data(), movie_src->size(), h);
    for (const auto& [rel, src] : langs) {
        h = fnv1a(rel.data(), rel.size(), h);
        h = fnv1a(src.data(), src.size(), h);
    }
    std::vector<std::pair<std::string, Bytes>> titles;
    for (const char* rel : kTitleMovies)
        if (auto t = read_file(hle_fs_game_file(rel))) {
            h = fnv1a(t->data(), t->size(), h);
            titles.emplace_back(rel, std::move(*t));
        }
    char stamp[32];
    std::snprintf(stamp, sizeof(stamp), "%016llx", static_cast<unsigned long long>(h));
    const auto have = read_file(out / "stamp");
    bool fresh = have && std::string(have->begin(), have->end()) == stamp && fs::exists(out / kMovie);
    for (const auto& [rel, src] : langs) fresh = fresh && fs::exists(out / rel);
    for (const auto& [rel, src] : titles) fresh = fresh && fs::exists(out / rel);
    if (!fresh) {
        try {
            const Bytes movie = build_option_movie(*movie_src);
            if (!write_file(out / kMovie, movie)) {
                host_log("menu assets: cannot write %s", out.string().c_str());
                return false;
            }
            int made = 0;
            for (const auto& [rel, src] : langs) {
                try {
                    if (!write_file(out / rel, build_menu_messages(src))) {
                        host_log("menu assets: cannot write %s", (out / rel).string().c_str());
                        continue;
                    }
                    ++made;
                } catch (const Fail& f) {
                    host_log("menu assets: %s not made: %s", rel.c_str(), f.why.c_str());
                }
            }
            for (const auto& [rel, src] : titles) {
                try {
                    if (!write_file(out / rel, add_list_slot(src, "CommandList")))
                        host_log("menu assets: cannot write %s", (out / rel).string().c_str());
                } catch (const Fail& f) {
                    host_log("menu assets: %s not made: %s", rel.c_str(), f.why.c_str());
                }
            }
            if (!write_file(out / "stamp", Bytes(stamp, stamp + std::strlen(stamp)))) {
                host_log("menu assets: cannot write %s", out.string().c_str());
                return false;
            }
            host_log("menu assets: made %s (%zu bytes) and %d of %zu language bundles in %s", kMovie, movie.size(), made, langs.size(),
                     out.string().c_str());
        } catch (const Fail& f) {
            host_log("menu assets: not made: %s", f.why.c_str());
            return false;
        } catch (const std::exception& e) {
            host_log("menu assets: not made: %s", e.what());
            return false;
        }
    }
    hle_fs_set_generated_root(out.string().c_str());
    return true;
}
