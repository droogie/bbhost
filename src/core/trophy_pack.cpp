#include "core/trophy_pack.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>

#include <zlib.h>

namespace {

// ---- AES-128 (FIPS 197), just enough for the trophy pack: the tables are
// built at first use, a block is encrypted or decrypted a byte at a time, and
// speed does not matter (a pack's text is a few kilobytes).

struct AesTables {
    std::uint8_t sbox[256];
    std::uint8_t inv[256];
};

const AesTables& aes_tables() {
    static const AesTables t = [] {
        AesTables r{};
        std::uint8_t p = 1, q = 1;
        const auto rotl = [](std::uint8_t v, int s) { return static_cast<std::uint8_t>((v << s) | (v >> (8 - s))); };
        do {
            p = static_cast<std::uint8_t>(p ^ (p << 1) ^ ((p & 0x80) ? 0x1b : 0));  // times 3
            q = static_cast<std::uint8_t>(q ^ (q << 1));                             // divided by 3
            q = static_cast<std::uint8_t>(q ^ (q << 2));
            q = static_cast<std::uint8_t>(q ^ (q << 4));
            if (q & 0x80) q = static_cast<std::uint8_t>(q ^ 0x09);
            r.sbox[p] = static_cast<std::uint8_t>(q ^ rotl(q, 1) ^ rotl(q, 2) ^ rotl(q, 3) ^ rotl(q, 4) ^ 0x63);
        } while (p != 1);
        r.sbox[0] = 0x63;
        for (int i = 0; i < 256; ++i) r.inv[r.sbox[i]] = static_cast<std::uint8_t>(i);
        return r;
    }();
    return t;
}

std::uint8_t xtime(std::uint8_t x) { return static_cast<std::uint8_t>((x << 1) ^ ((x & 0x80) ? 0x1b : 0)); }

std::uint8_t gmul(std::uint8_t x, std::uint8_t y) {
    std::uint8_t r = 0;
    while (y) {
        if (y & 1) r = static_cast<std::uint8_t>(r ^ x);
        x = xtime(x);
        y = static_cast<std::uint8_t>(y >> 1);
    }
    return r;
}

class Aes128 {
public:
    explicit Aes128(const std::uint8_t key[16]) {
        const AesTables& t = aes_tables();
        std::memcpy(rk_, key, 16);
        std::uint8_t rcon = 1;
        for (int i = 16; i < 176; i += 4) {
            std::uint8_t w[4];
            std::memcpy(w, rk_ + i - 4, 4);
            if (i % 16 == 0) {
                const std::uint8_t first = w[0];
                w[0] = static_cast<std::uint8_t>(t.sbox[w[1]] ^ rcon);
                w[1] = t.sbox[w[2]];
                w[2] = t.sbox[w[3]];
                w[3] = t.sbox[first];
                rcon = xtime(rcon);
            }
            for (int j = 0; j < 4; ++j) rk_[i + j] = static_cast<std::uint8_t>(rk_[i - 16 + j] ^ w[j]);
        }
    }

    void encrypt_block(const std::uint8_t in[16], std::uint8_t out[16]) const {
        const AesTables& t = aes_tables();
        std::uint8_t s[16];
        std::memcpy(s, in, 16);
        add_round_key(s, 0);
        for (int round = 1; round <= 10; ++round) {
            for (int i = 0; i < 16; ++i) s[i] = t.sbox[s[i]];
            std::uint8_t sh[16];
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r) sh[c * 4 + r] = s[((c + r) % 4) * 4 + r];
            std::memcpy(s, sh, 16);
            if (round != 10) {
                for (int c = 0; c < 4; ++c) {
                    std::uint8_t* a = s + c * 4;
                    const std::uint8_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
                    a[0] = static_cast<std::uint8_t>(xtime(a0) ^ xtime(a1) ^ a1 ^ a2 ^ a3);
                    a[1] = static_cast<std::uint8_t>(a0 ^ xtime(a1) ^ xtime(a2) ^ a2 ^ a3);
                    a[2] = static_cast<std::uint8_t>(a0 ^ a1 ^ xtime(a2) ^ xtime(a3) ^ a3);
                    a[3] = static_cast<std::uint8_t>(xtime(a0) ^ a0 ^ a1 ^ a2 ^ xtime(a3));
                }
            }
            add_round_key(s, round);
        }
        std::memcpy(out, s, 16);
    }

    void decrypt_block(const std::uint8_t in[16], std::uint8_t out[16]) const {
        const AesTables& t = aes_tables();
        std::uint8_t s[16];
        std::memcpy(s, in, 16);
        add_round_key(s, 10);
        for (int round = 9; round >= 0; --round) {
            std::uint8_t sh[16];
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r) sh[((c + r) % 4) * 4 + r] = s[c * 4 + r];
            for (int i = 0; i < 16; ++i) s[i] = t.inv[sh[i]];
            add_round_key(s, round);
            if (round != 0) {
                for (int c = 0; c < 4; ++c) {
                    std::uint8_t* a = s + c * 4;
                    const std::uint8_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
                    a[0] = static_cast<std::uint8_t>(gmul(a0, 14) ^ gmul(a1, 11) ^ gmul(a2, 13) ^ gmul(a3, 9));
                    a[1] = static_cast<std::uint8_t>(gmul(a0, 9) ^ gmul(a1, 14) ^ gmul(a2, 11) ^ gmul(a3, 13));
                    a[2] = static_cast<std::uint8_t>(gmul(a0, 13) ^ gmul(a1, 9) ^ gmul(a2, 14) ^ gmul(a3, 11));
                    a[3] = static_cast<std::uint8_t>(gmul(a0, 11) ^ gmul(a1, 13) ^ gmul(a2, 9) ^ gmul(a3, 14));
                }
            }
        }
        std::memcpy(out, s, 16);
    }

private:
    void add_round_key(std::uint8_t s[16], int round) const {
        for (int i = 0; i < 16; ++i) s[i] = static_cast<std::uint8_t>(s[i] ^ rk_[round * 16 + i]);
    }
    std::uint8_t rk_[176];
};

// CBC decryption of whole blocks; `len` is a multiple of 16.
void cbc_decrypt(const Aes128& aes, const std::uint8_t iv[16], const std::uint8_t* in, std::size_t len,
                 std::uint8_t* out) {
    std::uint8_t prev[16];
    std::memcpy(prev, iv, 16);
    for (std::size_t off = 0; off + 16 <= len; off += 16) {
        std::uint8_t block[16];
        std::memcpy(block, in + off, 16);  // kept: `out` may be `in`
        aes.decrypt_block(block, out + off);
        for (int i = 0; i < 16; ++i) out[off + i] = static_cast<std::uint8_t>(out[off + i] ^ prev[i]);
        std::memcpy(prev, block, 16);
    }
}

// ---- The .trp container: a 0x60-byte header, then 0x40-byte entries
// (name[32], position u64, size u64, flag u32, padding), all big-endian.

std::uint32_t be32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) | p[3];
}
std::uint64_t be64(const std::uint8_t* p) { return (static_cast<std::uint64_t>(be32(p)) << 32) | be32(p + 4); }

constexpr std::uint32_t kTrpMagic = 0xDCA24D00u;
constexpr std::size_t kTrpHeader = 0x60;
constexpr std::size_t kTrpEntry = 0x40;
constexpr std::uint64_t kMaxEsfm = 4u * 1024u * 1024u;

bool read_at(std::FILE* f, std::uint64_t pos, std::size_t n, std::vector<std::uint8_t>* out) {
    if (pos > 0x7fffffffu) return false;
    if (std::fseek(f, static_cast<long>(pos), SEEK_SET) != 0) return false;
    out->assign(n, 0);
    return std::fread(out->data(), 1, n, f) == n;
}

bool parse_hex_key(const std::string& hex, std::uint8_t key[16]) {
    std::string digits;
    for (char c : hex) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        digits += c;
    }
    if (digits.size() > 2 && digits[0] == '0' && (digits[1] == 'x' || digits[1] == 'X')) digits.erase(0, 2);
    if (digits.size() != 32) return false;
    for (int i = 0; i < 16; ++i) {
        unsigned v = 0;
        for (int j = 0; j < 2; ++j) {
            const char c = digits[static_cast<std::size_t>(i * 2 + j)];
            unsigned d;
            if (c >= '0' && c <= '9') d = static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') d = static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') d = static_cast<unsigned>(c - 'A' + 10);
            else return false;
            v = v * 16 + d;
        }
        key[i] = static_cast<std::uint8_t>(v);
    }
    return true;
}

// ---- The decrypted XML: <trophy id=".." hidden=".." ttype=".." ..> with
// <name> and <detail> inside.

void append_utf8(std::string* s, unsigned cp) {
    if (cp < 0x80) {
        *s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        *s += static_cast<char>(0xc0 | (cp >> 6));
        *s += static_cast<char>(0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
        *s += static_cast<char>(0xe0 | (cp >> 12));
        *s += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        *s += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
        *s += static_cast<char>(0xf0 | (cp >> 18));
        *s += static_cast<char>(0x80 | ((cp >> 12) & 0x3f));
        *s += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
        *s += static_cast<char>(0x80 | (cp & 0x3f));
    }
}

std::string decode_entities(std::string_view s) {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '&') {
            out += s[i];
            continue;
        }
        const std::size_t semi = s.find(';', i);
        if (semi == std::string_view::npos || semi - i > 10) {
            out += s[i];
            continue;
        }
        const std::string_view e = s.substr(i + 1, semi - i - 1);
        if (e == "amp") out += '&';
        else if (e == "lt") out += '<';
        else if (e == "gt") out += '>';
        else if (e == "quot") out += '"';
        else if (e == "apos") out += '\'';
        else if (e.size() > 1 && e[0] == '#') {
            const bool hex = e[1] == 'x' || e[1] == 'X';
            const std::string num(e.substr(hex ? 2 : 1));
            char* end = nullptr;
            const unsigned long cp = std::strtoul(num.c_str(), &end, hex ? 16 : 10);
            if (num.empty() || *end != 0 || cp > 0x10ffff) {
                out.append(s.substr(i, semi - i + 1));
            } else {
                append_utf8(&out, static_cast<unsigned>(cp));
            }
        } else {
            out.append(s.substr(i, semi - i + 1));
        }
        i = semi;
    }
    return out;
}

// The value of name="..." in a start tag; found = false when it is not there.
std::string attr_of(std::string_view tag, std::string_view name, bool* found) {
    const std::string needle = " " + std::string(name) + "=\"";
    const std::size_t at = tag.find(needle);
    if (found) *found = at != std::string_view::npos;
    if (at == std::string_view::npos) return {};
    const std::size_t from = at + needle.size();
    const std::size_t to = tag.find('"', from);
    if (to == std::string_view::npos) return {};
    return decode_entities(tag.substr(from, to - from));
}

std::string element_of(std::string_view body, std::string_view name) {
    const std::string open = "<" + std::string(name) + ">", close = "</" + std::string(name) + ">";
    const std::size_t a = body.find(open);
    if (a == std::string_view::npos) return {};
    const std::size_t from = a + open.size();
    const std::size_t b = body.find(close, from);
    if (b == std::string_view::npos) return {};
    return decode_entities(body.substr(from, b - from));
}

std::vector<TrophyInfo> parse_trophies(const std::string& xml) {
    std::vector<TrophyInfo> out;
    std::size_t pos = 0;
    while ((pos = xml.find("<trophy ", pos)) != std::string::npos) {
        const std::size_t tag_end = xml.find('>', pos);
        if (tag_end == std::string::npos) break;
        const std::string_view tag(xml.data() + pos, tag_end - pos);
        const bool self_closing = !tag.empty() && tag.back() == '/';
        std::size_t next = tag_end + 1;

        bool has_id = false;
        TrophyInfo t;
        const std::string id = attr_of(tag, "id", &has_id);
        if (has_id && !id.empty()) {
            t.id = std::atoi(id.c_str());
            const std::string type = attr_of(tag, "ttype", nullptr);
            t.grade = type.empty() ? 'B' : type[0];
            t.hidden = attr_of(tag, "hidden", nullptr) == "yes";
            if (!self_closing) {
                std::size_t close = xml.find("</trophy>", next);
                if (close == std::string::npos) close = xml.size();
                const std::string_view body(xml.data() + next, close - next);
                t.name = element_of(body, "name");
                t.detail = element_of(body, "detail");
                next = close;
            }
            out.push_back(std::move(t));
        } else if (!self_closing) {
            const std::size_t close = xml.find("</trophy>", next);
            if (close != std::string::npos) next = close;
        }
        pos = next;
    }
    return out;
}

std::mutex g_mu;
std::vector<TrophyInfo> g_pack;  // under g_mu
std::string g_trp_path;          // the pack they were read from (under g_mu)
void (*g_loader)() = nullptr;
std::once_flag g_loader_once;

}  // namespace

bool trophy_pack_parse(const std::string& trp_path, const std::string& np_comm_id, const std::string& key_hex,
                       std::vector<TrophyInfo>* out, std::string* err) {
    const auto fail = [&](const std::string& why) {
        if (err) *err = why;
        return false;
    };
    std::uint8_t trophy_key[16];
    if (!parse_hex_key(key_hex, trophy_key)) {
        return fail(key_hex.empty() ? "no trophy key (set BBHOST_TROPHY_KEY, or put it in <data>/trophy_key.txt)"
                                    : "the trophy key must be 32 hex digits");
    }
    if (np_comm_id.empty() || np_comm_id.size() > 16) return fail("the NP communication ID was not found");

    std::FILE* f = std::fopen(trp_path.c_str(), "rb");
    if (!f) return fail("cannot open " + trp_path);
    struct Closer {
        std::FILE* f;
        ~Closer() { std::fclose(f); }
    } closer{f};

    std::vector<std::uint8_t> head;
    if (!read_at(f, 0, kTrpHeader, &head)) return fail("the trophy pack is too short");
    if (be32(head.data()) != kTrpMagic) return fail("not a trophy pack (wrong magic)");
    const std::uint32_t count = be32(head.data() + 0x10), entry_size = be32(head.data() + 0x14);
    if (entry_size != kTrpEntry || count == 0 || count > 1024) return fail("unexpected trophy pack layout");

    std::vector<std::uint8_t> table;
    if (!read_at(f, kTrpHeader, static_cast<std::size_t>(count) * kTrpEntry, &table)) {
        return fail("the trophy pack's entry table is cut short");
    }
    std::uint64_t pos = 0, size = 0;
    bool found = false;
    for (std::uint32_t i = 0; i < count && !found; ++i) {
        const std::uint8_t* e = table.data() + static_cast<std::size_t>(i) * kTrpEntry;
        if (std::strncmp(reinterpret_cast<const char*>(e), "TROP.ESFM", 32) == 0) {
            pos = be64(e + 32);
            size = be64(e + 40);
            found = true;
        }
    }
    if (!found) return fail("TROP.ESFM is not in the trophy pack");
    if (size < 32 || size > kMaxEsfm) return fail("TROP.ESFM has an odd size");

    std::vector<std::uint8_t> blob;
    if (!read_at(f, pos, static_cast<std::size_t>(size), &blob)) return fail("TROP.ESFM is cut short");

    // The text key: the communication ID, zero-padded to a block, encrypted with
    // the trophy key (CBC with a zero IV is one plain block here).
    std::uint8_t id_block[16] = {};
    std::memcpy(id_block, np_comm_id.data(), np_comm_id.size());
    std::uint8_t text_key[16];
    Aes128(trophy_key).encrypt_block(id_block, text_key);

    // The first 16 bytes of the entry are the IV, the rest the ciphertext.
    const std::size_t clen = (blob.size() - 16) / 16 * 16;
    std::vector<std::uint8_t> plain(clen);
    cbc_decrypt(Aes128(text_key), blob.data(), blob.data() + 16, clen, plain.data());
    while (!plain.empty() && plain.back() == 0) plain.pop_back();
    const std::string xml(plain.begin(), plain.end());
    if (xml.find("<trophyconf") == std::string::npos) {
        return fail("the pack did not decrypt to XML: the trophy key or the communication ID is wrong");
    }

    std::vector<TrophyInfo> trophies = parse_trophies(xml);
    if (trophies.empty()) return fail("no trophies in the pack's XML");
    // Each trophy's icon: the entry TROPnnn.PNG, nnn its id.
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint8_t* e = table.data() + static_cast<std::size_t>(i) * kTrpEntry;
        char name[33] = {};
        std::memcpy(name, e, 32);
        if (std::strncmp(name, "TROP", 4) != 0 || std::strcmp(name + 7, ".PNG") != 0) continue;
        if (!(name[4] >= '0' && name[4] <= '9' && name[5] >= '0' && name[5] <= '9' && name[6] >= '0' && name[6] <= '9')) continue;
        const int id = (name[4] - '0') * 100 + (name[5] - '0') * 10 + (name[6] - '0');
        for (TrophyInfo& t : trophies) {
            if (t.id == id) {
                t.icon_pos = be64(e + 32);
                t.icon_size = be64(e + 40);
            }
        }
    }
    if (out) *out = std::move(trophies);
    return true;
}

std::string trophy_pack_comm_id(const std::string& npbind_path) {
    std::FILE* f = std::fopen(npbind_path.c_str(), "rb");
    if (!f) return {};
    std::vector<std::uint8_t> data(1u << 20);
    data.resize(std::fread(data.data(), 1, data.size(), f));
    std::fclose(f);
    const auto digit = [](std::uint8_t c) { return c >= '0' && c <= '9'; };
    for (std::size_t i = 0; i + 12 <= data.size(); ++i) {
        const std::uint8_t* p = data.data() + i;
        if (std::memcmp(p, "NPWR", 4) != 0) continue;
        bool ok = p[9] == '_';
        for (int k = 4; k < 9 && ok; ++k) ok = digit(p[k]);
        ok = ok && digit(p[10]) && digit(p[11]);
        if (ok) return std::string(reinterpret_cast<const char*>(p), 12);
    }
    return {};
}

bool trophy_pack_load(const std::string& trp_path, const std::string& npbind_path, const std::string& key_hex,
                      std::string* err) {
    const std::string comm_id = trophy_pack_comm_id(npbind_path);
    if (comm_id.empty()) {
        if (err) *err = "no NP communication ID in " + npbind_path;
        return false;
    }
    std::vector<TrophyInfo> trophies;
    if (!trophy_pack_parse(trp_path, comm_id, key_hex, &trophies, err)) return false;
    std::lock_guard<std::mutex> lock(g_mu);
    g_pack = std::move(trophies);
    g_trp_path = trp_path;
    return true;
}

void trophy_pack_set_loader(void (*loader)()) { g_loader = loader; }

bool trophy_pack_find(int id, TrophyInfo* out) {
    if (g_loader) std::call_once(g_loader_once, g_loader);  // outside the lock: the loader takes it
    std::lock_guard<std::mutex> lock(g_mu);
    for (const TrophyInfo& t : g_pack) {
        if (t.id == id) {
            if (out) *out = t;
            return true;
        }
    }
    return false;
}

std::size_t trophy_pack_count() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_pack.size();
}

bool trophy_pack_icon(int id, std::vector<std::uint8_t>* png) {
    if (g_loader) std::call_once(g_loader_once, g_loader);
    std::string path;
    std::uint64_t pos = 0, size = 0;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        for (const TrophyInfo& t : g_pack) {
            if (t.id == id) {
                pos = t.icon_pos;
                size = t.icon_size;
            }
        }
        path = g_trp_path;
    }
    if (!png || path.empty() || size < 8 || size > (8u << 20)) return false;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    const bool ok = read_at(f, pos, static_cast<std::size_t>(size), png);
    std::fclose(f);
    return ok;
}

namespace {

// One sample of a row of `depth`-bit samples (1, 2, 4, 8 or 16 bits), as stored.
unsigned png_sample(const std::uint8_t* row, std::size_t index, int depth) {
    if (depth == 8) return row[index];
    if (depth == 16) return (static_cast<unsigned>(row[index * 2]) << 8) | row[index * 2 + 1];
    const std::size_t bit = index * static_cast<std::size_t>(depth);
    const unsigned byte = row[bit / 8];
    const int shift = 8 - depth - static_cast<int>(bit % 8);
    return (byte >> shift) & ((1u << depth) - 1u);
}

}  // namespace

bool png_decode_rgba(const std::uint8_t* d, std::size_t n, int* width, int* height, std::vector<std::uint8_t>* rgba,
                     std::string* err) {
    const auto fail = [&](const char* why) {
        if (err) *err = why;
        return false;
    };
    static const std::uint8_t kSig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    if (!d || n < 8 + 25 || std::memcmp(d, kSig, 8) != 0) return fail("not a PNG");
    std::uint32_t w = 0, h = 0;
    int depth = 0, ctype = -1, interlace = 0;
    bool have_header = false;
    std::vector<std::uint8_t> palette, trns, idat;
    for (std::size_t pos = 8; pos + 12 <= n;) {
        const std::uint32_t len = be32(d + pos);
        const char* type = reinterpret_cast<const char*>(d + pos + 4);
        if (len > n - pos - 12) return fail("a PNG chunk runs past the end");
        const std::uint8_t* body = d + pos + 8;
        if (std::memcmp(type, "IHDR", 4) == 0 && len >= 13) {
            w = be32(body);
            h = be32(body + 4);
            depth = body[8];
            ctype = body[9];
            interlace = body[12];
            if (body[10] != 0 || body[11] != 0) return fail("an unknown PNG compression or filter");
            have_header = true;
        } else if (std::memcmp(type, "PLTE", 4) == 0) {
            palette.assign(body, body + len);
        } else if (std::memcmp(type, "tRNS", 4) == 0) {
            trns.assign(body, body + len);
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), body, body + len);
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            break;
        }
        pos += 12 + static_cast<std::size_t>(len);
    }
    if (!have_header || idat.empty()) return fail("a PNG without a header or pixels");
    if (w == 0 || h == 0 || w > 8192 || h > 8192) return fail("a PNG of an odd size");
    if (interlace != 0) return fail("an interlaced PNG");
    int channels = 0;
    bool depth_ok = false;
    switch (ctype) {
        case 0: channels = 1; depth_ok = depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16; break;
        case 2: channels = 3; depth_ok = depth == 8 || depth == 16; break;
        case 3: channels = 1; depth_ok = depth == 1 || depth == 2 || depth == 4 || depth == 8; break;
        case 4: channels = 2; depth_ok = depth == 8 || depth == 16; break;
        case 6: channels = 4; depth_ok = depth == 8 || depth == 16; break;
        default: return fail("an unknown PNG colour type");
    }
    if (!depth_ok) return fail("a PNG bit depth that its colour type does not allow");
    if (ctype == 3 && palette.size() < 3) return fail("a palette PNG without a palette");

    const std::size_t bits_pp = static_cast<std::size_t>(channels) * static_cast<std::size_t>(depth);
    const std::size_t bpp = std::max<std::size_t>(1, bits_pp / 8);
    const std::size_t row_bytes = (static_cast<std::size_t>(w) * bits_pp + 7) / 8;
    const std::size_t want = static_cast<std::size_t>(h) * (row_bytes + 1);
    std::vector<std::uint8_t> raw(want);
    uLongf got = static_cast<uLongf>(want);
    if (uncompress(raw.data(), &got, idat.data(), static_cast<uLong>(idat.size())) != Z_OK || got != want) {
        return fail("the PNG's pixels do not inflate to the size its header says");
    }

    // Undo the row filters, in place: raw[y * (row_bytes + 1)] is the filter, then the row.
    std::vector<std::uint8_t> zero(row_bytes, 0);
    for (std::uint32_t y = 0; y < h; ++y) {
        std::uint8_t* line = raw.data() + static_cast<std::size_t>(y) * (row_bytes + 1);
        const int filter = line[0];
        std::uint8_t* cur = line + 1;
        const std::uint8_t* up = y ? line - row_bytes : zero.data();  // the row above, already undone
        if (filter > 4) return fail("a PNG row with an unknown filter");
        for (std::size_t i = 0; i < row_bytes; ++i) {
            const int a = i >= bpp ? cur[i - bpp] : 0, b = up[i], c = i >= bpp ? up[i - bpp] : 0;
            int add = 0;
            switch (filter) {
                case 1: add = a; break;
                case 2: add = b; break;
                case 3: add = (a + b) / 2; break;
                case 4: {
                    const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
                    add = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
                    break;
                }
                default: break;
            }
            cur[i] = static_cast<std::uint8_t>(cur[i] + add);
        }
    }

    std::vector<std::uint8_t> out(static_cast<std::size_t>(w) * h * 4);
    const unsigned maxv = (1u << depth) - 1u;
    const auto to8 = [&](unsigned v) -> std::uint8_t {
        return static_cast<std::uint8_t>(depth == 16 ? (v >> 8) : depth == 8 ? v : (v * 255u) / maxv);
    };
    for (std::uint32_t y = 0; y < h; ++y) {
        const std::uint8_t* row = raw.data() + static_cast<std::size_t>(y) * (row_bytes + 1) + 1;
        std::uint8_t* px = out.data() + static_cast<std::size_t>(y) * w * 4;
        for (std::uint32_t x = 0; x < w; ++x, px += 4) {
            const std::size_t at = static_cast<std::size_t>(x) * static_cast<std::size_t>(channels);
            switch (ctype) {
                case 0: px[0] = px[1] = px[2] = to8(png_sample(row, at, depth)); px[3] = 255; break;
                case 2:
                    px[0] = to8(png_sample(row, at, depth));
                    px[1] = to8(png_sample(row, at + 1, depth));
                    px[2] = to8(png_sample(row, at + 2, depth));
                    px[3] = 255;
                    break;
                case 3: {
                    const std::size_t idx = png_sample(row, at, depth);
                    if (idx * 3 + 2 >= palette.size()) return fail("a PNG pixel past its palette");
                    px[0] = palette[idx * 3];
                    px[1] = palette[idx * 3 + 1];
                    px[2] = palette[idx * 3 + 2];
                    px[3] = idx < trns.size() ? trns[idx] : 255;
                    break;
                }
                case 4:
                    px[0] = px[1] = px[2] = to8(png_sample(row, at, depth));
                    px[3] = to8(png_sample(row, at + 1, depth));
                    break;
                default:
                    px[0] = to8(png_sample(row, at, depth));
                    px[1] = to8(png_sample(row, at + 1, depth));
                    px[2] = to8(png_sample(row, at + 2, depth));
                    px[3] = to8(png_sample(row, at + 3, depth));
                    break;
            }
        }
    }
    if (width) *width = static_cast<int>(w);
    if (height) *height = static_cast<int>(h);
    if (rgba) *rgba = std::move(out);
    return true;
}

std::vector<std::uint8_t> rgba_resize(const std::vector<std::uint8_t>& src, int sw, int sh, int dw, int dh) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(std::max(dw, 0)) * static_cast<std::size_t>(std::max(dh, 0)) * 4, 0);
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || src.size() < static_cast<std::size_t>(sw) * sh * 4) return out;
    const double fx = static_cast<double>(sw) / dw, fy = static_cast<double>(sh) / dh;
    for (int dy = 0; dy < dh; ++dy) {
        const double y0 = dy * fy, y1 = (dy + 1) * fy;
        for (int dx = 0; dx < dw; ++dx) {
            const double x0 = dx * fx, x1 = (dx + 1) * fx;
            double r = 0, g = 0, b = 0, a = 0, total = 0;
            for (int sy = static_cast<int>(y0); sy < sh && sy < y1; ++sy) {
                const double wy = std::min<double>(sy + 1, y1) - std::max<double>(sy, y0);
                for (int sx = static_cast<int>(x0); sx < sw && sx < x1; ++sx) {
                    const double wgt = wy * (std::min<double>(sx + 1, x1) - std::max<double>(sx, x0));
                    const std::uint8_t* p = src.data() + (static_cast<std::size_t>(sy) * sw + sx) * 4;
                    const double pa = p[3] / 255.0;
                    r += wgt * p[0] * pa;
                    g += wgt * p[1] * pa;
                    b += wgt * p[2] * pa;
                    a += wgt * pa;
                    total += wgt;
                }
            }
            std::uint8_t* o = out.data() + (static_cast<std::size_t>(dy) * dw + dx) * 4;
            if (total > 0 && a > 0) {
                o[0] = static_cast<std::uint8_t>(std::clamp(r / a + 0.5, 0.0, 255.0));
                o[1] = static_cast<std::uint8_t>(std::clamp(g / a + 0.5, 0.0, 255.0));
                o[2] = static_cast<std::uint8_t>(std::clamp(b / a + 0.5, 0.0, 255.0));
                o[3] = static_cast<std::uint8_t>(std::clamp(a / total * 255.0 + 0.5, 0.0, 255.0));
            }
        }
    }
    return out;
}
