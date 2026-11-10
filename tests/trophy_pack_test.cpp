// core/trophy_pack.cpp: a trophy pack's names, grades and icons, read from a
// made-up trophy00.trp (tests/trophy_test_pack.h, written by
// make_trophy_test_pack.py with openssl doing the AES, so the format is read
// two independent ways) - and refused cleanly when the key, the ID or the file
// is wrong. Plus the PNG decoder and the resizer the popup's icon goes through.
#include "core/trophy_pack.h"

#include "trophy_test_pack.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

int g_fail = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);  \
            ++g_fail;                                                 \
        }                                                             \
    } while (0)

namespace fs = std::filesystem;
fs::path g_dir;

std::vector<std::uint8_t> fixture() { return {kTestTrp, kTestTrp + kTestTrpSize}; }

std::string write_file(const char* name, const std::vector<std::uint8_t>& bytes) {
    const fs::path p = g_dir / name;
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return p.string();
}

std::string npbind(const std::string& comm_id) {
    std::vector<std::uint8_t> b(32, 0);  // the real file has a header and other records around the ID
    b.insert(b.end(), comm_id.begin(), comm_id.end());
    b.resize(b.size() + 24, 0);
    return write_file("npbind.dat", b);
}

struct Parsed {
    bool ok;
    std::string err;
    std::vector<TrophyInfo> t;
};
Parsed parse(const std::vector<std::uint8_t>& trp, const std::string& comm = kTestCommId, const std::string& key = kTestTrophyKey) {
    Parsed r;
    r.ok = trophy_pack_parse(write_file("p.trp", trp), comm, key, &r.t, &r.err);
    return r;
}

// Big-endian field in the TRP header/entries, for the mutants below.
void put32(std::vector<std::uint8_t>& b, std::size_t at, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b[at + i] = static_cast<std::uint8_t>(v >> (24 - 8 * i));
}

void pack_reading() {
    const Parsed p = parse(fixture());
    CHECK(p.ok);
    CHECK(p.err.empty());
    CHECK(p.t.size() == 4);
    if (p.t.size() != 4) return;
    CHECK(p.t[0].id == 0 && p.t[0].grade == 'G' && !p.t[0].hidden);
    CHECK(p.t[0].name == "Gold & Glory");                    // &amp; decoded
    CHECK(p.t[0].detail == "First line.\nSecond line.");     // a newline kept
    CHECK(p.t[1].id == 1 && p.t[1].grade == 'B' && p.t[1].hidden);
    CHECK(p.t[1].name == "Caf\xc3\xa9 Bronze");              // UTF-8 kept as it is
    CHECK(p.t[2].id == 2 && p.t[2].grade == 'S' && !p.t[2].hidden && p.t[2].name == "Silver");
    CHECK(p.t[3].id == 3 && p.t[3].grade == 'P' && p.t[3].name == "Everything");
    CHECK(p.t[0].icon_size > 8 && p.t[0].icon_pos > 0);      // TROP000.PNG
    CHECK(p.t[1].icon_size == 0 && p.t[2].icon_size == 0);   // the others have none

    CHECK(trophy_grade_code('P') == 1 && trophy_grade_code('G') == 2);
    CHECK(trophy_grade_code('S') == 3 && trophy_grade_code('B') == 4 && trophy_grade_code('?') == 4);
}

void pack_refusals() {
    // The key and the ID: either wrong and the text does not decrypt to XML.
    std::string key = kTestTrophyKey;
    key[5] = key[5] == '0' ? '1' : '0';
    Parsed p = parse(fixture(), kTestCommId, key);
    CHECK(!p.ok && !p.err.empty() && p.t.empty());
    p = parse(fixture(), "NPWR00001_00");
    CHECK(!p.ok && !p.err.empty());
    p = parse(fixture(), kTestCommId, "");  // no key supplied
    CHECK(!p.ok && !p.err.empty());
    p = parse(fixture(), kTestCommId, "0011");  // too short
    CHECK(!p.ok && !p.err.empty());
    p = parse(fixture(), kTestCommId, "zz0102030405060708090a0b0c0d0e0f");  // not hex
    CHECK(!p.ok && !p.err.empty());
    p = parse(fixture(), "");
    CHECK(!p.ok && !p.err.empty());

    // The file: not a trophy pack, cut short, an entry past its end, absent.
    auto bad = fixture();
    bad[0] ^= 0xff;
    CHECK(!parse(bad).ok);
    // Cut inside the header, the entries and TROP.ESFM itself (it starts at 0x131).
    for (const std::size_t keep : {std::size_t{0}, std::size_t{10}, std::size_t{0x50}, std::size_t{0x100}, std::size_t{0x159}}) {
        auto cut = fixture();
        cut.resize(keep);
        CHECK(!parse(cut).ok);
    }
    auto tail = fixture();  // only the last icon's last byte gone: the text is all there
    tail.pop_back();
    CHECK(parse(tail).ok);
    auto far = fixture();
    put32(far, 0x60 + 0x40 + 0x20 + 4, 0x0fffffff);  // TROP.ESFM's position (low half), far past the file
    CHECK(!parse(far).ok);
    auto many = fixture();
    put32(many, 0x10, 0x00ffffff);  // a count the file cannot hold
    CHECK(!parse(many).ok);
    std::string err;
    CHECK(!trophy_pack_parse((g_dir / "absent.trp").string(), kTestCommId, kTestTrophyKey, nullptr, &err) && !err.empty());
}

void comm_id() {
    CHECK(trophy_pack_comm_id(npbind(kTestCommId)) == kTestCommId);
    CHECK(trophy_pack_comm_id(write_file("empty.dat", {})).empty());
    CHECK(trophy_pack_comm_id(write_file("junk.dat", std::vector<std::uint8_t>(200, 'x'))).empty());
    CHECK(trophy_pack_comm_id((g_dir / "absent.dat").string()).empty());
    // Not an ID: letters where digits go, or a short one.
    CHECK(trophy_pack_comm_id(npbind("NPWRABCDE_00")).empty());
    CHECK(trophy_pack_comm_id(npbind("NPWR1234_00")).empty());
}

int g_loader_calls = 0;
std::string g_trp_path, g_npbind_path;
void test_loader() {
    ++g_loader_calls;
    std::string err;
    CHECK(trophy_pack_load(g_trp_path, g_npbind_path, kTestTrophyKey, &err));
}

// The process-wide pack: nothing before a load; the loader runs once, on the
// first ask; a pack that did not load leaves nothing to find.
void pack_in_the_process() {
    CHECK(trophy_pack_count() == 0);
    CHECK(!trophy_pack_find(0, nullptr));
    std::vector<std::uint8_t> png;
    CHECK(!trophy_pack_icon(0, &png));

    g_trp_path = write_file("trophy00.trp", fixture());
    g_npbind_path = npbind(kTestCommId);
    std::string err;
    CHECK(!trophy_pack_load(g_trp_path, g_npbind_path, "", &err) && !err.empty());      // no key
    CHECK(!trophy_pack_load(g_trp_path, (g_dir / "absent.dat").string(), kTestTrophyKey, &err) && !err.empty());
    CHECK(trophy_pack_count() == 0 && !trophy_pack_find(0, nullptr));

    trophy_pack_set_loader(test_loader);
    CHECK(g_loader_calls == 0);  // installing it reads nothing
    TrophyInfo t;
    CHECK(trophy_pack_find(0, &t) && t.name == "Gold & Glory");
    CHECK(g_loader_calls == 1);
    CHECK(trophy_pack_find(3, &t) && t.grade == 'P');
    CHECK(!trophy_pack_find(7, &t));
    CHECK(trophy_pack_find(1, nullptr));  // a null out is fine
    CHECK(trophy_pack_count() == 4);
    CHECK(g_loader_calls == 1);  // asked four times, loaded once
}

std::uint32_t px(const std::vector<std::uint8_t>& rgba, int w, int x, int y) {
    const std::uint8_t* p = rgba.data() + (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 4;
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) | (static_cast<std::uint32_t>(p[2]) << 8) | p[3];
}

void icons() {
    std::vector<std::uint8_t> png;
    CHECK(trophy_pack_icon(0, &png));
    CHECK(png.size() > 8 && std::memcmp(png.data(), "\x89PNG\r\n\x1a\n", 8) == 0);
    CHECK(!trophy_pack_icon(1, &png));   // in the pack, no icon
    CHECK(!trophy_pack_icon(99, &png));  // not in the pack
    CHECK(!trophy_pack_icon(0, nullptr));

    int w = 0, h = 0;
    std::vector<std::uint8_t> rgba;
    std::string err;
    CHECK(png_decode_rgba(png.data(), png.size(), &w, &h, &rgba, &err));
    CHECK(w == 8 && h == 8 && rgba.size() == 8 * 8 * 4);
    if (rgba.size() != 8 * 8 * 4) return;
    CHECK(px(rgba, 8, 0, 0) == 0xc82828ffu && px(rgba, 8, 3, 3) == 0xc82828ffu);  // opaque red
    CHECK(px(rgba, 8, 4, 0) == 0x28c828ffu);                                         // opaque green
    CHECK(px(rgba, 8, 0, 4) == 0x2828c880u);                                         // half-transparent blue, straight alpha
    CHECK((px(rgba, 8, 7, 7) & 0xffu) == 0);                                         // clear

    // Cut short, a wrong signature, nothing at all: refused with a reason.
    std::vector<std::uint8_t> cut(png.begin(), png.end() - 20);
    err.clear();
    CHECK(!png_decode_rgba(cut.data(), cut.size(), &w, &h, &rgba, &err) && !err.empty());
    std::vector<std::uint8_t> sig = png;
    sig[1] = 'X';
    err.clear();
    CHECK(!png_decode_rgba(sig.data(), sig.size(), &w, &h, &rgba, &err) && !err.empty());
    err.clear();
    CHECK(!png_decode_rgba(png.data(), 0, &w, &h, &rgba, &err) && !err.empty());
    // A header that promises more than the data has.
    std::vector<std::uint8_t> huge = png;
    huge[16] = 0x7f;  // IHDR width's top byte
    CHECK(!png_decode_rgba(huge.data(), huge.size(), &w, &h, &rgba, nullptr));

    // The resizer: each quadrant of the icon survives a shrink to 2x2 whole,
    // the half-transparent one with its colour (premultiplied, so the clear
    // quadrant does not darken it), a flat colour stays flat, same size is a copy.
    const std::vector<std::uint8_t> small = rgba_resize(rgba, 8, 8, 2, 2);
    CHECK(small.size() == 2 * 2 * 4);
    if (small.size() == 16) {
        CHECK(px(small, 2, 0, 0) == 0xc82828ffu && px(small, 2, 1, 0) == 0x28c828ffu);
        CHECK(px(small, 2, 0, 1) == 0x2828c880u);
        CHECK((px(small, 2, 1, 1) & 0xffu) == 0);
    }
    CHECK(rgba_resize(rgba, 8, 8, 8, 8) == rgba);
    const std::vector<std::uint8_t> flat(8 * 8 * 4, 0x80);
    const std::vector<std::uint8_t> flat3 = rgba_resize(flat, 8, 8, 3, 3);  // a factor that does not divide
    bool all = flat3.size() == 3 * 3 * 4;
    for (std::size_t i = 0; all && i < flat3.size(); ++i) all = flat3[i] == 0x80;
    CHECK(all);
}

}  // namespace

int main() {
    char tmpl[] = "/tmp/trophy_pack_test_XXXXXX";
    if (!mkdtemp(tmpl)) {
        std::printf("trophy_pack_test: no temp folder\n");
        return 1;
    }
    g_dir = tmpl;

    pack_reading();
    pack_refusals();
    comm_id();
    pack_in_the_process();  // once the pack is loaded, so after the rest
    icons();

    std::error_code ec;
    fs::remove_all(g_dir, ec);
    if (g_fail) {
        std::printf("trophy_pack_test: %d failed\n", g_fail);
        return 1;
    }
    std::printf("trophy_pack_test: ok\n");
    return 0;
}
