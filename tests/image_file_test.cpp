// core/image_file.h: a PNG it writes is a valid file - the signature, every
// chunk's CRC, an IHDR for 8-bit RGB - whose rows, unfiltered, are the
// pixels it was given, across several IDAT chunks; a PPM is the P6 header and
// the bytes as given.
#include "core/image_file.h"

#include <zlib.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

std::uint32_t be32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) << 24 | static_cast<std::uint32_t>(p[1]) << 16 | static_cast<std::uint32_t>(p[2]) << 8 | p[3];
}

std::vector<std::uint8_t> slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// A picture with flat areas, gradients and noise, so every filter wins somewhere.
std::vector<std::uint8_t> picture(std::uint32_t w, std::uint32_t h, std::uint32_t seed) {
    std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * h * 3);
    std::uint32_t r = seed;
    for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
            r = r * 1664525u + 1013904223u;
            std::uint8_t* p = &px[(static_cast<std::size_t>(y) * w + x) * 3];
            if (y < h / 4) {
                p[0] = 40, p[1] = 200, p[2] = 90;
            } else if (y < h / 2) {
                p[0] = static_cast<std::uint8_t>(x * 3), p[1] = static_cast<std::uint8_t>(y * 5), p[2] = static_cast<std::uint8_t>(x + y);
            } else {
                p[0] = static_cast<std::uint8_t>(r >> 24), p[1] = static_cast<std::uint8_t>(r >> 16), p[2] = static_cast<std::uint8_t>(r >> 8);
            }
        }
    }
    return px;
}

bool write(const std::string& path, std::uint32_t w, std::uint32_t h, const std::vector<std::uint8_t>& px) {
    ImageFile f;
    if (!f.open(path, w, h)) return false;
    for (std::uint32_t y = 0; y < h; ++y) {
        if (!f.row(&px[static_cast<std::size_t>(y) * w * 3])) return false;
    }
    return f.close();
}

int paeth(int a, int b, int c) {
    const int p = a + b - c, pa = p > a ? p - a : a - p, pb = p > b ? p - b : b - p, pc = p > c ? p - c : c - p;
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

// Decodes an 8-bit RGB PNG as the specification reads one.
bool decode(const std::vector<std::uint8_t>& file, std::uint32_t* w, std::uint32_t* h, std::vector<std::uint8_t>* px, int* idats) {
    static const std::uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    if (file.size() < 8 || std::memcmp(file.data(), sig, 8) != 0) return false;
    std::vector<std::uint8_t> z;
    std::size_t at = 8;
    bool ihdr = false, iend = false;
    *idats = 0;
    while (at + 12 <= file.size() && !iend) {
        const std::uint32_t len = be32(&file[at]);
        if (at + 12 + len > file.size()) return false;
        const std::uint8_t* type = &file[at + 4];
        const std::uint8_t* data = &file[at + 8];
        const uLong crc = crc32(crc32(0, type, 4), data, len);
        if (crc != be32(&file[at + 8 + len])) return false;
        if (!std::memcmp(type, "IHDR", 4)) {
            if (len != 13 || ihdr || at != 8) return false;
            *w = be32(data), *h = be32(data + 4);
            if (data[8] != 8 || data[9] != 2 || data[10] || data[11] || data[12]) return false;
            ihdr = true;
        } else if (!std::memcmp(type, "IDAT", 4)) {
            z.insert(z.end(), data, data + len);
            ++*idats;
        } else if (!std::memcmp(type, "IEND", 4)) {
            iend = len == 0;
        }
        at += 12 + len;
    }
    if (!ihdr || !iend || at != file.size()) return false;
    const std::size_t stride = static_cast<std::size_t>(*w) * 3;
    std::vector<std::uint8_t> raw(*h * (stride + 1));
    uLongf raw_len = static_cast<uLongf>(raw.size());
    if (uncompress(raw.data(), &raw_len, z.data(), static_cast<uLong>(z.size())) != Z_OK || raw_len != raw.size()) return false;
    px->assign(*h * stride, 0);
    for (std::uint32_t y = 0; y < *h; ++y) {
        const std::uint8_t filter = raw[y * (stride + 1)];
        const std::uint8_t* in = &raw[y * (stride + 1) + 1];
        std::uint8_t* row = &(*px)[y * stride];
        const std::uint8_t* up = y ? &(*px)[(y - 1) * stride] : nullptr;
        for (std::size_t i = 0; i < stride; ++i) {
            const int a = i >= 3 ? row[i - 3] : 0, b = up ? up[i] : 0, c = up && i >= 3 ? up[i - 3] : 0;
            int pred;
            switch (filter) {
            case 0: pred = 0; break;
            case 1: pred = a; break;
            case 2: pred = b; break;
            case 3: pred = (a + b) / 2; break;
            case 4: pred = paeth(a, b, c); break;
            default: return false;
            }
            row[i] = static_cast<std::uint8_t>(in[i] + pred);
        }
    }
    return true;
}

}  // namespace

int main() {
    // The files are this process's own, in the temp folder: two test runs at
    // once from one build folder (its working folder) would share them.
    const std::string base = (std::filesystem::temp_directory_path() / ("bbhost-image-file-test-" + std::to_string(getpid()))).string();
    const std::string png_path = base + ".png", ppm_path = base + ".ppm";
    // Small and odd-sized, then one large enough for several IDAT chunks.
    struct Size {
        std::uint32_t w, h;
    };
    for (const Size sz : {Size{1, 1}, Size{37, 23}, Size{640, 480}}) {
        const std::vector<std::uint8_t> px = picture(sz.w, sz.h, sz.w * 7 + sz.h);
        CHECK(write(png_path, sz.w, sz.h, px));
        std::uint32_t w = 0, h = 0;
        std::vector<std::uint8_t> back;
        int idats = 0;
        const std::vector<std::uint8_t> file = slurp(png_path);
        CHECK(decode(file, &w, &h, &back, &idats));
        CHECK(w == sz.w && h == sz.h);
        CHECK(back == px);
        if (sz.w == 640) CHECK(idats > 1 && file.size() < px.size());
        std::printf("image_file_test: %ux%u png %zu bytes in %d IDAT\n", sz.w, sz.h, file.size(), idats);

        CHECK(write(ppm_path, sz.w, sz.h, px));
        const std::vector<std::uint8_t> ppm = slurp(ppm_path);
        const std::string head = "P6\n" + std::to_string(sz.w) + " " + std::to_string(sz.h) + "\n255\n";
        CHECK(ppm.size() == head.size() + px.size());
        CHECK(ppm.size() >= head.size() && std::memcmp(ppm.data(), head.data(), head.size()) == 0);
        CHECK(ppm.size() == head.size() + px.size() && std::memcmp(ppm.data() + head.size(), px.data(), px.size()) == 0);
    }
    // A file closed short of its rows is reported, and a folder that is not there fails at open.
    {
        ImageFile f;
        CHECK(f.open(png_path, 4, 4));
        const std::uint8_t row[12] = {};
        CHECK(f.row(row));
        CHECK(!f.close());
        CHECK(!f.open("no-such-folder/x.png", 4, 4));
    }
    std::remove(png_path.c_str());
    std::remove(ppm_path.c_str());
    if (g_fail) {
        std::printf("%d failed\n", g_fail);
        return 1;
    }
    std::printf("image_file_test: ok\n");
    return 0;
}
