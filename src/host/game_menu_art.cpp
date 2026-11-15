#include "host/game_menu_art.h"

#if !defined(BBHOST_GAME_MENU_ART_NO_FS)
#include "hle/fs.h"
#include "log.h"
#endif

#include <zlib.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

namespace game_menu_art {

namespace {

std::uint32_t u32(const std::vector<std::uint8_t>& b, std::size_t i) {
    return static_cast<std::uint32_t>(b[i]) | static_cast<std::uint32_t>(b[i + 1]) << 8 | static_cast<std::uint32_t>(b[i + 2]) << 16 |
           static_cast<std::uint32_t>(b[i + 3]) << 24;
}
std::uint16_t u16(const std::vector<std::uint8_t>& b, std::size_t i) {
    return static_cast<std::uint16_t>(b[i] | b[i + 1] << 8);
}

// ---- BC7 (the format's tables; the partitions and anchors as Pillow's
// decoder has them, which tests/game_menu_art_test.cpp's vectors come from)

struct Mode {
    int subsets, partition_bits, rotation_bits, index_select_bits, colour_bits, alpha_bits, endpoint_pbits, shared_pbits,
        index_bits, index2_bits;
};
constexpr Mode kModes[8] = {
    {3, 4, 0, 0, 4, 0, 1, 0, 3, 0}, {2, 6, 0, 0, 6, 0, 0, 1, 3, 0}, {3, 6, 0, 0, 5, 0, 0, 0, 2, 0},
    {2, 6, 0, 0, 7, 0, 1, 0, 2, 0}, {1, 0, 2, 1, 5, 6, 0, 0, 2, 3}, {1, 0, 2, 0, 7, 8, 0, 0, 2, 2},
    {1, 0, 0, 0, 7, 7, 1, 0, 4, 0}, {2, 6, 0, 0, 5, 5, 1, 0, 2, 0},
};
constexpr std::uint16_t kPartition2[64] = {
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80,
    0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
    0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
    0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
};
constexpr char kPartition3[64][17] = {
    "0011001102212222", "0001001122112221", "0000200122112211", "0222002200110111",
    "0000000011221122", "0011001100220022", "0022002211111111", "0011001122112211",
    "0000000011112222", "0000111111112222", "0000111122222222", "0012001200120012",
    "0112011201120112", "0122012201220122", "0011011211221222", "0011200122002220",
    "0001001101121122", "0111001120012200", "0000112211221122", "0022002200221111",
    "0111011102220222", "0001000122212221", "0000001101220122", "0000110022102210",
    "0122012200110000", "0012001211222222", "0110122112210110", "0000011012211221",
    "0022110211020022", "0110011020022222", "0011012201220011", "0000200022112221",
    "0000000211221222", "0222002200120011", "0011001200220222", "0120012001200120",
    "0000111122220000", "0120120120120120", "0120201212010120", "0011220011220011",
    "0011112222000011", "0101010122222222", "0000000021212121", "0022112200221122",
    "0022001100220011", "0220122102201221", "0101222222220101", "0000212121212121",
    "0101010101012222", "0222011102220111", "0002111200021112", "0000211221122112",
    "0222011101110222", "0002111211120002", "0110011001102222", "0000000021122112",
    "0110011022222222", "0022001100110022", "0022112211220022", "0000000000002112",
    "0002000100020001", "0222122202221222", "0101222222222222", "0111201122012220",
};
constexpr std::uint8_t kAnchor2[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
    6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
};
constexpr std::uint8_t kAnchor3a[64] = {
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3,
    3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
    8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15,
    3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
};
constexpr std::uint8_t kAnchor3b[64] = {
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8,
    15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
    15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8,
    15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
};
constexpr std::uint8_t kWeights2[4] = {0, 21, 43, 64};
constexpr std::uint8_t kWeights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
constexpr std::uint8_t kWeights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

// The block's bits, least significant first.
struct Bits {
    const std::uint8_t* b;
    int at;
    std::uint32_t take(int n) {
        std::uint32_t v = 0;
        for (int i = 0; i < n; ++i, ++at) v |= static_cast<std::uint32_t>((b[at >> 3] >> (at & 7)) & 1) << i;
        return v;
    }
};

int subset_of(int subsets, int partition, int pixel) {
    if (subsets == 2) return (kPartition2[partition] >> pixel) & 1;
    if (subsets == 3) return kPartition3[partition][pixel] - '0';
    return 0;
}

int anchor_of(int subsets, int partition, int subset) {
    if (subset == 0) return 0;
    if (subsets == 2) return kAnchor2[partition];
    return subset == 1 ? kAnchor3a[partition] : kAnchor3b[partition];
}

std::uint8_t expand(std::uint32_t v, int bits) {
    v <<= 8 - bits;
    return static_cast<std::uint8_t>(v | (v >> bits));
}

std::uint8_t weight(int bits, std::uint32_t index) {
    return bits == 2 ? kWeights2[index] : bits == 3 ? kWeights3[index] : kWeights4[index];
}

std::uint8_t mix(std::uint8_t e0, std::uint8_t e1, std::uint8_t w) {
    return static_cast<std::uint8_t>(((64 - w) * e0 + w * e1 + 32) >> 6);
}

}  // namespace

void decode_bc7(const std::uint8_t block[16], std::uint8_t out[64]) {
    int mode = 0;
    while (mode < 8 && !(block[0] & (1 << mode))) ++mode;
    if (mode == 8) {
        std::memset(out, 0, 64);
        return;
    }
    const Mode& m = kModes[mode];
    Bits r{block, mode + 1};
    const int partition = static_cast<int>(r.take(m.partition_bits));
    const int rotation = static_cast<int>(r.take(m.rotation_bits));
    const int index_select = static_cast<int>(r.take(m.index_select_bits));
    std::uint32_t ep[3][2][4] = {};  // subset, endpoint, channel
    for (int c = 0; c < 3; ++c)
        for (int s = 0; s < m.subsets; ++s)
            for (int e = 0; e < 2; ++e) ep[s][e][c] = r.take(m.colour_bits);
    if (m.alpha_bits)
        for (int s = 0; s < m.subsets; ++s)
            for (int e = 0; e < 2; ++e) ep[s][e][3] = r.take(m.alpha_bits);
    int colour_bits = m.colour_bits, alpha_bits = m.alpha_bits;
    if (m.endpoint_pbits) {
        for (int s = 0; s < m.subsets; ++s)
            for (int e = 0; e < 2; ++e) {
                const std::uint32_t p = r.take(1);
                for (int c = 0; c < 4; ++c) ep[s][e][c] = (ep[s][e][c] << 1) | p;
            }
        ++colour_bits;
        if (alpha_bits) ++alpha_bits;
    }
    if (m.shared_pbits) {
        for (int s = 0; s < m.subsets; ++s) {
            const std::uint32_t p = r.take(1);
            for (int e = 0; e < 2; ++e)
                for (int c = 0; c < 3; ++c) ep[s][e][c] = (ep[s][e][c] << 1) | p;
        }
        ++colour_bits;
    }
    std::uint8_t col[3][2][4];
    for (int s = 0; s < m.subsets; ++s)
        for (int e = 0; e < 2; ++e) {
            for (int c = 0; c < 3; ++c) col[s][e][c] = expand(ep[s][e][c], colour_bits);
            col[s][e][3] = alpha_bits ? expand(ep[s][e][3], alpha_bits) : 255;
        }
    std::uint32_t index[16], index2[16] = {};
    for (int i = 0; i < 16; ++i) {
        const int s = subset_of(m.subsets, partition, i);
        index[i] = r.take(m.index_bits - (i == anchor_of(m.subsets, partition, s) ? 1 : 0));
    }
    if (m.index2_bits)
        for (int i = 0; i < 16; ++i) index2[i] = r.take(m.index2_bits - (i == 0 ? 1 : 0));
    for (int i = 0; i < 16; ++i) {
        const int s = subset_of(m.subsets, partition, i);
        std::uint8_t* px = out + i * 4;
        if (m.index2_bits) {
            const bool swap = index_select != 0;
            const std::uint8_t wc = swap ? weight(m.index2_bits, index2[i]) : weight(m.index_bits, index[i]);
            const std::uint8_t wa = swap ? weight(m.index_bits, index[i]) : weight(m.index2_bits, index2[i]);
            for (int c = 0; c < 3; ++c) px[c] = mix(col[s][0][c], col[s][1][c], wc);
            px[3] = mix(col[s][0][3], col[s][1][3], wa);
        } else {
            const std::uint8_t w = weight(m.index_bits, index[i]);
            for (int c = 0; c < 4; ++c) px[c] = mix(col[s][0][c], col[s][1][c], w);
        }
        if (rotation) std::swap(px[3], px[rotation - 1]);
    }
}

bool inflate_dcx(const std::vector<std::uint8_t>& file, std::vector<std::uint8_t>& out) {
    if (file.size() < 0x4c || std::memcmp(file.data(), "DCX\0", 4) != 0) return false;
    std::size_t at = 0;
    for (std::size_t i = 4; i + 8 <= file.size() && i < 0x100; ++i) {
        if (std::memcmp(file.data() + i, "DCA\0", 4) == 0) {
            at = i + 8;
            break;
        }
    }
    if (!at) return false;
    z_stream z{};
    if (inflateInit(&z) != Z_OK) return false;
    z.next_in = const_cast<Bytef*>(file.data() + at);
    z.avail_in = static_cast<uInt>(file.size() - at);
    out.clear();
    std::uint8_t chunk[1 << 16];
    int rc = Z_OK;
    while (rc == Z_OK) {
        z.next_out = chunk;
        z.avail_out = sizeof(chunk);
        rc = inflate(&z, Z_NO_FLUSH);
        if (rc != Z_OK && rc != Z_STREAM_END) break;
        out.insert(out.end(), chunk, chunk + (sizeof(chunk) - z.avail_out));
    }
    inflateEnd(&z);
    return rc == Z_STREAM_END;
}

bool parse_tpf(const std::vector<std::uint8_t>& tpf, std::vector<TpfTexture>& out) {
    out.clear();
    if (tpf.size() < 0x10 || std::memcmp(tpf.data(), "TPF\0", 4) != 0 || tpf[12] != 4) return false;
    const std::uint32_t count = u32(tpf, 8);
    if (count > 4096 || 0x10 + static_cast<std::size_t>(count) * 0x24 > tpf.size()) return false;
    for (std::uint32_t k = 0; k < count; ++k) {
        const std::size_t o = 0x10 + static_cast<std::size_t>(k) * 0x24;
        TpfTexture t;
        t.offset = u32(tpf, o);
        t.size = u32(tpf, o + 4);
        t.format = tpf[o + 8];
        t.width = u16(tpf, o + 12);
        t.height = u16(tpf, o + 14);
        std::size_t name = u32(tpf, o + 0x18);
        if (t.offset + t.size > tpf.size() || name >= tpf.size()) return false;
        for (; name + 1 < tpf.size() && (tpf[name] || tpf[name + 1]); name += 2) {
            const std::uint16_t ch = u16(tpf, name);
            t.name.push_back(ch < 0x80 ? static_cast<char>(ch) : '?');
        }
        out.push_back(std::move(t));
    }
    return true;
}

std::size_t tiled_block_offset(int bx, int by, int blocks_w, int bytes_per_block) {
    const int tiles_w = (blocks_w + 7) / 8;
    const int tile = (by / 8) * tiles_w + bx / 8;
    int morton = 0;
    for (int bit = 0; bit < 3; ++bit) {
        morton |= ((bx >> bit) & 1) << (2 * bit);
        morton |= ((by >> bit) & 1) << (2 * bit + 1);
    }
    return (static_cast<std::size_t>(tile) * 64 + static_cast<std::size_t>(morton)) * static_cast<std::size_t>(bytes_per_block);
}

bool decode_region(const std::uint8_t* data, std::size_t size, int width, int height, int x0, int y0, int x1, int y1,
                   std::vector<std::uint8_t>& out) {
    if (x0 < 0 || y0 < 0 || x1 > width || y1 > height || x0 >= x1 || y0 >= y1) return false;
    const int bw = (width + 3) / 4, bh = (height + 3) / 4;
    const std::size_t need = static_cast<std::size_t>((bw + 7) / 8 * 8) * static_cast<std::size_t>((bh + 7) / 8 * 8) * 16;
    if (size < need) return false;
    const int w = x1 - x0, h = y1 - y0;
    out.assign(static_cast<std::size_t>(w) * h * 4, 0);
    std::uint8_t px[64];
    for (int by = y0 / 4; by <= (y1 - 1) / 4; ++by) {
        for (int bx = x0 / 4; bx <= (x1 - 1) / 4; ++bx) {
            decode_bc7(data + tiled_block_offset(bx, by, bw, 16), px);
            for (int j = 0; j < 4; ++j) {
                const int y = by * 4 + j;
                if (y < y0 || y >= y1) continue;
                for (int i = 0; i < 4; ++i) {
                    const int x = bx * 4 + i;
                    if (x < x0 || x >= x1) continue;
                    std::memcpy(&out[(static_cast<std::size_t>(y - y0) * w + (x - x0)) * 4], px + (j * 4 + i) * 4, 4);
                }
            }
        }
    }
    return true;
}

std::vector<std::uint8_t> halve(const std::vector<std::uint8_t>& rgba, int w, int h) {
    const int hw = w / 2, hh = h / 2;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(hw) * hh * 4, 0);
    for (int y = 0; y < hh; ++y)
        for (int x = 0; x < hw; ++x) {
            int sum[4] = {0, 0, 0, 0};
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    const std::uint8_t* p = &rgba[(static_cast<std::size_t>(y * 2 + j) * w + (x * 2 + i)) * 4];
                    for (int c = 0; c < 3; ++c) sum[c] += p[c] * p[3];
                    sum[3] += p[3];
                }
            std::uint8_t* o = &out[(static_cast<std::size_t>(y) * hw + x) * 4];
            for (int c = 0; c < 3; ++c) o[c] = static_cast<std::uint8_t>(sum[3] ? (sum[c] + sum[3] / 2) / sum[3] : 0);
            o[3] = static_cast<std::uint8_t>((sum[3] + 2) / 4);
        }
    return out;
}

void additive_to_alpha(std::uint8_t* rgba, std::size_t pixels) {
    for (std::size_t i = 0; i < pixels; ++i) {
        std::uint8_t* p = rgba + i * 4;
        const int a = std::max({p[0], p[1], p[2]});
        if (a == 0) {
            p[0] = p[1] = p[2] = p[3] = 0;
            continue;
        }
        for (int c = 0; c < 3; ++c) p[c] = static_cast<std::uint8_t>(std::min(255, (p[c] * 255 + a / 2) / a));
        p[3] = static_cast<std::uint8_t>(a);
    }
}

// The options dialog's art (menu/optionsetting.gfx's shapes fill from these
// regions of the textures its DefineExternalImage2 tags name); the slate the
// game's other menus (the inventory, equipment, level up) lay their panels on,
// MENU_Common_00002's landscape one, as the dialog itself has none (it dims
// the screen behind it) and MENU_Common_00003's portrait slate fades out to
// its left; and the key guide's icons whole, KG_OK drawn as the circle and
// KG_Cancel as the cross.
const Source& source(Piece p) {
    static const Source kSources[kPieceCount] = {
        {"MENU_Common_00003", 75, 57, 406, 85, true},       // kOrnament: crest 194..286, the rule at y 83..84
        {"MENU_Common_00003", 52, 85, 429, 106, true},      // kBand
        {"MENU_Common_00003", 110, 4, 269, 46, true},       // kFiligree: its finial at x 188..190, the rule at y 29..30
        {"MENU_Common_00003", 40, 27, 116, 33, true},       // kRule: the rule at y 29..30
        {"MENU_Common_00003", 20, 238, 185, 242, true},     // kLabelLine
        {"MENU_Common_00003", 75, 74, 121, 85, true},       // kValueEnd: the ornament's left end, its rule at y 83..84
        {"MENU_Common_00003", 7, 814, 364, 908, true},      // kGlow: brightest at x 359
        {"MENU_Common_00003", 197, 110, 210, 133, false},   // kArrow
        {"MENU_Common_00002", 0, 0, 1154, 676, false, 2},  // kSlate: the menus' landscape slate, opaque
        {"MENU_Common_00001", 1566, 1533, 1877, 1547, false},  // kSliderTrack
        {"MENU_Common_00001", 1566, 1551, 1871, 1558, false},  // kSliderFill
        {"KG_OK", 0, 0, 32, 32, false},
        {"KG_Cancel", 0, 0, 32, 32, false},
        {"KG_R_U", 0, 0, 32, 32, false},
        {"KG_R_L", 0, 0, 32, 32, false},
        {"KG_L1", 0, 0, 32, 32, false},
        {"KG_R1", 0, 0, 32, 32, false},
        {"KG_L2", 0, 0, 32, 32, false},
        {"KG_R2", 0, 0, 32, 32, false},
        {"KG_L3", 0, 0, 32, 32, false},
        {"KG_R3", 0, 0, 32, 32, false},
        {"KG_L_UDLR", 0, 0, 32, 32, false},
        {"KG_L_UD", 0, 0, 32, 32, false},
        {"KG_L_LR", 0, 0, 32, 32, false},
    };
    return kSources[p];
}

const Slices& slices(Piece p) {
    // kOrnament: its scrolled ends (75..119, 361..406 of the texture), the
    // rule between them and the crest's curls (166..314, the crest itself
    // 194..286); kBand: its faded ends and its even middle; others whole.
    static const Slices kOrnamentSlices{{0, 44, 91, 239, 286, 331}};
    static const Slices kBandSlices{{0, 48, 188, 189, 329, 377}};
    static const Slices kWhole{{0, 0, 0, 0, 0, 0}};
    return p == kOrnament ? kOrnamentSlices : p == kBand ? kBandSlices : kWhole;
}

bool stretch(const Slices& s, float width, float scale, Span& out) {
    const float fixed = ((s.x[1] - s.x[0]) + (s.x[3] - s.x[2]) + (s.x[5] - s.x[4])) * scale;
    const float each = (width - fixed) * 0.5f;
    if (each < 0.0f || scale <= 0.0f) return false;
    const float w[5] = {(s.x[1] - s.x[0]) * scale, each, (s.x[3] - s.x[2]) * scale, each, (s.x[5] - s.x[4]) * scale};
    float x = 0.0f;
    for (int i = 0; i < 5; ++i) {
        out.x0[i] = x;
        x += w[i];
        out.x1[i] = x;
    }
    return true;
}

bool pack(const std::vector<std::vector<std::uint8_t>>& pixels, const std::vector<int>& widths, const std::vector<int>& heights,
          int width, int height, Atlas& out) {
    constexpr int kGap = 2;
    const std::size_t n = std::min({pixels.size(), widths.size(), heights.size(), static_cast<std::size_t>(kPieceCount)});
    out = Atlas{};
    out.width = width;
    out.height = height;
    out.rgba.assign(static_cast<std::size_t>(width) * height * 4, 0);
    // Tallest first, in shelves left to right.
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < n; ++i)
        if (widths[i] > 0 && heights[i] > 0) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return heights[a] > heights[b]; });
    int x = kGap, y = kGap, shelf = 0;
    for (const std::size_t i : order) {
        const int w = widths[i], h = heights[i];
        if (pixels[i].size() < static_cast<std::size_t>(w) * h * 4) return false;
        if (x + w + kGap > width) {
            x = kGap;
            y += shelf + kGap;
            shelf = 0;
        }
        if (x + w + kGap > width || y + h + kGap > height) return false;
        // The piece, and its edge repeated a gap's width around it.
        for (int j = -kGap; j < h + kGap; ++j) {
            const int sj = std::clamp(j, 0, h - 1);
            for (int k = -kGap; k < w + kGap; ++k) {
                const int sk = std::clamp(k, 0, w - 1);
                std::memcpy(&out.rgba[(static_cast<std::size_t>(y + j) * width + (x + k)) * 4],
                            &pixels[i][(static_cast<std::size_t>(sj) * w + sk) * 4], 4);
            }
        }
        Placed& p = out.piece[i];
        p.present = true;
        p.w = w;
        p.h = h;
        p.u0 = static_cast<float>(x) / width;
        p.v0 = static_cast<float>(y) / height;
        p.u1 = static_cast<float>(x + w) / width;
        p.v1 = static_cast<float>(y + h) / height;
        x += w + kGap * 2;
        shelf = std::max(shelf, h + kGap);
    }
    return true;
}

bool build(const std::vector<std::uint8_t>& tpf, Atlas& out, std::string* why) {
    std::vector<TpfTexture> textures;
    if (!parse_tpf(tpf, textures)) {
        if (why) *why = "not a PS4 TPF";
        return false;
    }
    std::vector<std::vector<std::uint8_t>> pixels(kPieceCount);
    std::vector<int> widths(kPieceCount, 0), heights(kPieceCount, 0);
    int found = 0;
    for (int p = 0; p < kPieceCount; ++p) {
        const Source& s = source(static_cast<Piece>(p));
        const auto t = std::find_if(textures.begin(), textures.end(), [&](const TpfTexture& x) { return x.name == s.texture; });
        if (t == textures.end() || t->format != 102) continue;
        if (!decode_region(tpf.data() + t->offset, t->size, t->width, t->height, s.x0, s.y0, s.x1, s.y1, pixels[p])) continue;
        if (s.additive) additive_to_alpha(pixels[p].data(), pixels[p].size() / 4);
        widths[p] = s.x1 - s.x0;
        heights[p] = s.y1 - s.y0;
        if (s.shrink == 2) {
            pixels[p] = halve(pixels[p], widths[p], heights[p]);
            widths[p] /= 2;
            heights[p] /= 2;
        }
        ++found;
    }
    if (!found) {
        if (why) *why = "none of the pieces' textures in it";
        return false;
    }
    if (!pack(pixels, widths, heights, 1024, 512, out)) {
        if (why) *why = "the pieces do not fit the atlas";
        return false;
    }
    return true;
}

#if !defined(BBHOST_GAME_MENU_ART_NO_FS)
namespace {

std::once_flag g_once;
// Never freed: a preload's thread may still be filling it as the process ends.
Atlas* g_atlas = nullptr;

void load() {
    if (const char* e = std::getenv("BBHOST_MENU_ART"); e && e[0] == '0') {
        host_log("menu: the game's menu art left out (BBHOST_MENU_ART=0): a plugin's panel is drawn without it");
        return;
    }
    const std::string path = hle_fs_map_path_base("/app0/dvdroot_ps4/menu/common.tpf.dcx");
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        host_log("menu: no menu/common.tpf.dcx in the dump; a plugin's panel is drawn without the game's art");
        return;
    }
    std::vector<std::uint8_t> file(static_cast<std::size_t>(std::max<std::streamoff>(0, f.tellg())));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(file.size()));
    std::vector<std::uint8_t> tpf;
    std::string why;
    auto* atlas = new Atlas;
    if (!f || !inflate_dcx(file, tpf)) {
        why = "it does not inflate";
    } else if (build(tpf, *atlas, &why)) {
        int n = 0;
        for (const Placed& p : atlas->piece) n += p.present;
        g_atlas = atlas;
        host_log("menu: the game's menu art from menu/common.tpf.dcx: %d of %d pieces in a %dx%d texture", n,
                 static_cast<int>(kPieceCount), atlas->width, atlas->height);
        return;
    }
    delete atlas;
    host_log("menu: menu/common.tpf.dcx is not menu art this reads (%s); a plugin's panel is drawn without it", why.c_str());
}

}  // namespace

const Atlas* menu_art() {
    std::call_once(g_once, load);
    return g_atlas;
}

void preload() {
    static std::atomic<bool> started{false};
    if (started.exchange(true)) return;
    std::thread([] { menu_art(); }).detach();
}
#else
const Atlas* menu_art() { return nullptr; }  // (the test reads the file itself)
void preload() {}
#endif

}  // namespace game_menu_art

#if !defined(BBHOST_GAME_MENU_ART_NO_FS) && defined(BBHOST_HAVE_SDL3)
#include "imgui.h"
#include "imgui_internal.h"

ImTextureData* game_menu_art_texture() {
    const game_menu_art::Atlas* art = game_menu_art::menu_art();
    if (!art) return nullptr;
    static ImTextureData* tex = nullptr;
    if (!tex) {
        tex = IM_NEW(ImTextureData)();
        tex->Create(ImTextureFormat_RGBA32, art->width, art->height);
        std::memcpy(tex->GetPixels(), art->rgba.data(), art->rgba.size());
        tex->UseColors = true;
    }
    // Once a context: the in-game menus remade with the swapchain (a new
    // context and backend, which may well sit where the old one did; the old
    // backend let the texture go with it) make it again from its pixels.
    if (ImGuiContext* now = ImGui::GetCurrentContext(); now && !now->UserTextures.contains(tex)) {
        tex->RefCount = 0;
        tex->SetStatus(ImTextureStatus_WantCreate);
        ImGui::RegisterUserTexture(tex);
    }
    return tex;
}
#else
ImTextureData* game_menu_art_texture() { return nullptr; }
#endif
