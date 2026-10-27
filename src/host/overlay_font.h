#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace overlay_font {
// Two texels of padding keep filtering inside each glyph's cell.
constexpr int kCellW = 28, kCellH = 32, kCols = 128;
constexpr int kAdvanceW = 12, kAdvanceH = 24, kOriginX = 2, kOriginY = 2;
constexpr int kAtlasW = kCellW * kCols, kAtlasH = 4096;
constexpr int kSolidX = kAtlasW - 2, kSolidY = kAtlasH - 2;
struct Glyph { int index; float advance; };
const std::vector<std::uint8_t>& atlas();
Glyph glyph(std::uint32_t codepoint);
// Decodes one UTF-8 scalar; malformed input becomes U+FFFD, always advancing.
std::uint32_t next(const char*& text);
float width(float scale, const char* text);
std::string wrap(float scale, float max_width, const char* text);
}
