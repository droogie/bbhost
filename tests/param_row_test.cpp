// Synthetic guest memory exercises the public registration and row lookup path.
// No eboot, game assets, configuration or mapped guest address space is needed.
#include "core/elf.h"
#include "engine/params.h"
#include "engine/paramdef.h"
#include "engine/param_lookup.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

namespace {
std::map<std::uint64_t, std::vector<std::uint8_t>> memory;
constexpr std::uint64_t repo = 0x10000000, cap = 0x10010000, holder = 0x10020000;
constexpr std::uint64_t file = 0x10030000, replacement = 0x10040000;

template<class T> void put(std::uint64_t base, std::size_t offset, T value) {
    auto& block = memory.at(base);
    CHECK(offset + sizeof(value) <= block.size());
    std::memcpy(block.data() + offset, &value, sizeof(value));
}

void make_file(std::uint64_t base, std::uint8_t format, std::uint8_t flags,
               const std::vector<std::uint32_t>& ids) {
    memory[base] = std::vector<std::uint8_t>(0x400, 0);
    put(base, 0xa, static_cast<std::uint16_t>(ids.size()));
    put(base, 0x2d, format);
    put(base, 0x2e, flags);
    const bool wide = format > 3 && (flags & 2);
    const std::size_t directory = format == 2 ? 0x34 : 0x40;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto at = directory + i * (wide ? 0x18 : 0xc);
        put(base, at, ids[i]);
        if (wide) put(base, at + 8, std::uint64_t(0x200 + i * 8));
        else put(base, at + 4, std::uint32_t(0x200 + i * 8));
        put(base, 0x200 + i * 8, std::uint32_t(100 + i));
    }
}

void register_file(std::uint64_t base) {
    put(holder, 0x70, base);
    // Registration walks on every 150th tick; this includes exactly one walk.
    for (unsigned i = 0; i < 150; ++i) params_tick();
}

void check_row(std::uint64_t base, std::uint32_t id, unsigned index) {
    std::size_t bytes = 777;
    const auto expected = reinterpret_cast<void*>(base + 0x200 + index * 8);
    CHECK(params_row("Test", id, &bytes) == expected);
    CHECK(bytes == 8);
    CHECK(params_row("Test", id, nullptr) == expected);
}

void public_lookup() {
    memory[0x5940340] = std::vector<std::uint8_t>(8);
    put(0x5940340, 0, repo);
    memory[repo] = std::vector<std::uint8_t>(0x1300);
    memory[cap] = std::vector<std::uint8_t>(0x80);
    memory[holder] = std::vector<std::uint8_t>(0x80);
    put(repo, 0x70, std::int32_t(1));
    put(repo, 0x78, cap);
    put(cap, 0x70, holder);
    for (unsigned i = 0; i < 4; ++i) put(cap, 0x18 + i * 2, std::uint16_t("Test"[i]));
    ElfImage image;
    image.mem.slide = 0x400000;
    params_install(&image);

    make_file(replacement, 4, 2, {10, 10, 10, 20});
    register_file(replacement);
    check_row(replacement, 10, 2); // Retain the original midpoint duplicate.

    // Cover each directory layout and the format > 3, narrow-offset branch.
    for (const auto [format, flags] : std::vector<std::pair<std::uint8_t, std::uint8_t>>{
             {2, 0}, {3, 0}, {4, 0}, {4, 2}}) {
        make_file(file, format, flags, {30, 10, 20, 10});
        register_file(file);
        const auto before = memory.at(file);
        check_row(file, 30, 0); // Binary search misses a present unsorted id.
        check_row(file, 10, 1); // Binary hit preserves its duplicate choice.
        std::size_t bytes = 777;
        CHECK(params_row("Test", 999, &bytes) == nullptr);
        CHECK(bytes == 777);
        CHECK(params_row("Missing", 30, &bytes) == nullptr);
        CHECK(params_row(nullptr, 30, &bytes) == nullptr);
        CHECK(bytes == 777);
        CHECK(memory.at(file) == before);
        // Force a new registration even when the next format reuses file.
        make_file(replacement, format, flags, {40, 50});
        register_file(replacement);
        check_row(replacement, 40, 0);
        CHECK(params_row("Test", 30, nullptr) == nullptr);
    }

    make_file(file, 4, 2, {30, 30, 10, 20});
    register_file(file);
    check_row(file, 30, 0); // Miss fallback chooses first directory duplicate.
    // A changed count at the same address must discard the old miss index.
    make_file(file, 4, 2, {70, 10, 20});
    register_file(file);
    check_row(file, 70, 0);
    CHECK(params_row("Test", 30, nullptr) == nullptr);
    make_file(replacement, 4, 2, {90, 10, 20});
    register_file(replacement); // Changed file, unchanged count.
    check_row(replacement, 90, 0);
    CHECK(params_row("Test", 70, nullptr) == nullptr);
}

void cache_lookup() {
    bbhost::param_detail::MissIndex cache;
    const std::uint32_t ids[] = {30, 10, 20, 10};
    unsigned reads = 0;
    bool fail = true;
    auto reader = [&](std::uint32_t i, std::uint32_t* id, std::uint64_t* data) {
        ++reads;
        if (fail && i == 1) return false; // Scan fails after its first entry; binary search visits 2 and 3.
        *id = ids[i]; *data = 100 + i;
        return true;
    };
    CHECK(bbhost::param_detail::lookup(30, 4, reader, cache) == 0);
    CHECK(!cache.ready);
    CHECK(cache.first.empty());
    fail = false;
    CHECK(bbhost::param_detail::lookup(30, 4, reader, cache) == 100);
    CHECK(cache.ready);
    reads = 0;
    CHECK(bbhost::param_detail::lookup(30, 4, reader, cache) == 100);
    CHECK(reads == 2); // Only the original binary search repeats.
    reads = 0;
    CHECK(bbhost::param_detail::lookup(999, 4, reader, cache) == 0);
    CHECK(reads == 2);
}
} // namespace

bool host_read_safe(const void* address, void* out, std::size_t bytes) {
    const auto at = reinterpret_cast<std::uintptr_t>(address);
    for (const auto& [base, data] : memory) {
        if (at >= base && at - base <= data.size() && bytes <= data.size() - (at - base)) {
            std::memcpy(out, data.data() + (at - base), bytes);
            return true;
        }
    }
    return false;
}

// These collaborators are intentionally inert: registration needs neither
// an override file nor a definition archive for the synthetic tables.
const char* hle_fs_mods_root() { return ""; }
bool config_parse_toml(const std::string&, std::map<std::string, std::string>*, std::string*) { return false; }
const ParamDefinition* paramdef_for(const std::string&) { return nullptr; }
std::size_t paramdef_count() { return 0; }
const ParamField* ParamDefinition::field(const std::string&) const { return nullptr; }
std::string paramdef_read(const ParamField&, const std::uint8_t*, std::uint32_t) { return {}; }
bool paramdef_write(const ParamField&, std::uint8_t*, const std::string&, std::uint32_t) { return false; }

int main() {
    public_lookup();
    cache_lookup();
    std::puts("param_row_test: passed");
}
