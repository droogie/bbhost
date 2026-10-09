#include "engine/params.h"
#include "engine/param_lookup.h"

#include "core/config.h"
#include "core/elf.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "engine/paramdef.h"
#include "hle/fs.h"
#include "log.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kRepositorySlot = 0x5940340;  // SoloParamRepository_ptr
constexpr unsigned kSlots = 63, kCapsPerSlot = 8;
constexpr std::size_t kSlotStride = 0x48, kSlotsAt = 0x70;

std::uint64_t g_slide = 0;

template <typename T>
bool rd(std::uint64_t at, T* out) {
    return host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(at)), out, sizeof(T));
}

struct Table {
    std::string name;
    std::uint64_t cap = 0, file = 0;
    std::uint32_t rows = 0;
    std::uint8_t format = 0, flags = 0;
    bool overridden = false;
    std::shared_ptr<bbhost::param_detail::MissIndex> misses;
    std::string param_type;  // the file's type name at +0xc ("EQUIP_PARAM_WEAPON_ST"), which names its definition
};
std::mutex g_mu;
std::map<std::string, Table> g_tables;  // by name
std::uint64_t g_overrides_applied = 0, g_override_rows = 0, g_named_fields = 0;
// Tables whose definition's row size is the table's (named fields allowed),
// is not (refused), or which have no definition.
std::uint64_t g_defs_agree = 0, g_defs_disagree = 0, g_defs_missing = 0;
int g_ticks = 0;

// A cap's name: std::wstring at +0x18 (inline for fewer than 8 characters,
// else a pointer), as narrow text.
std::string cap_name(std::uint64_t cap) {
    std::uint64_t capacity = 0;
    if (!rd(cap + 0x30, &capacity)) return "";
    std::uint64_t at = cap + 0x18;
    if (capacity >= 8 && !rd(cap + 0x18, &at)) return "";
    std::string s;
    for (int i = 0; i < 64; ++i) {
        std::uint16_t c = 0;
        if (!rd(at + 2u * static_cast<unsigned>(i), &c) || !c) break;
        s.push_back(c < 128 ? static_cast<char>(c) : '?');
    }
    return s;
}

// The record for row index i: its id and where its data lies in the file.
bool record(const Table& t, std::uint32_t i, std::uint32_t* id, std::uint64_t* data) {
    if (i >= t.rows) return false;
    if (t.format == 2) {
        std::uint32_t off = 0;
        if (!rd(t.file + 0x34 + 0xc * i, id) || !rd(t.file + 0x34 + 0xc * i + 4, &off)) return false;
        *data = t.file + off;
    } else if (t.format <= 3 || !(t.flags & 2)) {
        std::uint32_t off = 0;
        if (!rd(t.file + 0x40 + 0xc * i, id) || !rd(t.file + 0x40 + 0xc * i + 4, &off)) return false;
        *data = t.file + off;
    } else {
        std::uint64_t off = 0;
        if (!rd(t.file + 0x40 + 0x18 * i, id) || !rd(t.file + 0x40 + 0x18 * i + 8, &off)) return false;
        *data = t.file + off;
    }
    return true;
}

// The row size, from the gap between the first two rows' data (0 when the
// table has one row).
std::size_t row_size(const Table& t) {
    std::uint32_t id0 = 0, id1 = 0;
    std::uint64_t d0 = 0, d1 = 0;
    if (t.rows < 2 || !record(t, 0, &id0, &d0) || !record(t, 1, &id1, &d1) || d1 <= d0) return 0;
    return static_cast<std::size_t>(d1 - d0);
}

bool parse_typed(const std::string& key, std::string* type, std::uint32_t* off) {
    const std::size_t us = key.find('_');
    if (us == std::string::npos) return false;
    *type = key.substr(0, us);
    *off = static_cast<std::uint32_t>(std::strtoul(key.c_str() + us + 1, nullptr, 0));
    return *type == "f32" || *type == "i32" || *type == "u32" || *type == "i16" || *type == "u16" || *type == "i8" || *type == "u8";
}

void write_field(std::uint64_t row, const std::string& type, std::uint32_t off, const std::string& text) {
    auto* p = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(row + off));
    if (type == "f32") {
        const float v = std::strtof(text.c_str(), nullptr);
        std::memcpy(p, &v, 4);
    } else if (type == "i32" || type == "u32") {
        const std::uint32_t v = static_cast<std::uint32_t>(std::strtoll(text.c_str(), nullptr, 0));
        std::memcpy(p, &v, 4);
    } else if (type == "i16" || type == "u16") {
        const std::uint16_t v = static_cast<std::uint16_t>(std::strtol(text.c_str(), nullptr, 0));
        std::memcpy(p, &v, 2);
    } else {
        *p = static_cast<std::uint8_t>(std::strtol(text.c_str(), nullptr, 0));
    }
}

// The table's definition when it can be trusted: its row size is the one the
// table's records are laid out with. nullptr otherwise.
const ParamDefinition* trusted_definition(const Table& t) {
    const ParamDefinition* def = t.param_type.empty() ? nullptr : paramdef_for(t.param_type);
    if (!def) return nullptr;
    const std::size_t size = row_size(t);
    return !size || size == def->row_bytes ? def : nullptr;
}

// <mods>/params/<Name>.toml onto the table's rows in place. A key is
// "row.<id>.<field>" or "row.all.<field>", the field either the definition's
// name ("attackBasePhysics", "name[2]" for an array element) or a raw
// "<type>_<offset>" ("f32_0x10").
void apply_override(Table& t) {
    const char* mods = hle_fs_mods_root();
    if (!mods || !mods[0]) return;
    const std::string path = std::string(mods) + "/params/" + t.name + ".toml";
    std::map<std::string, std::string> kv;
    std::string err;
    if (!config_parse_toml(path, &kv, &err)) return;  // no file: nothing to do
    const std::size_t size = row_size(t);
    const ParamDefinition* def = trusted_definition(t);
    struct Op {
        bool all = false;
        std::uint32_t id = 0;
        std::string raw_type;
        std::uint32_t raw_off = 0;
        const ParamField* field = nullptr;
        std::uint32_t index = 0;
        std::string value;
    };
    std::vector<Op> ops;
    for (const auto& [k, v] : kv) {
        if (k.rfind("row.", 0) != 0) continue;
        const std::size_t dot = k.find('.', 4);
        if (dot == std::string::npos) continue;
        Op op;
        const std::string which = k.substr(4, dot - 4);
        op.all = which == "all";
        op.id = static_cast<std::uint32_t>(std::strtoul(which.c_str(), nullptr, 0));
        op.value = v;
        const std::string rest = k.substr(dot + 1);
        if (parse_typed(rest, &op.raw_type, &op.raw_off)) {
            if (size && op.raw_off + 4 > size) continue;
            ops.push_back(std::move(op));
            continue;
        }
        std::string name = rest;
        if (const std::size_t br = name.find('['); br != std::string::npos) {
            op.index = static_cast<std::uint32_t>(std::strtoul(name.c_str() + br + 1, nullptr, 10));
            name.resize(br);
        }
        op.field = def ? def->field(name) : nullptr;
        if (!op.field) {
            host_log("params: %s: \"%s\" is not a field of %s%s", path.c_str(), rest.c_str(), t.param_type.c_str(),
                     def ? "" : " (no definition this table's rows match)");
            continue;
        }
        ops.push_back(std::move(op));
    }
    std::uint64_t written = 0, rows_touched = 0;
    for (std::uint32_t i = 0; i < t.rows; ++i) {
        std::uint32_t id = 0;
        std::uint64_t data = 0;
        if (!record(t, i, &id, &data)) continue;
        bool touched = false;
        for (const Op& op : ops) {
            if (!op.all && op.id != id) continue;
            if (op.field) {
                if (!paramdef_write(*op.field, reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(data)), op.value, op.index)) continue;
                ++g_named_fields;
            } else {
                write_field(data, op.raw_type, op.raw_off, op.value);
            }
            ++written;
            touched = true;
        }
        if (touched) ++rows_touched;
    }
    t.overridden = true;
    ++g_overrides_applied;
    g_override_rows += rows_touched;
    host_log("params: %s: %s applied, %llu fields over %llu rows (row size %zu)", t.name.c_str(), path.c_str(),
             static_cast<unsigned long long>(written), static_cast<unsigned long long>(rows_touched), size);
}

// BBHOST_PARAM_DUMP="EquipParamWeapon:1000,LockCamParam:0": each named row's
// fields, by the game's definition - what a <mods>/params file can change.
void dump_rows(const Table& t) {
    static const std::string spec = [] {
        const char* e = std::getenv("BBHOST_PARAM_DUMP");
        return std::string(e ? e : "");
    }();
    if (spec.empty()) return;
    std::size_t at = 0;
    while (at < spec.size()) {
        const std::size_t comma = spec.find(',', at);
        const std::string item = spec.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? spec.size() : comma + 1;
        const std::size_t colon = item.find(':');
        if (colon == std::string::npos || item.substr(0, colon) != t.name) continue;
        const std::uint32_t want = static_cast<std::uint32_t>(std::strtoul(item.c_str() + colon + 1, nullptr, 0));
        const ParamDefinition* def = trusted_definition(t);
        if (!def) {
            host_log("params: dump %s: no definition matches its rows", item.c_str());
            continue;
        }
        for (std::uint32_t i = 0; i < t.rows; ++i) {
            std::uint32_t id = 0;
            std::uint64_t data = 0;
            if (!record(t, i, &id, &data) || id != want) continue;
            const auto* row = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(data));
            host_log("params: %s row %u (%s, %zu fields, %u bytes):", t.name.c_str(), id, def->type.c_str(), def->fields.size(), def->row_bytes);
            for (const ParamField& f : def->fields) {
                if (f.type == "dummy8") continue;
                std::string value = paramdef_read(f, row);
                for (std::uint32_t k = 1; k < f.count && k < 16; ++k) value += " " + paramdef_read(f, row, k);
                host_log("  %s = %s    (%s%s, %s)", f.name.c_str(), value.c_str(), f.type.c_str(), f.bits ? ":bits" : "", f.display.c_str());
            }
        }
    }
}

void walk() {
    std::uint64_t repo = 0;
    if (!g_slide || !rd(g_slide + (kRepositorySlot - kPreferredGuestSlide), &repo) || !repo) return;
    for (unsigned slot = 0; slot < kSlots; ++slot) {
        std::int32_t count = 0;
        if (!rd(repo + kSlotsAt + slot * kSlotStride, &count) || count <= 0) continue;
        for (unsigned k = 0; k < static_cast<unsigned>(count) && k < kCapsPerSlot; ++k) {
            std::uint64_t cap = 0, holder = 0, file = 0;
            if (!rd(repo + kSlotsAt + slot * kSlotStride + 8 + 8 * k, &cap) || !cap) continue;
            if (!rd(cap + 0x70, &holder) || !holder || !rd(holder + 0x70, &file) || !file) continue;
            std::uint16_t rows = 0;
            std::uint8_t format = 0, flags = 0;
            char type[33] = {};
            if (!rd(file + 0xa, &rows) || !rd(file + 0x2d, &format) || !rd(file + 0x2e, &flags)) continue;
            host_read_safe(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(file + 0xc)), type, 32);
            const std::string name = cap_name(cap);
            if (name.empty()) continue;
            std::lock_guard<std::mutex> lk(g_mu);
            Table& t = g_tables[name];
            if (t.file == file && t.rows == rows) continue;  // seen, unchanged
            const bool reloaded = t.file != 0;
            t = Table{name, cap, file, rows, format, flags, false,
                      std::make_shared<bbhost::param_detail::MissIndex>(), std::string(type)};
            host_log("params: slot %u %s: %s, %u rows, format %u/%u at 0x%llx%s", slot, name.c_str(), type, rows, format, flags,
                     static_cast<unsigned long long>(file), reloaded ? " (reloaded)" : "");
            if (!reloaded) {
                const ParamDefinition* def = paramdef_for(t.param_type);
                const std::size_t size = row_size(t);
                if (!def) {
                    ++g_defs_missing;
                } else if (size && size != def->row_bytes) {
                    ++g_defs_disagree;
                    host_log("params: %s: its definition lays a row out in %u bytes, the table's rows are %zu apart: named fields refused",
                             name.c_str(), def->row_bytes, size);
                } else {
                    ++g_defs_agree;
                }
            }
            apply_override(t);
            dump_rows(t);
        }
    }
}

}  // namespace

void params_install(ElfImage* image) { g_slide = image->mem.slide; }

void params_tick() {
    // The repository fills during the boot; a walk every ~5 s (150 frames
    // at 30) sees new tables and reloads without costing the frame.
    if (g_ticks++ % 150 != 0) return;
    walk();
}

void* params_row(const char* table, std::uint32_t id, std::size_t* bytes) {
    Table t;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_tables.find(table ? table : "");
        if (it == g_tables.end()) return nullptr;
        t = it->second;
    }
    const auto data = bbhost::param_detail::lookup(id, t.rows,
        [&](std::uint32_t i, std::uint32_t* rid, std::uint64_t* address) { return record(t, i, rid, address); }, *t.misses);
    if (data && bytes) *bytes = row_size(t);
    return data ? reinterpret_cast<void*>(static_cast<std::uintptr_t>(data)) : nullptr;
}

bool params_ids(const char* table, std::uint32_t* ids, std::size_t max, std::size_t* count) {
    Table t;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_tables.find(table ? table : "");
        if (it == g_tables.end()) return false;
        t = it->second;
    }
    if (count) *count = t.rows;
    for (std::uint32_t i = 0; ids && i < t.rows && i < max; ++i) {
        std::uint64_t data = 0;
        if (!record(t, i, &ids[i], &data)) return false;
    }
    return true;
}

const ParamDefinition* params_definition(const char* table) {
    Table t;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_tables.find(table ? table : "");
        if (it == g_tables.end()) return nullptr;
        t = it->second;
    }
    return trusted_definition(t);
}

void params_report() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_tables.empty()) return;
    host_log("params: %zu tables seen, %llu override files applied over %llu rows (%llu fields by name); definitions: %llu match "
             "their table's rows, %llu do not, %llu tables have none (of %zu loaded)",
             g_tables.size(), static_cast<unsigned long long>(g_overrides_applied), static_cast<unsigned long long>(g_override_rows),
             static_cast<unsigned long long>(g_named_fields), static_cast<unsigned long long>(g_defs_agree),
             static_cast<unsigned long long>(g_defs_disagree), static_cast<unsigned long long>(g_defs_missing), paramdef_count());
}
