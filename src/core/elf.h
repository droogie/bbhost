#pragma once

#include "core/memory.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct ImportReloc {
    std::uint64_t got_va = 0;
    std::string nid;
    std::string name;
};

// A PT_LOAD of the image, at its ELF address (runtime = mem.slide + va).
struct ElfSegment {
    std::uint64_t va = 0, filesz = 0, memsz = 0;
    std::uint32_t flags = 0;  // PF_X 1, PF_W 2, PF_R 4
};

struct ElfImage {
    GuestMemory mem;
    std::uint64_t entry = 0;
    std::uint64_t init = 0;
    std::uint64_t fini = 0;
    std::vector<ImportReloc> imports;
    std::string sha256;  // hex digest of the ELF file, gates address-based patches
    std::vector<ElfSegment> segments;
    // The R_X86_64_RELATIVE relocations the loader applied: the slot and the
    // address it was given, both ELF addresses (engine/image_text.h moves
    // the pointers to a string).
    std::vector<std::pair<std::uint32_t, std::uint32_t>> relative;
    // Room after the last segment, inside mem and so within a rip-relative
    // reach of all of the code: text that replaces the game's own
    // (engine/image_text.h). ELF addresses; read-write, never executed.
    std::uint64_t tail_va = 0, tail_size = 0, tail_used = 0;
    // After the tail, the routines of core/legacy_cpu.h when the processor
    // needs them (arena_size 0 when not): inside mem too, so in reach of all
    // of the code and inside every range that counts as the guest's. ELF
    // addresses.
    std::uint64_t arena_va = 0, arena_size = 0;
};

// arena: the bytes to reserve after the tail for core/legacy_cpu.h's routines
// (legacy_cpu_arena_size()).
bool load_orbis_elf(const char* path, ElfImage* image, std::size_t arena = 0);
void unload_elf(ElfImage* image);
