#include "core/elf.h"
#include "core/imports.h"
#include "core/sha256.h"
#include "core/tls_rewrite.h"
#include "log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

constexpr std::uint32_t PT_LOAD = 1;
constexpr std::uint32_t PT_SCE_DYNLIBDATA = 0x61000000;
constexpr std::uint32_t PT_SCE_PROC_PARAM = 0x61000001;
constexpr std::uint32_t PF_X = 1;

constexpr std::uint64_t DT_NULL = 0;
constexpr std::uint64_t DT_INIT = 12;
constexpr std::uint64_t DT_FINI = 13;
constexpr std::uint64_t DT_SCE_STRTAB = 0x6100003d;
constexpr std::uint64_t DT_SCE_SYMTAB = 0x61000039;
constexpr std::uint64_t DT_SCE_SYMENT = 0x61000035;
constexpr std::uint64_t DT_SCE_JMPREL = 0x61000029;
constexpr std::uint64_t DT_SCE_PLTRELSZ = 0x6100002d;
constexpr std::uint64_t DT_SCE_RELA = 0x61000033;
constexpr std::uint64_t DT_SCE_RELASZ = 0x61000031;
constexpr std::uint64_t DT_SCE_RELAENT = 0x6100002f;

constexpr std::uint32_t R_X86_64_64 = 1;
constexpr std::uint32_t R_X86_64_GLOB_DAT = 6;
constexpr std::uint32_t R_X86_64_JUMP_SLOT = 7;
constexpr std::uint32_t R_X86_64_RELATIVE = 8;

constexpr std::uint64_t kPreferredSlide = 0x400000;

struct Elf64_Ehdr {
    unsigned char e_ident[16];
    std::uint16_t e_type;
    std::uint16_t e_machine;
    std::uint32_t e_version;
    std::uint64_t e_entry;
    std::uint64_t e_phoff;
    std::uint64_t e_shoff;
    std::uint32_t e_flags;
    std::uint16_t e_ehsize;
    std::uint16_t e_phentsize;
    std::uint16_t e_phnum;
    std::uint16_t e_shentsize;
    std::uint16_t e_shnum;
    std::uint16_t e_shstrndx;
};

struct Elf64_Phdr {
    std::uint32_t p_type;
    std::uint32_t p_flags;
    std::uint64_t p_offset;
    std::uint64_t p_vaddr;
    std::uint64_t p_paddr;
    std::uint64_t p_filesz;
    std::uint64_t p_memsz;
    std::uint64_t p_align;
};

struct Elf64_Dyn {
    std::int64_t d_tag;
    std::uint64_t d_val;
};

struct Elf64_Rela {
    std::uint64_t r_offset;
    std::uint64_t r_info;
    std::int64_t r_addend;
};

struct Elf64_Sym {
    std::uint32_t st_name;
    unsigned char st_info;
    unsigned char st_other;
    std::uint16_t st_shndx;
    std::uint64_t st_value;
    std::uint64_t st_size;
};

std::string nid_from_strtab(const std::uint8_t* dynlib, std::size_t dynlib_size,
                            std::uint64_t strtab, std::uint32_t st_name) {
    if (st_name == 0 || strtab + st_name >= dynlib_size) {
        return {};
    }
    std::size_t p = static_cast<std::size_t>(strtab + st_name);
    while (p > strtab && dynlib[p - 1] != 0) {
        --p;
    }
    std::size_t end = p;
    while (end < dynlib_size && dynlib[end] != 0) {
        ++end;
    }
    std::string s(reinterpret_cast<const char*>(dynlib + p), end - p);
    auto hash = s.find('#');
    if (hash != std::string::npos) {
        s.resize(hash);
    }
    return s;
}

}  // namespace

bool load_orbis_elf(const char* path, ElfImage* image) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        host_log("cannot open %s", path);
        return false;
    }
    in.seekg(0, std::ios::end);
    const auto file_size = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    std::vector<std::uint8_t> file(file_size);
    in.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(file_size));
    static const unsigned char kElfMag[4] = {0x7f, 'E', 'L', 'F'};
    if (file.size() < sizeof(Elf64_Ehdr) || std::memcmp(file.data(), kElfMag, 4) != 0) {
        host_log("%s is not an ELF (decrypt SELF first)", path);
        return false;
    }

    image->sha256 = sha256_hex(file.data(), file.size());
    host_log("eboot %s", path);
    host_log("eboot sha256 %s", image->sha256.c_str());
    Elf64_Ehdr eh{};
    std::memcpy(&eh, file.data(), sizeof(eh));
    if (eh.e_machine != 62) {
        host_log("e_machine %u is not x86-64", eh.e_machine);
        return false;
    }

    std::uint64_t min_va = ~0ull;
    std::uint64_t max_va = 0;
    const Elf64_Phdr* ph = reinterpret_cast<const Elf64_Phdr*>(file.data() + eh.e_phoff);
    const Elf64_Phdr* dynlib_ph = nullptr;
    for (int i = 0; i < eh.e_phnum; ++i) {
        if (ph[i].p_type == PT_LOAD) {
            min_va = std::min(min_va, ph[i].p_vaddr);
            max_va = std::max(max_va, ph[i].p_vaddr + ph[i].p_memsz);
        }
        if (ph[i].p_type == PT_SCE_DYNLIBDATA) {
            dynlib_ph = &ph[i];
        }
    }
    if (min_va != 0 || max_va == 0 || !dynlib_ph) {
        host_log("unexpected Orbis layout min_va=0x%llx max_va=0x%llx dynlib=%d",
                 static_cast<unsigned long long>(min_va),
                 static_cast<unsigned long long>(max_va), dynlib_ph != nullptr);
        return false;
    }

    // The image and its tail (core/elf.h): 4 MiB for text that replaces the
    // game's own, reachable from every rip-relative instruction.
    constexpr std::uint64_t kTail = 4ull << 20;
    image->tail_va = (max_va + 0xfff) & ~0xfffull;
    image->tail_size = kTail;
    const std::size_t span = static_cast<std::size_t>(image->tail_va + kTail);
    // BBHOST_SLIDE=0x...: load at another slide (what a Windows boot does when
    // 0x400000 is taken), to prove the address-based patches on Linux.
    std::uint64_t slide = kPreferredSlide;
    if (const char* e = std::getenv("BBHOST_SLIDE")) slide = std::strtoull(e, nullptr, 0);
    if (!guest_alloc(&image->mem, slide, span)) {
        return false;
    }

    for (int i = 0; i < eh.e_phnum; ++i) {
        if (ph[i].p_type != PT_LOAD) {
            continue;
        }
        image->segments.push_back({ph[i].p_vaddr, ph[i].p_filesz, ph[i].p_memsz, ph[i].p_flags});
        void* dest = guest_ptr(image->mem, image->mem.slide + ph[i].p_vaddr);
        if (ph[i].p_filesz) {
            std::memcpy(dest, file.data() + ph[i].p_offset, ph[i].p_filesz);
        }
    }

    const std::uint8_t* dynlib = file.data() + dynlib_ph->p_offset;
    const std::size_t dynlib_size = static_cast<std::size_t>(dynlib_ph->p_filesz);

    std::uint64_t strtab = 0, symtab = 0, syment = 24;
    std::uint64_t jmprel = 0, pltrelsz = 0;
    std::uint64_t rela = 0, relasz = 0, relaent = 24;
    image->init = 0;
    image->fini = 0;

    // Sony's dynamic table sits at the end of the file (PT_DYNAMIC). Values for
    // SCE_* tags are offsets into PT_SCE_DYNLIBDATA except INIT/FINI which are VAs.
    const Elf64_Phdr* dyn_ph = nullptr;
    for (int i = 0; i < eh.e_phnum; ++i) {
        if (ph[i].p_type == 2) {
            dyn_ph = &ph[i];
        }
    }
    if (!dyn_ph) {
        host_log("no PT_DYNAMIC");
        return false;
    }
    const auto* dyn = reinterpret_cast<const Elf64_Dyn*>(file.data() + dyn_ph->p_offset);
    const std::size_t ndyn = static_cast<std::size_t>(dyn_ph->p_filesz / sizeof(Elf64_Dyn));
    for (std::size_t i = 0; i < ndyn && dyn[i].d_tag != DT_NULL; ++i) {
        switch (static_cast<std::uint64_t>(dyn[i].d_tag)) {
            case DT_INIT: image->init = dyn[i].d_val; break;
            case DT_FINI: image->fini = dyn[i].d_val; break;
            case DT_SCE_STRTAB: strtab = dyn[i].d_val; break;
            case DT_SCE_SYMTAB: symtab = dyn[i].d_val; break;
            case DT_SCE_SYMENT: syment = dyn[i].d_val; break;
            case DT_SCE_JMPREL: jmprel = dyn[i].d_val; break;
            case DT_SCE_PLTRELSZ: pltrelsz = dyn[i].d_val; break;
            case DT_SCE_RELA: rela = dyn[i].d_val; break;
            case DT_SCE_RELASZ: relasz = dyn[i].d_val; break;
            case DT_SCE_RELAENT: relaent = dyn[i].d_val; break;
            default: break;
        }
    }

    // This 1.09 eboot stores rela offset in DT_SCE_RELAENT (0xb1f0) and entsize 24
    // in DT_SCE_RELA. Identify by magnitude instead of trusting the tag names.
    if (rela == 24 && relaent > 24) {
        std::swap(rela, relaent);
    }
    if (relaent != 24) {
        relaent = 24;
    }
    if (syment != 24) {
        syment = 24;
    }

    std::size_t relative = 0, abs64 = 0, globdat = 0, other = 0;
    if (rela && relasz && rela + relasz <= dynlib_size) {
        const std::size_t n = relasz / relaent;
        for (std::size_t i = 0; i < n; ++i) {
            Elf64_Rela r{};
            std::memcpy(&r, dynlib + rela + i * relaent, sizeof(r));
            const std::uint32_t type = static_cast<std::uint32_t>(r.r_info);
            const std::uint32_t sym_i = static_cast<std::uint32_t>(r.r_info >> 32);
            auto* slot = reinterpret_cast<std::uint64_t*>(
                guest_ptr(image->mem, image->mem.slide + r.r_offset));
            if (type == R_X86_64_RELATIVE) {
                *slot = image->mem.slide + static_cast<std::uint64_t>(r.r_addend);
                if (r.r_offset < 0x100000000ull && r.r_addend >= 0 && r.r_addend < 0x100000000ll)
                    image->relative.push_back({static_cast<std::uint32_t>(r.r_offset), static_cast<std::uint32_t>(r.r_addend)});
                ++relative;
            } else if (type == R_X86_64_64) {
                std::uint64_t symval = 0;
                if (symtab && syment && symtab + (sym_i + 1) * syment <= dynlib_size) {
                    Elf64_Sym sym{};
                    std::memcpy(&sym, dynlib + symtab + sym_i * syment, sizeof(sym));
                    symval = sym.st_value;
                }
                *slot = image->mem.slide + symval + static_cast<std::uint64_t>(r.r_addend);
                ++abs64;
            } else if (type == R_X86_64_GLOB_DAT) {
                ImportReloc imp;
                imp.got_va = image->mem.slide + r.r_offset;
                unsigned st_info = 0;
                if (symtab && syment && symtab + (sym_i + 1) * syment <= dynlib_size) {
                    Elf64_Sym sym{};
                    std::memcpy(&sym, dynlib + symtab + sym_i * syment, sizeof(sym));
                    st_info = sym.st_info;
                    imp.nid = nid_from_strtab(dynlib, dynlib_size, strtab, sym.st_name);
                    imp.name = lookup_nid_name(imp.nid);
                }
                *slot = bind_glob_dat(imp.name, imp.nid, r.r_offset, st_info);
                ++globdat;
            } else {
                ++other;
            }
        }
    }
    host_log("relocs: %zu relative, %zu abs64, %zu globdat, %zu other",
             relative, abs64, globdat, other);

    if (jmprel && pltrelsz && jmprel + pltrelsz <= dynlib_size) {
        const std::size_t n = pltrelsz / 24;
        for (std::size_t i = 0; i < n; ++i) {
            Elf64_Rela r{};
            std::memcpy(&r, dynlib + jmprel + i * 24, sizeof(r));
            const std::uint32_t type = static_cast<std::uint32_t>(r.r_info);
            const std::uint32_t sym_i = static_cast<std::uint32_t>(r.r_info >> 32);
            if (type != R_X86_64_JUMP_SLOT) {
                continue;
            }
            ImportReloc imp;
            imp.got_va = image->mem.slide + r.r_offset;
            if (symtab + (sym_i + 1) * syment <= dynlib_size) {
                Elf64_Sym sym{};
                std::memcpy(&sym, dynlib + symtab + sym_i * syment, sizeof(sym));
                imp.nid = nid_from_strtab(dynlib, dynlib_size, strtab, sym.st_name);
                imp.name = lookup_nid_name(imp.nid);
            }
            auto* slot = reinterpret_cast<std::uint64_t*>(guest_ptr(image->mem, imp.got_va));
            *slot = bind_import(imp.name, imp.nid, imp.got_va, static_cast<int>(i));
            image->imports.push_back(std::move(imp));
        }
    }
    std::size_t named = 0;
    for (const auto& imp : image->imports) {
        if (!imp.name.empty()) {
            ++named;
        }
    }
    host_log("bound %zu JUMP_SLOT imports (%zu named)", image->imports.size(), named);

    TlsRewriteStats tls_st;
    for (int i = 0; i < eh.e_phnum; ++i) {
        if (ph[i].p_type == PT_LOAD && (ph[i].p_flags & PF_X)) {
            const std::size_t sz = static_cast<std::size_t>((ph[i].p_memsz + 0xfff) & ~0xfffull);
            // The guest's TLS reads move from FS to GS before the text is
            // sealed (core/tls_rewrite.h): the Windows shape, on by request here.
            if (tls_gs_mode()) {
                tls_rewrite_to_gs(image, image->mem.slide + ph[i].p_vaddr, static_cast<std::size_t>(ph[i].p_filesz),
                                  &tls_st);
            }
            if (!guest_protect_rx(&image->mem, image->mem.slide + ph[i].p_vaddr, sz)) {
                return false;
            }
            guest_mark_executable(&image->mem, image->mem.slide + ph[i].p_vaddr, sz);
        }
    }
    if (tls_gs_mode()) tls_rewrite_report(tls_st);

    image->entry = image->mem.slide + eh.e_entry;
    if (image->init) {
        image->init += image->mem.slide;
    }
    if (image->fini) {
        image->fini += image->mem.slide;
    }
    host_log("entry 0x%llx init 0x%llx slide 0x%llx",
             static_cast<unsigned long long>(image->entry),
             static_cast<unsigned long long>(image->init),
             static_cast<unsigned long long>(image->mem.slide));
    return true;
}

void unload_elf(ElfImage* image) {
    guest_free(&image->mem);
    image->imports.clear();
}
