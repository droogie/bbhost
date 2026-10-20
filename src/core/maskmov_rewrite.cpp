#include "core/maskmov_rewrite.h"

#include "core/elf.h"
#include "core/memory.h"
#include "decomp/insn.h"
#include "log.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include "core/mac_vm.h"

#include <sys/mman.h>
#include <sys/sysctl.h>
#endif

namespace {

using ull = unsigned long long;

// 1.09 sites where a branch lands just after the store, so the jump covers
// the instructions before it instead (no branch lands in [start, site)).
struct Before {
    std::uint64_t site, start;  // ELF virtual addresses
};
// sub_2851790 stores a vec3 and returns; `je` at 0x2851797 lands on its ret.
constexpr Before kBefore109[] = {{0x28517a9, 0x28517a1}};

struct Emit {
    std::uint8_t* p;
    std::size_t n = 0, cap;
    bool ok = true;
    void b(std::initializer_list<std::uint8_t> bytes) {
        for (std::uint8_t v : bytes) {
            if (n >= cap) {
                ok = false;
                return;
            }
            p[n++] = v;
        }
    }
    void d32(std::int32_t v) {
        std::uint8_t t[4];
        std::memcpy(t, &v, 4);
        b({t[0], t[1], t[2], t[3]});
    }
    // A jcc/jmp rel8 to be aimed later: returns where its offset is.
    std::size_t j8(std::uint8_t op) {
        b({op, 0});
        return n - 1;
    }
    void aim(std::size_t at) {
        const std::size_t d = n - (at + 1);
        if (d > 127) ok = false;
        else p[at] = static_cast<std::uint8_t>(d);
    }
};

std::atomic<std::uint64_t*> g_counters{nullptr};
std::atomic<std::size_t> g_counter_n{0};
std::vector<std::uint64_t> g_sites;  // each counter's store, as an ELF address

}  // namespace

bool maskmov_decode(const std::uint8_t* p, std::size_t avail, MaskmovInsn* out) {
    // VEX.128.66.0F.WIG F7 /r, register form only (mod 11).
    if (avail >= 4 && p[0] == 0xc5 && (p[1] & 0x7f) == 0x79 && p[2] == 0xf7 && p[3] >= 0xc0) {
        out->len = 4;
        out->data = static_cast<std::uint8_t>((((p[1] >> 7) ^ 1) << 3) | ((p[3] >> 3) & 7));
        out->mask = p[3] & 7;
        return true;
    }
    if (avail >= 5 && p[0] == 0xc4 && (p[1] & 0x1f) == 1 && (p[2] & 0x7f) == 0x79 && p[3] == 0xf7 && p[4] >= 0xc0) {
        out->len = 5;
        out->data = static_cast<std::uint8_t>((((p[1] >> 7) ^ 1) << 3) | ((p[4] >> 3) & 7));
        out->mask = static_cast<std::uint8_t>(((((p[1] >> 5) & 1) ^ 1) << 3) | (p[4] & 7));
        return true;
    }
    return false;
}

std::size_t maskmov_emit_stub(const std::uint8_t* region, std::size_t region_len, std::size_t site_off, std::uint64_t from,
                              std::uint8_t* dst, std::uint64_t to, std::size_t cap, std::uint64_t* counter) {
    MaskmovInsn mi;
    if (site_off >= region_len || !maskmov_decode(region + site_off, region_len - site_off, &mi) ||
        site_off + mi.len > region_len) {
        return 0;
    }
    Emit e{dst, 0, cap};
    if (site_off) {
        const std::size_t moved = x86_relocate(region, site_off, from, dst, to, cap);
        if (!moved) return 0;
        e.n = moved;
    }
    const std::uint8_t x = mi.data, m = mi.mask;
    e.b({0x48, 0x8d, 0x64, 0x24, 0x80});  // lea rsp, [rsp-0x80]: below the red zone
    e.b({0x9c, 0x50, 0x51, 0x52});        // pushfq; push rax; push rcx; push rdx
    e.b({0x48, 0x83, 0xec, 0x10});        // sub rsp, 16
    // vmovdqu [rsp], xmm<x>
    e.b({0xc4, static_cast<std::uint8_t>((((x >> 3) ^ 1) << 7) | 0x61), 0x7a, 0x7f,
         static_cast<std::uint8_t>(((x & 7) << 3) | 4), 0x24});
    // vpmovmskb eax, xmm<m>
    e.b({0xc4, static_cast<std::uint8_t>(0xc1 | ((((m >> 3) ^ 1)) << 5)), 0x79, 0xd7, static_cast<std::uint8_t>(0xc0 | (m & 7))});
    if (counter) {
        // inc qword [rip+counter]
        e.b({0x48, 0xff, 0x05});
        const std::int64_t rel = static_cast<std::int64_t>(reinterpret_cast<std::uint64_t>(counter)) -
                                 static_cast<std::int64_t>(to + e.n + 4);
        if (rel < INT32_MIN || rel > INT32_MAX) return 0;
        e.d32(static_cast<std::int32_t>(rel));
    }
    std::size_t done[3];
    // All 16 bytes.
    e.b({0x3d, 0xff, 0xff, 0, 0});  // cmp eax, 0xffff
    std::size_t next = e.j8(0x75);
    e.b({0x48, 0x8b, 0x0c, 0x24, 0x48, 0x89, 0x0f});              // mov rcx, [rsp]; mov [rdi], rcx
    e.b({0x48, 0x8b, 0x4c, 0x24, 0x08, 0x48, 0x89, 0x4f, 0x08});  // mov rcx, [rsp+8]; mov [rdi+8], rcx
    done[0] = e.j8(0xeb);
    e.aim(next);
    // The first 12 (a vec3).
    e.b({0x3d, 0xff, 0x0f, 0, 0});  // cmp eax, 0xfff
    next = e.j8(0x75);
    e.b({0x48, 0x8b, 0x0c, 0x24, 0x48, 0x89, 0x0f});  // mov rcx, [rsp]; mov [rdi], rcx
    e.b({0x8b, 0x4c, 0x24, 0x08, 0x89, 0x4f, 0x08});  // mov ecx, [rsp+8]; mov [rdi+8], ecx
    done[1] = e.j8(0xeb);
    e.aim(next);
    // The first 8 (a vec2).
    e.b({0x3d, 0xff, 0, 0, 0});  // cmp eax, 0xff
    next = e.j8(0x75);
    e.b({0x48, 0x8b, 0x0c, 0x24, 0x48, 0x89, 0x0f});  // mov rcx, [rsp]; mov [rdi], rcx
    done[2] = e.j8(0xeb);
    e.aim(next);
    // Any other mask, a byte at a time.
    e.b({0x31, 0xc9});                    // xor ecx, ecx
    const std::size_t loop = e.n;
    e.b({0x0f, 0xa3, 0xc8});              // bt eax, ecx
    e.b({0x73, 0x06});                    // jnc +6
    e.b({0x8a, 0x14, 0x0c});              // mov dl, [rsp+rcx]
    e.b({0x88, 0x14, 0x0f});              // mov [rdi+rcx], dl
    e.b({0xff, 0xc1, 0x83, 0xf9, 0x10});  // inc ecx; cmp ecx, 16
    e.b({0x72, static_cast<std::uint8_t>(-static_cast<int>(e.n + 2 - loop))});  // jb loop
    for (std::size_t at : done) e.aim(at);
    e.b({0x48, 0x83, 0xc4, 0x10});                          // add rsp, 16
    e.b({0x5a, 0x59, 0x58, 0x9d});                          // pop rdx; pop rcx; pop rax; popfq
    e.b({0x48, 0x8d, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00});  // lea rsp, [rsp+0x80]
    if (!e.ok) return 0;
    const std::size_t after = site_off + mi.len;
    if (after < region_len) {
        const std::size_t moved = x86_relocate(region + after, region_len - after, from + after, dst + e.n, to + e.n, cap - e.n);
        if (!moved) return 0;
        e.n += moved;
    }
    // jmp from + region_len
    e.b({0xe9});
    const std::int64_t rel = static_cast<std::int64_t>(from + region_len) - static_cast<std::int64_t>(to + e.n + 4);
    if (rel < INT32_MIN || rel > INT32_MAX) return 0;
    e.d32(static_cast<std::int32_t>(rel));
    return e.ok ? e.n : 0;
}

bool maskmov_patch(std::uint8_t* region, std::size_t region_len, std::uint64_t from, std::uint64_t stub) {
    if (region_len < 5) return false;
    const std::int64_t rel = static_cast<std::int64_t>(stub) - static_cast<std::int64_t>(from + 5);
    if (rel < INT32_MIN || rel > INT32_MAX) return false;
    std::memset(region + 5, 0xcc, region_len - 5);
    const auto r = static_cast<std::int32_t>(rel);
    std::memcpy(region + 1, &r, 4);
    region[0] = 0xe9;
    return true;
}

bool maskmov_rewrite_wanted() {
    const char* e = std::getenv("BBHOST_MASKMOV");
    if (e && e[0] == '0') return false;
#if defined(__APPLE__)
    if (e && e[0] == '1') return true;
    int v = 0;
    std::size_t n = sizeof(v);
    return sysctlbyname("sysctl.proc_translated", &v, &n, nullptr, 0) == 0 && v == 1;
#else
    return false;  // a processor's own masked store
#endif
}

void maskmov_rewrite(ElfImage* image, std::uint64_t va, std::size_t size, bool is_109, MaskmovRewriteStats* st) {
    auto* text = static_cast<std::uint8_t*>(guest_ptr(image->mem, va));
    const std::uint64_t elf_base = image->mem.slide;  // ELF VA = guest VA - slide
    struct Site {
        std::size_t start, len, site_off;
    };
    std::vector<Site> sites;
    std::size_t prev_end = 0;
    for (std::size_t i = 0; i + 5 <= size; ++i) {
        if (text[i] != 0xc4 && text[i] != 0xc5) continue;
        MaskmovInsn mi;
        if (!maskmov_decode(text + i, size - i, &mi)) continue;
        ++st->found;
        const std::uint64_t elf_va = va + i - elf_base;
        std::size_t start = i, end = i + mi.len;
        for (const Before& b : kBefore109) {
            if (is_109 && b.site == elf_va) start = static_cast<std::size_t>(b.start + elf_base - va);
        }
        // Whole instructions until the 5-byte jump fits.
        bool whole = true;
        while (end - start < 5) {
            const X86Insn in = x86_decode(text + end, size - end);
            if (!in.len) {
                whole = false;
                break;
            }
            end += in.len;
        }
        if (start != i) {
            // The instructions before it must end exactly at the store.
            std::size_t at = start;
            while (at < i) {
                const X86Insn in = x86_decode(text + at, size - at);
                if (!in.len) break;
                at += in.len;
            }
            whole = whole && at == i && start >= prev_end;
        }
        if (!whole || !is_109) {
            host_log("maskmov: the store at 0x%llx stays as it is (%s)", static_cast<ull>(elf_va),
                     !is_109 ? "not the 1.09 eboot, whose branch targets the sites were checked against" : "undecodable neighbours");
            i += mi.len - 1;
            continue;
        }
        sites.push_back({start, end - start, i - start});
        prev_end = end;
        i = end - 1;
    }
    if (sites.empty()) return;
#if defined(__APPLE__)
    // Counters on their own non-executable pages: under Rosetta 2 every write
    // to a page whose code has run retranslates it.
    constexpr std::size_t kStub = 256;
    const std::size_t counter_bytes = (sites.size() * 8 + 0xfff) & ~std::size_t{0xfff};
    const std::size_t stub_bytes = (sites.size() * kStub + 0xfff) & ~std::size_t{0xfff};
    auto* page = static_cast<std::uint8_t*>(mac_vm_map_near(image->mem.slide, image->mem.slide + image->mem.size,
                                                            counter_bytes + stub_bytes, PROT_READ | PROT_WRITE | PROT_EXEC));
    if (!page) {
        host_log("maskmov: no executable memory within reach of the image; the %zu stores stay as they are", sites.size());
        return;
    }
    auto* counters = reinterpret_cast<std::uint64_t*>(page);
    std::uint8_t* stubs = page + counter_bytes;
    std::size_t used = 0;
    for (std::size_t k = 0; k < sites.size(); ++k) {
        const Site& s = sites[k];
        std::uint8_t* dst = stubs + used;
        const std::uint64_t from = va + s.start;
        const std::size_t n = maskmov_emit_stub(text + s.start, s.len, s.site_off, from, dst, reinterpret_cast<std::uint64_t>(dst),
                                                kStub, &counters[k]);
        if (!n || !maskmov_patch(text + s.start, s.len, from, reinterpret_cast<std::uint64_t>(dst))) {
            host_log("maskmov: the store at 0x%llx stays as it is (no stub)", static_cast<ull>(from + s.site_off - elf_base));
            continue;
        }
        used += (n + 15) & ~std::size_t{15};
        ++st->rewritten;
    }
    mprotect(page, counter_bytes, PROT_READ | PROT_WRITE);
    mprotect(stubs, stub_bytes, PROT_READ | PROT_EXEC);
    for (const Site& s : sites) g_sites.push_back(va + s.start + s.site_off - elf_base);
    g_counters.store(counters);
    g_counter_n.store(sites.size());
#else
    (void)image;
#endif
}

void maskmov_rewrite_report(const MaskmovRewriteStats& st) {
    host_log("maskmov: %zu of %zu vmaskmovdqu sites store only their selected bytes, from stubs (Rosetta 2 writes all 16 back; "
             "BBHOST_MASKMOV=0 leaves them)%s",
             st.rewritten, st.found, st.found == 53 ? "" : " - not the 53 of the 1.09 build");
}

void maskmov_rewrite_exit_report() {
    const std::uint64_t* c = g_counters.load();
    const std::size_t n = g_counter_n.load();
    if (!c) return;
    std::uint64_t total = 0;
    std::size_t ran = 0;
    char list[1024];
    int len = 0;
    for (std::size_t k = 0; k < n; ++k) {
        total += c[k];
        if (!c[k]) continue;
        ++ran;
        if (len < static_cast<int>(sizeof(list)) - 40 && k < g_sites.size())
            len += std::snprintf(list + len, sizeof(list) - static_cast<std::size_t>(len), " 0x%llx:%llu", static_cast<ull>(g_sites[k]),
                                 static_cast<ull>(c[k]));
    }
    host_log("maskmov: the stubs stored %llu times, %zu of %zu sites ran%s%s", static_cast<ull>(total), ran, n, ran ? ":" : "",
             ran ? list : "");
}
