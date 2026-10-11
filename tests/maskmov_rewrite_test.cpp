// core/maskmov_rewrite.cpp: the stub must store only the selected bytes
// (Rosetta 2 also writes back the others' old values), keep flags and
// registers, and run the instructions its jump covers.
#include "core/maskmov_rewrite.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

#include <sys/mman.h>
#include <unistd.h>

namespace {
int g_fail = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                        \
        }                                                                    \
    } while (0)

constexpr std::size_t kPage = 4096;

// f(dst, data, mask): loads the data and mask registers, eax = 41, CF = 1,
// the masked store to [rdi], adc eax, 0; ret - 42 when the flags survive.
using Fn = int (*)(void* dst, const void* data, const void* mask);

struct Variant {
    const char* name;
    std::uint8_t loads[8];  // vmovdqu data, [rsi]; vmovdqu mask, [rdx]
    std::uint8_t site[5];
    std::size_t site_len;
};
const Variant kVariants[] = {
    // vmovdqu xmm1, [rsi]; vmovdqu xmm2, [rdx]; vmaskmovdqu xmm1, xmm2
    {"VEX2 xmm1/xmm2", {0xc5, 0xfa, 0x6f, 0x0e, 0xc5, 0xfa, 0x6f, 0x12}, {0xc5, 0xf9, 0xf7, 0xca}, 4},
    // vmovdqu xmm13, [rsi]; vmovdqu xmm14, [rdx]; vmaskmovdqu xmm13, xmm14
    {"VEX3 xmm13/xmm14", {0xc5, 0x7a, 0x6f, 0x2e, 0xc5, 0x7a, 0x6f, 0x32}, {0xc4, 0x41, 0x79, 0xf7, 0xee}, 5},
};

// Writes f at code; returns the offset of `mov eax, 41` and of the store.
void write_fn(std::uint8_t* code, const Variant& v, std::size_t* mov_at, std::size_t* site_at, std::size_t* adc_end) {
    std::size_t n = 0;
    std::memcpy(code + n, v.loads, 8);
    n += 8;
    *mov_at = n;
    const std::uint8_t mov[] = {0xb8, 41, 0, 0, 0, 0xf9};  // mov eax, 41; stc
    std::memcpy(code + n, mov, sizeof(mov));
    n += sizeof(mov);
    *site_at = n;
    std::memcpy(code + n, v.site, v.site_len);
    n += v.site_len;
    const std::uint8_t tail[] = {0x83, 0xd0, 0x00, 0xc3};  // adc eax, 0; ret
    std::memcpy(code + n, tail, sizeof(tail));
    *adc_end = n + 3;
}

const std::uint16_t kMasks[] = {0xffff, 0x0fff, 0x00ff, 0x5a3c, 0x8001, 0x0000};

void check_stores(Fn f, const char* what) {
    alignas(16) std::uint8_t data[16], mask[16], buf[48];
    for (int k = 0; k < 16; ++k) data[k] = static_cast<std::uint8_t>(0x10 + k);
    for (std::uint16_t m : kMasks) {
        for (int k = 0; k < 16; ++k) mask[k] = (m >> k) & 1 ? 0x80 | k : 0x7f;
        std::memset(buf, 0xee, sizeof(buf));
        const int r = f(buf + 16, data, mask);
        bool ok = r == 42;
        for (int k = 0; k < 48; ++k) {
            const bool sel = k >= 16 && k < 32 && ((m >> (k - 16)) & 1);
            ok = ok && buf[k] == (sel ? data[k - 16] : 0xee);
        }
        if (!ok) std::fprintf(stderr, "%s, mask 0x%04x: returned %d\n", what, m, r);
        CHECK(ok);
    }
}

// Stores of another thread's to the bytes after a 12-byte mask, lost while
// f stores in a loop: Rosetta 2's masked store writes them back.
unsigned lost_stores(Fn f, int iterations) {
    alignas(64) static std::uint8_t buf[64];
    alignas(16) static std::uint8_t data[16], mask[16];
    for (int k = 0; k < 16; ++k) {
        data[k] = static_cast<std::uint8_t>(k + 1);
        mask[k] = k < 12 ? 0xff : 0;
    }
    std::memset(buf, 0, sizeof(buf));
    std::atomic<bool> stop{false};
    std::thread storer([&] {
        while (!stop.load(std::memory_order_relaxed)) f(buf, data, mask);
    });
    auto* counter = reinterpret_cast<volatile std::uint32_t*>(buf + 12);
    std::uint32_t mine = 0;
    unsigned lost = 0;
    for (int i = 0; i < iterations; ++i) {
        const std::uint32_t v = *counter;
        if (v != mine) ++lost;
        mine = v + 1;
        *counter = mine;
    }
    stop = true;
    storer.join();
    return lost;
}
}  // namespace

int main() {
    // One mapping for the functions and their stubs, so a rel32 reaches.
    auto* mem = static_cast<std::uint8_t*>(mmap(nullptr, 4 * kPage, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0));
    if (mem == MAP_FAILED) {
        std::printf("no executable memory; nothing to test\n");
        return 0;
    }
    // A page whose first bytes follow a store's selected ones: read-only, so
    // a stub that writes them faults.
    auto* guard = static_cast<std::uint8_t*>(mmap(nullptr, 2 * kPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
    CHECK(guard != MAP_FAILED);
    std::memset(guard, 0xee, 2 * kPage);
    mprotect(guard + kPage, kPage, PROT_READ);

    MaskmovInsn mi;
    CHECK(maskmov_decode(kVariants[0].site, 4, &mi) && mi.len == 4 && mi.data == 1 && mi.mask == 2);
    CHECK(maskmov_decode(kVariants[1].site, 5, &mi) && mi.len == 5 && mi.data == 13 && mi.mask == 14);
    const std::uint8_t not_one[] = {0xc5, 0xf9, 0xf7, 0x0a, 0x90};  // a memory operand: not an encoding of it
    CHECK(!maskmov_decode(not_one, 5, &mi));

    int slot = 0;
    for (const Variant& v : kVariants) {
        for (int before = 0; before < 2; ++before) {
            std::uint8_t* code = mem + 128 * slot;
            std::uint8_t* stub = mem + 2 * kPage + 256 * slot;
            ++slot;
            std::size_t mov_at, site_at, adc_end;
            write_fn(code, v, &mov_at, &site_at, &adc_end);
            const auto f = reinterpret_cast<Fn>(code);
            char what[96];
            std::snprintf(what, sizeof(what), "%s, the processor's", v.name);
            if (!before) check_stores(f, what);  // the instruction itself, as reference
            // The jump covers the store and the adc after it, or the
            // mov/stc before it and the store.
            const std::size_t start = before ? mov_at : site_at;
            const std::size_t end = before ? site_at + v.site_len : adc_end;
            // The counter beside the stubs, within a rel32 of them.
            auto& count = *reinterpret_cast<std::uint64_t*>(mem + 3 * kPage + 8 * slot);
            count = 0;
            const std::size_t n = maskmov_emit_stub(code + start, end - start, site_at - start, reinterpret_cast<std::uint64_t>(code + start),
                                                    stub, reinterpret_cast<std::uint64_t>(stub), 256, &count);
            CHECK(n > 0);
            if (!n) continue;
            CHECK(maskmov_patch(code + start, end - start, reinterpret_cast<std::uint64_t>(code + start), reinterpret_cast<std::uint64_t>(stub)));
            std::snprintf(what, sizeof(what), "%s, from the stub (%s)", v.name, before ? "instructions before" : "instructions after");
            check_stores(f, what);
            CHECK(count == sizeof(kMasks) / sizeof(kMasks[0]));
            // A vec3 store ending where the read-only page starts.
            alignas(16) std::uint8_t data[16], mask[16];
            for (int k = 0; k < 16; ++k) {
                data[k] = static_cast<std::uint8_t>(k);
                mask[k] = k < 12 ? 0xff : 0;
            }
            CHECK(f(guard + kPage - 12, data, mask) == 42);
            CHECK(std::memcmp(guard + kPage - 12, data, 12) == 0);
            if (!before) {
                const unsigned lost = lost_stores(f, 2000000);
                if (lost) std::fprintf(stderr, "%s: %u stores beside the selected bytes lost\n", what, lost);
                CHECK(lost == 0);
            }
        }
    }
    // Rosetta 2's own masked store loses neighbouring stores: shown, not checked.
    {
        std::size_t mov_at, site_at, adc_end;
        std::uint8_t* code = mem + kPage + 512;
        write_fn(code, kVariants[0], &mov_at, &site_at, &adc_end);
        std::printf("the processor's vmaskmovdqu: %u of 4000000 neighbouring stores lost%s\n",
                    lost_stores(reinterpret_cast<Fn>(code), 4000000), maskmov_rewrite_wanted() ? " (Rosetta 2: rewritten in the eboot)" : "");
    }
    if (g_fail) {
        std::fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("maskmov_rewrite_test: ok\n");
    return 0;
}
