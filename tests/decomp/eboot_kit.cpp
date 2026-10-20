#include "eboot_kit.h"

#include "core/elf.h"
#include "core/imports.h"
#include "core/sha256.h"
#include "core/tls_rewrite.h"
#include "decomp/decomp.h"
#include "engine/addr.h"
#include "hle/guest_fs.h"

#include "core/plt_names.inc"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <xmmintrin.h>

namespace eboot_kit {
namespace {

std::string g_path;
std::vector<std::uint8_t> g_bytes;
std::vector<Function> g_functions;
ElfImage g_image;
bool g_loaded = false;
void* g_tcb = nullptr;

struct Segment {
    std::uint64_t va, off, filesz;
    std::uint32_t type;
};

std::vector<Segment> segments(const std::vector<std::uint8_t>& f) {
    std::uint64_t phoff = 0;
    std::uint16_t phnum = 0;
    std::memcpy(&phoff, f.data() + 0x20, 8);
    std::memcpy(&phnum, f.data() + 0x38, 2);
    std::vector<Segment> out;
    for (int i = 0; i < phnum; ++i) {
        const std::uint8_t* ph = f.data() + phoff + 56 * i;
        Segment s{};
        std::memcpy(&s.type, ph, 4);
        std::memcpy(&s.off, ph + 8, 8);
        std::memcpy(&s.va, ph + 16, 8);
        std::memcpy(&s.filesz, ph + 32, 8);
        out.push_back(s);
    }
    return out;
}

// An ELF address's offset in the file, or ~0 outside every PT_LOAD.
std::uint64_t file_offset(std::uint64_t va) {
    for (const Segment& s : segments(g_bytes))
        if (s.type == 1 && va >= s.va && va < s.va + s.filesz) return s.off + (va - s.va);
    return ~0ull;
}

template <class T>
T read(std::uint64_t va) {
    T v{};
    const std::uint64_t o = file_offset(va);
    if (o != ~0ull && o + sizeof(T) <= g_bytes.size()) std::memcpy(&v, g_bytes.data() + o, sizeof(T));
    return v;
}

// Every import a stub of its own that names it: mov edi, index; jmp [rip+0]
// to the trap.
struct Import {
    std::string name, nid;
    std::uint64_t got = 0;  // where the game's code finds it
};
std::vector<Import> g_imports;
std::uint8_t* g_stubs = nullptr;
constexpr std::size_t kStub = 32, kStubSpace = 0x10000;
alignas(64) std::uint8_t g_data_imports[64 * 1024];
std::size_t g_data_used = 0;

[[noreturn]] void import_trap(std::uint32_t i) {
    const Import& imp = g_imports[i];
    std::fprintf(stderr, "eboot_kit: the game's code called an import (%s, NID %s): a test has no system library behind it\n",
                 imp.name.c_str(), imp.nid.c_str());
    std::abort();
}

}  // namespace

const std::vector<std::uint8_t>& bytes(const char* test) {
    if (!g_bytes.empty()) return g_bytes;
    const char* env = std::getenv("BBHOST_EBOOT");
    g_path = env && *env ? env : std::string(BBHOST_SOURCE_DIR) + "/eboot-109-decrypted.bin";
    std::ifstream in(g_path, std::ios::binary);
    if (!in) {
        std::printf("%s skipped: no eboot at %s (BBHOST_EBOOT)\n", test, g_path.c_str());
        std::exit(kSkip);
    }
    g_bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (sha256_hex(g_bytes.data(), g_bytes.size()) != kEboot109Sha256) {
        std::printf("%s skipped: %s is not the 1.09 eboot\n", test, g_path.c_str());
        std::exit(kSkip);
    }
    return g_bytes;
}

Range text(const char* test) {
    bytes(test);
    std::uint64_t phoff = 0;
    std::uint16_t phnum = 0;
    std::memcpy(&phoff, g_bytes.data() + 0x20, 8);
    std::memcpy(&phnum, g_bytes.data() + 0x38, 2);
    for (int i = 0; i < phnum; ++i) {
        const std::uint8_t* ph = g_bytes.data() + phoff + 56 * i;
        std::uint32_t type = 0, flags = 0;
        std::uint64_t va = 0, filesz = 0;
        std::memcpy(&type, ph, 4);
        std::memcpy(&flags, ph + 4, 4);
        std::memcpy(&va, ph + 16, 8);
        std::memcpy(&filesz, ph + 32, 8);
        if (type == 1 && (flags & 1)) return {va + kBase, va + filesz + kBase};
    }
    return {0, 0};
}

const std::uint8_t* file_at(const char* test, std::uint64_t bn) {
    bytes(test);
    const std::uint64_t o = file_offset(bn - kBase);
    return o == ~0ull ? nullptr : g_bytes.data() + o;
}

const std::vector<Function>& functions(const char* test) {
    if (!g_functions.empty()) return g_functions;
    bytes(test);
    // PT_GNU_EH_FRAME: the header, then a table of (start, FDE) pairs
    // relative to the header (datarel sdata4), sorted by start.
    std::uint64_t hdr = 0;
    for (const Segment& s : segments(g_bytes))
        if (s.type == 0x6474e550) hdr = s.va;
    const auto enc = read<std::uint32_t>(hdr);
    const std::uint8_t eh_ptr_enc = (enc >> 8) & 0xff, count_enc = (enc >> 16) & 0xff, table_enc = enc >> 24;
    if (!hdr || eh_ptr_enc != 0x1b || count_enc != 0x03 || table_enc != 0x3b) {
        std::printf("%s: the eboot's .eh_frame_hdr is not the shape this reads\n", test);
        std::exit(1);
    }
    const std::uint32_t count = read<std::uint32_t>(hdr + 8);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint64_t row = hdr + 12 + 8ull * i;
        const std::uint64_t start = hdr + static_cast<std::uint64_t>(static_cast<std::int64_t>(read<std::int32_t>(row)));
        const std::uint64_t fde = hdr + static_cast<std::uint64_t>(static_cast<std::int64_t>(read<std::int32_t>(row + 4)));
        // FDE: length, CIE pointer, pc_begin (pcrel sdata4), pc_range.
        const std::uint64_t begin = fde + 8 + static_cast<std::uint64_t>(static_cast<std::int64_t>(read<std::int32_t>(fde + 8)));
        if (begin != start) {
            std::printf("%s: FDE %u at 0x%llx begins at 0x%llx, the table says 0x%llx\n", test, i,
                        static_cast<unsigned long long>(fde), static_cast<unsigned long long>(begin),
                        static_cast<unsigned long long>(start));
            std::exit(1);
        }
        g_functions.push_back({start + kBase, read<std::uint32_t>(fde + 12)});
    }
    return g_functions;
}

void guest_fp() {
    _mm_setcsr(0x9fc0);
    const std::uint16_t cw = 0x037f;
    asm volatile("fldcw %0" : : "m"(cw));
}

void host_fp() {
    _mm_setcsr(0x1f80);
    const std::uint16_t cw = 0x037f;
    asm volatile("fldcw %0" : : "m"(cw));
}

void load(const char* test) {
    if (g_loaded) return;
    bytes(test);
    g_stubs = static_cast<std::uint8_t*>(
        mmap(nullptr, kStubSpace, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (g_stubs == MAP_FAILED || !load_orbis_elf(g_path.c_str(), &g_image) || g_image.mem.slide != kBase) {
        std::printf("%s: cannot load the eboot at 0x%llx\n", test, static_cast<unsigned long long>(kBase));
        std::exit(1);
    }
    // The calling thread as a guest thread: its TCB where the rewritten
    // fs:[0] reads look for it (core/tls_rewrite.h), the console's FP mode.
    if (tls_gs_mode()) {
        auto* block = static_cast<std::uint8_t*>(std::calloc(1, kGuestTls + sizeof(GuestTcb) + 32));
        auto* tcb = reinterpret_cast<GuestTcb*>(block + kGuestTls);
        tcb->self = tcb;
        tcb->magic = kTcbMagic;
        g_tcb = tcb;
#if defined(__APPLE__)
        // macOS: the slot is a pthread key of ours, read at gs:[8*key].
        pthread_setspecific(static_cast<pthread_key_t>(tls_gs_slot()), tcb);
#else
        auto* teb = static_cast<std::uint8_t*>(std::calloc(1, kTebStandInSize));
        std::memcpy(teb + tls_gs_disp(), &tcb, sizeof(tcb));
        syscall(SYS_arch_prctl, ARCH_SET_GS, reinterpret_cast<unsigned long>(teb));
#endif
    }
    guest_fp();
    g_loaded = true;
}

void* tcb() { return g_tcb; }

void stub(std::uint64_t bn, const void* host) {
    const std::uint64_t page = bn & ~0xfffull, end = (bn + 14 + 0xfff) & ~0xfffull;
    mprotect(reinterpret_cast<void*>(page), end - page, PROT_READ | PROT_WRITE | PROT_EXEC);
    std::uint8_t* p = at(bn);
    p[0] = 0xff;
    p[1] = 0x25;  // jmp [rip+0]
    std::memset(p + 2, 0, 4);
    const auto to = reinterpret_cast<std::uint64_t>(host);
    std::memcpy(p + 6, &to, 8);
    mprotect(reinterpret_cast<void*>(page), end - page, PROT_READ | PROT_EXEC);
}

const std::vector<Relocated>& relocated() {
    static std::vector<Relocated> out;
    if (out.empty())
        for (const auto& [slot, target] : g_image.relative) out.push_back({slot + kBase, target + kBase});
    return out;
}

bool bind(const char* name, const void* fn) {
    bool found = false;
    for (const Import& imp : g_imports)
        if (imp.got && imp.name == name) {
            *reinterpret_cast<const void**>(static_cast<std::uintptr_t>(imp.got)) = fn;
            found = true;
        }
    return found;
}

float Rng::value(float lo, float hi) {
    if (specials && chance(0.15)) {
        static const float kSpecial[] = {0.0f,     -0.0f,   1.0f,   -1.0f,   0.5f,   1e-40f, -1e-40f, FLT_MAX,
                                         -FLT_MAX, 1e20f,   -1e20f, 127.0f,  255.0f, 2.0f,   -2.0f,
                                         std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                         std::numeric_limits<float>::quiet_NaN()};
        return kSpecial[std::uniform_int_distribution<std::size_t>(0, std::size(kSpecial) - 1)(g_)];
    }
    return uniform(lo, hi);
}

}  // namespace eboot_kit

// What a decomp source reaches of the list's own code (decomp/decomp.h), for
// a test that builds the source alone: the image at its preferred slide,
// nothing placed, no compare run. Weak, so decomp.cpp's win where a test
// links it.
__attribute__((weak)) std::uint64_t g_decomp_slide = eboot_kit::kBase;
__attribute__((weak)) std::uint64_t decomp_guest(std::uint64_t bn) { return bn; }
__attribute__((weak)) void decomp_add(const DecompFunction&) {}
__attribute__((weak)) bool decomp_comparing() { return false; }

// What the loader binds imports with (core/imports.h), for a test: traps.
std::string lookup_nid_name(std::string_view nid) { return std::string(nid); }

std::uint64_t bind_import(const std::string& nid_name, const std::string& nid, std::uint64_t got, int plt_index) {
    using namespace eboot_kit;
    // The PLT's order names the slot the game calls (core/imports.cpp).
    const std::string name =
        plt_index >= 0 && plt_index < kPltNameCount && kPltNames[plt_index] ? std::string(kPltNames[plt_index]) : nid_name;
    static const struct {
        const char* name;
        const void* fn;
    } kHost[] = {
        {"memset", reinterpret_cast<const void*>(&::memset)},   {"memcpy", reinterpret_cast<const void*>(&::memcpy)},
        {"memmove", reinterpret_cast<const void*>(&::memmove)}, {"memcmp", reinterpret_cast<const void*>(&::memcmp)},
        {"strlen", reinterpret_cast<const void*>(&::strlen)},   {"strcmp", reinterpret_cast<const void*>(&::strcmp)},
        {"strncmp", reinterpret_cast<const void*>(&::strncmp)}, {"strcpy", reinterpret_cast<const void*>(&::strcpy)},
    };
    for (const auto& h : kHost)
        if (name == h.name) return reinterpret_cast<std::uint64_t>(h.fn);
    const std::uint32_t i = static_cast<std::uint32_t>(g_imports.size());
    if ((i + 1) * kStub > kStubSpace) return 0;
    g_imports.push_back({name, nid, got});
    std::uint8_t* s = g_stubs + i * kStub;
    s[0] = 0xbf;  // mov edi, i
    std::memcpy(s + 1, &i, 4);
    s[5] = 0xff;
    s[6] = 0x25;  // jmp [rip+0]
    std::memset(s + 7, 0, 4);
    const auto trap = reinterpret_cast<std::uint64_t>(&import_trap);
    std::memcpy(s + 11, &trap, 8);
    return reinterpret_cast<std::uint64_t>(s);
}

std::uint64_t bind_glob_dat(const std::string&, const std::string&, std::uint64_t, unsigned) {
    using namespace eboot_kit;
    // A data import: a zeroed kilobyte of its own.
    if (g_data_used + 1024 > sizeof(g_data_imports)) return 0;
    const auto p = reinterpret_cast<std::uint64_t>(g_data_imports + g_data_used);
    g_data_used += 1024;
    return p;
}
