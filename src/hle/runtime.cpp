#include "hle/hle.h"
#include "hle/fs.h"
#include "hle/np.h"
#include "hle/common.h"
#include "hle/guest_fs.h"
#include "hle/modules.h"
#include "hle/platform.h"
#include "core/elf.h"
#include "core/thunk.h"
#include "core/config.h"
#include "engine/addr.h"
#include "engine/key_prompts.h"
#include "engine/menu_pointer.h"
#include "engine/option_menu.h"
#include "engine/summon_invite.h"
#include "engine/np_test.h"
#include "engine/gx_resources.h"
#include "engine/gx_state.h"
#include "engine/graphics_patch.h"
#include "engine/mouse_camera.h"
#include "engine/frame_pool.h"
#include "engine/yebis.h"
#include "engine/guest.h"
#include "decomp/decomp.h"
#include "host/fmod_project_init.h"
#include "decomp/events/flag_store.h"
#include "engine/event_flags.h"
#include "engine/player_data.h"
#include "engine/world_chr.h"
#include "engine/playlog.h"
#include "engine/sos_signs.h"
#include "engine/change_appearance.h"
#include "engine/rebirth.h"
#include "engine/five_players.h"
#include "engine/lua_events.h"
#include "engine/params.h"
#include "engine/patch_manifest.h"
#include "log.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

thread_local int g_unknown_calls = 0;
}  // namespace

thread_local int t_guest_errno = 0;

namespace {

std::uintptr_t g_boot_stack_base = 0;
std::size_t g_boot_stack_size = 0;

struct StubInfo {
    std::string name;
    std::string nid;
    std::uint64_t got_va = 0;
};
std::vector<StubInfo> g_stubs;

void* wrap_hle_impl(void* fn) { return thunk_wrap(fn); }

// The image the DL_PANIC backtrace resolves against. DL_PANIC's own file
// and line name the Dantelion source, but the hot ones (DLLightMutex's
// "Mutex is not initialized." is inlined at 81 sites, all reporting line
// 153) say nothing about which caller tripped it, so log the guest frames.
std::uint64_t g_panic_slide = 0;
std::uint64_t g_panic_size = 0;

// BN addresses (the eboot loaded at its preferred 0x400000) so a frame can
// be pasted into the database as-is.
bool panic_in_image(std::uint64_t va) {
    return g_panic_size != 0 && va >= g_panic_slide && va - g_panic_slide < g_panic_size;
}

void panic_backtrace(std::uint64_t rsp, std::uint64_t rbp) {
    if (!g_panic_size || !rsp) {
        return;
    }
    // The call that reached DL_PANIC pushed its return address; the patch
    // replaced the entry in place, so nothing else is on the stack yet.
    const auto* ret0 = reinterpret_cast<const std::uint64_t*>(rsp);
    if (panic_in_image(*ret0)) {
        host_log("DL_PANIC from 0x%llx",
                 static_cast<unsigned long long>(*ret0 - g_panic_slide + kPreferredGuestSlide));
    } else {
        host_log("DL_PANIC from 0x%llx (outside the image)",
                 static_cast<unsigned long long>(*ret0));
    }
    // The eboot keeps frame pointers. Walk them, but only while each frame
    // is above the last one and within a stack's reach of the first, so a
    // leaf without a frame or a tail call stops the walk instead of
    // wandering into unmapped memory.
    std::uint64_t fp = rbp;
    std::uint64_t prev = rsp;
    for (int depth = 0; depth < 24; ++depth) {
        if (fp <= prev || fp - rsp > (16u << 20) || (fp & 7u) != 0) {
            break;
        }
        const auto* frame = reinterpret_cast<const std::uint64_t*>(fp);
        const std::uint64_t ret = frame[1];
        if (!panic_in_image(ret)) {
            break;
        }
        host_log("  #%d 0x%llx", depth,
                 static_cast<unsigned long long>(ret - g_panic_slide + kPreferredGuestSlide));
        prev = fp;
        fp = frame[0];
    }
}

GUEST_ABI void hle_dl_panic(const char* file, int line, const char* msg, std::uint64_t rsp,
                            std::uint64_t rbp) {
    // BBHOST_PANIC_CONTINUE=1 returns instead of aborting, so one panic in a
    // path the port does not need yet does not end the run, and a session can
    // collect every site it reaches. The guest carries on with whatever state
    // tripped the assert, so it is a diagnostic switch, not a fix.
    static const bool keep_going = [] {
        const char* e = std::getenv("BBHOST_PANIC_CONTINUE");
        return e && e[0] == '1';
    }();
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 64) {
        host_log("DL_PANIC %s(%d): %s", file ? file : "?", line, msg ? msg : "?");
        panic_backtrace(rsp, rbp);
        if (const std::string files = hle_fs_problem_files(); !files.empty())
            host_log("DL_PANIC: game files found missing or empty, newest last: %s", files.c_str());
        hle_fs_log_recent_opens();
        if (file && std::strstr(file, "FileTransferTask"))
            host_log("DL_PANIC: the game's file loader stopped on a file it could not read: a game file is missing or "
                     "empty. The 1.09 update's files must be copied over the game folder (its sce_sys/param.sfo then "
                     "says APP_VER 01.09), and the dump must be complete.");
    }
    if (keep_going) {
        return;
    }
    std::abort();
}

}  // namespace

GUEST_ABI int hle_ok() { return 0; }

extern "C" GUEST_ABI int hle_unknown_log(unsigned idx) {
    if (g_unknown_calls < 64) {
        if (idx < g_stubs.size()) {
            const StubInfo& s = g_stubs[idx];
            host_log("HLE stub #%u %s nid=%s got=0x%llx", idx,
                     s.name.empty() ? "?" : s.name.c_str(),
                     s.nid.empty() ? "-" : s.nid.c_str(),
                     static_cast<unsigned long long>(s.got_va));
        } else {
            host_log("HLE stub #%u", idx);
        }
    }
    ++g_unknown_calls;
    return 0;
}

#include "hle/stub_table.inc"

GUEST_ABI int hle_unknown() {
    return hle_unknown_log(0xffffffffu);
}

std::uint64_t hle_make_stub(const std::string& name, const std::string& nid, std::uint64_t got_va) {
    const unsigned idx = static_cast<unsigned>(g_stubs.size());
    g_stubs.push_back(StubInfo{name, nid, got_va});
    if (idx < static_cast<unsigned>(kHleStubCount)) {
        return reinterpret_cast<std::uint64_t>(kHleStubs[idx]);
    }
    return reinterpret_cast<std::uint64_t>(&hle_unknown);
}

void* hle_wrap_fn(void* fn) { return wrap_hle_impl(fn); }

void hle_set_guest_stack(void* base, std::size_t size) {
    g_boot_stack_base = reinterpret_cast<std::uintptr_t>(base);
    g_boot_stack_size = size;
    host_log("guest boot stack %p size=0x%zx", base, size);
}

void* hle_thread_enter_guest() { return guest_thread_enter(); }

void hle_thread_leave_guest(void* tcb) { guest_thread_leave(tcb); }

// The slide of the 1.09 image, for reading a known guest object in a
// diagnostic (0 when the eboot is not the build the addresses belong to).
std::atomic<std::uint64_t> g_known_slide{0}, g_known_size{0};
std::uint64_t hle_image_slide() { return g_known_slide.load(std::memory_order_relaxed); }

// The pool the frame loop spins on when the game freezes a few seconds into
// a world: `data_58b8788` holds it, its entry count sits at +8 and
// the most it can hold at +0x80, and it is filled once - leaving the count at
// zero if that fill failed. Says nothing when the eboot is another build.
// How many of the pool's entries are free right now, -1 when it cannot be
// read: the watcher below logs every change, which says whether they leak
// one at a time or all at once.
int hle_guest_pool_free() {
    const std::uint64_t slide = g_known_slide.load(std::memory_order_relaxed);
    constexpr std::uint64_t kPoolHolderElf = 0x54b8788;
    if (!slide || kPoolHolderElf + 8 > g_known_size.load(std::memory_order_relaxed)) return -1;
    std::uint64_t pool = 0;
    std::memcpy(&pool, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(slide + kPoolHolderElf)), 8);
    if (!pool || !hle_kernel_va_mapped(pool, 0x88)) return -1;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(pool));
    std::int32_t count = 0;
    std::uint64_t flags = 0;
    std::memcpy(&count, bytes + 8, 4);
    std::memcpy(&flags, bytes + 0x50, 8);
    if (count <= 0 || count > 4096 || !flags || !hle_kernel_va_mapped(flags, static_cast<std::uint64_t>(count) * 4)) return -1;
    const auto* f = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(flags));
    int free_now = 0;
    for (int i = 0; i < count; ++i) free_now += f[i] != 0;
    // A second container would put its own pool in the global and leave the
    // entries of the first one to be released into a pool that never held
    // them - which the release quietly does nothing about.
    static std::atomic<std::uint64_t> seen{0};
    const std::uint64_t was = seen.exchange(pool, std::memory_order_relaxed);
    if (was && was != pool) {
        host_log("pool: the frame loop's pool changed from 0x%llx to 0x%llx", static_cast<unsigned long long>(was),
                 static_cast<unsigned long long>(pool));
    }
    return free_now;
}

void hle_guest_pool_report() {
    const std::uint64_t slide = g_known_slide.load(std::memory_order_relaxed);
    if (!slide) return;
    constexpr std::uint64_t kPoolHolderElf = 0x54b8788;  // Binary Ninja 0x58b8788
    const std::uint64_t holder = slide + kPoolHolderElf;
    if (kPoolHolderElf + 8 > g_known_size.load(std::memory_order_relaxed)) return;  // the image, not guest memory
    std::uint64_t pool = 0;
    std::memcpy(&pool, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(holder)), 8);
    if (!pool || !hle_kernel_va_mapped(pool, 0x88)) {
        host_log("hang: the frame loop's object pool is at 0x%llx (not readable)", static_cast<unsigned long long>(pool));
        return;
    }
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(pool));
    std::int32_t count = 0, most = 0;
    std::uint64_t flags = 0, objects = 0;
    std::memcpy(&count, bytes + 8, 4);
    std::memcpy(&most, bytes + 0x80, 4);
    std::memcpy(&objects, bytes + 0x28, 8);
    std::memcpy(&flags, bytes + 0x50, 8);
    int free_now = 0;
    if (count > 0 && count < 4096 && flags && hle_kernel_va_mapped(flags, static_cast<std::uint64_t>(count) * 4)) {
        const auto* f = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(flags));
        for (int i = 0; i < count; ++i) free_now += f[i] != 0;
    }
    host_log("hang: the frame loop's object pool at 0x%llx holds %d of at most %d (%d free; objects 0x%llx, flags 0x%llx)",
             static_cast<unsigned long long>(pool), count, most, free_now, static_cast<unsigned long long>(objects),
             static_cast<unsigned long long>(flags));
    // An entry nobody points at any more was taken and then dropped - the
    // frame loop will spin for it forever. Hand those back (the flag the pool
    // reads is one dword an entry), which turns the freeze into a hitch.
    // BBHOST_POOL_RECOVER=0 leaves them lost.
    // Off by default: handing an entry back buys the frame loop exactly one
    // flip before the next one is lost, so it is a way to watch the leak, not
    // a way out of it.
    static const bool recover = [] {
        const char* e = std::getenv("BBHOST_POOL_RECOVER");
        return e && e[0] == '1';
    }();
    if (free_now == 0 && count > 0 && count <= 16 && objects && flags &&
        hle_kernel_va_mapped(objects, static_cast<std::uint64_t>(count) * 8) &&
        hle_kernel_va_mapped(flags, static_cast<std::uint64_t>(count) * 4)) {
        const auto* slot = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(objects));
        std::uint64_t taken[16] = {};
        const int n = count < 16 ? count : 16;
        for (int i = 0; i < n; ++i) taken[i] = slot[i];
        int counts[16] = {};
        std::uint64_t outside[16] = {};
        hle_kernel_find_pointers(taken, n, 16, counts, outside);
        if (recover) {
            auto* f = reinterpret_cast<std::uint32_t*>(static_cast<std::uintptr_t>(flags));
            int given_back = 0;
            for (int i = 0; i < n; ++i) {
                // Only an entry whose every pointer is its own bookkeeping:
                // one that something still holds is not ours to hand out.
                if (outside[i] || !taken[i]) continue;
                __atomic_store_n(&f[i], 1u, __ATOMIC_RELEASE);
                ++given_back;
            }
            if (given_back) {
                host_log("HLE fake: %d of the pool's %d entries were taken and dropped, and nothing points at them any more; "
                         "handed back (one flip each, then they are lost again)", given_back, n);
            }
        }
    }
    // Each entry's head: its class (the vtable) and the first fields, so a
    // held one can be told from a free one.
    if (count > 0 && count <= 16 && objects && hle_kernel_va_mapped(objects, static_cast<std::uint64_t>(count) * 8)) {
        const auto* slot = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(objects));
        const auto* f = flags && hle_kernel_va_mapped(flags, static_cast<std::uint64_t>(count) * 4)
                            ? reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(flags))
                            : nullptr;
        for (int i = 0; i < count; ++i) {
            if (!slot[i] || !hle_kernel_va_mapped(slot[i], 0x20)) continue;
            const auto* q = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(slot[i]));
            host_log("hang:   entry %d at 0x%llx%s: %llx %llx %llx %llx", i, static_cast<unsigned long long>(slot[i]),
                     f && f[i] ? " (free)" : " (taken)", static_cast<unsigned long long>(q[0]),
                     static_cast<unsigned long long>(q[1]), static_cast<unsigned long long>(q[2]),
                     static_cast<unsigned long long>(q[3]));
            // The buffer the entry carries: its first bytes name what it is
            // (a file's magic, a command stream, a decoded block).
            if (q[1] && hle_kernel_va_mapped(q[1], 32)) {
                const auto* b = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(q[1]));
                char text[33] = {};
                for (int k = 0; k < 32; ++k) text[k] = b[k] >= 0x20 && b[k] < 0x7f ? static_cast<char>(b[k]) : '.';
                host_log("hang:     its buffer 0x%llx: %02x %02x %02x %02x %02x %02x %02x %02x  \"%s\"",
                         static_cast<unsigned long long>(q[1]), b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], text);
                // Its class, through the vtable the first field points at: the
                // methods are eboot addresses to look up.
                std::uint64_t vt = 0;
                std::memcpy(&vt, b, 8);
                if (i == 0 && vt && hle_kernel_va_mapped(vt, 32)) {
                    const auto* m = reinterpret_cast<const std::uint64_t*>(static_cast<std::uintptr_t>(vt));
                    host_log("hang:     vtable 0x%llx: %llx %llx %llx %llx", static_cast<unsigned long long>(vt),
                             static_cast<unsigned long long>(m[0]), static_cast<unsigned long long>(m[1]),
                             static_cast<unsigned long long>(m[2]), static_cast<unsigned long long>(m[3]));
                }
            }
        }
    }
}

void hle_patch_guest(ElfImage* image) {
    engine_bind(image);
    if (image->sha256 == kEboot109Sha256) {
        g_known_size.store(image->mem.size, std::memory_order_relaxed);
        g_known_slide.store(image->mem.slide, std::memory_order_relaxed);
    }
    if (image->sha256 != kEboot109Sha256) {
        host_log("eboot is not the known 1.09 build (sha256 %s); guest patches skipped",
                 image->sha256.c_str());
        return;
    }
    patch_manifests_apply(image);  // patches/*.toml: skip-intro, fov-uncap, a mod's own
    constexpr std::uint64_t kDlPanicElf = 0x20b55b0;
    const std::uint64_t va = image->mem.slide + kDlPanicElf;
    if (va < image->mem.slide || va + 16 > image->mem.slide + image->mem.size) {
        host_log("DL_PANIC va 0x%llx out of image", static_cast<unsigned long long>(va));
        return;
    }
    if (!guest_protect_rwx(&image->mem, va & ~0xfffull, 0x1000)) {
        host_log("cannot unprotect DL_PANIC page");
        return;
    }
    auto* p = static_cast<std::uint8_t*>(guest_ptr(image->mem, va));
    // Through the FS thunk: the panic logs with host libc.
    g_panic_slide = image->mem.slide;
    g_panic_size = image->mem.size;
    // Not wrap_hle_impl: the panic wants the guest's rsp and rbp, which only
    // survive if the capture runs before the thunk's own prologue.
    const std::uint64_t dest = reinterpret_cast<std::uint64_t>(
        thunk_wrap_capture_frame(reinterpret_cast<void*>(&hle_dl_panic)));
    p[0] = 0x48;
    p[1] = 0xb8;
    std::memcpy(p + 2, &dest, 8);
    p[10] = 0xff;
    p[11] = 0xe0;
    guest_protect_rx(&image->mem, va & ~0xfffull, 0x1000);
    host_log("patched DL_PANIC at 0x%llx -> host abort", static_cast<unsigned long long>(va));

    // The PC port's options, in the game's own System menu (engine/option_menu.h).
    option_menu_install(image);
    // The mouse in every menu list, DS3's way (engine/menu_pointer.h).
    menu_pointer_install(image);
    // The pad glyphs in the key guide and the tutorial notes become the keys
    // the player bound (engine/key_prompts.h).
    key_prompts_install(image);
    yebis_install(image);
    // The frame loop's pooled object, borrowed and given back as our source:
    // the game's own keeps the entry on a shared context and loses it
    // (engine/frame_pool.h). BBHOST_POOL_FIX=0 keeps the game's.
    frame_pool_install(image);
    graphics_patch_install(image);
    // The co-op invite as a type-1 item for the game's own summon manager
    // (engine/summon_invite.h): off, the game's own invite is used;
    // BBHOST_SUMMON_INVITE=1 brings ours back for experiments.
    summon_invite_install(image);
    np_test_install(image);
    gx_resources_install(image);
    // GX's blend, depth-stencil and rasterizer states as host objects
    // (engine/gx_state.h). BBHOST_GX_STATE=0 leaves them out, =2 compares.
    gx_state_install(image);
    params_install(image);
    lua_events_install(image);
    event_flags_install(image);
    player_data_install(image);
    world_chr_install(image);
    // How often the play log goes to the private server (engine/playlog.h).
    playlog_install(image);
    // How long a touched summon sign may take (engine/sos_signs.h).
    sos_signs_install(image);
    // The Hunter's Dream mirror, the developers' "Put on Disguise" (engine/change_appearance.h).
    change_appearance_install(image);
    // The Altar of Despair's "Rebirth in the Nightmare" (engine/rebirth.h).
    rebirth_install(image);
    five_players_install(image);
    // The game's functions as our source, in its place (decomp/, docs/decomp.md).
    decomp_add_areas();
    fmod_project_init::add_hooks();
    decomp_install(image);
    fmod_project_init::qualify();
    // After the list: the mouse turns our follow camera when it is in place,
    // else the game's by hooks (engine/mouse_camera.h).
    mouse_camera_install(image);

    // The command-arena acquire (guest 0x2ad3b80) becomes hle_gx_arena_acquire
    // (gnm_exec.cpp): the same scan, but when no chunk is free it waits for the
    // GPU to free one instead of failing, since the refill that crashes at
    // 0x2ab0a98 cannot handle a failure. BBHOST_ARENA_WAIT=0 keeps the original.
    if (const char* e = std::getenv("BBHOST_ARENA_WAIT"); !(e && e[0] == '0')) {
        constexpr std::uint64_t kArenaAcquire = 0x2ad3b80;  // guest VA
        constexpr std::uint8_t kPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53};
        const std::uint64_t at = image->mem.slide + (kArenaAcquire - kPreferredGuestSlide);
        auto* q = at >= image->mem.slide && at + 16 <= image->mem.slide + image->mem.size
                      ? static_cast<std::uint8_t*>(guest_ptr(image->mem, at))
                      : nullptr;
        if (!q || std::memcmp(q, kPrologue, sizeof(kPrologue)) != 0) {
            host_log("patch: command-arena acquire refused: unexpected bytes at 0x%llx", static_cast<unsigned long long>(at));
        } else if (!guest_protect_rwx(&image->mem, at & ~0xfffull, 0x1000)) {
            host_log("patch: cannot unprotect the command-arena acquire page");
        } else {
            const std::uint64_t host = reinterpret_cast<std::uint64_t>(wrap_hle_impl(reinterpret_cast<void*>(&hle_gx_arena_acquire)));
            q[0] = 0x48;  // mov rax, host
            q[1] = 0xb8;
            std::memcpy(q + 2, &host, 8);
            q[10] = 0xff;  // jmp rax
            q[11] = 0xe0;
            guest_protect_rx(&image->mem, at & ~0xfffull, 0x1000);
            host_log("patched the command-arena acquire at 0x%llx -> host (waits for a free chunk)", static_cast<unsigned long long>(at));
        }
    }

    // BBHOST_BP_TESS=1: int3 at the Scaleform shape tessellator (guest VA
    // 0x4b71b0) so we can read arg3, the mesh packing Matrix2x4.
    if (const char* e = std::getenv("BBHOST_BP_TESS"); e && e[0] == '1') {
        // BBHOST_BP_ELF=<hex> retargets the breakpoint (default: the tessellator
        // at guest 0x4b71b0). 0xb7ab0 is its caller sub_4b7ab0.
        std::uint64_t elf_off = 0xb71b0;
        if (const char* a = std::getenv("BBHOST_BP_ELF"); a && *a) {
            elf_off = std::strtoull(a, nullptr, 16);
        }
        hle_bp_arm(image->mem.slide + elf_off);
    }
    hle_gx_trace_arm(&image->mem);
}

void register_hle() {
    thunk_set_lookups(&hle_fn_name, &hle_image_slide);
    static bool once = false;
    if (once) {
        return;
    }
    once = true;
    hle_register_libc();
    hle_register_pthread();
    hle_register_kernel();
    hle_register_fs();
    hle_register_fios();
    hle_register_dialog();
    hle_register_avplayer();
    hle_register_video();
    hle_register_gnm();
    hle_register_audio();
    hle_register_net();
    hle_register_http();
    hle_register_ajm();
    hle_register_system();
    hle_register_np_matching2();
    hle_register_np_signaling();
}

void hle_register_all() { register_hle(); }
