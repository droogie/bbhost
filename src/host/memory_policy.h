#pragma once
// How bbhost holds the game's memory on the GPU it runs on: one decision,
// made once for the device (gpu.cpp host_gpu_memory_policy, from the device's
// kind as first known - the probe before the device is made, else the
// device's own) and logged once (`gpu: memory: ...`). Four parts:
// - GX's buffers bound by handle from copies of their own, or today's path
//   (BBHOST_GXVK_BUFFERS; host/gxvk.h buffers_mode_for);
// - a budget for those copies (BBHOST_GXVK_BUFFERS_BUDGET_MB;
//   host/gxvk_budget.h budget_bytes);
// - the game's memory imported whole or a 16 MB window at a time
//   (BBHOST_IMPORT_WINDOWS, gpu.import_windows; host/import_mode.h decide);
// - the upload staging kept or given back down to its floor in play
//   (BBHOST_STAGING_RETURN, gpu.staging_return; host/apu_memory.h
//   staging_return).
// Each part's switch wins on any device. Unset, the device decides:
//
//   the device                                  GX's buffers         imports          upload staging
//   a discrete GPU with room (7.5 GiB+ free)    by handle            whole            kept
//   one with small video memory                 by handle, budgeted  by window        given back (**)
//   one with tight memory (heaps under 12 GiB)  today's path         whole            kept
//   an integrated GPU with 8 GiB or more        by handle            by platform (*)  by platform (*)
//   a smaller integrated GPU (a Steam Deck)     today's path         by platform (*)  by platform (*)
//   (*) by window and given back under Windows; whole and kept under Linux,
//   where the Steam Deck's kit asks for window imports in its own toml.
//   (**) kept on NVIDIA's driver, whose free of a 64 MiB staging chunk stalls
//   the frame it falls in.
//
// Why: a card with room has the video memory for the copies and the PCIe
// link for whole imports, and its staging is kept: each 64 MiB chunk's
// vkFreeMemory took 22-29 ms on an RTX 4070, and the frame it fell in ran
// 29-47 ms against 14-17 around it, one every two seconds for the 16 s after
// every load (gpu.cpp start_memory_reserve) - on NVIDIA's driver a small card
// keeps it too, for the ~0.5 GiB of system memory the return would give
// back; AMD's frees one in 0.4-3.7 ms (a Radeon 680M under Windows), so a
// small card there gives it back. A small card's whole imports pin 4-5 GiB of
// system memory beside it, which Windows counts as shared GPU memory, and
// without the copies the buffer shadow's mirrors carried the meshes: refused
// room at a 4 GB card's budget, 16.8 M draws read their vertices across the
// bus. The copies there are budgeted (an eighth of the video memory free at
// the start, 256-512 MiB): the least recently bound go first, and the pool
// gives room back before the mirrors are refused. An integrated GPU's video
// memory is the system's: Windows' driver pins whole imports and counts them
// as shared (a Radeon 680M: 8.0 -> 6.2 GB with windows at the same frame
// rate); a smaller one reads GX's buffers where they lie, and one with 8 GiB
// or more (a Strix Halo's carve-out) reads them from copies in the carve-out:
// the Radeon 8060S's frame took ~7.5% less GPU time for 256 MiB of copies
// (docs/list-resolve.md). A discrete GPU with tight memory holds nothing
// more than it must.
//
// Kept apart from Vulkan so a test can drive it.
#include <cstdint>
#include <cstdio>
#include <string>

#include "host/apu_memory.h"
#include "host/gxvk.h"
#include "host/gxvk_budget.h"
#include "host/import_mode.h"

namespace memory_policy {

// The device as the defaults see it (gxvk.h's classes).
using Device = gxvk::BuffersDevice;

// From the device's kind: integrated first (with 8 GiB of video memory or
// more, and not tight, a roomy one), then tight memory, then small video
// memory (gpu.cpp small_vram: a discrete GPU under 7.5 GiB free at the start).
inline Device device_of(bool integrated, bool integrated_room, bool tight, bool small_vram) {
    if (integrated) return integrated_room && !tight ? Device::kIntegratedRoomy : Device::kIntegrated;
    if (tight) return Device::kTight;
    return small_vram ? Device::kSmallVram : Device::kRoomy;
}
inline bool is_integrated(Device d) { return d == Device::kIntegrated || d == Device::kIntegratedRoomy; }

// The switches as given (null or empty: unset): the environment's, and the
// config's for the two parts the config has.
struct Switches {
    const char* buffers = nullptr;      // BBHOST_GXVK_BUFFERS
    const char* budget = nullptr;       // BBHOST_GXVK_BUFFERS_BUDGET_MB
    const char* imports_env = nullptr;  // BBHOST_IMPORT_WINDOWS
    const char* imports_cfg = nullptr;  // gpu.import_windows
    const char* staging_env = nullptr;  // BBHOST_STAGING_RETURN
    const char* staging_cfg = nullptr;  // gpu.staging_return
    const char* small = nullptr;        // BBHOST_SMALL_VRAM (only said: gpu.cpp small_vram reads it for the device's kind)
};
inline bool given(const char* s) { return s && *s; }

struct Policy {
    Device device = Device::kRoomy;
    bool windows_host = false;
    bool nvidia = false;                        // NVIDIA's driver (the staging's return)
    std::uint64_t free_at_start = 0;            // video memory free for us at the start, bytes (the budget's base)
    gxvk::Buffers buffers = gxvk::Buffers::kOff;  // GX's buffers
    std::uint64_t budget = 0;                   // their copies' budget in bytes, 0 none
    bool windows = false;                       // the game's memory imported a window at a time
    bool staging_return = false;                // the upload staging given back in play
};

// `nvidia`: the device's driver is NVIDIA's (PCI vendor 0x10de).
inline Policy decide(Device d, bool windows_host, std::uint64_t free_at_start, bool nvidia, const Switches& s) {
    Policy p;
    p.device = d;
    p.windows_host = windows_host;
    p.nvidia = nvidia;
    p.free_at_start = free_at_start;
    const int integrated = is_integrated(d) ? 1 : 0, small = d == Device::kSmallVram ? 1 : 0;
    p.buffers = gxvk::buffers_mode_for(s.buffers, d);
    // A budget only where the copies are made, and never on an integrated GPU (its video memory is the system's: no
    // reserve trims it, and BBHOST_GXVK_BUFFERS_BUDGET_MB is for a discrete GPU).
    p.budget = p.buffers != gxvk::Buffers::kOff && !integrated ? gxvk_budget::budget_bytes(s.budget, small == 1, free_at_start) : 0;
    p.windows = import_mode::decide(s.imports_env, s.imports_cfg, integrated, windows_host, small) == import_mode::Mode::kWindows;
    // The staging's return on a small card only where the driver frees a chunk without stalling a frame.
    p.staging_return = apu_memory::staging_return(s.staging_env, s.staging_cfg, integrated, windows_host, small && !nvidia ? 1 : 0);
    return p;
}

// The log's line for it (after `gpu: memory: `): the device, then each part and what decided it.
inline std::string describe(const Policy& p, const Switches& s) {
    const unsigned long long free_mib = p.free_at_start >> 20;
    char b[640];
    std::string out;
    switch (p.device) {
        case Device::kRoomy:
            if (free_mib)
                std::snprintf(b, sizeof b, "a graphics card with room in its video memory (%llu MiB free at the start)", free_mib);
            else
                std::snprintf(b, sizeof b, "a graphics card with room in its video memory");  // (no device asked: BBHOST_GPU=0)
            break;
        case Device::kSmallVram:
            std::snprintf(b, sizeof b,
                          "a graphics card with small video memory (%llu MiB free at the start, %s): no spare image heap blocks in play, "
                          "less untile scratch kept",
                          free_mib, given(s.small) ? "BBHOST_SMALL_VRAM" : "under 7680; BBHOST_SMALL_VRAM=0 says otherwise");
            break;
        case Device::kIntegratedRoomy:
            std::snprintf(b, sizeof b, "an integrated GPU with 8 GiB of video memory or more");
            break;
        case Device::kIntegrated:
            std::snprintf(b, sizeof b, "an integrated GPU");
            break;
        case Device::kTight:
            std::snprintf(b, sizeof b, "a graphics card with tight memory (under 12 GiB in all its heaps)");
            break;
    }
    out = b;
    // GX's buffers and their budget.
    const char* by = given(s.buffers) ? " (BBHOST_GXVK_BUFFERS)" : "";
    switch (p.buffers) {
        case gxvk::Buffers::kOn:
        case gxvk::Buffers::kCompare:
            if (p.budget) {
                std::snprintf(b, sizeof b,
                              "; GX's buffers bound by handle%s%s, their copies within a budget of %llu MiB (%s), the least recently bound "
                              "going first",
                              p.buffers == gxvk::Buffers::kCompare ? " and compared with today's path" : "", by,
                              static_cast<unsigned long long>(p.budget >> 20),
                              given(s.budget) ? "BBHOST_GXVK_BUFFERS_BUDGET_MB" : "an eighth of the free memory, 256-512 MiB");
            } else {
                std::snprintf(b, sizeof b, "; GX's buffers bound by handle%s%s%s",
                              p.buffers == gxvk::Buffers::kCompare ? " and compared with today's path" : "", by,
                              is_integrated(p.device) ? ", from their copies in its video memory" : "");
            }
            break;
        case gxvk::Buffers::kOff:
            std::snprintf(b, sizeof b, "; GX's buffers on today's path%s", by[0] ? by : ", read where they are");
            break;
    }
    out += b;
    // The imports.
    const char* imp = given(s.imports_env) ? " (BBHOST_IMPORT_WINDOWS)" : given(s.imports_cfg) ? " (gpu.import_windows)" : "";
    std::snprintf(b, sizeof b, "; the game's memory imported %s%s", p.windows ? "a 16 MB window at a time as the GPU first needs it" : "whole", imp);
    out += b;
    // The upload staging.
    const char* st = given(s.staging_env) ? " (BBHOST_STAGING_RETURN)" : given(s.staging_cfg) ? " (gpu.staging_return)" : "";
    std::snprintf(b, sizeof b, "; the upload staging %s%s%s", p.staging_return ? "given back down to its floor in play" : "kept", st,
                  !p.staging_return && !st[0] && p.device == Device::kSmallVram && p.nvidia
                      ? ", as NVIDIA's driver stalls the frame a staging chunk's free falls in"
                      : "");
    out += b;
    out += " - BBHOST_GXVK_BUFFERS, BBHOST_GXVK_BUFFERS_BUDGET_MB, BBHOST_IMPORT_WINDOWS and BBHOST_STAGING_RETURN change a part (docs/running.md)";
    return out;
}

}  // namespace memory_policy
