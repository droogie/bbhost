// GX's CreateBuffer as source (docs/decomp.md): gx_create_buffer
// (0x2565c10), ID3D11Device::CreateBuffer as the engine's GX layer has it on
// the console. Dark Souls III's PC build keeps the same layer over D3D11, and
// its creator checks the description the same way before it calls the
// driver.
//
// The description is D3D11_BUFFER_DESC's shape: the byte width, the usage
// (0 default, 1 immutable, 2 dynamic, 3 staging), the bind flags (vertex 0x1,
// index 0x2, constant 0x4, shader resource 0x8, stream output 0x10, unordered
// access 0x80), the CPU access (bit 0 write, bit 1 read), the misc flags and
// the structure stride. The creator refuses it (8) when:
// - it has no bytes, misc flags outside what the layer makes, or the
//   depth-stencil bind;
// - a structured buffer's stride does not divide its bytes, is not a whole
//   number of dwords, or exceeds 0x800 bytes;
// - the device is not fully made, or a field is past its range;
// - the usage does not fit the rest: default without CPU access; immutable
//   with initial data, no CPU access and some of the low four binds; dynamic
//   with CPU writes, some of the low four binds and not shared; staging with
//   CPU access, no bind and none of misc 0x18;
// - a constant buffer binds as anything else, or two binds that exclude each
//   other are both set;
// - no allocator is left.
// Then the resource - the memory, the 24-bit id, the description, a constant
// buffer's rename slots, the initial data copied in - comes from
// gx_buffer_create_internal (0x256d420), and the holder from the object
// allocator: 0x58 bytes, the vtable, the allocator, a reference count of 1,
// and the resource at +0x18. A holder that cannot be allocated gives the
// resource back (9).
//
// Ours makes the same calls with the same arguments in the same order, and
// then hands what it made to the host (decomp/gx/resources.h): the GX
// resource registry, where its return hook did that before, and stage 1 of
// GX on Vulkan, which gives the buffer a host object and writes its handle
// in the holder (host/gxvk.h). The resource's dword at +0x2c - holder +0x44,
// which the game's left as whatever its frame held - and its byte at +0x29
// are 0 in ours.
//
// A leaf: it reads the game's memory and calls the game's functions, and of
// the host's only that entry, which switches to the host's FS itself.
#include "decomp/gx/resources.h"

#include "decomp/decomp.h"
#include "decomp/guest.h"

#include "core/thunk.h"
#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstring>

using namespace decomp;

// tests/decomp/gx_resources_test.cpp builds this file with mutants: small
// wrong turns it switches on one at a time and must catch.
#ifdef DECOMP_GX_RESOURCES_MUTANTS
extern int g_resources_mutant;
#define MUTANT(n) (g_resources_mutant == (n))
#else
#define MUTANT(n) false
#endif

void* decomp_gx::g_buffer_made_entry = nullptr;
void* decomp_gx::g_buffer_made_host = nullptr;

namespace {

constexpr u64 kCreateBuffer = 0x2565c10;
// push rbp; mov rbp, rsp; push r15, r14, r13, r12, rbx; sub rsp, 0x58
constexpr u8 kEntryCreateBuffer[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x58};

// What the creator calls and reads, by Binary Ninja address (symbols/1.09-names.tsv).
constexpr u64 kMakeResource = 0x256d420;    // gx_buffer_create_internal(resource*, -, desc*, initial data*, object allocator, memory allocator, placement*)
constexpr u64 kFreeRenameSlot = 0x2aae900;  // a constant buffer's rename slot given back in a context (context + 0xb3c0, slot)
constexpr u64 kContexts = 0x5aa63c0;        // GX's contexts by index (the context constructor's table); the creator walks the first 16
constexpr u64 kHolderVtable = 0x57b6240;    // the holders' base class, then the buffer's
constexpr u64 kBufferVtable = 0x575f3a0;
constexpr u64 kDeviceObjectAllocator = 0xdc10, kDeviceMemoryAllocator = 0xdc20;  // the device's two allocators
constexpr u64 kHolderBytes = 0x58, kResourceBytes = 0x40;

using MakeFn = GUEST_ABI u32 (*)(u64, u64, u64, u64, u64, u64, u64);
using AllocFn = GUEST_ABI u64 (*)(u64, u64, u64);
using FreeFn = GUEST_ABI void (*)(u64, u64);
using FreeSlotFn = GUEST_ABI void (*)(u64, u32);
using MadeFn = GUEST_ABI void (*)(u64, u64, u64, u64);

inline u64 vfunc(u64 object, u64 slot) { return load<u64>(load<u64>(object) + slot); }

// The creator's first half: whether the description passes (the creator
// returns 8 otherwise), and the allocators it goes on with. It only reads.
inline bool description_passes(u64 device, u64 desc, u64 initial_data, u64 object_override, u64 memory_override, u64& object_alloc,
                               u64& memory_alloc) {
    const u32 bytes = load<u32>(desc), bind = load<u32>(desc + 8), misc = load<u32>(desc + 0x10);
    bool ok = bytes != 0 && (misc & 0x87c5) == 0 && ((bind >> 6) & 1) == 0;
    if (misc & 0x20) {  // structured
        const u32 stride = load<u32>(desc + 0x14);
        if (MUTANT(1) ? stride == 0 || stride >= bytes : stride - 1 >= bytes) return false;  // u32: a stride of 0 wraps
        if ((bytes % stride) != 0 || (stride & 3) != 0) return false;
        if (stride > 0x800) ok = false;
    }
    if (!ok) return false;
    object_alloc = load<u64>(device + kDeviceObjectAllocator);
    memory_alloc = load<u64>(device + kDeviceMemoryAllocator);
    bool made = load<u64>(device) != 0 && load<u64>(device + 8) != 0 && load<u64>(device + 0x20) != 0;
    const u32 cpu = load<u32>(desc + 0xc), usage = load<u32>(desc + 4);
    if (bind > 0xff || cpu > 3 || misc > 0xffff) made = false;
    bool fits = false;
    switch (usage) {
    case 0:  // default
        fits = made && cpu == 0;
        break;
    case 1:  // immutable
        fits = (MUTANT(2) || initial_data != 0) && cpu == 0 && bind != 0 && (bind & 0xf0) == 0 && made;
        break;
    case 2:  // dynamic
        fits = (cpu & 1) != 0 && bind != 0 && (bind & 0xf0) == 0 && (misc & 2) == 0 && made;
        break;
    case 3:  // staging
        fits = bind == 0 && cpu != 0 && (misc & 0x18) == 0 && made;
        break;
    default:
        break;
    }
    bool refused;
    if ((bind & 4) != 0 && (bind & ~4u) != 0) refused = true;  // a constant buffer binds as nothing else
    else if ((bind & 0x60) == 0x60 || (bind & 0xc0) == 0xc0) refused = true;
    else refused = (bind & 0x90) == 0x90 || !fits;
    if (object_override) object_alloc = object_override;
    if (memory_override) memory_alloc = memory_override;
    return memory_alloc != 0 && object_alloc != 0 && !refused;
}

// Ours' own counts (leaf-safe: atomics of our globals): buffers made, and
// refusals by where.
std::atomic<u64> g_made{0}, g_refused_desc{0}, g_refused_resource{0}, g_refused_holder{0};

inline void tell_host(u64 holder, u64 desc, u64 initial_data, u32 result) {
    if (void* e = decomp_gx::g_buffer_made_entry) reinterpret_cast<MadeFn>(e)(holder, desc, initial_data != 0 ? 1 : 0, result);
}

}  // namespace

DECOMP_LEAF u32 gx_create_buffer(u64 device, u64 out, u64 desc, u64 initial_data, u64 object_override, u64 memory_override, u64 placement) {
    u64 object_alloc = 0, memory_alloc = 0;
    if (!description_passes(device, desc, initial_data, object_override, memory_override, object_alloc, memory_alloc)) {
        g_refused_desc.fetch_add(1, std::memory_order_relaxed);
        tell_host(0, desc, initial_data, 8);
        return 8;
    }
    // The resource, as the game's frame holds it: its allocator, memory, id,
    // description and rename slots (+0x30 and +0x38 zeroed before the call,
    // which may fail before it writes them).
    u64 resource[kResourceBytes / 8] = {};
    const u32 result = game_function<MakeFn>(kMakeResource)(reinterpret_cast<u64>(resource), 0, desc, initial_data, object_alloc,
                                                           memory_alloc, placement);
    if (result) {
        g_refused_resource.fetch_add(1, std::memory_order_relaxed);
        tell_host(0, desc, initial_data, result);
        return result;
    }
    const u64 allocator = MUTANT(3) ? memory_alloc : object_alloc;
    const u64 holder = reinterpret_cast<AllocFn>(vfunc(allocator, 0x58))(allocator, kHolderBytes, 8);
    if (!holder) {
        // No holder: the resource's memory and a constant buffer's rename
        // slots go back.
        store<u64>(out, 0);
        if (resource[0] && resource[1]) {
            reinterpret_cast<FreeFn>(vfunc(resource[0], 0x70))(resource[0], resource[1]);
            resource[1] = 0;
        }
        if (!MUTANT(4) && (reinterpret_cast<const u8*>(resource)[0x23] & 4) != 0) {
            const u64 slots = resource[7];
            for (u64 i = 0; i < 16; ++i) {
                const u64 ctx = load<u64>(game_address(kContexts) + 8 * i);
                if (ctx) game_function<FreeSlotFn>(kFreeRenameSlot)(ctx + 0xb3c0, load<u16>(slots + 2 * i));
            }
            if (resource[6] && slots) {
                reinterpret_cast<FreeFn>(vfunc(resource[6], 0x70))(resource[6], slots);
                resource[7] = 0;
            }
        }
        g_refused_holder.fetch_add(1, std::memory_order_relaxed);
        tell_host(0, desc, initial_data, 9);
        return 9;
    }
    store<u64>(holder, game_address(kHolderVtable));
    store<u64>(holder + 0x08, object_alloc);
    store<u32>(holder + 0x10, 1);
    store<u64>(holder, game_address(kBufferVtable));
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(holder + 0x18)), resource, MUTANT(5) ? kResourceBytes - 8 : kResourceBytes);
    store<u64>(out, holder);
    g_made.fetch_add(1, std::memory_order_relaxed);
    tell_host(holder, desc, initial_data, 0);
    return 0;
}

// ---- The compare run: the game's creator makes the buffer, ours must agree --

namespace {

void* g_game_create = nullptr;
DecompCompare g_cmp_create;

// Ours' first half against the game's whole call: a description ours refuses
// the game's refuses (8, *out untouched); one ours passes the game's makes,
// or refuses for the resource's or the holder's allocation (its 5, 8 or 9),
// and a holder it makes is ours' - the buffer's vtable, the object allocator,
// a count of 1, the resource's allocator (the memory allocator, the object
// allocator for misc 0x4000) and its object allocator at +0x48.
GUEST_ABI u32 compare_create_buffer(u64 device, u64 out, u64 desc, u64 initial_data, u64 object_override, u64 memory_override,
                                    u64 placement) {
    u64 object_alloc = 0, memory_alloc = 0;
    const bool passes = description_passes(device, desc, initial_data, object_override, memory_override, object_alloc, memory_alloc);
    const u64 out_before = out ? load<u64>(out) : 0;
    GuestCallRegs r{};
    r.fn = g_game_create;
    r.ints[0] = device;
    r.ints[1] = out;
    r.ints[2] = desc;
    r.ints[3] = initial_data;
    r.ints[4] = object_override;
    r.ints[5] = memory_override;
    r.stack[0] = placement;
    hle_call_guest_regs(&r);
    const u32 theirs = static_cast<u32>(r.ret);
    bool same = true;
    u64 holder = 0;
    if (!passes) {
        same = theirs == 8 && (!out || load<u64>(out) == out_before);
    } else if (theirs == 0) {
        holder = load<u64>(out);
        const u32 misc = load<u32>(desc + 0x10);
        same = holder != 0 && load<u64>(holder) == game_address(kBufferVtable) && load<u64>(holder + 0x08) == object_alloc &&
               load<u32>(holder + 0x10) == 1 && load<u64>(holder + 0x18) == ((misc & 0x4000) ? object_alloc : memory_alloc) &&
               load<u64>(holder + 0x48) == object_alloc && load<u8>(holder + 0x38) == 1;
    }
    g_cmp_create.calls.fetch_add(1, std::memory_order_relaxed);
    if (!same && g_cmp_create.differ.fetch_add(1, std::memory_order_relaxed) < 8) {
        host_log("decomp: gx_create_buffer differs: the game's returned %u%s for a description ours %s (%u bytes, usage %u, bind 0x%x, cpu %u, "
                 "misc 0x%x)",
                 theirs, theirs == 0 ? " and a holder unlike ours" : "", passes ? "passes" : "refuses", load<u32>(desc), load<u32>(desc + 4),
                 load<u32>(desc + 8), load<u32>(desc + 0xc), load<u32>(desc + 0x10));
    }
    // Hosted here: the host's function itself, not its thunk.
    if (void* h = decomp_gx::g_buffer_made_host) reinterpret_cast<MadeFn>(h)(holder, desc, initial_data != 0 ? 1 : 0, theirs);
    return theirs;
}

void report() {
    if (!g_made.load() && !g_refused_desc.load() && !g_refused_resource.load() && !g_refused_holder.load()) return;  // the compare run's
    host_log("decomp: gx_create_buffer made %llu buffers; refused %llu descriptions, %llu for the resource, %llu for the holder",
             static_cast<ull>(g_made.load()), static_cast<ull>(g_refused_desc.load()), static_cast<ull>(g_refused_resource.load()),
             static_cast<ull>(g_refused_holder.load()));
}

}  // namespace

void decomp_gx_resources_add() {
    DecompFunction fn{"gx_create_buffer", "gx", kCreateBuffer, kEntryCreateBuffer, sizeof(kEntryCreateBuffer),
                      reinterpret_cast<void*>(&gx_create_buffer), DecompKind::Leaf};
    fn.original = &g_game_create;
    fn.compare = reinterpret_cast<void*>(&compare_create_buffer);
    fn.counts = &g_cmp_create;
    fn.report = &report;
    decomp_add(fn);
}
