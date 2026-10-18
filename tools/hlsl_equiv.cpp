// hlsl_equiv: a lifted shader and its HLSL path's module, run on the same
// random inputs, their outputs compared bit for bit.
//
//   hlsl_equiv [--size N] [--vertices N] [--seed S] [--shard I/N] [--list FILE] [--only SUBSTR]
//              [--small-ints] [--float-words] [--timeout S] [--progress FILE] <dir>...
//   hlsl_equiv --compile [--only SUBSTR] [--list FILE] <dir>...
//              every compute module (*.spv) under the folders built into a
//              pipeline against a layout from its own bindings, the device's
//              verdict printed: a driver that refuses DXC's module of a lift
//              it takes shows here, before a player meets it.
//
// <dir> holds what `gcn2hlsl --out` writes: <bundle>/<program>.<stage>.lift.spv
// and .dxc.spv. For each pixel or vertex shader pair the tool builds one
// draw around the stage from the lifted module's interface alone - its
// descriptor bindings filled with random words (the params block's buffer
// biases 0, its ranges and strides set so loads stay inside the buffers),
// random textures with mip levels, samplers, random vertex attributes - and
// runs it twice, once with each module:
//   pixel shader: a full-screen triangle from a generated vertex shader whose
//     outputs are random per vertex, interpolated as the inputs ask (flat,
//     noperspective, centroid); every colour output and the depth output read
//     back;
//   vertex shader (and a tessellation LS, whose control point is an array
//     output): points through a generated geometry shader that turns each
//     vertex's outputs (position, every location, clip distances, point size)
//     into a row of texels, written raw;
//   domain shader: a generated hull gives four patches random control points
//     and patch constants and the factors 3, in point mode; each tessellator
//     point's outputs go through the same geometry shader;
//   hull shader: a generated vertex shader gives four patches random control
//     points; a generated domain shader (point mode; the domain, spacing and
//     winding DXC's module declares, as the lift's hull declares none) writes
//     every control point's outputs, the patch constants and the factors the
//     hull made into a buffer, which is compared.
// The generated stages are HLSL compiled by DXC (host/dxc.h). Run it on
// lavapipe (VK_DRIVER_FILES=<lvp_icd.json>): the CPU driver evaluates both
// modules the same way every time, so a difference is the modules', not the
// scheduling's. Prints one line per program and the totals; exit 0 when no
// pair differed.
#include "host/dxc.h"
#include "gcn/hlsl.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <atomic>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& why) { throw Failure(why); }
void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) fail(std::string(what) + " failed (" + std::to_string(static_cast<int>(r)) + ")");
}
struct Skip : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::vector<std::uint32_t> read_words(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<std::uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), w.size() * 4);
    return w;
}

// ---- the module's interface ------------------------------------------------------
struct TypeInfo {
    std::uint16_t op = 0;
    std::vector<std::uint32_t> w;  // the instruction's words after the result id
};
struct Var {
    std::uint32_t id = 0, sc = 0, pointee = 0;
    std::map<std::uint32_t, std::vector<std::uint32_t>> dec;
    bool has(std::uint32_t d) const { return dec.count(d) != 0; }
    std::uint32_t get(std::uint32_t d) const { return dec.at(d).empty() ? 0 : dec.at(d)[0]; }
};
struct Module {
    std::uint32_t model = 0;
    std::map<std::uint32_t, TypeInfo> types;
    std::map<std::uint32_t, std::uint32_t> consts;  // id -> value (32-bit)
    std::vector<Var> vars;
    std::set<std::uint32_t> interface_ids;
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> member_offset;
    std::set<std::uint32_t> dref_samplers;  // sampler variables used in a depth-compare sample
    std::set<std::uint32_t> plain_samplers;
    std::set<std::uint32_t> caps;
    std::map<std::uint32_t, std::vector<std::uint32_t>> modes;  // OpExecutionMode: mode -> its literals
};

Module parse(const std::vector<std::uint32_t>& w) {
    Module m;
    std::map<std::uint32_t, std::map<std::uint32_t, std::vector<std::uint32_t>>> dec;
    std::map<std::uint32_t, std::uint32_t> load_of;          // OpLoad result -> variable
    std::map<std::uint32_t, std::uint32_t> sampled_sampler;  // OpSampledImage result -> sampler variable
    for (std::size_t i = 5; i < w.size();) {
        const std::uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || i + n > w.size()) fail("malformed module");
        const std::uint32_t* a = &w[i + 1];
        switch (op) {
        case 17: m.caps.insert(a[0]); break;
        case 16: m.modes[a[1]] = std::vector<std::uint32_t>(a + 2, a + n - 1); break;  // OpExecutionMode
        case 15: {  // OpEntryPoint
            m.model = a[0];
            std::size_t k = 2;
            for (; k < n - 1; ++k) {
                const std::uint32_t x = a[k];
                if (!(x & 0xffu) || !(x & 0xff00u) || !(x & 0xff0000u) || !(x & 0xff000000u)) {
                    ++k;
                    break;
                }
            }
            for (; k < n - 1; ++k) m.interface_ids.insert(a[k]);
            break;
        }
        case 71: dec[a[0]][a[1]] = std::vector<std::uint32_t>(a + 2, a + n - 1); break;
        case 72:
            if (n >= 5 && a[2] == 35) m.member_offset[{a[0], a[1]}] = a[3];
            break;
        case 43: m.consts[a[1]] = a[2]; break;  // OpConstant
        case 59: {
            Var v;
            v.id = a[1];
            v.sc = a[2];
            v.pointee = m.types.count(a[0]) ? m.types[a[0]].w.at(1) : 0;
            m.vars.push_back(v);
            break;
        }
        case 61: load_of[a[1]] = a[2]; break;                      // OpLoad
        case 86: sampled_sampler[a[1]] = load_of.count(a[3]) ? load_of[a[3]] : 0; break;  // OpSampledImage
        case 89: case 90: case 97:                                 // Dref samples, DrefGather
            if (sampled_sampler.count(a[2])) m.dref_samplers.insert(sampled_sampler[a[2]]);
            break;
        case 87: case 88: case 96: case 105:  // other samples, gather, QueryLod
            if (sampled_sampler.count(a[2])) m.plain_samplers.insert(sampled_sampler[a[2]]);
            break;
        default:
            if (op >= 19 && op <= 39) {
                TypeInfo t;
                t.op = static_cast<std::uint16_t>(op);
                t.w.assign(a + 1, a + n - 1);
                m.types[a[0]] = t;
            }
            break;
        }
        i += n;
    }
    for (Var& v : m.vars) v.dec = dec[v.id];
    return m;
}

// The scalar kind and component count of a stage variable's type ('f', 'u', 'i').
bool scalar_shape(const Module& m, std::uint32_t t, char& kind, std::uint32_t& comps) {
    const TypeInfo& ty = m.types.at(t);
    comps = 1;
    std::uint32_t s = t;
    if (ty.op == 23) {
        s = ty.w[0];
        comps = ty.w[1];
    }
    const TypeInfo& st = m.types.at(s);
    if (st.op == 22 && st.w[0] == 32) kind = 'f';
    else if (st.op == 21 && st.w[0] == 32) kind = st.w[1] ? 'i' : 'u';
    else if (st.op == 20) kind = 'b';
    else return false;
    return true;
}

// ---- a hull's outputs, stored by the hull ------------------------------------------
// A hull's factors decide whether its patch reaches the domain shader at all,
// and random inputs give the game's hulls factors that cull every patch. So
// the hull stores what it made itself: before each return of the entry
// point, every invocation writes its control point's outputs, then the patch
// constants and the factors, to a storage buffer (set `set`, binding 0), at
// record (primitive ID x output points + invocation ID). Both modules are
// changed the same way, finding their outputs by location and built-in, not
// by id: per-control-point outputs by location, patch outputs by location,
// TessLevelOuter[4], TessLevelInner[2]; every component as a uint.
struct HullRecord {
    std::uint32_t cp_words = 0, patch_words = 0;
    std::uint32_t words() const { return cp_words + patch_words; }
};
std::vector<std::uint32_t> instrument_hull(const std::vector<std::uint32_t>& in, std::uint32_t set, std::uint32_t out_points, HullRecord& rec) {
    struct In {
        std::size_t at;
        std::uint32_t op, n;
    };
    std::vector<In> list;
    for (std::size_t i = 5; i < in.size();) {
        const std::uint32_t n = in[i] >> 16, op = in[i] & 0xffff;
        if (!n || i + n > in.size()) fail("malformed module");
        list.push_back({i, op, n});
        i += n;
    }
    std::uint32_t bound = in[3];
    std::map<std::uint32_t, std::vector<std::uint32_t>> types;  // id -> opcode, operands
    std::map<std::uint32_t, std::map<std::uint32_t, std::uint32_t>> deco;
    struct Out {
        std::uint32_t var, pointee;
    };
    std::vector<Out> outputs;
    std::map<std::uint32_t, std::uint32_t> var_pointee, var_class;
    std::uint32_t entry_fn = 0, uint_t = 0;
    std::size_t entry_at = 0, last_annotation = 0, first_function = 0;
    std::map<std::uint32_t, std::uint32_t> int_consts;  // value -> id, of type uint_t (found after uint_t)
    for (const In& x : list) {
        const std::uint32_t* a = &in[x.at + 1];
        if (x.op == 15 && !entry_at) {  // OpEntryPoint
            entry_at = x.at;
            entry_fn = a[1];
        } else if (x.op == 71 || x.op == 72 || x.op == 332 || x.op == 5632 || x.op == 5633) {
            last_annotation = x.at + x.n;
            if (x.op == 71) deco[a[0]][a[1]] = x.n > 3 ? a[2] : 0;
        } else if (x.op >= 19 && x.op <= 39) {
            types[a[0]] = std::vector<std::uint32_t>(&in[x.at], &in[x.at] + x.n);
            types[a[0]][0] = x.op;
            if (x.op == 21 && a[1] == 32 && a[2] == 0 && !uint_t) uint_t = a[0];
        } else if (x.op == 59) {  // OpVariable
            const std::uint32_t pointee = types.count(a[0]) ? types[a[0]].at(3) : 0;
            var_pointee[a[1]] = pointee;
            var_class[a[1]] = a[2];
            if (a[2] == 3) outputs.push_back({a[1], pointee});
        } else if (x.op == 54 && !first_function) {
            first_function = x.at;
        }
    }
    if (!entry_at || !first_function || !last_annotation) fail("a hull module without its entry point, functions or decorations");
    {
        int returns = 0;
        bool inside = false;
        for (const In& x : list) {
            if (x.op == 54) inside = in[x.at + 2] == entry_fn;
            if (x.op == 253 && inside) ++returns;
        }
        if (returns != 1) fail("a hull entry point with " + std::to_string(returns) + " returns");
    }
    std::vector<std::uint32_t> head, globals, decorations, iface;  // what goes in
    const auto emit = [](std::vector<std::uint32_t>& v, std::uint32_t op, std::initializer_list<std::uint32_t> ops) {
        v.push_back(static_cast<std::uint32_t>(ops.size() + 1) << 16 | op);
        v.insert(v.end(), ops);
    };
    if (!uint_t) {
        uint_t = bound++;
        emit(globals, 21, {uint_t, 32, 0});
    }
    const auto constant = [&](std::uint32_t v) {
        const auto it = int_consts.find(v);
        if (it != int_consts.end()) return it->second;
        const std::uint32_t id = bound++;
        emit(globals, 43, {uint_t, id, v});
        int_consts[v] = id;
        return id;
    };
    const auto pointer = [&](std::uint32_t sc, std::uint32_t t) {
        const std::uint32_t id = bound++;
        emit(globals, 32, {id, sc, t});
        return id;
    };
    // The buffer.
    const std::uint32_t rta = bound++, st = bound++, ssbo = bound++;
    emit(globals, 29, {rta, uint_t});
    emit(globals, 30, {st, rta});
    const std::uint32_t ptr_st = pointer(12, st), ptr_u = pointer(12, uint_t);
    emit(globals, 59, {ptr_st, ssbo, 12});
    emit(decorations, 71, {rta, 6, 4});        // ArrayStride 4
    emit(decorations, 71, {st, 2});            // Block
    emit(decorations, 72, {st, 0, 35, 0});     // Offset 0
    emit(decorations, 71, {ssbo, 34, set});    // DescriptorSet
    emit(decorations, 71, {ssbo, 33, 0});      // Binding
    if (in[1] >= 0x00010400u) iface.push_back(ssbo);
    // The invocation's and the patch's IDs: the module's built-ins, or new ones.
    const auto builtin_input = [&](std::uint32_t b) {
        for (const auto& [var, c] : var_class) {
            if (c == 1 && deco.count(var) && deco[var].count(11) && deco[var][11] == b) return var;
        }
        const std::uint32_t var = bound++;
        emit(globals, 59, {pointer(1, uint_t), var, 1});
        emit(decorations, 71, {var, 11, b});
        iface.push_back(var);
        var_pointee[var] = uint_t;
        return var;
    };
    const std::uint32_t inv_var = builtin_input(8), prim_var = builtin_input(7);
    // The values, in order: (variable, pointee, per control point).
    struct Val {
        std::uint32_t var, pointee, location;
        bool per_cp;
        int builtin;
    };
    std::vector<Val> vals;
    for (const Out& o : outputs) {
        const auto& d = deco[o.var];
        if (d.count(11)) {
            if (d.at(11) == 11 || d.at(11) == 12) vals.push_back({o.var, o.pointee, d.at(11) == 11 ? 1000u : 1001u, false, static_cast<int>(d.at(11))});
            else fail("a hull output built-in other than the factors");
            continue;
        }
        if (!d.count(30)) continue;
        vals.push_back({o.var, o.pointee, d.at(30), !d.count(15), -1});
    }
    std::stable_sort(vals.begin(), vals.end(), [](const Val& x, const Val& y) {
        if (x.per_cp != y.per_cp) return x.per_cp;
        return x.location < y.location;
    });
    // Element types: (type, count) after the per-control-point level.
    const auto shape = [&](std::uint32_t t, std::uint32_t& count) {
        count = 0;
        if (types.at(t)[0] == 28) {
            const std::uint32_t len_id = types.at(t)[3];
            std::uint32_t len = 0;
            for (const In& x : list) {
                if (x.op == 43 && in[x.at + 2] == len_id) len = in[x.at + 3];
            }
            if (!len) fail("an output array of unknown length");
            count = len;
            t = types.at(t)[2];
        }
        return t;
    };
    std::vector<std::uint32_t> code;  // before each OpReturn of the entry point
    const auto load_u = [&](std::uint32_t var) {
        const std::uint32_t t = var_pointee.at(var), v = bound++;
        emit(code, 61, {t, v, var});
        if (t == uint_t) return v;
        const std::uint32_t u = bound++;
        emit(code, 124, {uint_t, u, v});  // OpBitcast
        return u;
    };
    const std::uint32_t inv = load_u(inv_var), prim = load_u(prim_var);
    const std::uint32_t t1 = bound++, slot = bound++;
    emit(code, 132, {uint_t, t1, prim, constant(out_points)});  // OpIMul
    emit(code, 128, {uint_t, slot, t1, inv});                    // OpIAdd
    std::vector<std::uint32_t> store_code;  // the stores; the record's size is known after
    std::uint32_t word = 0;
    const std::uint32_t base = bound++;
    const auto store = [&](std::uint32_t value) {
        const std::uint32_t addr = bound++, ptr = bound++;
        emit(store_code, 128, {uint_t, addr, base, constant(word++)});
        emit(store_code, 65, {ptr_u, ptr, ssbo, constant(0), addr});
        emit(store_code, 62, {ptr, value});
    };
    std::map<std::uint32_t, std::uint32_t> out_ptr;  // element type -> Output pointer
    for (const Val& v : vals) {
        std::uint32_t t = v.pointee, count = 0;
        if (v.per_cp) {
            if (types.at(t)[0] != 28) fail("a per-control-point output that is not an array");
            t = types.at(t)[2];
        }
        const std::uint32_t elem = shape(t, count);
        const auto& et = types.at(elem);
        std::uint32_t comps = 1, comp_t = elem;
        if (et[0] == 23) {
            comps = et[3];
            comp_t = et[2];
        }
        if (types.at(comp_t)[0] != 21 && types.at(comp_t)[0] != 22) fail("an output of another component type");
        if (!out_ptr.count(elem)) out_ptr[elem] = pointer(3, elem);
        for (std::uint32_t k = 0; k < (count ? count : 1); ++k) {
            std::vector<std::uint32_t> idx;
            if (v.per_cp) idx.push_back(inv);
            if (count) idx.push_back(constant(k));
            const std::uint32_t p = bound++, x = bound++;
            store_code.push_back(static_cast<std::uint32_t>(4 + idx.size()) << 16 | 65);  // OpAccessChain
            store_code.push_back(out_ptr[elem]);
            store_code.push_back(p);
            store_code.push_back(v.var);
            store_code.insert(store_code.end(), idx.begin(), idx.end());
            emit(store_code, 61, {elem, x, p});
            for (std::uint32_t c = 0; c < comps; ++c) {
                std::uint32_t s = x;
                if (comps > 1) {
                    s = bound++;
                    emit(store_code, 81, {comp_t, s, x, c});  // OpCompositeExtract
                }
                if (comp_t != uint_t) {
                    const std::uint32_t u = bound++;
                    emit(store_code, 124, {uint_t, u, s});
                    s = u;
                }
                store(s);
            }
        }
        if (v.per_cp) rec.cp_words = word;
    }
    rec.patch_words = word - rec.cp_words;
    emit(code, 132, {uint_t, base, slot, constant(rec.words())});
    code.insert(code.end(), store_code.begin(), store_code.end());
    // The module again, with the additions.
    std::vector<std::uint32_t> out(in.begin(), in.begin() + 5);
    bool in_entry = false;
    for (const In& x : list) {
        if (x.at == last_annotation) out.insert(out.end(), decorations.begin(), decorations.end());
        if (x.at == first_function) out.insert(out.end(), globals.begin(), globals.end());
        if (x.op == 54) in_entry = in[x.at + 2] == entry_fn;
        if (x.op == 253 && in_entry) out.insert(out.end(), code.begin(), code.end());  // before OpReturn
        if (x.at == entry_at) {
            std::vector<std::uint32_t> ep(&in[x.at], &in[x.at] + x.n);
            ep.insert(ep.end(), iface.begin(), iface.end());
            ep[0] = static_cast<std::uint32_t>(ep.size()) << 16 | 15;
            out.insert(out.end(), ep.begin(), ep.end());
            continue;
        }
        out.insert(out.end(), &in[x.at], &in[x.at] + x.n);
    }
    out[3] = bound;
    return out;
}

// ---- Vulkan ----------------------------------------------------------------------
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* map = nullptr;
    VkDeviceSize size = 0;
};
struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0, height = 0, layers = 1, levels = 1;
};

struct Device {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem{};
    std::string name;
    bool quad_ops = false, runtime_offsets = false;
    std::string robust = "robustness2 unavailable";

    void init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "hlsl_equiv";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");
        std::uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance, &n, devs.data());
        std::uint32_t family = 0;
        for (VkPhysicalDevice d : devs) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.apiVersion < VK_API_VERSION_1_3) continue;
            std::uint32_t nq = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
            std::vector<VkQueueFamilyProperties> qs(nq);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
            for (std::uint32_t i = 0; i < nq; ++i) {
                if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    phys = d;
                    family = i;
                    name = p.deviceName;
                    break;
                }
            }
            if (phys) break;
        }
        if (!phys) fail("no Vulkan 1.3 device");
        vkGetPhysicalDeviceMemoryProperties(phys, &mem);
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(phys, &p2);
        quad_ops = (sg.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) && (sg.supportedStages & VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPhysicalDeviceFeatures supported{};
        vkGetPhysicalDeviceFeatures(phys, &supported);
        // Random words index buffers and texels out of range: every access
        // bounds-checked, an out-of-range one reading 0 (robustness2), the same
        // for both modules.
        std::uint32_t next = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, nullptr);
        std::vector<VkExtensionProperties> props(next);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, props.data());
        bool r2 = false, m8 = false;
        for (const VkExtensionProperties& e : props) {
            r2 |= std::strcmp(e.extensionName, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME) == 0;
            m8 |= std::strcmp(e.extensionName, VK_KHR_MAINTENANCE_8_EXTENSION_NAME) == 0;
        }
        VkPhysicalDeviceVulkan13Features q13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceRobustness2FeaturesEXT qr2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
        VkPhysicalDeviceMaintenance8FeaturesKHR q8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
        q13.pNext = &qr2;
        if (m8) qr2.pNext = &q8;
        VkPhysicalDeviceFeatures2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        q2.pNext = &q13;
        vkGetPhysicalDeviceFeatures2(phys, &q2);
        std::vector<const char*> dext;
        // A sample's run-time offset (a lift made with gcn2hlsl
        // --runtime-offsets, as on a device with the feature) needs maintenance8.
        VkPhysicalDeviceMaintenance8FeaturesKHR fm8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
        runtime_offsets = m8 && q8.maintenance8;
        if (runtime_offsets) {
            dext.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
            fm8.maintenance8 = VK_TRUE;
        }
        VkPhysicalDeviceRobustness2FeaturesEXT fr2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
        if (r2) {
            dext.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
            fr2.robustBufferAccess2 = qr2.robustBufferAccess2;
            fr2.robustImageAccess2 = qr2.robustImageAccess2;
            robust = std::string("robustness2 buffers ") + (qr2.robustBufferAccess2 ? "yes" : "no") + ", images " + (qr2.robustImageAccess2 ? "yes" : "no");
        }
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        // A compute lift's raw addresses (PhysicalStorageBuffer64): buffer device address.
        VkPhysicalDeviceVulkan12Features q12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        {
            VkPhysicalDeviceFeatures2 qf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            qf.pNext = &q12;
            vkGetPhysicalDeviceFeatures2(phys, &qf);
        }
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.bufferDeviceAddress = q12.bufferDeviceAddress;
        f12.shaderInt8 = q12.shaderInt8;
        f12.shaderFloat16 = q12.shaderFloat16;
        f12.storageBuffer8BitAccess = q12.storageBuffer8BitAccess;
        f13.dynamicRendering = VK_TRUE;
        f13.synchronization2 = VK_TRUE;
        f13.robustImageAccess = q13.robustImageAccess;
        if (r2) f13.pNext = &fr2;
        if (runtime_offsets) {
            fm8.pNext = f13.pNext;
            f13.pNext = &fm8;
        }
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f12.pNext = &f13;
        f2.pNext = &f12;
        f2.features.shaderInt64 = supported.shaderInt64;
        f2.features.shaderInt16 = supported.shaderInt16;
        f2.features.shaderStorageImageReadWithoutFormat = supported.shaderStorageImageReadWithoutFormat;
        f2.features.shaderStorageImageWriteWithoutFormat = supported.shaderStorageImageWriteWithoutFormat;
        f2.features.geometryShader = supported.geometryShader;
        f2.features.shaderClipDistance = supported.shaderClipDistance;
        f2.features.shaderTessellationAndGeometryPointSize = supported.shaderTessellationAndGeometryPointSize;
        f2.features.robustBufferAccess = supported.robustBufferAccess;
        f2.features.independentBlend = supported.independentBlend;
        f2.features.sampleRateShading = supported.sampleRateShading;
        f2.features.shaderImageGatherExtended = supported.shaderImageGatherExtended;
        f2.features.tessellationShader = supported.tessellationShader;
        f2.features.vertexPipelineStoresAndAtomics = supported.vertexPipelineStoresAndAtomics;
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<std::uint32_t>(dext.size());
        dci.ppEnabledExtensionNames = dext.data();
        check(vkCreateDevice(phys, &dci, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, family, 0, &queue);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = family;
        check(vkCreateCommandPool(device, &pci, nullptr, &pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &cai, &cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device, &fci, nullptr, &fence), "vkCreateFence");
    }
    std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags) const {
        for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        fail("no memory type");
    }
    Buffer buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        Buffer b;
        b.size = std::max<VkDeviceSize>(size, 16);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = b.size;
        bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device, &bci, nullptr, &b.buffer), "vkCreateBuffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &mai, nullptr, &b.memory), "vkAllocateMemory");
        check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "vkBindBufferMemory");
        check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
        std::memset(b.map, 0, b.size);
        return b;
    }
    void destroy(Buffer& b) {
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = Buffer{};
    }
    Image image(VkFormat format, std::uint32_t w, std::uint32_t h, std::uint32_t layers, std::uint32_t levels, VkImageUsageFlags usage,
                VkImageAspectFlags aspect, bool array_view) {
        Image img;
        img.format = format;
        img.width = w;
        img.height = h;
        img.layers = layers;
        img.levels = levels;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format;
        ci.extent = {w, h, 1};
        ci.mipLevels = levels;
        ci.arrayLayers = layers;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(vkCreateImage(device, &ci, nullptr, &img.image), "vkCreateImage");
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, img.image, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, 0);
        check(vkAllocateMemory(device, &mai, nullptr, &img.memory), "vkAllocateMemory (image)");
        check(vkBindImageMemory(device, img.image, img.memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img.image;
        vi.viewType = array_view ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {aspect, 0, levels, 0, layers};
        check(vkCreateImageView(device, &vi, nullptr, &img.view), "vkCreateImageView");
        return img;
    }
    void destroy(Image& img) {
        if (img.view) vkDestroyImageView(device, img.view, nullptr);
        if (img.image) vkDestroyImage(device, img.image, nullptr);
        if (img.memory) vkFreeMemory(device, img.memory, nullptr);
        img = Image{};
    }
    void submit(const std::function<void(VkCommandBuffer)>& record) {
        check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
        record(cmd);
        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        check(vkResetFences(device, 1, &fence), "vkResetFences");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        check(vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit");
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, 120'000'000'000ull), "vkWaitForFences");
    }
};

void barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

std::uint16_t half_bits(float f) {
    // Round to nearest for random texel data (any value is as good as another).
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000;
    const int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    if (exp <= 0) return static_cast<std::uint16_t>(sign);
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7bff);
    return static_cast<std::uint16_t>(sign | (exp << 10) | ((x >> 13) & 0x3ff));
}

// ---- one program -----------------------------------------------------------------
struct Inputs {
    std::uint64_t seed = 1;
    // Buffer and params words small integers (0-63) rather than floats: a
    // loop whose count a random float's bits give runs for ever.
    bool small_ints = false;
    // Buffer and params words all floats in [-2, 2] (no raw words): a raw
    // word in a matrix's translation swamps every point a domain shader
    // makes, so its outputs barely depend on the tessellation coordinate.
    bool float_words = false;
    std::uint32_t size = 32;      // the pixel shader's target: size x size
    std::uint32_t vertices = 48;  // the vertex shader's points
};

struct Run {
    std::vector<std::vector<std::uint8_t>> outputs;  // each target's texels
};

class Harness {
public:
    Harness(Device& d, gpu::Dxc& dxc) : dev(d), dxc_(dxc) {}

    // --compile: a compute module's pipeline, built against a layout made from
    // its own bindings - the driver's verdict, VK_SUCCESS or its error.
    VkResult compile_compute(const std::vector<std::uint32_t>& words) {
        const Module m = parse(words);
        if (m.model != 5) throw Skip("not a compute shader");
        std::map<std::uint32_t, std::vector<VkDescriptorSetLayoutBinding>> per_set;
        std::set<std::pair<std::uint32_t, std::uint32_t>> seen;
        for (const Var& v : m.vars) {
            if (v.sc != 0 && v.sc != 2 && v.sc != 12) continue;
            if (!v.has(34) || !v.has(33)) continue;
            if (!seen.insert({v.get(34), v.get(33)}).second) continue;
            std::uint32_t t = v.pointee, count = 1;
            if (m.types.at(t).op == 28) {  // an array of descriptors
                count = m.consts.at(m.types.at(t).w.at(1));
                t = m.types.at(t).w.at(0);
            }
            const TypeInfo& ty = m.types.at(t);
            VkDescriptorType type;
            if (v.sc == 2) type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            else if (v.sc == 12) type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            else if (ty.op == 26) type = VK_DESCRIPTOR_TYPE_SAMPLER;
            else if (ty.op == 27) type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            else if (ty.op == 25 && ty.w.at(1) == 5) type = ty.w.at(5) == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
            else if (ty.op == 25) type = ty.w.at(5) == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            else throw Skip("a descriptor of another kind");
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = v.get(33);
            lb.descriptorType = type;
            lb.descriptorCount = count;
            lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            per_set[v.get(34)].push_back(lb);
        }
        std::uint32_t max_set = 0;
        for (const auto& [set, b] : per_set) max_set = std::max(max_set, set);
        std::vector<VkDescriptorSetLayout> layouts;
        for (std::uint32_t set = 0; set <= max_set; ++set) {
            VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            const auto it = per_set.find(set);
            if (it != per_set.end()) {
                li.bindingCount = static_cast<std::uint32_t>(it->second.size());
                li.pBindings = it->second.data();
            }
            VkDescriptorSetLayout l;
            check(vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &l), "vkCreateDescriptorSetLayout");
            layouts.push_back(l);
        }
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = static_cast<std::uint32_t>(layouts.size());
        pli.pSetLayouts = layouts.data();
        VkPipelineLayout layout;
        check(vkCreatePipelineLayout(dev.device, &pli, nullptr, &layout), "vkCreatePipelineLayout");
        const VkShaderModule sm = module(words);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.layout = layout;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = sm;
        ci.stage.pName = "main";
        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult r = vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline);
        if (pipeline) vkDestroyPipeline(dev.device, pipeline, nullptr);
        vkDestroyShaderModule(dev.device, sm, nullptr);
        vkDestroyPipelineLayout(dev.device, layout, nullptr);
        for (VkDescriptorSetLayout l : layouts) vkDestroyDescriptorSetLayout(dev.device, l, nullptr);
        return r;
    }

    // "identical", "different: ...", or throws Skip / Failure.
    std::string compare(const fs::path& lift_path, const fs::path& dxc_path, const Inputs& in) {
        const std::vector<std::uint32_t> lift = read_words(lift_path), other = read_words(dxc_path);
        const Module m = parse(lift);
        if (m.model != 4 && m.model != 0 && m.model != 1 && m.model != 2) throw Skip("execution model " + std::to_string(m.model));
        const Module declared = m.model == 1 ? parse(other) : Module();  // a hull's domain and spacing, which only DXC's module declares
        if (m.caps.count(68) && !dev.quad_ops) throw Skip("quad operations unsupported by the device");
        for (const std::uint32_t s : m.dref_samplers) {
            if (m.plain_samplers.count(s)) throw Skip("a sampler used with and without a depth compare");
        }
        Resources r = make_resources(m, in);
        std::string result;
        try {
            const auto run = [&](const std::vector<std::uint32_t>& w) {
                return m.model == 4 ? run_pixel(m, w, r, in) : m.model == 0 ? run_vertex(m, w, r, in) : m.model == 2 ? run_domain(m, w, r, in)
                                                                                                         : run_hull(m, w, r, in, declared);
            };
            Run a = run(lift);
            Run b = run(other);
            if (const char* dump = std::getenv("HLSL_EQUIV_DUMP"); dump && *dump) {
                // Both runs' targets, raw, for a closer look at a difference.
                for (std::size_t t = 0; t < a.outputs.size(); ++t) {
                    for (int k = 0; k < 2; ++k) {
                        const auto& o = (k ? b : a).outputs[t];
                        std::ofstream(std::string(dump) + "/" + (k ? "hlsl" : "lift") + "-" + std::to_string(t) + ".bin", std::ios::binary)
                            .write(reinterpret_cast<const char*>(o.data()), static_cast<std::streamsize>(o.size()));
                    }
                }
            }
            result = diff(a, b, m.model == 4 ? in.size : in.vertices);
        } catch (...) {
            free_resources(r);
            throw;
        }
        free_resources(r);
        return result;
    }

private:
    Device& dev;
    gpu::Dxc& dxc_;
    std::map<std::string, VkShaderModule> helper_cache;

    struct Binding {
        std::uint32_t set = 0, binding = 0;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
        Buffer buffer;
        Image image;
        VkSampler sampler = VK_NULL_HANDLE;
    };
    struct Resources {
        std::vector<Binding> bindings;
        std::map<std::uint32_t, VkDescriptorSetLayout> layouts;  // set -> layout
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::map<std::uint32_t, VkDescriptorSet> sets;
        Buffer vertex;  // the vertex shader's attributes: one uint4 a location, per vertex
        Buffer capture;  // a hull's run: what the hull stores; a domain shader's: the point counter (set capture_set, binding 0)
        std::uint32_t capture_set = 0;
        std::vector<std::uint32_t> vertex_locations;
        std::vector<std::uint32_t> attr_words;  // the pixel shader's inputs' per-vertex values (3 vertices x location x 4)
    };

    void free_resources(Resources& r) {
        for (Binding& b : r.bindings) {
            dev.destroy(b.buffer);
            dev.destroy(b.image);
            if (b.sampler) vkDestroySampler(dev.device, b.sampler, nullptr);
        }
        if (r.pool) vkDestroyDescriptorPool(dev.device, r.pool, nullptr);
        if (r.layout) vkDestroyPipelineLayout(dev.device, r.layout, nullptr);
        for (auto& [s, l] : r.layouts) vkDestroyDescriptorSetLayout(dev.device, l, nullptr);
        dev.destroy(r.vertex);
    }

    Resources make_resources(const Module& m, const Inputs& in) {
        Resources r;
        std::mt19937_64 rng(in.seed);
        const auto word_any = [&] { return static_cast<std::uint32_t>(rng()); };
        const auto word = [&] { return in.small_ints ? static_cast<std::uint32_t>(rng() % 64) : word_any(); };
        // Floats mostly in [-2, 2], a few special values: the inputs a shader computes with.
        const auto float_word = [&] {
            if (in.small_ints) return word();
            const std::uint32_t pick = word() % 64;
            float f;
            if (pick == 0) f = 0.0f;
            else if (pick == 1) f = -0.0f;
            else if (pick == 2) f = 1.0f;
            else if (pick == 3) f = 1e30f;
            else if (pick == 4) f = -1e-30f;
            else f = static_cast<float>(static_cast<int>(word() % 4001) - 2000) / 1000.0f;
            std::uint32_t bits;
            std::memcpy(&bits, &f, 4);
            return bits;
        };
        const VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS;
        std::map<std::uint32_t, std::vector<VkDescriptorSetLayoutBinding>> per_set;
        std::set<std::pair<std::uint32_t, std::uint32_t>> seen;
        for (const Var& v : m.vars) {
            if (v.sc != 0 && v.sc != 2 && v.sc != 12) continue;
            if (!v.has(34) || !v.has(33)) continue;
            Binding b;
            b.set = v.get(34);
            b.binding = v.get(33);
            if (!seen.insert({b.set, b.binding}).second) continue;
            const TypeInfo& t = m.types.at(v.pointee);
            if (v.sc == 2) {  // the params block (std140)
                b.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                b.buffer = dev.buffer(512, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
                auto* p = static_cast<std::uint32_t*>(b.buffer.map);
                for (std::uint32_t k = 0; k < 128; ++k) p[k] = word();
                // StageParams: the buffers' biases 0 (offset 96), their ranges
                // and strides inside the 16 KiB each is bound with (160), every
                // buffer bound (cb_valid, 80).
                p[80 / 4] = 0xffffffffu;
                for (std::uint32_t k = 0; k < 16; ++k) p[96 / 4 + k] = 0;
                for (std::uint32_t k = 0; k < 16; ++k) p[160 / 4 + k] = 16 + (word() % 4) * 16;
            } else if (v.sc == 12) {
                b.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                b.buffer = dev.buffer(16384, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                auto* p = static_cast<std::uint32_t*>(b.buffer.map);
                for (std::uint32_t k = 0; k < 4096; ++k) p[k] = (k & 3) == 3 && !in.float_words ? word() : float_word();
            } else if (t.op == 26) {  // sampler
                b.type = VK_DESCRIPTOR_TYPE_SAMPLER;
                VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                const bool linear = (word() & 1) != 0;
                si.magFilter = si.minFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
                si.mipmapMode = (word() & 1) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
                si.addressModeU = si.addressModeV = si.addressModeW = (word() & 1) ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                si.maxLod = 16.0f;
                if (m.dref_samplers.count(v.id)) {
                    si.compareEnable = VK_TRUE;
                    si.compareOp = VK_COMPARE_OP_LESS;
                }
                check(vkCreateSampler(dev.device, &si, nullptr, &b.sampler), "vkCreateSampler");
            } else if (t.op == 25) {  // image
                const std::uint32_t sampled = t.w.at(0), dim = t.w.at(1), arrayed = t.w.at(3), ms = t.w.at(4), mode = t.w.at(5);
                if (dim != 1 || ms || mode != 1) throw Skip("an image that is not a sampled 2D image");
                const TypeInfo& st = m.types.at(sampled);
                const bool depth = t.w.at(2) == 1;
                const char kind = st.op == 22 ? 'f' : st.w.at(1) ? 'i' : 'u';
                b.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                const std::uint32_t w = 16, h = 16, layers = arrayed ? 6 : 1, levels = 5;
                VkFormat format = depth ? VK_FORMAT_D32_SFLOAT
                                  : kind == 'f' ? VK_FORMAT_R16G16B16A16_SFLOAT
                                  : kind == 'u' ? VK_FORMAT_R32G32B32A32_UINT
                                                : VK_FORMAT_R32G32B32A32_SINT;
                const VkImageAspectFlags aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
                b.image = dev.image(format, w, h, layers, levels, VK_IMAGE_USAGE_SAMPLED_BIT, aspect, arrayed != 0);
                // Every level and layer from a staging buffer.
                const std::uint32_t texel = depth ? 4 : kind == 'f' ? 8 : 16;
                std::vector<VkBufferImageCopy> copies;
                VkDeviceSize total = 0;
                for (std::uint32_t l = 0; l < levels; ++l) total += static_cast<VkDeviceSize>(std::max(1u, w >> l)) * std::max(1u, h >> l) * layers * texel;
                Buffer staging = dev.buffer(total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                auto* bytes = static_cast<std::uint8_t*>(staging.map);
                VkDeviceSize at = 0;
                for (std::uint32_t l = 0; l < levels; ++l) {
                    const std::uint32_t lw = std::max(1u, w >> l), lh = std::max(1u, h >> l);
                    VkBufferImageCopy c{};
                    c.bufferOffset = at;
                    c.imageSubresource = {aspect, l, 0, layers};
                    c.imageExtent = {lw, lh, 1};
                    copies.push_back(c);
                    for (std::uint32_t k = 0; k < lw * lh * layers; ++k) {
                        if (depth) {
                            const float d = static_cast<float>(word() % 1001) / 1000.0f;
                            std::memcpy(bytes + at, &d, 4);
                        } else if (kind == 'f') {
                            for (int c4 = 0; c4 < 4; ++c4) {
                                const float f = static_cast<float>(static_cast<int>(word() % 3001) - 1000) / 1000.0f;
                                const std::uint16_t hb = half_bits(f);
                                std::memcpy(bytes + at + 2 * c4, &hb, 2);
                            }
                        } else {
                            for (int c4 = 0; c4 < 4; ++c4) {
                                const std::uint32_t x = word();
                                std::memcpy(bytes + at + 4 * c4, &x, 4);
                            }
                        }
                        at += texel;
                    }
                }
                dev.submit([&](VkCommandBuffer cmd) {
                    barrier(cmd, b.image.image, aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                    vkCmdCopyBufferToImage(cmd, staging.buffer, b.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           static_cast<std::uint32_t>(copies.size()), copies.data());
                    barrier(cmd, b.image.image, aspect, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                });
                dev.destroy(staging);
            } else {
                throw Skip("a descriptor of another kind");
            }
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = b.binding;
            lb.descriptorType = b.type;
            lb.descriptorCount = 1;
            lb.stageFlags = stages;
            per_set[b.set].push_back(lb);
            r.bindings.push_back(std::move(b));
        }
        // A hull's run: the buffer the hull stores its outputs to; a domain
        // shader's: the geometry shader's point counter. In a set past the module's.
        if (m.model == 1 || m.model == 2) {
            std::uint32_t top = 0;
            for (const auto& [s, v] : per_set) top = std::max(top, s + 1);
            r.capture_set = top;
            Binding b;
            b.set = top;
            b.binding = 0;
            b.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.buffer = dev.buffer(1u << 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            r.capture = b.buffer;
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = 0;
            lb.descriptorType = b.type;
            lb.descriptorCount = 1;
            lb.stageFlags = stages;
            per_set[top].push_back(lb);
            r.bindings.push_back(std::move(b));
        }
        // Every set index up to the highest one (empty layouts in between).
        std::uint32_t max_set = 0;
        for (const auto& [s, v] : per_set) max_set = std::max(max_set, s);
        std::vector<VkDescriptorSetLayout> ordered;
        for (std::uint32_t s = 0; s <= max_set; ++s) {
            VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            const auto it = per_set.find(s);
            if (it != per_set.end()) {
                li.bindingCount = static_cast<std::uint32_t>(it->second.size());
                li.pBindings = it->second.data();
            }
            VkDescriptorSetLayout l;
            check(vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &l), "vkCreateDescriptorSetLayout");
            r.layouts[s] = l;
            ordered.push_back(l);
        }
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = static_cast<std::uint32_t>(ordered.size());
        pli.pSetLayouts = ordered.data();
        check(vkCreatePipelineLayout(dev.device, &pli, nullptr, &r.layout), "vkCreatePipelineLayout");
        const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64},
                                              {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64}, {VK_DESCRIPTOR_TYPE_SAMPLER, 64}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 16;
        dpi.poolSizeCount = 4;
        dpi.pPoolSizes = sizes;
        check(vkCreateDescriptorPool(dev.device, &dpi, nullptr, &r.pool), "vkCreateDescriptorPool");
        for (std::uint32_t s = 0; s <= max_set; ++s) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = r.pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &r.layouts[s];
            VkDescriptorSet ds;
            check(vkAllocateDescriptorSets(dev.device, &ai, &ds), "vkAllocateDescriptorSets");
            r.sets[s] = ds;
        }
        for (const Binding& b : r.bindings) {
            VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr.dstSet = r.sets[b.set];
            wr.dstBinding = b.binding;
            wr.descriptorCount = 1;
            wr.descriptorType = b.type;
            VkDescriptorBufferInfo bi{b.buffer.buffer, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo ii{b.sampler, b.image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            if (b.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) wr.pBufferInfo = &bi;
            else wr.pImageInfo = &ii;
            vkUpdateDescriptorSets(dev.device, 1, &wr, 0, nullptr);
        }
        // Stage inputs.
        if (m.model == 0) {
            for (const Var& v : m.vars) {
                if (v.sc == 1 && v.has(30) && m.interface_ids.count(v.id)) r.vertex_locations.push_back(v.get(30));
            }
            std::sort(r.vertex_locations.begin(), r.vertex_locations.end());
            const std::size_t per = r.vertex_locations.size() * 4;
            r.vertex = dev.buffer(std::max<std::size_t>(16, per * 4 * in.vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            auto* p = static_cast<std::uint32_t*>(r.vertex.map);
            for (std::size_t k = 0; k < per * in.vertices; ++k) p[k] = (k & 1) ? word() : float_word();
        } else {
            r.attr_words.resize(3 * 32 * 4);
            for (std::uint32_t& x : r.attr_words) x = float_word();
            for (std::size_t k = 0; k < r.attr_words.size(); k += 7) r.attr_words[k] = word();  // some raw bits too
        }
        return r;
    }

    VkShaderModule module(const std::vector<std::uint32_t>& words) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = words.size() * 4;
        ci.pCode = words.data();
        VkShaderModule s;
        check(vkCreateShaderModule(dev.device, &ci, nullptr, &s), "vkCreateShaderModule");
        return s;
    }
    VkShaderModule helper(const std::string& source, const char* profile) {
        const std::string key = std::string(profile) + "\n" + source;
        const auto it = helper_cache.find(key);
        if (it != helper_cache.end()) return it->second;
        gcn::HlslResult h;
        h.source = source;
        h.profile = profile;
        gpu::DxcOptions o;
        std::vector<std::uint32_t> spirv;
        std::string error;
        if (!dxc_.compile(h, o, spirv, error)) fail("a generated stage did not compile: " + error.substr(0, 300));
        const VkShaderModule s = module(spirv);
        helper_cache[key] = s;
        return s;
    }

    static std::string hex(std::uint32_t v) {
        char b[16];
        std::snprintf(b, sizeof(b), "0x%08xu", v);
        return b;
    }

    // ---- pixel shaders: a full-screen triangle -------------------------------------
    Run run_pixel(const Module& m, const std::vector<std::uint32_t>& words, Resources& r, const Inputs& in) {
        // The vertex shader feeding its inputs: each a value per vertex, as the input interpolates.
        std::string vs = "struct VOut {\n    float4 pos : SV_Position;\n";
        std::string body;
        struct Out {
            std::uint32_t location, comps;
            char kind;
            bool depth = false;
        };
        std::vector<Out> outs;
        bool frag_depth = false;
        for (const Var& v : m.vars) {
            if (!m.interface_ids.count(v.id)) continue;
            if (v.sc == 1 && v.has(30)) {
                char kind;
                std::uint32_t comps;
                if (!scalar_shape(m, v.pointee, kind, comps)) throw Skip("a pixel input of an aggregate type");
                const std::uint32_t loc = v.get(30);
                if (loc >= 32) throw Skip("an input past location 31");
                std::string q = v.has(14) || kind != 'f' ? "nointerpolation " : "";
                if (v.has(13)) q += "noperspective ";
                if (v.has(16)) q += "centroid ";
                const std::string tn = std::string(kind == 'f' ? "float" : kind == 'u' ? "uint" : "int") + (comps > 1 ? std::to_string(comps) : "");
                vs += "    [[vk::location(" + std::to_string(loc) + ")]] " + q + tn + " a" + std::to_string(loc) + " : A" + std::to_string(loc) + ";\n";
                std::string per_vertex[3];
                for (int vx = 0; vx < 3; ++vx) {
                    std::string e = tn + "(";
                    for (std::uint32_t c = 0; c < comps; ++c) {
                        const std::uint32_t bits = r.attr_words[(vx * 32 + loc) * 4 + c];
                        e += (c ? ", " : "") + std::string(kind == 'f' ? "asfloat(" : kind == 'u' ? "(" : "asint(") + hex(bits) + ")";
                    }
                    per_vertex[vx] = e + ")";
                }
                body += "    o.a" + std::to_string(loc) + " = select(vid == 0, " + per_vertex[0] + ", select(vid == 1, " + per_vertex[1] + ", " + per_vertex[2] +
                        "));\n";
            }
            if (v.sc == 3) {
                if (v.has(11) && v.get(11) == 22) {
                    frag_depth = true;
                    continue;
                }
                if (!v.has(30)) throw Skip("a pixel output that is no location");
                char kind;
                std::uint32_t comps;
                if (!scalar_shape(m, v.pointee, kind, comps) || kind == 'b') throw Skip("a pixel output of an aggregate type");
                outs.push_back({v.get(30), comps, kind});
            }
        }
        vs += "};\nVOut main(uint vid : SV_VertexID) {\n    VOut o;\n"
              "    float2 uv = float2((vid << 1) & 2, vid & 2);\n"
              "    o.pos = float4(uv * 2.0f - 1.0f, 0.5f, 1.0f);\n" +
              body + "    return o;\n}\n";
        const VkShaderModule vsm = helper(vs, "vs_6_6");
        const VkShaderModule fsm = module(words);
        // Targets: each output's location, 32-bit channels.
        std::sort(outs.begin(), outs.end(), [](const Out& a, const Out& b) { return a.location < b.location; });
        const std::uint32_t n = in.size;
        std::vector<Image> targets;
        std::vector<VkFormat> formats;
        std::uint32_t max_loc = 0;
        for (const Out& o : outs) max_loc = std::max(max_loc, o.location);
        std::vector<VkRenderingAttachmentInfo> colors(outs.empty() ? 0 : max_loc + 1, VkRenderingAttachmentInfo{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO});
        std::vector<VkFormat> color_formats(colors.size(), VK_FORMAT_UNDEFINED);
        for (const Out& o : outs) {
            const VkFormat f = o.kind == 'f' ? VK_FORMAT_R32G32B32A32_SFLOAT : o.kind == 'u' ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SINT;
            targets.push_back(dev.image(f, n, n, 1, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false));
            VkRenderingAttachmentInfo& a = colors[o.location];
            a.imageView = targets.back().view;
            a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            a.clearValue.color.uint32[0] = 0xdeadbeefu;
            a.clearValue.color.uint32[1] = 0x7fc0dead;
            a.clearValue.color.uint32[2] = 0x12345678u;
            a.clearValue.color.uint32[3] = 0xfedcba98u;
            color_formats[o.location] = f;
        }
        Image depth;
        if (frag_depth) depth = dev.image(VK_FORMAT_D32_SFLOAT, n, n, 1, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, false);
        VkPipeline pipeline = graphics_pipeline(r.layout, vsm, VK_NULL_HANDLE, fsm, color_formats, frag_depth, false, {});
        Run run;
        std::vector<Buffer> readback;
        dev.submit([&](VkCommandBuffer cmd) {
            for (Image& t : targets) barrier(cmd, t.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            if (frag_depth) barrier(cmd, depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
            VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            da.imageView = depth.view;
            da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            da.clearValue.depthStencil.depth = 0.25f;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {n, n}};
            ri.layerCount = 1;
            ri.colorAttachmentCount = static_cast<std::uint32_t>(colors.size());
            ri.pColorAttachments = colors.data();
            if (frag_depth) ri.pDepthAttachment = &da;
            vkCmdBeginRendering(cmd, &ri);
            draw(cmd, pipeline, r, n, n, 3, 0);
            vkCmdEndRendering(cmd);
            for (Image& t : targets) {
                barrier(cmd, t.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                readback.push_back(dev.buffer(static_cast<VkDeviceSize>(n) * n * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
                VkBufferImageCopy c{};
                c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                c.imageExtent = {n, n, 1};
                vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.back().buffer, 1, &c);
            }
            if (frag_depth) {
                barrier(cmd, depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                readback.push_back(dev.buffer(static_cast<VkDeviceSize>(n) * n * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
                VkBufferImageCopy c{};
                c.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
                c.imageExtent = {n, n, 1};
                vkCmdCopyImageToBuffer(cmd, depth.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.back().buffer, 1, &c);
            }
        });
        for (Buffer& b : readback) {
            const auto* p = static_cast<const std::uint8_t*>(b.map);
            run.outputs.emplace_back(p, p + b.size);
            dev.destroy(b);
        }
        vkDestroyPipeline(dev.device, pipeline, nullptr);
        vkDestroyShaderModule(dev.device, fsm, nullptr);
        for (Image& t : targets) dev.destroy(t);
        dev.destroy(depth);
        return run;
    }

    // ---- vertex shaders: each vertex's outputs as a row of texels ------------------
    // The geometry shader that turns each point of a vertex or domain shader
    // `m` into a column of texels, one row an output (position, every location,
    // clip distances, point size): the column is the point's primitive ID.
    struct Rows {
        std::string gs;
        std::uint32_t rows = 0;
    };
    // counter_set: the column from a counter in that set's buffer (binding 0)
    // instead - after tessellation the primitive ID is the patch's, the same
    // for each of its points.
    Rows gs_rows(const Module& m, std::uint32_t columns, int counter_set = -1) {
        struct Out {
            std::string decl, read;  // the input's declaration; its value as uint4
        };
        std::vector<Out> outs;
        for (const Var& v : m.vars) {
            if (v.sc != 3 || !m.interface_ids.count(v.id)) continue;
            if (v.has(30)) {
                // A scalar or vector; or an array of them (an LS's control
                // point, one location an element), a row an element.
                std::uint32_t elem = v.pointee, count = 0;
                if (m.types.at(elem).op == 28) {
                    count = m.consts.at(m.types.at(elem).w.at(1));
                    elem = m.types.at(elem).w.at(0);
                }
                char kind;
                std::uint32_t comps;
                if (!scalar_shape(m, elem, kind, comps) || kind == 'b') throw Skip("a vertex output of an aggregate type");
                const std::string loc = std::to_string(v.get(30));
                const std::string tn = std::string(kind == 'f' ? "float" : kind == 'u' ? "uint" : "int") + (comps > 1 ? std::to_string(comps) : "");
                std::string pad = comps == 4 ? "" : comps == 3 ? ", 0u" : comps == 2 ? ", 0u, 0u" : ", 0u, 0u, 0u";
                if (!count) {
                    outs.push_back({"    [[vk::location(" + loc + ")]] nointerpolation " + tn + " o" + loc + " : O" + loc + ";\n",
                                    "uint4(asuint(v[0].o" + loc + ")" + pad + ")"});
                    continue;
                }
                for (std::uint32_t k = 0; k < count; ++k)
                    outs.push_back({k ? std::string() : "    [[vk::location(" + loc + ")]] nointerpolation " + tn + " o" + loc + "[" + std::to_string(count) + "] : O" + loc + ";\n",
                                    "uint4(asuint(v[0].o" + loc + "[" + std::to_string(k) + "])" + pad + ")"});
                continue;
            }
            if (!v.has(11)) continue;
            const std::uint32_t bi = v.get(11);
            if (bi == 0) {
                outs.push_back({"    float4 pos : SV_Position;\n", "asuint(v[0].pos)"});
            } else if (bi == 1) {
                outs.push_back({"    [[vk::builtin(\"PointSize\")]] float psize : PSIZE;\n", "uint4(asuint(v[0].psize), 0u, 0u, 0u)"});
            } else if (bi == 3) {
                const TypeInfo& at = m.types.at(v.pointee);
                const std::uint32_t count = m.consts.at(at.w.at(1));
                for (std::uint32_t g = 0; g * 4 < count; ++g) {
                    const std::uint32_t c = std::min<std::uint32_t>(4, count - g * 4);
                    const std::string tn = c == 1 ? "float" : "float" + std::to_string(c);
                    const std::string pad = c == 4 ? "" : c == 3 ? ", 0u" : c == 2 ? ", 0u, 0u" : ", 0u, 0u, 0u";
                    outs.push_back({"    " + tn + " clip" + std::to_string(g) + " : SV_ClipDistance" + std::to_string(g) + ";\n",
                                    "uint4(asuint(v[0].clip" + std::to_string(g) + ")" + pad + ")"});
                }
            } else {
                throw Skip("vertex output built-in " + std::to_string(bi));
            }
        }
        Rows r;
        r.rows = static_cast<std::uint32_t>(outs.size());
        if (!r.rows) throw Skip("no outputs");
        std::string& gs = r.gs;
        gs = "struct VOut {\n";
        for (const Out& o : outs) gs += o.decl;
        gs += "};\nstruct GOut {\n    float4 pos : SV_Position;\n    [[vk::builtin(\"PointSize\")]] float psize : PSIZE;\n"
              "    [[vk::location(0)]] nointerpolation uint4 bits : BITS;\n};\n";
        if (counter_set >= 0) gs += "[[vk::binding(0, " + std::to_string(counter_set) + ")]] RWByteAddressBuffer counter;\n";
        gs += "[maxvertexcount(" + std::to_string(r.rows) + ")]\nvoid main(point VOut v[1], uint prim_id : SV_PrimitiveID, inout PointStream<GOut> s) {\n";
        gs += counter_set >= 0 ? "    uint prim;\n    counter.InterlockedAdd(0u, 1u, prim);\n" : "    uint prim = prim_id;\n";
        gs += "    GOut g;\n    g.psize = 1.0f;\n";
        for (std::uint32_t k = 0; k < r.rows; ++k) {
            gs += "    g.pos = float4((float(prim) + 0.5f) / " + std::to_string(columns) + ".0f * 2.0f - 1.0f, (" + std::to_string(k) + ".5f) / " +
                  std::to_string(r.rows) + ".0f * 2.0f - 1.0f, 0.5f, 1.0f);\n";
            gs += "    g.bits = " + outs[k].read + ";\n    s.Append(g);\n";
        }
        gs += "}\n";
        return r;
    }
    // Draws `vertices` with `pipeline` into a columns x rows R32G32B32A32_UINT
    // target cleared to 0xdeadbeef, and reads it back.
    Run draw_rows(VkPipeline pipeline, Resources& r, std::uint32_t columns, std::uint32_t rows, std::uint32_t vertices, bool vertex_buffer) {
        Image target = dev.image(VK_FORMAT_R32G32B32A32_UINT, columns, rows, 1, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false);
        Run run;
        Buffer rb = dev.buffer(static_cast<VkDeviceSize>(columns) * rows * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        dev.submit([&](VkCommandBuffer cmd) {
            barrier(cmd, target.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            a.imageView = target.view;
            a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            for (std::uint32_t& c : a.clearValue.color.uint32) c = 0xdeadbeefu;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {columns, rows}};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &a;
            vkCmdBeginRendering(cmd, &ri);
            if (vertex_buffer) {
                const VkDeviceSize zero = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &r.vertex.buffer, &zero);
            }
            draw(cmd, pipeline, r, columns, rows, vertices, 0);
            vkCmdEndRendering(cmd);
            barrier(cmd, target.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy c{};
            c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.imageExtent = {columns, rows, 1};
            vkCmdCopyImageToBuffer(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buffer, 1, &c);
        });
        const auto* p = static_cast<const std::uint8_t*>(rb.map);
        run.outputs.emplace_back(p, p + rb.size);
        dev.destroy(rb);
        dev.destroy(target);
        return run;
    }
    Run run_vertex(const Module& m, const std::vector<std::uint32_t>& words, Resources& r, const Inputs& in) {
        const Rows rows = gs_rows(m, in.vertices);
        const std::string ps = "[[vk::location(0)]] uint4 main([[vk::location(0)]] nointerpolation uint4 bits : BITS) : SV_Target0 { return bits; }\n";
        const VkShaderModule gsm = helper(rows.gs, "gs_6_6"), psm = helper(ps, "ps_6_6"), vsm = module(words);
        // One vertex buffer binding a location, uint4 each.
        std::vector<VkVertexInputBindingDescription> vb;
        std::vector<VkVertexInputAttributeDescription> va;
        const std::uint32_t stride = static_cast<std::uint32_t>(r.vertex_locations.size()) * 16;
        if (!r.vertex_locations.empty()) vb.push_back({0, stride, VK_VERTEX_INPUT_RATE_VERTEX});
        for (std::size_t k = 0; k < r.vertex_locations.size(); ++k)
            va.push_back({r.vertex_locations[k], 0, VK_FORMAT_R32G32B32A32_UINT, static_cast<std::uint32_t>(k * 16)});
        const std::vector<VkFormat> formats = {VK_FORMAT_R32G32B32A32_UINT};
        VkPipeline pipeline = graphics_pipeline(r.layout, vsm, gsm, psm, formats, false, true, {vb, va});
        Run run = draw_rows(pipeline, r, in.vertices, rows.rows, in.vertices, !r.vertex_locations.empty());
        vkDestroyPipeline(dev.device, pipeline, nullptr);
        vkDestroyShaderModule(dev.device, vsm, nullptr);
        return run;
    }

    // ---- tessellation: the stage around the tested one generated --------------------
    // Random control points and patch constants come from tables in the
    // generated stages: words of floats mostly in [-2, 2] (every seventh raw
    // unless --float-words), as the lifts' uint4 control points carry floats' bits.
    static constexpr std::uint32_t kPatches = 4, kTessColumns = 256;
    static std::string u4(std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d) {
        return "uint4(" + hex(a) + ", " + hex(b) + ", " + hex(c) + ", " + hex(d) + ")";
    }
    static std::uint32_t table_word(std::mt19937_64& rng, std::size_t k, bool raw) {
        if (raw && k % 7 == 3) return static_cast<std::uint32_t>(rng());
        const std::uint32_t pick = static_cast<std::uint32_t>(rng() % 64);
        float f = pick == 0 ? 0.0f : pick == 1 ? -0.0f : pick == 2 ? 1.0f : static_cast<float>(static_cast<int>(rng() % 4001) - 2000) / 1000.0f;
        std::uint32_t bits;
        std::memcpy(&bits, &f, 4);
        return bits;
    }
    // A stage variable as a struct member: its element type (a scalar or
    // vector) and array count (0: none) after `outer` levels of per-control-point
    // arrays are taken off.
    struct Member {
        std::uint32_t location = 0, count = 0, comps = 0;
        char kind = 'u';
        std::string name, tn;
        std::uint32_t elements() const { return count ? count : 1; }
        std::string decl(const std::string& semantic) const {
            return "    [[vk::location(" + std::to_string(location) + ")]] " + tn + " " + name + (count ? "[" + std::to_string(count) + "]" : "") + " : " +
                   semantic + std::to_string(location) + ";\n";
        }
        // Element k from a uint4 expression.
        std::string from_u4(const std::string& e) const {
            static const char* sw[] = {"", ".x", ".xy", ".xyz", ""};
            const std::string v = comps == 4 ? e : "(" + e + ")" + sw[comps];
            return kind == 'f' ? "asfloat(" + v + ")" : kind == 'i' ? "asint(" + v + ")" : v;
        }
        std::string to_u4(const std::string& e) const {
            static const char* pad[] = {"", ", 0u, 0u, 0u", ", 0u, 0u", ", 0u", ""};
            return comps == 4 ? "asuint(" + e + ")" : "uint4(asuint(" + e + ")" + pad[comps] + ")";
        }
        std::string at(std::uint32_t k) const { return name + (count ? "[" + std::to_string(k) + "]" : ""); }
    };
    static Member member_of(const Module& m, const Var& v, std::uint32_t outer, std::uint32_t& outer_count) {
        std::uint32_t t = v.pointee;
        outer_count = 0;
        for (std::uint32_t k = 0; k < outer; ++k) {
            const TypeInfo& ty = m.types.at(t);
            if (ty.op != 28) throw Skip("a control-point variable that is not an array");
            outer_count = m.consts.at(ty.w.at(1));
            t = ty.w.at(0);
        }
        Member mb;
        mb.location = v.get(30);
        mb.name = "m" + std::to_string(mb.location);
        if (m.types.at(t).op == 28) {
            mb.count = m.consts.at(m.types.at(t).w.at(1));
            t = m.types.at(t).w.at(0);
        }
        if (!scalar_shape(m, t, mb.kind, mb.comps) || mb.kind == 'b') throw Skip("a tessellation variable of an aggregate type");
        mb.tn = std::string(mb.kind == 'f' ? "float" : mb.kind == 'u' ? "uint" : "int") + (mb.comps > 1 ? std::to_string(mb.comps) : "");
        return mb;
    }
    static const char* partitioning(const Module& m) {
        return m.modes.count(1) ? "integer" : m.modes.count(2) ? "fractional_even" : m.modes.count(3) ? "fractional_odd" : nullptr;
    }

    // Domain shaders: a generated hull gives each of kPatches patches random
    // control points and patch constants and the factors 3, in point mode; the
    // tessellator's points go through the geometry shader's rows, a column a point.
    Run run_domain(const Module& m, const std::vector<std::uint32_t>& words, Resources& r, const Inputs& in) {
        const bool tri = m.modes.count(22) != 0, quad = m.modes.count(24) != 0;
        const char* part = partitioning(m);
        if ((!tri && !quad) || !part) throw Skip("a domain shader without a triangle or quad domain and its spacing");
        std::vector<Member> cps, pcs;
        std::uint32_t patch = 0;
        for (const Var& v : m.vars) {
            if (v.sc != 1 || !m.interface_ids.count(v.id)) continue;
            if (v.has(11)) {
                const std::uint32_t bi = v.get(11);
                if (bi != 13 && bi != 11 && bi != 12 && bi != 7 && bi != 14) throw Skip("domain input built-in " + std::to_string(bi));
                continue;
            }
            if (!v.has(30)) continue;
            std::uint32_t n = 0;
            if (v.has(15)) {
                pcs.push_back(member_of(m, v, 0, n));
            } else {
                cps.push_back(member_of(m, v, 1, n));
                if (patch && n != patch) throw Skip("control-point inputs of different patch sizes");
                patch = n;
            }
        }
        if (!patch) patch = 1;
        std::mt19937_64 rng(in.seed ^ 0x7e55a11dull);
        std::size_t wk = 0;
        std::string hs = "struct VOut { [[vk::location(0)]] uint id : VID; };\nstruct CP {\n";
        std::uint32_t cpe = 0, pce = 0;
        for (const Member& c : cps) {
            hs += c.decl("CP");
            cpe += c.elements();
        }
        if (cps.empty()) hs += "    [[vk::location(31)]] uint4 unused : CPU;\n";
        hs += std::string("};\nstruct PC {\n    float outer[") + (tri ? "3" : "4") + "] : SV_TessFactor;\n    float inner" + (tri ? "" : "[2]") +
              " : SV_InsideTessFactor;\n";
        for (const Member& c : pcs) {
            hs += c.decl("PC");
            pce += c.elements();
        }
        hs += "};\n";
        const auto table = [&](const char* name, std::uint32_t n) {
            std::string t = "static const uint4 " + std::string(name) + "[" + std::to_string(std::max(1u, n)) + "] = {";
            for (std::uint32_t k = 0; k < std::max(1u, n); ++k) {
                std::uint32_t w4[4];
                for (std::uint32_t& x : w4) x = table_word(rng, wk++, !in.float_words);
                t += (k ? ", " : "") + u4(w4[0], w4[1], w4[2], w4[3]);
            }
            return t + "};\n";
        };
        hs += table("cpv", kPatches * patch * cpe) + table("pcv", kPatches * pce);
        hs += "PC pcf(InputPatch<VOut, " + std::to_string(patch) + "> ip, uint pid : SV_PrimitiveID) {\n    PC o;\n";
        for (int k = 0; k < (tri ? 3 : 4); ++k) hs += "    o.outer[" + std::to_string(k) + "] = 3.0f;\n";
        hs += tri ? "    o.inner = 3.0f;\n" : "    o.inner[0] = 3.0f;\n    o.inner[1] = 3.0f;\n";
        hs += "    uint b = (pid % " + std::to_string(kPatches) + "u) * " + std::to_string(pce) + "u;\n";
        std::uint32_t at = 0;
        for (const Member& c : pcs) {
            for (std::uint32_t k = 0; k < c.elements(); ++k, ++at) hs += "    o." + c.at(k) + " = " + c.from_u4("pcv[b + " + std::to_string(at) + "u]") + ";\n";
        }
        hs += "    return o;\n}\n";
        hs += std::string("[domain(\"") + (tri ? "tri" : "quad") + "\")]\n[partitioning(\"" + part + "\")]\n[outputtopology(\"point\")]\n[outputcontrolpoints(" +
              std::to_string(patch) + ")]\n[patchconstantfunc(\"pcf\")]\n";
        hs += "CP main(InputPatch<VOut, " + std::to_string(patch) + "> ip, uint i : SV_OutputControlPointID, uint pid : SV_PrimitiveID) {\n    CP o;\n";
        hs += "    uint b = ((pid % " + std::to_string(kPatches) + "u) * " + std::to_string(patch) + "u + i) * " + std::to_string(cpe) + "u;\n";
        at = 0;
        for (const Member& c : cps) {
            for (std::uint32_t k = 0; k < c.elements(); ++k, ++at) hs += "    o." + c.at(k) + " = " + c.from_u4("cpv[b + " + std::to_string(at) + "u]") + ";\n";
        }
        if (cps.empty()) hs += "    o.unused = uint4(0u, 0u, 0u, 0u);\n";
        hs += "    return o;\n}\n";
        const std::string vs = "struct VOut { [[vk::location(0)]] uint id : VID; };\nVOut main(uint vid : SV_VertexID) { VOut o; o.id = vid; return o; }\n";
        const Rows rows = gs_rows(m, kTessColumns, static_cast<int>(r.capture_set));
        std::memset(r.capture.map, 0, 16);
        const std::string ps = "[[vk::location(0)]] uint4 main([[vk::location(0)]] nointerpolation uint4 bits : BITS) : SV_Target0 { return bits; }\n";
        const VkShaderModule vsm = helper(vs, "vs_6_6"), hsm = helper(hs, "hs_6_6"), gsm = helper(rows.gs, "gs_6_6"), psm = helper(ps, "ps_6_6"),
                             dsm = module(words);
        VkPipeline pipeline = graphics_pipeline(r.layout, vsm, gsm, psm, {VK_FORMAT_R32G32B32A32_UINT}, false, true, {}, hsm, dsm, patch);
        Run run = draw_rows(pipeline, r, kTessColumns, rows.rows, patch * kPatches, false);
        vkDestroyPipeline(dev.device, pipeline, nullptr);
        vkDestroyShaderModule(dev.device, dsm, nullptr);
        return run;
    }

    // Hull shaders: a generated vertex shader gives each of kPatches patches
    // random control points; both hull modules store what they made
    // themselves (instrument_hull), whatever the factors; a generated domain
    // shader (point mode; the domain, spacing and winding `declared` says -
    // DXC's module, as the lift's hull declares none) completes the pipeline.
    Run run_hull(const Module& m, const std::vector<std::uint32_t>& words, Resources& r, const Inputs& in, const Module& declared) {
        const bool tri = declared.modes.count(22) != 0, quad = declared.modes.count(24) != 0;
        const char* part = partitioning(declared);
        if ((!tri && !quad) || !part) throw Skip("a hull shader whose HLSL module declares no triangle or quad domain and spacing");
        const std::uint32_t order = declared.modes.count(5) ? 5 : 4;
        if (!m.modes.count(26)) throw Skip("a hull shader without OutputVertices");
        const std::uint32_t out_points = m.modes.at(26).at(0);
        std::vector<Member> ins;
        std::uint32_t in_points = 0;
        for (const Var& v : m.vars) {
            if (!m.interface_ids.count(v.id) || v.sc != 1) continue;
            if (v.has(11)) {
                const std::uint32_t bi = v.get(11);
                if (bi != 8 && bi != 7 && bi != 14) throw Skip("hull input built-in " + std::to_string(bi));
                continue;
            }
            if (!v.has(30)) continue;
            std::uint32_t n = 0;
            ins.push_back(member_of(m, v, 1, n));
            if (in_points && n != in_points) throw Skip("control-point inputs of different patch sizes");
            in_points = n;
        }
        if (!in_points) throw Skip("a hull shader without control-point inputs");
        std::mt19937_64 rng(in.seed ^ 0x4a11c0deull);
        std::size_t wk = 0;
        // The vertex shader: control point `vid` of the draw.
        std::uint32_t ine = 0;
        std::string vs = "struct VOut {\n";
        for (const Member& c : ins) {
            vs += c.decl("V");
            ine += c.elements();
        }
        vs += "};\nstatic const uint4 vv[" + std::to_string(kPatches * in_points * ine) + "] = {";
        for (std::uint32_t k = 0; k < kPatches * in_points * ine; ++k) {
            std::uint32_t w4[4];
            for (std::uint32_t& x : w4) x = table_word(rng, wk++, !in.float_words);
            vs += (k ? ", " : "") + u4(w4[0], w4[1], w4[2], w4[3]);
        }
        vs += "};\nVOut main(uint vid : SV_VertexID) {\n    VOut o;\n    uint b = vid * " + std::to_string(ine) + "u;\n";
        std::uint32_t at = 0;
        for (const Member& c : ins) {
            for (std::uint32_t k = 0; k < c.elements(); ++k, ++at) vs += "    o." + c.at(k) + " = " + c.from_u4("vv[b + " + std::to_string(at) + "u]") + ";\n";
        }
        vs += "    return o;\n}\n";
        std::string ds = std::string("struct PC {\n    float outer[") + (tri ? "3" : "4") + "] : SV_TessFactor;\n    float inner" + (tri ? "" : "[2]") +
                         " : SV_InsideTessFactor;\n};\n";
        ds += std::string("[domain(\"") + (tri ? "tri" : "quad") + "\")]\nfloat4 main(PC pc) : SV_Position {\n";
        ds += std::string("    vk::ext_execution_mode(") + (std::strcmp(part, "integer") == 0 ? "1" : std::strcmp(part, "fractional_even") == 0 ? "2" : "3") +
              ");\n    vk::ext_execution_mode(" + std::to_string(order) + ");\n    vk::ext_execution_mode(10);  // PointMode\n";
        ds += "    return float4(0.0f, 0.0f, 0.0f, 1.0f);\n}\n";
        HullRecord rec;
        std::vector<std::uint32_t> instrumented;
        try {
            instrumented = instrument_hull(words, r.capture_set, out_points, rec);
        } catch (const Failure& f) {
            throw Skip(std::string("not instrumented: ") + f.what());
        }
        if (const char* dump = std::getenv("HLSL_EQUIV_DUMP"); dump && *dump) {
            static int dumped = 0;
            std::ofstream(std::string(dump) + "/hull-" + std::to_string(dumped++) + ".spv", std::ios::binary)
                .write(reinterpret_cast<const char*>(instrumented.data()), static_cast<std::streamsize>(instrumented.size() * 4));
        }
        const std::size_t bytes = static_cast<std::size_t>(rec.words()) * 4 * out_points * kPatches;
        if (bytes > r.capture.size) throw Skip("a hull's outputs past the capture buffer");
        const VkShaderModule vsm = helper(vs, "vs_6_6"), dsm = helper(ds, "ds_6_6"), hsm = module(instrumented);
        VkPipeline pipeline = graphics_pipeline(r.layout, vsm, VK_NULL_HANDLE, VK_NULL_HANDLE, {}, false, false, {}, hsm, dsm, in_points);
        std::memset(r.capture.map, 0xef, static_cast<std::size_t>(r.capture.size));  // 0xefefefef: not written
        dev.submit([&](VkCommandBuffer cmd) {
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {1, 1}};
            ri.layerCount = 1;
            vkCmdBeginRendering(cmd, &ri);
            draw(cmd, pipeline, r, 1, 1, in_points * kPatches, 0);
            vkCmdEndRendering(cmd);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        });
        Run run;
        const auto* p = static_cast<const std::uint8_t*>(r.capture.map);
        run.outputs.emplace_back(p, p + bytes);
        // The patch's words are its own only in invocation 0's record: in the
        // others they may be read before invocation 0 wrote them.
        auto& o = run.outputs.back();
        for (std::uint32_t patch = 0; patch < kPatches; ++patch) {
            for (std::uint32_t inv = 1; inv < out_points; ++inv) {
                const std::size_t rec_at = (static_cast<std::size_t>(patch) * out_points + inv) * rec.words() * 4;
                std::memset(o.data() + rec_at + static_cast<std::size_t>(rec.cp_words) * 4, 0xef, static_cast<std::size_t>(rec.patch_words) * 4);
            }
        }
        // The record's shape, so two modules that store other things differ.
        run.outputs.push_back(std::vector<std::uint8_t>(8));
        std::memcpy(run.outputs.back().data(), &rec.cp_words, 4);
        std::memcpy(run.outputs.back().data() + 4, &rec.patch_words, 4);
        vkDestroyPipeline(dev.device, pipeline, nullptr);
        vkDestroyShaderModule(dev.device, hsm, nullptr);
        return run;
    }

    struct VertexInput {
        std::vector<VkVertexInputBindingDescription> bindings;
        std::vector<VkVertexInputAttributeDescription> attributes;
    };
    // tcs and tes: tessellation stages, patch_points control points a patch;
    // no fs: the rasterizer discards (a hull's run writes a buffer only).
    VkPipeline graphics_pipeline(VkPipelineLayout layout, VkShaderModule vs, VkShaderModule gs, VkShaderModule fs, const std::vector<VkFormat>& colors,
                                 bool depth, bool points, const VertexInput& vi, VkShaderModule tcs = VK_NULL_HANDLE, VkShaderModule tes = VK_NULL_HANDLE,
                                 std::uint32_t patch_points = 0) {
        std::vector<VkPipelineShaderStageCreateInfo> st;
        const auto stage = [&](VkShaderStageFlagBits s, VkShaderModule m) {
            VkPipelineShaderStageCreateInfo i{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            i.stage = s;
            i.module = m;
            i.pName = "main";
            st.push_back(i);
        };
        stage(VK_SHADER_STAGE_VERTEX_BIT, vs);
        if (tcs) stage(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, tcs);
        if (tes) stage(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, tes);
        if (gs) stage(VK_SHADER_STAGE_GEOMETRY_BIT, gs);
        if (fs) stage(VK_SHADER_STAGE_FRAGMENT_BIT, fs);
        VkPipelineTessellationStateCreateInfo ts{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
        ts.patchControlPoints = patch_points;
        VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vis.vertexBindingDescriptionCount = static_cast<std::uint32_t>(vi.bindings.size());
        vis.pVertexBindingDescriptions = vi.bindings.data();
        vis.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(vi.attributes.size());
        vis.pVertexAttributeDescriptions = vi.attributes.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = tcs ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : points ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        rs.rasterizerDiscardEnable = fs ? VK_FALSE : VK_TRUE;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = depth;
        ds.depthWriteEnable = depth;
        ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
        std::vector<VkPipelineColorBlendAttachmentState> blend(colors.size());
        for (auto& b : blend) b.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = static_cast<std::uint32_t>(blend.size());
        cb.pAttachments = blend.data();
        const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2;
        dy.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo pr{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        pr.colorAttachmentCount = static_cast<std::uint32_t>(colors.size());
        pr.pColorAttachmentFormats = colors.data();
        pr.depthAttachmentFormat = depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED;
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.pNext = &pr;
        gp.stageCount = static_cast<std::uint32_t>(st.size());
        gp.pStages = st.data();
        gp.pVertexInputState = &vis;
        gp.pInputAssemblyState = &ia;
        if (tcs) gp.pTessellationState = &ts;
        gp.pViewportState = &vp;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dy;
        gp.layout = layout;
        VkPipeline p;
        check(vkCreateGraphicsPipelines(dev.device, VK_NULL_HANDLE, 1, &gp, nullptr, &p), "vkCreateGraphicsPipelines");
        return p;
    }

    void draw(VkCommandBuffer cmd, VkPipeline pipeline, Resources& r, std::uint32_t w, std::uint32_t h, std::uint32_t vertices, std::uint32_t) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        for (const auto& [s, set] : r.sets) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.layout, s, 1, &set, 0, nullptr);
        const VkViewport vp{0.0f, 0.0f, static_cast<float>(w), static_cast<float>(h), 0.0f, 1.0f};
        const VkRect2D sc{{0, 0}, {w, h}};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdDraw(cmd, vertices, 1, 0, 0);
    }

    static std::string diff(const Run& a, const Run& b, std::uint32_t width) {
        if (a.outputs.size() != b.outputs.size()) return "different: target count";
        // A word both runs wrote as a NaN, with another sign or payload: SPIR-V
        // and Vulkan leave a NaN's bits to the implementation, and lavapipe's
        // compiler moves them between equal computations of different shapes.
        const auto nan = [](std::uint32_t v) { return (v & 0x7f800000u) == 0x7f800000u && (v & 0x7fffffu); };
        std::size_t nan_only = 0;
        std::size_t texels = 0, first_target = 0, first_at = 0;
        bool found = false;
        for (std::size_t t = 0; t < a.outputs.size(); ++t) {
            const auto& x = a.outputs[t];
            const auto& y = b.outputs[t];
            if (x.size() != y.size()) return "different: target size";
            const std::size_t texel = x.size() / std::max<std::size_t>(1, width) ? 16 : 16;
            (void)texel;
            for (std::size_t k = 0; k + 4 <= x.size(); k += 4) {
                if (std::memcmp(&x[k], &y[k], 4) == 0) continue;
                std::uint32_t xa, ya;
                std::memcpy(&xa, &x[k], 4);
                std::memcpy(&ya, &y[k], 4);
                if (nan(xa) && nan(ya)) {
                    ++nan_only;
                    continue;
                }
                ++texels;
                if (!found) {
                    found = true;
                    first_target = t;
                    first_at = k;
                }
            }
        }
        if (!found) {
            // How much of the output the draw wrote (texel words that are not
            // the clear pattern) and how varied it is: an identical output that
            // is all clear colour proves little.
            std::size_t written = 0, words = 0;
            std::set<std::uint32_t> distinct;
            for (const auto& x : a.outputs) {
                for (std::size_t k = 0; k + 4 <= x.size(); k += 4) {
                    std::uint32_t v;
                    std::memcpy(&v, &x[k], 4);
                    ++words;
                    if (v != 0xdeadbeefu && v != 0x7fc0deadu && v != 0x12345678u && v != 0xfedcba98u && v != 0x3e800000u && v != 0xefefefefu) ++written;
                    if (distinct.size() < 100000) distinct.insert(v);
                }
            }
            char buf[160];
            if (nan_only) {
                std::snprintf(buf, sizeof(buf), "nan-bits: identical but for %zu NaN words with other bits; %zu%% of %zu words written, %zu distinct", nan_only,
                              words ? written * 100 / words : 0, words, distinct.size());
                return buf;
            }
            std::snprintf(buf, sizeof(buf), "identical: %zu%% of %zu words written, %zu distinct", words ? written * 100 / words : 0, words, distinct.size());
            return buf;
        }
        std::uint32_t va, vb;
        std::memcpy(&va, &a.outputs[first_target][first_at], 4);
        std::memcpy(&vb, &b.outputs[first_target][first_at], 4);
        char buf[200];
        std::snprintf(buf, sizeof(buf), "different: %zu words; first in target %zu word %zu: %08x (lift) against %08x (HLSL)", texels, first_target,
                      first_at / 4, va, vb);
        return buf;
    }
};

}  // namespace

int main(int argc, char** argv) {
    Inputs in;
    std::uint32_t shard = 0, shards = 1;
    std::string only;
    const char* list_path = nullptr;
    bool verbose = false;
    const char* progress_path = nullptr;  // the program being run, written before it runs: a crash names it
    int timeout_s = 60;                   // a program still running after this is left out (a loop counting random words)
    std::vector<fs::path> dirs;
    bool compile_only = false;  // --compile: every compute module (*.spv) built into a pipeline, the driver's result printed
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--size" && a + 1 < argc) in.size = static_cast<std::uint32_t>(std::atoi(argv[++a]));
        else if (arg == "--vertices" && a + 1 < argc) in.vertices = static_cast<std::uint32_t>(std::atoi(argv[++a]));
        else if (arg == "--seed" && a + 1 < argc) in.seed = std::strtoull(argv[++a], nullptr, 0);
        else if (arg == "--shard" && a + 1 < argc) std::sscanf(argv[++a], "%u/%u", &shard, &shards);
        else if (arg == "--only" && a + 1 < argc) only = argv[++a];
        else if (arg == "--list" && a + 1 < argc) list_path = argv[++a];
        else if (arg == "--verbose") verbose = true;
        else if (arg == "--progress" && a + 1 < argc) progress_path = argv[++a];
        else if (arg == "--small-ints") in.small_ints = true;
        else if (arg == "--float-words") in.float_words = true;
        else if (arg == "--timeout" && a + 1 < argc) timeout_s = std::atoi(argv[++a]);
        else if (arg == "--compile") compile_only = true;
        else dirs.push_back(arg);
    }
    if (dirs.empty() || !shards) {
        std::fprintf(stderr, "usage: hlsl_equiv [--size N] [--vertices N] [--seed S] [--shard I/N] [--only SUBSTR] [--list FILE] [--small-ints] [--float-words] "
                             "[--timeout S] [--progress FILE] [--compile] <dir>...\n");
        return 2;
    }
    std::vector<fs::path> lifts;
    for (const fs::path& d : dirs) {
        for (const auto& e : fs::recursive_directory_iterator(d)) {
            const std::string n = e.path().filename().string();
            if (n.size() > 12 && n.compare(n.size() - 9, 9, ".lift.spv") == 0 &&
                (n.find(".ps.lift") != std::string::npos || n.find(".vs.lift") != std::string::npos || n.find(".ls.lift") != std::string::npos ||
                 n.find(".hs.lift") != std::string::npos || n.find(".ds.lift") != std::string::npos))
                lifts.push_back(e.path());
        }
    }
    std::sort(lifts.begin(), lifts.end());
    gpu::Dxc dxc;
    if (!dxc.load()) {
        std::fprintf(stderr, "DXC: %s\n", dxc.error().c_str());
        return 2;
    }
    Device dev;
    try {
        dev.init();
    } catch (const Failure& f) {
        std::fprintf(stderr, "%s\n", f.what());
        return 2;
    }
    std::printf("device: %s; quad operations in pixel shaders: %s; %s; run-time sample offsets (maintenance8): %s\n", dev.name.c_str(), dev.quad_ops ? "yes" : "no", dev.robust.c_str(),
                dev.runtime_offsets ? "yes" : "no");
    // --progress: a shard that crashed (a shader faulting inside the driver)
    // resumes past the program its progress file names, which is listed as crashed.
    std::string resume_after;
    if (progress_path) {
        if (FILE* f = std::fopen(progress_path, "r")) {
            char line[4096] = {};
            if (std::fgets(line, sizeof(line), f)) resume_after = line;
            while (!resume_after.empty() && (resume_after.back() == '\n' || resume_after.back() == '\r')) resume_after.pop_back();
            std::fclose(f);
        }
    }
    FILE* list = list_path ? std::fopen(list_path, resume_after.empty() ? "w" : "a") : nullptr;
    Harness h(dev, dxc);
    if (compile_only) {
        // Every compute module under the folders, the lift's and DXC's (or any
        // other variant), each built into a pipeline on this device.
        std::vector<fs::path> modules;
        for (const fs::path& d : dirs) {
            for (const auto& e : fs::recursive_directory_iterator(d)) {
                const std::string n = e.path().filename().string();
                if (n.size() > 4 && n.compare(n.size() - 4, 4, ".spv") == 0 && (only.empty() || e.path().string().find(only) != std::string::npos))
                    modules.push_back(e.path());
            }
        }
        std::sort(modules.begin(), modules.end());
        std::map<std::string, std::size_t> results;
        for (const fs::path& f : modules) {
            std::string r;
            try {
                const VkResult v = h.compile_compute(read_words(f));
                r = v == VK_SUCCESS ? "compiled" : "refused: VkResult " + std::to_string(static_cast<int>(v));
            } catch (const Skip&) {
                continue;
            } catch (const std::exception& e) {
                r = std::string("error: ") + e.what();
            }
            std::printf("%s %s\n", f.string().c_str(), r.c_str());
            if (list) std::fprintf(list, "%s %s\n", f.string().c_str(), r.c_str());
            results[r.substr(0, r.find(':'))]++;
        }
        std::printf("totals:");
        for (const auto& [k, n] : results) std::printf(" %s %zu;", k.c_str(), n);
        std::printf("\n");
        if (list) std::fclose(list);
        return results.count("refused") || results.count("error") ? 1 : 0;
    }
    std::map<std::string, std::size_t> totals;
    std::size_t index = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const fs::path& lift : lifts) {
        if (index++ % shards != shard) continue;
        const std::string ls = lift.string();
        if (!only.empty() && ls.find(only) == std::string::npos) continue;
        const fs::path other = ls.substr(0, ls.size() - 9) + ".dxc.spv";
        if (!fs::exists(other)) continue;
        if (!resume_after.empty()) {
            if (ls == resume_after || "done:" + ls == resume_after) {
                if (ls == resume_after) {
                    const std::string name = fs::relative(lift, lift.parent_path().parent_path()).string();
                    std::printf("%s crashed: the driver faulted running it\n", name.c_str());
                    if (list) std::fprintf(list, "%s crashed: the driver faulted running it\n", name.c_str());
                    totals["crashed"]++;
                }
                resume_after.clear();
            }
            continue;
        }
        if (progress_path) {
            if (FILE* f = std::fopen(progress_path, "w")) {
                std::fprintf(f, "%s\n", ls.c_str());
                std::fclose(f);
            }
        }
        std::string result;
        // The watchdog: a draw still running past the timeout (a loop over a
        // random count) is listed as such and the process ends; the runner
        // resumes past it (the progress file says it is done).
        static std::atomic<std::uint64_t> deadline{0};
        static std::string watched_name, watched_path;
        static FILE* watched_list = nullptr;
        static const char* watched_progress = nullptr;
        static bool watchdog_started = false;
        watched_name = fs::relative(lift, lift.parent_path().parent_path()).string();
        watched_path = ls;
        watched_list = list;
        watched_progress = progress_path;
        deadline = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count()) +
                   static_cast<std::uint64_t>(timeout_s);
        if (!watchdog_started) {
            watchdog_started = true;
            std::thread([] {
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    const std::uint64_t now = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
                    const std::uint64_t d = deadline.load();
                    if (!d || now < d) continue;
                    std::printf("%s timeout: still running after the time limit\n", watched_name.c_str());
                    std::fflush(stdout);
                    if (watched_list) {
                        std::fprintf(watched_list, "%s timeout: still running after the time limit\n", watched_name.c_str());
                        std::fflush(watched_list);
                    }
                    if (watched_progress) {
                        if (FILE* f = std::fopen(watched_progress, "w")) {
                            std::fprintf(f, "done:%s\n", watched_path.c_str());
                            std::fclose(f);
                        }
                    }
                    _exit(3);
                }
            }).detach();
        }
        try {
            result = h.compare(lift, other, in);
        } catch (const Skip& s) {
            result = std::string("skipped: ") + s.what();
        } catch (const Failure& f) {
            result = std::string("error: ") + f.what();
        }
        deadline = 0;
        const std::string key = result.substr(0, result.find(':'));
        totals[key]++;
        const std::string name = fs::relative(lift, lift.parent_path().parent_path()).string();
        if (key != "identical" || verbose) std::printf("%s %s\n", name.c_str(), result.c_str());
        if (list) {
            std::fprintf(list, "%s %s\n", name.c_str(), result.c_str());
            std::fflush(list);
        }
        std::fflush(stdout);
    }
    if (list) std::fclose(list);
    if (progress_path) std::remove(progress_path);
    std::printf("totals (%.0f s):", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    for (const auto& [k, n] : totals) std::printf(" %s %zu;", k.c_str(), n);
    std::printf("\n");
    return totals.count("different") ? 1 : 0;
}
