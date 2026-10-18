#include "pipeline_probe.h"

#include "gcn/half.h"
#include "gcn/hlsl.h"
#include "gcn/translate.h"
#include "host/dxc.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace probe {

namespace {

// ---- a module's interface -------------------------------------------------------
enum : std::uint32_t {
    kOpEntryPoint = 15, kOpExecutionMode = 16, kOpTypeBool = 20, kOpTypeInt = 21, kOpTypeFloat = 22, kOpTypeVector = 23,
    kOpTypeImage = 25, kOpTypeSampler = 26, kOpTypeSampledImage = 27, kOpTypeArray = 28, kOpTypeRuntimeArray = 29, kOpTypeStruct = 30,
    kOpTypePointer = 32, kOpConstant = 43, kOpVariable = 59, kOpDecorate = 71, kOpMemberDecorate = 72,
};
enum : std::uint32_t { kDecBlock = 2, kDecBufferBlock = 3, kDecBuiltIn = 11, kDecNoPerspective = 13, kDecFlat = 14, kDecPatch = 15,
                       kDecCentroid = 16, kDecSample = 17, kDecLocation = 30, kDecComponent = 31, kDecBinding = 33, kDecDescriptorSet = 34 };
enum : std::uint32_t { kScUniformConstant = 0, kScInput = 1, kScUniform = 2, kScOutput = 3, kScPushConstant = 9, kScStorageBuffer = 12 };
enum : std::uint32_t { kModelVertex = 0, kModelTessControl = 1, kModelTessEval = 2, kModelFragment = 4, kModelCompute = 5 };
enum : std::uint32_t { kBuiltinPosition = 0, kBuiltinClipDistance = 3, kBuiltinCullDistance = 4, kBuiltinFragDepth = 22, kBuiltinStencilRef = 5014 };

struct Type {
    std::uint32_t op = 0;
    std::vector<std::uint32_t> w;  // the instruction's words after its result id
};
struct Var {
    std::uint32_t id = 0, sc = 0, pointee = 0;
    std::map<std::uint32_t, std::vector<std::uint32_t>> dec;
    bool has(std::uint32_t d) const { return dec.count(d) != 0; }
    std::uint32_t get(std::uint32_t d) const {
        const auto it = dec.find(d);
        return it == dec.end() || it->second.empty() ? 0 : it->second[0];
    }
};
struct Module {
    std::uint32_t model = ~0u;
    std::map<std::uint32_t, Type> types;
    std::map<std::uint32_t, std::uint32_t> consts;  // OpConstant: id -> its first word
    std::vector<Var> vars;
    std::set<std::uint32_t> interface_ids;
    std::map<std::uint32_t, std::vector<std::uint32_t>> modes;     // OpExecutionMode: mode -> literals
    std::map<std::uint32_t, std::set<std::uint32_t>> decorated;    // id -> its decorations (a struct's Block)
    std::map<std::uint32_t, std::set<std::uint32_t>> member_builtins;  // struct id -> its members' built-ins
};
struct Unplaced : std::runtime_error {  // what a probe cannot place: the verdict is "untried"
    using std::runtime_error::runtime_error;
};

Module parse(const std::vector<std::uint32_t>& w) {
    if (w.size() < 5 || w[0] != 0x07230203u) throw Unplaced("not a SPIR-V module");
    Module m;
    std::map<std::uint32_t, std::map<std::uint32_t, std::vector<std::uint32_t>>> dec;
    for (std::size_t i = 5; i < w.size();) {
        const std::uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n || i + n > w.size()) throw Unplaced("a malformed module");
        const std::uint32_t* a = &w[i + 1];
        switch (op) {
        case kOpEntryPoint:
            if (m.model == ~0u && n >= 3) {
                m.model = a[0];
                std::size_t k = 2;
                for (; k < n - 1; ++k) {  // past the name: the word holding its terminating zero
                    const std::uint32_t x = a[k];
                    if (!(x & 0xffu) || !(x & 0xff00u) || !(x & 0xff0000u) || !(x & 0xff000000u)) {
                        ++k;
                        break;
                    }
                }
                for (; k < n - 1; ++k) m.interface_ids.insert(a[k]);
            }
            break;
        case kOpExecutionMode:
            if (n >= 3) m.modes[a[1]] = std::vector<std::uint32_t>(a + 2, a + n - 1);
            break;
        case kOpDecorate:
            if (n >= 3) dec[a[0]][a[1]] = std::vector<std::uint32_t>(a + 2, a + n - 1);
            break;
        case kOpMemberDecorate:
            if (n >= 5 && a[2] == kDecBuiltIn) m.member_builtins[a[0]].insert(a[3]);
            break;
        case kOpConstant:
            if (n >= 4) m.consts[a[1]] = a[2];
            break;
        case kOpVariable:
            if (n >= 4) {
                Var v;
                v.id = a[1];
                v.sc = a[2];
                const auto pt = m.types.find(a[0]);
                if (pt != m.types.end() && pt->second.op == kOpTypePointer && pt->second.w.size() >= 2) v.pointee = pt->second.w[1];
                m.vars.push_back(v);
            }
            break;
        default:
            if (op >= 19 && op <= 39 && n >= 2) {  // the OpType* instructions
                Type t;
                t.op = op;
                t.w.assign(a + 1, a + n - 1);
                m.types[a[0]] = t;
            }
            break;
        }
        i += n;
    }
    for (Var& v : m.vars) {
        const auto it = dec.find(v.id);
        if (it != dec.end()) v.dec = it->second;
    }
    for (const auto& [id, d] : dec) {
        for (const auto& kv : d) m.decorated[id].insert(kv.first);
    }
    return m;
}

const Type& type_of(const Module& m, std::uint32_t id) {
    const auto it = m.types.find(id);
    if (it == m.types.end()) throw Unplaced("a type the probe does not read");
    return it->second;
}
std::uint32_t array_length(const Module& m, const Type& t) {
    const auto it = m.consts.find(t.w.at(1));
    if (it == m.consts.end()) throw Unplaced("an array whose length is no constant");
    return it->second;
}
// A scalar or vector: its kind (f, i, u), width in bits and component count.
struct Shape {
    char kind = 'f';
    std::uint32_t bits = 32, comps = 1;
};
bool shape_of(const Module& m, std::uint32_t id, Shape& s) {
    const Type* t = &type_of(m, id);
    s.comps = 1;
    if (t->op == kOpTypeVector) {
        s.comps = t->w.at(1);
        t = &type_of(m, t->w.at(0));
    }
    if (t->op == kOpTypeFloat) {
        s.kind = 'f';
        s.bits = t->w.at(0);
        return true;
    }
    if (t->op == kOpTypeInt) {
        s.kind = t->w.at(1) ? 'i' : 'u';
        s.bits = t->w.at(0);
        return true;
    }
    return false;
}
// An interface variable's element: a scalar or vector, or an array of them
// (`count`), after `outer` per-vertex or per-control-point arrays.
struct Element {
    Shape shape;
    std::uint32_t count = 0;  // 0: not an array
};
Element element_of(const Module& m, const Var& v, std::uint32_t outer, std::uint32_t* outer_count = nullptr) {
    std::uint32_t t = v.pointee;
    for (std::uint32_t k = 0; k < outer; ++k) {
        const Type& ty = type_of(m, t);
        if (ty.op != kOpTypeArray) throw Unplaced("a per-control-point variable that is no array");
        if (outer_count) *outer_count = array_length(m, ty);
        t = ty.w.at(0);
    }
    Element e;
    const Type& ty = type_of(m, t);
    if (ty.op == kOpTypeArray) {
        e.count = array_length(m, ty);
        t = ty.w.at(0);
    }
    if (!shape_of(m, t, e.shape)) throw Unplaced("a stage variable of an aggregate type");
    if (e.shape.bits != 32) throw Unplaced("a " + std::to_string(e.shape.bits) + "-bit stage variable");
    return e;
}
std::string hlsl_type(const Shape& s) {
    const std::string base = s.kind == 'f' ? "float" : s.kind == 'i' ? "int" : "uint";
    return s.comps > 1 ? base + std::to_string(s.comps) : base;
}
std::uint32_t builtin_block_of(const Module& m, std::uint32_t pointee) {
    // gl_PerVertex arrays: the struct inside, or 0
    std::uint32_t t = pointee;
    for (int k = 0; k < 2; ++k) {
        const auto it = m.types.find(t);
        if (it == m.types.end()) return 0;
        if (it->second.op == kOpTypeStruct) return m.member_builtins.count(t) ? t : 0;
        if (it->second.op != kOpTypeArray) return 0;
        t = it->second.w.at(0);
    }
    const auto it = m.types.find(t);
    return it != m.types.end() && it->second.op == kOpTypeStruct && m.member_builtins.count(t) ? t : 0;
}

// ---- descriptor bindings ------------------------------------------------------------
struct Binding {
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    std::uint32_t count = 1;
};
using Bindings = std::map<std::pair<std::uint32_t, std::uint32_t>, Binding>;  // (set, binding)

void add_bindings(const Module& m, Bindings& out, bool& push) {
    for (const Var& v : m.vars) {
        if (v.sc == kScPushConstant) {
            push = true;
            continue;
        }
        if (v.sc != kScUniformConstant && v.sc != kScUniform && v.sc != kScStorageBuffer) continue;
        if (!v.has(kDecDescriptorSet) || !v.has(kDecBinding)) continue;
        std::uint32_t t = v.pointee, count = 1;
        const Type& outer = type_of(m, t);
        if (outer.op == kOpTypeArray) {
            count = array_length(m, outer);
            t = outer.w.at(0);
        } else if (outer.op == kOpTypeRuntimeArray) {
            throw Unplaced("a run-time array of descriptors");
        }
        const Type& ty = type_of(m, t);
        Binding b;
        b.count = count;
        if (v.sc == kScStorageBuffer) {
            b.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        } else if (v.sc == kScUniform) {
            const auto d = m.decorated.find(t);
            b.type = d != m.decorated.end() && d->second.count(kDecBufferBlock) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        } else if (ty.op == kOpTypeSampler) {
            b.type = VK_DESCRIPTOR_TYPE_SAMPLER;
        } else if (ty.op == kOpTypeSampledImage) {
            b.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        } else if (ty.op == kOpTypeImage) {
            const std::uint32_t dim = ty.w.at(1), sampled = ty.w.at(5);
            if (dim == 5) b.type = sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
            else if (dim == 6) b.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            else b.type = sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        } else {
            throw Unplaced("a descriptor of a kind the probe does not bind (type op " + std::to_string(ty.op) + ")");
        }
        const std::pair<std::uint32_t, std::uint32_t> key{v.get(kDecDescriptorSet), v.get(kDecBinding)};
        const auto [it, inserted] = out.try_emplace(key, b);
        if (!inserted && (it->second.type != b.type || it->second.count != b.count))
            throw Unplaced("the stages bind set " + std::to_string(key.first) + " binding " + std::to_string(key.second) + " differently");
    }
}

// ---- stage interfaces ----------------------------------------------------------------
struct VertexInput {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
};
VkFormat attribute_format(const Shape& s) {
    static const VkFormat f[3][4] = {
        {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
        {VK_FORMAT_R32_SINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32A32_SINT},
        {VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32A32_UINT},
    };
    return f[s.kind == 'f' ? 0 : s.kind == 'i' ? 1 : 2][std::clamp<std::uint32_t>(s.comps, 1, 4) - 1];
}
// A vertex stage's inputs as attributes of one binding, 16 bytes a location.
VertexInput vertex_input_of(const Module& m) {
    VertexInput vi;
    std::uint32_t offset = 0;
    std::vector<std::pair<std::uint32_t, VkFormat>> locs;
    for (const Var& v : m.vars) {
        if (v.sc != kScInput || !m.interface_ids.count(v.id) || v.has(kDecBuiltIn) || !v.has(kDecLocation)) continue;
        const Element e = element_of(m, v, 0);
        for (std::uint32_t k = 0; k < std::max(1u, e.count); ++k) locs.emplace_back(v.get(kDecLocation) + k, attribute_format(e.shape));
    }
    std::sort(locs.begin(), locs.end());
    for (const auto& [loc, format] : locs) {
        vi.attributes.push_back({loc, 0, format, offset});
        offset += 16;
    }
    if (offset) vi.bindings.push_back({0, offset, VK_VERTEX_INPUT_RATE_VERTEX});
    return vi;
}
struct Outputs {
    std::vector<VkFormat> colors;  // by location; VK_FORMAT_UNDEFINED where nothing is written
    VkFormat depth = VK_FORMAT_UNDEFINED, stencil = VK_FORMAT_UNDEFINED;
};
Outputs outputs_of(const Module* ps, VkFormat depth_stencil, bool has_stencil) {
    Outputs o;
    o.depth = depth_stencil;
    o.stencil = has_stencil ? depth_stencil : VK_FORMAT_UNDEFINED;
    if (!ps) return o;
    for (const Var& v : ps->vars) {
        if (v.sc != kScOutput || !ps->interface_ids.count(v.id) || v.has(kDecBuiltIn) || !v.has(kDecLocation)) continue;
        const Element e = element_of(*ps, v, 0);
        const VkFormat f = e.shape.kind == 'f' ? VK_FORMAT_R16G16B16A16_SFLOAT
                           : e.shape.kind == 'i' ? VK_FORMAT_R32G32B32A32_SINT
                                                 : VK_FORMAT_R32G32B32A32_UINT;
        for (std::uint32_t k = 0; k < std::max(1u, e.count); ++k) {
            const std::uint32_t loc = v.get(kDecLocation) + k;
            if (loc >= 8) throw Unplaced("a colour output past location 7");
            if (o.colors.size() <= loc) o.colors.resize(loc + 1, VK_FORMAT_UNDEFINED);
            o.colors[loc] = f;
        }
    }
    return o;
}

// The vertex shader feeding a pixel shader: a full-screen triangle and each
// input the pixel shader reads, zero, at its location with its type and
// interpolation (which a library link may require to agree).
std::string vertex_shader_for(const Module& ps) {
    std::string decl = "struct VOut {\n    float4 pos : SV_Position;\n";
    for (const Var& v : ps.vars) {
        if (v.sc != kScInput || !ps.interface_ids.count(v.id)) continue;
        if (v.has(kDecBuiltIn)) {
            const std::uint32_t bi = v.get(kDecBuiltIn);
            if (bi != kBuiltinClipDistance && bi != kBuiltinCullDistance) continue;  // the rasterizer's own
            const Type& at = type_of(ps, v.pointee);
            if (at.op != kOpTypeArray) throw Unplaced("a clip or cull distance that is no array");
            const std::uint32_t n = array_length(ps, at);
            const char* sem = bi == kBuiltinClipDistance ? "SV_ClipDistance" : "SV_CullDistance";
            for (std::uint32_t g = 0; g * 4 < n; ++g) {
                const std::uint32_t c = std::min<std::uint32_t>(4, n - g * 4);
                decl += "    float" + (c > 1 ? std::to_string(c) : std::string()) + " " + (bi == kBuiltinClipDistance ? "clip" : "cull") +
                        std::to_string(g) + " : " + sem + std::to_string(g) + ";\n";
            }
            continue;
        }
        if (!v.has(kDecLocation)) continue;
        if (v.has(kDecComponent)) throw Unplaced("a pixel shader input packed with others in a location");
        const Element e = element_of(ps, v, 0);
        std::string q;
        if (v.has(kDecFlat) || e.shape.kind != 'f') q += "nointerpolation ";
        if (v.has(kDecNoPerspective)) q += "noperspective ";
        if (v.has(kDecCentroid)) q += "centroid ";
        if (v.has(kDecSample)) q += "sample ";
        const std::string loc = std::to_string(v.get(kDecLocation));
        decl += "    [[vk::location(" + loc + ")]] " + q + hlsl_type(e.shape) + " a" + loc + (e.count ? "[" + std::to_string(e.count) + "]" : "") +
                " : A" + loc + ";\n";
    }
    return decl +
           "};\nVOut main(uint vid : SV_VertexID) {\n    VOut o = (VOut)0;\n    float2 uv = float2((vid << 1) & 2, vid & 2);\n"
           "    o.pos = float4(uv * 2.0f - 1.0f, 0.5f, 1.0f);\n    return o;\n}\n";
}

// The vertex and hull shaders in front of a domain shader with no game hull:
// each of its control points and patch constants, zero, and factors of 3.
struct HullFor {
    std::string vs, hs;
    std::uint32_t control_points = 1;
};
HullFor hull_for(const Module& ds, const Module& modes) {
    const auto mode = [&](std::uint32_t k) { return modes.modes.count(k) != 0 || ds.modes.count(k) != 0; };
    const bool tri = mode(22), quad = mode(24), iso = mode(25);
    if (!tri && !quad && !iso) throw Unplaced("a domain shader with no domain");
    const char* domain = tri ? "tri" : quad ? "quad" : "isoline";
    const char* partitioning = mode(2) ? "fractional_even" : mode(3) ? "fractional_odd" : "integer";
    const char* topology = mode(10) ? "point" : iso ? "line" : mode(5) ? "triangle_ccw" : "triangle_cw";
    std::string cp, pc;
    std::uint32_t patch = 0;
    for (const Var& v : ds.vars) {
        if (v.sc != kScInput || !ds.interface_ids.count(v.id)) continue;
        if (v.has(kDecBuiltIn)) continue;  // the tessellator's: coordinate, levels, patch size, primitive
        if (!v.has(kDecLocation)) {
            if (const std::uint32_t block = builtin_block_of(ds, v.pointee)) {
                const std::set<std::uint32_t>& members = ds.member_builtins.at(block);
                if (members.count(kBuiltinPosition)) cp += "    float4 pos : SV_Position;\n";
                continue;
            }
            throw Unplaced("a domain shader input with neither location nor built-in");
        }
        const std::string loc = std::to_string(v.get(kDecLocation));
        if (v.has(kDecPatch)) {
            const Element e = element_of(ds, v, 0);
            pc += "    [[vk::location(" + loc + ")]] " + hlsl_type(e.shape) + " p" + loc + (e.count ? "[" + std::to_string(e.count) + "]" : "") +
                  " : PC" + loc + ";\n";
            continue;
        }
        std::uint32_t n = 0;
        const Element e = element_of(ds, v, 1, &n);
        if (patch && n != patch) throw Unplaced("control-point inputs of different patch sizes");
        patch = n;
        cp += "    [[vk::location(" + loc + ")]] " + hlsl_type(e.shape) + " c" + loc + (e.count ? "[" + std::to_string(e.count) + "]" : "") + " : CP" +
              loc + ";\n";
    }
    if (!patch) patch = 1;
    if (cp.empty()) cp = "    [[vk::location(31)]] uint4 unused : CPU;\n";
    HullFor h;
    h.control_points = patch;
    const std::string n = std::to_string(patch);
    const std::string vout = "struct VOut { [[vk::location(0)]] uint id : VID; };\n";
    h.vs = vout + "VOut main(uint vid : SV_VertexID) {\n    VOut o;\n    o.id = vid;\n    return o;\n}\n";
    const int outer = tri ? 3 : quad ? 4 : 2;
    std::string factors;
    for (int k = 0; k < outer; ++k) factors += "    o.outer[" + std::to_string(k) + "] = 3.0f;\n";
    if (tri) factors += "    o.inner = 3.0f;\n";
    if (quad) factors += "    o.inner[0] = 3.0f;\n    o.inner[1] = 3.0f;\n";
    h.hs = vout + "struct CP {\n" + cp + "};\nstruct PC {\n    float outer[" + std::to_string(outer) + "] : SV_TessFactor;\n" +
           (tri ? "    float inner : SV_InsideTessFactor;\n" : quad ? "    float inner[2] : SV_InsideTessFactor;\n" : "") + pc +
           "};\nPC pcf(InputPatch<VOut, " + n + "> ip, uint pid : SV_PrimitiveID) {\n    PC o = (PC)0;\n" + factors + "    return o;\n}\n" +
           "[domain(\"" + domain + "\")]\n[partitioning(\"" + partitioning + "\")]\n[outputtopology(\"" + topology + "\")]\n" +
           "[outputcontrolpoints(" + n + ")]\n[patchconstantfunc(\"pcf\")]\n" + "CP main(InputPatch<VOut, " + n +
           "> ip, uint i : SV_OutputControlPointID) {\n    CP o = (CP)0;\n    return o;\n}\n";
    return h;
}

// What the driver and the validation layer say while a creation runs on this thread.
thread_local std::string* t_messages = nullptr;
thread_local std::string* t_errors = nullptr;
VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                          const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (!data || !data->pMessage) return VK_FALSE;
    const std::string m(data->pMessage, std::min<std::size_t>(std::strlen(data->pMessage), 400));
    if (t_messages && (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT))) {
        if (!t_messages->empty()) *t_messages += " | ";
        *t_messages += m;
    }
    if (t_errors && (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) && t_errors->empty()) *t_errors = m;
    return VK_FALSE;
}

bool env_off(const char* name) {
    const char* e = std::getenv(name);
    return e && e[0] == '0';
}

}  // namespace

std::string Verdict::str() const {
    if (ok) return note.empty() ? "built" : "built (" + note + ")";
    if (skipped) return "not tried: " + what;
    std::string s = "refused: " + what + " VkResult " + std::to_string(result);
    if (!message.empty()) s += " (" + message + ")";
    return s;
}
std::string Verdict::token() const {
    if (ok) return "built";
    if (skipped) return "untried";
    std::string w = what;
    for (char& c : w) {
        if (c == ' ') c = '-';
    }
    return "refused:" + w + ":" + std::to_string(result);
}

std::string DeviceInfo::describe() const {
    char api_s[32];
    std::snprintf(api_s, sizeof(api_s), "%u.%u.%u", VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), VK_API_VERSION_PATCH(api));
    std::string s = "device " + std::to_string(index) + ": " + name + " (" + driver + ", Vulkan " + api_s + ")" + (validation ? ", validation layer on" : "") + "\n";
    s += std::string("  pipelines: ") +
         (libraries ? "as the renderer builds them, from graphics pipeline libraries: the stage's own library, then a fast link and an optimized link"
                    : "whole pipelines (no fast-linking graphics pipeline libraries, or BBHOST_PIPELINE_LIBRARY=0 / --monolithic)") +
         "; depth clamp " + (dynamic_depth_clamp ? "dynamic" : "in the pipeline") + "\n";
    s += std::string("  lifter as bbhost sets it up here: run-time sample offsets ") + (runtime_offsets ? "on (maintenance8)" : "off") +
         ", native half rounding " + (native_half_rtz ? "on" : "off") + "; pixel shaders at ";
    if (ps_wave == 1) s += "the driver's choice of subgroup size where they compute the same in 32 lanes (sizes " + std::to_string(subgroup_min) + "-" + std::to_string(subgroup_max) + ")";
    else if (ps_wave == 32) s += "a required subgroup size of 32 where they compute the same in 32 lanes, " + std::to_string(subgroup_size) + " for the rest";
    else s += "the device's subgroup size " + std::to_string(subgroup_size);
    return s;
}

struct Prober::Impl {
    gpu::Dxc& dxc;
    DeviceInfo info;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger = nullptr;
    VkFormat depth_stencil = VK_FORMAT_D32_SFLOAT;
    bool has_stencil = false;
    std::uint32_t push_size = 128;
    std::mutex helper_mu;
    std::map<std::string, std::vector<std::uint32_t>> helpers;  // generated stage's source -> DXC's SPIR-V, under helper_mu

    explicit Impl(gpu::Dxc& d) : dxc(d) {}
    ~Impl() {
        if (device) vkDestroyDevice(device, nullptr);
        if (messenger && destroy_messenger) destroy_messenger(instance, messenger, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    // A generated stage's SPIR-V (compiled once a source).
    std::vector<std::uint32_t> helper(const std::string& source, const char* profile) {
        const std::string key = std::string(profile) + "\n" + source;
        {
            std::lock_guard<std::mutex> lk(helper_mu);
            const auto it = helpers.find(key);
            if (it != helpers.end()) return it->second;
        }
        gcn::HlslResult h;
        h.source = source;
        h.profile = profile;
        gpu::DxcOptions o;
        std::vector<std::uint32_t> spirv;
        std::string error;
        if (!dxc.compile(h, o, spirv, error)) throw Unplaced(std::string("a generated ") + profile + " stage did not compile: " + error.substr(0, 200));
        std::lock_guard<std::mutex> lk(helper_mu);
        helpers.emplace(key, spirv);
        return spirv;
    }

    // ---- what one probe creates, destroyed with it ----
    struct Scope {
        Impl& p;
        Verdict& v;
        std::vector<VkShaderModule> modules;
        std::vector<VkDescriptorSetLayout> set_layouts;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::vector<VkPipeline> libraries, pipelines;
        Scope(Impl& impl, Verdict& verdict) : p(impl), v(verdict) {}
        ~Scope() {
            for (VkPipeline x : pipelines) vkDestroyPipeline(p.device, x, nullptr);
            for (VkPipeline x : libraries) vkDestroyPipeline(p.device, x, nullptr);
            if (layout) vkDestroyPipelineLayout(p.device, layout, nullptr);
            for (VkDescriptorSetLayout l : set_layouts) vkDestroyDescriptorSetLayout(p.device, l, nullptr);
            for (VkShaderModule m : modules) vkDestroyShaderModule(p.device, m, nullptr);
        }
        VkShaderModule module(const std::vector<std::uint32_t>& words) {
            VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            ci.codeSize = words.size() * 4;
            ci.pCode = words.data();
            VkShaderModule m = VK_NULL_HANDLE;
            if (vkCreateShaderModule(p.device, &ci, nullptr, &m) != VK_SUCCESS) throw Unplaced("the driver made no shader module of it");
            modules.push_back(m);
            return m;
        }
        void make_layout(const Bindings& b, bool push) {
            std::uint32_t top = 0;
            for (const auto& [key, binding] : b) top = std::max(top, key.first + 1);
            for (std::uint32_t set = 0; set < top; ++set) {
                std::vector<VkDescriptorSetLayoutBinding> list;
                for (const auto& [key, binding] : b) {
                    if (key.first == set) list.push_back({key.second, binding.type, binding.count, VK_SHADER_STAGE_ALL, nullptr});
                }
                VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                li.bindingCount = static_cast<std::uint32_t>(list.size());
                li.pBindings = list.data();
                VkDescriptorSetLayout l = VK_NULL_HANDLE;
                if (vkCreateDescriptorSetLayout(p.device, &li, nullptr, &l) != VK_SUCCESS) throw Unplaced("no descriptor set layout of its bindings");
                set_layouts.push_back(l);
            }
            VkPushConstantRange range{VK_SHADER_STAGE_ALL, 0, p.push_size};
            VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            pli.setLayoutCount = static_cast<std::uint32_t>(set_layouts.size());
            pli.pSetLayouts = set_layouts.data();
            pli.pushConstantRangeCount = push ? 1 : 0;
            pli.pPushConstantRanges = &range;
            if (vkCreatePipelineLayout(p.device, &pli, nullptr, &layout) != VK_SUCCESS) throw Unplaced("no pipeline layout of its bindings");
        }
        // One creation. `ours`: it holds the module the probe is for, and a
        // refusal is the module's; otherwise it is the probe's own failing.
        VkPipeline create(VkGraphicsPipelineCreateInfo& gpci, const char* what, bool ours, bool library) {
            if (!v.ok) return VK_NULL_HANDLE;
            std::string messages, errors;
            t_messages = &messages;
            t_errors = &errors;
            VkPipeline pipe = VK_NULL_HANDLE;
            const VkResult r = vkCreateGraphicsPipelines(p.device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipe);
            t_messages = t_errors = nullptr;
            ++v.created;
            return finish(r, pipe, messages, errors, what, ours, library);
        }
        VkPipeline finish(VkResult r, VkPipeline pipe, const std::string& messages, const std::string& errors, const char* what, bool ours, bool library) {
            if (r != VK_SUCCESS) {
                if (pipe) vkDestroyPipeline(p.device, pipe, nullptr);
                v.ok = false;
                v.skipped = !ours;
                v.what = ours ? what : std::string("the driver refused the probe's own ") + what + " (VkResult " + std::to_string(r) + ")";
                v.result = r;
                v.message = messages;
                return VK_NULL_HANDLE;
            }
            if (v.warning.empty() && !errors.empty()) v.warning = std::string(what) + ": " + errors;
            (library ? libraries : pipelines).push_back(pipe);
            return pipe;
        }
    };

    struct Stage {
        VkShaderStageFlagBits stage;
        VkShaderModule module;
        bool ours;
    };
    struct Plan {
        std::vector<Stage> pre;  // the pre-rasterization stages
        VkShaderModule fragment = VK_NULL_HANDLE;
        bool fragment_ours = false;
        std::uint32_t wave64_needs = gcn::kWave64NotPixel;
        VertexInput vi;
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        std::uint32_t patch_points = 0;
        Outputs out;
    };

    // The fragment stage's subgroup size as the renderer asks for it
    // (render.cpp fragment_stage_wave): false where it asks for none.
    bool size_fragment(VkPipelineShaderStageCreateInfo& si, VkPipelineShaderStageRequiredSubgroupSizeCreateInfo& req, std::uint32_t wave64_needs) const {
        if (!info.ps_wave || !si.module) return false;
        const bool wave32 = wave64_needs == 0;
        if (wave32 && info.ps_wave == 1) {
            si.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT;
            return true;
        }
        const std::uint32_t size = wave32 ? 32 : info.subgroup_size;
        if (!info.fragment_sizes || size < info.subgroup_min || size > info.subgroup_max) return false;
        req = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        req.requiredSubgroupSize = size;
        si.pNext = &req;
        return true;
    }

    Verdict build(Scope& s, const Plan& plan) {
        Verdict& v = s.v;
        const bool pre_ours = std::any_of(plan.pre.begin(), plan.pre.end(), [](const Stage& st) { return st.ours; });
        std::vector<VkPipelineShaderStageCreateInfo> pre;
        for (const Stage& st : plan.pre) {
            VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            si.stage = st.stage;
            si.module = st.module;
            si.pName = "main";
            pre.push_back(si);
        }
        VkPipelineShaderStageCreateInfo fsi{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        fsi.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fsi.module = plan.fragment;
        fsi.pName = "main";
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req{};
        const bool sized = size_fragment(fsi, req, plan.wave64_needs);

        // The state, as render.cpp's libraries have it.
        VkPipelineVertexInputStateCreateInfo vis{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vis.vertexBindingDescriptionCount = static_cast<std::uint32_t>(plan.vi.bindings.size());
        vis.pVertexBindingDescriptions = plan.vi.bindings.data();
        vis.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(plan.vi.attributes.size());
        vis.pVertexAttributeDescriptions = plan.vi.attributes.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = plan.topology;
        VkPipelineTessellationStateCreateInfo ts{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
        ts.patchControlPoints = std::max(1u, plan.patch_points);
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth = 1.0f;
        rs.depthBiasEnable = VK_TRUE;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        std::vector<VkPipelineColorBlendAttachmentState> blends(plan.out.colors.size());
        for (VkPipelineColorBlendAttachmentState& b : blends) b.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = static_cast<std::uint32_t>(blends.size());
        cb.pAttachments = blends.data();
        VkPipelineRenderingCreateInfo formats{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        formats.colorAttachmentCount = static_cast<std::uint32_t>(plan.out.colors.size());
        formats.pColorAttachmentFormats = plan.out.colors.data();
        formats.depthAttachmentFormat = plan.out.depth;
        formats.stencilAttachmentFormat = plan.out.stencil;
        VkPipelineRenderingCreateInfo no_formats{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        std::vector<VkDynamicState> pre_dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE,
                                               VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_DEPTH_BIAS};
        if (info.dynamic_depth_clamp) pre_dyn.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
        std::vector<VkDynamicState> frag_dyn = {VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,   VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
                                                VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP,         VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                                VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK};
        if (info.depth_bounds) {
            frag_dyn.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE);
            frag_dyn.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
        }
        const std::vector<VkDynamicState> out_dyn = {VK_DYNAMIC_STATE_BLEND_CONSTANTS};

        // A refused creation with the subgroup size asked for is made again at
        // the driver's default, as the renderer does (fragment_stage_unsized):
        // when that one is taken the stage is built, with a note; when it is
        // refused too the size was not why, and the first refusal stands.
        const auto unsized_again = [&](const std::function<bool()>& attempt) {
            if (attempt() || !sized || !v.refused()) return;
            const Verdict first = v;
            v.ok = true;
            v.what.clear();
            v.message.clear();
            v.result = 0;
            fsi.pNext = nullptr;
            fsi.flags &= ~static_cast<VkPipelineShaderStageCreateFlags>(VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT);
            if (attempt()) {
                v.note = "the subgroup size asked for was refused (VkResult " + std::to_string(first.result) +
                         "); built at the driver's default, as the renderer retries";
            } else {
                const std::uint32_t created = v.created;
                v = first;
                v.created = created;
            }
        };

        if (!info.libraries) {
            std::vector<VkDynamicState> all = pre_dyn;
            all.insert(all.end(), frag_dyn.begin(), frag_dyn.end());
            all.insert(all.end(), out_dyn.begin(), out_dyn.end());
            VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dsi.dynamicStateCount = static_cast<std::uint32_t>(all.size());
            dsi.pDynamicStates = all.data();
            unsized_again([&] {
                std::vector<VkPipelineShaderStageCreateInfo> stages = pre;
                if (plan.fragment) stages.push_back(fsi);
                VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
                gpci.pNext = &formats;
                gpci.stageCount = static_cast<std::uint32_t>(stages.size());
                gpci.pStages = stages.data();
                gpci.pVertexInputState = &vis;
                gpci.pInputAssemblyState = &ia;
                gpci.pTessellationState = plan.patch_points ? &ts : nullptr;
                gpci.pViewportState = &vp;
                gpci.pRasterizationState = &rs;
                gpci.pMultisampleState = &ms;
                gpci.pDepthStencilState = &ds;
                gpci.pColorBlendState = &cb;
                gpci.pDynamicState = &dsi;
                gpci.layout = s.layout;
                return s.create(gpci, "pipeline", true, false) != VK_NULL_HANDLE;
            });
            return v;
        }

        const auto library = [&](VkGraphicsPipelineCreateInfo& gpci, VkGraphicsPipelineLibraryFlagsEXT flags, const void* rendering, const char* what,
                                 bool ours) {
            VkGraphicsPipelineLibraryCreateInfoEXT li{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT};
            li.flags = flags;
            li.pNext = const_cast<void*>(rendering);
            gpci.pNext = &li;
            gpci.flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
            return s.create(gpci, what, ours, true);
        };
        VkPipeline libs[4] = {};
        {
            VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gpci.pVertexInputState = &vis;
            gpci.pInputAssemblyState = &ia;
            libs[0] = library(gpci, VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, nullptr, "vertex input library", false);
        }
        {
            VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dsi.dynamicStateCount = static_cast<std::uint32_t>(pre_dyn.size());
            dsi.pDynamicStates = pre_dyn.data();
            VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gpci.stageCount = static_cast<std::uint32_t>(pre.size());
            gpci.pStages = pre.data();
            gpci.pTessellationState = plan.patch_points ? &ts : nullptr;
            gpci.pViewportState = &vp;
            gpci.pRasterizationState = &rs;
            gpci.pDynamicState = &dsi;
            gpci.layout = s.layout;
            libs[1] = library(gpci, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, &no_formats, "pre-rasterization library", pre_ours);
        }
        {
            VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dsi.dynamicStateCount = static_cast<std::uint32_t>(frag_dyn.size());
            dsi.pDynamicStates = frag_dyn.data();
            unsized_again([&] {
                VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
                gpci.stageCount = plan.fragment ? 1 : 0;
                gpci.pStages = &fsi;
                gpci.pMultisampleState = &ms;
                gpci.pDepthStencilState = &ds;
                gpci.pDynamicState = &dsi;
                gpci.layout = s.layout;
                libs[2] = library(gpci, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, &no_formats, "fragment library", plan.fragment_ours);
                return libs[2] != VK_NULL_HANDLE;
            });
        }
        {
            VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dsi.dynamicStateCount = static_cast<std::uint32_t>(out_dyn.size());
            dsi.pDynamicStates = out_dyn.data();
            VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gpci.pColorBlendState = &cb;
            gpci.pMultisampleState = &ms;
            gpci.pDynamicState = &dsi;
            libs[3] = library(gpci, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, &formats, "fragment output library", false);
        }
        if (!v.ok) return v;
        for (const bool optimize : {false, true}) {
            VkPipelineLibraryCreateInfoKHR lci{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
            lci.libraryCount = 4;
            lci.pLibraries = libs;
            VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gpci.pNext = &lci;
            gpci.flags = optimize ? VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT : 0;
            gpci.layout = s.layout;
            s.create(gpci, optimize ? "optimized link" : "fast link", true, false);
        }
        return v;
    }

    Verdict run(const std::function<void(Scope&, Plan&, Bindings&, bool&)>& setup) {
        Verdict v;
        try {
            Scope s(*this, v);
            Plan plan;
            Bindings b;
            bool push = false;
            setup(s, plan, b, push);
            s.make_layout(b, push);
            build(s, plan);
        } catch (const Unplaced& u) {
            v.ok = false;
            v.skipped = true;
            v.what = u.what();
        }
        return v;
    }
};

Prober::Prober(gpu::Dxc& dxc) : impl_(std::make_unique<Impl>(dxc)) {}
Prober::~Prober() = default;
const DeviceInfo& Prober::info() const { return impl_->info; }

bool Prober::init(const std::string& select, bool validate, bool monolithic, std::string& error) {
    Impl& p = *impl_;
    std::uint32_t nl = 0;
    vkEnumerateInstanceLayerProperties(&nl, nullptr);
    std::vector<VkLayerProperties> layers(nl);
    vkEnumerateInstanceLayerProperties(&nl, layers.data());
    std::uint32_t ne = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> iexts(ne);
    vkEnumerateInstanceExtensionProperties(nullptr, &ne, iexts.data());
    const auto has_iext = [&](const char* n) {
        return std::any_of(iexts.begin(), iexts.end(), [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, n) == 0; });
    };
    std::vector<const char*> want_layers, want_exts;
    if (validate) {
        if (!std::any_of(layers.begin(), layers.end(), [](const VkLayerProperties& l) { return std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0; })) {
            error = "--validate: VK_LAYER_KHRONOS_validation is not installed";
            return false;
        }
        want_layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    const bool debug_utils = has_iext(VK_EXT_DEBUG_UTILS_EXTENSION_NAME) || validate;
    if (debug_utils) want_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gcn2hlsl sweep";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = static_cast<std::uint32_t>(want_layers.size());
    ici.ppEnabledLayerNames = want_layers.data();
    ici.enabledExtensionCount = static_cast<std::uint32_t>(want_exts.size());
    ici.ppEnabledExtensionNames = want_exts.data();
    if (vkCreateInstance(&ici, nullptr, &p.instance) != VK_SUCCESS) {
        error = "vkCreateInstance failed (no Vulkan 1.3 loader or driver?)";
        return false;
    }
    if (debug_utils) {
        const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(p.instance, "vkCreateDebugUtilsMessengerEXT"));
        p.destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(p.instance, "vkDestroyDebugUtilsMessengerEXT"));
        VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        mci.pfnUserCallback = on_message;
        if (create) create(p.instance, &mci, nullptr, &p.messenger);
    }
    p.info.validation = validate;

    // The device bbhost would take (host/gpu.cpp choose_device).
    std::uint32_t nd = 0;
    vkEnumeratePhysicalDevices(p.instance, &nd, nullptr);
    std::vector<VkPhysicalDevice> devs(nd);
    vkEnumeratePhysicalDevices(p.instance, &nd, devs.data());
    std::string want = select;
    if (want.empty()) {
        if (const char* e = std::getenv("BBHOST_GPU_DEVICE")) want = e;
    }
    int chosen = -1, best = -1;
    int best_rank = -1;
    VkDeviceSize best_local = 0;
    std::uint32_t family = 0, best_family = 0;
    for (std::uint32_t index = 0; index < devs.size(); ++index) {
        VkPhysicalDeviceProperties dp{};
        vkGetPhysicalDeviceProperties(devs[index], &dp);
        std::uint32_t nx = 0;
        vkEnumerateDeviceExtensionProperties(devs[index], nullptr, &nx, nullptr);
        std::vector<VkExtensionProperties> xs(nx);
        vkEnumerateDeviceExtensionProperties(devs[index], nullptr, &nx, xs.data());
        const bool host_import = std::any_of(xs.begin(), xs.end(), [](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, "VK_EXT_external_memory_host") == 0; });
        if (!host_import || dp.apiVersion < VK_API_VERSION_1_3) continue;
        std::uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[index], &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[index], &nq, qs.data());
        int fam = -1;
        for (std::uint32_t i = 0; i < nq && fam < 0; ++i) {
            if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) fam = static_cast<int>(i);
        }
        if (fam < 0) continue;
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(devs[index], &mp);
        VkDeviceSize local = 0;
        for (std::uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local += mp.memoryHeaps[h].size;
        }
        const int rank = dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 4 : dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3
                         : dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 2 : dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? 1 : 0;
        if (!want.empty() && chosen < 0) {
            char* end = nullptr;
            const long n = std::strtol(want.c_str(), &end, 10);
            std::string lname(dp.deviceName), lwant(want);
            for (char& c : lname) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            for (char& c : lwant) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if ((end && *end == 0 && n == static_cast<long>(index)) || lname.find(lwant) != std::string::npos) {
                chosen = static_cast<int>(index);
                family = static_cast<std::uint32_t>(fam);
            }
        }
        if (rank > best_rank || (rank == best_rank && local > best_local)) {
            best = static_cast<int>(index);
            best_rank = rank;
            best_local = local;
            best_family = static_cast<std::uint32_t>(fam);
        }
    }
    if (chosen < 0) {
        chosen = best;
        family = best_family;
    }
    if (chosen < 0) {
        error = "no Vulkan 1.3 device bbhost could run on";
        return false;
    }
    p.phys = devs[static_cast<std::size_t>(chosen)];
    p.info.index = chosen;

    std::uint32_t nx = 0;
    vkEnumerateDeviceExtensionProperties(p.phys, nullptr, &nx, nullptr);
    std::vector<VkExtensionProperties> xs(nx);
    vkEnumerateDeviceExtensionProperties(p.phys, nullptr, &nx, xs.data());
    const auto has = [&](const char* n) {
        return std::any_of(xs.begin(), xs.end(), [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, n) == 0; });
    };
    const bool ext_m8 = has(VK_KHR_MAINTENANCE_8_EXTENSION_NAME), ext_gpl = has("VK_EXT_graphics_pipeline_library") && has("VK_KHR_pipeline_library");
    const bool ext_eds3 = has("VK_EXT_extended_dynamic_state3"), ext_push = has("VK_KHR_push_descriptor");

    VkPhysicalDeviceVulkan11Properties p11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties p12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT pgpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
    p11.pNext = &p12;
    p12.pNext = &p13;
    if (ext_gpl) p13.pNext = &pgpl;
    VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    pp.pNext = &p11;
    vkGetPhysicalDeviceProperties2(p.phys, &pp);
    VkPhysicalDeviceVulkan12Features q12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features q13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceMaintenance8FeaturesKHR q8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT qgpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT q3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    q12.pNext = &q13;
    void** tail = &q13.pNext;
    if (ext_m8) {
        *tail = &q8;
        tail = &q8.pNext;
    }
    if (ext_gpl) {
        *tail = &qgpl;
        tail = &qgpl.pNext;
    }
    if (ext_eds3) {
        *tail = &q3;
        tail = &q3.pNext;
    }
    VkPhysicalDeviceFeatures2 q2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    q2.pNext = &q12;
    vkGetPhysicalDeviceFeatures2(p.phys, &q2);

    DeviceInfo& d = p.info;
    d.name = pp.properties.deviceName;
    d.vendor = pp.properties.vendorID;
    d.api = pp.properties.apiVersion;
    d.driver = std::string(p12.driverName) + (p12.driverInfo[0] ? std::string(" ") + p12.driverInfo : std::string());
    p.push_size = std::min<std::uint32_t>(pp.properties.limits.maxPushConstantsSize, 256);
    // host/gpu.cpp's switches.
    const bool gpl = ext_gpl && qgpl.graphicsPipelineLibrary && pgpl.graphicsPipelineLibraryFastLinking;
    d.libraries = gpl && !monolithic && !env_off("BBHOST_PIPELINE_LIBRARY") && !env_off("BBHOST_SET_LAYOUTS");
    d.dynamic_depth_clamp = gpl && ext_eds3 && q3.extendedDynamicState3DepthClampEnable && !env_off("BBHOST_DYNAMIC_DEPTH_CLAMP");
    d.runtime_offsets = ext_m8 && q8.maintenance8 && !env_off("BBHOST_RUNTIME_OFFSETS");
    d.native_half_rtz = q12.shaderFloat16 && p12.shaderRoundingModeRTZFloat16 && p12.shaderDenormPreserveFloat16 && !env_off("BBHOST_HALF_NATIVE");
    d.depth_bounds = q2.features.depthBounds == VK_TRUE;
    d.subgroup_size = p11.subgroupSize;
    d.subgroup_min = p13.minSubgroupSize;
    d.subgroup_max = p13.maxSubgroupSize;
    d.fragment_sizes = (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_FRAGMENT_BIT) != 0;
    {
        const char* e = std::getenv("BBHOST_PS_WAVE32");
        const bool driver_choice = !(e && (std::strcmp(e, "32") == 0 || std::strcmp(e, "1") == 0));
        if (e && e[0] == '0') d.ps_wave = 0;
        else if (p11.subgroupSize <= 32) d.ps_wave = 0;
        else if (!q13.subgroupSizeControl) d.ps_wave = 0;
        else if (p13.minSubgroupSize > 32 || p13.maxSubgroupSize < 32) d.ps_wave = 0;
        else if (driver_choice && p13.minSubgroupSize == 32) d.ps_wave = 1;
        else if (!d.fragment_sizes) d.ps_wave = 0;
        else d.ps_wave = 32;
    }
    gcn::set_runtime_sample_offsets(d.runtime_offsets);
    gcn::set_native_half_rtz(d.native_half_rtz);

    // The features bbhost turns on that a compile can depend on, where the device has them.
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    const VkPhysicalDeviceFeatures& have = q2.features;
    VkPhysicalDeviceFeatures& f = f2.features;
    f.shaderInt64 = have.shaderInt64;
    f.shaderInt16 = have.shaderInt16;
    f.shaderStorageImageReadWithoutFormat = have.shaderStorageImageReadWithoutFormat;
    f.shaderStorageImageWriteWithoutFormat = have.shaderStorageImageWriteWithoutFormat;
    f.geometryShader = have.geometryShader;
    f.tessellationShader = have.tessellationShader;
    f.independentBlend = have.independentBlend;
    f.fillModeNonSolid = have.fillModeNonSolid;
    f.depthClamp = have.depthClamp;
    f.depthBiasClamp = have.depthBiasClamp;
    f.depthBounds = have.depthBounds;
    f.dualSrcBlend = have.dualSrcBlend;
    f.shaderClipDistance = have.shaderClipDistance;
    f.shaderCullDistance = have.shaderCullDistance;
    f.samplerAnisotropy = have.samplerAnisotropy;
    f.imageCubeArray = have.imageCubeArray;
    f.fragmentStoresAndAtomics = have.fragmentStoresAndAtomics;
    f.vertexPipelineStoresAndAtomics = have.vertexPipelineStoresAndAtomics;
    f.shaderImageGatherExtended = have.shaderImageGatherExtended;
    if (const char* e = std::getenv("BBHOST_ROBUST"); e && e[0] == '1') f.robustBufferAccess = have.robustBufferAccess;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = q12.bufferDeviceAddress;
    f12.shaderFloat16 = q12.shaderFloat16;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    if (!q13.dynamicRendering) {
        error = d.name + ": no dynamic rendering (bbhost needs Vulkan 1.3's)";
        return false;
    }
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = q13.synchronization2;
    f13.subgroupSizeControl = d.ps_wave ? VK_TRUE : VK_FALSE;
    f13.pipelineCreationCacheControl = q13.pipelineCreationCacheControl;
    VkPhysicalDeviceMaintenance8FeaturesKHR fm8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
    fm8.maintenance8 = VK_TRUE;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT fgpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    fgpl.graphicsPipelineLibrary = VK_TRUE;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT feds3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    feds3.extendedDynamicState3DepthClampEnable = VK_TRUE;
    std::vector<const char*> dext;
    f2.pNext = &f12;
    f12.pNext = &f13;
    void** ftail = &f13.pNext;
    if (ext_m8 && q8.maintenance8) {
        dext.push_back(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
        *ftail = &fm8;
        ftail = &fm8.pNext;
    }
    if (gpl) {
        dext.push_back("VK_KHR_pipeline_library");
        dext.push_back("VK_EXT_graphics_pipeline_library");
        *ftail = &fgpl;
        ftail = &fgpl.pNext;
    }
    if (d.dynamic_depth_clamp) {
        dext.push_back("VK_EXT_extended_dynamic_state3");
        *ftail = &feds3;
        ftail = &feds3.pNext;
    }
    if (ext_push) dext.push_back("VK_KHR_push_descriptor");
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
    if (vkCreateDevice(p.phys, &dci, nullptr, &p.device) != VK_SUCCESS) {
        error = d.name + ": vkCreateDevice failed";
        return false;
    }
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(p.phys, VK_FORMAT_D32_SFLOAT_S8_UINT, &fp);
    p.has_stencil = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
    p.depth_stencil = p.has_stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
    return true;
}

Verdict Prober::pixel(const std::vector<std::uint32_t>& spirv, std::uint32_t wave64_needs) {
    Impl& p = *impl_;
    return p.run([&](Impl::Scope& s, Impl::Plan& plan, Bindings& b, bool& push) {
        const Module m = parse(spirv);
        if (m.model != kModelFragment) throw Unplaced("not a pixel shader");
        add_bindings(m, b, push);
        plan.pre.push_back({VK_SHADER_STAGE_VERTEX_BIT, s.module(p.helper(vertex_shader_for(m), "vs_6_0")), false});
        plan.fragment = s.module(spirv);
        plan.fragment_ours = true;
        plan.wave64_needs = wave64_needs;
        plan.out = outputs_of(&m, p.depth_stencil, p.has_stencil);
    });
}

Verdict Prober::vertex(const std::vector<std::uint32_t>& spirv) {
    Impl& p = *impl_;
    return p.run([&](Impl::Scope& s, Impl::Plan& plan, Bindings& b, bool& push) {
        const Module m = parse(spirv);
        if (m.model != kModelVertex) throw Unplaced("not a vertex shader");
        add_bindings(m, b, push);
        plan.pre.push_back({VK_SHADER_STAGE_VERTEX_BIT, s.module(spirv), true});
        plan.vi = vertex_input_of(m);
        plan.out = outputs_of(nullptr, p.depth_stencil, p.has_stencil);
    });
}

Verdict Prober::domain(const std::vector<std::uint32_t>& spirv, const std::vector<std::uint32_t>& modes) {
    Impl& p = *impl_;
    return p.run([&](Impl::Scope& s, Impl::Plan& plan, Bindings& b, bool& push) {
        const Module m = parse(spirv);
        if (m.model != kModelTessEval) throw Unplaced("not a domain shader");
        const Module mm = parse(modes.empty() ? spirv : modes);
        add_bindings(m, b, push);
        const HullFor h = hull_for(m, mm);
        plan.pre.push_back({VK_SHADER_STAGE_VERTEX_BIT, s.module(p.helper(h.vs, "vs_6_0")), false});
        plan.pre.push_back({VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, s.module(p.helper(h.hs, "hs_6_0")), false});
        plan.pre.push_back({VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, s.module(spirv), true});
        plan.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        plan.patch_points = h.control_points;
        plan.out = outputs_of(nullptr, p.depth_stencil, p.has_stencil);
    });
}

Verdict Prober::hull_set(const std::vector<std::uint32_t>& ls, const std::vector<std::uint32_t>& hs, const std::vector<std::uint32_t>& ds,
                         std::uint32_t control_points) {
    Impl& p = *impl_;
    return p.run([&](Impl::Scope& s, Impl::Plan& plan, Bindings& b, bool& push) {
        const Module ml = parse(ls), mh = parse(hs), md = parse(ds);
        if (ml.model != kModelVertex || mh.model != kModelTessControl || md.model != kModelTessEval) throw Unplaced("not an LS, a hull and a domain shader");
        add_bindings(ml, b, push);
        add_bindings(mh, b, push);
        add_bindings(md, b, push);
        plan.pre.push_back({VK_SHADER_STAGE_VERTEX_BIT, s.module(ls), true});
        plan.pre.push_back({VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, s.module(hs), true});
        plan.pre.push_back({VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, s.module(ds), true});
        plan.vi = vertex_input_of(ml);
        plan.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        plan.patch_points = std::max(1u, control_points);
        plan.out = outputs_of(nullptr, p.depth_stencil, p.has_stencil);
    });
}

Verdict Prober::compute(const std::vector<std::uint32_t>& spirv) {
    Impl& p = *impl_;
    Verdict v;
    try {
        Impl::Scope s(p, v);
        const Module m = parse(spirv);
        if (m.model != kModelCompute) throw Unplaced("not a compute shader");
        Bindings b;
        bool push = false;
        add_bindings(m, b, push);
        s.make_layout(b, push);
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = s.module(spirv);
        ci.stage.pName = "main";
        ci.layout = s.layout;
        std::string messages, errors;
        t_messages = &messages;
        t_errors = &errors;
        VkPipeline pipe = VK_NULL_HANDLE;
        const VkResult r = vkCreateComputePipelines(p.device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe);
        t_messages = t_errors = nullptr;
        ++v.created;
        s.finish(r, pipe, messages, errors, "compute pipeline", true, false);
    } catch (const Unplaced& u) {
        v.ok = false;
        v.skipped = true;
        v.what = u.what();
    }
    return v;
}

}  // namespace probe
