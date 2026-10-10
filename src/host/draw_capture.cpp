// BBHOST_CAPTURE_DRAW: see draw_capture.h.
//
// A capture is a directory holding manifest.json and blobs/*.bin. The manifest
// names every blob with its size and SHA-256; tools/drawreplay refuses a
// capture whose blobs do not match. Nothing here changes what the live draw
// renders: the readbacks copy out of its images before and after it.
#include "host/draw_capture.h"

#include "core/sha256.h"
#include "gcn/isa.h"
#include "hle/modules.h"
#include "log.h"
#include "replay/compare.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <system_error>

namespace gpu {

namespace {

constexpr std::uint64_t kPage = 1ull << gcn::kPageShift;
constexpr std::size_t kMaxPages = 512;                  // 32 MiB of guest pages per capture
constexpr std::uint64_t kMaxVsharpBytes = 16ull << 20;  // one vertex buffer
constexpr std::uint64_t kMaxVertexBytes = 64ull << 20;  // one vertex-input binding

// Bytes of one element of a buffer data format (render.cpp vertex_format_bytes).
std::uint32_t vertex_element_bytes(std::uint32_t dfmt) {
    static const std::uint8_t kBytes[16] = {0, 1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 8, 8, 12, 16, 0};
    return kBytes[dfmt & 0xf];
}
constexpr int kMaxRejections = 64;

struct Config {
    bool enabled = false;
    std::string prefix;
    std::uint32_t min_count = 0;
    std::uint64_t min_flip = 0;
    std::string dir = "tmp/captures";
    int count = 1;
    int every = 1;  // capture one of every n matching draws (distinct light instances)
    bool each = false;  // '*': the first draw of each pipeline whose no-fallback variant is ready (tools/lift_verify.py)
    bool each_ps = false;  // '*ps': the same, one draw for each pixel shader
    bool armed = false;    // BBHOST_CAPTURE_ARMED=1: only while armed (host_gpu_capture_arm)
};

const Config& config() {
    static const Config c = [] {
        Config c;
        const char* e = std::getenv("BBHOST_CAPTURE_DRAW");
        if (!e || !*e) return c;
        std::string s(e);
        if (const std::size_t pc = s.find('%'); pc != std::string::npos) {
            c.min_count = static_cast<std::uint32_t>(std::strtoul(s.c_str() + pc + 1, nullptr, 10));
            s.resize(pc);
        }
        c.prefix = s;
        c.each_ps = s == "*ps";
        c.each = s == "*" || c.each_ps;
        c.enabled = !s.empty();
        if (const char* a = std::getenv("BBHOST_CAPTURE_ARMED")) c.armed = a[0] == '1';
        if (const char* f = std::getenv("BBHOST_CAPTURE_MIN_FLIP")) c.min_flip = std::strtoull(f, nullptr, 10);
        if (const char* d = std::getenv("BBHOST_CAPTURE_DIR"); d && *d) c.dir = d;
        if (const char* n = std::getenv("BBHOST_CAPTURE_COUNT")) c.count = std::max(1, std::atoi(n));
        if (const char* n = std::getenv("BBHOST_CAPTURE_EVERY")) c.every = std::max(1, std::atoi(n));
        return c;
    }();
    return c;
}

int g_captured = 0;  // under g.mu
std::set<std::string> g_each_seen;  // BBHOST_CAPTURE_DRAW=*: pipelines (*ps: pixel shaders) already captured (under g.mu)
int g_rejected = 0;
// BBHOST_CAPTURE_ARMED=1: the draws still to take, and where (under g.mu).
int g_armed = 0;
std::string g_armed_dir;
std::uint64_t g_matches = 0;  // matching draws seen, for BBHOST_CAPTURE_EVERY

struct Readback {
    std::string blob;
    DevBuffer staging;
    std::uint64_t bytes = 0;
};

json::Value hex_words(const std::uint32_t* w, std::size_t n) {
    json::Value a = json::Value::make_array();
    for (std::size_t i = 0; i < n; ++i) a.push(json::hex(w[i]));
    return a;
}

json::Value image_info_json(const VkImageCreateInfo& i) {
    json::Value v = json::Value::make_object();
    v.set("flags", i.flags);
    v.set("type", static_cast<unsigned>(i.imageType));
    v.set("format", static_cast<unsigned>(i.format));
    json::Value ext = json::Value::make_array();
    ext.push(i.extent.width);
    ext.push(i.extent.height);
    ext.push(i.extent.depth);
    v.set("extent", ext);
    v.set("mip_levels", i.mipLevels);
    v.set("array_layers", i.arrayLayers);
    v.set("samples", static_cast<unsigned>(i.samples));
    v.set("usage", i.usage);
    return v;
}

json::Value view_info_json(const VkImageViewCreateInfo& vi) {
    json::Value v = json::Value::make_object();
    v.set("type", static_cast<unsigned>(vi.viewType));
    v.set("format", static_cast<unsigned>(vi.format));
    json::Value comp = json::Value::make_array();
    comp.push(static_cast<unsigned>(vi.components.r));
    comp.push(static_cast<unsigned>(vi.components.g));
    comp.push(static_cast<unsigned>(vi.components.b));
    comp.push(static_cast<unsigned>(vi.components.a));
    v.set("components", comp);
    v.set("aspect", vi.subresourceRange.aspectMask);
    v.set("base_level", vi.subresourceRange.baseMipLevel);
    v.set("levels", vi.subresourceRange.levelCount);
    v.set("base_layer", vi.subresourceRange.baseArrayLayer);
    v.set("layers", vi.subresourceRange.layerCount);
    return v;
}

json::Value sampler_json(const VkSamplerCreateInfo& s) {
    json::Value v = json::Value::make_object();
    v.set("flags", s.flags);
    v.set("mag_filter", static_cast<unsigned>(s.magFilter));
    v.set("min_filter", static_cast<unsigned>(s.minFilter));
    v.set("mipmap_mode", static_cast<unsigned>(s.mipmapMode));
    v.set("address_u", static_cast<unsigned>(s.addressModeU));
    v.set("address_v", static_cast<unsigned>(s.addressModeV));
    v.set("address_w", static_cast<unsigned>(s.addressModeW));
    v.set("mip_lod_bias", json::f32(s.mipLodBias));
    v.set("anisotropy_enable", s.anisotropyEnable == VK_TRUE);
    v.set("max_anisotropy", json::f32(s.maxAnisotropy));
    v.set("compare_enable", s.compareEnable == VK_TRUE);
    v.set("compare_op", static_cast<unsigned>(s.compareOp));
    v.set("min_lod", json::f32(s.minLod));
    v.set("max_lod", json::f32(s.maxLod));
    v.set("border_color", static_cast<unsigned>(s.borderColor));
    v.set("unnormalized_coordinates", s.unnormalizedCoordinates == VK_TRUE);
    return v;
}

json::Value stencil_json(const VkStencilOpState& s) {
    json::Value v = json::Value::make_object();
    v.set("fail_op", static_cast<unsigned>(s.failOp));
    v.set("pass_op", static_cast<unsigned>(s.passOp));
    v.set("depth_fail_op", static_cast<unsigned>(s.depthFailOp));
    v.set("compare_op", static_cast<unsigned>(s.compareOp));
    return v;
}

json::Value fixed_json(const GfxFixedState& f) {
    json::Value v = json::Value::make_object();
    v.set("topology", static_cast<unsigned>(f.topology));
    v.set("polygon_mode", static_cast<unsigned>(f.polygon_mode));
    v.set("cull_mode", f.cull_mode);
    v.set("front_face", static_cast<unsigned>(f.front_face));
    v.set("depth_clamp", f.depth_clamp);
    v.set("depth_test", f.depth_test);
    v.set("depth_write", f.depth_write);
    v.set("depth_compare", static_cast<unsigned>(f.depth_compare));
    v.set("depth_bounds_test", f.depth_bounds_test);
    v.set("stencil_test", f.stencil_test);
    v.set("stencil_front", stencil_json(f.front));
    v.set("stencil_back", stencil_json(f.back));
    json::Value atts = json::Value::make_array();
    for (std::size_t a = 0; a < f.blends.size(); ++a) {
        const VkPipelineColorBlendAttachmentState& b = f.blends[a];
        json::Value o = json::Value::make_object();
        o.set("slot", f.color_slots[a]);
        o.set("format", static_cast<unsigned>(f.color_formats[a]));
        o.set("blend_enable", b.blendEnable == VK_TRUE);
        o.set("src_color", static_cast<unsigned>(b.srcColorBlendFactor));
        o.set("dst_color", static_cast<unsigned>(b.dstColorBlendFactor));
        o.set("color_op", static_cast<unsigned>(b.colorBlendOp));
        o.set("src_alpha", static_cast<unsigned>(b.srcAlphaBlendFactor));
        o.set("dst_alpha", static_cast<unsigned>(b.dstAlphaBlendFactor));
        o.set("alpha_op", static_cast<unsigned>(b.alphaBlendOp));
        o.set("write_mask", b.colorWriteMask);
        atts.push(o);
    }
    v.set("attachments", atts);
    v.set("depth_format", static_cast<unsigned>(f.depth_format));
    v.set("stencil_format", static_cast<unsigned>(f.stencil_format));
    return v;
}

bool op_writes(VkStencilOp op) { return op != VK_STENCIL_OP_KEEP; }

// Guest memory a stage's program can walk through the page table: the tables
// its s_load_* instructions read from user-data pointers and the vertex
// buffers (V#s) found there or in user data. The replay's sink-sentinel run
// is what proves the set complete for a draw; this only has to cover it.
void program_ranges(const std::vector<std::uint32_t>* words, const std::uint32_t* user,
                    std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
    if (!words || words->empty()) return;
    const gcn::Program p = gcn::decode(words->data(), words->size());
    const auto vsharp = [&](const std::uint32_t* w) {
        const std::uint64_t base = static_cast<std::uint64_t>(w[0]) | (static_cast<std::uint64_t>(w[1] & 0xff) << 32);
        const std::uint64_t bytes = static_cast<std::uint64_t>((w[1] >> 16) & 0x3fff) * w[2];
        if (base && bytes && bytes <= kMaxVsharpBytes && hle_kernel_va_mapped(base, 16)) ranges.push_back({base, bytes});
    };
    for (const gcn::Inst& in : p.insts) {
        if (in.enc == gcn::Enc::SMRD && in.src0 + 1 < 16) {
            const std::uint64_t hi = in.op < 8 ? user[in.src0 + 1] : (user[in.src0 + 1] & 0xff);
            const std::uint64_t at = (static_cast<std::uint64_t>(user[in.src0]) | (hi << 32)) +
                                     static_cast<std::uint64_t>(static_cast<std::int64_t>(in.imm) * 4);
            if (!hle_kernel_va_mapped(at, 16)) continue;
            ranges.push_back({at, 64});
            if (in.op < 8) {
                std::uint32_t w[4] = {};
                read_guest_locked(at, sizeof(w), w);
                vsharp(w);
            }
        } else if (in.enc == gcn::Enc::MUBUF && in.srsrc + 3 < 16) {
            vsharp(user + in.srsrc);
        }
    }
}

}  // namespace

class CaptureSession {
public:
    CaptureDraw draw;
    std::string dir;
    json::Value manifest = json::Value::make_object();
    std::vector<Readback> readbacks;
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> host_blobs;
    bool done = false;
    bool quiet = false;  // rejected before anything worth reporting

    void blob(const std::string& name, const std::uint8_t* data, std::size_t bytes) {
        host_blobs.emplace_back(name, std::vector<std::uint8_t>(data, data + bytes));
    }

    // Records a copy of one aspect of the given levels and layers into a
    // host-visible buffer, read after the next flush. Subresources are laid
    // out level-major, then layer, as `subresources` lists them.
    bool readback(const std::string& name, VkImage image, VkImageAspectFlags aspect, VkFormat format, VkImageType type,
                  VkExtent3D extent, std::uint32_t base_level, std::uint32_t levels, std::uint32_t base_layer,
                  std::uint32_t layers, json::Value& subresources, std::string& why) {
        std::vector<VkBufferImageCopy> regions;
        std::uint64_t total = 0;
        for (std::uint32_t l = base_level; l < base_level + levels; ++l) {
            const std::uint32_t w = std::max(1u, extent.width >> l);
            const std::uint32_t h = type == VK_IMAGE_TYPE_1D ? 1u : std::max(1u, extent.height >> l);
            const std::uint32_t d = type == VK_IMAGE_TYPE_3D ? std::max(1u, extent.depth >> l) : 1u;
            const std::uint64_t bytes = replay::subresource_bytes(format, aspect, w, h, d);
            if (!bytes) {
                why = "no texel layout for format " + std::to_string(format) + " aspect " + std::to_string(aspect);
                return false;
            }
            for (std::uint32_t k = base_layer; k < base_layer + layers; ++k) {
                VkBufferImageCopy r{};
                r.bufferOffset = total;
                r.imageSubresource = {aspect, l, k, 1};
                r.imageExtent = {w, h, d};
                regions.push_back(r);
                json::Value sj = json::Value::make_object();
                sj.set("level", l);
                sj.set("layer", k);
                json::Value ext = json::Value::make_array();
                ext.push(w);
                ext.push(h);
                ext.push(d);
                sj.set("extent", ext);
                sj.set("offset", total);
                sj.set("bytes", bytes);
                subresources.push(sj);
                total += bytes;
            }
        }
        Readback rb;
        rb.blob = name;
        rb.bytes = total;
        if (!create_dev_buffer(rb.staging, total, true)) {
            why = "no host-visible staging buffer of " + std::to_string(total) + " bytes";
            return false;
        }
        readbacks.push_back(rb);
        begin_recording_locked();
        transfer_flush_locked();
        render_end_pass_locked();
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(g_cmd(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
        vkCmdCopyImageToBuffer(g_cmd(), image, VK_IMAGE_LAYOUT_GENERAL, rb.staging.buffer, static_cast<std::uint32_t>(regions.size()),
                               regions.data());
        return true;
    }

    bool target_readbacks(const char* when, json::Value& targets, std::string& why) {
        const CaptureDraw& d = draw;
        json::Value& colors = const_cast<json::Value&>(json::member(targets, "color"));
        std::size_t n = 0;  // the targets array lists bound slots only (a gap has no image)
        for (std::size_t a = 0; a < d.fixed.color_slots.size(); ++a) {
            const RtImage* rt = d.color[d.fixed.color_slots[a]];
            if (!rt) continue;
            const std::string name = "color" + std::to_string(d.fixed.color_slots[a]) + "-" + when;
            json::Value subs = json::Value::make_array();
            if (!readback(name, rt->image, VK_IMAGE_ASPECT_COLOR_BIT, rt->format, VK_IMAGE_TYPE_2D, {rt->width, rt->height, 1}, 0, 1,
                          d.color_layer[d.fixed.color_slots[a]], 1, subs, why)) {
                return false;
            }
            colors.array[n++].set(when, name);
        }
        if (d.depth) {
            json::Value& depth = const_cast<json::Value&>(json::member(targets, "depth"));
            json::Value subs = json::Value::make_array();
            const std::string name = std::string("depth-") + when;
            if (!readback(name, d.depth->image, VK_IMAGE_ASPECT_DEPTH_BIT, d.depth->format, VK_IMAGE_TYPE_2D,
                          {d.depth->width, d.depth->height, 1}, 0, 1, d.depth_layer, 1, subs, why)) {
                return false;
            }
            depth.set(std::string("depth_") + when, name);
            depth.set("layer", d.depth_layer);
            depth.set("layers", d.depth->layers);
            if (d.fixed.stencil_format != VK_FORMAT_UNDEFINED) {
                const std::string sname = std::string("stencil-") + when;
                if (!readback(sname, d.depth->image, VK_IMAGE_ASPECT_STENCIL_BIT, d.depth->format, VK_IMAGE_TYPE_2D,
                              {d.depth->width, d.depth->height, 1}, 0, 1, d.depth_layer, 1, subs, why)) {
                    return false;
                }
                depth.set(std::string("stencil_") + when, sname);
            }
        }
        return true;
    }
};

void CaptureSessionDeleter::operator()(CaptureSession* s) const {
    if (!s->done) {
        if (!s->quiet) {
            host_log("capture: %s draw %llu abandoned before it was written", s->draw.pipeline.c_str(),
                     static_cast<unsigned long long>(s->draw.draw_index));
        }
        // The copies may still be queued: free the buffers with that submission.
        for (Readback& r : s->readbacks) defer_destroy(r.staging);
    }
    delete s;
}

bool capture_candidate(const std::string& pipeline, std::uint32_t count, std::uint64_t flip, bool lean_ready) {
    const Config& c = config();
    if (!(c.enabled && g_captured < c.count && g_rejected < kMaxRejections && flip >= c.min_flip && count >= c.min_count)) {
        return false;
    }
    if (c.armed && g_armed <= 0) return false;
    if (c.each) {
        // A pipeline is named <vertex shader>+<pixel shader>.
        const std::size_t plus = pipeline.rfind('+');
        return lean_ready && g_each_seen.insert(c.each_ps && plus != std::string::npos ? pipeline.substr(plus + 1) : pipeline).second;
    }
    // A comma-separated list captures any of them: two passes of one surface
    // in the same frame, which is how their inputs can be compared.
    bool any = false;
    for (std::size_t at = 0; at <= c.prefix.size() && !any;) {
        const std::size_t comma = std::min(c.prefix.find(',', at), c.prefix.size());
        const std::size_t n = comma - at;
        if (n && pipeline.compare(0, n, c.prefix, at, n) == 0) any = true;
        at = comma + 1;
    }
    if (!any) return false;
    return g_matches++ % static_cast<std::uint64_t>(c.every) == 0;
}

void capture_settle_locked() { flush_locked(); }

CapturePtr capture_begin_locked(CaptureDraw&& in) {
    const Config& cfg = config();
    CapturePtr session(new CaptureSession);
    CaptureSession& s = *session;
    s.draw = std::move(in);
    CaptureDraw& d = s.draw;
    const auto reject = [&](const std::string& why) {
        s.quiet = true;
        ++g_rejected;
        static std::set<std::string> seen;
        if (seen.size() < 32 && seen.insert(why).second) {
            host_log("capture: %s draw %llu (flip %llu) not captured: %s", d.pipeline.c_str(),
                     static_cast<unsigned long long>(d.draw_index), static_cast<unsigned long long>(d.flip), why.c_str());
        }
        if (g_rejected == kMaxRejections) host_log("capture: %d candidate draws rejected; no more attempts", kMaxRejections);
        return CapturePtr();
    };

    // What the first supported draws must not do.
    if (d.indirect_va) return reject("indirect draw: its arguments are GPU data");
    for (int t = 0; t < 8; ++t) {
        if (d.color[t] && !d.color[t]->initialised) return reject("colour target " + std::to_string(t) + " has undefined contents");
    }
    if (d.depth && !d.depth->initialised) return reject("depth target has undefined contents");
    const bool depth_stencil_writes =
        d.fixed.depth_write || (d.fixed.stencil_test && (op_writes(d.fixed.front.failOp) || op_writes(d.fixed.front.passOp) ||
                                                         op_writes(d.fixed.front.depthFailOp) || op_writes(d.fixed.back.failOp) ||
                                                         op_writes(d.fixed.back.passOp) || op_writes(d.fixed.back.depthFailOp)));

    namespace fs = std::filesystem;
    char leaf[160];
    std::snprintf(leaf, sizeof(leaf), "%s-f%llu-d%llu", d.pipeline.c_str(), static_cast<unsigned long long>(d.flip),
                  static_cast<unsigned long long>(d.draw_index));
    s.dir = (cfg.armed && !g_armed_dir.empty() ? g_armed_dir : cfg.dir) + "/" + leaf;
    std::error_code ec;
    if (fs::exists(s.dir, ec)) return reject("capture directory " + s.dir + " already exists");
    if (!fs::create_directories(fs::path(s.dir) / "blobs", ec)) return reject("cannot create " + s.dir + ": " + ec.message());

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::map<VkImageView, std::string> image_ids;
    std::map<VkSampler, std::string> sampler_ids;
    json::Value images = json::Value::make_object();
    json::Value samplers = json::Value::make_object();
    json::Value stages = json::Value::make_array();
    static const char* const kStageName[2] = {"vertex", "pixel"};
    for (int st = 0; st < 2; ++st) {
        CaptureStage& cs = d.stages[st];
        const std::string tag = st ? "ps" : "vs";
        json::Value sj = json::Value::make_object();
        sj.set("stage", kStageName[st]);
        sj.set("present", cs.spirv != nullptr);
        if (!cs.spirv) {
            stages.push(sj);
            continue;
        }
        if (!cs.meta || !cs.images || cs.images->images.size() != cs.meta->images.size() ||
            cs.images->samplers.size() != cs.meta->samplers.size() || cs.buffers.size() != cs.meta->buffers.size()) {
            return reject(std::string(kStageName[st]) + " bindings and resolved resources disagree");
        }
        const gcn::TranslateResult& meta = *cs.meta;
        const std::uint32_t* user = cs.params.user_sgpr;
        s.blob(tag + "-spirv", reinterpret_cast<const std::uint8_t*>(cs.spirv->data()), cs.spirv->size() * 4);
        sj.set("spirv", tag + "-spirv");
        if (cs.gcn_words && !cs.gcn_words->empty()) {
            s.blob(tag + "-gcn", reinterpret_cast<const std::uint8_t*>(cs.gcn_words->data()), cs.gcn_words->size() * 4);
            sj.set("gcn", tag + "-gcn");
        }
        s.blob(tag + "-params", reinterpret_cast<const std::uint8_t*>(&cs.params), sizeof(cs.params));
        json::Value pj = json::Value::make_object();
        pj.set("blob", tag + "-params");
        pj.set("bytes", static_cast<unsigned>(sizeof(cs.params)));
        pj.set("user_sgpr", hex_words(user, 16));
        pj.set("cb_valid", json::hex(cs.params.cb_valid));
        json::Value bias = json::Value::make_array();
        for (std::uint32_t b : cs.params.cb_bias_dw) bias.push(b);
        pj.set("cb_bias_dw", bias);
        pj.set("l1_table_replaced_on_replay", json::hex(cs.params.l1_table));
        sj.set("params", pj);

        json::Value ij = json::Value::make_array();
        for (std::size_t k = 0; k < meta.images.size(); ++k) {
            const gcn::ImageBinding& b = meta.images[k];
            const StageImages::Image& im = cs.images->images[k];
            const std::string where = std::string(kStageName[st]) + " image binding " + std::to_string(b.binding) + " (" + b.path.str() + ")";
            if (b.storage) return reject(where + " is a storage image");
            if (!im.resolved || !im.view || im.dim != b.dim || im.arrayed != b.arrayed) return reject(where + " is bound to a dummy image");
            ViewRecord rec;
            if (!describe_view_locked(im.view, rec)) return reject(where + " has a view the texture cache did not record");
            std::uint32_t words[8] = {};
            resolve_resource(b.path, user, b.r128 ? 4 : 8, words, &ranges);
            std::string id;
            if (auto found = image_ids.find(im.view); found != image_ids.end()) {
                id = found->second;
            } else {
                id = "image" + std::to_string(image_ids.size());
                image_ids[im.view] = id;
                json::Value rj = json::Value::make_object();
                rj.set("kind", rec.render_target ? "render-target" : "surface");
                rj.set("guest_base", json::hex(rec.guest_base));
                rj.set("image", image_info_json(rec.image_info));
                rj.set("view", view_info_json(rec.view_info));
                for (int t = 0; t < 8; ++t) {
                    if (d.color[t] && d.color[t]->image == rec.image) return reject(where + " samples colour target " + std::to_string(t) + " while the draw writes it");
                }
                if (d.depth && d.depth->image == rec.image) {
                    if (depth_stencil_writes) return reject(where + " samples the depth target while the draw can write depth or stencil");
                    rj.set("alias", "depth");
                } else {
                    rj.set("alias", nullptr);
                    json::Value subs = json::Value::make_array();
                    std::string why;
                    const VkImageSubresourceRange& r = rec.view_info.subresourceRange;
                    if (!s.readback(id, rec.image, r.aspectMask, rec.image_info.format, rec.image_info.imageType, rec.image_info.extent,
                                    r.baseMipLevel, r.levelCount, r.baseArrayLayer, r.layerCount, subs, why)) {
                        return reject(where + ": " + why);
                    }
                    rj.set("blob", id);
                    rj.set("subresources", subs);
                }
                images.set(id, rj);
            }
            json::Value bj = json::Value::make_object();
            bj.set("binding", b.binding);
            bj.set("path", b.path.str());
            bj.set("dim", b.dim);
            bj.set("arrayed", b.arrayed);
            bj.set("r128", b.r128);
            bj.set("da", b.da);
            bj.set("depth_compare", b.depth);
            bj.set("cube", b.cube);
            bj.set("tsharp", hex_words(im.w, 8));
            bj.set("image", id);
            ij.push(bj);
        }
        sj.set("images", ij);

        json::Value smj = json::Value::make_array();
        for (std::size_t k = 0; k < meta.samplers.size(); ++k) {
            const gcn::SamplerBinding& b = meta.samplers[k];
            const std::string where = std::string(kStageName[st]) + " sampler binding " + std::to_string(b.binding) + " (" + b.path.str() + ")";
            const VkSampler smp = cs.images->samplers[k];
            if (!smp) return reject(where + " is bound to the dummy sampler");
            VkSamplerCreateInfo sci{};
            if (!describe_sampler_locked(smp, sci)) return reject(where + " has a sampler the texture cache did not record");
            std::uint32_t words[4] = {};
            resolve_resource(b.path, user, 4, words, &ranges);
            std::string id;
            if (auto found = sampler_ids.find(smp); found != sampler_ids.end()) {
                id = found->second;
            } else {
                id = "sampler" + std::to_string(sampler_ids.size());
                sampler_ids[smp] = id;
                samplers.set(id, sampler_json(sci));
            }
            json::Value bj = json::Value::make_object();
            bj.set("binding", b.binding);
            bj.set("path", b.path.str());
            bj.set("compare", b.compare);
            bj.set("ssharp", hex_words(words, 4));
            bj.set("sampler", id);
            smj.push(bj);
        }
        sj.set("samplers", smj);

        json::Value bufj = json::Value::make_array();
        for (std::size_t i = 0; i < meta.buffers.size(); ++i) {
            const gcn::BufferBinding& b = meta.buffers[i];
            const VkDescriptorBufferInfo& info = cs.buffers[i];
            const std::string where = std::string(kStageName[st]) + " buffer binding " + std::to_string(b.binding) + " (" + b.path.str() + ")";
            if (!((cs.params.cb_valid >> i) & 1u) || info.buffer == g.dummy_ssbo || info.range == VK_WHOLE_SIZE) {
                return reject(where + " is not bound; the shader would walk the page table");
            }
            std::uint32_t words[2] = {};
            resolve_resource(b.path, user, 2, words, &ranges);
            const std::string name = tag + "-buffer" + std::to_string(i);
            json::Value bj = json::Value::make_object();
            bj.set("binding", b.binding);
            bj.set("path", b.path.str());
            bj.set("pointer", b.pointer);
            bj.set("max_dw", b.max_dw);
            bj.set("bias_dw", cs.params.cb_bias_dw[i]);
            bj.set("bytes", info.range);
            if (info.buffer == g.sink.buffer) {
                // A null table: the page table's sink page, never written.
                bj.set("kind", "sink");
                const std::vector<std::uint8_t> zeros(info.range, 0);
                s.blob(name, zeros.data(), zeros.size());
            } else {
                std::uint64_t va = 0;
                const std::uint8_t* bytes = imported_bytes_locked(info.buffer, info.offset, info.range, &va);
                if (!bytes) return reject(where + " is not over imported guest memory");
                bj.set("kind", "guest");
                bj.set("guest_va", json::hex(va));
                s.blob(name, bytes, info.range);
                if (va) ranges.push_back({va, info.range});
            }
            bj.set("blob", name);
            bufj.push(bj);
        }
        sj.set("buffers", bufj);
        program_ranges(cs.gcn_words, user, ranges);
        if (st == 0) program_ranges(d.fetch_words, user, ranges);
        stages.push(sj);
    }

    // Guest pages in the GPU page table that the shaders can reach.
    std::set<std::uint64_t> pages;
    for (const auto& [va, bytes] : ranges) {
        if (!bytes || va + bytes < va) continue;
        for (std::uint64_t p = va & ~(kPage - 1); p < va + bytes; p += kPage) {
            if (!page_in_gpu_table_locked(p) || !hle_kernel_va_mapped(p, kPage)) continue;
            pages.insert(p);
            if (pages.size() > kMaxPages) return reject("shader-reachable guest memory exceeds " + std::to_string(kMaxPages) + " pages");
        }
    }
    json::Value memory = json::Value::make_object();
    memory.set("page_bytes", kPage);
    json::Value page_list = json::Value::make_array();
    std::vector<std::uint8_t> page_bytes(pages.size() * kPage);
    std::size_t at = 0;
    for (std::uint64_t p : pages) {
        page_list.push(json::hex(p));
        read_guest_locked(p, kPage, page_bytes.data() + at);
        at += kPage;
    }
    memory.set("pages", page_list);
    memory.set("blob", "memory");
    s.blob("memory", page_bytes.data(), page_bytes.size());

    json::Value draw = json::Value::make_object();
    draw.set("vertex_count", d.count);
    draw.set("instance_count", d.instances);
    draw.set("base_vertex", d.base_vertex);
    draw.set("indexed", d.index_va != 0);
    std::uint64_t last_vertex = d.count ? d.count - 1 : 0;  // the highest vertex read, before base_vertex
    if (d.index_va) {
        const std::size_t bytes = static_cast<std::size_t>(d.count) * (d.index_type ? 4 : 2);
        if (!hle_kernel_va_mapped(d.index_va, bytes)) return reject("index buffer is not mapped");
        std::vector<std::uint8_t> indices(bytes);
        read_guest_locked(d.index_va, bytes, indices.data());
        s.blob("index", indices.data(), indices.size());
        last_vertex = 0;
        for (std::size_t k = 0; k < d.count; ++k) {
            std::uint32_t v = 0;
            if (d.index_type) {
                std::memcpy(&v, indices.data() + k * 4, 4);
            } else {
                std::uint16_t v16 = 0;
                std::memcpy(&v16, indices.data() + k * 2, 2);
                v = v16;
            }
            last_vertex = std::max<std::uint64_t>(last_vertex, v);
        }
        draw.set("index_type", d.index_type ? "uint32" : "uint16");
        draw.set("index_blob", "index");
        draw.set("index_va", json::hex(d.index_va));
    }

    // BBHOST_VERTEX_INPUT: elements, attributes, and each binding's bytes up to
    // the last element the draw reads.
    json::Value vertex_input;  // null for a draw that runs its fetch shader
    if (d.vertex_input) {
        if (d.base_vertex < 0) return reject("vertex input with a negative base vertex");
        const std::uint64_t base = static_cast<std::uint64_t>(d.base_vertex);
        vertex_input = json::Value::make_object();
        json::Value elements = json::Value::make_array();
        for (const gcn::VertexElement& e : d.vinput.elements) {
            json::Value ej = json::Value::make_object();
            ej.set("location", e.location);
            ej.set("vdata", e.vdata);
            ej.set("count", e.count);
            ej.set("w3", json::hex(e.w3));
            elements.push(ej);
        }
        vertex_input.set("elements", elements);
        json::Value bindings = json::Value::make_array();
        for (std::size_t b = 0; b < d.vinput.bindings.size(); ++b) {
            const VkVertexInputBindingDescription& bd = d.vinput.bindings[b];
            if (bd.binding != b) return reject("vertex input bindings are not numbered in order");
            std::uint64_t bytes = 0;
            for (const VkVertexInputAttributeDescription& a : d.vinput.attributes) {
                if (a.binding != bd.binding) continue;
                const std::uint32_t w3 = a.location < d.vinput.elements.size() ? d.vinput.elements[a.location].w3 : 0;
                bytes = std::max<std::uint64_t>(bytes, a.offset + (base + last_vertex) * bd.stride + vertex_element_bytes((w3 >> 15) & 0xf));
            }
            const std::uint64_t va = b < d.vinput.binding_va.size() ? d.vinput.binding_va[b] : 0;
            const std::string where = "vertex input binding " + std::to_string(b);
            if (!va || !bytes || bytes > kMaxVertexBytes) return reject(where + " has no guest address or is too large to capture");
            if (!hle_kernel_va_mapped(va, bytes)) return reject(where + " is not mapped");
            std::vector<std::uint8_t> data(bytes);
            read_guest_locked(va, bytes, data.data());
            const std::string name = "vertex-binding" + std::to_string(b);
            s.blob(name, data.data(), data.size());
            json::Value bj = json::Value::make_object();
            bj.set("binding", bd.binding);
            bj.set("stride", bd.stride);
            bj.set("guest_va", json::hex(va));
            bj.set("bytes", bytes);
            bj.set("blob", name);
            bindings.push(bj);
        }
        vertex_input.set("bindings", bindings);
        json::Value attributes = json::Value::make_array();
        for (const VkVertexInputAttributeDescription& a : d.vinput.attributes) {
            json::Value aj = json::Value::make_object();
            aj.set("location", a.location);
            aj.set("binding", a.binding);
            aj.set("format", static_cast<unsigned>(a.format));
            aj.set("offset", a.offset);
            attributes.push(aj);
        }
        vertex_input.set("attributes", attributes);
    }

    json::Value shaders = json::Value::make_object();
    shaders.set("geometry", d.gs_spirv ? json::Value("gs-spirv") : json::Value());
    if (d.gs_spirv) s.blob("gs-spirv", reinterpret_cast<const std::uint8_t*>(d.gs_spirv->data()), d.gs_spirv->size() * 4);
    shaders.set("fetch_gcn", d.fetch_words ? json::Value("fetch-gcn") : json::Value());
    if (d.fetch_words) s.blob("fetch-gcn", reinterpret_cast<const std::uint8_t*>(d.fetch_words->data()), d.fetch_words->size() * 4);

    json::Value targets = json::Value::make_object();
    json::Value colors = json::Value::make_array();
    for (std::size_t a = 0; a < d.fixed.color_slots.size(); ++a) {
        const RtImage* rt = d.color[d.fixed.color_slots[a]];
        if (!rt) continue;  // a gap below the last bound slot (fixed_state)
        json::Value tj = json::Value::make_object();
        tj.set("slot", d.fixed.color_slots[a]);
        tj.set("layer", d.color_layer[d.fixed.color_slots[a]]);  // of the target's `layers`; only this layer is read
        tj.set("layers", rt->layers);
        tj.set("guest_base", json::hex(rt->base));
        tj.set("format", static_cast<unsigned>(rt->format));
        json::Value ext = json::Value::make_array();
        ext.push(rt->width);
        ext.push(rt->height);
        tj.set("extent", ext);
        colors.push(tj);
    }
    targets.set("color", colors);
    if (d.depth) {
        json::Value dj = json::Value::make_object();
        dj.set("guest_base", json::hex(d.depth->base));
        dj.set("format", static_cast<unsigned>(d.depth->format));
        json::Value ext = json::Value::make_array();
        ext.push(d.depth->width);
        ext.push(d.depth->height);
        dj.set("extent", ext);
        targets.set("depth", dj);
    } else {
        targets.set("depth", nullptr);
    }
    std::string why;
    if (!s.target_readbacks("initial", targets, why)) return reject("target readback: " + why);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g.phys, &props);
    json::Value identity = json::Value::make_object();
    identity.set("pipeline", d.pipeline);
    identity.set("variant", d.lean ? "cb-lean (no page-table fallback)" : "fallback");
    identity.set("flip", d.flip);
    identity.set("draw_index", d.draw_index);
    identity.set("source", d.source);
    identity.set("device", props.deviceName);
    identity.set("vendor_id", json::hex(props.vendorID));
    identity.set("device_id", json::hex(props.deviceID));
    identity.set("driver_version", json::hex(props.driverVersion));
    identity.set("api_version", json::hex(props.apiVersion));
    identity.set("rejected_candidates_before", g_rejected);

    json::Value& m = s.manifest;
    m.set("schema", "bbhost-draw-capture");
    m.set("version", 1);
    m.set("identity", identity);
    m.set("translate", d.translate);
    m.set("shaders", shaders);
    m.set("pipeline", fixed_json(d.fixed));
    m.set("draw", draw);
    m.set("vertex_input", vertex_input);
    m.set("stages", stages);
    m.set("images", images);
    m.set("samplers", samplers);
    m.set("memory", memory);
    m.set("targets", targets);
    m.set("registers", d.registers);
    json::Value notes = json::Value::make_array();
    notes.push("The GPU was settled (every earlier submission completed) before this draw's descriptor sets were allocated.");
    notes.push("Inputs were copied after that; *-initial targets were read in the draw's own submission just before it, *-live just after.");
    notes.push("Guest pages cover the tables and vertex buffers the programs reach through the page table; replay's sink-sentinel run checks that.");
    m.set("notes", notes);
    return session;
}

void capture_finish_locked(CapturePtr session, const CaptureDynamic& dyn) {
    if (!session) return;
    CaptureSession& s = *session;
    json::Value& m = s.manifest;
    json::Value& targets = const_cast<json::Value&>(json::member(m, "targets"));
    std::string why;
    if (!s.target_readbacks("live", targets, why)) {
        host_log("capture: %s: live target readback failed: %s", s.dir.c_str(), why.c_str());
        return;
    }
    json::Value dj = json::Value::make_object();
    json::Value vp = json::Value::make_object();
    vp.set("x", json::f32(dyn.viewport.x));
    vp.set("y", json::f32(dyn.viewport.y));
    vp.set("width", json::f32(dyn.viewport.width));
    vp.set("height", json::f32(dyn.viewport.height));
    vp.set("min_depth", json::f32(dyn.viewport.minDepth));
    vp.set("max_depth", json::f32(dyn.viewport.maxDepth));
    dj.set("viewport", vp);
    json::Value sc = json::Value::make_array();
    sc.push(dyn.scissor.offset.x);
    sc.push(dyn.scissor.offset.y);
    sc.push(dyn.scissor.extent.width);
    sc.push(dyn.scissor.extent.height);
    dj.set("scissor", sc);
    json::Value area = json::Value::make_array();
    area.push(dyn.render_area.width);
    area.push(dyn.render_area.height);
    dj.set("render_area", area);
    if (dyn.depth_bounds_dynamic) {
        json::Value bounds = json::Value::make_array();
        bounds.push(json::f32(dyn.depth_bounds[0]));
        bounds.push(json::f32(dyn.depth_bounds[1]));
        dj.set("depth_bounds", bounds);
    } else {
        dj.set("depth_bounds", nullptr);
    }
    static const char* const kStencil[6] = {"reference_front", "reference_back", "compare_mask_front",
                                            "compare_mask_back", "write_mask_front", "write_mask_back"};
    json::Value stencil = json::Value::make_object();
    for (int k = 0; k < 6; ++k) stencil.set(kStencil[k], dyn.stencil[k]);
    dj.set("stencil", stencil);
    json::Value blend = json::Value::make_array();
    for (float f : dyn.blend_constants) blend.push(json::f32(f));
    dj.set("blend_constants", blend);
    m.set("dynamic", dj);

    flush_locked();

    json::Value blobs = json::Value::make_object();
    std::uint64_t total = 0;
    bool ok = true;
    const auto write = [&](const std::string& name, const std::uint8_t* data, std::size_t bytes) {
        const std::string rel = "blobs/" + name + ".bin";
        FILE* f = std::fopen((s.dir + "/" + rel).c_str(), "wb");
        if (!f) return false;
        const bool wrote = bytes == 0 || std::fwrite(data, 1, bytes, f) == bytes;
        if (std::fclose(f) != 0 || !wrote) return false;
        json::Value b = json::Value::make_object();
        b.set("file", rel);
        b.set("bytes", bytes);
        b.set("sha256", sha256_hex(data, bytes));
        blobs.set(name, b);
        total += bytes;
        return true;
    };
    for (Readback& rb : s.readbacks) {
        ok = ok && write(rb.blob, static_cast<const std::uint8_t*>(rb.staging.map), rb.bytes);
        vkDestroyBuffer(g.device, rb.staging.buffer, nullptr);
        vkFreeMemory(g.device, rb.staging.memory, nullptr);
    }
    s.readbacks.clear();
    for (const auto& [name, bytes] : s.host_blobs) ok = ok && write(name, bytes.data(), bytes.size());
    s.done = true;
    if (!ok) {
        host_log("capture: %s: writing blobs failed; no manifest written", s.dir.c_str());
        return;
    }
    m.set("blobs", blobs);
    const std::string text = json::dump(m);
    const std::string tmp = s.dir + "/manifest.json.tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    const bool wrote = f && std::fwrite(text.data(), 1, text.size(), f) == text.size();
    if (!f || std::fclose(f) != 0 || !wrote || std::rename(tmp.c_str(), (s.dir + "/manifest.json").c_str()) != 0) {
        host_log("capture: %s: writing the manifest failed", s.dir.c_str());
        return;
    }
    ++g_captured;
    if (g_armed > 0) --g_armed;
    host_log("capture: wrote %s/manifest.json (%s, flip %llu, draw %llu, %zu blobs, %llu bytes)", s.dir.c_str(),
             s.draw.pipeline.c_str(), static_cast<unsigned long long>(s.draw.flip), static_cast<unsigned long long>(s.draw.draw_index),
             blobs.object.size(), static_cast<unsigned long long>(total));
}

}  // namespace gpu

void host_gpu_capture_arm(int draws, const char* dir) {
    using namespace gpu;
    std::lock_guard<GpuMutex> lock(g.mu);
    g_armed = draws > 0 ? draws : 0;
    if (dir && *dir) g_armed_dir = dir;
    g_rejected = 0;  // each arming gets the attempts a run has
}
