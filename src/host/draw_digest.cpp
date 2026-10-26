#include "host/draw_digest.h"

#include "hle/modules.h"
#include "host/gpu.h"
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

// A record: a draw (kind 0, its hash) or a frame's end (1, the flips done then).
struct Record {
    std::uint32_t kind;
    std::uint32_t frame;  // the command processor's frames so far
    std::uint64_t digest;
};
static_assert(sizeof(Record) == 16);

struct Digest {
    std::FILE* out = nullptr;
    std::vector<Record> buf;
    std::uint32_t frame = 0, every = 1, frames_max = 20000, recorded = 0;
    std::uint64_t draws = 0;
    bool recording = true;
    std::mutex mu;  // the exit's flush against the command processor's thread
};

Digest* g_d = nullptr;

// FNV-1a over 64-bit words.
struct Hash {
    std::uint64_t h = 0xcbf29ce484222325ull;
    void add(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (8 * i)) & 0xff;
            h *= 0x100000001b3ull;
        }
    }
    void add_f(float f) {
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        add(u);
    }
};

void flush(Digest& d) {
    if (d.out && !d.buf.empty()) {
        std::fwrite(d.buf.data(), sizeof(Record), d.buf.size(), d.out);
        std::fflush(d.out);
    }
    d.buf.clear();
}

void at_exit() {
    if (!g_d) return;
    std::lock_guard<std::mutex> lk(g_d->mu);
    flush(*g_d);
    if (g_d->out) std::fclose(g_d->out);
    g_d->out = nullptr;
    host_log("draw digest: %u frames recorded, %llu draws", g_d->recorded, static_cast<unsigned long long>(g_d->draws));
}

// Opened at the first draw or frame's end, on the command processor's thread.
bool start() {
    const char* path = std::getenv("BBHOST_DRAW_DIGEST");
    if (!path || !*path) return false;
    auto* d = new Digest;
    d->out = std::fopen(path, "wb");
    if (!d->out) {
        host_log("draw digest: cannot write %s; off", path);
        delete d;
        return false;
    }
    if (const char* e = std::getenv("BBHOST_DRAW_DIGEST_EVERY"); e && std::atoi(e) > 0) d->every = static_cast<std::uint32_t>(std::atoi(e));
    if (const char* e = std::getenv("BBHOST_DRAW_DIGEST_FRAMES"); e && std::atoi(e) > 0) d->frames_max = static_cast<std::uint32_t>(std::atoi(e));
    const char magic[16] = {'D', 'R', 'A', 'W', 'D', 'I', 'G', '1'};
    std::fwrite(magic, 1, sizeof magic, d->out);
    d->buf.reserve(1u << 16);
    g_d = d;
    std::atexit(at_exit);
    host_log("draw digest: on, to %s (every %u frames, at most %u)", path, d->every, d->frames_max);
    return true;
}

bool g_started = false;

Digest* digest() {
    if (!g_started) {
        g_started = true;
        if (!start()) g_draw_digest = false;
    }
    return g_d;
}

}  // namespace

bool g_draw_digest = [] {
    const char* e = std::getenv("BBHOST_DRAW_DIGEST");
    return e && *e;
}();

void draw_digest_note(const GpuDrawInputs& in, const GxDrawObjects& o) {
    Digest* dp = digest();
    if (!dp) return;
    Digest& d = *dp;
    if (!d.recording || d.frame % d.every != 0) return;
    Hash h;
    h.add(o.caller);
    for (const std::uint64_t s : o.shader) h.add(s);
    h.add(o.vs_prog_id);
    h.add(o.ps_prog_id);
    h.add(o.il_id);
    h.add(o.blend_id);
    h.add(o.depth_stencil_id);
    h.add(o.raster_id);
    h.add((o.geometry ? 1u : 0u) | (o.indexed ? 2u : 0u) | (o.index_known ? 4u : 0u));
    h.add(o.count);
    h.add(o.instances);
    h.add(o.index_type);
    for (int i = 0; i < 2; ++i) {
        h.add(in.vs_pgm[i]);
        h.add(in.ps_pgm[i]);
    }
    h.add((in.tess ? 1u : 0u) | (in.gs ? 2u : 0u));
    h.add(in.prim);
    h.add(in.target_mask);
    h.add(in.cb_shader_mask);
    for (int t = 0; t < 8; ++t) {
        if (!((in.target_mask >> (4 * t)) & 0xf)) continue;
        for (int k = 0; k < 5; ++k) h.add(in.color[t][k]);
        h.add(in.blend[t]);
    }
    h.add(in.depth_control);
    h.add(in.stencil_control);
    h.add(in.z_info);
    h.add(in.z_read_base);
    h.add(in.su_sc_mode);
    h.add(in.clip_cntl);
    for (const float v : in.vport) h.add_f(v);
    for (int k = 0; k < 2; ++k) {
        h.add(in.screen_scissor[k]);
        h.add(in.generic_scissor[k]);
    }
    h.add(static_cast<std::uint32_t>(in.base_vertex));
    std::lock_guard<std::mutex> lk(d.mu);
    d.buf.push_back(Record{0, d.frame, h.h});
    ++d.draws;
}

void draw_digest_frame() {
    Digest* dp = digest();
    if (!dp) return;
    Digest& d = *dp;
    std::lock_guard<std::mutex> lk(d.mu);
    if (d.recording && d.frame % d.every == 0) {
        d.buf.push_back(Record{1, d.frame, static_cast<std::uint64_t>(hle_video_flip_count())});
        if (++d.recorded >= d.frames_max) d.recording = false;
        if (d.buf.size() >= (1u << 16)) flush(d);
    }
    ++d.frame;
}
