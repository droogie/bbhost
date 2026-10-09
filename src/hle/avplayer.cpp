// sceAvPlayer: movie playback. With ffmpeg (BBHOST_HAVE_FFMPEG) the mp4 is
// demuxed and decoded on a host thread; video frames are delivered as NV12
// in buffers obtained from the game's allocateTexture callback (so they are
// GPU-visible and become textures through the normal T# path), audio as
// 16-bit stereo PCM. Without ffmpeg a player reports playback finished on the
// first poll so the game skips the movie.
#include "hle/common.h"
#include "hle/hle.h"
#include "hle/modules.h"
#include "core/thunk.h"
#include "hle/fs.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(BBHOST_HAVE_FFMPEG)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#endif

namespace {

constexpr int kErrInvalidParams = static_cast<int>(0x806A0001u);
constexpr int kErrInvalidHandle = static_cast<int>(0x806A0002u);

// SceAvPlayerInitData (120 bytes)
struct InitData {
    void* mem_object;
    void* allocate;          // void* (*)(void* obj, uint32 align, uint32 size)
    void* deallocate;        // void (*)(void* obj, void* mem)
    void* allocate_texture;  // same as allocate
    void* deallocate_texture;
    void* file_object;
    void* file_open;
    void* file_close;
    void* file_read_offset;
    void* file_size;
    void* event_object;
    void* event_callback;    // void (*)(void* obj, int32 event, int32 source_id, void* data)
    std::uint32_t debug_level;
    std::uint32_t base_priority;
    std::int32_t num_video_buffers;
    std::uint8_t auto_start;
    std::uint8_t reserved[3];
    const char* default_language;
};
static_assert(sizeof(InitData) == 120, "SceAvPlayerInitData layout");

constexpr std::int32_t kStateStop = 1, kStateReady = 2, kStatePlay = 3;

struct VideoFrame {
    std::vector<std::uint8_t> nv12;  // luma then interleaved chroma, pitch-aligned
    std::uint64_t pts_us = 0;
};
struct AudioChunk {
    std::vector<std::uint8_t> pcm;   // s16 stereo
    std::uint64_t pts_us = 0;
};

struct StreamDesc {
    std::uint32_t type;  // 0 video, 1 audio, 2 timed text, 3 unknown
    int ff_index;
    std::uint32_t width = 0, height = 0, channels = 0, rate = 0;
    char lang[4] = {'e', 'n', 'g', 0};
};

struct Player {
    InitData init{};
    std::vector<StreamDesc> streams;  // container order: the game picks by index
    int audio_sel = -1;               // ffmpeg stream index of the enabled audio stream
    std::string source;
    bool started = false;
    bool looping = false;
    bool auto_start = false;
    int polls = 0;
    // Media
    bool has_media = false;
    std::uint32_t width = 0, height = 0, pitch = 0;
    std::uint32_t framerate = 30;
    std::uint64_t duration_us = 0;
    std::uint32_t audio_rate = 48000, audio_channels = 2;
    // Guest buffers
    std::vector<void*> video_buffers;
    std::size_t next_video = 0;
    std::vector<void*> audio_buffers;
    std::size_t next_audio = 0;
    // Playback clock
    std::chrono::steady_clock::time_point start_time{};
    std::uint64_t paused_at_us = 0;
    bool paused = false;
    std::uint64_t last_video_pts = 0;
    bool clock_anchored = false;  // see anchor_clock()
    bool stop_sent = false;
    // Decode thread
    std::thread thread;
    std::mutex qmu;
    std::condition_variable qcv;
    std::deque<VideoFrame> video;
    std::deque<AudioChunk> audio;
    bool eof = false;
    std::atomic<bool> quit{false};
    // Event delivery. libSceAvPlayer reports state changes from its own
    // controller thread after the API call that caused them returns; the
    // game's READY handler assumes sceAvPlayerAddSource has completed
    // (it creates its movie thread from an object initialised afterwards).
    std::thread event_thread;
    std::mutex emu;
    std::condition_variable ecv;
    std::deque<std::int32_t> events;
    bool equit = false;
    std::uint64_t threads_at_source = 0;  // hle_threads_made() when the source was added (hold_ready)
};

std::mutex g_mu;
std::unordered_map<std::uintptr_t, Player*> g_players;
std::uintptr_t g_next = 0x10;

Player* get(std::uintptr_t h) {
    auto it = g_players.find(h);
    return it == g_players.end() ? nullptr : it->second;
}

std::uint64_t clock_us(const Player& p) {
    if (!p.started) return 0;
    if (p.paused) return p.paused_at_us;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - p.start_time).count());
}

// The playback clock starts when the game asks for playback, but opening the
// file and decoding the first frames takes long enough that it would then stay
// permanently ahead of the media: every queued frame reads as "late", the
// catch-up loop in GetVideoDataEx drains the whole queue on one call and the
// next calls find it empty, and a 30 fps movie is shown at 8-17 fps. Anchor the
// clock to the first sample the game actually takes instead, so "due" means due
// relative to real playback. Returns true when it moved the clock.
bool anchor_clock(Player& p, std::uint64_t pts_us) {
    if (p.clock_anchored) return false;
    p.clock_anchored = true;
    p.start_time = std::chrono::steady_clock::now() - std::chrono::microseconds(pts_us);
    p.paused_at_us = pts_us;
    return true;
}

// First calls of every AvPlayer entry point (the game's sequence matters).
std::atomic<int> g_av_calls{0};
std::atomic<std::uint64_t> g_video_frames_delivered{0}, g_audio_chunks_delivered{0};
// Movie health: how often the game asks for a frame, how often we have none
// queued for it, and how many we skip to catch up. A healthy movie has
// get-video == movie-frames and zero of the other two.
std::atomic<std::uint64_t> g_video_calls{0}, g_video_starved{0}, g_video_dropped{0};
void av_trace(const char* what, std::uintptr_t h) {
    if (g_av_calls.fetch_add(1) < 80) host_log("avplayer: %s (0x%llx)", what, static_cast<unsigned long long>(h));
}

// The console's player reports READY once it has opened and buffered the
// source, well after sceAvPlayerAddSource returns; ours had it ready at once.
// The game makes its movie's audio thread (Mv_AudioPlayThread) just after
// that call, and its READY handler (sub_2cff6a0) takes that thread not
// running yet for a failed setup: CSMovieIns::STEP_Wait_Setup closes the
// player, which frees the thread object the new thread is about to lock -
// DL_PANIC "Mutex is not initialized" (DLLightMutex.cpp) as the title's
// attract movie opens, whenever a thread is slow to start. So READY waits
// until a thread made after the source was added has reached its entry: at
// most a second, for an owner that makes none, and no longer than the player
// stays open (a close joins this thread). BBHOST_MOVIE_READY_NOW=1
// reports it at once, as before (an A/B switch for the race, which
// BBHOST_TEST_THREAD_START_MS brings about on demand).
void hold_ready(Player& p) {
    static const bool now = [] {
        const char* e = std::getenv("BBHOST_MOVIE_READY_NOW");
        return e && e[0] == '1';
    }();
    if (now) return;
    const auto t0 = std::chrono::steady_clock::now();
    const auto ran = [&] { return hle_threads_made() > p.threads_at_source && hle_threads_starting() == 0; };
    const auto closing = [&] {
        std::lock_guard<std::mutex> lk(p.emu);
        return p.equit;
    };
    bool timed_out = false;
    while (!ran()) {
        if (closing()) return;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(1)) {
            timed_out = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Its first steps: the handler reads the state the thread sets first thing.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 8) {
        host_log("avplayer: READY after %lld ms%s", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                        std::chrono::steady_clock::now() - t0).count()),
                 timed_out ? ", no new thread ran" : ", once the thread made after the source was running");
    }
}

void event_thread_main(Player* p) {
    // A host thread that calls into the guest needs a guest TLS block; run
    // host code with host FS and let hle_call_guest6 switch for the call.
    void* tcb = hle_thread_enter_guest();
    hle_fs_host();
    for (;;) {
        std::int32_t ev;
        {
            std::unique_lock<std::mutex> lk(p->emu);
            p->ecv.wait(lk, [&] { return p->equit || !p->events.empty(); });
            if (p->events.empty()) break;
            ev = p->events.front();
            p->events.pop_front();
        }
        if (ev == kStateReady) hold_ready(*p);
        if (g_av_calls.load() < 80) host_log("avplayer: event %d delivered", ev);
        hle_call_guest6(p->init.event_callback, reinterpret_cast<std::int64_t>(p->init.event_object), ev, 0, 0, 0, 0);
    }
    hle_thread_leave_guest(tcb);
}

// Queue a state event for the delivery thread (never runs guest code on the
// caller: the caller may hold g_mu or be inside another AvPlayer call).
void send_event(Player& p, std::int32_t event) {
    if (!p.init.event_callback) return;
    std::lock_guard<std::mutex> lk(p.emu);
    if (!p.event_thread.joinable()) {
        p.equit = false;
        p.event_thread = std::thread(event_thread_main, &p);
    }
    p.events.push_back(event);
    p.ecv.notify_all();
}

void stop_event_thread(Player& p) {
    {
        std::lock_guard<std::mutex> lk(p.emu);
        p.equit = true;
        p.ecv.notify_all();
    }
    if (!p.event_thread.joinable()) return;
    if (p.event_thread.get_id() == std::this_thread::get_id()) {
        p.event_thread.detach();  // closed from inside the callback
    } else {
        p.event_thread.join();
    }
}

// The game's allocator callbacks. Never called under g_mu: the allocator
// takes the game's own heap lock, and the game's threads call AvPlayer
// (which takes g_mu) while holding it.
void* guest_alloc(const InitData& init, bool texture, std::uint32_t size) {
    void* fn = texture ? init.allocate_texture : init.allocate;
    if (!fn) return nullptr;
    return reinterpret_cast<void*>(hle_call_guest6(fn, reinterpret_cast<std::int64_t>(init.mem_object), 256, size, 0, 0, 0));
}
void guest_free(const InitData& init, bool texture, void* mem) {
    void* fn = texture ? init.deallocate_texture : init.deallocate;
    if (!fn || !mem) return;
    hle_call_guest6(fn, reinterpret_cast<std::int64_t>(init.mem_object), reinterpret_cast<std::int64_t>(mem), 0, 0, 0, 0);
}

#if defined(BBHOST_HAVE_FFMPEG)
// Half a second of video, and enough audio that the audio queue never gates the
// single demux thread: the game only takes audio a few hundred ms ahead of the
// clock, so a short audio queue blocked decode_thread in push_audio and video
// then arrived in bursts of a dozen frames separated by gaps in which the game
// found nothing to show. Video is the queue that should limit the demuxer.
constexpr std::size_t kMaxQueuedVideo = 16;
constexpr std::size_t kMaxQueuedAudio = 256;

void decode_thread(Player* p, std::string path) {
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0 || avformat_find_stream_info(fmt, nullptr) < 0) {
        host_log("avplayer: cannot open %s", path.c_str());
        std::lock_guard<std::mutex> lk(p->qmu);
        p->eof = true;
        p->qcv.notify_all();
        return;
    }
    int vidx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    int aidx = p->audio_sel >= 0 && p->audio_sel < static_cast<int>(fmt->nb_streams) &&
                       fmt->streams[p->audio_sel]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO
                   ? p->audio_sel
                   : av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    AVCodecContext* vctx = nullptr;
    AVCodecContext* actx = nullptr;
    SwsContext* sws = nullptr;
    SwrContext* swr = nullptr;
    auto open_codec = [&](int idx) -> AVCodecContext* {
        if (idx < 0) return nullptr;
        const AVCodec* codec = avcodec_find_decoder(fmt->streams[idx]->codecpar->codec_id);
        if (!codec) return nullptr;
        AVCodecContext* ctx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(ctx, fmt->streams[idx]->codecpar);
        ctx->thread_count = 4;
        if (avcodec_open2(ctx, codec, nullptr) < 0) {
            avcodec_free_context(&ctx);
            return nullptr;
        }
        return ctx;
    };
    vctx = open_codec(vidx);
    actx = open_codec(aidx);
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    const std::uint32_t pitch = p->pitch, w = p->width, h = p->height;
    auto push_video = [&](AVFrame* f) {
        if (!sws) {
            sws = sws_getContext(f->width, f->height, static_cast<AVPixelFormat>(f->format), static_cast<int>(w), static_cast<int>(h),
                                 AV_PIX_FMT_NV12, SWS_BILINEAR, nullptr, nullptr, nullptr);
        }
        VideoFrame vf;
        vf.nv12.resize(static_cast<std::size_t>(pitch) * h * 3 / 2);
        std::uint8_t* planes[2] = {vf.nv12.data(), vf.nv12.data() + static_cast<std::size_t>(pitch) * h};
        int strides[2] = {static_cast<int>(pitch), static_cast<int>(pitch)};
        sws_scale(sws, f->data, f->linesize, 0, f->height, planes, strides);
        const AVRational tb = fmt->streams[vidx]->time_base;
        const std::int64_t pts = f->best_effort_timestamp != AV_NOPTS_VALUE ? f->best_effort_timestamp : f->pts;
        vf.pts_us = pts == AV_NOPTS_VALUE ? 0 : static_cast<std::uint64_t>(av_rescale_q(pts, tb, AVRational{1, 1000000}));
        std::unique_lock<std::mutex> lk(p->qmu);
        p->qcv.wait(lk, [&] { return p->video.size() < kMaxQueuedVideo || p->quit.load(); });
        p->video.push_back(std::move(vf));
        p->qcv.notify_all();
    };
    auto push_audio = [&](AVFrame* f) {
        if (!swr) {
            AVChannelLayout out;
            av_channel_layout_default(&out, 2);
            swr_alloc_set_opts2(&swr, &out, AV_SAMPLE_FMT_S16, static_cast<int>(p->audio_rate), &f->ch_layout,
                                static_cast<AVSampleFormat>(f->format), f->sample_rate, 0, nullptr);
            swr_init(swr);
        }
        const int out_samples = static_cast<int>(av_rescale_rnd(swr_get_delay(swr, f->sample_rate) + f->nb_samples, p->audio_rate,
                                                                 f->sample_rate, AV_ROUND_UP));
        AudioChunk ac;
        ac.pcm.resize(static_cast<std::size_t>(out_samples) * 4);
        std::uint8_t* out_ptr = ac.pcm.data();
        const int got = swr_convert(swr, &out_ptr, out_samples, const_cast<const std::uint8_t**>(f->data), f->nb_samples);
        if (got <= 0) return;
        ac.pcm.resize(static_cast<std::size_t>(got) * 4);
        const AVRational tb = fmt->streams[aidx]->time_base;
        ac.pts_us = f->pts == AV_NOPTS_VALUE ? 0 : static_cast<std::uint64_t>(av_rescale_q(f->pts, tb, AVRational{1, 1000000}));
        std::unique_lock<std::mutex> lk(p->qmu);
        p->qcv.wait(lk, [&] { return p->audio.size() < kMaxQueuedAudio || p->quit.load(); });
        p->audio.push_back(std::move(ac));
        p->qcv.notify_all();
    };
    while (!p->quit.load()) {
        const int rr = av_read_frame(fmt, pkt);
        if (rr < 0) {
            // Flush decoders.
            for (AVCodecContext* ctx : {vctx, actx}) {
                if (!ctx) continue;
                avcodec_send_packet(ctx, nullptr);
                while (avcodec_receive_frame(ctx, frame) == 0 && !p->quit.load()) {
                    if (ctx == vctx) push_video(frame); else push_audio(frame);
                }
            }
            if (p->looping) {
                av_seek_frame(fmt, -1, 0, AVSEEK_FLAG_BACKWARD);
                if (vctx) avcodec_flush_buffers(vctx);
                if (actx) avcodec_flush_buffers(actx);
                continue;
            }
            break;
        }
        AVCodecContext* ctx = pkt->stream_index == vidx ? vctx : pkt->stream_index == aidx ? actx : nullptr;
        if (ctx && avcodec_send_packet(ctx, pkt) == 0) {
            while (avcodec_receive_frame(ctx, frame) == 0 && !p->quit.load()) {
                if (ctx == vctx) push_video(frame); else push_audio(frame);
            }
        }
        av_packet_unref(pkt);
    }
    {
        std::lock_guard<std::mutex> lk(p->qmu);
        p->eof = true;
        p->qcv.notify_all();
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    if (sws) sws_freeContext(sws);
    if (swr) swr_free(&swr);
    if (vctx) avcodec_free_context(&vctx);
    if (actx) avcodec_free_context(&actx);
    avformat_close_input(&fmt);
}

bool probe_media(Player& p, const std::string& path) {
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0) return false;
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        avformat_close_input(&fmt);
        return false;
    }
    const int vidx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    const int aidx = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    p.streams.clear();
    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
        const AVCodecParameters* cp = fmt->streams[i]->codecpar;
        StreamDesc d;
        d.ff_index = static_cast<int>(i);
        d.type = cp->codec_type == AVMEDIA_TYPE_VIDEO ? 0 : cp->codec_type == AVMEDIA_TYPE_AUDIO ? 1 : cp->codec_type == AVMEDIA_TYPE_SUBTITLE ? 2 : 3;
        d.width = static_cast<std::uint32_t>(cp->width);
        d.height = static_cast<std::uint32_t>(cp->height);
        d.channels = 2;  // what GetAudioData delivers (the file's 5.1 is downmixed)
        d.rate = cp->sample_rate > 0 ? static_cast<std::uint32_t>(cp->sample_rate) : 48000;
        if (const AVDictionaryEntry* e = av_dict_get(fmt->streams[i]->metadata, "language", nullptr, 0); e && e->value && std::strlen(e->value) >= 3) {
            std::memcpy(d.lang, e->value, 3);
        }
        p.streams.push_back(d);
    }
    p.audio_sel = aidx;
    if (vidx >= 0) {
        const AVCodecParameters* cp = fmt->streams[vidx]->codecpar;
        p.width = static_cast<std::uint32_t>(cp->width);
        p.height = static_cast<std::uint32_t>(cp->height);
        p.pitch = (p.width + 63) & ~63u;
        const AVRational fr = fmt->streams[vidx]->avg_frame_rate;
        p.framerate = fr.den ? static_cast<std::uint32_t>((fr.num + fr.den / 2) / fr.den) : 30;
    }
    if (aidx >= 0) {
        p.audio_rate = static_cast<std::uint32_t>(fmt->streams[aidx]->codecpar->sample_rate);
        if (!p.audio_rate) p.audio_rate = 48000;
        p.audio_channels = 2;
    }
    p.duration_us = fmt->duration > 0 ? static_cast<std::uint64_t>(fmt->duration) : 0;
    p.has_media = vidx >= 0;
    avformat_close_input(&fmt);
    return p.has_media;
}
#endif

void stop_thread(Player& p) {
    p.quit.store(true);
    p.qcv.notify_all();
    if (p.thread.joinable()) p.thread.join();
    p.quit.store(false);
    std::lock_guard<std::mutex> lk(p.qmu);
    p.video.clear();
    p.audio.clear();
    p.eof = false;
}

// Without g_mu: a closed player, no longer in g_players.
void release_buffers(Player& p) {
    for (void* b : p.video_buffers) guest_free(p.init, true, b);
    for (void* b : p.audio_buffers) guest_free(p.init, false, b);
    p.video_buffers.clear();
    p.audio_buffers.clear();
}

// The frame buffers playback needs, from the game's allocator: `lk` (g_mu) is
// let go over the calls and taken again. The player again by its handle, or
// null when it was closed meanwhile.
Player* ensure_buffers(std::unique_lock<std::mutex>& lk, std::uintptr_t h, Player* p) {
#if defined(BBHOST_HAVE_FFMPEG)
    if (!p->has_media || !p->video_buffers.empty()) return p;
    const InitData init = p->init;
    const int n = init.num_video_buffers > 0 ? init.num_video_buffers : 3;
    const std::uint32_t bytes = p->pitch * p->height * 3 / 2;
    lk.unlock();
    std::vector<void*> video, audio;
    for (int i = 0; i < n; ++i) {
        void* b = guest_alloc(init, true, bytes);
        if (!b) break;
        std::memset(b, 0, bytes);
        video.push_back(b);
    }
    for (int i = 0; i < 4; ++i) {
        void* b = guest_alloc(init, false, 16384);
        if (!b) break;
        audio.push_back(b);
    }
    lk.lock();
    p = get(h);
    if (!p || !p->video_buffers.empty()) {  // closed, or given buffers by another call meanwhile
        lk.unlock();
        for (void* b : video) guest_free(init, true, b);
        for (void* b : audio) guest_free(init, false, b);
        lk.lock();
        return get(h);
    }
    p->video_buffers = std::move(video);
    p->audio_buffers = std::move(audio);
    host_log("avplayer: %ux%u pitch %u, %u fps, %u Hz audio, %zu video buffers from the game's allocator", p->width, p->height,
             p->pitch, p->framerate, p->audio_rate, p->video_buffers.size());
#else
    (void)lk;
    (void)h;
#endif
    return p;
}

// Under g_mu, the buffers already made (ensure_buffers).
void begin_playback(Player& p) {
#if defined(BBHOST_HAVE_FFMPEG)
    if (!p.has_media) return;
    stop_thread(p);
    p.start_time = std::chrono::steady_clock::now();
    p.paused = false;
    p.stop_sent = false;
    p.last_video_pts = 0;
    p.clock_anchored = false;
    p.started = true;
    Player* pp = &p;
    p.thread = std::thread(decode_thread, pp, hle_fs_map_path(p.source.c_str()));
    send_event(p, kStatePlay);
#else
    p.started = true;
    p.polls = 0;
#endif
}

// SceAvPlayerHandle sceAvPlayerInit(SceAvPlayerInitData*)
GUEST_ABI std::uintptr_t hle_av_init(const InitData* init) {
    if (!init) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    std::uintptr_t h = g_next;
    g_next += 0x10;
    Player* p = new Player();
    p->init = *init;
    p->auto_start = init->auto_start != 0;
    g_players[h] = p;
    host_log("sceAvPlayerInit -> 0x%llx (video buffers %d, autostart %d)", static_cast<unsigned long long>(h),
             init->num_video_buffers, init->auto_start);
    return h;
}
GUEST_ABI int hle_av_post_init(std::uintptr_t h, const void*) {
    av_trace("post_init", h);
    std::lock_guard<std::mutex> lock(g_mu);
    return get(h) ? 0 : kErrInvalidHandle;
}
GUEST_ABI int hle_av_add_source(std::uintptr_t h, const char* path) {
    std::unique_lock<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p) {
        return kErrInvalidHandle;
    }
    p->source = path ? path : "";
    // BBHOST_NO_MOVIE=1 mirrors shadPS4 (AvPlayer stubbed): accept the source
    // but never decode or signal READY, so the game skips the movie and
    // proceeds. Diagnostic for the menu-preload divergence.
    static const bool no_movie = [] {
        const char* e = std::getenv("BBHOST_NO_MOVIE");
        return e && e[0] == '1';
    }();
    if (no_movie) {
        host_log("HLE fake: sceAvPlayerAddSource %s (BBHOST_NO_MOVIE: movie skipped)", p->source.c_str());
        return 0;
    }
#if defined(BBHOST_HAVE_FFMPEG)
    const std::string host = hle_fs_map_path(p->source.c_str());
    if (!probe_media(*p, host)) {
        host_log("HLE fake: sceAvPlayerAddSource %s (cannot decode; movie skipped)", p->source.c_str());
        return 0;
    }
    host_log("sceAvPlayerAddSource %s -> %s", p->source.c_str(), host.c_str());
    p->threads_at_source = hle_threads_made();
    send_event(*p, kStateReady);
    if (p->auto_start && (p = ensure_buffers(lock, h, p)) != nullptr) begin_playback(*p);
#else
    host_log("HLE fake: sceAvPlayerAddSource %s (movie skipped: built without ffmpeg)", p->source.c_str());
#endif
    return 0;
}
GUEST_ABI int hle_av_start(std::uintptr_t h) {
    av_trace("start", h);
    std::unique_lock<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p) {
        return kErrInvalidHandle;
    }
    if (p->has_media && p->started && p->paused) {
        p->paused = false;
        p->start_time = std::chrono::steady_clock::now() - std::chrono::microseconds(p->paused_at_us);
        return 0;
    }
    p = ensure_buffers(lock, h, p);
    if (!p) return kErrInvalidHandle;
    begin_playback(*p);
    return 0;
}
GUEST_ABI int hle_av_stop(std::uintptr_t h) {
    av_trace("stop", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p) {
        return kErrInvalidHandle;
    }
    if (p->started && p->has_media) {
        stop_thread(*p);
        if (!p->stop_sent) {
            p->stop_sent = true;
            send_event(*p, kStateStop);
        }
    }
    p->started = false;
    return 0;
}
GUEST_ABI int hle_av_pause(std::uintptr_t h) {
    av_trace("pause", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p) return kErrInvalidHandle;
    if (p->started && !p->paused) {
        p->paused_at_us = clock_us(*p);
        p->paused = true;
    }
    return 0;
}
GUEST_ABI int hle_av_resume(std::uintptr_t h) { return hle_av_start(h); }
GUEST_ABI int hle_av_set_looping(std::uintptr_t h, bool loop) {
    av_trace("set_looping", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p) {
        return kErrInvalidHandle;
    }
    p->looping = loop;
    return 0;
}
// bool sceAvPlayerIsActive(handle)
GUEST_ABI bool hle_av_is_active(std::uintptr_t h) {
    av_trace("is_active", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p || !p->started) {
        return false;
    }
    if (!p->has_media) {
        if (++p->polls > 2) {
            p->started = false;
            return false;
        }
        return true;
    }
    bool done;
    {
        std::lock_guard<std::mutex> lk(p->qmu);
        done = p->eof && p->video.empty() && p->audio.empty();
    }
    if (done) {
        p->started = false;
        stop_thread(*p);
        if (!p->stop_sent) {
            p->stop_sent = true;
            send_event(*p, kStateStop);
        }
        host_log("avplayer: %s finished", p->source.c_str());
        return false;
    }
    return true;
}

// bool sceAvPlayerGetVideoDataEx(handle, SceAvPlayerFrameInfoEx*)
GUEST_ABI bool hle_av_get_video(std::uintptr_t h, std::uint8_t* info) {
    av_trace("get_video", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p || !p->started || !p->has_media || !info || p->video_buffers.empty()) return false;
    g_video_calls.fetch_add(1);
    std::uint64_t now = clock_us(*p);
    VideoFrame vf;
    {
        std::lock_guard<std::mutex> lk(p->qmu);
        if (!p->video.empty() && anchor_clock(*p, p->video.front().pts_us)) now = clock_us(*p);
        if (p->video.empty()) {
            g_video_starved.fetch_add(1);
            return false;
        }
        // Deliver the oldest frame that is due. Catch up by dropping only what
        // is hopelessly late: frame-threaded decoding hands us frames in bursts
        // of two or three, and dropping every frame that was merely "due"
        // delivered one of each burst and then left the queue empty for the
        // next few calls - a 30 fps movie played at 17. Never drop the last
        // queued frame; showing it a little late beats showing nothing.
        constexpr std::uint64_t kTooLateUs = 150000;
        // Dropping only helps if the decoder can then catch up. When it is
        // producing at exactly the media rate - which it does once the host is
        // busy - lateness never shrinks and we would throw away five frames in
        // six for the rest of the movie. Past half a second behind, accept the
        // offset instead: move the clock onto the queue and keep every frame.
        if (p->video.front().pts_us + 500000 < now) {
            p->start_time += std::chrono::microseconds(now - p->video.front().pts_us);
            now = clock_us(*p);
            static std::atomic<int> logs{0};
            if (logs.fetch_add(1) < 4) {
                host_log("avplayer: %llu ms behind; moved the clock onto the queue instead of dropping",
                         static_cast<unsigned long long>(500));
            }
        }
        while (p->video.size() > 1 && p->video.front().pts_us + kTooLateUs < now) {
            p->video.pop_front();
            g_video_dropped.fetch_add(1);
        }
        if (p->video.front().pts_us > now + 2000) return false;  // not due yet
        vf = std::move(p->video.front());
        p->video.pop_front();
        p->qcv.notify_all();
    }
    void* buf = p->video_buffers[p->next_video++ % p->video_buffers.size()];
    std::memcpy(buf, vf.nv12.data(), vf.nv12.size());
    p->last_video_pts = vf.pts_us;
    g_video_frames_delivered.fetch_add(1);
    static std::atomic<int> logs{0};
    if (logs.fetch_add(1) < 3) {
        host_log("avplayer: video frame %llu ms into buffer %p (%ux%u NV12)", static_cast<unsigned long long>(vf.pts_us / 1000), buf,
                 p->width, p->height);
    }
    std::memset(info, 0, 120);
    const std::uint64_t ptr = reinterpret_cast<std::uint64_t>(buf);
    const std::uint64_t ts_ms = vf.pts_us / 1000;
    std::memcpy(info + 0, &ptr, 8);
    std::memcpy(info + 16, &ts_ms, 8);
    std::memcpy(info + 24, &p->width, 4);
    std::memcpy(info + 28, &p->height, 4);
    const float aspect = static_cast<float>(p->width) / static_cast<float>(p->height);
    std::memcpy(info + 32, &aspect, 4);
    std::memcpy(info + 36, "eng", 4);
    std::memcpy(info + 40, &p->framerate, 4);
    std::memcpy(info + 60, &p->pitch, 4);
    info[64] = 8;
    info[65] = 8;
    info[66] = 0;
    return true;
}
// bool sceAvPlayerGetAudioData(handle, SceAvPlayerFrameInfo*)
GUEST_ABI bool hle_av_get_audio(std::uintptr_t h, std::uint8_t* info) {
    av_trace("get_audio", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p || !p->started || !p->has_media || !info || p->audio_buffers.empty()) return false;
    AudioChunk ac;
    {
        std::lock_guard<std::mutex> lk(p->qmu);
        if (p->audio.empty()) return false;
        // Keep audio roughly in step with the clock: do not run more than
        // ~300 ms ahead of it.
        if (p->audio.front().pts_us > clock_us(*p) + 300000) return false;
        anchor_clock(*p, p->audio.front().pts_us);
        ac = std::move(p->audio.front());
        p->audio.pop_front();
        p->qcv.notify_all();
    }
    void* buf = p->audio_buffers[p->next_audio++ % p->audio_buffers.size()];
    const std::size_t n = ac.pcm.size() > 16384 ? 16384 : ac.pcm.size();
    std::memcpy(buf, ac.pcm.data(), n);
    std::memset(info, 0, 48);
    const std::uint64_t ptr = reinterpret_cast<std::uint64_t>(buf);
    const std::uint64_t ts_ms = ac.pts_us / 1000;
    std::memcpy(info + 0, &ptr, 8);
    std::memcpy(info + 16, &ts_ms, 8);
    const std::uint16_t channels = static_cast<std::uint16_t>(p->audio_channels);
    std::memcpy(info + 24, &channels, 2);
    std::memcpy(info + 28, &p->audio_rate, 4);
    const std::uint32_t size = static_cast<std::uint32_t>(n);
    std::memcpy(info + 32, &size, 4);
    std::memcpy(info + 36, "eng", 4);
    g_audio_chunks_delivered.fetch_add(1);
    return true;
}
GUEST_ABI std::uint64_t hle_av_current_time(std::uintptr_t h) {
    av_trace("current_time", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    return p ? clock_us(*p) / 1000 : 0;
}
GUEST_ABI int hle_av_stream_count(std::uintptr_t h) {
    av_trace("stream_count", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p) return kErrInvalidHandle;
    return p->has_media ? static_cast<int>(p->streams.size()) : 0;
}
// int sceAvPlayerGetStreamInfo(handle, uint32 id, SceAvPlayerStreamInfo*)
GUEST_ABI int hle_av_stream_info(std::uintptr_t h, std::uint32_t id, std::uint8_t* info) {
    av_trace("stream_info", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p || !info) return kErrInvalidParams;
    if (!p->has_media || id >= p->streams.size()) return kErrInvalidParams;
    // SceAvPlayerStreamInfo: +0 type, +8 details (video: w, h, aspect, lang;
    // audio: u16 channels, pad, rate, size, lang), +24 duration ms, +32 start.
    const StreamDesc& d = p->streams[id];
    std::memset(info, 0, 40);
    std::memcpy(info + 0, &d.type, 4);
    if (d.type == 0) {
        std::memcpy(info + 8, &d.width, 4);
        std::memcpy(info + 12, &d.height, 4);
        const float aspect = d.height ? static_cast<float>(d.width) / static_cast<float>(d.height) : 1.0f;
        std::memcpy(info + 16, &aspect, 4);
        std::memcpy(info + 20, d.lang, 4);
    } else if (d.type == 1) {
        const std::uint16_t channels = static_cast<std::uint16_t>(d.channels);
        std::memcpy(info + 8, &channels, 2);
        std::memcpy(info + 12, &d.rate, 4);
        const std::uint32_t size = 4096;
        std::memcpy(info + 16, &size, 4);
        std::memcpy(info + 20, d.lang, 4);
    }
    const std::uint64_t duration_ms = p->duration_us / 1000;
    std::memcpy(info + 24, &duration_ms, 8);
    if (g_av_calls.load() < 80) {
        host_log("avplayer: stream %u is %s (%s)", id, d.type == 0 ? "video" : d.type == 1 ? "audio" : "other", d.lang);
    }
    return 0;
}
GUEST_ABI int hle_av_enable_stream(std::uintptr_t h, std::uint32_t id) {
    av_trace("enable_stream", h);
    std::lock_guard<std::mutex> lock(g_mu);
    Player* p = get(h);
    if (!p) return kErrInvalidHandle;
    if (id < p->streams.size() && p->streams[id].type == 1) {
        p->audio_sel = p->streams[id].ff_index;
        host_log("avplayer: audio stream %u selected", id);
    }
    return 0;
}

GUEST_ABI int hle_av_close(std::uintptr_t h) {
    av_trace("close", h);
    Player* p = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        p = get(h);
        if (!p) return kErrInvalidHandle;
        stop_thread(*p);
        g_players.erase(h);
    }
    // Outside g_mu: the delivery thread may be inside another AvPlayer call,
    // and the game's deallocator takes its own heap lock.
    stop_event_thread(*p);
    release_buffers(*p);
    delete p;
    return 0;
}

}  // namespace

void hle_register_avplayer() {
#define REG(name, fn) register_hle_fn(name, reinterpret_cast<void*>(fn))
    REG("sceAvPlayerInit", hle_av_init);
    REG("sceAvPlayerPostInit", hle_av_post_init);
    REG("sceAvPlayerAddSource", hle_av_add_source);
    REG("sceAvPlayerStart", hle_av_start);
    REG("sceAvPlayerStop", hle_av_stop);
    REG("sceAvPlayerPause", hle_av_pause);
    REG("sceAvPlayerResume", hle_av_resume);
    REG("sceAvPlayerSetLooping", hle_av_set_looping);
    REG("sceAvPlayerIsActive", hle_av_is_active);
    REG("sceAvPlayerGetVideoDataEx", hle_av_get_video);
    REG("sceAvPlayerGetAudioData", hle_av_get_audio);
    REG("sceAvPlayerCurrentTime", hle_av_current_time);
    REG("sceAvPlayerStreamCount", hle_av_stream_count);
    REG("sceAvPlayerGetStreamInfo", hle_av_stream_info);
    REG("sceAvPlayerEnableStream", hle_av_enable_stream);
    REG("sceAvPlayerClose", hle_av_close);
#undef REG
}

std::string hle_av_stats() {
    char buf[200];
    std::snprintf(buf, sizeof(buf), " movie-frames=%llu movie-audio-chunks=%llu get-video=%llu starved=%llu dropped=%llu",
                  static_cast<unsigned long long>(g_video_frames_delivered.load()),
                  static_cast<unsigned long long>(g_audio_chunks_delivered.load()),
                  static_cast<unsigned long long>(g_video_calls.load()), static_cast<unsigned long long>(g_video_starved.load()),
                  static_cast<unsigned long long>(g_video_dropped.load()));
    return buf;
}
