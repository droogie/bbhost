#include "host/shader_overrides.h"

#include "core/portable.h"
#include "host/translation_cache.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gpu {

std::atomic<std::uint64_t> g_override_gen{0};

namespace {

constexpr std::size_t kMaxText = 4u << 20;  // a source's size: the largest printed shaders are ~1 MiB

// FNV-1a over bytes and a final mix: a text's identity, a file's, a set's salt
// (never 0, which is no override).
struct Hash64 {
    std::uint64_t h = 0xcbf29ce484222325ull;
    void bytes(const void* p, std::size_t n) {
        const auto* b = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
    }
    void u64(std::uint64_t v) { bytes(&v, sizeof(v)); }
    void str(const std::string& s) {
        u64(s.size());
        bytes(s.data(), s.size());
    }
    std::uint64_t done() const {
        std::uint64_t x = h;
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ull;
        x ^= x >> 33;
        return x ? x : 1;
    }
};

bool hex16(const std::string& s) {
    if (s.size() != 16) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

// A set: what applies to one program, by the names its stages look for.
using Set = std::unordered_map<std::string, std::shared_ptr<const OverrideSource>>;

struct Owner {
    std::string name;  // for the logs
    std::string dir;   // relative folders start here
    bool on = true;
    std::map<std::string, std::shared_ptr<const OverrideSource>> texts;
    struct Folder {
        std::string dir;
        std::map<std::string, std::shared_ptr<const OverrideSource>> files;
    };
    std::vector<Folder> folders;
};

struct Outcome {
    std::uint64_t taken = 0, refused = 0;
    std::string why;  // the last refusal's
};

// A changed source waiting for its compile ahead (override_prepare): a text,
// or a folder's file (`folder` its folder).
struct Pending {
    std::shared_ptr<const OverrideSource> src;
    std::string folder;
};

struct Registry {
    std::mutex mu;
    bool on = false;
    std::atomic<bool> on_flag{false};  // `on`, read without the lock (overrides_on)
    std::string why_off;
    std::map<int, Owner> owners;  // in their order: the player's folder first
    // What applies: each name's source, the first owner's that has one.
    std::unordered_map<std::string, std::shared_ptr<const OverrideSource>> effective;
    std::unordered_map<std::string, std::vector<std::string>> by_stable;  // a program's stable name -> its names
    std::unordered_map<std::uint64_t, std::vector<std::string>> by_lift;  // a lift's hash -> its <hash>.<stage> names
    std::unordered_map<std::string, std::set<std::uint64_t>> program_lifts;  // such hashes met for a program
    std::unordered_map<std::uint64_t, std::shared_ptr<const Set>> sets;   // by salt; never erased in a run
    std::map<std::tuple<int, std::string, std::uint64_t>, Outcome> outcomes;  // by owner, name and identity
    std::map<std::pair<int, std::string>, Pending> pending;  // by owner and name: compiling ahead
    std::size_t player_unnamed = 0;  // the player's .hlsl files named after no shader
    std::atomic<std::uint64_t> compiles{0}, compile_us{0}, stages{0}, pipelines{0};
    std::atomic<std::uint64_t> window_compiles{0}, window_us{0}, window_stages{0}, window_pipelines{0};
    std::set<std::pair<int, std::string>> shadow_logged;
    std::atomic<bool> fallback{false};
};
// Never destroyed: the renderer's threads ask it until the process ends.
Registry& reg() {
    static Registry* const r = new Registry;
    return *r;
}

// The source for the logs (override_label), under mu.
std::string label_locked(const Registry& r, const OverrideSource& s) {
    if (s.owner == kOverridePlayer) return "shaders/" + s.name + ".hlsl";
    const auto o = r.owners.find(s.owner);
    const std::string owner = o != r.owners.end() && !o->second.name.empty() ? o->second.name : "owner " + std::to_string(s.owner);
    return owner + "'s " + (s.path.empty() ? s.name : s.path);
}

// The source of a file as it is now; null when it is not a file.
std::shared_ptr<const OverrideSource> file_source(int owner, const std::string& name, const std::filesystem::path& p) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(p, ec)) return nullptr;
    const std::uint64_t size = std::filesystem::file_size(p, ec);
    if (ec) return nullptr;
    const std::int64_t mtime = static_cast<std::int64_t>(std::filesystem::last_write_time(p, ec).time_since_epoch().count());
    if (ec) return nullptr;
    auto s = std::make_shared<OverrideSource>();
    s->name = name;
    s->owner = owner;
    s->path = p.string();
    s->size = size;
    s->mtime = mtime;
    Hash64 h;
    h.str(s->path);
    h.u64(size);
    h.u64(static_cast<std::uint64_t>(mtime));
    s->identity = h.done();
    return s;
}

// Every <shader>.hlsl in `dir`, by name; false when it is not a folder.
// `unnamed`: the .hlsl files whose names are no shader's.
bool list_folder(int owner, const std::string& dir, std::map<std::string, std::shared_ptr<const OverrideSource>>& out,
                 std::size_t* unnamed) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return false;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        const std::string file = e.path().filename().string();
        if (file.size() <= 5 || file.compare(file.size() - 5, 5, ".hlsl") != 0) continue;
        const std::string name = file.substr(0, file.size() - 5);
        std::string stable, stage, hash16;
        if (!override_parse(name, stable, stage, hash16)) {
            if (unnamed) ++*unnamed;
            continue;
        }
        if (auto s = file_source(owner, name, e.path())) out[name] = std::move(s);
    }
    return true;
}

// What applies, made again from the owners' sources; the generation moves
// when it changed. Under mu.
bool rebuild_locked(Registry& r) {
    std::unordered_map<std::string, std::shared_ptr<const OverrideSource>> eff;
    if (r.on) {
        const auto offer = [&](const std::shared_ptr<const OverrideSource>& s) {
            const auto [it, fresh] = eff.try_emplace(s->name, s);
            if (!fresh && it->second->owner != s->owner && r.shadow_logged.emplace(s->owner, s->name).second) {
                host_log("render: HLSL override %s not used: %s comes first", label_locked(r, *s).c_str(), label_locked(r, *it->second).c_str());
            }
        };
        for (const auto& [id, o] : r.owners) {
            if (!o.on) continue;
            for (const auto& [name, s] : o.texts) offer(s);
            for (const Owner::Folder& f : o.folders) {
                for (const auto& [name, s] : f.files) offer(s);
            }
        }
    }
    bool same = eff.size() == r.effective.size();
    for (auto it = eff.begin(); same && it != eff.end(); ++it) {
        const auto was = r.effective.find(it->first);
        same = was != r.effective.end() && was->second == it->second;
    }
    if (same) return false;
    r.effective = std::move(eff);
    r.by_stable.clear();
    r.by_lift.clear();
    for (const auto& [name, s] : r.effective) {
        std::string stable, stage, hash16;
        if (!override_parse(name, stable, stage, hash16)) continue;
        if (stable.empty()) r.by_lift[std::strtoull(hash16.c_str(), nullptr, 16)].push_back(name);
        else r.by_stable[stable].push_back(name);
    }
    r.fallback.store(!r.by_lift.empty(), std::memory_order_release);
    g_override_gen.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

// An owner's own source of a name, in effect or not: its text, else its first
// folder's file; null for none. Under mu.
std::shared_ptr<const OverrideSource> own_locked(const Registry& r, int owner, const std::string& name) {
    const auto o = r.owners.find(owner);
    if (o == r.owners.end()) return nullptr;
    if (const auto t = o->second.texts.find(name); t != o->second.texts.end()) return t->second;
    for (const Owner::Folder& f : o->second.folders) {
        if (const auto it = f.files.find(name); it != f.files.end()) return it->second;
    }
    return nullptr;
}

int state_locked(const Registry& r, int owner, const std::string& name, std::string* why) {
    const std::shared_ptr<const OverrideSource> own = own_locked(r, owner, name);
    const auto p = r.pending.find({owner, name});
    const auto say = [&](std::string w) {
        if (why) *why = std::move(w);
    };
    if (!own && p == r.pending.end()) return kOverrideNone;
    if (!r.on) {
        say(r.why_off.empty() ? std::string("edited shaders do not apply") : r.why_off);
        return kOverrideOff;
    }
    if (const auto o = r.owners.find(owner); o != r.owners.end() && !o->second.on) {
        say("its owner is off");
        return kOverrideWaiting;
    }
    if (p != r.pending.end()) {
        say("compiling ahead for the variants of its shader met so far");
        return kOverrideWaiting;
    }
    if (const auto e = r.effective.find(name); e != r.effective.end() && e->second->owner != owner) {
        say(label_locked(r, *e->second) + " comes first");
        return kOverrideShadowed;
    }
    const auto oc = r.outcomes.find({owner, name, own->identity});
    if (oc == r.outcomes.end() || (!oc->second.taken && !oc->second.refused)) {
        say("none of its shader's stages built with it yet");
        return kOverrideWaiting;
    }
    if (oc->second.taken) {
        if (oc->second.refused) say("refused for a variant: " + oc->second.why);
        return kOverrideTaken;
    }
    say(oc->second.why);
    return kOverrideRefused;
}

// Every name an owner gives a source for (texts, folders' files, pending). Under mu.
std::vector<std::string> names_locked(const Registry& r, int owner) {
    std::set<std::string> names;
    if (const auto o = r.owners.find(owner); o != r.owners.end()) {
        for (const auto& [name, s] : o->second.texts) names.insert(name);
        for (const Owner::Folder& f : o->second.folders) {
            for (const auto& [name, s] : f.files) names.insert(name);
        }
    }
    for (const auto& [key, p] : r.pending) {
        if (key.first == owner) names.insert(key.second);
    }
    return {names.begin(), names.end()};
}

// Whether an owner before `owner` has a source named `name` (so one of its
// own of that name would not apply: nothing to compile ahead). Under mu.
bool shadowed_locked(const Registry& r, int owner, const std::string& name) {
    for (const auto& [id, o] : r.owners) {
        if (id >= owner) break;
        if (!o.on) continue;
        if (o.texts.count(name)) return true;
        for (const Owner::Folder& f : o.folders) {
            if (f.files.count(name)) return true;
        }
    }
    return false;
}

// A pending source taken where it goes (its owner's texts or folder), unless
// another took its place or it was taken away; under mu, rebuilt by the caller.
bool install_locked(Registry& r, const std::shared_ptr<const OverrideSource>& s) {
    const auto p = r.pending.find({s->owner, s->name});
    if (p == r.pending.end() || p->second.src != s) return false;
    const std::string folder = p->second.folder;
    r.pending.erase(p);
    const auto o = r.owners.find(s->owner);
    if (o == r.owners.end()) return false;
    if (folder.empty()) {
        o->second.texts[s->name] = s;
        return true;
    }
    const auto f = std::find_if(o->second.folders.begin(), o->second.folders.end(), [&](const Owner::Folder& x) { return x.dir == folder; });
    if (f == o->second.folders.end()) return false;
    f->files[s->name] = s;
    return true;
}

// A changed file of a folder `dir`: gone at once, or its new source compiled
// ahead (override_prepare) and taken when that is done. Not under mu.
void file_changed(int owner, const std::string& dir, const std::string& name, std::shared_ptr<const OverrideSource> now) {
    Registry& r = reg();
    {
        std::lock_guard<std::mutex> lk(r.mu);
        const auto o = r.owners.find(owner);
        if (o == r.owners.end()) return;
        const auto f = std::find_if(o->second.folders.begin(), o->second.folders.end(), [&](const Owner::Folder& x) { return x.dir == dir; });
        if (f == o->second.folders.end()) return;
        if (!now) {
            r.pending.erase({owner, name});
            if (f->files.erase(name)) rebuild_locked(r);
            return;
        }
        const auto was = f->files.find(name);
        if (was != f->files.end() && was->second->identity == now->identity) return;  // listed already
        const auto p = r.pending.find({owner, name});
        if (p != r.pending.end() && p->second.src->identity == now->identity) return;  // compiling ahead already
        r.pending[{owner, name}] = {now, dir};
    }
    if (!override_prepare(now)) override_install(now);
}

// A file found changed where it was read: listed again (its new identity, or
// gone), so its shader's next draw keys and builds with what it holds now.
void relist(const OverrideSource& s) {
    std::shared_ptr<const OverrideSource> now = file_source(s.owner, s.name, s.path);
    std::string dir;
    {
        Registry& r = reg();
        std::lock_guard<std::mutex> lk(r.mu);
        const auto o = r.owners.find(s.owner);
        if (o == r.owners.end()) return;
        for (const Owner::Folder& f : o->second.folders) {
            const auto it = f.files.find(s.name);
            if (it != f.files.end() && it->second->path == s.path) dir = f.dir;
        }
        if (dir.empty()) return;
        host_log("render: HLSL override %s changed since it was listed: its shader is built with it again at its next draw",
                 label_locked(r, s).c_str());
    }
    file_changed(s.owner, dir, s.name, std::move(now));
}

}  // namespace

bool override_parse(const std::string& name, std::string& stable, std::string& stage, std::string& hash16) {
    std::string base = name, variant;
    if (const std::size_t tilde = base.find('~'); tilde != std::string::npos) {
        variant = base.substr(tilde + 1);
        base.resize(tilde);
        if (!hex16(variant)) return false;
    }
    const std::size_t dot = base.rfind('.');
    if (dot == std::string::npos || dot == 0) return false;
    stage = base.substr(dot + 1);
    static const char* const kStages[] = {"vs", "ps", "hs", "ds", "gs", "cs"};
    if (std::none_of(std::begin(kStages), std::end(kStages), [&](const char* k) { return stage == k; })) return false;
    const std::string head = base.substr(0, dot);
    for (const char c : head) {
        if (std::strchr("<>:\"/\\|?*", c) || static_cast<unsigned char>(c) < 32) return false;
    }
    if (hex16(head)) {  // a lift's hash: no GX name is 16 digits long
        if (!variant.empty()) return false;
        stable.clear();
        hash16 = head;
        return true;
    }
    stable = head;
    hash16 = variant;
    return true;
}

std::string override_label(const OverrideSource& s) {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    return label_locked(r, s);
}

void overrides_start(bool on, const std::string& why_off, const std::string& bbhost_dir) {
    Registry& r = reg();
    std::map<std::string, std::shared_ptr<const OverrideSource>> files;
    std::size_t unnamed = 0;
    const std::string dir = bbhost_dir.empty() ? std::string() : (std::filesystem::path(bbhost_dir) / "shaders").string();
    if (!dir.empty()) list_folder(kOverridePlayer, dir, files, &unnamed);  // listed off too, for the report
    std::lock_guard<std::mutex> lk(r.mu);
    r.on = on;
    r.on_flag.store(on, std::memory_order_release);
    r.why_off = why_off;
    Owner& player = r.owners[kOverridePlayer];
    player.name = "shaders/";
    player.dir = dir;
    player.folders.clear();
    if (!dir.empty()) player.folders.push_back({dir, std::move(files)});
    r.player_unnamed = unnamed;
    rebuild_locked(r);
    // BBHOST_SHADER_WATCH=1 (or a period in milliseconds): every folder listed
    // again while the game runs, a change applied once compiled.
    static std::once_flag watching;
    if (const char* e = std::getenv("BBHOST_SHADER_WATCH"); on && e && *e && std::strcmp(e, "0") != 0) {
        const int ms = std::atoi(e) > 1 ? std::atoi(e) : 500;
        std::call_once(watching, [ms, dir] {
            host_log("render: HLSL overrides: %s and the plugins' folders watched every %d ms (BBHOST_SHADER_WATCH): an edited shader applies "
                     "once it compiles", dir.empty() ? "no player's folder" : dir.c_str(), ms);
            std::thread([ms] {
                host_thread_set_name("bb-shwatch");
                host_thread_background();
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                    overrides_watch_scan();
                }
            }).detach();
        });
    }
    // The start's report: every edited shader, who gave it (the player's
    // folder, a plugin), and the ones another comes first for; or what turned
    // them off, and how many that leaves out.
    std::size_t listed = 0, shadowed = 0, all = 0;
    std::string list;
    for (const auto& [id, o] : r.owners) {
        for (const std::string& name : names_locked(r, id)) {
            ++all;
            if (!r.on) continue;
            const auto e = r.effective.find(name);
            if (e != r.effective.end() && e->second->owner != id) {
                ++shadowed;
                continue;
            }
            if (listed++ < 32) list += (list.empty() ? "" : ", ") + name + " (" + (id == kOverridePlayer ? std::string("your folder") : o.name) + ")";
        }
    }
    if (!r.on) {
        if (all) host_log("render: HLSL overrides off (%s): %zu edited shader%s not used", r.why_off.c_str(), all, all == 1 ? "" : "s");
    } else if (all) {
        host_log("render: HLSL overrides: %zu - %s%s%s; each compiled where its stage is first built%s", listed, list.c_str(),
                 listed > 32 ? (", and " + std::to_string(listed - 32) + " more").c_str() : "",
                 shadowed ? (" (" + std::to_string(shadowed) + " more not used: another of the same name comes first)").c_str() : "",
                 unnamed ? "; .hlsl files named after no shader in your folder: nothing reads them" : "");
    }
}

bool overrides_on() { return reg().on_flag.load(std::memory_order_acquire); }

bool overrides_watch_scan() {
    Registry& r = reg();
    struct Folder {
        int owner;
        std::string dir;
        std::map<std::string, std::uint64_t> files;  // name -> identity: the listing's, or the one compiling ahead
    };
    std::vector<Folder> folders;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        if (!r.on) return false;
        for (const auto& [id, o] : r.owners) {
            for (const Owner::Folder& f : o.folders) {
                Folder w{id, f.dir, {}};
                for (const auto& [name, s] : f.files) w.files[name] = s->identity;
                for (const auto& [key, p] : r.pending) {
                    if (key.first == id && p.folder == f.dir) w.files[key.second] = p.src->identity;
                }
                folders.push_back(std::move(w));
            }
        }
    }
    static std::atomic<int> logs{0};
    bool changed = false;
    for (const Folder& w : folders) {
        std::map<std::string, std::shared_ptr<const OverrideSource>> now;
        list_folder(w.owner, w.dir, now, nullptr);  // a folder gone: every file in it gone
        const auto note = [&](const std::string& name, const char* what) {
            changed = true;
            if (logs.fetch_add(1) < 64) {
                host_log("render: HLSL overrides: %s/%s.hlsl %s - its shader's next draw takes what applies now", w.dir.c_str(), name.c_str(), what);
            }
        };
        for (const auto& [name, s] : now) {
            const auto was = w.files.find(name);
            if (was != w.files.end() && was->second == s->identity) continue;
            note(name, was == w.files.end() ? "is new" : "changed");
            file_changed(w.owner, w.dir, name, s);
        }
        for (const auto& [name, identity] : w.files) {
            if (now.count(name)) continue;
            note(name, "is gone");
            file_changed(w.owner, w.dir, name, nullptr);
        }
    }
    return changed;
}

void overrides_owner(int owner, const std::string& name, const std::string& own_dir, bool on) {
    if (owner == kOverridePlayer) return;
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    Owner& o = r.owners[owner];
    o.name = name;
    o.dir = own_dir;
    o.on = on;
    rebuild_locked(r);
}

void overrides_owner_on(int owner, bool on) {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    const auto o = r.owners.find(owner);
    if (o == r.owners.end() || owner == kOverridePlayer || o->second.on == on) return;
    o->second.on = on;
    rebuild_locked(r);
}

void overrides_drop_owner(int owner) {
    if (owner == kOverridePlayer) return;
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    for (auto it = r.pending.begin(); it != r.pending.end();) it = it->first.first == owner ? r.pending.erase(it) : std::next(it);
    if (r.owners.erase(owner)) rebuild_locked(r);
}

int overrides_set_text(int owner, const char* shader, const char* hlsl) {
    std::string stable, stage, hash16;
    if (!shader || owner == kOverridePlayer || !override_parse(shader, stable, stage, hash16)) return 1;
    std::shared_ptr<OverrideSource> s;
    if (hlsl) {
        const std::size_t n = std::strlen(hlsl);
        if (n > kMaxText) return 1;
        s = std::make_shared<OverrideSource>();
        s->name = shader;
        s->owner = owner;
        s->text = std::make_shared<const std::string>(hlsl, n);
        Hash64 h;
        h.str(*s->text);
        s->identity = h.done();
    }
    Registry& r = reg();
    {
        std::lock_guard<std::mutex> lk(r.mu);
        const auto o = r.owners.find(owner);
        if (o == r.owners.end()) return 1;
        if (!s) {  // taken away: at once, with any text compiling ahead for it
            r.pending.erase({owner, shader});
            if (o->second.texts.erase(shader)) rebuild_locked(r);
            return 0;
        }
        const auto slot = o->second.texts.find(shader);
        if (slot != o->second.texts.end() && slot->second->identity == s->identity) {  // the same text again
            r.pending.erase({owner, shader});
            return 0;
        }
        const auto p = r.pending.find({owner, shader});
        if (p != r.pending.end() && p->second.src->identity == s->identity) return 0;  // compiling ahead already
        r.pending[{owner, shader}] = {s, std::string()};
        if (shadowed_locked(r, owner, shader)) {  // another owner's comes first: nothing of it to compile ahead
            if (install_locked(r, s)) rebuild_locked(r);
            return 0;
        }
    }
    // Compiled ahead for the variants met so far, then taken; at once when none was met.
    if (!override_prepare(s)) override_install(s);
    return 0;
}


void override_install(const std::shared_ptr<const OverrideSource>& s) {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    if (install_locked(r, s)) rebuild_locked(r);
}

bool overrides_ahead_idle() {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    return r.pending.empty();
}

void override_attribute(std::uint64_t lift_hash, const std::string& program) {
    if (program.empty()) return;
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    r.program_lifts[program].insert(lift_hash);
}

int overrides_add_folder(int owner, const char* dir) {
    if (owner == kOverridePlayer) return -1;
    Registry& r = reg();
    std::string base;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        const auto o = r.owners.find(owner);
        if (o == r.owners.end()) return -1;
        if (!dir || !*dir) {
            if (!o->second.folders.empty()) {
                o->second.folders.clear();
                rebuild_locked(r);
            }
            return 0;
        }
        base = o->second.dir;
    }
    std::filesystem::path p(dir);
    if (p.is_relative() && !base.empty()) p = std::filesystem::path(base) / p;
    const std::string where = p.lexically_normal().string();
    std::map<std::string, std::shared_ptr<const OverrideSource>> files;
    if (!list_folder(owner, where, files, nullptr)) return -1;
    const int n = static_cast<int>(files.size());
    // A file met before in another form (a folder listed again) is compiled
    // ahead, as a changed text is; the rest - the shaders not met yet, all of
    // them at the start - apply at once, together.
    std::vector<std::shared_ptr<const OverrideSource>> ahead;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        const auto o = r.owners.find(owner);
        if (o == r.owners.end()) return -1;
        auto f = std::find_if(o->second.folders.begin(), o->second.folders.end(), [&](const Owner::Folder& x) { return x.dir == where; });
        if (f == o->second.folders.end()) {
            o->second.folders.push_back({where, {}});
            f = o->second.folders.end() - 1;
        }
        std::map<std::string, std::shared_ptr<const OverrideSource>> now;
        for (auto& [name, s] : files) {
            const auto was = f->files.find(name);
            if (was != f->files.end() && was->second->identity == s->identity) {
                now[name] = was->second;  // unchanged: the same source, so what applies stays
                continue;
            }
            r.pending[{owner, name}] = {s, where};
            ahead.push_back(s);
            if (was != f->files.end()) now[name] = was->second;  // the old one until the new one is compiled
        }
        f->files = std::move(now);
        rebuild_locked(r);
    }
    std::vector<std::shared_ptr<const OverrideSource>> direct;
    for (const std::shared_ptr<const OverrideSource>& s : ahead) {
        if (!override_prepare(s)) direct.push_back(s);
    }
    if (!direct.empty()) {
        std::lock_guard<std::mutex> lk(r.mu);
        bool any = false;
        for (const std::shared_ptr<const OverrideSource>& s : direct) any |= install_locked(r, s);
        if (any) rebuild_locked(r);
    }
    return n;
}

std::uint64_t override_salt(const std::string& program) {
    if (program.empty() || !override_gen()) return 0;
    Registry& r = reg();
    const std::string stable = hlsl_file_name(program);  // outside the lock: the bundles' lookup
    std::lock_guard<std::mutex> lk(r.mu);
    if (r.effective.empty()) return 0;
    std::vector<std::pair<std::string, std::shared_ptr<const OverrideSource>>> parts;
    if (const auto it = r.by_stable.find(stable); it != r.by_stable.end()) {
        for (const std::string& name : it->second) parts.emplace_back(name, r.effective.at(name));
    }
    if (const auto it = r.program_lifts.find(program); it != r.program_lifts.end()) {
        for (const std::uint64_t hash : it->second) {
            if (const auto b = r.by_lift.find(hash); b != r.by_lift.end()) {
                for (const std::string& name : b->second) parts.emplace_back(name, r.effective.at(name));
            }
        }
    }
    if (parts.empty()) return 0;
    std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    Hash64 h;
    for (const auto& [name, s] : parts) {
        h.str(name);
        h.u64(s->identity);
        h.u64(static_cast<std::uint64_t>(s->owner));
    }
    const std::uint64_t salt = h.done();
    if (!r.sets.count(salt)) {
        auto set = std::make_shared<Set>();
        for (auto& [name, s] : parts) (*set)[name] = s;
        r.sets.emplace(salt, std::move(set));
    }
    return salt;
}

bool override_fallback_wanted() { return reg().fallback.load(std::memory_order_acquire); }

void override_lift_seen(std::uint64_t lift_hash, const std::string& program) {
    if (program.empty()) return;
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    if (!r.by_lift.count(lift_hash) || !r.program_lifts[program].insert(lift_hash).second) return;
    // The program's next draw takes the name into its salt.
    g_override_gen.fetch_add(1, std::memory_order_acq_rel);
}

bool override_pick(std::uint64_t salt, const std::string& stable, const std::string& stage, const std::string& hash16,
                   std::shared_ptr<const OverrideSource>& out) {
    if (!salt) return false;
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    const auto it = r.sets.find(salt);
    if (it == r.sets.end()) return false;
    const Set& set = *it->second;
    const std::string names[3] = {stable.empty() ? std::string() : stable + "." + stage + "~" + hash16,
                                  stable.empty() ? std::string() : stable + "." + stage, hash16 + "." + stage};
    for (const std::string& name : names) {
        if (name.empty()) continue;
        if (const auto f = set.find(name); f != set.end()) {
            out = f->second;
            return true;
        }
    }
    return false;
}

bool override_read(const OverrideSource& s, std::string& text, std::string& why) {
    if (s.text) {
        text = *s.text;
        return true;
    }
    {
        std::ifstream in(s.path, std::ios::binary);
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    // As listed, or another file than the source names: listed again.
    std::error_code ec;
    const std::uint64_t size = std::filesystem::file_size(s.path, ec);
    const std::int64_t mtime = ec ? 0 : static_cast<std::int64_t>(std::filesystem::last_write_time(s.path, ec).time_since_epoch().count());
    if (ec || size != s.size || mtime != s.mtime || text.size() != s.size) {
        why = "the file changed since it was listed";
        relist(s);
        return false;
    }
    return true;
}

std::string override_dxc_text(const OverrideSource& s, const std::string& text) {
    // A byte-order mark goes: past the #line it would be a character in the source.
    const std::size_t from = text.compare(0, 3, "\xEF\xBB\xBF") == 0 ? 3 : 0;
    return "#line 1 \"" + s.name + ".hlsl\"\n" + text.substr(from);
}

void override_outcome(const OverrideSource& s, bool taken, const std::string& why) {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    Outcome& o = r.outcomes[{s.owner, s.name, s.identity}];
    if (taken) {
        ++o.taken;
    } else {
        ++o.refused;
        o.why = why;
    }
}


int overrides_state(int owner, const std::string& name, std::string* why) {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    return state_locked(r, owner, name, why);
}

void override_note_compile(std::uint64_t us) {
    Registry& r = reg();
    r.compiles.fetch_add(1, std::memory_order_relaxed);
    r.compile_us.fetch_add(us, std::memory_order_relaxed);
    r.window_compiles.fetch_add(1, std::memory_order_relaxed);
    r.window_us.fetch_add(us, std::memory_order_relaxed);
}
void override_note_stage() {
    reg().stages.fetch_add(1, std::memory_order_relaxed);
    reg().window_stages.fetch_add(1, std::memory_order_relaxed);
}
void override_note_pipeline() {
    reg().pipelines.fetch_add(1, std::memory_order_relaxed);
    reg().window_pipelines.fetch_add(1, std::memory_order_relaxed);
}

std::vector<std::string> overrides_report() {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    std::vector<std::string> lines;
    std::size_t n[6] = {}, all = 0;
    for (const auto& [id, o] : r.owners) {
        for (const std::string& name : names_locked(r, id)) {
            std::string why;
            const int st = state_locked(r, id, name, &why);
            ++n[st];
            ++all;
            if (lines.size() >= 64) continue;
            std::string what;
            switch (st) {
                case kOverrideTaken: {
                    const std::shared_ptr<const OverrideSource> own = own_locked(r, id, name);
                    const auto oc = own ? r.outcomes.find({id, name, own->identity}) : r.outcomes.end();
                    what = "taken by " + std::to_string(oc != r.outcomes.end() ? oc->second.taken : 0) + " stage builds";
                    if (!why.empty()) what += "; " + why;
                    break;
                }
                case kOverrideRefused: what = "refused: " + why; break;
                case kOverrideShadowed: what = "not used: " + why; break;
                case kOverrideOff: what = "off: " + why; break;
                default: what = "not met: " + why; break;
            }
            lines.push_back("render: HLSL override " + name + " (" + (id == kOverridePlayer ? std::string("your folder") : o.name) + "): " + what);
        }
    }
    if (!all) return {};
    char b[400];
    std::snprintf(b, sizeof(b),
                  "render: HLSL overrides at exit: %zu (%zu taken, %zu refused, %zu not met, %zu not used, %zu off)%s; %llu modules compiled "
                  "(%.1f s of DXC), %llu stages and %llu pipelines built with them",
                  all, n[kOverrideTaken], n[kOverrideRefused], n[kOverrideWaiting], n[kOverrideShadowed], n[kOverrideOff],
                  all > lines.size() ? (", " + std::to_string(all - lines.size()) + " not listed").c_str() : "",
                  static_cast<unsigned long long>(r.compiles.load()), r.compile_us.load() / 1e6, static_cast<unsigned long long>(r.stages.load()),
                  static_cast<unsigned long long>(r.pipelines.load()));
    lines.insert(lines.begin(), b);
    return lines;
}

std::string overrides_window() {
    Registry& r = reg();
    const std::uint64_t compiles = r.window_compiles.exchange(0), us = r.window_us.exchange(0), stages = r.window_stages.exchange(0),
                        pipelines = r.window_pipelines.exchange(0);
    if (!compiles && !stages && !pipelines) return {};
    char b[160];
    std::snprintf(b, sizeof(b), "HLSL overrides: %llu stages and %llu pipelines built, %llu modules compiled (%llu ms)",
                  static_cast<unsigned long long>(stages), static_cast<unsigned long long>(pipelines), static_cast<unsigned long long>(compiles),
                  static_cast<unsigned long long>(us / 1000));
    return b;
}

}  // namespace gpu
