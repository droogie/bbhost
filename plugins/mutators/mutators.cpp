// Mutators: an official bbhost plugin (docs/plugins.md) - rules on top of
// the game, each one a switch. Any mix of them makes a different run.
//
//   [plugins]
//   mutators = true
//   [mutators]
//   speed = 1.0          # game time against real time: 0.75 a slower hunt, 1.25 a frantic one
//   gravity = 1.0        # 0.2: the moon is close - long falls, floating corpses
//   bullet_time = false  # under 30% HP the world slows to half speed
//   one_hit = false      # anything that hurts you kills you
//   blood_drain = 0      # % of max HP lost every 10 s (rally it back by hitting things)
//   enemy_hp = 1.0       # every enemy's HP times this (2.0: twice as tough)
//   echoes = 1.0         # Blood Echoes gained times this
//   chaos = 0            # every N seconds a random consumable's effect lands on you (0: off)
//   seed = ""            # the chaos order
//
// Everything is in memory (enemy_hp is NpcParam's hp, as the game reads it):
// turn a mutator off and the next start plays as the game does. Echoes and HP it changed stay in the save, as play would.
#include "bbhost/sdk.hpp"

#include <algorithm>
#include <cmath>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN | BB_PLUGIN_GAMEPLAY,
    "mutators",
    "ゲームルール変更",
    "1.0.0",
    "bbhost",
    "重力、速度、スロー効果、一撃死、HP 減少、敵の強化、血の遺志倍率、ランダム効果を個別に設定します。",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, nullptr, "ワールド", "ゲーム内メニューで直ちに反映されます。"},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "gravity", "重力", "0.2 では落下が遅くなり、死体も浮くように動きます。", "1.0", 0.05, 2.0},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "speed", "ゲーム速度", "0.75 で遅く、1.25 で速くなります。PC の性能に依存します。", "1.0", 0.25, 2.0},
    {BB_OPT_BOOL | BB_OPT_LIVE, "bullet_time", "瀕死時のスロー効果", "HP が 30% 未満になるとワールドの速度が半分になります。", "false"},
    {BB_OPT_HEADING, nullptr, "狩人"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "one_hit", "一撃死", "ダメージを受けると即死します。", "false"},
    {BB_OPT_INT | BB_OPT_LIVE, "blood_drain", "HP の自然減少", "10秒ごとに最大 HP の指定割合を失います。攻撃のリゲインで取り戻せます。", "0", 0, 20},
    {BB_OPT_FLOAT | BB_OPT_LIVE, "echoes", "獲得する血の遺志", nullptr, "1.0", 0.0, 10.0},
    {BB_OPT_INT | BB_OPT_LIVE, "chaos", "ランダム効果の間隔（秒）", "消耗品の効果をランダムに受けます。0 で無効。", "0", 0, 120},
    {BB_OPT_TEXT | BB_OPT_LIVE, "seed", "ランダム効果のシード", nullptr, ""},
    {BB_OPT_HEADING, nullptr, "敵"},
    {BB_OPT_FLOAT, "enemy_hp", "敵の HP 倍率", "すべての敵の HP に適用します（次回起動時）。", "1.0", 0.25, 10.0},
    {BB_OPT_END}};

namespace {

bb::Plugin g;
constexpr int kFps = 60;

float g_speed = 1.0f;
float g_gravity = 1.0f;
bool g_bullet_time = false;
bool g_one_hit = false;
float g_drain = 0.0f;
float g_enemy_hp = 1.0f;
float g_echoes = 1.0f;
int g_chaos = 0;

std::uint64_t g_frame = 0;
float g_scale_now = 1.0f;
std::int64_t g_last_hp = -1, g_last_echoes = -1;
// enemy_hp is written into NpcParam's hp, which the game derives every
// character's max HP from (it recomputes an active one's each update, so a
// character's own HP field would only be overwritten). The first row it
// wrote, to see the game load the table anew.
const void* g_npc_scaled_at = nullptr;
std::vector<std::int32_t> g_chaos_pool;
bb::Rng g_rng(1);

// ChrIns -> modules (+0x3b0) -> data module (+0x20): HP +0xf8, max HP +0xfc.
std::int32_t* hp_fields(void* ins) {
    auto* modules = *reinterpret_cast<std::uint8_t**>(static_cast<std::uint8_t*>(ins) + 0x3b0);
    auto* data = modules ? *reinterpret_cast<std::uint8_t**>(modules + 0x20) : nullptr;
    return data ? reinterpret_cast<std::int32_t*>(data + 0xf8) : nullptr;
}

// The SpEffects the game's consumables apply (EquipParamGoods refCategory 2
// names a SpEffectParam row in refId), less key items and anything that
// reaches the network or the player's echoes.
void build_chaos_pool() {
    g.each_row<bb::params::EquipParamGoods>([](std::uint32_t, const bb::EQUIP_PARAM_GOODS_ST& goods) {
        if (goods.refCategory != 2 || goods.refId <= 0 || goods.goodsType == 1) return;
        const auto* e = g.param<bb::params::SpEffectParam>(static_cast<std::uint32_t>(goods.refId));
        if (!e) return;
        if (e->requestSOS || e->requestBlackSOS || e->requestForceJoinBlackSOS || e->requestKickSession ||
            e->requestLeaveSession || e->requestNpcInveda || e->requestLeaveColiseumSession || e->clearSoul)
            return;
        if (std::find(g_chaos_pool.begin(), g_chaos_pool.end(), goods.refId) == g_chaos_pool.end())
            g_chaos_pool.push_back(goods.refId);
    });
    g.log("mutators: chaos draws from %zu consumable effects", g_chaos_pool.size());
}

void scale_enemy_hp() {
    const std::vector<std::uint32_t> ids = g.param_ids<bb::params::NpcParam>();
    if (ids.empty()) return;
    auto* first = g.param<bb::params::NpcParam>(ids.front());
    if (!first || first == g_npc_scaled_at) return;
    for (std::uint32_t id : ids) {
        if (auto* npc = g.param<bb::params::NpcParam>(id)) {
            const float hp = static_cast<float>(npc->hp) * g_enemy_hp;
            npc->hp = static_cast<std::uint32_t>(std::max(1.0f, std::min(hp, 4.0e9f)));
        }
    }
    g_npc_scaled_at = first;
    g.log("mutators: enemy HP x%.2f in %zu NpcParam rows", static_cast<double>(g_enemy_hp), ids.size());
}

void set_scale(float want) {
    // Eased, so bullet time slides in and out instead of snapping.
    g_scale_now += (want - g_scale_now) * 0.08f;
    if (std::fabs(g_scale_now - want) < 0.01f) g_scale_now = want;
    g.api->set_time_scale(g_scale_now);
}

void tick() {
    ++g_frame;
    const std::int64_t hp = g.stat("hp", -1), max_hp = g.stat("max_hp", -1);
    const bool in_world = hp >= 0 && max_hp > 0;

    float want = g_speed;
    if (g_bullet_time && in_world && hp > 0 && hp * 10 < max_hp * 3) want = g_speed * 0.5f;
    if (want != g_scale_now) set_scale(want);
    if (g_gravity != 1.0f) {
        // The world's vector, every frame: a load can make the world anew.
        if (float* grav = g.gravity(); grav && grav[1] != -9.8f * g_gravity) {
            grav[1] = -9.8f * g_gravity;
            g.log("mutators: world gravity %.2f", static_cast<double>(grav[1]));
        }
    }
    if (!in_world) {
        g_last_hp = -1;
        g_last_echoes = -1;
        return;
    }

    if (g_one_hit && g_last_hp > 0 && hp > 0 && hp < g_last_hp) {
        g.set_stat("hp", 0);
        g.message("一撃死。", 3.0f);
    }
    if (g_drain > 0.0f && g_frame % (10 * kFps) == 0 && hp > 1) {
        const std::int64_t loss = std::max<std::int64_t>(1, static_cast<std::int64_t>(static_cast<float>(max_hp) * g_drain / 100.0f));
        g.set_stat("hp", std::max<std::int64_t>(1, hp - loss));  // drains to 1, never kills
    }
    g_last_hp = g.stat("hp", hp);

    if (g_echoes != 1.0f) {
        const std::int64_t echoes = g.stat("echoes", -1);
        if (g_last_echoes >= 0 && echoes > g_last_echoes) {
            const std::int64_t gained = echoes - g_last_echoes;
            const std::int64_t now = g_last_echoes + static_cast<std::int64_t>(static_cast<float>(gained) * g_echoes);
            g.set_stat("echoes", std::max<std::int64_t>(0, now));
        }
        g_last_echoes = g.stat("echoes", echoes);
    }

    if (g_chaos > 0 && !g_chaos_pool.empty() && g_frame % static_cast<std::uint64_t>(g_chaos * kFps) == 0) {
        const std::int32_t effect = g_chaos_pool[g_rng.below(static_cast<std::uint32_t>(g_chaos_pool.size()))];
        g.sp_effect(effect);
        g.message("ランダム効果！", 2.0f);
        g.log("mutators: chaos applies SpEffect %d", effect);
    }
}

void read_options() {
    g_speed = static_cast<float>(g.config_number("mutators.speed", 1.0));
    g_gravity = static_cast<float>(g.config_number("mutators.gravity", 1.0));
    if (!(g_gravity > 0.01f)) g_gravity = 0.01f;
    g_bullet_time = g.config_bool("mutators.bullet_time", false);
    g_one_hit = g.config_bool("mutators.one_hit", false);
    g_drain = static_cast<float>(g.config_number("mutators.blood_drain", 0));
    g_enemy_hp = static_cast<float>(g.config_number("mutators.enemy_hp", 1.0));
    g_echoes = static_cast<float>(g.config_number("mutators.echoes", 1.0));
    g_chaos = static_cast<int>(g.config_number("mutators.chaos", 0));
    g_rng = bb::Rng(bb::seed_of(g.config("mutators.seed", "chaos")));
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (!g.attach(api, 9)) return 1;
    if (!api->eboot_is_109()) return 1;
    read_options();
    g.log("mutators: gravity x%.2f, speed %.2f, bullet time %s, one hit %s, blood drain %.0f%%, enemy HP x%.2f, echoes x%.2f, chaos %ds",
          static_cast<double>(g_gravity), static_cast<double>(g_speed), g_bullet_time ? "on" : "off", g_one_hit ? "on" : "off", static_cast<double>(g_drain),
          static_cast<double>(g_enemy_hp), static_cast<double>(g_echoes), g_chaos);
    g.every_frame([] {
        if (g_chaos > 0 && g_chaos_pool.empty() && g_frame % kFps == 0 && !g.param_ids<bb::params::EquipParamGoods>().empty())
            build_chaos_pool();
        if (g_enemy_hp != 1.0f && g_frame % 30 == 0) scale_enemy_hp();
        tick();
    });
    // A change in the in-game menu: read them all again (enemy_hp waits for
    // the next start - the table is scaled once).
    g.api->on_option("mutators", [](const char* key, const char*, void*) {
        const float enemy_hp = g_enemy_hp;
        const float gravity = g_gravity;
        read_options();
        g_enemy_hp = enemy_hp;
        if (gravity != 1.0f && g_gravity == 1.0f) {
            if (float* grav = g.gravity()) grav[1] = -9.8f;  // back to the game's own
        }
        g.log("mutators: %s changed", key);
    }, nullptr);
    g.on_world_load([](std::uint32_t) {
        g_last_hp = g_last_echoes = -1;
    });
    return 0;
}
