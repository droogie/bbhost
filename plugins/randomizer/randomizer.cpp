// Randomizer: an official bbhost plugin (docs/plugins.md).
//
// Every one-time pickup in the world - the items lying in Yharnam, on corpses,
// in chests, the ones bosses leave - gets another pickup's item, from a seed,
// and the shops sell each other's wares. Key items (doors, elevators, the
// story) stay where the game put them, so a run can always be finished.
// Regular enemies become other kinds: from their own area, or from the whole
// game, and bosses can turn up among them.
//
// It changes the game's tables in memory and the map layouts and AI bundles
// in its overlay, nothing on disk: turn it off and the next start is vanilla
// again. Items already picked up stay in the save.
//
//   [plugins]
//   randomizer = true
//   [randomizer]
//   seed = "my run"     # empty: one is made the first time and kept in <data>/plugins/randomizer/seed.txt
//   items = true        # world pickups
//   shops = true        # shop lineups
//   chalices = true     # chalices go into the pool with everything else
//   enemies = true      # regular enemies become other kinds
//   enemy_from = "same area"   # or "whole game"
//   enemy_mix = "evenly"       # or "like the original"
//   enemy_npcs = false  # hostile hunters (c0000) too
//   enemy_bosses = "none"      # "a few", "many": roaming bosses among them
//
// <data>/plugins/randomizer/spoiler.txt lists where everything went, and
// enemies.txt what each area's enemies became.
#include "bbhost/sdk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <cstring>
#include <tuple>
#include <set>

extern "C" BB_PLUGIN_EXPORT const BbPluginInfo bb_plugin_info = {
    BB_PLUGIN_API_VERSION,
    BB_PLUGIN_OPT_IN | BB_PLUGIN_GAMEPLAY | BB_PLUGIN_ADOPTS_RULES,
    "randomizer",
    "ランダマイザー",
    "1.1.0",
    "bbhost",
    "シードに基づき配置アイテム、店の商品、敵を入れ替えます。重要アイテムは動かさず、クリア可能な状態を維持します。",
};

extern "C" BB_PLUGIN_EXPORT const BbOption bb_plugin_options[] = {
    {BB_OPT_HEADING, nullptr, "シードの設定", "同じシードでは同じ配置になります。アイテムと店は即時、敵は次のエリア読み込み時に反映します。他プレイヤーの世界では相手のシードを使います。"},
    {BB_OPT_TEXT | BB_OPT_LIVE, "seed", "シード", "空欄では初回に生成し plugins/randomizer/seed.txt に保存します。「ランダムなシード」で再生成できます。同じシードを共有すると同じ配置で遊べます。", ""},
    {BB_OPT_HEADING, nullptr, "アイテム"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "items", "配置アイテムを入れ替える", "地面・遺体・宝箱・ボス報酬のアイテムを入れ替えます。鍵や物語に必要な品は元の場所を維持します。", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "chalices", "聖杯も入れ替える", "無効では聖杯を元の場所に残します。", "true"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "shops", "店の商品を入れ替える", "店同士で商品を入れ替えます。各商品の価格は維持します。", "true"},
    {BB_OPT_HEADING, nullptr, "敵"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "enemies", "敵を入れ替える", "通常の敵を別種に入れ替えます。ボスと会話できる NPC は維持します。", "true"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "enemy_from", "置き換える敵の範囲", "同じエリアでは元からいる種類を使用します。全エリアでは独自の AI を含めて使い、終盤の敵は元の強さを維持します。", "same area", 0, 0, "same area|whole game"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "enemy_mix", "各種の出現頻度", "均等では全種類を同じ確率にします。原作準拠では元の多さを維持します。", "evenly", 0, 0, "evenly|like the original"},
    {BB_OPT_BOOL | BB_OPT_LIVE, "enemy_npcs", "敵対する狩人を含める", "狩人装備の人型の敵も入れ替えます。会話できる NPC は変更しません。", "false"},
    {BB_OPT_CHOICE | BB_OPT_LIVE, "enemy_bosses", "通常の敵としてボスを出す", "ガスコイン神父、エミーリア、マリアなどが通常の敵に混ざって出現します。能力はボスのままです。本来のボス戦は移動しません。", "none", 0, 0, "none|a few|many"},
    {BB_OPT_END}};

namespace {

using bb::params::EquipParamGoods;
using bb::params::ItemLotParam;
using bb::params::ShopLineupParam;

bb::Plugin g;

// ItemLotParam categories (ITEMLOT_ITEMCATEGORY as the table uses them).
enum : std::int32_t { kWeapon = 0, kArmor = 1, kGoods = 4, kRune = 8, kGem = 15 };
// EquipParamGoods.goodsType.
enum : std::uint8_t { kGoodsKeyItem = 1, kGoodsChalice = 6 };
// ShopLineupParam.equipType.
enum : std::uint8_t { kShopGoods = 3 };
// Lots at and above this id are the chalice dungeons' generated loot.
constexpr std::uint32_t kDungeonLots = 100000000;

struct Item {
    std::int32_t category, id;
    std::uint8_t num;
};

struct Shop {
    std::int32_t equip_id, value;
    std::uint8_t equip_type;
    std::int16_t value_san;
};

std::string g_seed;        // the seed as the player wrote it (or seed.txt's)
std::string g_token;       // bb::seed_token(g_seed): what the world is made from, and what others see
// The enemies a world has: a seed and the enemy settings (an empty token:
// the game's own). The layouts depend on nothing else, so this is what the
// rules carry and what a guest needs to make a host's world.
struct World {
    std::string token;
    bool even = true, npcs = false;
    bool anywhere = false;  // kinds from the whole game, not only the area's own
    int bosses = 0;         // roaming bosses among the enemies: 0 none, 1 a few, 2 many
    std::string key() const {
        if (token.empty()) return {};
        return token + (even ? "|even" : "|found") + (npcs ? "|npcs" : "") + (anywhere ? "|any" : "") +
               (bosses ? "|bosses" + std::to_string(bosses) : "");
    }
};
std::string g_world;       // World::key() of the layouts written (a host's while a guest)
bool g_visitor = false;    // off, loaded only to play other players' worlds (visitor())
std::mutex g_layout_mu;    // layouts are rewritten on the main thread and on the network thread (a join);
                           // it also guards g_token, g_world, g_enemies and g_visiting across them
bool g_visiting = false;   // in another player's world (its layouts written, ours on the way home)
bool g_items = true, g_shops = true, g_chalices = true, g_enemies = true, g_enemy_even = true, g_enemy_npcs = false;
bool g_enemy_anywhere = false;
int g_enemy_bosses = 0;
bool g_enemies_done = false;
// The movable rows as the game made them: every plan is made from these, and
// written back before a new one is applied (a re-seed, a setting turned off).
std::vector<std::pair<std::uint32_t, Item>> g_lot_orig;
std::vector<std::pair<std::uint32_t, Shop>> g_shop_orig;
// The plan: the row each item goes to.
std::vector<std::pair<std::uint32_t, Item>> g_lot_plan;
std::vector<std::pair<std::uint32_t, Shop>> g_shop_plan;
std::string g_pending_banner;  // shown once the next world has loaded (under g_layout_mu)
// The row the plan was last written to, to see the game reload a table.
const void* g_lots_at = nullptr;
const void* g_shops_at = nullptr;
bool g_announced = false;

std::uint8_t goods_type(std::int32_t id) {
    auto* goods = g.param<EquipParamGoods>(static_cast<std::uint32_t>(id));
    return goods ? goods->goodsType : 0;
}

// A lot the shuffle may move: one item in its first slot, picked up once.
bool movable_lot(std::uint32_t id, const bb::ITEMLOT_PARAM_ST& r) {
    if (id >= kDungeonLots || r.getItemFlagId <= 0 || r.lotItemId01 <= 0 || r.lotItemNum01 == 0) return false;
    if (r.lotItemId02 > 0 || r.lotItemId03 > 0 || r.lotItemId04 > 0 || r.lotItemId05 > 0 || r.lotItemId06 > 0 ||
        r.lotItemId07 > 0 || r.lotItemId08 > 0)
        return false;
    const std::int32_t c = r.lotItemCategory01;
    if (c != kWeapon && c != kArmor && c != kGoods && c != kRune && c != kGem) return false;
    if (c == kGoods) {
        const std::uint8_t t = goods_type(r.lotItemId01);
        if (t == kGoodsKeyItem) return false;  // chalices: in the snapshot, left out of a plan by make_plan
    }
    return true;
}

bool movable_shop(const bb::SHOP_LINEUP_PARAM& r) {
    if (r.equipId <= 0 || r.value <= 0) return false;
    return !(r.equipType == kShopGoods && goods_type(r.equipId) == kGoodsKeyItem);
}

const char* category_name(std::int32_t c) {
    switch (c) {
    case kWeapon: return "weapon";
    case kArmor: return "armor";
    case kGoods: return "item";
    case kRune: return "rune";
    case kGem: return "gem";
    default: return "?";
    }
}

// The movable rows, once the tables are in: what every plan starts from.
void snapshot_originals() {
    g.each_row<ItemLotParam>([&](std::uint32_t id, const bb::ITEMLOT_PARAM_ST& r) {
        if (movable_lot(id, r)) g_lot_orig.emplace_back(id, Item{r.lotItemCategory01, r.lotItemId01, r.lotItemNum01});
    });
    g.each_row<ShopLineupParam>([&](std::uint32_t id, const bb::SHOP_LINEUP_PARAM& r) {
        if (movable_shop(r)) g_shop_orig.emplace_back(id, Shop{r.equipId, r.value, r.equipType, r.value_SAN});
    });
}

void make_plan() {
    g_lot_plan.clear();
    g_shop_plan.clear();
    bb::Rng rng(bb::seed_of(g_token));
    if (g_items) {
        std::vector<std::uint32_t> rows;
        std::vector<Item> items;
        for (const auto& [id, it] : g_lot_orig) {
            if (!g_chalices && it.category == kGoods && goods_type(it.id) == kGoodsChalice) continue;
            rows.push_back(id);
            items.push_back(it);
        }
        rng.shuffle(items);
        for (std::size_t i = 0; i < rows.size(); ++i) g_lot_plan.emplace_back(rows[i], items[i]);
    }
    if (g_shops) {
        // A shop is listed once per stage of the game it changes at (the
        // 100000-140000 blocks are one shop five times), so wares are
        // swapped ware for ware - every row selling A sells B, at B's price -
        // and each stage of a shop sells the same things.
        std::map<std::pair<int, std::int32_t>, Shop> first;  // a ware's first listing
        for (const auto& [id, w] : g_shop_orig) first.emplace(std::make_pair(static_cast<int>(w.equip_type), w.equip_id), w);
        std::vector<Shop> from, to;
        for (const auto& [key, ware] : first) from.push_back(ware);
        to = from;
        rng.shuffle(to);
        std::map<std::pair<int, std::int32_t>, Shop> swap;
        for (std::size_t i = 0; i < from.size(); ++i) swap[{from[i].equip_type, from[i].equip_id}] = to[i];
        for (const auto& [id, w] : g_shop_orig) g_shop_plan.emplace_back(id, swap[{w.equip_type, w.equip_id}]);
    }
}

void write_spoiler() {
    const std::string dir = g.api->plugin_dir("randomizer");
    if (dir.empty()) return;
    std::ofstream f(dir + "/spoiler.txt");
    f << "bbhost randomizer " << bb_plugin_info.version << " - seed \"" << g_seed << "\"\n\n";
    f << "# pickups: lot (its pickup flag) -> what it now gives\n";
    for (const auto& [row, it] : g_lot_plan) {
        const auto* r = g.param<ItemLotParam>(row);
        f << "lot " << row << " (flag " << (r ? r->getItemFlagId : 0) << "): " << category_name(it.category) << ' ' << it.id
          << " x" << static_cast<int>(it.num) << '\n';
    }
    f << "\n# shops: lineup row -> what it now sells\n";
    for (const auto& [row, w] : g_shop_plan) {
        f << "shop " << row << ": type " << static_cast<int>(w.equip_type) << " id " << w.equip_id << " for " << w.value
          << '\n';
    }
}

void write_lot(std::uint32_t row, const Item& it) {
    if (auto* r = g.param<ItemLotParam>(row)) {
        r->lotItemCategory01 = it.category;
        r->lotItemId01 = it.id;
        r->lotItemNum01 = it.num;
    }
}
void write_shop(std::uint32_t row, const Shop& w) {
    if (auto* r = g.param<ShopLineupParam>(row)) {
        r->equipId = w.equip_id;
        r->equipType = w.equip_type;
        r->value = w.value;
        r->value_SAN = w.value_san;
    }
}

// Writes the plan into the live tables: the original rows first, then the
// plan over them. Again whenever the game has loaded a table anew (a new
// buffer is vanilla) or the plan changed (force).
void apply(bool force = false) {
    if (g_lot_orig.empty() && g_shop_orig.empty()) return;
    auto* lots = g_lot_orig.empty() ? nullptr : g.param<ItemLotParam>(g_lot_orig.front().first);
    if (lots && (force || lots != g_lots_at)) {
        for (const auto& [row, it] : g_lot_orig) write_lot(row, it);
        for (const auto& [row, it] : g_lot_plan) write_lot(row, it);
        g_lots_at = lots;
        g.log("randomizer: %zu pickups placed", g_lot_plan.size());
    }
    auto* shops = g_shop_orig.empty() ? nullptr : g.param<ShopLineupParam>(g_shop_orig.front().first);
    if (shops && (force || shops != g_shops_at)) {
        for (const auto& [row, w] : g_shop_orig) write_shop(row, w);
        for (const auto& [row, w] : g_shop_plan) write_shop(row, w);
        g_shops_at = shops;
        g.log("randomizer: %zu shop wares placed", g_shop_plan.size());
    }
}

// --- enemies: the map layouts, rewritten into the plugin's overlay ---------
//
// A map's layout (map/mapstudio/<map>.msb.dcx) lists its parts; an enemy
// part names a model (an index into the layout's model list), an NpcParam
// row, an AI (NpcThinkParam) and gear (CharaInitParam) in 0x40 bytes of type
// data. Those are fixed-size fields, rewritten in place; the model list's
// per-model instance counts are kept equal to the parts using them.
//
// A kind from the area itself has its model and AI loaded there already. One
// from elsewhere (enemies from the whole game, roaming bosses) needs its
// model in the layout's model list - an entry is inserted after the last
// enemy model and the parts' model indices past it renumbered - and its AI
// programs in the area's script bundle (the next section).

constexpr const char* kMaps[] = {
    "m21_00_00_00", "m21_01_00_00", "m22_00_00_00", "m23_00_00_00", "m23_00_00_01", "m24_00_00_00",
    "m24_00_00_01", "m24_01_00_00", "m24_01_00_01", "m24_01_00_11", "m24_02_00_00", "m24_02_00_01",
    "m25_00_00_00", "m26_00_00_00", "m27_00_00_00", "m27_00_00_01", "m28_00_00_00", "m28_00_00_01",
    "m32_00_00_00", "m32_00_00_01", "m33_00_00_00", "m34_00_00_00", "m35_00_00_00", "m36_00_00_00",
};

// Bosses that fight as ordinary enemies away from their arenas, by NpcParam
// row: each was put, alone, in the slot of the huntsman who kills the frozen
// seed's hunter in Central Yharnam, and killed the hunter too. Father
// Gascoigne as a man, Laurence, Ludwig, the Cleric Beast and Darkbeast Paarl
// stood where they were put (their arenas' events start them), the
// Blood-starved Beast of Old Yharnam (209000) circled without striking; Rom,
// Amygdala, Ebrietas, the One Reborn and the Moon Presence need their arenas,
// the Witch of Hemwick her Mad Ones, and the lesser Amygdalas only watch.
constexpr std::int32_t kRoamingBosses[] = {
    209010,                  // Blood-starved Beast (the Hunter's Nightmare's)
    212700, 212710, 212720,  // the Shadows of Yharnam
    232000,                  // Martyr Logarius
    272000,                  // Father Gascoigne, the beast
    452000,                  // Lady Maria of the Astral Clocktower
    454000,                  // Orphan of Kos
    502000,                  // Vicar Amelia
};

// Each enemy model's files (chr/<model>.chrbnd and .anibnd) in MB, rounded
// up, for the budget below; a model not listed counts as kModelMbDefault.
constexpr std::pair<int, int> kModelMb[] = {
    {0, 3}, {1000, 7}, {1001, 8}, {1050, 9}, {1051, 3}, {1060, 10}, {1090, 5}, {1091, 5}, {1100, 6}, {1110, 4},
    {1111, 1}, {1120, 9}, {1140, 8}, {1170, 4}, {1171, 1}, {1180, 3}, {1190, 6}, {1220, 9}, {1240, 9}, {1241, 2},
    {1250, 6}, {1260, 6}, {1270, 5}, {1271, 3}, {1290, 4}, {1300, 8}, {1400, 8}, {2000, 10}, {2020, 8}, {2040, 6},
    {2050, 5}, {2090, 14}, {2100, 7}, {2110, 8}, {2120, 9}, {2121, 2}, {2130, 4}, {2150, 10}, {2170, 25}, {2180, 14},
    {2190, 1}, {2250, 9}, {2260, 9}, {2320, 12}, {2330, 8}, {2400, 12}, {2500, 5}, {2501, 2}, {2520, 5}, {2530, 7},
    {2540, 3}, {2550, 4}, {2560, 14}, {2561, 1}, {2570, 11}, {2571, 1}, {2600, 14}, {2610, 16}, {2620, 14}, {2630, 11},
    {2631, 8}, {2632, 5}, {2640, 9}, {2700, 15}, {2710, 16}, {2720, 10}, {2730, 17}, {2740, 8}, {3060, 9}, {3100, 16},
    {4000, 10}, {4010, 14}, {4020, 21}, {4021, 6}, {4022, 1}, {4030, 7}, {4031, 1}, {4040, 15}, {4050, 4}, {4060, 15},
    {4070, 10}, {4080, 6}, {4100, 6}, {4110, 17}, {4120, 4}, {4130, 4}, {4140, 5}, {4500, 10}, {4510, 14}, {4520, 15},
    {4540, 16}, {4550, 1}, {5000, 10}, {5020, 7}, {5033, 2}, {5080, 16}, {5510, 13}, {7100, 7}, {7110, 6}, {7500, 8},
};
constexpr int kModelMbDefault = 8;
// The model files an area's enemies may be drawn from, unless its own come to
// more: Central Yharnam ran with the game's 16 largest, 244 MB, in a test.
constexpr int kAreaModelBudgetMb = 250;
// Roaming bosses: one slot in N becomes one ("a few", "many"), drawn from
// this many bosses per area.
constexpr std::uint32_t kBossOdds[3] = {0, 40, 10};
constexpr std::size_t kBossesPerArea[3] = {0, 1, 2};
// NpcThinkParam goal 11000 does nothing (a part of something bigger, a
// decoration); such kinds stay out of the whole game's pool.
constexpr std::int32_t kGoalNothing = 11000;

int model_mb(const std::string& model) {
    const int n = std::atoi(model.c_str() + 1);
    for (const auto& [m, mb] : kModelMb) {
        if (m == n) return mb;
    }
    return kModelMbDefault;
}

template <typename T>
T rd(const std::vector<std::uint8_t>& b, std::size_t at) {
    T v{};
    if (at + sizeof(T) <= b.size()) std::memcpy(&v, b.data() + at, sizeof(T));
    return v;
}
template <typename T>
void wr(std::vector<std::uint8_t>& b, std::size_t at, T v) {
    if (at + sizeof(T) <= b.size()) std::memcpy(b.data() + at, &v, sizeof(T));
}
template <typename T>
void put(std::vector<std::uint8_t>& b, T v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void pad_to(std::vector<std::uint8_t>& b, std::size_t align) {
    b.resize((b.size() + align - 1) / align * align, 0);
}

// The entry offsets of the MSB's four lists (MODEL, EVENT, POINT, PARTS).
bool msb_lists(const std::vector<std::uint8_t>& b, std::vector<std::uint64_t> lists[4]) {
    if (b.size() < 0x20 || std::memcmp(b.data(), "MSB ", 4) != 0) return false;
    std::uint64_t at = 0x10;
    for (int l = 0; l < 4; ++l) {
        const auto count1 = rd<std::int32_t>(b, at + 4);
        if (count1 < 1 || count1 > 200000 || at + 16 + static_cast<std::uint64_t>(count1) * 8 > b.size()) return false;
        for (int i = 0; i < count1 - 1; ++i) lists[l].push_back(rd<std::uint64_t>(b, at + 16 + static_cast<std::uint64_t>(i) * 8));
        at = rd<std::uint64_t>(b, at + 16 + static_cast<std::uint64_t>(count1 - 1) * 8);
        if (l < 3 && (at == 0 || at >= b.size())) return false;
    }
    return true;
}

std::string utf16_at(const std::vector<std::uint8_t>& b, std::size_t at, std::size_t max = 64) {
    std::string s;
    for (; at + 1 < b.size() && s.size() < max; at += 2) {
        const std::uint16_t c = rd<std::uint16_t>(b, at);
        if (!c) break;
        s += static_cast<char>(c < 0x80 ? c : '?');
    }
    return s;
}

// Text as the game's files keep it: UTF-16, ended by a zero.
std::vector<std::uint8_t> utf16z(const std::string& s) {
    std::vector<std::uint8_t> out;
    for (const char c : s) {
        out.push_back(static_cast<std::uint8_t>(c));
        out.push_back(0);
    }
    out.push_back(0);
    out.push_back(0);
    return out;
}

// The layout with models added to its model list - each after the last model
// of its type (enemies 2, c0000 4), as the game orders the list - and every
// part's model index past an insertion moved up. The lists are laid out
// again as the game lays them out; with no models to add the result is the
// input, which write_layouts_locked checks first.
bool msb_add_models(std::vector<std::uint8_t>& b, const std::vector<std::string>& names) {
    struct List {
        std::int32_t version = 0;
        std::vector<std::uint8_t> name;  // UTF-16, without its end
        std::vector<std::vector<std::uint8_t>> entries;
    };
    std::vector<List> lists;
    if (b.size() < 0x20 || std::memcmp(b.data(), "MSB ", 4) != 0) return false;
    for (std::uint64_t at = 0x10;;) {
        if (at + 16 > b.size() || lists.size() >= 16) return false;
        List l;
        l.version = rd<std::int32_t>(b, at);
        const auto count1 = rd<std::int32_t>(b, at + 4);
        const auto name_at = rd<std::uint64_t>(b, at + 8);
        if (count1 < 1 || count1 > 200000 || at + 16 + static_cast<std::uint64_t>(count1) * 8 > b.size()) return false;
        std::vector<std::uint64_t> offs(static_cast<std::size_t>(count1));
        for (std::size_t i = 0; i < offs.size(); ++i) offs[i] = rd<std::uint64_t>(b, at + 16 + i * 8);
        for (std::uint64_t n = name_at; n + 1 < b.size() && (b[n] || b[n + 1]); n += 2) l.name.insert(l.name.end(), {b[n], b[n + 1]});
        const std::uint64_t next = offs.back();
        for (std::size_t i = 0; i + 1 < offs.size(); ++i) {
            const std::uint64_t end = i + 2 < offs.size() ? offs[i + 1] : (next ? next : b.size());
            if (offs[i] > end || end > b.size()) return false;
            l.entries.emplace_back(b.begin() + static_cast<std::ptrdiff_t>(offs[i]), b.begin() + static_cast<std::ptrdiff_t>(end));
        }
        lists.push_back(std::move(l));
        if (!next) break;
        at = next;
    }
    if (lists.size() < 4) return false;
    auto& models = lists[0].entries;
    auto& parts = lists[3].entries;
    for (const std::string& name : names) {
        const std::int32_t type = name == "c0000" ? 4 : 2;
        std::size_t at = 0;
        std::int32_t type_index = 0;
        for (std::size_t i = 0; i < models.size(); ++i) {
            const auto t = rd<std::int32_t>(models[i], 0x08);
            if (t <= type) at = i + 1;
            if (t == type) ++type_index;
        }
        const std::vector<std::uint8_t> nb = utf16z(name);
        const std::vector<std::uint8_t> sb = utf16z("N:\\SPRJ\\data\\Model\\chr\\" + name + "\\sib\\" + name + (type == 4 ? ".SIB" : ".sib"));
        std::vector<std::uint8_t> e;
        put<std::int64_t>(e, 0x28);
        put<std::int32_t>(e, type);
        put<std::int32_t>(e, type_index);
        put<std::int64_t>(e, static_cast<std::int64_t>(0x28 + nb.size()));
        e.resize(0x28, 0);  // instances (counted again later) and padding
        e.insert(e.end(), nb.begin(), nb.end());
        e.insert(e.end(), sb.begin(), sb.end());
        pad_to(e, 8);
        models.insert(models.begin() + static_cast<std::ptrdiff_t>(at), std::move(e));
        for (auto& p : parts) {
            const auto m = rd<std::int32_t>(p, 0x1c);
            if (m >= static_cast<std::int32_t>(at)) wr<std::int32_t>(p, 0x1c, m + 1);
        }
    }
    std::vector<std::uint8_t> out(b.begin(), b.begin() + 0x10);
    for (std::size_t li = 0; li < lists.size(); ++li) {
        const List& l = lists[li];
        const std::size_t base = out.size(), count1 = l.entries.size() + 1;
        const std::size_t name_at = base + 16 + 8 * count1;
        std::size_t pos = (name_at + l.name.size() + 2 + 7) & ~std::size_t(7);
        put<std::int32_t>(out, l.version);
        put<std::int32_t>(out, static_cast<std::int32_t>(count1));
        put<std::int64_t>(out, static_cast<std::int64_t>(name_at));
        for (const auto& e : l.entries) {
            put<std::int64_t>(out, static_cast<std::int64_t>(pos));
            pos += (e.size() + 7) & ~std::size_t(7);
        }
        put<std::int64_t>(out, li + 1 == lists.size() ? 0 : static_cast<std::int64_t>(pos));
        out.insert(out.end(), l.name.begin(), l.name.end());
        out.insert(out.end(), {0, 0});
        pad_to(out, 8);
        for (const auto& e : l.entries) {
            out.insert(out.end(), e.begin(), e.end());
            pad_to(out, 8);
        }
    }
    b.swap(out);
    return true;
}

// What an enemy part is to the shuffle: a regular enemy (a c0000 hunter is
// kHuman), a boss, or something it leaves alone.
enum class Part { kKeep, kRegular, kHuman, kBoss };
Part part_class(const std::string& model, std::int32_t npc_id, std::int32_t talk) {
    const auto* npc = npc_id > 0 ? g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(npc_id)) : nullptr;
    if (!npc || talk > 0 || npc->teamType == 26) return Part::kKeep;  // NPCs, friends, talkers
    if (model.size() != 5 || model[0] != 'c' || model[1] == '8' || model[1] == '9') return Part::kKeep;  // dolls, messengers
    if (npc->npcType == 1) return Part::kBoss;
    if (npc->npcType != 0) return Part::kKeep;
    if (model == "c2570" || model == "c4030") return Part::kKeep;  // Celestial Emissary, Living Failures: boss fights
    return model == "c0000" ? Part::kHuman : Part::kRegular;
}

// --- AI programs: what a kind from elsewhere needs in its new area ----------
//
// An area's AI is compiled Lua 5.0, in script/<area>_00_00.luabnd (a BND4),
// with two lists beside the programs: <area>.luainfo, the goals the engine
// can start (an id, battle or logic, the program's name), and <area>.luagnl,
// every global the programs define. aiCommon.luabnd has the shared ones and
// is loaded everywhere. A kind's NpcThinkParam row names a battle and a logic
// goal; one registered in neither aiCommon nor the area's bundle never
// starts, and the enemy stands where it was put. (aiCommon's battle goals
// are used by enemies of a dozen areas none of whose bundles has them, so
// the engine looks there too.)
//
// So once enemies are shuffled, every story bundle carries the programs of
// every kind that can be put anywhere - the whole pool's, not one seed's,
// written once a session. The game keeps a bundle it holds (the area the
// player stands in, across a death or a co-op summons into it) and does not
// read the file again, so what it holds must serve whatever the next layout
// brings. Each program goes in once (a program is byte for byte the same in
// every area that has one), after the bundle's own (extended_bundle), with
// its goals added to the list and its globals to the other, and a program
// that calls a function another program of its home bundle defines brings
// that one too (400000 and 400001). The largest story area then has ~2.8 MB
// of programs, under the ~3.3 MB a chalice dungeon loads; aiCommon and the
// chalice bundle are left as they are.

struct BndFile {
    std::uint32_t flags = 0;
    std::int32_t id = 0;
    std::vector<std::uint8_t> name;  // UTF-16 without its end, as the bundle has it
    std::string base;                // the name after its last '\': "126000_battle.lua"
    std::vector<std::uint8_t> data;
};
struct Bnd {
    std::vector<std::uint8_t> head;  // the 0x40 header
    std::vector<BndFile> files;
};

bool ends_with(const std::string& s, const char* tail) {
    const std::size_t n = std::strlen(tail);
    return s.size() >= n && s.compare(s.size() - n, n, tail) == 0;
}

// A BND4 as the luabnds have it: 0x24-byte entries, UTF-16 names, no hashes.
bool bnd_read(const std::vector<std::uint8_t>& b, Bnd& out) {
    if (b.size() < 0x40 || std::memcmp(b.data(), "BND4", 4) != 0 || rd<std::uint64_t>(b, 0x20) != 0x24) return false;
    const auto count = rd<std::uint32_t>(b, 0x0c);
    if (count > 100000 || 0x40 + static_cast<std::uint64_t>(count) * 0x24 > b.size()) return false;
    out.head.assign(b.begin(), b.begin() + 0x40);
    out.files.clear();
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t e = 0x40 + static_cast<std::size_t>(i) * 0x24;
        BndFile f;
        f.flags = rd<std::uint32_t>(b, e);
        const auto size = rd<std::uint64_t>(b, e + 0x10);
        const auto at = rd<std::uint32_t>(b, e + 0x18);
        f.id = rd<std::int32_t>(b, e + 0x1c);
        if (at > b.size() || size > b.size() - at) return false;
        for (std::size_t n = rd<std::uint32_t>(b, e + 0x20); n + 1 < b.size() && (b[n] || b[n + 1]); n += 2) {
            f.name.insert(f.name.end(), {b[n], b[n + 1]});
            const char c = b[n + 1] ? '?' : static_cast<char>(b[n]);
            if (c == '\\' || c == '/') f.base.clear();
            else f.base += c;
        }
        f.data.assign(b.begin() + at, b.begin() + static_cast<std::ptrdiff_t>(at + size));
        out.files.push_back(std::move(f));
    }
    return true;
}

std::vector<std::uint8_t> bnd_write(const Bnd& bnd) {
    const std::size_t n = bnd.files.size(), names_at = 0x40 + n * 0x24;
    std::vector<std::uint8_t> names;
    std::vector<std::uint32_t> name_at;
    for (const BndFile& f : bnd.files) {
        name_at.push_back(static_cast<std::uint32_t>(names_at + names.size()));
        names.insert(names.end(), f.name.begin(), f.name.end());
        names.insert(names.end(), {0, 0});
    }
    const std::size_t data_start = names_at + names.size();
    std::vector<std::uint8_t> out(bnd.head);
    wr<std::uint32_t>(out, 0x0c, static_cast<std::uint32_t>(n));
    wr<std::uint64_t>(out, 0x28, data_start);
    std::size_t pos = (data_start + 15) & ~std::size_t(15);
    for (std::size_t i = 0; i < n; ++i) {
        const BndFile& f = bnd.files[i];
        put<std::uint32_t>(out, f.flags);
        put<std::int32_t>(out, -1);
        put<std::uint64_t>(out, f.data.size());
        put<std::uint64_t>(out, f.data.size());
        put<std::uint32_t>(out, static_cast<std::uint32_t>(pos));
        put<std::int32_t>(out, f.id);
        put<std::uint32_t>(out, name_at[i]);
        pos = (pos + f.data.size() + 15) & ~std::size_t(15);
    }
    out.insert(out.end(), names.begin(), names.end());
    pad_to(out, 16);
    for (const BndFile& f : bnd.files) {
        out.insert(out.end(), f.data.begin(), f.data.end());
        pad_to(out, 16);
    }
    return out;
}

// A goal of a .luainfo: its id, whether it is a battle or a logic goal, the
// program's name and, for a logic goal, its interrupt handler's.
struct Goal {
    std::int32_t id = 0;
    std::uint8_t battle = 0, logic = 0;
    std::string name, interrupt;
};

bool info_read(const std::vector<std::uint8_t>& b, std::vector<Goal>& out) {
    if (b.size() < 0x10 || std::memcmp(b.data(), "LUAI", 4) != 0) return false;
    const auto n = rd<std::uint32_t>(b, 8);
    if (0x10 + static_cast<std::uint64_t>(n) * 0x18 > b.size()) return false;
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::size_t at = 0x10 + static_cast<std::size_t>(i) * 0x18;
        Goal goal;
        goal.id = rd<std::int32_t>(b, at);
        goal.battle = b[at + 4];
        goal.logic = b[at + 5];
        goal.name = utf16_at(b, rd<std::uint64_t>(b, at + 8), 256);
        if (const auto intr = rd<std::uint64_t>(b, at + 16)) goal.interrupt = utf16_at(b, intr, 256);
        out.push_back(std::move(goal));
    }
    return true;
}

std::vector<std::uint8_t> info_write(const std::vector<Goal>& goals) {
    std::vector<std::uint8_t> out = {'L', 'U', 'A', 'I'}, strings;
    put<std::uint32_t>(out, 1);
    put<std::uint32_t>(out, static_cast<std::uint32_t>(goals.size()));
    put<std::uint32_t>(out, 0);
    const std::size_t base = 0x10 + goals.size() * 0x18;
    for (const Goal& goal : goals) {
        put<std::int32_t>(out, goal.id);
        out.insert(out.end(), {goal.battle, goal.logic, 0, 0});
        put<std::uint64_t>(out, base + strings.size());
        const auto nb = utf16z(goal.name);
        strings.insert(strings.end(), nb.begin(), nb.end());
        if (goal.interrupt.empty()) {
            put<std::uint64_t>(out, 0);
        } else {
            put<std::uint64_t>(out, base + strings.size());
            const auto ib = utf16z(goal.interrupt);
            strings.insert(strings.end(), ib.begin(), ib.end());
        }
    }
    out.insert(out.end(), strings.begin(), strings.end());
    pad_to(out, 16);
    return out;
}

bool gnl_read(const std::vector<std::uint8_t>& b, std::vector<std::string>& out) {
    for (std::size_t k = 0;; ++k) {
        if ((k + 1) * 8 > b.size()) return false;
        const auto at = rd<std::uint64_t>(b, k * 8);
        if (!at) return true;
        out.push_back(utf16_at(b, at, 256));
    }
}

std::vector<std::uint8_t> gnl_write(const std::vector<std::string>& names) {
    std::vector<std::uint8_t> out, strings;
    const std::size_t base = (names.size() + 1) * 8;
    for (const std::string& s : names) {
        put<std::uint64_t>(out, base + strings.size());
        const auto nb = utf16z(s);
        strings.insert(strings.end(), nb.begin(), nb.end());
    }
    put<std::uint64_t>(out, 0);
    out.insert(out.end(), strings.begin(), strings.end());
    pad_to(out, 16);
    return out;
}

// The globals a compiled Lua 5.0 chunk ("\x1bLuaP") sets and reads, in all of
// its functions: SETGLOBAL and GETGLOBAL of a constant. What a program
// defines is what its bundle's .luagnl lists (in every bundle of the game).
struct LuaGlobals {
    std::set<std::string> sets, gets;
    std::set<std::string> defines;  // set by the chunk itself as it loads: its functions, its constants
};

struct LuaIn {
    const std::vector<std::uint8_t>& b;
    std::size_t at = 0;
    bool ok = true;
    template <typename T>
    T get() {
        T v{};
        if (at + sizeof(T) > b.size()) {
            ok = false;
            return v;
        }
        std::memcpy(&v, b.data() + at, sizeof(T));
        at += sizeof(T);
        return v;
    }
    void skip(std::uint64_t n) {
        if (n > b.size() - at) ok = false;
        else at += n;
    }
    std::string str() {  // a size_t length with its end counted, then the text
        const auto n = get<std::uint64_t>();
        if (!ok || n == 0) return {};
        if (n > b.size() - at) {
            ok = false;
            return {};
        }
        std::string s(reinterpret_cast<const char*>(b.data() + at), n - 1);
        at += n;
        return s;
    }
    std::int32_t count() {
        const auto n = get<std::int32_t>();
        if (n < 0 || static_cast<std::uint64_t>(n) > b.size()) ok = false;
        return ok ? n : 0;
    }
};

bool lua_function(LuaIn& in, LuaGlobals& out, int depth) {
    if (depth > 64) return false;
    in.str();      // source
    in.skip(4 + 4);  // line defined; upvalues, parameters, varargs, stack size
    in.skip(4ull * static_cast<std::uint64_t>(in.count()));  // line info
    for (std::int32_t i = 0, n = in.count(); i < n && in.ok; ++i) {  // locals
        in.str();
        in.skip(8);
    }
    for (std::int32_t i = 0, n = in.count(); i < n && in.ok; ++i) in.str();  // upvalue names
    std::vector<std::string> k(static_cast<std::size_t>(in.count()));
    for (std::string& c : k) {
        const auto type = in.get<std::uint8_t>();
        if (type == 3) in.skip(8);  // a number
        else if (type == 4) c = in.str();
        else if (type != 0) return false;  // nil is the only other constant
        if (!in.ok) return false;
    }
    for (std::int32_t i = 0, n = in.count(); i < n && in.ok; ++i) {
        if (!lua_function(in, out, depth + 1)) return false;
    }
    for (std::int32_t i = 0, n = in.count(); i < n && in.ok; ++i) {
        const auto ins = in.get<std::uint32_t>();
        const std::uint32_t op = ins & 0x3f, bx = (ins >> 6) & 0x3ffff;
        if ((op == 5 || op == 7) && bx < k.size() && !k[bx].empty()) (op == 5 ? out.gets : out.sets).insert(k[bx]);
        if (op == 7 && depth == 0 && bx < k.size() && !k[bx].empty()) out.defines.insert(k[bx]);
    }
    return in.ok;
}

bool lua_globals(const std::vector<std::uint8_t>& b, LuaGlobals& out) {
    if (b.size() < 22 || std::memcmp(b.data(), "\x1bLuaP", 5) != 0) return false;
    LuaIn in{b};
    in.at = 22;  // the header: sizes and a test number
    return lua_function(in, out, 0) && in.at == b.size();
}

// --- the whole game's enemies ----------------------------------------------

// A kind of enemy: what a part becomes, as one set.
struct Kind {
    std::string model;  // "c1260"
    std::int32_t npc = 0, think = 0, chara_init = -1;
    std::string area;   // the area it is first found in ("m27_00_00")
    int count = 0;      // its parts in the game's layouts
    bool human = false, boss = false;
    bool usable = false;  // may be put anywhere: it fights, and its programs were found
};

// Every kind of the story layouts as the game has them, what each area has
// of its own, and the AI bundles: what the whole game's pool and the roaming
// bosses are drawn from. Made on the first world that needs it.
struct Catalog {
    bool built = false;
    std::vector<Kind> kinds;
    std::map<std::string, std::set<std::string>> area_models;  // area -> its regular slots' models (not c0000)
    std::set<std::string> hunter_areas;                        // areas with c0000 hunters
    std::map<std::string, Bnd> bundles;                        // "m24_01_00_00" -> its AI bundle
    std::set<std::tuple<std::int32_t, int, int>> common_goals;  // aiCommon's (id, battle, logic)
    std::set<std::string> common_sets;                         // the globals aiCommon defines
    std::map<std::string, LuaGlobals> globals;                 // "<bundle>/<program>" -> what it sets and reads
    std::set<std::string> helpers;                             // globals three or more programs define as they load
    std::map<std::string, std::set<std::string>> bundle_programs;  // the programs each written bundle has
};
Catalog g_catalog;  // under g_layout_mu

std::string bundle_of(const std::string& area) { return area.substr(0, 6) + "_00_00"; }

const BndFile* bundle_file(const std::string& bundle, const std::string& base) {
    auto it = g_catalog.bundles.find(bundle);
    if (it == g_catalog.bundles.end()) return nullptr;
    for (const BndFile& f : it->second.files) {
        if (f.base == base) return &f;
    }
    return nullptr;
}

// The programs a kind's AI needs that aiCommon does not have: "<id>_battle.lua"
// and "<id>_logic.lua" for the goals of its part's think row (`need`), and
// for the default think row its NpcParam row names (aiThinkId; `extra`,
// carried when found), should the part's give way to it. False when the
// part's think row is missing.
bool kind_programs(const Kind& k, std::vector<std::string>& need, std::vector<std::string>* extra = nullptr) {
    const auto* think = k.think > 0 ? g.param<bb::params::NpcThinkParam>(static_cast<std::uint32_t>(k.think)) : nullptr;
    if (!think) return false;
    const auto* npc = g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(k.npc));
    const auto* fallback = extra && npc && npc->aiThinkId > 0 && npc->aiThinkId != k.think
                               ? g.param<bb::params::NpcThinkParam>(static_cast<std::uint32_t>(npc->aiThinkId))
                               : nullptr;
    for (const auto* row : {think, fallback}) {
        if (!row) continue;
        std::vector<std::string>& out = row == think ? need : *extra;
        for (const auto& [goal, battle] : {std::pair<std::int32_t, bool>{row->battleGoalID, true}, {row->logicId, false}}) {
            if (goal <= 0 || g_catalog.common_goals.count({goal, battle ? 1 : 0, battle ? 0 : 1})) continue;
            char name[32];
            std::snprintf(name, sizeof(name), "%06d_%s.lua", goal, battle ? "battle" : "logic");
            if (std::find(out.begin(), out.end(), name) == out.end()) out.push_back(name);
        }
    }
    return true;
}

// Where a program is: the kind's own area's bundle first, then any story
// bundle (they are the same everywhere). Empty when no bundle has it.
std::string program_home(const std::string& base, const std::string& area) {
    if (bundle_file(bundle_of(area), base)) return bundle_of(area);
    for (const auto& [bundle, bnd] : g_catalog.bundles) {
        if (bundle_file(bundle, base)) return bundle;
    }
    return {};
}

// With g_layout_mu held.
void build_catalog_locked() {
    if (g_catalog.built) return;
    g_catalog.built = true;
    auto read = [](const std::string& path, std::vector<std::uint8_t>& out) {
        const void* data = nullptr;
        std::size_t size = 0;
        if (g.api->game_file(path.c_str(), &data, &size) != 0) return false;
        out.assign(static_cast<const std::uint8_t*>(data), static_cast<const std::uint8_t*>(data) + size);
        return true;
    };
    // aiCommon's goals and globals: what every area has.
    std::vector<std::uint8_t> raw;
    Bnd common;
    if (!read("/dvdroot_ps4/script/aicommon.luabnd.dcx", raw) || !bnd_read(raw, common)) {
        g.log("randomizer: aiCommon.luabnd is not a bundle this reads; enemies stay in their own areas");
        return;
    }
    for (const BndFile& f : common.files) {
        std::vector<Goal> goals;
        LuaGlobals lg;
        if (ends_with(f.base, ".luainfo") && info_read(f.data, goals)) {
            for (const Goal& goal : goals) g_catalog.common_goals.insert({goal.id, goal.battle, goal.logic});
        } else if (ends_with(f.base, ".lua") && lua_globals(f.data, lg)) {
            g_catalog.common_sets.insert(lg.sets.begin(), lg.sets.end());
        }
    }
    // The story areas' bundles, kept as the game has them.
    int unread = 0;  // programs whose code this could not follow
    for (const char* map : kMaps) {
        const std::string bundle = bundle_of(map);
        if (g_catalog.bundles.count(bundle)) continue;
        Bnd bnd;
        if (!read("/dvdroot_ps4/script/" + bundle + ".luabnd.dcx", raw) || !bnd_read(raw, bnd)) continue;  // m21_01 has none
        // what is rewritten must rebuild exactly as it was: the bundle, its lists
        bool exact = bnd_write(bnd) == raw;
        for (const BndFile& f : bnd.files) {
            std::vector<Goal> goals;
            std::vector<std::string> names;
            if (ends_with(f.base, ".luainfo")) exact = exact && info_read(f.data, goals) && info_write(goals) == f.data;
            if (ends_with(f.base, ".luagnl")) exact = exact && gnl_read(f.data, names) && gnl_write(names) == f.data;
        }
        if (!exact) {
            g.log("randomizer: %s.luabnd does not rebuild as it was; left as it is", bundle.c_str());
            continue;
        }
        for (const BndFile& f : bnd.files) {
            LuaGlobals lg;
            if (!ends_with(f.base, ".lua")) continue;
            if (lua_globals(f.data, lg)) g_catalog.globals[bundle + "/" + f.base] = std::move(lg);
            else ++unread;
        }
        g_catalog.bundles[bundle] = std::move(bnd);
    }
    // Globals three or more programs define as they load: copied helpers
    // (GetWellSpace_Odds), the same in each.
    std::map<std::string, std::set<std::string>> definers;
    for (const auto& [key, lg] : g_catalog.globals) {
        for (const std::string& n : lg.defines) definers[n].insert(key.substr(key.find('/') + 1));
    }
    for (const auto& [n, by] : definers) {
        if (by.size() >= 3) g_catalog.helpers.insert(n);
    }
    // Every kind of the layouts.
    std::map<std::tuple<std::string, std::int32_t, std::int32_t, std::int32_t>, std::size_t> index;
    for (const char* map : kMaps) {
        std::vector<std::uint64_t> lists[4];
        if (!read(std::string("/dvdroot_ps4/map/mapstudio/") + map + ".msb.dcx", raw) || !msb_lists(raw, lists)) continue;
        std::vector<std::string> models;
        for (std::uint64_t m : lists[0]) models.push_back(utf16_at(raw, m + rd<std::uint64_t>(raw, m)));
        const std::string area = std::string(map).substr(0, 9);
        for (std::uint64_t part : lists[3]) {
            if (rd<std::int32_t>(raw, part + 0x14) != 2) continue;
            const std::uint64_t data = part + rd<std::uint64_t>(raw, part + 0xb8);
            const auto mi = rd<std::int32_t>(raw, part + 0x1c);
            if (mi < 0 || static_cast<std::size_t>(mi) >= models.size()) continue;
            const std::string& model = models[static_cast<std::size_t>(mi)];
            const auto npc = rd<std::int32_t>(raw, data + 0x0c);
            const Part cls = part_class(model, npc, rd<std::int32_t>(raw, data + 0x10));
            if (cls == Part::kKeep) continue;
            if (cls == Part::kRegular) g_catalog.area_models[area].insert(model);
            if (cls == Part::kHuman) g_catalog.hunter_areas.insert(area);
            const auto key = std::make_tuple(model, npc, rd<std::int32_t>(raw, data + 0x08), rd<std::int32_t>(raw, data + 0x18));
            auto [it, fresh] = index.emplace(key, g_catalog.kinds.size());
            if (fresh) {
                Kind k;
                k.model = model;
                k.npc = npc;
                k.think = std::get<2>(key);
                k.chara_init = std::get<3>(key);
                k.area = area;
                k.human = cls == Part::kHuman;
                k.boss = cls == Part::kBoss;
                g_catalog.kinds.push_back(k);
            }
            ++g_catalog.kinds[it->second].count;
        }
    }
    // Which may be put anywhere: a fighter (not a do-nothing part, not an
    // unkillable one: hp 999 or 99999), on the roaming list if a boss, its
    // programs found.
    int usable = 0, bosses = 0;
    for (Kind& k : g_catalog.kinds) {
        if (k.boss && std::find(std::begin(kRoamingBosses), std::end(kRoamingBosses), k.npc) == std::end(kRoamingBosses)) continue;
        const auto* npc = g.param<bb::params::NpcParam>(static_cast<std::uint32_t>(k.npc));
        const auto* think = k.think > 0 ? g.param<bb::params::NpcThinkParam>(static_cast<std::uint32_t>(k.think)) : nullptr;
        if (!npc || !think || think->battleGoalID == kGoalNothing || think->logicId == kGoalNothing) continue;
        if (!k.boss && (npc->hp == 999 || npc->hp >= 99999)) continue;  // the game's "cannot be killed"
        std::vector<std::string> programs;
        if (!kind_programs(k, programs)) continue;
        k.usable = std::all_of(programs.begin(), programs.end(), [&](const std::string& p) { return !program_home(p, k.area).empty(); });
        if (k.usable) ++(k.boss ? bosses : usable);
    }
    g.log("randomizer: %zu kinds of enemy in the story maps, %d of them can go anywhere, %d roaming bosses; %zu AI bundles%s",
          g_catalog.kinds.size(), usable, bosses, g_catalog.bundles.size(),
          unread ? (", " + std::to_string(unread) + " programs unread").c_str() : "");
}

// The programs every story bundle carries: every kind that can go anywhere -
// the whole game's pool with its hunters, and the roaming bosses - its battle
// and logic programs, and what they call in their home bundles: program ->
// the bundle it is taken from. The same for every seed and setting.
std::map<std::string, std::string> pool_programs() {
    std::map<std::string, std::string> out;
    std::vector<std::pair<std::string, std::string>> todo;
    for (const Kind& k : g_catalog.kinds) {
        if (!k.usable) continue;
        std::vector<std::string> programs, extra;
        kind_programs(k, programs, &extra);
        programs.insert(programs.end(), extra.begin(), extra.end());
        for (const std::string& p : programs) {
            if (out.count(p)) continue;
            const std::string home = program_home(p, k.area);
            if (home.empty()) continue;  // an aiThinkId default no bundle has
            out[p] = home;
            todo.emplace_back(p, home);
        }
    }
    while (!todo.empty()) {
        const auto [program, bundle] = todo.back();
        todo.pop_back();
        const LuaGlobals& lg = g_catalog.globals[bundle + "/" + program];
        for (const std::string& name : lg.gets) {
            if (lg.sets.count(name) || g_catalog.common_sets.count(name)) continue;
            // defined by exactly one other program of the same bundle: that one comes too
            std::string definer;
            int definers = 0;
            for (const BndFile& f : g_catalog.bundles[bundle].files) {
                auto it = g_catalog.globals.find(bundle + "/" + f.base);
                if (f.base != program && it != g_catalog.globals.end() && it->second.sets.count(name)) {
                    definer = f.base;
                    ++definers;
                }
            }
            if (definers == 1 && !out.count(definer)) {
                out[definer] = bundle;
                todo.emplace_back(definer, bundle);
            }
        }
    }
    return out;
}

// A story bundle with the programs it lacks added, their goals registered and
// their globals listed. They go after the bundle's own, which keep their
// order and numbers (with the added ones first, the area's own enemies lost
// their AI in a test). A bundle's programs run in order and the last
// definition of a global stands, and a few functions have a variant in
// another program (OnIf_210000 in 210000 and 210020): a program that would
// define again what one of the bundle's own defines as it loads is left out
// of that bundle - its kinds are not put in that area (AreaPlan) - unless the
// name is a helper three or more programs share. `present`: the programs the
// bundle ends up with.
std::vector<std::uint8_t> extended_bundle(const std::string& bundle, const std::map<std::string, std::string>& programs,
                                          std::set<std::string>& present) {
    Bnd bnd = g_catalog.bundles[bundle];
    BndFile *info = nullptr, *gnl = nullptr;
    std::set<std::string> own_defines;
    std::int32_t next_id = 0;
    for (BndFile& f : bnd.files) {
        if (ends_with(f.base, ".luainfo")) info = &f;
        if (ends_with(f.base, ".luagnl")) gnl = &f;
        if (ends_with(f.base, ".lua")) {
            present.insert(f.base);
            const LuaGlobals& lg = g_catalog.globals[bundle + "/" + f.base];
            own_defines.insert(lg.defines.begin(), lg.defines.end());
        }
        next_id = std::max(next_id, f.id + 1);
    }
    std::vector<Goal> goals;
    std::vector<std::string> names;
    if (!info || !gnl || !info_read(info->data, goals) || !gnl_read(gnl->data, names)) return bnd_write(bnd);
    std::set<std::tuple<std::int32_t, int, int>> registered;
    for (const Goal& goal : goals) registered.insert({goal.id, goal.battle, goal.logic});
    std::set<std::string> listed(names.begin(), names.end());
    std::vector<BndFile> added;
    for (const auto& [program, from] : programs) {
        if (present.count(program)) continue;
        const BndFile* src = bundle_file(from, program);
        if (!src) continue;
        const LuaGlobals& lg = g_catalog.globals[from + "/" + program];
        if (std::any_of(lg.defines.begin(), lg.defines.end(),
                        [&](const std::string& n) { return own_defines.count(n) && !g_catalog.helpers.count(n); }))
            continue;
        BndFile f = *src;
        f.id = next_id++;
        added.push_back(std::move(f));
        present.insert(program);
        // its goals: the source list's entries named for a function it defines
        std::vector<Goal> theirs;
        for (const BndFile& sf : g_catalog.bundles[from].files) {
            if (ends_with(sf.base, ".luainfo")) info_read(sf.data, theirs);
        }
        for (const Goal& goal : theirs) {
            bool its = lg.sets.count(goal.name) > 0;
            for (auto it = lg.sets.lower_bound(goal.name + "_"); !its && it != lg.sets.end() && it->compare(0, goal.name.size() + 1, goal.name + "_") == 0; ++it) its = true;
            if (its && registered.insert({goal.id, goal.battle, goal.logic}).second) goals.push_back(goal);
        }
        for (const std::string& name : lg.sets) {
            if (listed.insert(name).second) names.push_back(name);
        }
    }
    info->data = info_write(goals);
    gnl->data = gnl_write(names);
    bnd.files.insert(bnd.files.end(), added.begin(), added.end());
    return bnd_write(bnd);
}

// Every story bundle with the pool's programs, into the overlay: once, on the
// first world with shuffled enemies (the player's own or a host's), and kept
// for the session - the programs only serve kinds a layout puts there, so a
// vanilla world loses nothing by them, and switching settings or worlds later
// finds them already in the bundle the game holds for the area it stands in.
bool g_bundles_written = false;  // under g_layout_mu
void write_bundles_locked() {
    if (g_bundles_written || g_catalog.bundles.empty()) return;
    g_bundles_written = true;
    const std::map<std::string, std::string> programs = pool_programs();
    std::size_t bytes = 0;
    for (const auto& [bundle, bnd] : g_catalog.bundles) {
        const std::vector<std::uint8_t> out = extended_bundle(bundle, programs, g_catalog.bundle_programs[bundle]);
        bytes = std::max(bytes, out.size());
        if (g.api->overlay_file("randomizer", ("/dvdroot_ps4/script/" + bundle + ".luabnd.dcx").c_str(), out.data(), out.size()) != 0) {
            g.log("randomizer: could not write %s.luabnd", bundle.c_str());
        }
    }
    g.log("randomizer: AI bundles with the programs of every kind that can go anywhere (%zu programs, the largest bundle %zu KB)",
          programs.size(), bytes / 1024);
}

// What one area's enemies are drawn from with the whole game's pool or
// roaming bosses - the same for each of the area's state variants, so a slot
// gets the same kind in all of them.
struct AreaPlan {
    std::vector<std::string> models;          // whole game: the models drawn from
    std::map<std::string, std::string> found;  // whole game, "like the original": own model -> its new one
    std::vector<std::size_t> bosses;           // the roaming bosses' kinds (catalog indices)
    std::map<std::string, std::vector<std::size_t>> kinds;  // model -> its kinds that can go anywhere
};

// Whether a kind's AI is in an area once the bundles are written: each of
// its programs in the area's bundle (aiCommon's goals need none).
bool kind_fits(const Kind& k, const std::string& area) {
    std::vector<std::string> need;
    if (!kind_programs(k, need)) return false;
    const auto it = g_catalog.bundle_programs.find(bundle_of(area));
    return std::all_of(need.begin(), need.end(), [&](const std::string& p) { return it != g_catalog.bundle_programs.end() && it->second.count(p); });
}

AreaPlan area_plan(const World& w, const std::string& area) {
    AreaPlan plan;
    std::set<std::string> own = g_catalog.area_models[area];
    if (w.npcs && g_catalog.hunter_areas.count(area)) own.insert("c0000");
    int used = 0;
    for (const std::string& m : own) used += model_mb(m);
    const int budget = std::max(kAreaModelBudgetMb, used);
    for (std::size_t i = 0; i < g_catalog.kinds.size(); ++i) {
        const Kind& k = g_catalog.kinds[i];
        if (k.usable && !k.boss && (!k.human || w.npcs) && kind_fits(k, area)) plan.kinds[k.model].push_back(i);
    }
    if (w.anywhere) {
        // as many models as the area has of its own, from the whole game's
        std::vector<std::string> candidates;
        for (const auto& [model, kinds] : plan.kinds) candidates.push_back(model);
        bb::Rng rng(bb::seed_of(w.token + "|" + area + "|models"));
        rng.shuffle(candidates);
        used = 0;
        for (const std::string& m : candidates) {
            if (plan.models.size() >= own.size()) break;
            if (used + model_mb(m) > budget) continue;
            plan.models.push_back(m);
            used += model_mb(m);
        }
        if (!w.even && !plan.models.empty()) {
            std::vector<std::string> to = plan.models;
            bb::Rng shuffle(bb::seed_of(w.token + "|" + area + "|found"));
            shuffle.shuffle(to);
            std::size_t i = 0;
            for (const std::string& m : own) plan.found[m] = to[i++ % to.size()];
        }
    }
    if (w.bosses) {
        std::map<std::string, std::vector<std::size_t>> by_model;
        for (std::size_t i = 0; i < g_catalog.kinds.size(); ++i) {
            if (g_catalog.kinds[i].usable && g_catalog.kinds[i].boss && kind_fits(g_catalog.kinds[i], area)) by_model[g_catalog.kinds[i].model].push_back(i);
        }
        std::vector<std::string> candidates;
        for (const auto& [model, kinds] : by_model) candidates.push_back(model);
        bb::Rng rng(bb::seed_of(w.token + "|" + area + "|bosses"));
        rng.shuffle(candidates);
        std::size_t taken = 0;
        for (const std::string& m : candidates) {
            if (taken >= kBossesPerArea[w.bosses]) break;
            if (used + model_mb(m) > budget) continue;
            used += model_mb(m);
            ++taken;
            plan.bosses.insert(plan.bosses.end(), by_model[m].begin(), by_model[m].end());
        }
    }
    return plan;
}

struct EnemyKind {
    std::int32_t model, npc, think, chara_init;
    bool operator<(const EnemyKind& o) const {
        return std::tie(model, npc, think, chara_init) < std::tie(o.model, o.npc, o.think, o.chara_init);
    }
};

// Rewrites one layout; the number of enemies changed. `plan`: the area's,
// with the whole game's pool or roaming bosses (nullptr: its own kinds only).
int randomize_layout(std::vector<std::uint8_t>& b, const std::string& map, const World& w, const AreaPlan* plan) {
    std::vector<std::uint64_t> lists[4];
    if (!msb_lists(b, lists)) return -1;
    std::vector<std::string> model_names;
    for (std::uint64_t m : lists[0]) model_names.push_back(utf16_at(b, m + rd<std::uint64_t>(b, m)));
    struct Slot {
        std::size_t part;  // its index in the parts list
        std::string name, model;
    };
    std::vector<Slot> slots;
    std::vector<EnemyKind> found;  // one per slot, for "as found"
    std::set<EnemyKind> kinds;
    for (std::size_t i = 0; i < lists[3].size(); ++i) {
        const std::uint64_t part = lists[3][i];
        if (rd<std::int32_t>(b, part + 0x14) != 2) continue;  // enemies only
        const std::uint64_t data = part + rd<std::uint64_t>(b, part + 0xb8);
        const auto model = rd<std::int32_t>(b, part + 0x1c);
        if (model < 0 || static_cast<std::size_t>(model) >= model_names.size()) continue;
        const std::string& mname = model_names[static_cast<std::size_t>(model)];
        const Part cls = part_class(mname, rd<std::int32_t>(b, data + 0x0c), rd<std::int32_t>(b, data + 0x10));
        if (cls != Part::kRegular && !(cls == Part::kHuman && w.npcs)) continue;  // bosses, NPCs, talkers; hunters unless asked
        const EnemyKind k{model, rd<std::int32_t>(b, data + 0x0c), rd<std::int32_t>(b, data + 0x08), rd<std::int32_t>(b, data + 0x18)};
        slots.push_back({i, utf16_at(b, part + rd<std::uint64_t>(b, part + 0x08)), mname});
        found.push_back(k);
        kinds.insert(k);
    }
    if (slots.empty() || (!plan && kinds.size() < 2)) return 0;
    const std::vector<EnemyKind> pool_even(kinds.begin(), kinds.end());
    const std::vector<EnemyKind>& pool = w.even ? pool_even : found;
    // A map's state variants (m24_01_00_00, _01, _11) share their slots' names:
    // a slot's pick is keyed by the area and its name, so they agree.
    const std::string area = map.substr(0, 9);
    struct Pick {
        std::string model;
        std::int32_t npc, think, chara_init;
    };
    std::vector<Pick> picks;
    for (const Slot& s : slots) {
        if (!plan) {
            bb::Rng rng(bb::seed_of(w.token + "|" + area + "|" + s.name));
            const EnemyKind& k = pool[rng.below(static_cast<std::uint32_t>(pool.size()))];
            picks.push_back({model_names[static_cast<std::size_t>(k.model)], k.npc, k.think, k.chara_init});
            continue;
        }
        bb::Rng rng(bb::seed_of(w.token + "|" + area + "|" + s.name + "|wild"));
        const Kind* k = nullptr;
        if (!plan->bosses.empty() && rng.below(kBossOdds[w.bosses]) == 0) {
            k = &g_catalog.kinds[plan->bosses[rng.below(static_cast<std::uint32_t>(plan->bosses.size()))]];
        } else if (w.anywhere && !plan->models.empty()) {
            auto f = plan->found.find(s.model);
            const std::string& model = !w.even && f != plan->found.end() ? f->second : plan->models[rng.below(static_cast<std::uint32_t>(plan->models.size()))];
            // one of the model's kinds, the common ones more often
            const auto at = plan->kinds.find(model);
            static const std::vector<std::size_t> none;
            const std::vector<std::size_t>& of = at == plan->kinds.end() ? none : at->second;
            std::uint32_t total = 0;
            for (std::size_t i : of) total += static_cast<std::uint32_t>(g_catalog.kinds[i].count);
            std::uint32_t r = rng.below(total);
            for (std::size_t i : of) {
                k = &g_catalog.kinds[i];
                if (r < static_cast<std::uint32_t>(k->count)) break;
                r -= static_cast<std::uint32_t>(k->count);
            }
        }
        if (k) {
            picks.push_back({k->model, k->npc, k->think, k->chara_init});
        } else {
            const EnemyKind& own = pool[rng.below(static_cast<std::uint32_t>(pool.size()))];
            picks.push_back({model_names[static_cast<std::size_t>(own.model)], own.npc, own.think, own.chara_init});
        }
    }
    // Models from elsewhere into the model list.
    std::vector<std::string> missing;
    for (const Pick& p : picks) {
        if (std::find(model_names.begin(), model_names.end(), p.model) == model_names.end() &&
            std::find(missing.begin(), missing.end(), p.model) == missing.end())
            missing.push_back(p.model);
    }
    if (!missing.empty()) {
        if (!msb_add_models(b, missing)) return -1;
        for (auto& l : lists) l.clear();
        if (!msb_lists(b, lists)) return -1;
        model_names.clear();
        for (std::uint64_t m : lists[0]) model_names.push_back(utf16_at(b, m + rd<std::uint64_t>(b, m)));
    }
    int changed = 0;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const Slot& s = slots[i];
        const Pick& k = picks[i];
        const auto mi = std::find(model_names.begin(), model_names.end(), k.model) - model_names.begin();
        if (static_cast<std::size_t>(mi) >= model_names.size()) continue;
        const std::uint64_t part = lists[3][s.part];
        const std::uint64_t data = part + rd<std::uint64_t>(b, part + 0xb8);
        wr<std::int32_t>(b, part + 0x1c, static_cast<std::int32_t>(mi));
        // The part's name starts with its model's ("c2630_0012"): five
        // characters, rewritten in place to the new model's.
        const std::uint64_t name_at = part + rd<std::uint64_t>(b, part + 0x08);
        if (k.model.size() == 5 && s.name.size() > 5 && s.name[0] == 'c' && s.name[5] == '_') {
            for (int c = 0; c < 5; ++c) wr<std::uint16_t>(b, name_at + static_cast<std::uint64_t>(c) * 2, static_cast<std::uint16_t>(k.model[static_cast<std::size_t>(c)]));
        }
        wr<std::int32_t>(b, data + 0x08, k.think);
        wr<std::int32_t>(b, data + 0x0c, k.npc);
        wr<std::int32_t>(b, data + 0x18, k.chara_init);
        ++changed;
    }
    // Every model's instance count from the parts again.
    std::vector<std::int32_t> counts(lists[0].size(), 0);
    for (std::uint64_t part : lists[3]) {
        const auto m = rd<std::int32_t>(b, part + 0x1c);
        if (m >= 0 && static_cast<std::size_t>(m) < counts.size()) ++counts[static_cast<std::size_t>(m)];
    }
    for (std::size_t i = 0; i < lists[0].size(); ++i) wr<std::int32_t>(b, lists[0][i] + 0x18, counts[i]);
    return changed;
}

// Writes every story layout into the plugin's overlay for world `w` - or as
// the game has it, for an empty token - for the next load of each map
// (bbhost makes a load of a rewritten layout read it again, the map the
// player stands in too), and the AI bundles the world needs. With
// g_layout_mu held.
void write_layouts_locked(const World& w) {
    const bool shuffle = !w.token.empty();
    if (shuffle) {
        build_catalog_locked();
        write_bundles_locked();
    }
    const bool wild = shuffle && (w.anywhere || w.bosses) && !g_catalog.bundles.empty();
    int maps = 0, enemies = 0;
    std::ofstream spoiler;  // the player's own world's; a host's is not theirs to read
    if (const std::string dir = g.api->plugin_dir("randomizer"); !dir.empty() && shuffle && !g_visiting) spoiler.open(dir + "/enemies.txt");
    if (spoiler) spoiler << "enemies for seed " << w.token << "\n";
    std::map<std::string, AreaPlan> plans;
    for (const char* map : kMaps) {
        const std::string path = std::string("/dvdroot_ps4/map/mapstudio/") + map + ".msb.dcx";
        const void* data = nullptr;
        std::size_t size = 0;
        if (g.api->game_file(path.c_str(), &data, &size) != 0) continue;
        std::vector<std::uint8_t> b(static_cast<const std::uint8_t*>(data), static_cast<const std::uint8_t*>(data) + size);
        const AreaPlan* plan = nullptr;
        if (wild) {
            // a layout this cannot lay out again exactly keeps its own kinds
            std::vector<std::uint8_t> again = b;
            if (msb_add_models(again, {}) && again == b) {
                const std::string area = std::string(map).substr(0, 9);
                auto it = plans.find(area);
                if (it == plans.end()) it = plans.emplace(area, area_plan(w, area)).first;
                plan = &it->second;
            } else {
                g.log("randomizer: %s does not lay out again as it was; its enemies stay its own", map);
            }
        }
        const int n = shuffle ? randomize_layout(b, map, w, plan) : 0;
        if (n < 0) {
            g.log("randomizer: %s is not a layout this reads; left as it is", map);
            continue;
        }
        if (g.api->overlay_file("randomizer", path.c_str(), b.data(), b.size()) != 0) {
            g.log("randomizer: could not write %s", map);
            continue;
        }
        ++maps;
        enemies += n;
        if (spoiler) spoiler << map << ": " << n << " enemies\n";
    }
    if (spoiler && !plans.empty()) {
        spoiler << "\n# what each area's enemies are drawn from\n";
        for (const auto& [area, plan] : plans) {
            spoiler << area << ":";
            for (const std::string& m : plan.models) spoiler << ' ' << m;
            if (!plan.bosses.empty()) {
                spoiler << "  bosses:";
                std::set<std::string> seen;
                for (std::size_t i : plan.bosses) {
                    if (seen.insert(g_catalog.kinds[i].model).second) spoiler << ' ' << g_catalog.kinds[i].model;
                }
            }
            spoiler << '\n';
        }
    }
    g_world = w.key();
    g.log("randomizer: %d enemies in %d map layouts%s", enemies, maps, shuffle ? (" (seed " + w.token + ")").c_str() : " (as the game has them)");
}

// A seed token as bbhost sends it (bb::seed_token: [a-z0-9_.-], up to 40):
// the host's comes from the server, so anything else is not a world to build.
bool plain_token(const std::string& t) {
    if (t.empty() || t.size() > 40) return false;
    for (const char ch : t) {
        if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == '-')) return false;
    }
    return true;
}

std::string load_or_make_seed() {
    std::string seed = g.config("randomizer.seed");
    if (!seed.empty()) return seed;
    const std::string dir = g.api->plugin_dir("randomizer");
    const std::string path = dir + "/seed.txt";
    if (std::ifstream in(path); in && std::getline(in, seed) && !seed.empty()) return seed;
    // A new seed the first time, kept so the world stays the same between
    // sessions of one run.
    const auto now = static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    char text[32];
    std::snprintf(text, sizeof(text), "%08llx", static_cast<unsigned long long>(bb::Rng(now).next() & 0xffffffffull));
    seed = text;
    if (!dir.empty()) std::ofstream(path) << seed << '\n';
    return seed;
}

void read_settings() {
    g_items = g.config_bool("randomizer.items", true);
    g_shops = g.config_bool("randomizer.shops", true);
    g_chalices = g.config_bool("randomizer.chalices", true);
    g_enemies = g.config_bool("randomizer.enemies", true);
    // "as found" and "even" were 1.0's names for the mixes
    const std::string mix = g.config("randomizer.enemy_mix", "evenly");
    g_enemy_even = mix != "like the original" && mix != "as found";
    g_enemy_npcs = g.config_bool("randomizer.enemy_npcs", false);
    g_enemy_anywhere = g.config("randomizer.enemy_from", "same area") == "whole game";
    const std::string bosses = g.config("randomizer.enemy_bosses", "none");
    g_enemy_bosses = bosses == "many" ? 2 : bosses == "a few" ? 1 : 0;
    g_seed = load_or_make_seed();
    g_token = bb::seed_token(g_seed);
}

// The player's own world, from the settings (under g_layout_mu).
World own_world() {
    World w;
    if (g_enemies) w = {g_token, g_enemy_even, g_enemy_npcs, g_enemy_anywhere, g_enemy_bosses};
    return w;
}

// What the session plays by: the seed, and the enemy settings that differ
// from the defaults (two worlds with different enemies are different rules).
std::string own_rules() {
    std::string r = "randomizer;seed=" + g_token;
    if (!g_enemies) return r + ";enemies=0";
    if (!g_enemy_even) r += ";enemy_mix=found";
    if (g_enemy_npcs) r += ";enemy_npcs=1";
    if (g_enemy_anywhere) r += ";enemy_from=any";
    if (g_enemy_bosses) r += ";enemy_bosses=" + std::to_string(g_enemy_bosses);
    return r;
}

// A host's entry ("randomizer;seed=x;enemy_mix=found") -> its world; false
// for anything this does not make a world from.
bool host_world(const std::string& rules, World* out) {
    World w;
    bool enemies = true;
    std::size_t at = rules.find(';');
    while (at != std::string::npos) {
        const std::size_t next = rules.find(';', at + 1);
        const std::string kv = rules.substr(at + 1, next == std::string::npos ? std::string::npos : next - at - 1);
        const std::size_t eq = kv.find('=');
        const std::string k = kv.substr(0, eq), v = eq == std::string::npos ? std::string() : kv.substr(eq + 1);
        if (k == "seed") w.token = v;
        else if (k == "enemies") enemies = v != "0";
        else if (k == "enemy_mix") w.even = v != "found";
        else if (k == "enemy_npcs") w.npcs = v == "1";
        else if (k == "enemy_from") w.anywhere = v == "any";
        else if (k == "enemy_bosses") w.bosses = v == "2" ? 2 : v == "1" ? 1 : 0;
        at = next;
    }
    if (!w.token.empty() && !plain_token(w.token)) return false;
    if (!enemies) w = World{};
    *out = w;
    return true;
}

// The settings changed while playing: a new plan into the tables now, and
// new layouts for the next load of each map (unless a host's world is being
// played - the next load then is the host's, and ours follows on the way
// home).
void reseed() {
    std::string token, rules;
    {
        std::lock_guard<std::mutex> lk(g_layout_mu);
        read_settings();
        token = g_token;
        // in another player's world its layouts stay; ours are written on the way home
        if (!g_visiting) write_layouts_locked(own_world());
        rules = own_rules();
    }
    g.api->set_rules("randomizer", rules.c_str());
    if (!g_lot_orig.empty() || !g_shop_orig.empty()) {
        make_plan();
        apply(true);
        write_spoiler();
    }
    g.log("randomizer: now seed \"%s\" (%s)", g_seed.c_str(), token.c_str());
    g.message("ランダマイザー - シード " + g_seed + ": アイテムは即時、敵は次のエリアから", 5.0f);
}

}  // namespace

extern "C" BB_PLUGIN_EXPORT int bb_plugin_image(const BbHostApi* api) {
    if (!g.attach(api, 10)) return 1;
    if (!api->eboot_is_109()) {
        g.log("randomizer: not the 1.09 eboot; off");
        return 1;
    }
    // Off, but loaded to play other randomizer players' worlds: nothing of
    // the player's own changes - only a host's layouts while a guest there.
    g_visitor = api->visitor("randomizer") != 0;
    if (!g_visitor) {
        read_settings();
        g.log("randomizer: seed \"%s\" (items %s, shops %s, enemies %s)", g_seed.c_str(), g_items ? "on" : "off",
              g_shops ? "on" : "off", g_enemies ? "on" : "off");
        g.api->set_rules("randomizer", own_rules().c_str());
    } else {
        g.log("randomizer: off - here only to play another randomizer player's world when summoned into it");
    }
    g.every_frame([] {
        if (g_visitor) return;
        // The tables load during the boot. The layouts are written once
        // NpcParam is in (the classes come from it) - before the first map is
        // read; the plan once the item tables are.
        if (!g_enemies_done && !g.param_ids<bb::params::NpcParam>().empty()) {
            g_enemies_done = true;
            std::lock_guard<std::mutex> lk(g_layout_mu);
            if (!g_visiting) write_layouts_locked(own_world());
        }
        if (g_lot_orig.empty() && g_shop_orig.empty()) {
            if (!g.param<EquipParamGoods>(1) && g.param_ids<EquipParamGoods>().empty()) return;
            if (g.param_ids<ItemLotParam>().empty() || g.param_ids<ShopLineupParam>().empty()) return;
            snapshot_originals();
            make_plan();
            write_spoiler();
        }
        apply();
    });
    // A setting changed in the in-game menu: everything again from it.
    g.api->on_option("randomizer", [](const char*, const char*, void*) {
        if (!g_visitor) reseed();
    }, nullptr);
    // Joining a randomizer player's world as a guest: its enemies (the
    // network thread, before that world loads); back home: ours. Pickups and
    // shops stay the player's own - a guest takes neither in another world.
    // rules: the host's entry ("randomizer;seed=x"), "" for a host without
    // the randomizer (its world is the game's), NULL going home.
    g.api->on_world_rules("randomizer", [](const char* rules, void*) {
        World host;
        if (rules && !host_world(rules, &host)) {
            g.log("randomizer: the host's seed is not one this builds a world from; staying as we are");
            return;
        }
        std::lock_guard<std::mutex> lk(g_layout_mu);   // reseed() on the main thread decides under it too
        const World mine = g_visitor ? World{} : own_world();
        const World& want = rules ? host : mine;
        const bool was_visiting = g_visiting;
        const bool changed = want.key() != g_world;
        g_visiting = rules != nullptr;
        // On screen once the world it is about has loaded: the summons' load,
        // or the one home. A vanilla player in a vanilla world hears nothing.
        if (rules) {
            if (!host.token.empty()) g_pending_banner = "ランダマイザー - ホストの世界のシード: " + host.token + (host.key() == mine.key() ? " （自分と同じ）" : "");
            else if (!mine.token.empty()) g_pending_banner = "ランダマイザー - ホストは通常のゲームです。敵は原作の配置です";
        } else if (was_visiting && (changed || !g_visitor)) {
            g_pending_banner = mine.token.empty() ? "ランダマイザー - 自分の世界に戻りました" : "ランダマイザー - 自分の世界に戻りました。シード: " + g_seed;
        }
        if (!changed) return;  // already that world
        g.log("randomizer: %s", rules ? (host.token.empty() ? "ホストの世界は原作の配置です" : ("ホストの世界のシード: " + host.token).c_str())
                                      : (g_visitor ? "自分の世界: 原作の配置に戻りました" : "自分の世界に戻りました"));
        write_layouts_locked(want);
    }, nullptr);
    // The seed on screen once, a few seconds into the first world (the player
    // exists while the loading screen is still up), and after a summons'
    // load: whose world it is.
    static int countdown = -1;
    static std::string banner;
    g.on_world_load([](std::uint32_t) {
        std::lock_guard<std::mutex> lk(g_layout_mu);
        if (!g_pending_banner.empty()) {
            banner = g_pending_banner;
            g_pending_banner.clear();
            g_announced = true;  // the join's banner says the seed instead
            countdown = 4 * 60;
        } else if (!g_announced && !g_visitor) {   // a visitor changes nothing of its own: no banner
            banner = "ランダマイザー - シード " + g_seed;
            g_announced = true;
            countdown = 5 * 60;
        }
    });
    g.every_frame([] {
        if (countdown > 0 && --countdown == 0) g.message(banner, 6.0f);
    });
    return 0;
}
