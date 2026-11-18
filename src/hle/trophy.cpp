// SPDX-License-Identifier: GPL-3.0-or-later
#include "hle/trophy.h"
#include "hle/common.h"
#include "core/trophies.h"
#include <array>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace trophy_hle {
namespace {
std::mutex context_mutex;
int contexts = 0, handles = 0;
std::array<bool, 8> registered{};
constexpr std::uint64_t kUnixEpochTicks = 62135596800000000ull;
int validate(int ctx, int handle, bool need_registered = true) {
    std::lock_guard lock(context_mutex);
    if (ctx <= 0 || ctx > contexts) return kInvalidContext;
    if (handle <= 0 || handle > handles) return kInvalidHandle;
    if (need_registered && !registered[ctx - 1]) return kNotRegistered;
    return 0;
}
template<class T> bool fits(T* value) { return !value || value->size >= sizeof(T); }
int result(trophies::Status status) {
    switch (status) {
    case trophies::Status::Ok: return 0;
    case trophies::Status::AlreadyUnlocked: return kAlreadyUnlocked;
    case trophies::Status::InvalidId: return kInvalidId;
    case trophies::Status::AutomaticPlatinum: return kAutomaticPlatinum;
    default: return kIoError;
    }
}
}
GUEST_ABI int create_context(int* ctx, int, unsigned, std::uint64_t) {
    if (!ctx) return kInvalidArgument;
    std::lock_guard lock(context_mutex);
    if (contexts >= static_cast<int>(registered.size())) return kContextLimit;
    *ctx = ++contexts;
    return 0;
}
GUEST_ABI int create_handle(int* handle) {
    if (!handle) return kInvalidArgument;
    std::lock_guard lock(context_mutex);
    if (handles >= 4) return kHandleLimit;
    *handle = ++handles;
    return 0;
}
GUEST_ABI int register_context(int ctx, int handle, std::uint64_t) {
    if (const int error = validate(ctx, handle, false)) return error;
    const auto award = trophies::runtime().reconcile_platinum(std::time(nullptr));
    if (award.status == trophies::Status::IoError) {
        host_log("trophies: %s", trophies::runtime().snapshot().error.c_str());
        return kIoError;
    }
    {
        std::lock_guard lock(context_mutex);
        registered[ctx - 1] = true;
    }
    return result(award.status);
}
GUEST_ABI int unlock(int ctx, int handle, int id, int* platinum) {
    if (platinum) *platinum = -1;
    if (const int error = validate(ctx, handle)) return error;
    const auto award = trophies::runtime().unlock(id, std::time(nullptr));
    if (platinum) *platinum = award.platinum;
    if (award.status == trophies::Status::Ok)
        host_log("trophies: unlocked %d (%s)%s", id, trophies::catalog()[id].name, award.platinum == 0 ? ", platinum awarded" : "");
    else if (award.status == trophies::Status::IoError)
        host_log("trophies: %s", trophies::runtime().snapshot().error.c_str());
    return result(award.status);
}
GUEST_ABI int game_info(int ctx, int handle, GameDetails* details, GameData* data) {
    if (const int error = validate(ctx, handle)) return error;
    if ((!details && !data) || !fits(details) || !fits(data)) return kInvalidArgument;
    const auto state = trophies::runtime().snapshot();
    if (!state.error.empty()) return kIoError;
    std::array<unsigned, 5> totals{}, earned{};
    for (int i = 0; i < trophies::kCount; ++i) {
        const int grade = static_cast<int>(trophies::catalog()[i].grade);
        ++totals[grade];
        if (state.earned[i]) ++earned[grade];
    }
    if (details) {
        const auto size = details->size;
        *details = {};
        details->size = size;
        details->groups = 1;  // one add-on group, in addition to the base game (-1)
        details->trophies = trophies::kCount;
        details->platinum = totals[1]; details->gold = totals[2]; details->silver = totals[3]; details->bronze = totals[4];
        std::snprintf(details->title, sizeof(details->title), "%s", "Bloodborne");
        std::snprintf(details->description, sizeof(details->description), "%s", "Trophies for the base game and The Old Hunters");
    }
    if (data) {
        const auto size = data->size;
        *data = {};
        data->size = size;
        data->trophies = state.count();
        data->platinum = earned[1]; data->gold = earned[2]; data->silver = earned[3]; data->bronze = earned[4];
        data->progress = data->trophies * 100 / trophies::kCount;
    }
    return 0;
}
GUEST_ABI int info(int ctx, int handle, int id, Details* details, Data* data) {
    if (const int error = validate(ctx, handle)) return error;
    if (id < 0 || id >= trophies::kCount) return kInvalidId;
    if ((!details && !data) || !fits(details) || !fits(data)) return kInvalidArgument;
    const auto state = trophies::runtime().snapshot();
    if (!state.error.empty()) return kIoError;
    const auto& item = trophies::catalog()[id];
    if (details) {
        const auto size = details->size;
        *details = {};
        details->size = size; details->id = id; details->grade = static_cast<int>(item.grade);
        details->group = id < trophies::kBaseCount ? -1 : 1;
        details->hidden = item.hidden;
        std::snprintf(details->name, sizeof(details->name), "%s", item.name);
        std::snprintf(details->description, sizeof(details->description), "%s", item.description);
    }
    if (data) {
        const auto size = data->size;
        *data = {};
        data->size = size; data->id = id; data->unlocked = state.earned[id] != 0;
        if (data->unlocked) data->timestamp = state.earned[id] * 1000000ull + kUnixEpochTicks;
    }
    return 0;
}
}  // namespace trophy_hle
void hle_register_trophies() {
    HLE_REG("sceNpTrophyCreateContext", trophy_hle::create_context);
    HLE_REG("sceNpTrophyCreateHandle", trophy_hle::create_handle);
    HLE_REG("sceNpTrophyRegisterContext", trophy_hle::register_context);
    HLE_REG("sceNpTrophyUnlockTrophy", trophy_hle::unlock);
    HLE_REG("sceNpTrophyGetGameInfo", trophy_hle::game_info);
    HLE_REG("sceNpTrophyGetTrophyInfo", trophy_hle::info);
}
