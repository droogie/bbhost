// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "guest_abi.h"
#include <cstddef>
#include <cstdint>

// Orbis NP Trophy ABI. Layout verified against shadPS4's np_trophy.h as
// included by ayozetr/bloodborne_linux at 804651e4 (GPL-2.0-or-later).
namespace trophy_hle {
struct Details {
    std::uint64_t size;
    std::int32_t id, grade, group;
    std::uint8_t hidden, reserved[3];
    char name[128], description[1024];
};
struct Data {
    std::uint64_t size;
    std::int32_t id;
    std::uint8_t unlocked, reserved[3];
    std::uint64_t timestamp;
};
struct GameDetails {
    std::uint64_t size;
    std::uint32_t groups, trophies, platinum, gold, silver, bronze;
    char title[128], description[1024];
};
struct GameData {
    std::uint64_t size;
    std::uint32_t trophies, platinum, gold, silver, bronze, progress;
};
static_assert(sizeof(Details) == 1176 && offsetof(Details, name) == 24);
static_assert(sizeof(Data) == 24 && offsetof(Data, timestamp) == 16);
static_assert(sizeof(GameDetails) == 1184 && offsetof(GameDetails, title) == 32);
static_assert(sizeof(GameData) == 32);
constexpr int kInvalidArgument = static_cast<int>(0x80551604u);
constexpr int kInvalidHandle = static_cast<int>(0x80551608u);
constexpr int kInvalidContext = static_cast<int>(0x80551609u);
constexpr int kInvalidId = static_cast<int>(0x8055160au);
constexpr int kAlreadyUnlocked = static_cast<int>(0x8055160cu);
constexpr int kAutomaticPlatinum = static_cast<int>(0x8055160du);
constexpr int kNotRegistered = static_cast<int>(0x8055160fu);
constexpr int kContextLimit = static_cast<int>(0x80551623u);
constexpr int kHandleLimit = static_cast<int>(0x80551624u);
constexpr int kIoError = static_cast<int>(0x80551600u);
GUEST_ABI int create_context(int* ctx, int user, unsigned label, std::uint64_t options);
GUEST_ABI int create_handle(int* handle);
GUEST_ABI int register_context(int ctx, int handle, std::uint64_t options);
GUEST_ABI int unlock(int ctx, int handle, int id, int* platinum);
GUEST_ABI int game_info(int ctx, int handle, GameDetails* details, GameData* data);
GUEST_ABI int info(int ctx, int handle, int id, Details* details, Data* data);
}  // namespace trophy_hle
void hle_register_trophies();
