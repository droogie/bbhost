#undef NDEBUG
#include "core/trophies.h"
#include "hle/trophy.h"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using trophies::Status;
std::map<std::string, void*> imports;
void register_hle_fn(const char* name, void* fn) { imports[name] = fn; }
std::string contents(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}
int main() {
    const auto root = fs::temp_directory_path() / ("bbhost-trophies-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const auto file = root / fs::u8path("記録") / "trophies.txt";
    fs::create_directories(file.parent_path());
    const std::string original = "# legacy bbhost records\n14 1700000000\nmalformed\n999 1700000001\n14 1700000002\n15 1700000003";
    std::ofstream(file, std::ios::binary) << original;
    trophies::Store store;
    store.open(file);
    assert(store.snapshot().count() == 2 && store.snapshot().earned[14] == 1700000000);
    assert(store.unlock(-1, 1700000100).status == Status::InvalidId);
    assert(store.unlock(40, 1700000100).status == Status::InvalidId);
    assert(store.unlock(0, 1700000100).status == Status::AutomaticPlatinum);
    assert(store.unlock(14, 1700000100).status == Status::AlreadyUnlocked);
    assert(store.pop_notice() == -1 && contents(file) == original);
    assert(store.unlock(16, 1700000100).status == Status::Ok);
    auto backup = file; backup += ".pre-trophies.bak";
    assert(contents(backup) == original);
    assert(contents(file).starts_with(original + "\n"));
    assert(store.pop_notice() == 16 && store.pop_notice() == -1);
    store.open(file);
    assert(store.snapshot().count() == 3 && store.snapshot().earned[16] == 1700000100);
    assert(store.unlock(17, 1700000200).status == Status::Ok && contents(backup) == original);

    // Concurrent duplicates persist and notify exactly once.
    store.open(root / "threads.txt");
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) threads.emplace_back([&] { store.unlock(21, 1700000300); });
    for (auto& t : threads) t.join();
    assert(store.snapshot().count() == 1 && store.pop_notice() == 21 && store.pop_notice() == -1);

    // DLC does not contribute to platinum; base completion does not require DLC.
    store.open(root / "base-only.txt");
    for (int i = 1; i < trophies::kBaseCount; ++i) {
        const auto a = store.unlock(i, 1700000400 + i);
        assert(a.status == Status::Ok && a.platinum == (i == 33 ? 0 : -1));
    }
    assert(store.snapshot().count() == 34 && store.snapshot().count(34, 40) == 0);
    for (int i = 1; i < trophies::kBaseCount; ++i) assert(store.pop_notice() == i);
    assert(store.pop_notice() == 0 && store.pop_notice() == -1);
    assert(store.reconcile_platinum(1700000500).platinum == -1);
    store.open(root / "dlc-only.txt");
    for (int i = 34; i < 40; ++i) assert(store.unlock(i, 1700000600).platinum == -1);
    assert(store.snapshot().count() == 6 && !store.snapshot().earned[0]);

    // An existing completed bbhost history gains platinum once at registration.
    const auto legacy = root / "complete-legacy.txt";
    { std::ofstream out(legacy); for (int i = 1; i < 34; ++i) out << i << " 1700000700\n"; }
    const auto before = contents(legacy);
    store.open(legacy);
    assert(store.snapshot().count() == 33 && contents(legacy) == before);
    assert(store.reconcile_platinum(1700000800).platinum == 0);
    assert(store.pop_notice() == 0 && store.pop_notice() == -1);

    // Failed persistence must neither acknowledge nor notify an unlock.
    const auto blocker = root / "not-a-directory";
    std::ofstream(blocker) << "preserve me";
    store.open(blocker / "trophies.txt");
    assert(store.unlock(21, 1700000900).status == Status::IoError);
    assert(!store.snapshot().earned[21] && store.pop_notice() == -1 && contents(blocker) == "preserve me");

    // Exercise the actual guest API, including exact buffer sizes and timestamps.
    trophies::runtime().open(root / "api.txt");
    using namespace trophy_hle;
    hle_register_trophies(); assert(imports.size() == 6);
    int ctx = 0, handle = 0;
    assert(create_context(nullptr, 1, 0, 0) == kInvalidArgument);
    assert(create_handle(nullptr) == kInvalidArgument);
    assert(create_context(&ctx, 1, 0, 0) == 0 && create_handle(&handle) == 0);
    int platinum = 123;
    assert(unlock(ctx, handle, 14, &platinum) == kNotRegistered && platinum == -1);
    trophies::runtime().open(root);
    assert(register_context(ctx, handle, 0) == kIoError);
    assert(unlock(ctx, handle, 14, &platinum) == kNotRegistered);
    trophies::runtime().open(root / "api.txt");
    assert(register_context(ctx, handle, 0) == 0);
    assert(unlock(-1, handle, 14, &platinum) == kInvalidContext);
    assert(unlock(ctx, -1, 14, &platinum) == kInvalidHandle);
    assert(unlock(ctx, handle, 40, &platinum) == kInvalidId);
    assert(unlock(ctx, handle, 0, &platinum) == kAutomaticPlatinum);
    assert(unlock(ctx, handle, 14, &platinum) == 0 && platinum == -1);
    assert(unlock(ctx, handle, 14, &platinum) == kAlreadyUnlocked);
    struct Guarded { std::uint64_t before; Data data; std::uint64_t after[3]; } guard{};
    guard.before = 0x12345678; for (auto& v : guard.after) v = 0x98765432;
    guard.data.size = sizeof(Data);
    Details detail{}; detail.size = sizeof(detail);
    assert(info(ctx, handle, 14, &detail, &guard.data) == 0);
    assert(guard.before == 0x12345678); for (auto v : guard.after) assert(v == 0x98765432);
    assert(guard.data.id == 14 && guard.data.unlocked == 1);
    assert(guard.data.timestamp == trophies::runtime().snapshot().earned[14] * 1000000ull + 62135596800000000ull);
    assert(detail.grade == 4 && detail.group == -1 && detail.hidden && std::string(detail.name) == "Father Gascoigne");
    guard.data.size = 8;
    assert(info(ctx, handle, 14, nullptr, &guard.data) == kInvalidArgument);
    assert(info(ctx, handle, -1, &detail, nullptr) == kInvalidId);
    assert(info(ctx, handle, 39, &detail, nullptr) == 0 && detail.group == 1);
    GameDetails gd{}; gd.size = sizeof(gd);
    GameData progress{}; progress.size = sizeof(progress);
    assert(game_info(ctx, handle, &gd, &progress) == 0);
    assert(gd.trophies == 40 && gd.platinum == 1 && gd.gold == 7 && gd.silver == 8 && gd.bronze == 24);
    assert(progress.trophies == 1 && progress.bronze == 1 && progress.progress == 2);
    for (int i = 1; i < 34; ++i) if (i != 14) assert(unlock(ctx, handle, i, &platinum) == 0);
    assert(platinum == 0);
    assert(game_info(ctx, handle, &gd, &progress) == 0 && progress.platinum == 1 && progress.trophies == 34);
    for (int i = 1; i < 8; ++i) assert(create_context(&ctx, 1, 0, 0) == 0);
    ctx = 123;
    assert(create_context(&ctx, 1, 0, 0) == static_cast<int>(0x80551623u) && ctx == 123);
    for (int i = 1; i < 4; ++i) assert(create_handle(&handle) == 0);
    handle = 123;
    assert(create_handle(&handle) == static_cast<int>(0x80551624u) && handle == 123);
    fs::remove_all(root);
    std::cout << "PASS: persistence, backup, duplicate/concurrent unlocks, platinum, DLC, notifications, I/O failures, guest ABI\n";
}
