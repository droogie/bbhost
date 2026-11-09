/*
 * ffx_providers_vk.cpp -- the providers bbhost's Linux FidelityFX library carries: FSR 3's upscaler alone (3.1.6, the
 * Windows package's version). AMD's own build lists each library's providers in a file of its own, which the SDK's
 * published source does not include; api/internal/ffx_provider.h declares the two functions and does the choosing.
 * No FSR 2 (the Windows DLL's older 2.3.4, which nothing in bbhost asks for), no frame generation.
 * Copyright (c) 2026 the bbhost authors. MIT licence: see LICENSE in this folder.
 */
#include <ffx_provider.h>

#include <upscalers/fsr3/include/ffx_provider_fsr3upscale.h>

namespace {

std::span<ffxProvider* const> providers() {
    static ffxProvider* const list[] = {&ffxProvider_FSR3Upscale::GetInstance()};
    return list;
}

}  // namespace

ffxProvider* GetProvider(ffxStructType_t descType, uint64_t overrideId, void* device, std::optional<ffxProviderExternal>& extProviderSlot) {
    return GetProvider(descType, overrideId, device, extProviderSlot, providers());
}

uint64_t GetProviderVersions(ffxStructType_t descType, void* device, uint64_t capacity, uint64_t* versionIds, const char** versionNames,
                             std::optional<ffxProviderExternal>& extProviderSlot) {
    return GetProviderVersions(descType, device, capacity, versionIds, versionNames, extProviderSlot, providers());
}
