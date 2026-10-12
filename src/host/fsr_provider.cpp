// The SDK host graph asks for D3D shader blobs. The Vulkan bridge resolves
// pipelines from its embedded SPIR-V catalog by pass name instead.
#include "ffx_fsr3upscaler_shaderblobs.h"

extern "C" FfxErrorCode fsr3UpscalerGetPermutationBlobByIndex(
    FfxFsr3UpscalerPass, uint32_t, FfxShaderBlob* blob) {
    if (!blob) return FFX_ERROR_INVALID_POINTER;
    *blob = {};
    return FFX_OK;
}

extern "C" FfxErrorCode fsr3UpscalerIsWave64(uint32_t, bool& wave64) {
    wave64 = false;
    return FFX_OK;
}
