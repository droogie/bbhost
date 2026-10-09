#!/bin/bash
# What a macOS CI runner needs before tools/build_macos.sh: Rosetta 2 and the
# LunarG Vulkan SDK in ~/VulkanSDK/<version>, installed without its GUI. The
# workflows cache ~/VulkanSDK keyed on this file.
#   tools/ci_macos_setup.sh
set -eu
VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION:-1.4.363.0}
root=$HOME/VulkanSDK/$VULKAN_SDK_VERSION

if [ "$(uname -m)" = arm64 ] && ! /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null; then
    echo "installing Rosetta 2"
    sudo softwareupdate --install-rosetta --agree-to-license
fi
/usr/bin/arch -x86_64 /usr/bin/true || { echo "Rosetta 2 cannot run x86_64 code here"; exit 1; }

if [ ! -f "$root/macOS/lib/libMoltenVK.dylib" ]; then
    work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
    echo "fetching the Vulkan SDK $VULKAN_SDK_VERSION"
    curl -fsSL -m 1200 -o "$work/sdk.zip" "https://sdk.lunarg.com/sdk/download/$VULKAN_SDK_VERSION/mac/vulkan_sdk.zip"
    unzip -q "$work/sdk.zip" -d "$work/sdk"
    app=$(ls -d "$work"/sdk/*.app | head -1)
    "$app/Contents/MacOS/$(basename "$app" .app)" --root "$root" --accept-licenses --default-answer \
        --confirm-command install copy_only=1 > "$work/install.log" 2>&1 \
        || { tail -30 "$work/install.log"; exit 1; }
    # Only what the build and the package use.
    rm -rf "$root/Applications" "$root/macOS/bin" "$root/macOS/Frameworks" "$root/macOS/lib/KosmicKrisp.framework"
    find "$root/macOS/lib" -mindepth 1 -maxdepth 1 ! -name 'libvulkan.*' ! -name libMoltenVK.dylib \
        ! -name libSPIRV-Tools.a ! -name libSPIRV-Tools-opt.a ! -name libVkLayer_khronos_validation.dylib \
        ! -name cmake -exec rm -rf {} +
fi
for f in lib/libvulkan.1.dylib lib/libMoltenVK.dylib share/vulkan/icd.d/MoltenVK_icd.json include/vulkan/vulkan.h; do
    [ -e "$root/macOS/$f" ] || { echo "the Vulkan SDK in $root lacks macOS/$f"; exit 1; }
done
echo "Vulkan SDK: $root/macOS ($(du -sh "$root" | cut -f1))"
