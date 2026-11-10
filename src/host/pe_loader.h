#pragma once
// Lightweight x86-64 PE32+ DLL loader and Win32/Vulkan ABI bridge for Linux.
// Enables loading the Windows x64 FidelityFX SDK v2.4.0 (FSR 4 + FrameGen) and
// upscaler_bridge DLLs natively in-process on Linux x86-64.

#if defined(__linux__) && defined(__x86_64__)

#include <vulkan/vulkan.h>

#define BBHOST_MS_ABI __attribute__((ms_abi))

// Load a PE32+ DLL from disk (either full path or filename resolved against
// previously registered search directories).
void* pe_load_library(const char* path);

// Look up an exported symbol from a loaded PE32+ DLL.
void* pe_get_proc_address(void* module, const char* name);

// Release a loaded PE32+ DLL.
void pe_free_library(void* module);

// Register a directory to search when loaded PE DLLs call LoadLibraryA/W with
// a relative filename (e.g. amd_fidelityfx_upscaler_vk.dll).
void pe_add_search_dir(const char* dir);

// Set the native Vulkan proc-addr resolvers before initializing the DLLs.
void pe_set_vk_resolvers(PFN_vkGetInstanceProcAddr gipa,
                         PFN_vkGetDeviceProcAddr gdpa);

// MS-ABI wrappers around vkGetInstanceProcAddr / vkGetDeviceProcAddr that
// return MS-ABI-wrapped Vulkan function pointers to PE DLL callers.
VKAPI_ATTR PFN_vkVoidFunction BBHOST_MS_ABI
pe_vk_get_instance_proc_addr(VkInstance instance, const char* pName);

VKAPI_ATTR PFN_vkVoidFunction BBHOST_MS_ABI
pe_vk_get_device_proc_addr(VkDevice device, const char* pName);

// Create 3 virtual VkQueue handles backed by a single physical VkQueue when
// the driver (e.g. Mesa RADV) exposes fewer than 5 queues on the graphics family.
// All queue submissions and presents through either the real queue or the virtual
// queues from PE DLLs are serialized with host_gpu_queue_lock_only().
void pe_create_virtual_queues(VkQueue real_queue, VkQueue out_queues[3]);

// Destroy virtual VkQueue handles created by pe_create_virtual_queues().
void pe_destroy_virtual_queues(VkQueue queues[3]);

// Unwrap a virtual VkQueue to the underlying physical VkQueue (or return q as-is).
VkQueue pe_unwrap_queue(VkQueue q);

// Check if a given VkQueue handle is a virtual queue created by pe_create_virtual_queues().
bool pe_is_virtual_queue(VkQueue q);

#endif // defined(__linux__) && defined(__x86_64__)
