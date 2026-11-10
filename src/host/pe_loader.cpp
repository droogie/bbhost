#include "host/pe_loader.h"

#if defined(__linux__) && defined(__x86_64__)

#include "host/gpu.h"
#include "log.h"

__attribute__((weak)) void host_gpu_queue_lock_only() {}
__attribute__((weak)) void host_gpu_queue_unlock() {}

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <locale.h>
#include <malloc.h>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <fcntl.h>
#include <dlfcn.h>

#pragma pack(push, 1)

struct IMAGE_DOS_HEADER {
    uint16_t e_magic;    // 0x5A4D "MZ"
    uint16_t e_cblp;
    uint16_t e_cp;
    uint16_t e_crlc;
    uint16_t e_cparhdr;
    uint16_t e_minalloc;
    uint16_t e_maxalloc;
    uint16_t e_ss;
    uint16_t e_sp;
    uint16_t e_csum;
    uint16_t e_ip;
    uint16_t e_cs;
    uint16_t e_lfarlc;
    uint16_t e_ovno;
    uint16_t e_res[4];
    uint16_t e_oemid;
    uint16_t e_oeminfo;
    uint16_t e_res2[10];
    int32_t  e_lfanew;   // Offset to NT headers
};

struct IMAGE_FILE_HEADER {
    uint16_t Machine;              // 0x8664 = IMAGE_FILE_MACHINE_AMD64
    uint16_t NumberOfSections;
    uint32_t TimeDateStamp;
    uint32_t PointerToSymbolTable;
    uint32_t NumberOfSymbols;
    uint16_t SizeOfOptionalHeader;
    uint16_t Characteristics;
};

struct IMAGE_DATA_DIRECTORY {
    uint32_t VirtualAddress;
    uint32_t Size;
};

struct IMAGE_OPTIONAL_HEADER64 {
    uint16_t Magic;                // 0x20B = PE32+
    uint8_t  MajorLinkerVersion;
    uint8_t  MinorLinkerVersion;
    uint32_t SizeOfCode;
    uint32_t SizeOfInitializedData;
    uint32_t SizeOfUninitializedData;
    uint32_t AddressOfEntryPoint;
    uint32_t BaseOfCode;
    uint64_t ImageBase;
    uint32_t SectionAlignment;
    uint32_t FileAlignment;
    uint16_t MajorOperatingSystemVersion;
    uint16_t MinorOperatingSystemVersion;
    uint16_t MajorImageVersion;
    uint16_t MinorImageVersion;
    uint16_t MajorSubsystemVersion;
    uint16_t MinorSubsystemVersion;
    uint32_t Win32VersionValue;
    uint32_t SizeOfImage;
    uint32_t SizeOfHeaders;
    uint32_t CheckSum;
    uint16_t Subsystem;
    uint16_t DllCharacteristics;
    uint64_t SizeOfStackReserve;
    uint64_t SizeOfStackCommit;
    uint64_t SizeOfHeapReserve;
    uint64_t SizeOfHeapCommit;
    uint32_t LoaderFlags;
    uint32_t NumberOfRvaAndSizes;
    IMAGE_DATA_DIRECTORY DataDirectory[16];
};

struct IMAGE_NT_HEADERS64 {
    uint32_t Signature;            // 0x00004550 "PE\0\0"
    IMAGE_FILE_HEADER FileHeader;
    IMAGE_OPTIONAL_HEADER64 OptionalHeader;
};

struct IMAGE_SECTION_HEADER {
    uint8_t  Name[8];
    uint32_t VirtualSize;
    uint32_t VirtualAddress;
    uint32_t SizeOfRawData;
    uint32_t PointerToRawData;
    uint32_t PointerToRelocations;
    uint32_t PointerToLinenumbers;
    uint16_t NumberOfRelocations;
    uint16_t NumberOfLinenumbers;
    uint32_t Characteristics;
};

struct IMAGE_EXPORT_DIRECTORY {
    uint32_t Characteristics;
    uint32_t TimeDateStamp;
    uint16_t MajorVersion;
    uint16_t MinorVersion;
    uint32_t Name;
    uint32_t Base;
    uint32_t NumberOfFunctions;
    uint32_t NumberOfNames;
    uint32_t AddressOfFunctions;
    uint32_t AddressOfNames;
    uint32_t AddressOfNameOrdinals;
};

struct IMAGE_IMPORT_DESCRIPTOR {
    uint32_t OriginalFirstThunk;
    uint32_t TimeDateStamp;
    uint32_t ForwarderChain;
    uint32_t Name;
    uint32_t FirstThunk;
};

struct IMAGE_TLS_DIRECTORY64 {
    uint64_t StartAddressOfRawData;
    uint64_t EndAddressOfRawData;
    uint64_t AddressOfIndex;
    uint64_t AddressOfCallbacks;
    uint32_t SizeOfZeroFill;
    uint32_t Characteristics;
};

struct VS_FIXEDFILEINFO {
    uint32_t dwSignature;        // 0xFEEF04BD
    uint32_t dwStrucVersion;     // 0x00010000
    uint32_t dwFileVersionMS;
    uint32_t dwFileVersionLS;
    uint32_t dwProductVersionMS;
    uint32_t dwProductVersionLS;
    uint32_t dwFileFlagsMask;
    uint32_t dwFileFlags;
    uint32_t dwFileOS;           // 0x00040004
    uint32_t dwFileType;         // 0x00000002
    uint32_t dwFileSubtype;
    uint32_t dwFileDateMS;
    uint32_t dwFileDateLS;
};

#pragma pack(pop)

// Win32 handle structure
enum WinHandleType {
    WIN_HANDLE_EVENT,
    WIN_HANDLE_THREAD,
    WIN_HANDLE_FILE
};

struct WinHandleObj {
    uint32_t magic; // 0x57484E44 "WHND"
    WinHandleType type;
    std::atomic<int> ref_count{1};

    // Event & Thread synchronization
    bool manual_reset = false;
    std::atomic<bool> signaled{false};
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cv = PTHREAD_COND_INITIALIZER;

    // Thread specific
    pthread_t thread = 0;
    void* thread_arg = nullptr;
    void* (*thread_func)(void*) = nullptr;
    uint32_t exit_code = 0;

    // File specific
    int fd = -1;
};

static WinHandleObj* alloc_win_handle(WinHandleType type) {
    auto* h = new WinHandleObj();
    h->magic = 0x57484E44;
    h->type = type;
    pthread_mutex_init(&h->mu, nullptr);
    pthread_cond_init(&h->cv, nullptr);
    return h;
}

static void release_win_handle(WinHandleObj* h) {
    if (!h) return;
    if (h->ref_count.fetch_sub(1) == 1) {
        if (h->type == WIN_HANDLE_FILE && h->fd >= 0) {
            close(h->fd);
            h->fd = -1;
        }
        pthread_mutex_destroy(&h->mu);
        pthread_cond_destroy(&h->cv);
        delete h;
    }
}

// Module representation
struct LoadedModule {
    std::string name;       // Lowercase filename (e.g. "upscaler_bridge.dll")
    std::string full_path;
    uint8_t* base = nullptr;
    size_t mapped_size = 0;
    size_t image_size = 0;
    int ref_count = 1;
    uint32_t tls_index = 0xFFFFFFFF;
    void* tls_template = nullptr;
    size_t tls_template_size = 0;
    size_t tls_zero_size = 0;
    uint8_t* stub_arena = nullptr;
    size_t stub_arena_used = 0;
    size_t stub_arena_cap = 0;
    VS_FIXEDFILEINFO fixed_file_info{};
    bool has_fixed_file_info = false;
};

static std::mutex g_modules_mu;
static std::vector<LoadedModule*> g_modules;
static std::vector<std::string> g_search_dirs;
static uint32_t g_next_tls_index = 0;

static PFN_vkGetInstanceProcAddr g_real_vkGetInstanceProcAddr = nullptr;
static PFN_vkGetDeviceProcAddr g_real_vkGetDeviceProcAddr = nullptr;

// Thread-local TLS and TEB state
static constexpr size_t MAX_PE_TLS_MODULES = 64;
static thread_local void* t_pe_tls_slots[MAX_PE_TLS_MODULES] = {nullptr};
static thread_local void* t_pe_tls_array_ptr = t_pe_tls_slots;
static thread_local uint8_t t_fake_teb[0x100] = {0};
static thread_local uint32_t t_last_error = 0;
static thread_local int t_queue_lock_depth = 0;

static void pe_lock_queue() {
    if (t_queue_lock_depth++ == 0) {
        host_gpu_queue_lock_only();
    }
}

static void pe_unlock_queue() {
    if (--t_queue_lock_depth == 0) {
        host_gpu_queue_unlock();
    }
}

static void init_thread_pe_tls() {
    if (*(void**)(t_fake_teb + 0x30) == nullptr) {
        *(void**)(t_fake_teb + 0x30) = t_fake_teb;               // Self pointer at gs:0x30
        *(void***)(t_fake_teb + 0x58) = &t_pe_tls_array_ptr;      // ThreadLocalStoragePointer at gs:0x58
        *(uint64_t*)(t_fake_teb + 0x10) = 0x1000;                // StackLimit at gs:0x10
    }

    std::lock_guard<std::mutex> lk(g_modules_mu);
    for (auto* m : g_modules) {
        if (m->tls_index != 0xFFFFFFFF && m->tls_index < MAX_PE_TLS_MODULES) {
            if (t_pe_tls_slots[m->tls_index] == nullptr) {
                size_t total = m->tls_template_size + m->tls_zero_size + 0x400;
                void* mem = calloc(1, total ? total : 0x1000);
                if (m->tls_template && m->tls_template_size) {
                    memcpy(mem, m->tls_template, m->tls_template_size);
                }
                t_pe_tls_slots[m->tls_index] = mem;
            }
        }
    }
}

extern "C" void* get_pe_tls_array_c() {
    init_thread_pe_tls();
    return t_pe_tls_slots;
}

extern "C" void* get_pe_teb_c() {
    init_thread_pe_tls();
    return t_fake_teb;
}

// Forward declarations
static void* resolve_import(const char* dll_name, const char* func_name, uint16_t ordinal);
static void* create_ms2sysv_stub(LoadedModule* mod, void* target_sysv_fn);

// Assembly bridge from MS-ABI callers to SysV-ABI callees (up to 12 arguments)
__attribute__((naked))
static void pe_ms2sysv_bridge() {
    __asm__ volatile (
        "push %rbp\n\t"
        "mov %rsp, %rbp\n\t"
        "push %rdi\n\t"
        "push %rsi\n\t"
        "sub $176, %rsp\n\t"
        "movdqu %xmm6, 0(%rsp)\n\t"
        "movdqu %xmm7, 16(%rsp)\n\t"
        "movdqu %xmm8, 32(%rsp)\n\t"
        "movdqu %xmm9, 48(%rsp)\n\t"
        "movdqu %xmm10, 64(%rsp)\n\t"
        "movdqu %xmm11, 80(%rsp)\n\t"
        "movdqu %xmm12, 96(%rsp)\n\t"
        "movdqu %xmm13, 112(%rsp)\n\t"
        "movdqu %xmm14, 128(%rsp)\n\t"
        "movdqu %xmm15, 144(%rsp)\n\t"

        // SysV stack args (args 7..12)
        "sub $64, %rsp\n\t"
        "mov 64(%rbp), %rax\n\t"
        "mov %rax, 0(%rsp)\n\t"
        "mov 72(%rbp), %rax\n\t"
        "mov %rax, 8(%rsp)\n\t"
        "mov 80(%rbp), %rax\n\t"
        "mov %rax, 16(%rsp)\n\t"
        "mov 88(%rbp), %rax\n\t"
        "mov %rax, 24(%rsp)\n\t"
        "mov 96(%rbp), %rax\n\t"
        "mov %rax, 32(%rsp)\n\t"
        "mov 104(%rbp), %rax\n\t"
        "mov %rax, 40(%rsp)\n\t"

        // SysV register args
        "mov %rcx, %rdi\n\t"
        "mov %rdx, %rsi\n\t"
        "mov %r8, %rdx\n\t"
        "mov %r9, %rcx\n\t"
        "mov 48(%rbp), %r8\n\t"
        "mov 56(%rbp), %r9\n\t"

        "call *%r10\n\t"

        "add $64, %rsp\n\t"
        "movdqu 0(%rsp), %xmm6\n\t"
        "movdqu 16(%rsp), %xmm7\n\t"
        "movdqu 32(%rsp), %xmm8\n\t"
        "movdqu 48(%rsp), %xmm9\n\t"
        "movdqu 64(%rsp), %xmm10\n\t"
        "movdqu 80(%rsp), %xmm11\n\t"
        "movdqu 96(%rsp), %xmm12\n\t"
        "movdqu 112(%rsp), %xmm13\n\t"
        "movdqu 128(%rsp), %xmm14\n\t"
        "movdqu 144(%rsp), %xmm15\n\t"
        "add $176, %rsp\n\t"
        "pop %rsi\n\t"
        "pop %rdi\n\t"
        "pop %rbp\n\t"
        "ret\n\t"
    );
}

static void* create_ms2sysv_stub(LoadedModule* mod, void* target_sysv_fn) {
    if (!mod || !target_sysv_fn) return nullptr;
    if (mod->stub_arena_used + 32 > mod->stub_arena_cap) {
        host_log("pe_loader: stub arena exhausted for module %s", mod->name.c_str());
        return nullptr;
    }
    uint8_t* stub = mod->stub_arena + mod->stub_arena_used;
    mod->stub_arena_used += 32;

    // movabs %r10, target_sysv_fn (10 bytes: 49 ba <8 bytes>)
    stub[0] = 0x49; stub[1] = 0xba;
    *(uint64_t*)&stub[2] = (uint64_t)target_sysv_fn;
    // movabs %r11, pe_ms2sysv_bridge (10 bytes: 49 bb <8 bytes>)
    stub[10] = 0x49; stub[11] = 0xbb;
    *(uint64_t*)&stub[12] = (uint64_t)&pe_ms2sysv_bridge;
    // jmp *%r11 (3 bytes: 41 ff e3)
    stub[20] = 0x41; stub[21] = 0xff; stub[22] = 0xe3;
    // NOP pad remainder
    memset(&stub[23], 0x90, 9);

    return stub;
}

// Fallback module for creating standalone Vulkan stubs
static LoadedModule* get_or_create_bridge_module() {
    static LoadedModule* bridge_mod = nullptr;
    if (!bridge_mod) {
        bridge_mod = new LoadedModule();
        bridge_mod->name = "vulkan_bridge_stubs";
        bridge_mod->stub_arena_cap = 128 * 1024;
        bridge_mod->stub_arena = (uint8_t*)mmap(nullptr, bridge_mod->stub_arena_cap,
                                                PROT_READ | PROT_WRITE | PROT_EXEC,
                                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    return bridge_mod;
}

static std::string to_lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static std::string get_file_name(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static std::string get_dir_name(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

void pe_add_search_dir(const char* dir) {
    if (!dir || !*dir) return;
    std::lock_guard<std::mutex> lk(g_modules_mu);
    for (const auto& d : g_search_dirs) {
        if (d == dir) return;
    }
    g_search_dirs.push_back(dir);
}

void pe_set_vk_resolvers(PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa) {
    g_real_vkGetInstanceProcAddr = gipa;
    g_real_vkGetDeviceProcAddr = gdpa;
}

// Virtual Queue implementation
struct VirtualQueueObj {
    uint64_t magic; // 0x5651554555450001ULL
    VkQueue real_queue;
};

static VirtualQueueObj g_virtual_queues[3];
static bool g_virtual_queues_active = false;

void pe_create_virtual_queues(VkQueue real_queue, VkQueue out_queues[3]) {
    for (int i = 0; i < 3; ++i) {
        g_virtual_queues[i].magic = 0x5651554555450001ULL + i;
        g_virtual_queues[i].real_queue = real_queue;
        out_queues[i] = (VkQueue)&g_virtual_queues[i];
    }
    g_virtual_queues_active = true;
    host_log("pe_loader: virtual VkQueues created for Mesa RADV backed by %p", (void*)real_queue);
}

void pe_destroy_virtual_queues(VkQueue queues[3]) {
    g_virtual_queues_active = false;
    for (int i = 0; i < 3; ++i) {
        queues[i] = VK_NULL_HANDLE;
    }
}

VkQueue pe_unwrap_queue(VkQueue q) {
    if (!q) return VK_NULL_HANDLE;
    if (g_virtual_queues_active) {
        for (int i = 0; i < 3; ++i) {
            if (q == (VkQueue)&g_virtual_queues[i]) {
                return g_virtual_queues[i].real_queue;
            }
        }
    }
    return q;
}

bool pe_is_virtual_queue(VkQueue q) {
    if (!q || !g_virtual_queues_active) return false;
    for (int i = 0; i < 3; ++i) {
        if (q == (VkQueue)&g_virtual_queues[i]) {
            return true;
        }
    }
    return false;
}

// Cached real function pointers
static VkDevice g_last_device = VK_NULL_HANDLE;
static PFN_vkQueueSubmit g_real_vkQueueSubmit = nullptr;
static PFN_vkQueueSubmit2 g_real_vkQueueSubmit2 = nullptr;
static PFN_vkQueuePresentKHR g_real_vkQueuePresentKHR = nullptr;
static PFN_vkQueueWaitIdle g_real_vkQueueWaitIdle = nullptr;
static PFN_vkDeviceWaitIdle g_real_vkDeviceWaitIdle = nullptr;

// Intercepted Vulkan functions
static VKAPI_ATTR VkResult BBHOST_MS_ABI
pe_vkQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence) {
    VkQueue real_q = pe_unwrap_queue(queue);
    pe_lock_queue();
    auto pfn = g_real_vkQueueSubmit;
    if (!pfn && g_real_vkGetDeviceProcAddr && g_last_device) {
        pfn = (PFN_vkQueueSubmit)g_real_vkGetDeviceProcAddr(g_last_device, "vkQueueSubmit");
    }
    if (!pfn) pfn = (PFN_vkQueueSubmit)dlsym(RTLD_DEFAULT, "vkQueueSubmit");
    VkResult res = pfn ? pfn(real_q, submitCount, pSubmits, fence) : VK_SUCCESS;
    pe_unlock_queue();
    return res;
}

static VKAPI_ATTR VkResult BBHOST_MS_ABI
pe_vkQueueSubmit2(VkQueue queue, uint32_t submitCount, const VkSubmitInfo2* pSubmits, VkFence fence) {
    VkQueue real_q = pe_unwrap_queue(queue);
    pe_lock_queue();
    auto pfn = g_real_vkQueueSubmit2;
    if (!pfn && g_real_vkGetDeviceProcAddr && g_last_device) {
        pfn = (PFN_vkQueueSubmit2)g_real_vkGetDeviceProcAddr(g_last_device, "vkQueueSubmit2");
        if (!pfn) pfn = (PFN_vkQueueSubmit2)g_real_vkGetDeviceProcAddr(g_last_device, "vkQueueSubmit2KHR");
    }
    if (!pfn) pfn = (PFN_vkQueueSubmit2)dlsym(RTLD_DEFAULT, "vkQueueSubmit2");
    if (!pfn) pfn = (PFN_vkQueueSubmit2)dlsym(RTLD_DEFAULT, "vkQueueSubmit2KHR");
    VkResult res = pfn ? pfn(real_q, submitCount, pSubmits, fence) : VK_ERROR_INITIALIZATION_FAILED;
    pe_unlock_queue();
    return res;
}

static VKAPI_ATTR VkResult BBHOST_MS_ABI
pe_vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
    VkQueue real_q = pe_unwrap_queue(queue);
    auto pfn = g_real_vkQueuePresentKHR;
    if (!pfn && g_real_vkGetDeviceProcAddr && g_last_device) {
        pfn = (PFN_vkQueuePresentKHR)g_real_vkGetDeviceProcAddr(g_last_device, "vkQueuePresentKHR");
    }
    if (!pfn) pfn = (PFN_vkQueuePresentKHR)dlsym(RTLD_DEFAULT, "vkQueuePresentKHR");
    return pfn ? pfn(real_q, pPresentInfo) : VK_SUCCESS;
}

static VKAPI_ATTR VkResult BBHOST_MS_ABI
pe_vkQueueWaitIdle(VkQueue queue) {
    VkQueue real_q = pe_unwrap_queue(queue);
    auto pfn = g_real_vkQueueWaitIdle;
    if (!pfn && g_real_vkGetDeviceProcAddr && g_last_device) {
        pfn = (PFN_vkQueueWaitIdle)g_real_vkGetDeviceProcAddr(g_last_device, "vkQueueWaitIdle");
    }
    if (!pfn) pfn = (PFN_vkQueueWaitIdle)dlsym(RTLD_DEFAULT, "vkQueueWaitIdle");
    return pfn ? pfn(real_q) : VK_SUCCESS;
}

static VKAPI_ATTR VkResult BBHOST_MS_ABI
pe_vkDeviceWaitIdle(VkDevice device) {
    auto pfn = g_real_vkDeviceWaitIdle;
    if (!pfn && g_real_vkGetDeviceProcAddr && device) {
        pfn = (PFN_vkDeviceWaitIdle)g_real_vkGetDeviceProcAddr(device, "vkDeviceWaitIdle");
    }
    if (!pfn) pfn = (PFN_vkDeviceWaitIdle)dlsym(RTLD_DEFAULT, "vkDeviceWaitIdle");
    return pfn ? pfn(device) : VK_SUCCESS;
}

static VKAPI_ATTR VkResult BBHOST_MS_ABI
pe_vkSetDebugUtilsObjectNameEXT(VkDevice device, const VkDebugUtilsObjectNameInfoEXT* pNameInfo) {
    if (!pNameInfo) return VK_SUCCESS;
    VkDebugUtilsObjectNameInfoEXT info = *pNameInfo;
    if (info.objectType == VK_OBJECT_TYPE_QUEUE) {
        info.objectHandle = (uint64_t)pe_unwrap_queue((VkQueue)info.objectHandle);
    }
    auto pfn = (PFN_vkSetDebugUtilsObjectNameEXT)(g_real_vkGetDeviceProcAddr ? g_real_vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT") : nullptr);
    if (pfn) return pfn(device, &info);
    return VK_SUCCESS;
}

static uint8_t g_adapter_luid[8] = {1, 0, 0, 0, 0, 0, 0, 0};

static VKAPI_ATTR void BBHOST_MS_ABI
pe_vkGetPhysicalDeviceProperties2(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* pProperties) {
    auto pfn = (PFN_vkGetPhysicalDeviceProperties2)(g_real_vkGetInstanceProcAddr ? g_real_vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkGetPhysicalDeviceProperties2") : nullptr);
    if (!pfn) pfn = (PFN_vkGetPhysicalDeviceProperties2)dlsym(RTLD_DEFAULT, "vkGetPhysicalDeviceProperties2");
    if (pfn) pfn(physicalDevice, pProperties);

    if (pProperties) {
        auto* header = (VkBaseOutStructure*)pProperties->pNext;
        while (header) {
            if (header->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES) {
                auto* idProps = (VkPhysicalDeviceIDProperties*)header;
                idProps->deviceLUIDValid = VK_TRUE;
                memcpy(idProps->deviceLUID, g_adapter_luid, 8);
            }
            header = header->pNext;
        }
    }
}

VKAPI_ATTR PFN_vkVoidFunction BBHOST_MS_ABI
pe_vk_get_instance_proc_addr(VkInstance instance, const char* pName) {
    if (!pName) return nullptr;
    if (strcmp(pName, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)pe_vk_get_instance_proc_addr;
    if (strcmp(pName, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)pe_vk_get_device_proc_addr;
    if (strcmp(pName, "vkGetPhysicalDeviceProperties2") == 0 ||
        strcmp(pName, "vkGetPhysicalDeviceProperties2KHR") == 0) {
        return (PFN_vkVoidFunction)pe_vkGetPhysicalDeviceProperties2;
    }
    if (strcmp(pName, "vkQueueSubmit") == 0) return (PFN_vkVoidFunction)pe_vkQueueSubmit;
    if (strcmp(pName, "vkQueueSubmit2") == 0 || strcmp(pName, "vkQueueSubmit2KHR") == 0) {
        return (PFN_vkVoidFunction)pe_vkQueueSubmit2;
    }
    if (strcmp(pName, "vkQueuePresentKHR") == 0) return (PFN_vkVoidFunction)pe_vkQueuePresentKHR;
    if (strcmp(pName, "vkQueueWaitIdle") == 0) return (PFN_vkVoidFunction)pe_vkQueueWaitIdle;
    if (strcmp(pName, "vkDeviceWaitIdle") == 0) return (PFN_vkVoidFunction)pe_vkDeviceWaitIdle;
    if (strcmp(pName, "vkSetDebugUtilsObjectNameEXT") == 0) return (PFN_vkVoidFunction)pe_vkSetDebugUtilsObjectNameEXT;

    PFN_vkVoidFunction pfn = nullptr;
    if (g_real_vkGetInstanceProcAddr) {
        pfn = g_real_vkGetInstanceProcAddr(instance, pName);
    }
    if (!pfn) {
        auto gipa = (PFN_vkGetInstanceProcAddr)dlsym(RTLD_DEFAULT, "vkGetInstanceProcAddr");
        if (gipa) pfn = gipa(instance, pName);
    }
    if (!pfn) {
        pfn = (PFN_vkVoidFunction)dlsym(RTLD_DEFAULT, pName);
    }
    if (!pfn) return nullptr;

    return (PFN_vkVoidFunction)create_ms2sysv_stub(get_or_create_bridge_module(), (void*)pfn);
}

VKAPI_ATTR PFN_vkVoidFunction BBHOST_MS_ABI
pe_vk_get_device_proc_addr(VkDevice device, const char* pName) {
    if (!pName) return nullptr;
    if (device) g_last_device = device;
    if (strcmp(pName, "vkGetDeviceProcAddr") == 0) return (PFN_vkVoidFunction)pe_vk_get_device_proc_addr;
    if (strcmp(pName, "vkQueueSubmit") == 0) {
        if (!g_real_vkQueueSubmit && g_real_vkGetDeviceProcAddr && device) {
            g_real_vkQueueSubmit = (PFN_vkQueueSubmit)g_real_vkGetDeviceProcAddr(device, "vkQueueSubmit");
        }
        return (PFN_vkVoidFunction)pe_vkQueueSubmit;
    }
    if (strcmp(pName, "vkQueueSubmit2") == 0 || strcmp(pName, "vkQueueSubmit2KHR") == 0) {
        if (!g_real_vkQueueSubmit2 && g_real_vkGetDeviceProcAddr && device) {
            g_real_vkQueueSubmit2 = (PFN_vkQueueSubmit2)g_real_vkGetDeviceProcAddr(device, pName);
        }
        return (PFN_vkVoidFunction)pe_vkQueueSubmit2;
    }
    if (strcmp(pName, "vkQueuePresentKHR") == 0) {
        if (!g_real_vkQueuePresentKHR && g_real_vkGetDeviceProcAddr && device) {
            g_real_vkQueuePresentKHR = (PFN_vkQueuePresentKHR)g_real_vkGetDeviceProcAddr(device, "vkQueuePresentKHR");
        }
        return (PFN_vkVoidFunction)pe_vkQueuePresentKHR;
    }
    if (strcmp(pName, "vkQueueWaitIdle") == 0) {
        if (!g_real_vkQueueWaitIdle && g_real_vkGetDeviceProcAddr && device) {
            g_real_vkQueueWaitIdle = (PFN_vkQueueWaitIdle)g_real_vkGetDeviceProcAddr(device, "vkQueueWaitIdle");
        }
        return (PFN_vkVoidFunction)pe_vkQueueWaitIdle;
    }
    if (strcmp(pName, "vkDeviceWaitIdle") == 0) {
        if (!g_real_vkDeviceWaitIdle && g_real_vkGetDeviceProcAddr && device) {
            g_real_vkDeviceWaitIdle = (PFN_vkDeviceWaitIdle)g_real_vkGetDeviceProcAddr(device, "vkDeviceWaitIdle");
        }
        return (PFN_vkVoidFunction)pe_vkDeviceWaitIdle;
    }
    if (strcmp(pName, "vkSetDebugUtilsObjectNameEXT") == 0) return (PFN_vkVoidFunction)pe_vkSetDebugUtilsObjectNameEXT;

    PFN_vkVoidFunction pfn = nullptr;
    if (g_real_vkGetDeviceProcAddr) {
        pfn = g_real_vkGetDeviceProcAddr(device, pName);
    }
    if (!pfn) {
        auto gdpa = (PFN_vkGetDeviceProcAddr)dlsym(RTLD_DEFAULT, "vkGetDeviceProcAddr");
        if (gdpa) pfn = gdpa(device, pName);
    }
    if (!pfn && g_real_vkGetInstanceProcAddr) {
        pfn = g_real_vkGetInstanceProcAddr(VK_NULL_HANDLE, pName);
    }
    if (!pfn) {
        pfn = (PFN_vkVoidFunction)dlsym(RTLD_DEFAULT, pName);
    }
    if (!pfn) return nullptr;

    return (PFN_vkVoidFunction)create_ms2sysv_stub(get_or_create_bridge_module(), (void*)pfn);
}

// D3DKMT Mock functions for gdi32.dll
struct D3DKMT_ADAPTERINFO {
    uint32_t hAdapter;
    uint8_t  AdapterLuid[8];
    uint32_t NumOfSources;
    uint32_t bPresentMoveRegionsPreferred;
};

struct D3DKMT_ENUMADAPTERS {
    uint32_t NumAdapters;
    D3DKMT_ADAPTERINFO Adapters[16];
};

struct D3DKMT_ENUMADAPTERS2 {
    uint32_t NumAdapters;
    D3DKMT_ADAPTERINFO* pAdapters;
};

struct D3DKMT_QUERYADAPTERINFO {
    uint32_t hAdapter;
    uint32_t Type;
    void*    pPrivateDriverData;
    uint32_t PrivateDriverDataSize;
};

static uint32_t BBHOST_MS_ABI pe_D3DKMTEnumAdapters(D3DKMT_ENUMADAPTERS* pEnum) {
    if (!pEnum) return 0xC000000D; // STATUS_INVALID_PARAMETER
    pEnum->NumAdapters = 1;
    pEnum->Adapters[0].hAdapter = 1;
    memcpy(pEnum->Adapters[0].AdapterLuid, g_adapter_luid, 8);
    pEnum->Adapters[0].NumOfSources = 1;
    pEnum->Adapters[0].bPresentMoveRegionsPreferred = 0;
    return 0; // STATUS_SUCCESS
}

static uint32_t BBHOST_MS_ABI pe_D3DKMTEnumAdapters2(D3DKMT_ENUMADAPTERS2* pEnum2) {
    if (!pEnum2) return 0xC000000D;
    if (!pEnum2->pAdapters) {
        pEnum2->NumAdapters = 1;
        return 0;
    }
    pEnum2->NumAdapters = 1;
    pEnum2->pAdapters[0].hAdapter = 1;
    memcpy(pEnum2->pAdapters[0].AdapterLuid, g_adapter_luid, 8);
    pEnum2->pAdapters[0].NumOfSources = 1;
    pEnum2->pAdapters[0].bPresentMoveRegionsPreferred = 0;
    return 0;
}

static uint32_t BBHOST_MS_ABI pe_D3DKMTCloseAdapter(void*) {
    return 0;
}

// Linux DRM AMDGPU query structure
struct drm_amdgpu_info_device_data {
    uint32_t device_id;
    uint32_t chip_rev;
    uint32_t external_rev;
    uint32_t pci_rev;
    uint32_t family;
    uint32_t num_shader_engines;
    uint32_t num_shader_arrays_per_engine;
    uint32_t gpu_counter_freq;
    uint64_t max_engine_clock;
    uint64_t max_memory_clock;
    uint32_t cu_active_number;
};

struct drm_amdgpu_info_req {
    uint64_t return_pointer;
    uint32_t return_size;
    uint32_t query;
    uint8_t  pad[16];
};

static uint32_t BBHOST_MS_ABI pe_D3DKMTQueryAdapterInfo(const D3DKMT_QUERYADAPTERINFO* pQuery) {
    if (!pQuery || !pQuery->pPrivateDriverData) return 0xC000000D;

    uint32_t family = 0x91; // Default to Navi 31 / RDNA 3 (RX 7900 XTX)
    uint32_t ext_rev = 0x1;

    // Query local DRM node
    int fd = open("/dev/dri/renderD128", O_RDWR);
    if (fd < 0) fd = open("/dev/dri/renderD129", O_RDWR);
    if (fd >= 0) {
        drm_amdgpu_info_device_data dev_info{};
        drm_amdgpu_info_req req{};
        req.return_pointer = (uint64_t)&dev_info;
        req.return_size = sizeof(dev_info);
        req.query = 0x16; // AMDGPU_INFO_DEV_INFO
        // DRM_IOCTL_AMDGPU_INFO = 0x40206445
        if (ioctl(fd, 0x40206445, &req) == 0 && dev_info.family != 0) {
            family = dev_info.family;
            ext_rev = dev_info.external_rev;
        }
        close(fd);
    }

    auto* out = (uint8_t*)pQuery->pPrivateDriverData;
    if (pQuery->PrivateDriverDataSize >= 0x38) {
        *(uint32_t*)(out + 0x30) = family;
        *(uint32_t*)(out + 0x34) = ext_rev;
    }
    return 0; // STATUS_SUCCESS
}

typedef uint16_t win_wchar;

static std::string win_utf16_to_utf8(const win_wchar* ws, int max_chars = -1) {
    if (!ws) return "";
    std::string s;
    for (int i = 0; ws[i] && (max_chars < 0 || i < max_chars); ++i) {
        uint16_t c = ws[i];
        if (c == '\\') c = '/';
        if (c < 0x80) {
            s.push_back((char)c);
        } else if (c < 0x800) {
            s.push_back((char)(0xC0 | (c >> 6)));
            s.push_back((char)(0x80 | (c & 0x3F)));
        } else {
            s.push_back((char)(0xE0 | (c >> 12)));
            s.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
            s.push_back((char)(0x80 | (c & 0x3F)));
        }
    }
    return s;
}

static size_t win_utf8_to_utf16(const char* s, win_wchar* out, size_t out_max) {
    if (!out || out_max == 0) return 0;
    if (!s) { out[0] = 0; return 0; }
    size_t i = 0;
    while (*s && i + 1 < out_max) {
        out[i++] = (win_wchar)(uint8_t)*s++;
    }
    out[i] = 0;
    return i;
}

static size_t win_wcslen(const win_wchar* s) {
    if (!s) return 0;
    size_t len = 0;
    while (s[len]) ++len;
    return len;
}

static std::string locate_pe_file(const char* path);

// VERSION.dll functions
static uint32_t BBHOST_MS_ABI pe_GetFileVersionInfoSizeW(const win_wchar* filename, uint32_t* handle) {
    if (handle) *handle = 0;
    return 4096;
}

static uint32_t BBHOST_MS_ABI pe_GetFileVersionInfoSizeA(const char* filename, uint32_t* handle) {
    if (handle) *handle = 0;
    return 4096;
}

static uint32_t BBHOST_MS_ABI pe_GetFileVersionInfoW(const win_wchar* filename, uint32_t handle, uint32_t len, void* data) {
    if (!data || len < sizeof(VS_FIXEDFILEINFO)) return 0;
    memset(data, 0, len);

    // Search loaded modules
    std::lock_guard<std::mutex> lk(g_modules_mu);
    for (auto* m : g_modules) {
        if (m->has_fixed_file_info) {
            memcpy(data, &m->fixed_file_info, sizeof(VS_FIXEDFILEINFO));
            return 1;
        }
    }

    // Default version info: FSR SDK 2.4.0 / 4.1.2
    auto* fi = (VS_FIXEDFILEINFO*)data;
    fi->dwSignature = 0xFEEF04BD;
    fi->dwStrucVersion = 0x00010000;
    fi->dwFileVersionMS = 0x00040001; // 4.1
    fi->dwFileVersionLS = 0x000211CB; // 2.x
    fi->dwProductVersionMS = 0x00040001;
    fi->dwProductVersionLS = 0x00020000;
    fi->dwFileOS = 0x00040004;
    fi->dwFileType = 0x00000002;
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_GetFileVersionInfoA(const char* filename, uint32_t handle, uint32_t len, void* data) {
    return pe_GetFileVersionInfoW(nullptr, handle, len, data);
}

static uint32_t BBHOST_MS_ABI pe_VerQueryValueW(const void* block, const win_wchar* sub_block, void** buffer, uint32_t* len) {
    if (!block || !sub_block || !buffer || !len) return 0;
    std::string sub = win_utf16_to_utf8(sub_block);
    if (sub == "/" || sub == "\\") {
        *buffer = (void*)block;
        *len = sizeof(VS_FIXEDFILEINFO);
        return 1;
    }
    if (sub.find("Translation") != std::string::npos) {
        static uint16_t s_lang[2] = {0x0409, 0x04B0};
        *buffer = (void*)s_lang;
        *len = 4;
        return 1;
    }
    if (sub.find("ProductName") != std::string::npos) {
        static const win_wchar s_name[] = {
            'A','M','D',' ','F','i','d','e','l','i','t','y','F','X',' ','S','D','K',0
        };
        *buffer = (void*)s_name;
        *len = sizeof(s_name) / sizeof(win_wchar);
        return 1;
    }
    *buffer = (void*)block;
    *len = 0;
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_VerQueryValueA(const void* block, const char* sub_block, void** buffer, uint32_t* len) {
    if (!block || !sub_block || !buffer || !len) return 0;
    if (strcmp(sub_block, "\\") == 0) {
        *buffer = (void*)block;
        *len = sizeof(VS_FIXEDFILEINFO);
        return 1;
    }
    *buffer = (void*)block;
    *len = 0;
    return 1;
}

// Synchronization: Critical Sections
static void BBHOST_MS_ABI pe_InitializeCriticalSection(void* cs) {
    auto* mu = new pthread_mutex_t;
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(mu, &attr);
    pthread_mutexattr_destroy(&attr);
    *(pthread_mutex_t**)cs = mu;
}

static uint32_t BBHOST_MS_ABI pe_InitializeCriticalSectionAndSpinCount(void* cs, uint32_t) {
    pe_InitializeCriticalSection(cs);
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_InitializeCriticalSectionEx(void* cs, uint32_t, uint32_t) {
    pe_InitializeCriticalSection(cs);
    return 1;
}

static void BBHOST_MS_ABI pe_EnterCriticalSection(void* cs) {
    auto* mu = *(pthread_mutex_t**)cs;
    if (!mu) {
        pe_InitializeCriticalSection(cs);
        mu = *(pthread_mutex_t**)cs;
    }
    pthread_mutex_lock(mu);
}

static uint32_t BBHOST_MS_ABI pe_TryEnterCriticalSection(void* cs) {
    auto* mu = *(pthread_mutex_t**)cs;
    if (!mu) {
        pe_InitializeCriticalSection(cs);
        mu = *(pthread_mutex_t**)cs;
    }
    return pthread_mutex_trylock(mu) == 0 ? 1 : 0;
}

static void BBHOST_MS_ABI pe_LeaveCriticalSection(void* cs) {
    auto* mu = *(pthread_mutex_t**)cs;
    if (mu) pthread_mutex_unlock(mu);
}

static void BBHOST_MS_ABI pe_DeleteCriticalSection(void* cs) {
    auto* mu = *(pthread_mutex_t**)cs;
    if (mu) {
        pthread_mutex_destroy(mu);
        delete mu;
        *(pthread_mutex_t**)cs = nullptr;
    }
}

// Synchronization: SRW Locks
static void BBHOST_MS_ABI pe_AcquireSRWLockExclusive(void** srw) {
    if (!*srw) {
        static pthread_mutex_t s_srw_init_mu = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_lock(&s_srw_init_mu);
        if (!*srw) {
            auto* m = new pthread_mutex_t;
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
            pthread_mutex_init(m, &attr);
            pthread_mutexattr_destroy(&attr);
            *srw = m;
        }
        pthread_mutex_unlock(&s_srw_init_mu);
    }
    pthread_mutex_lock((pthread_mutex_t*)*srw);
}

static void BBHOST_MS_ABI pe_ReleaseSRWLockExclusive(void** srw) {
    if (*srw) pthread_mutex_unlock((pthread_mutex_t*)*srw);
}

static uint32_t BBHOST_MS_ABI pe_TryAcquireSRWLockExclusive(void** srw) {
    if (!*srw) pe_AcquireSRWLockExclusive(srw);
    return pthread_mutex_trylock((pthread_mutex_t*)*srw) == 0 ? 1 : 0;
}

static void BBHOST_MS_ABI pe_SleepConditionVariableSRW(void* cv, void** srw, uint32_t ms, uint32_t) {
    if (ms == 0xFFFFFFFF) {
        pe_ReleaseSRWLockExclusive(srw);
        usleep(1000);
        pe_AcquireSRWLockExclusive(srw);
    } else {
        pe_ReleaseSRWLockExclusive(srw);
        usleep(ms * 1000);
        pe_AcquireSRWLockExclusive(srw);
    }
}

static void BBHOST_MS_ABI pe_WakeAllConditionVariable(void*) {}

// MSVCP Mutex implementation
static void BBHOST_MS_ABI pe_Mtx_init_in_situ(void* mtx, int) {
    pe_InitializeCriticalSection(mtx);
}

static int BBHOST_MS_ABI pe_Mtx_lock(void* mtx) {
    pe_EnterCriticalSection(mtx);
    return 0;
}

static int BBHOST_MS_ABI pe_Mtx_unlock(void* mtx) {
    pe_LeaveCriticalSection(mtx);
    return 0;
}

// Events, Threads, Handles
static void* BBHOST_MS_ABI pe_CreateEventW(void*, uint32_t manual_reset, uint32_t initial_state, const win_wchar*) {
    WinHandleObj* h = alloc_win_handle(WIN_HANDLE_EVENT);
    h->manual_reset = manual_reset != 0;
    h->signaled.store(initial_state != 0);
    return h;
}

static uint32_t BBHOST_MS_ABI pe_SetEvent(void* handle) {
    auto* h = (WinHandleObj*)handle;
    if (!h || h->magic != 0x57484E44) return 0;
    pthread_mutex_lock(&h->mu);
    h->signaled.store(true);
    pthread_cond_broadcast(&h->cv);
    pthread_mutex_unlock(&h->mu);
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_ResetEvent(void* handle) {
    auto* h = (WinHandleObj*)handle;
    if (!h || h->magic != 0x57484E44) return 0;
    h->signaled.store(false);
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_WaitForSingleObject(void* handle, uint32_t ms) {
    auto* h = (WinHandleObj*)handle;
    if (!h || h->magic != 0x57484E44) return 0xFFFFFFFF; // WAIT_FAILED

    pthread_mutex_lock(&h->mu);
    uint32_t ret = 0; // WAIT_OBJECT_0

    if (ms == 0) {
        if (!h->signaled.load()) ret = 0x102; // WAIT_TIMEOUT
    } else if (ms == 0xFFFFFFFF) {
        while (!h->signaled.load()) {
            pthread_cond_wait(&h->cv, &h->mu);
        }
    } else {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += ms / 1000;
        ts.tv_nsec += (ms % 1000) * 1000000L;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000L;
        }
        while (!h->signaled.load()) {
            if (pthread_cond_timedwait(&h->cv, &h->mu, &ts) == ETIMEDOUT) {
                if (!h->signaled.load()) ret = 0x102;
                break;
            }
        }
    }

    if (ret == 0 && !h->manual_reset && h->type != WIN_HANDLE_THREAD) {
        h->signaled.store(false);
    }
    pthread_mutex_unlock(&h->mu);
    return ret;
}

static uint32_t BBHOST_MS_ABI pe_CloseHandle(void* handle) {
    auto* h = (WinHandleObj*)handle;
    if (!h || h->magic != 0x57484E44) return 0;
    release_win_handle(h);
    return 1;
}

struct ThreadStartWrapper {
    WinHandleObj* handle = nullptr;
    uint32_t (BBHOST_MS_ABI *func)(void*) = nullptr;
    void* arg = nullptr;
};

static void* pe_thread_entry(void* p) {
    auto* wrap = (ThreadStartWrapper*)p;
    init_thread_pe_tls();

    uint32_t (BBHOST_MS_ABI *f)(void*) = wrap->func;
    void* a = wrap->arg;
    WinHandleObj* h = wrap->handle;
    delete wrap;

    uint32_t exit_code = f(a);

    pthread_mutex_lock(&h->mu);
    h->exit_code = exit_code;
    h->signaled.store(true);
    pthread_cond_broadcast(&h->cv);
    pthread_mutex_unlock(&h->mu);

    release_win_handle(h);
    return nullptr;
}

static void* BBHOST_MS_ABI pe_CreateThread(void*, size_t, uint32_t (BBHOST_MS_ABI *start_addr)(void*),
                                           void* param, uint32_t, uint32_t* thread_id) {
    WinHandleObj* h = alloc_win_handle(WIN_HANDLE_THREAD);
    h->ref_count.fetch_add(1); // One for thread itself, one for caller handle

    auto* wrap = new ThreadStartWrapper{h, start_addr, param};
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&h->thread, &attr, pe_thread_entry, wrap);
    pthread_attr_destroy(&attr);

    if (thread_id) *thread_id = (uint32_t)h->thread;
    return h;
}

// Timing and System Info
static uint32_t BBHOST_MS_ABI pe_QueryPerformanceCounter(int64_t* lpPerformanceCount) {
    if (!lpPerformanceCount) return 0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *lpPerformanceCount = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_QueryPerformanceFrequency(int64_t* lpFrequency) {
    if (!lpFrequency) return 0;
    *lpFrequency = 1000000000LL;
    return 1;
}

static void BBHOST_MS_ABI pe_GetSystemTimeAsFileTime(uint64_t* lpSystemTimeAsFileTime) {
    if (!lpSystemTimeAsFileTime) return;
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    // Convert to 100-ns intervals since Jan 1, 1601
    *lpSystemTimeAsFileTime = (uint64_t)(tv.tv_sec + 11644473600ULL) * 10000000ULL + tv.tv_usec * 10ULL;
}

static uint64_t BBHOST_MS_ABI pe_GetTickCount64() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;
}

static uint32_t BBHOST_MS_ABI pe_GetCurrentProcessId() { return (uint32_t)getpid(); }
static uint32_t BBHOST_MS_ABI pe_GetCurrentThreadId() { return (uint32_t)syscall(SYS_gettid); }
static void* BBHOST_MS_ABI pe_GetCurrentProcess() { return (void*)-1; }
static void* BBHOST_MS_ABI pe_GetCurrentThread() { return (void*)-2; }

static void BBHOST_MS_ABI pe_GetSystemInfo(void* lpSystemInfo) {
    if (!lpSystemInfo) return;
    memset(lpSystemInfo, 0, 48);
    auto* p = (uint32_t*)lpSystemInfo;
    p[1] = 4096; // Page size
    p[5] = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN); // Number of processors
}

// Memory Allocation
static void* BBHOST_MS_ABI pe_GetProcessHeap() { return (void*)0x12345678ULL; }

static void* BBHOST_MS_ABI pe_HeapAlloc(void* hHeap, uint32_t flags, size_t bytes) {
    size_t total = (bytes + 16 + 4095) & ~4095ULL;
    void* p = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    *(size_t*)p = total;
    void* mem = (void*)((uint8_t*)p + 16);
    if (flags & 8) memset(mem, 0, bytes);
    return mem;
}

static uint32_t BBHOST_MS_ABI pe_HeapFree(void* hHeap, uint32_t flags, void* mem) {
    if (!mem) return 1;
    void* p = (void*)((uint8_t*)mem - 16);
    size_t total = *(size_t*)p;
    munmap(p, total);
    return 1;
}

static void* BBHOST_MS_ABI pe_HeapReAlloc(void* hHeap, uint32_t flags, void* mem, size_t bytes) {
    if (!mem) return pe_HeapAlloc(hHeap, flags, bytes);
    void* p = (void*)((uint8_t*)mem - 16);
    size_t old_total = *(size_t*)p;
    size_t old_bytes = old_total - 16;
    void* new_mem = pe_HeapAlloc(hHeap, flags, bytes);
    if (!new_mem) return nullptr;
    memcpy(new_mem, mem, std::min(old_bytes, bytes));
    pe_HeapFree(hHeap, flags, mem);
    return new_mem;
}

static size_t BBHOST_MS_ABI pe_HeapSize(void*, uint32_t, const void* mem) {
    if (!mem) return 0;
    void* p = (void*)((uint8_t*)mem - 16);
    size_t sz = *(size_t*)p - 16;
    return sz;
}

static void* BBHOST_MS_ABI pe_LocalAlloc(uint32_t flags, size_t bytes) {
    return pe_HeapAlloc(nullptr, flags, bytes);
}

static void* BBHOST_MS_ABI pe_LocalFree(void* mem) {
    pe_HeapFree(nullptr, 0, mem);
    return nullptr;
}

static void* BBHOST_MS_ABI pe_malloc(size_t sz) { return malloc(sz ? sz : 1); }
static void* BBHOST_MS_ABI pe_calloc(size_t n, size_t sz) { return calloc(n ? n : 1, sz ? sz : 1); }
static void* BBHOST_MS_ABI pe_realloc(void* p, size_t sz) { return realloc(p, sz ? sz : 1); }
static void BBHOST_MS_ABI pe_free(void* p) { free(p); }
static void* BBHOST_MS_ABI pe_memcpy(void* dst, const void* src, size_t n) { return memcpy(dst, src, n); }
static void* BBHOST_MS_ABI pe_memmove(void* dst, const void* src, size_t n) { return memmove(dst, src, n); }
static void* BBHOST_MS_ABI pe_memset(void* dst, int c, size_t n) { return memset(dst, c, n); }
static int BBHOST_MS_ABI pe_memcmp(const void* s1, const void* s2, size_t n) { return memcmp(s1, s2, n); }
static int BBHOST_MS_ABI pe_strcmp(const char* s1, const char* s2) { return strcmp(s1, s2); }
static size_t BBHOST_MS_ABI pe_strlen(const char* s) { return strlen(s); }
static float BBHOST_MS_ABI pe_atanf(float x) { return atanf(x); }
static float BBHOST_MS_ABI pe_cosf(float x) { return cosf(x); }
static float BBHOST_MS_ABI pe_sinf(float x) { return sinf(x); }
static float BBHOST_MS_ABI pe_tanf(float x) { return tanf(x); }
static float BBHOST_MS_ABI pe_exp2f(float x) { return exp2f(x); }
static double BBHOST_MS_ABI pe_log2(double x) { return log2(x); }
static float BBHOST_MS_ABI pe_log2f(float x) { return log2f(x); }
static float BBHOST_MS_ABI pe_powf(float x, float y) { return powf(x, y); }
static double BBHOST_MS_ABI pe_floor(double x) { return floor(x); }
static float BBHOST_MS_ABI pe_floorf(float x) { return floorf(x); }
static float BBHOST_MS_ABI pe_roundf(float x) { return roundf(x); }
static double BBHOST_MS_ABI pe_sqrt(double x) { return sqrt(x); }
static float BBHOST_MS_ABI pe_sqrtf(float x) { return sqrtf(x); }

static uint32_t BBHOST_MS_ABI pe_VirtualProtect(void*, size_t, uint32_t, uint32_t* old_protect) {
    if (old_protect) *old_protect = 0x40; // PAGE_EXECUTE_READWRITE
    return 1;
}

// Strings & CodePages
static int BBHOST_MS_ABI pe_MultiByteToWideChar(uint32_t, uint32_t, const char* lpMultiByteStr, int cbMultiByte,
                                                win_wchar* lpWideCharStr, int cchWideChar) {
    if (!lpMultiByteStr) return 0;
    size_t in_len = cbMultiByte < 0 ? strlen(lpMultiByteStr) + 1 : (size_t)cbMultiByte;
    if (cchWideChar <= 0) return (int)in_len;
    size_t out_len = std::min((size_t)cchWideChar, in_len);
    for (size_t i = 0; i < out_len; ++i) {
        lpWideCharStr[i] = (win_wchar)(uint8_t)lpMultiByteStr[i];
    }
    return (int)out_len;
}

static int BBHOST_MS_ABI pe_WideCharToMultiByte(uint32_t, uint32_t, const win_wchar* lpWideCharStr, int cchWideChar,
                                                char* lpMultiByteStr, int cbMultiByte, const char*, int*) {
    if (!lpWideCharStr) return 0;
    size_t in_len = cchWideChar < 0 ? win_wcslen(lpWideCharStr) + 1 : (size_t)cchWideChar;
    if (cbMultiByte <= 0) return (int)in_len;
    size_t out_len = std::min((size_t)cbMultiByte, in_len);
    for (size_t i = 0; i < out_len; ++i) {
        lpMultiByteStr[i] = (char)(uint8_t)lpWideCharStr[i];
    }
    return (int)out_len;
}

// Error Handling & Environment
static uint32_t BBHOST_MS_ABI pe_GetLastError() { return t_last_error; }
static void BBHOST_MS_ABI pe_SetLastError(uint32_t err) { t_last_error = err; }

static uint32_t BBHOST_MS_ABI pe_GetEnvironmentVariableW(const win_wchar* lpName, win_wchar* lpBuffer, uint32_t nSize) {
    if (!lpName) return 0;
    std::string name = win_utf16_to_utf8(lpName);
    const char* val = getenv(name.c_str());
    if (!val) {
        t_last_error = 203; // ERROR_ENVVAR_NOT_FOUND
        return 0;
    }
    size_t len = strlen(val);
    if (nSize <= len) return (uint32_t)(len + 1);
    return (uint32_t)win_utf8_to_utf16(val, lpBuffer, nSize);
}

static uint32_t BBHOST_MS_ABI pe_SetEnvironmentVariableW(const win_wchar* lpName, const win_wchar* lpValue) {
    if (!lpName) return 0;
    std::string name = win_utf16_to_utf8(lpName);
    if (lpValue) {
        std::string val = win_utf16_to_utf8(lpValue);
        setenv(name.c_str(), val.c_str(), 1);
    } else {
        unsetenv(name.c_str());
    }
    return 1;
}

// Files & Paths
static void* BBHOST_MS_ABI pe_CreateFileW(const win_wchar* lpFileName, uint32_t dwDesiredAccess,
                                          uint32_t, void*, uint32_t dwCreationDisposition,
                                          uint32_t, void*) {
    if (!lpFileName) return (void*)-1;
    std::string path = win_utf16_to_utf8(lpFileName);
    int flags = 0;
    if ((dwDesiredAccess & 0xC0000000) == 0xC0000000) flags = O_RDWR;
    else if (dwDesiredAccess & 0x40000000) flags = O_WRONLY;
    else flags = O_RDONLY;

    if (dwCreationDisposition == 2) flags |= O_CREAT | O_TRUNC;
    else if (dwCreationDisposition == 4) flags |= O_CREAT;

    int fd = open(path.c_str(), flags, 0666);
    if (fd < 0) {
        std::string resolved = locate_pe_file(path.c_str());
        if (!resolved.empty()) {
            fd = open(resolved.c_str(), flags, 0666);
        }
    }
    if (fd < 0) {
        t_last_error = 2; // ERROR_FILE_NOT_FOUND
        return (void*)-1;
    }
    WinHandleObj* h = alloc_win_handle(WIN_HANDLE_FILE);
    h->fd = fd;
    return h;
}

static uint32_t BBHOST_MS_ABI pe_ReadFile(void* hFile, void* lpBuffer, uint32_t nNumberOfBytesToRead,
                                          uint32_t* lpNumberOfBytesRead, void*) {
    auto* h = (WinHandleObj*)hFile;
    if (!h || h->type != WIN_HANDLE_FILE || h->fd < 0) return 0;
    ssize_t n = read(h->fd, lpBuffer, nNumberOfBytesToRead);
    if (n < 0) return 0;
    if (lpNumberOfBytesRead) *lpNumberOfBytesRead = (uint32_t)n;
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_WriteFile(void* hFile, const void* lpBuffer, uint32_t nNumberOfBytesToWrite,
                                           uint32_t* lpNumberOfBytesWritten, void*) {
    auto* h = (WinHandleObj*)hFile;
    int fd = (h && h->type == WIN_HANDLE_FILE) ? h->fd : (hFile == (void*)1 ? 1 : 2);
    ssize_t n = write(fd, lpBuffer, nNumberOfBytesToWrite);
    if (n < 0) return 0;
    if (lpNumberOfBytesWritten) *lpNumberOfBytesWritten = (uint32_t)n;
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_GetFileSizeEx(void* hFile, int64_t* lpFileSize) {
    auto* h = (WinHandleObj*)hFile;
    if (!h || h->type != WIN_HANDLE_FILE || h->fd < 0 || !lpFileSize) return 0;
    struct stat st;
    if (fstat(h->fd, &st) != 0) return 0;
    *lpFileSize = st.st_size;
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_SetFilePointerEx(void* hFile, int64_t liDistanceToMove,
                                                  int64_t* lpNewFilePointer, uint32_t dwMoveMethod) {
    auto* h = (WinHandleObj*)hFile;
    if (!h || h->type != WIN_HANDLE_FILE || h->fd < 0) return 0;
    int whence = (dwMoveMethod == 1) ? SEEK_CUR : (dwMoveMethod == 2) ? SEEK_END : SEEK_SET;
    off_t pos = lseek(h->fd, (off_t)liDistanceToMove, whence);
    if (pos == (off_t)-1) return 0;
    if (lpNewFilePointer) *lpNewFilePointer = pos;
    return 1;
}

static uint32_t BBHOST_MS_ABI pe_SetEndOfFile(void* hFile) {
    auto* h = (WinHandleObj*)hFile;
    if (!h || h->type != WIN_HANDLE_FILE || h->fd < 0) return 0;
    off_t pos = lseek(h->fd, 0, SEEK_CUR);
    return ftruncate(h->fd, pos) == 0 ? 1 : 0;
}

static uint32_t BBHOST_MS_ABI pe_FlushFileBuffers(void* hFile) {
    auto* h = (WinHandleObj*)hFile;
    if (!h || h->type != WIN_HANDLE_FILE || h->fd < 0) return 0;
    fsync(h->fd);
    return 1;
}

static std::string locate_pe_file(const char* path);

static uint32_t BBHOST_MS_ABI pe_GetFileAttributesW(const win_wchar* lpFileName) {
    if (!lpFileName) return 0xFFFFFFFF;
    std::string path = win_utf16_to_utf8(lpFileName);
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0x10 : 0x80; // FILE_ATTRIBUTE_DIRECTORY / FILE_ATTRIBUTE_NORMAL
    }
    std::string resolved = locate_pe_file(path.c_str());
    if (!resolved.empty() && stat(resolved.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0x10 : 0x80;
    }
    return 0xFFFFFFFF;
}

static uint32_t BBHOST_MS_ABI pe_GetFullPathNameW(const win_wchar* lpFileName, uint32_t nBufferLength,
                                                  win_wchar* lpBuffer, win_wchar** lpFilePart) {
    if (!lpFileName || !lpBuffer) return 0;
    size_t len = win_wcslen(lpFileName);
    if (nBufferLength <= len) return (uint32_t)(len + 1);
    for (size_t i = 0; i <= len; ++i) lpBuffer[i] = lpFileName[i];
    if (lpFilePart) {
        win_wchar* p = nullptr;
        for (size_t i = len; i > 0; --i) {
            if (lpBuffer[i - 1] == '/' || lpBuffer[i - 1] == '\\') {
                p = &lpBuffer[i];
                break;
            }
        }
        *lpFilePart = p ? p : lpBuffer;
    }
    return (uint32_t)len;
}

static uint32_t BBHOST_MS_ABI pe_GetTempPathW(uint32_t nBufferLength, win_wchar* lpBuffer) {
    static const win_wchar s_tmp[] = { '/', 't', 'm', 'p', '/', 0 };
    if (!lpBuffer || nBufferLength < 6) return 6;
    for (size_t i = 0; i < 6; ++i) lpBuffer[i] = s_tmp[i];
    return 5;
}

static void* BBHOST_MS_ABI pe_FindFirstFileExW(const win_wchar*, int, void*, int, void*, uint32_t) {
    t_last_error = 18; // ERROR_NO_MORE_FILES
    return (void*)-1;
}

static uint32_t BBHOST_MS_ABI pe_FindNextFileW(void*, void*) { return 0; }
static uint32_t BBHOST_MS_ABI pe_FindClose(void*) { return 1; }

// Dynamic Loading & Modules
static void* BBHOST_MS_ABI pe_GetModuleHandleW(const win_wchar* lpModuleName) {
    if (!lpModuleName) {
        std::lock_guard<std::mutex> lk(g_modules_mu);
        return g_modules.empty() ? nullptr : g_modules[0]->base;
    }
    std::string name_utf8 = win_utf16_to_utf8(lpModuleName);
    std::string base_name = to_lower(get_file_name(name_utf8));
    std::lock_guard<std::mutex> lk(g_modules_mu);
    for (auto* m : g_modules) {
        if (m->name == base_name) return m->base;
    }
    return nullptr;
}

static void* BBHOST_MS_ABI pe_GetModuleHandleA(const char* lpModuleName) {
    if (!lpModuleName) return pe_GetModuleHandleW(nullptr);
    std::string base_name = to_lower(get_file_name(lpModuleName));
    std::lock_guard<std::mutex> lk(g_modules_mu);
    for (auto* m : g_modules) {
        if (m->name == base_name) return m->base;
    }
    return nullptr;
}

static uint32_t BBHOST_MS_ABI pe_GetModuleHandleExW(uint32_t, const win_wchar* lpModuleName, void** phModule) {
    if (!phModule) return 0;
    *phModule = pe_GetModuleHandleW(lpModuleName);
    return *phModule != nullptr ? 1 : 0;
}

static uint32_t BBHOST_MS_ABI pe_GetModuleFileNameW(void* hModule, win_wchar* lpFilename, uint32_t nSize) {
    if (!lpFilename || nSize == 0) return 0;
    std::string path;
    if (!hModule) {
        char exe_buf[1024] = {0};
        ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
        if (n > 0) path.assign(exe_buf, n);
    } else {
        std::lock_guard<std::mutex> lk(g_modules_mu);
        for (auto* m : g_modules) {
            if (m->base == hModule) {
                path = m->full_path;
                break;
            }
        }
    }
    if (path.empty()) {
        char exe_buf[1024] = {0};
        ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
        if (n > 0) path.assign(exe_buf, n);
        else path = "bbhost";
    }
    return (uint32_t)win_utf8_to_utf16(path.c_str(), lpFilename, nSize);
}

static uint32_t BBHOST_MS_ABI pe_GetModuleFileNameA(void* hModule, char* lpFilename, uint32_t nSize) {
    if (!lpFilename || nSize == 0) return 0;
    std::string path;
    if (!hModule) {
        char exe_buf[1024] = {0};
        ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
        if (n > 0) path.assign(exe_buf, n);
    } else {
        std::lock_guard<std::mutex> lk(g_modules_mu);
        for (auto* m : g_modules) {
            if (m->base == hModule) {
                path = m->full_path;
                break;
            }
        }
    }
    if (path.empty()) {
        char exe_buf[1024] = {0};
        ssize_t n = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
        if (n > 0) path.assign(exe_buf, n);
        else path = "bbhost";
    }
    size_t len = path.length();
    if (nSize <= len) len = nSize - 1;
    memcpy(lpFilename, path.c_str(), len);
    lpFilename[len] = '\0';
    return (uint32_t)len;
}

static void* BBHOST_MS_ABI pe_LoadLibraryW(const win_wchar* lpLibFileName);

static void* BBHOST_MS_ABI pe_LoadLibraryA(const char* lpLibFileName) {
    if (!lpLibFileName) return nullptr;
    std::string name = to_lower(get_file_name(lpLibFileName));

    // Handle dummy Windows sentinels
    if (name == "gdi32.dll" || name == "gdi32") return (void*)0xDEADBEEF00000001ULL;
    if (name == "vulkan-1.dll" || name == "vulkan-1") return (void*)0xDEADBEEF00000002ULL;
    if (name == "kernel32.dll" || name == "kernel32") return (void*)0xDEADBEEF00000003ULL;

    // Explicitly reject non-Vulkan/inapplicable vendor libraries
    if (name == "amdihk64.dll" || name == "nvngx_dlss.dll" || name == "_nvngx.dll" ||
        name.find("_dx12.dll") != std::string::npos) {
        return nullptr;
    }

    return pe_load_library(lpLibFileName);
}

static void* BBHOST_MS_ABI pe_LoadLibraryW(const win_wchar* lpLibFileName) {
    if (!lpLibFileName) return nullptr;
    std::string path = win_utf16_to_utf8(lpLibFileName);
    return pe_LoadLibraryA(path.c_str());
}

static void* BBHOST_MS_ABI pe_LoadLibraryExA(const char* lpLibFileName, void*, uint32_t) {
    return pe_LoadLibraryA(lpLibFileName);
}

static void* BBHOST_MS_ABI pe_LoadLibraryExW(const win_wchar* lpLibFileName, void*, uint32_t) {
    return pe_LoadLibraryW(lpLibFileName);
}

static void* BBHOST_MS_ABI pe_GetProcAddress(void* hModule, const char* lpProcName) {
    if (!hModule || !lpProcName) return nullptr;

    if (hModule == (void*)0xDEADBEEF00000001ULL) {
        // gdi32.dll mock
        if (strcmp(lpProcName, "D3DKMTEnumAdapters") == 0) return (void*)&pe_D3DKMTEnumAdapters;
        if (strcmp(lpProcName, "D3DKMTEnumAdapters2") == 0) return (void*)&pe_D3DKMTEnumAdapters2;
        if (strcmp(lpProcName, "D3DKMTQueryAdapterInfo") == 0) return (void*)&pe_D3DKMTQueryAdapterInfo;
        if (strcmp(lpProcName, "D3DKMTCloseAdapter") == 0) return (void*)&pe_D3DKMTCloseAdapter;
        return nullptr;
    }
    if (hModule == (void*)0xDEADBEEF00000002ULL) {
        return (void*)pe_vk_get_instance_proc_addr(VK_NULL_HANDLE, lpProcName);
    }
    if (hModule == (void*)0xDEADBEEF00000003ULL) {
        return resolve_import("KERNEL32.dll", lpProcName, 0);
    }

    return pe_get_proc_address(hModule, lpProcName);
}

static uint32_t BBHOST_MS_ABI pe_FreeLibrary(void* hModule) {
    if (!hModule || (uintptr_t)hModule >= 0xDEADBEEF00000000ULL) return 1;
    pe_free_library(hModule);
    return 1;
}

// User32 & Display Stubs
static uint32_t BBHOST_MS_ABI pe_EnumWindows(void*, void*) { return 1; }
static uint32_t BBHOST_MS_ABI pe_GetClientRect(void*, void* lpRect) {
    if (lpRect) memset(lpRect, 0, 16);
    return 1;
}
static uint32_t BBHOST_MS_ABI pe_IsWindowVisible(void*) { return 1; }
static uint32_t BBHOST_MS_ABI pe_EqualRect(const void* r1, const void* r2) {
    return (r1 && r2 && memcmp(r1, r2, 16) == 0) ? 1 : 0;
}
static uint32_t BBHOST_MS_ABI pe_OffsetRect(void* lpRect, int dx, int dy) {
    if (!lpRect) return 0;
    auto* p = (int32_t*)lpRect;
    p[0] += dx; p[2] += dx;
    p[1] += dy; p[3] += dy;
    return 1;
}
static void* BBHOST_MS_ABI pe_MonitorFromWindow(void*, uint32_t) { return (void*)1; }
static uint32_t BBHOST_MS_ABI pe_GetWindowThreadProcessId(void*, uint32_t* lpdwProcessId) {
    if (lpdwProcessId) *lpdwProcessId = pe_GetCurrentProcessId();
    return pe_GetCurrentThreadId();
}
static uint32_t BBHOST_MS_ABI pe_ClientToScreen(void*, void*) { return 1; }
static uint32_t BBHOST_MS_ABI pe_GetMonitorInfoW(void*, void* lpmi) {
    if (!lpmi) return 0;
    auto* p = (int32_t*)lpmi;
    // Fill monitor rect: 0, 0, 3840, 2160
    p[1] = 0; p[2] = 0; p[3] = 3840; p[4] = 2160;
    p[5] = 0; p[6] = 0; p[7] = 3840; p[8] = 2160;
    p[9] = 1; // MONITORINFOF_PRIMARY
    return 1;
}

// Registry Stubs
static uint32_t BBHOST_MS_ABI pe_RegOpenKeyExW(void*, const win_wchar*, uint32_t, uint32_t, void**) { return 2; }
static uint32_t BBHOST_MS_ABI pe_RegQueryValueExW(void*, const win_wchar*, uint32_t*, uint32_t*, void*, uint32_t*) { return 2; }
static uint32_t BBHOST_MS_ABI pe_RegCloseKey(void*) { return 0; }

// MSVCP140 Stubs
static const char* BBHOST_MS_ABI pe_Syserror_map(int) { return "system error"; }
static int BBHOST_MS_ABI pe_Winerror_map(int) { return 0; }
static void BBHOST_MS_ABI pe_Throw_Cpp_error(int) {}
static void BBHOST_MS_ABI pe_Xbad_alloc() {}
static void BBHOST_MS_ABI pe_Xlength_error(const char*) {}
static void BBHOST_MS_ABI pe_Xout_of_range(const char*) {}

static void BBHOST_MS_ABI pe_Locinfo_ctor(void* self, const char*) { memset(self, 0, 32); }
static void BBHOST_MS_ABI pe_Locinfo_dtor(void*) {}
static void BBHOST_MS_ABI pe_Lockit_ctor(void*, int) {}
static void BBHOST_MS_ABI pe_Lockit_dtor(void*) {}
static void BBHOST_MS_ABI pe_facet_ctor(void* self, size_t) { memset(self, 0, 16); }
static void BBHOST_MS_ABI pe_facet_dtor(void*) {}
static void* BBHOST_MS_ABI pe_facet_Decref(void*) { return nullptr; }
static void BBHOST_MS_ABI pe_facet_Incref(void*) {}

struct Cvtvec {
    uint32_t Page;
    uint32_t Mbcurmax;
    uint32_t Isclocale;
    uint8_t  Isleadbyte[32];
};

static void* BBHOST_MS_ABI pe_Locinfo_Getcvt(void* ret_buf, void*) {
    if (!ret_buf) return nullptr;
    auto* cvt = (Cvtvec*)ret_buf;
    cvt->Page = 65001; // CP_UTF8
    cvt->Mbcurmax = 4;
    cvt->Isclocale = 1;
    memset(cvt->Isleadbyte, 0, 32);
    return ret_buf;
}

static const char* BBHOST_MS_ABI pe_Locinfo_Getfalse(void*) { return "false"; }
static const char* BBHOST_MS_ABI pe_Locinfo_Gettrue(void*) { return "true"; }
static void* BBHOST_MS_ABI pe_locale_Getgloballocale() { static uint8_t s_loc[64]; return s_loc; }
static const struct lconv* BBHOST_MS_ABI pe_Locinfo_Getlconv(void*) { return localeconv(); }
static int pe_locale_id_cnt = 0;
static void* BBHOST_MS_ABI pe_locale_Init(void*, bool) { static uint8_t s_loc[64]; return s_loc; }
static uint32_t pe_id_numpunct_D = 0;
static uint32_t pe_id_numpunct_W = 0;

static int BBHOST_MS_ABI pe_Mbrtowc(win_wchar* pwc, const char* s, size_t n, mbstate_t* pst, const void*) {
    if (!s || n == 0) return 0;
    if (pwc) *pwc = (win_wchar)(uint8_t)*s;
    return 1;
}

// CRT Stdio Common Formatting
struct SysVVaList {
    unsigned int gp_offset;
    unsigned int fp_offset;
    void* overflow_arg_area;
    void* reg_save_area;
};

static int BBHOST_MS_ABI pe_stdio_common_vsprintf(uint64_t, char* buf, size_t len, const char* format, void*, void* ms_args) {
    if (!format) return -1;
    SysVVaList vl{48, 304, ms_args, nullptr};
    va_list ap;
    memcpy(&ap, &vl, sizeof(vl));
    if (!buf || len == 0) {
        char dummy[1];
        return vsnprintf(dummy, 0, format, ap);
    }
    return vsnprintf(buf, len, format, ap);
}

static int BBHOST_MS_ABI pe_stdio_common_vswprintf(uint64_t, win_wchar* buf, size_t len, const win_wchar* format, void*, void* ms_args) {
    if (!buf || !format || len == 0) return -1;
    std::string u8_fmt = win_utf16_to_utf8(format);
    std::vector<char> u8_buf(std::max<size_t>(len * 4, 1024));
    SysVVaList vl{48, 304, ms_args, nullptr};
    va_list ap;
    memcpy(&ap, &vl, sizeof(vl));
    int ret = vsnprintf(u8_buf.data(), u8_buf.size(), u8_fmt.c_str(), ap);
    if (ret >= 0) {
        size_t n = win_utf8_to_utf16(u8_buf.data(), buf, len);
        return (int)n;
    }
    buf[0] = 0;
    return 0;
}

// CRT Math Helpers
static int BBHOST_MS_ABI pe_dclass(double x) { return std::fpclassify(x); }
static int BBHOST_MS_ABI pe_dsign(double x) { return std::signbit(x) ? 1 : 0; }
static int BBHOST_MS_ABI pe_fdclass(float x) { return std::fpclassify(x); }
static int BBHOST_MS_ABI pe_fdsign(float x) { return std::signbit(x) ? 1 : 0; }
static int BBHOST_MS_ABI pe_ldclass(long double x) { return std::fpclassify(x); }
static int BBHOST_MS_ABI pe_ldsign(long double x) { return std::signbit(x) ? 1 : 0; }

// Miscellaneous CRT & Windows API Stubs
static void BBHOST_MS_ABI pe_OutputDebugStringA(const char* s) {
    if (s && *s) host_log("pe_loader: dbg: %s", s);
}

static void BBHOST_MS_ABI pe_OutputDebugStringW(const win_wchar* s) {
    if (s && *s) {
        std::string u8 = win_utf16_to_utf8(s);
        host_log("pe_loader: dbg: %s", u8.c_str());
    }
}

static uint32_t BBHOST_MS_ABI pe_FormatMessageA(uint32_t, const void*, uint32_t, uint32_t, char* lpBuffer, uint32_t nSize, void*) {
    if (!lpBuffer || nSize == 0) return 0;
    snprintf(lpBuffer, nSize, "error");
    return (uint32_t)strlen(lpBuffer);
}

static void* BBHOST_MS_ABI pe_GetStdHandle(uint32_t nStdHandle) {
    return (nStdHandle == 0xFFFFFFF5) ? (void*)1 : (void*)2;
}
static uint32_t BBHOST_MS_ABI pe_GetFileType(void*) { return 2; } // FILE_TYPE_CHAR
static uint32_t BBHOST_MS_ABI pe_GetConsoleMode(void*, uint32_t* mode) {
    if (mode) *mode = 7;
    return 1;
}
static uint32_t BBHOST_MS_ABI pe_WriteConsoleA(void*, const void* buf, uint32_t n, uint32_t* written, void*) {
    write(1, buf, n);
    if (written) *written = n;
    return 1;
}
static uint32_t BBHOST_MS_ABI pe_WriteConsoleW(void*, const win_wchar* buf, uint32_t n, uint32_t* written, void*) {
    if (buf && n) {
        std::string u8 = win_utf16_to_utf8(buf, (int)n);
        write(1, u8.data(), u8.size());
    }
    if (written) *written = n;
    return 1;
}
static uint32_t BBHOST_MS_ABI pe_IsDebuggerPresent() { return 0; }
static uint32_t BBHOST_MS_ABI pe_IsProcessorFeaturePresent(uint32_t feature) {
    if (feature == 23) return 0; // PF_FASTFAIL_AVAILABLE
    return 1;
}
static void* BBHOST_MS_ABI pe_EncodePointer(void* ptr) { return ptr; }
static void* BBHOST_MS_ABI pe_DecodePointer(void* ptr) { return ptr; }
static uint32_t BBHOST_MS_ABI pe_TlsAlloc() { return 0; }
static uint32_t BBHOST_MS_ABI pe_TlsFree(uint32_t) { return 1; }
static void* BBHOST_MS_ABI pe_TlsGetValue(uint32_t) { return nullptr; }
static uint32_t BBHOST_MS_ABI pe_TlsSetValue(uint32_t, void*) { return 1; }
static uint32_t BBHOST_MS_ABI pe_FlsAlloc(void*) { return 0; }
static uint32_t BBHOST_MS_ABI pe_FlsFree(uint32_t) { return 1; }
static void* BBHOST_MS_ABI pe_FlsGetValue(uint32_t) { return nullptr; }
static uint32_t BBHOST_MS_ABI pe_FlsSetValue(uint32_t, void*) { return 1; }
static void BBHOST_MS_ABI pe_InitializeSListHead(void* head) { if (head) memset(head, 0, 16); }
static void* BBHOST_MS_ABI pe_InterlockedFlushSList(void* head) { if (head) memset(head, 0, 16); return nullptr; }
static void BBHOST_MS_ABI pe_TerminateProcess(void*, uint32_t) { abort(); }
static void BBHOST_MS_ABI pe_ExitProcess(uint32_t code) { exit((int)code); }
static void BBHOST_MS_ABI pe_RaiseException(uint32_t, uint32_t, uint32_t, const void*) { abort(); }
static void* BBHOST_MS_ABI pe_SetUnhandledExceptionFilter(void*) { return nullptr; }
static int32_t BBHOST_MS_ABI pe_UnhandledExceptionFilter(void*) { return 0; }
static void BBHOST_MS_ABI pe_RtlCaptureContext(void*) {}
static void* BBHOST_MS_ABI pe_RtlLookupFunctionEntry(uint64_t, uint64_t*, void*) { return nullptr; }
static void* BBHOST_MS_ABI pe_RtlVirtualUnwind(uint32_t, uint64_t, uint64_t, void*, void*, void**, uint64_t*, void*) { return nullptr; }
static void* BBHOST_MS_ABI pe_RtlPcToFileHeader(void*, void**) { return nullptr; }

static uint32_t BBHOST_MS_ABI pe_initterm(void (**start)(), void (**end)()) {
    while (start < end) {
        if (*start) (*start)();
        ++start;
    }
    return 0;
}

static uint32_t BBHOST_MS_ABI pe_initterm_e(int (BBHOST_MS_ABI **start)(), int (BBHOST_MS_ABI **end)()) {
    while (start < end) {
        if (*start) {
            int err = (*start)();
            if (err) return err;
        }
        ++start;
    }
    return 0;
}

static int BBHOST_MS_ABI pe_crt_atexit(void (*)()) { return 0; }
static int BBHOST_MS_ABI pe_configure_narrow_argv(int) { return 0; }
static int BBHOST_MS_ABI pe_initialize_narrow_environment() { return 0; }
static int BBHOST_MS_ABI pe_initialize_onexit_table(void*) { return 0; }
static int BBHOST_MS_ABI pe_register_onexit_function(void*, void*) { return 0; }
static int BBHOST_MS_ABI pe_execute_onexit_table(void*) { return 0; }
static void BBHOST_MS_ABI pe_cexit() {}
static int BBHOST_MS_ABI pe_seh_filter_dll(uint32_t, void*) { return 0; }
static void BBHOST_MS_ABI pe_terminate() { abort(); }
static void BBHOST_MS_ABI pe_purecall() { abort(); }
static void BBHOST_MS_ABI pe_std_terminate() { abort(); }
static void BBHOST_MS_ABI pe_std_type_info_destroy_list(void*) {}
static void BBHOST_MS_ABI pe_std_exception_copy(void* dst, const void* src) { memcpy(dst, src, 16); }
static void BBHOST_MS_ABI pe_std_exception_destroy(void*) {}
static int BBHOST_MS_ABI pe_callnewh(size_t) { return 0; }

static int BBHOST_MS_ABI pe_getenv_s(size_t* pReturnValue, char* buffer, size_t bufferSize, const char* varname) {
    if (!varname) return EINVAL;
    const char* val = getenv(varname);
    if (!val) {
        if (pReturnValue) *pReturnValue = 0;
        if (buffer && bufferSize > 0) buffer[0] = '\0';
        return 0;
    }
    size_t len = strlen(val);
    if (pReturnValue) *pReturnValue = len + 1;
    if (!buffer || bufferSize <= len) return ERANGE;
    memcpy(buffer, val, len + 1);
    return 0;
}

static int BBHOST_MS_ABI pe_wgetenv_s(size_t* pReturnValue, win_wchar* buffer, size_t numberOfElements, const win_wchar* varname) {
    if (!varname) return EINVAL;
    std::string name_utf8 = win_utf16_to_utf8(varname);
    const char* val = getenv(name_utf8.c_str());
    if (!val) {
        if (pReturnValue) *pReturnValue = 0;
        if (buffer && numberOfElements > 0) buffer[0] = 0;
        return 0;
    }
    size_t len = strlen(val);
    if (pReturnValue) *pReturnValue = len + 1;
    if (!buffer || numberOfElements <= len) return ERANGE;
    win_utf8_to_utf16(val, buffer, numberOfElements);
    return 0;
}

static int BBHOST_MS_ABI pe_strncpy_s(char* strDest, size_t numberOfElements, const char* strSource, size_t count) {
    if (!strDest || numberOfElements == 0) return EINVAL;
    if (!strSource) { strDest[0] = '\0'; return EINVAL; }
    size_t src_len = strlen(strSource);
    size_t n = count < src_len ? count : src_len;
    if (n >= numberOfElements) { strDest[0] = '\0'; return ERANGE; }
    memcpy(strDest, strSource, n);
    strDest[n] = '\0';
    return 0;
}

static int BBHOST_MS_ABI pe_wcscpy_s(win_wchar* strDestination, size_t numberOfElements, const win_wchar* strSource) {
    if (!strDestination || numberOfElements == 0) return EINVAL;
    if (!strSource) { strDestination[0] = 0; return EINVAL; }
    size_t len = win_wcslen(strSource);
    if (len >= numberOfElements) { strDestination[0] = 0; return ERANGE; }
    memcpy(strDestination, strSource, (len + 1) * sizeof(win_wchar));
    return 0;
}

// Exception Handling Frame stubs
static int BBHOST_MS_ABI pe_C_specific_handler(void*, void*, void*, void*) { return 1; }
static int BBHOST_MS_ABI pe_CxxFrameHandler4(void*, void*, void*, void*) { return 1; }
static void BBHOST_MS_ABI pe_CxxThrowException(void*, void*) { abort(); }
static void* BBHOST_MS_ABI pe_current_exception() { return nullptr; }
static void* BBHOST_MS_ABI pe_current_exception_context() { return nullptr; }

static void BBHOST_MS_ABI pe_GetStartupInfoW(void* lpStartupInfo) {
    if (lpStartupInfo) {
        memset(lpStartupInfo, 0, 104);
        *(uint32_t*)lpStartupInfo = 104;
    }
}
static const char* BBHOST_MS_ABI pe_GetCommandLineA() { return "bbhost"; }
static const win_wchar* BBHOST_MS_ABI pe_GetCommandLineW() {
    static const win_wchar s_cmd[] = { 'b', 'b', 'h', 'o', 's', 't', 0 };
    return s_cmd;
}
static uint32_t BBHOST_MS_ABI pe_GetACP() { return 65001; }
static uint32_t BBHOST_MS_ABI pe_GetOEMCP() { return 65001; }
static uint32_t BBHOST_MS_ABI pe_GetCPInfo(uint32_t, void* lpCPInfo) {
    if (lpCPInfo) memset(lpCPInfo, 0, 16);
    return 1;
}
static uint32_t BBHOST_MS_ABI pe_IsValidCodePage(uint32_t) { return 1; }
static uint32_t BBHOST_MS_ABI pe_GetConsoleOutputCP() { return 65001; }
static void* BBHOST_MS_ABI pe_GetConsoleWindow() { return nullptr; }
static uint32_t BBHOST_MS_ABI pe_AllocConsole() { return 1; }
static uint32_t BBHOST_MS_ABI pe_SetConsoleTitleA(const char*) { return 1; }
static uint32_t BBHOST_MS_ABI pe_SetStdHandle(uint32_t, void*) { return 1; }
static uint32_t BBHOST_MS_ABI pe_AreFileApisANSI() { return 1; }
static int BBHOST_MS_ABI pe_CompareStringW(uint32_t, uint32_t, const win_wchar* lpString1, int cchCount1,
                                           const win_wchar* lpString2, int cchCount2) {
    if (!lpString1 || !lpString2) return 2;
    std::string s1 = win_utf16_to_utf8(lpString1, cchCount1);
    std::string s2 = win_utf16_to_utf8(lpString2, cchCount2);
    int cmp = s1.compare(s2);
    return cmp < 0 ? 1 : (cmp > 0 ? 3 : 2);
}
static int BBHOST_MS_ABI pe_LCMapStringW(uint32_t, uint32_t, const win_wchar* lpSrcStr, int cchSrc,
                                         win_wchar* lpDestStr, int cchDest) {
    if (!lpSrcStr) return 0;
    int len = cchSrc < 0 ? (int)win_wcslen(lpSrcStr) + 1 : cchSrc;
    if (cchDest <= 0) return len;
    int copy_len = std::min(len, cchDest);
    if (lpDestStr) memcpy(lpDestStr, lpSrcStr, copy_len * sizeof(win_wchar));
    return copy_len;
}
static uint32_t BBHOST_MS_ABI pe_GetStringTypeW(uint32_t, const win_wchar*, int, uint16_t* lpCharType) {
    if (lpCharType) *lpCharType = 1;
    return 1;
}
static int BBHOST_MS_ABI pe_GetLocaleInfoEx(const win_wchar*, uint32_t, win_wchar* lpLCData, int cchData) {
    if (lpLCData && cchData > 0) lpLCData[0] = 0;
    return 1;
}
static int BBHOST_MS_ABI pe_GetDateFormatW(uint32_t, uint32_t, const void*, const win_wchar*, win_wchar* lpDateStr, int cchDate) {
    if (lpDateStr && cchDate > 0) lpDateStr[0] = 0;
    return 1;
}
static int BBHOST_MS_ABI pe_GetTimeFormatW(uint32_t, uint32_t, const void*, const win_wchar*, win_wchar* lpTimeStr, int cchTime) {
    if (lpTimeStr && cchTime > 0) lpTimeStr[0] = 0;
    return 1;
}
static uint32_t BBHOST_MS_ABI pe_GetTimeZoneInformation(void* lpTimeZoneInformation) {
    if (lpTimeZoneInformation) memset(lpTimeZoneInformation, 0, 172);
    return 0;
}
static win_wchar* BBHOST_MS_ABI pe_GetEnvironmentStringsW() {
    static win_wchar s_env[] = { '=', 0, 0 };
    return s_env;
}
static uint32_t BBHOST_MS_ABI pe_FreeEnvironmentStringsW(win_wchar*) { return 1; }
static uint32_t BBHOST_MS_ABI pe_GetSystemDirectoryW(win_wchar* lpBuffer, uint32_t uSize) {
    static const win_wchar s_sys[] = { '/', 'u', 's', 'r', '/', 'l', 'i', 'b', 0 };
    if (!lpBuffer || uSize < 9) return 9;
    for (size_t i = 0; i < 9; ++i) lpBuffer[i] = s_sys[i];
    return 8;
}
static uint32_t BBHOST_MS_ABI pe_ReadConsoleW(void*, void*, uint32_t, uint32_t* read, void*) {
    if (read) *read = 0;
    return 1;
}
static void BBHOST_MS_ABI pe_RtlUnwindEx(void*, void*, void*, void*, void*, void*) {}
static void BBHOST_MS_ABI pe_DebugBreak() {}
static uint64_t BBHOST_MS_ABI pe_VerSetConditionMask(uint64_t condMask, uint32_t, uint8_t) { return condMask; }
static uint32_t BBHOST_MS_ABI pe_VerifyVersionInfoW(void*, uint32_t, uint64_t) { return 1; }
static uint64_t BBHOST_MS_ABI pe_dummy_stub() { return 0; }

static void* resolve_import(const char* dll_name, const char* func_name, uint16_t) {
    if (!func_name) return nullptr;
    std::string dname = to_lower(dll_name ? dll_name : "");

    // Vulkan imports
    if (dname == "vulkan-1.dll" || dname == "vulkan-1") {
        void* p = (void*)pe_vk_get_instance_proc_addr(VK_NULL_HANDLE, func_name);
        if (!p) {
            void* sym = dlsym(RTLD_DEFAULT, func_name);
            if (sym) {
                p = create_ms2sysv_stub(get_or_create_bridge_module(), sym);
            }
        }
        return p;
    }

    // GDI32 imports
    if (dname == "gdi32.dll" || dname == "gdi32") {
        if (strcmp(func_name, "D3DKMTEnumAdapters") == 0) return (void*)&pe_D3DKMTEnumAdapters;
        if (strcmp(func_name, "D3DKMTEnumAdapters2") == 0) return (void*)&pe_D3DKMTEnumAdapters2;
        if (strcmp(func_name, "D3DKMTQueryAdapterInfo") == 0) return (void*)&pe_D3DKMTQueryAdapterInfo;
        if (strcmp(func_name, "D3DKMTCloseAdapter") == 0) return (void*)&pe_D3DKMTCloseAdapter;
        return nullptr;
    }

    // VERSION.dll imports
    if (dname == "version.dll") {
        if (strcmp(func_name, "GetFileVersionInfoSizeW") == 0) return (void*)&pe_GetFileVersionInfoSizeW;
        if (strcmp(func_name, "GetFileVersionInfoSizeA") == 0) return (void*)&pe_GetFileVersionInfoSizeA;
        if (strcmp(func_name, "GetFileVersionInfoW") == 0) return (void*)&pe_GetFileVersionInfoW;
        if (strcmp(func_name, "GetFileVersionInfoA") == 0) return (void*)&pe_GetFileVersionInfoA;
        if (strcmp(func_name, "VerQueryValueW") == 0) return (void*)&pe_VerQueryValueW;
        if (strcmp(func_name, "VerQueryValueA") == 0) return (void*)&pe_VerQueryValueA;
    }

    // USER32.dll imports
    if (dname == "user32.dll") {
        if (strcmp(func_name, "EnumWindows") == 0) return (void*)&pe_EnumWindows;
        if (strcmp(func_name, "GetClientRect") == 0) return (void*)&pe_GetClientRect;
        if (strcmp(func_name, "IsWindowVisible") == 0) return (void*)&pe_IsWindowVisible;
        if (strcmp(func_name, "EqualRect") == 0) return (void*)&pe_EqualRect;
        if (strcmp(func_name, "OffsetRect") == 0) return (void*)&pe_OffsetRect;
        if (strcmp(func_name, "MonitorFromWindow") == 0) return (void*)&pe_MonitorFromWindow;
        if (strcmp(func_name, "GetWindowThreadProcessId") == 0) return (void*)&pe_GetWindowThreadProcessId;
        if (strcmp(func_name, "ClientToScreen") == 0) return (void*)&pe_ClientToScreen;
        if (strcmp(func_name, "GetMonitorInfoW") == 0) return (void*)&pe_GetMonitorInfoW;
    }

    // ADVAPI32.dll imports
    if (dname == "advapi32.dll") {
        if (strcmp(func_name, "RegOpenKeyExW") == 0) return (void*)&pe_RegOpenKeyExW;
        if (strcmp(func_name, "RegQueryValueExW") == 0) return (void*)&pe_RegQueryValueExW;
        if (strcmp(func_name, "RegCloseKey") == 0) return (void*)&pe_RegCloseKey;
    }

    // MSVCP140.dll imports
    if (dname == "msvcp140.dll") {
        if (strcmp(func_name, "_Mtx_init_in_situ") == 0) return (void*)&pe_Mtx_init_in_situ;
        if (strcmp(func_name, "_Mtx_lock") == 0) return (void*)&pe_Mtx_lock;
        if (strcmp(func_name, "_Mtx_unlock") == 0) return (void*)&pe_Mtx_unlock;
        if (strcmp(func_name, "?_Syserror_map@std@@YAPEBDH@Z") == 0) return (void*)&pe_Syserror_map;
        if (strcmp(func_name, "?_Winerror_map@std@@YAHH@Z") == 0) return (void*)&pe_Winerror_map;
        if (strcmp(func_name, "?_Throw_Cpp_error@std@@YAXH@Z") == 0) return (void*)&pe_Throw_Cpp_error;
        if (strcmp(func_name, "?_Xbad_alloc@std@@YAXXZ") == 0) return (void*)&pe_Xbad_alloc;
        if (strcmp(func_name, "?_Xlength_error@std@@YAXPEBD@Z") == 0) return (void*)&pe_Xlength_error;
        if (strcmp(func_name, "?_Xout_of_range@std@@YAXPEBD@Z") == 0) return (void*)&pe_Xout_of_range;
        if (strcmp(func_name, "??0_Locinfo@std@@QEAA@PEBD@Z") == 0) return (void*)&pe_Locinfo_ctor;
        if (strcmp(func_name, "??1_Locinfo@std@@QEAA@XZ") == 0) return (void*)&pe_Locinfo_dtor;
        if (strcmp(func_name, "??0_Lockit@std@@QEAA@H@Z") == 0) return (void*)&pe_Lockit_ctor;
        if (strcmp(func_name, "??1_Lockit@std@@QEAA@XZ") == 0) return (void*)&pe_Lockit_dtor;
        if (strcmp(func_name, "??0facet@locale@std@@IEAA@_K@Z") == 0) return (void*)&pe_facet_ctor;
        if (strcmp(func_name, "??1facet@locale@std@@MEAA@XZ") == 0) return (void*)&pe_facet_dtor;
        if (strcmp(func_name, "?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ") == 0) return (void*)&pe_facet_Decref;
        if (strcmp(func_name, "?_Incref@facet@locale@std@@UEAAXXZ") == 0) return (void*)&pe_facet_Incref;
        if (strcmp(func_name, "?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ") == 0) return (void*)&pe_Locinfo_Getcvt;
        if (strcmp(func_name, "?_Getfalse@_Locinfo@std@@QEBAPEBDXZ") == 0) return (void*)&pe_Locinfo_Getfalse;
        if (strcmp(func_name, "?_Gettrue@_Locinfo@std@@QEBAPEBDXZ") == 0) return (void*)&pe_Locinfo_Gettrue;
        if (strcmp(func_name, "?_Getgloballocale@locale@std@@CAPEAV_Locimp@12@XZ") == 0) return (void*)&pe_locale_Getgloballocale;
        if (strcmp(func_name, "?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ") == 0) return (void*)&pe_Locinfo_Getlconv;
        if (strcmp(func_name, "?_Id_cnt@id@locale@std@@0HA") == 0) return (void*)&pe_locale_id_cnt;
        if (strcmp(func_name, "?_Init@locale@std@@CAPEAV_Locimp@12@_N@Z") == 0) return (void*)&pe_locale_Init;
        if (strcmp(func_name, "?id@?$numpunct@D@std@@2V0locale@2@A") == 0) return (void*)&pe_id_numpunct_D;
        if (strcmp(func_name, "?id@?$numpunct@_W@std@@2V0locale@2@A") == 0) return (void*)&pe_id_numpunct_W;
        if (strcmp(func_name, "_Mbrtowc") == 0) return (void*)&pe_Mbrtowc;
    }

    // VCRUNTIME140.dll & VCRUNTIME140_1.dll imports
    if (dname == "vcruntime140.dll" || dname == "vcruntime140_1.dll") {
        if (strcmp(func_name, "memcpy") == 0) return (void*)&pe_memcpy;
        if (strcmp(func_name, "memmove") == 0) return (void*)&pe_memmove;
        if (strcmp(func_name, "memset") == 0) return (void*)&pe_memset;
        if (strcmp(func_name, "memcmp") == 0) return (void*)&pe_memcmp;
        if (strcmp(func_name, "__C_specific_handler") == 0) return (void*)&pe_C_specific_handler;
        if (strcmp(func_name, "__CxxFrameHandler4") == 0) return (void*)&pe_CxxFrameHandler4;
        if (strcmp(func_name, "_CxxThrowException") == 0) return (void*)&pe_CxxThrowException;
        if (strcmp(func_name, "__current_exception") == 0) return (void*)&pe_current_exception;
        if (strcmp(func_name, "__current_exception_context") == 0) return (void*)&pe_current_exception_context;
        if (strcmp(func_name, "__std_exception_copy") == 0) return (void*)&pe_std_exception_copy;
        if (strcmp(func_name, "__std_exception_destroy") == 0) return (void*)&pe_std_exception_destroy;
        if (strcmp(func_name, "__std_terminate") == 0) return (void*)&pe_std_terminate;
        if (strcmp(func_name, "__std_type_info_destroy_list") == 0) return (void*)&pe_std_type_info_destroy_list;
        if (strcmp(func_name, "_purecall") == 0) return (void*)&pe_purecall;
    }

    // api-ms-win-crt-* imports
    if (dname.find("api-ms-win-crt-") != std::string::npos) {
        if (strcmp(func_name, "malloc") == 0) return (void*)&pe_malloc;
        if (strcmp(func_name, "calloc") == 0) return (void*)&pe_calloc;
        if (strcmp(func_name, "realloc") == 0) return (void*)&pe_realloc;
        if (strcmp(func_name, "free") == 0) return (void*)&pe_free;
        if (strcmp(func_name, "memcpy") == 0) return (void*)&pe_memcpy;
        if (strcmp(func_name, "memmove") == 0) return (void*)&pe_memmove;
        if (strcmp(func_name, "memset") == 0) return (void*)&pe_memset;
        if (strcmp(func_name, "memcmp") == 0) return (void*)&pe_memcmp;
        if (strcmp(func_name, "_callnewh") == 0) return (void*)&pe_callnewh;
        if (strcmp(func_name, "strcmp") == 0) return (void*)&pe_strcmp;
        if (strcmp(func_name, "strlen") == 0) return (void*)&pe_strlen;
        if (strcmp(func_name, "strncpy_s") == 0) return (void*)&pe_strncpy_s;
        if (strcmp(func_name, "wcscpy_s") == 0) return (void*)&pe_wcscpy_s;
        if (strcmp(func_name, "getenv_s") == 0) return (void*)&pe_getenv_s;
        if (strcmp(func_name, "_wgetenv_s") == 0) return (void*)&pe_wgetenv_s;
        if (strcmp(func_name, "terminate") == 0) return (void*)&pe_terminate;
        if (strcmp(func_name, "_cexit") == 0) return (void*)&pe_cexit;
        if (strcmp(func_name, "_initterm") == 0) return (void*)&pe_initterm;
        if (strcmp(func_name, "_initterm_e") == 0) return (void*)&pe_initterm_e;
        if (strcmp(func_name, "_crt_atexit") == 0) return (void*)&pe_crt_atexit;
        if (strcmp(func_name, "_configure_narrow_argv") == 0) return (void*)&pe_configure_narrow_argv;
        if (strcmp(func_name, "_initialize_narrow_environment") == 0) return (void*)&pe_initialize_narrow_environment;
        if (strcmp(func_name, "_initialize_onexit_table") == 0) return (void*)&pe_initialize_onexit_table;
        if (strcmp(func_name, "_register_onexit_function") == 0) return (void*)&pe_register_onexit_function;
        if (strcmp(func_name, "_execute_onexit_table") == 0) return (void*)&pe_execute_onexit_table;
        if (strcmp(func_name, "_seh_filter_dll") == 0) return (void*)&pe_seh_filter_dll;
        if (strcmp(func_name, "_invoke_watson") == 0) return (void*)&pe_terminate;
        if (strcmp(func_name, "__stdio_common_vsprintf") == 0) return (void*)&pe_stdio_common_vsprintf;
        if (strcmp(func_name, "__stdio_common_vswprintf") == 0) return (void*)&pe_stdio_common_vswprintf;
        if (strcmp(func_name, "__stdio_common_vswprintf_s") == 0) return (void*)&pe_stdio_common_vswprintf;
        if (strcmp(func_name, "___lc_codepage_func") == 0) return (void*)&pe_TlsAlloc; // returns 0
        // Math
        if (strcmp(func_name, "atanf") == 0) return (void*)&pe_atanf;
        if (strcmp(func_name, "cosf") == 0) return (void*)&pe_cosf;
        if (strcmp(func_name, "sinf") == 0) return (void*)&pe_sinf;
        if (strcmp(func_name, "tanf") == 0) return (void*)&pe_tanf;
        if (strcmp(func_name, "exp2f") == 0) return (void*)&pe_exp2f;
        if (strcmp(func_name, "log2") == 0) return (void*)&pe_log2;
        if (strcmp(func_name, "log2f") == 0) return (void*)&pe_log2f;
        if (strcmp(func_name, "powf") == 0) return (void*)&pe_powf;
        if (strcmp(func_name, "floor") == 0) return (void*)&pe_floor;
        if (strcmp(func_name, "floorf") == 0) return (void*)&pe_floorf;
        if (strcmp(func_name, "roundf") == 0) return (void*)&pe_roundf;
        if (strcmp(func_name, "sqrt") == 0) return (void*)&pe_sqrt;
        if (strcmp(func_name, "sqrtf") == 0) return (void*)&pe_sqrtf;
        if (strcmp(func_name, "_dclass") == 0) return (void*)&pe_dclass;
        if (strcmp(func_name, "_dsign") == 0) return (void*)&pe_dsign;
        if (strcmp(func_name, "_fdclass") == 0) return (void*)&pe_fdclass;
        if (strcmp(func_name, "_fdsign") == 0) return (void*)&pe_fdsign;
        if (strcmp(func_name, "_ldclass") == 0) return (void*)&pe_ldclass;
        if (strcmp(func_name, "_ldsign") == 0) return (void*)&pe_ldsign;
    }

    // KERNEL32.dll imports
    if (dname == "kernel32.dll" || dname == "kernel32") {
        if (strcmp(func_name, "AcquireSRWLockExclusive") == 0) return (void*)&pe_AcquireSRWLockExclusive;
        if (strcmp(func_name, "ReleaseSRWLockExclusive") == 0) return (void*)&pe_ReleaseSRWLockExclusive;
        if (strcmp(func_name, "TryAcquireSRWLockExclusive") == 0) return (void*)&pe_TryAcquireSRWLockExclusive;
        if (strcmp(func_name, "SleepConditionVariableSRW") == 0) return (void*)&pe_SleepConditionVariableSRW;
        if (strcmp(func_name, "WakeAllConditionVariable") == 0) return (void*)&pe_WakeAllConditionVariable;
        if (strcmp(func_name, "InitializeCriticalSection") == 0) return (void*)&pe_InitializeCriticalSection;
        if (strcmp(func_name, "InitializeCriticalSectionAndSpinCount") == 0) return (void*)&pe_InitializeCriticalSectionAndSpinCount;
        if (strcmp(func_name, "InitializeCriticalSectionEx") == 0) return (void*)&pe_InitializeCriticalSectionEx;
        if (strcmp(func_name, "EnterCriticalSection") == 0) return (void*)&pe_EnterCriticalSection;
        if (strcmp(func_name, "TryEnterCriticalSection") == 0) return (void*)&pe_TryEnterCriticalSection;
        if (strcmp(func_name, "LeaveCriticalSection") == 0) return (void*)&pe_LeaveCriticalSection;
        if (strcmp(func_name, "DeleteCriticalSection") == 0) return (void*)&pe_DeleteCriticalSection;
        if (strcmp(func_name, "CreateEventW") == 0) return (void*)&pe_CreateEventW;
        if (strcmp(func_name, "SetEvent") == 0) return (void*)&pe_SetEvent;
        if (strcmp(func_name, "ResetEvent") == 0) return (void*)&pe_ResetEvent;
        if (strcmp(func_name, "WaitForSingleObject") == 0) return (void*)&pe_WaitForSingleObject;
        if (strcmp(func_name, "CloseHandle") == 0) return (void*)&pe_CloseHandle;
        if (strcmp(func_name, "CreateThread") == 0) return (void*)&pe_CreateThread;
        if (strcmp(func_name, "QueryPerformanceCounter") == 0) return (void*)&pe_QueryPerformanceCounter;
        if (strcmp(func_name, "QueryPerformanceFrequency") == 0) return (void*)&pe_QueryPerformanceFrequency;
        if (strcmp(func_name, "GetSystemTimeAsFileTime") == 0) return (void*)&pe_GetSystemTimeAsFileTime;
        if (strcmp(func_name, "GetTickCount64") == 0) return (void*)&pe_GetTickCount64;
        if (strcmp(func_name, "GetCurrentProcessId") == 0) return (void*)&pe_GetCurrentProcessId;
        if (strcmp(func_name, "GetCurrentThreadId") == 0) return (void*)&pe_GetCurrentThreadId;
        if (strcmp(func_name, "GetCurrentProcess") == 0) return (void*)&pe_GetCurrentProcess;
        if (strcmp(func_name, "GetCurrentThread") == 0) return (void*)&pe_GetCurrentThread;
        if (strcmp(func_name, "GetSystemInfo") == 0) return (void*)&pe_GetSystemInfo;
        if (strcmp(func_name, "GetProcessHeap") == 0) return (void*)&pe_GetProcessHeap;
        if (strcmp(func_name, "HeapAlloc") == 0) return (void*)&pe_HeapAlloc;
        if (strcmp(func_name, "HeapReAlloc") == 0) return (void*)&pe_HeapReAlloc;
        if (strcmp(func_name, "HeapFree") == 0) return (void*)&pe_HeapFree;
        if (strcmp(func_name, "HeapSize") == 0) return (void*)&pe_HeapSize;
        if (strcmp(func_name, "LocalAlloc") == 0) return (void*)&pe_LocalAlloc;
        if (strcmp(func_name, "LocalFree") == 0) return (void*)&pe_LocalFree;
        if (strcmp(func_name, "VirtualProtect") == 0) return (void*)&pe_VirtualProtect;
        if (strcmp(func_name, "MultiByteToWideChar") == 0) return (void*)&pe_MultiByteToWideChar;
        if (strcmp(func_name, "WideCharToMultiByte") == 0) return (void*)&pe_WideCharToMultiByte;
        if (strcmp(func_name, "GetLastError") == 0) return (void*)&pe_GetLastError;
        if (strcmp(func_name, "SetLastError") == 0) return (void*)&pe_SetLastError;
        if (strcmp(func_name, "GetEnvironmentVariableW") == 0) return (void*)&pe_GetEnvironmentVariableW;
        if (strcmp(func_name, "SetEnvironmentVariableW") == 0) return (void*)&pe_SetEnvironmentVariableW;
        if (strcmp(func_name, "CreateFileW") == 0) return (void*)&pe_CreateFileW;
        if (strcmp(func_name, "ReadFile") == 0) return (void*)&pe_ReadFile;
        if (strcmp(func_name, "WriteFile") == 0) return (void*)&pe_WriteFile;
        if (strcmp(func_name, "GetFileSizeEx") == 0) return (void*)&pe_GetFileSizeEx;
        if (strcmp(func_name, "SetFilePointerEx") == 0) return (void*)&pe_SetFilePointerEx;
        if (strcmp(func_name, "SetEndOfFile") == 0) return (void*)&pe_SetEndOfFile;
        if (strcmp(func_name, "FlushFileBuffers") == 0) return (void*)&pe_FlushFileBuffers;
        if (strcmp(func_name, "GetFileAttributesW") == 0) return (void*)&pe_GetFileAttributesW;
        if (strcmp(func_name, "GetFullPathNameW") == 0) return (void*)&pe_GetFullPathNameW;
        if (strcmp(func_name, "GetTempPathW") == 0) return (void*)&pe_GetTempPathW;
        if (strcmp(func_name, "FindFirstFileExW") == 0) return (void*)&pe_FindFirstFileExW;
        if (strcmp(func_name, "FindNextFileW") == 0) return (void*)&pe_FindNextFileW;
        if (strcmp(func_name, "FindClose") == 0) return (void*)&pe_FindClose;
        if (strcmp(func_name, "GetModuleHandleW") == 0) return (void*)&pe_GetModuleHandleW;
        if (strcmp(func_name, "GetModuleHandleA") == 0) return (void*)&pe_GetModuleHandleA;
        if (strcmp(func_name, "GetModuleHandleExW") == 0) return (void*)&pe_GetModuleHandleExW;
        if (strcmp(func_name, "GetModuleFileNameW") == 0) return (void*)&pe_GetModuleFileNameW;
        if (strcmp(func_name, "GetModuleFileNameA") == 0) return (void*)&pe_GetModuleFileNameA;
        if (strcmp(func_name, "LoadLibraryA") == 0) return (void*)&pe_LoadLibraryA;
        if (strcmp(func_name, "LoadLibraryW") == 0) return (void*)&pe_LoadLibraryW;
        if (strcmp(func_name, "LoadLibraryExA") == 0) return (void*)&pe_LoadLibraryExA;
        if (strcmp(func_name, "LoadLibraryExW") == 0) return (void*)&pe_LoadLibraryExW;
        if (strcmp(func_name, "GetProcAddress") == 0) return (void*)&pe_GetProcAddress;
        if (strcmp(func_name, "FreeLibrary") == 0) return (void*)&pe_FreeLibrary;
        if (strcmp(func_name, "OutputDebugStringA") == 0) return (void*)&pe_OutputDebugStringA;
        if (strcmp(func_name, "OutputDebugStringW") == 0) return (void*)&pe_OutputDebugStringW;
        if (strcmp(func_name, "FormatMessageA") == 0) return (void*)&pe_FormatMessageA;
        if (strcmp(func_name, "GetStdHandle") == 0) return (void*)&pe_GetStdHandle;
        if (strcmp(func_name, "GetFileType") == 0) return (void*)&pe_GetFileType;
        if (strcmp(func_name, "GetConsoleMode") == 0) return (void*)&pe_GetConsoleMode;
        if (strcmp(func_name, "WriteConsoleA") == 0) return (void*)&pe_WriteConsoleA;
        if (strcmp(func_name, "WriteConsoleW") == 0) return (void*)&pe_WriteConsoleW;
        if (strcmp(func_name, "IsDebuggerPresent") == 0) return (void*)&pe_IsDebuggerPresent;
        if (strcmp(func_name, "IsProcessorFeaturePresent") == 0) return (void*)&pe_IsProcessorFeaturePresent;
        if (strcmp(func_name, "EncodePointer") == 0) return (void*)&pe_EncodePointer;
        if (strcmp(func_name, "DecodePointer") == 0) return (void*)&pe_DecodePointer;
        if (strcmp(func_name, "TlsAlloc") == 0) return (void*)&pe_TlsAlloc;
        if (strcmp(func_name, "TlsFree") == 0) return (void*)&pe_TlsFree;
        if (strcmp(func_name, "TlsGetValue") == 0) return (void*)&pe_TlsGetValue;
        if (strcmp(func_name, "TlsSetValue") == 0) return (void*)&pe_TlsSetValue;
        if (strcmp(func_name, "FlsAlloc") == 0) return (void*)&pe_FlsAlloc;
        if (strcmp(func_name, "FlsFree") == 0) return (void*)&pe_FlsFree;
        if (strcmp(func_name, "FlsGetValue") == 0) return (void*)&pe_FlsGetValue;
        if (strcmp(func_name, "FlsSetValue") == 0) return (void*)&pe_FlsSetValue;
        if (strcmp(func_name, "InitializeSListHead") == 0) return (void*)&pe_InitializeSListHead;
        if (strcmp(func_name, "InterlockedFlushSList") == 0) return (void*)&pe_InterlockedFlushSList;
        if (strcmp(func_name, "TerminateProcess") == 0) return (void*)&pe_TerminateProcess;
        if (strcmp(func_name, "ExitProcess") == 0) return (void*)&pe_ExitProcess;
        if (strcmp(func_name, "RaiseException") == 0) return (void*)&pe_RaiseException;
        if (strcmp(func_name, "SetUnhandledExceptionFilter") == 0) return (void*)&pe_SetUnhandledExceptionFilter;
        if (strcmp(func_name, "UnhandledExceptionFilter") == 0) return (void*)&pe_UnhandledExceptionFilter;
        if (strcmp(func_name, "RtlCaptureContext") == 0) return (void*)&pe_RtlCaptureContext;
        if (strcmp(func_name, "RtlLookupFunctionEntry") == 0) return (void*)&pe_RtlLookupFunctionEntry;
        if (strcmp(func_name, "RtlVirtualUnwind") == 0) return (void*)&pe_RtlVirtualUnwind;
        if (strcmp(func_name, "RtlPcToFileHeader") == 0) return (void*)&pe_RtlPcToFileHeader;
        if (strcmp(func_name, "DisableThreadLibraryCalls") == 0) return (void*)&pe_TlsFree; // returns 1
        if (strcmp(func_name, "SetThreadPriority") == 0) return (void*)&pe_TlsFree;
        if (strcmp(func_name, "SetThreadDescription") == 0) return (void*)&pe_TlsAlloc; // returns 0
        if (strcmp(func_name, "GetStartupInfoW") == 0) return (void*)&pe_GetStartupInfoW;
        if (strcmp(func_name, "GetCommandLineA") == 0) return (void*)&pe_GetCommandLineA;
        if (strcmp(func_name, "GetCommandLineW") == 0) return (void*)&pe_GetCommandLineW;
        if (strcmp(func_name, "GetACP") == 0) return (void*)&pe_GetACP;
        if (strcmp(func_name, "GetOEMCP") == 0) return (void*)&pe_GetOEMCP;
        if (strcmp(func_name, "GetCPInfo") == 0) return (void*)&pe_GetCPInfo;
        if (strcmp(func_name, "IsValidCodePage") == 0) return (void*)&pe_IsValidCodePage;
        if (strcmp(func_name, "GetConsoleOutputCP") == 0) return (void*)&pe_GetConsoleOutputCP;
        if (strcmp(func_name, "GetConsoleWindow") == 0) return (void*)&pe_GetConsoleWindow;
        if (strcmp(func_name, "AllocConsole") == 0) return (void*)&pe_AllocConsole;
        if (strcmp(func_name, "SetConsoleTitleA") == 0) return (void*)&pe_SetConsoleTitleA;
        if (strcmp(func_name, "SetStdHandle") == 0) return (void*)&pe_SetStdHandle;
        if (strcmp(func_name, "AreFileApisANSI") == 0) return (void*)&pe_AreFileApisANSI;
        if (strcmp(func_name, "CompareStringW") == 0) return (void*)&pe_CompareStringW;
        if (strcmp(func_name, "LCMapStringW") == 0) return (void*)&pe_LCMapStringW;
        if (strcmp(func_name, "GetStringTypeW") == 0) return (void*)&pe_GetStringTypeW;
        if (strcmp(func_name, "GetLocaleInfoEx") == 0) return (void*)&pe_GetLocaleInfoEx;
        if (strcmp(func_name, "GetDateFormatW") == 0) return (void*)&pe_GetDateFormatW;
        if (strcmp(func_name, "GetTimeFormatW") == 0) return (void*)&pe_GetTimeFormatW;
        if (strcmp(func_name, "GetTimeZoneInformation") == 0) return (void*)&pe_GetTimeZoneInformation;
        if (strcmp(func_name, "GetEnvironmentStringsW") == 0) return (void*)&pe_GetEnvironmentStringsW;
        if (strcmp(func_name, "FreeEnvironmentStringsW") == 0) return (void*)&pe_FreeEnvironmentStringsW;
        if (strcmp(func_name, "GetSystemDirectoryW") == 0) return (void*)&pe_GetSystemDirectoryW;
        if (strcmp(func_name, "ReadConsoleW") == 0) return (void*)&pe_ReadConsoleW;
        if (strcmp(func_name, "RtlUnwindEx") == 0) return (void*)&pe_RtlUnwindEx;
        if (strcmp(func_name, "DebugBreak") == 0) return (void*)&pe_DebugBreak;
        if (strcmp(func_name, "VerSetConditionMask") == 0) return (void*)&pe_VerSetConditionMask;
        if (strcmp(func_name, "VerifyVersionInfoW") == 0) return (void*)&pe_VerifyVersionInfoW;
        return (void*)&pe_dummy_stub;
    }

    // Check if it's another loaded PE module
    {
        std::lock_guard<std::mutex> lk(g_modules_mu);
        for (auto* m : g_modules) {
            if (m->name == dname) {
                return pe_get_proc_address(m->base, func_name);
            }
        }
    }

    // Try loading dependent PE module
    void* mod = pe_load_library(dll_name);
    if (mod) {
        return pe_get_proc_address(mod, func_name);
    }

    host_log("pe_loader: unresolved import %s!%s", dll_name ? dll_name : "unknown", func_name);
    return nullptr;
}

static std::string locate_pe_file(const char* path) {
    if (!path || !*path) return "";
    std::string s = path;
    for (char& c : s) if (c == '\\') c = '/';
    std::filesystem::path p(s);
    std::error_code ec;
    if (std::filesystem::is_regular_file(p, ec)) {
        return std::filesystem::canonical(p, ec).string();
    }

    std::string fname = get_file_name(s);
    std::lock_guard<std::mutex> lk(g_modules_mu);

    for (const auto& d : g_search_dirs) {
        std::filesystem::path cand = std::filesystem::path(d) / fname;
        if (std::filesystem::is_regular_file(cand, ec)) {
            return std::filesystem::canonical(cand, ec).string();
        }
    }

    // Search beside exe
    char exe_path[1024];
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n > 0) {
        exe_path[n] = '\0';
        std::filesystem::path exe_dir = std::filesystem::path(exe_path).parent_path();
        std::filesystem::path cand = exe_dir / fname;
        if (std::filesystem::is_regular_file(cand, ec)) return cand.string();

        cand = exe_dir / "third_party/upscaler_bridge/bin" / fname;
        if (std::filesystem::is_regular_file(cand, ec)) return cand.string();
    }

    return "";
}

void* pe_load_library(const char* path) {
    std::string resolved = locate_pe_file(path);
    if (resolved.empty()) {
        host_log("pe_loader: cannot find PE library '%s'", path ? path : "null");
        return nullptr;
    }

    std::string base_name = to_lower(get_file_name(resolved));

    {
        std::lock_guard<std::mutex> lk(g_modules_mu);
        for (auto* m : g_modules) {
            if (m->name == base_name || m->full_path == resolved) {
                m->ref_count++;
                return m->base;
            }
        }
    }

    std::ifstream file(resolved, std::ios::binary | std::ios::ate);
    if (!file) {
        host_log("pe_loader: failed to open file '%s'", resolved.c_str());
        return nullptr;
    }
    size_t file_size = (size_t)file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> file_buf(file_size);
    if (!file.read((char*)file_buf.data(), file_size)) {
        host_log("pe_loader: failed to read file '%s'", resolved.c_str());
        return nullptr;
    }

    if (file_size < sizeof(IMAGE_DOS_HEADER)) return nullptr;
    auto* dos = (const IMAGE_DOS_HEADER*)file_buf.data();
    if (dos->e_magic != 0x5A4D || dos->e_lfanew <= 0 || (size_t)dos->e_lfanew + sizeof(IMAGE_NT_HEADERS64) > file_size) {
        host_log("pe_loader: '%s' is not a valid PE file", resolved.c_str());
        return nullptr;
    }

    auto* nt = (const IMAGE_NT_HEADERS64*)(file_buf.data() + dos->e_lfanew);
    if (nt->Signature != 0x00004550 || nt->OptionalHeader.Magic != 0x20B || nt->FileHeader.Machine != 0x8664) {
        host_log("pe_loader: '%s' is not an x86-64 PE32+ binary", resolved.c_str());
        return nullptr;
    }

    size_t image_size = nt->OptionalHeader.SizeOfImage;
    size_t page_aligned_size = (image_size + 4095) & ~4095ULL;
    size_t stub_arena_cap = 64 * 1024;
    size_t total_map_size = page_aligned_size + stub_arena_cap;

    void* mmap_base = mmap(nullptr, total_map_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mmap_base == MAP_FAILED) {
        host_log("pe_loader: failed to mmap %zu bytes for '%s'", total_map_size, resolved.c_str());
        return nullptr;
    }

    auto* mod = new LoadedModule();
    mod->name = base_name;
    mod->full_path = resolved;
    mod->base = (uint8_t*)mmap_base;
    mod->mapped_size = total_map_size;
    mod->image_size = image_size;
    mod->stub_arena = mod->base + page_aligned_size;
    mod->stub_arena_used = 0;
    mod->stub_arena_cap = stub_arena_cap;

    // Search for embedded VS_FIXEDFILEINFO in the file
    for (size_t i = 0; i + sizeof(VS_FIXEDFILEINFO) <= file_size; i += 4) {
        if (*(const uint32_t*)(file_buf.data() + i) == 0xFEEF04BD) {
            memcpy(&mod->fixed_file_info, file_buf.data() + i, sizeof(VS_FIXEDFILEINFO));
            mod->has_fixed_file_info = true;
            break;
        }
    }

    // Copy PE headers
    size_t headers_size = std::min((size_t)nt->OptionalHeader.SizeOfHeaders, file_size);
    memcpy(mod->base, file_buf.data(), headers_size);

    // Copy Sections
    auto* sec_headers = (const IMAGE_SECTION_HEADER*)((const uint8_t*)&nt->OptionalHeader + nt->FileHeader.SizeOfOptionalHeader);
    for (uint16_t i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& sec = sec_headers[i];
        if (!sec.VirtualAddress) continue;
        uint8_t* dest = mod->base + sec.VirtualAddress;
        if (sec.PointerToRawData && sec.SizeOfRawData) {
            size_t copy_size = std::min((size_t)sec.SizeOfRawData, file_size - (size_t)sec.PointerToRawData);
            memcpy(dest, file_buf.data() + sec.PointerToRawData, copy_size);
        }
    }

    // Register search directory for this DLL
    pe_add_search_dir(get_dir_name(resolved).c_str());

    {
        std::lock_guard<std::mutex> lk(g_modules_mu);
        g_modules.push_back(mod);
    }

    // Apply Base Relocations
    intptr_t delta = (intptr_t)mod->base - (intptr_t)nt->OptionalHeader.ImageBase;
    if (delta != 0 && nt->OptionalHeader.NumberOfRvaAndSizes > 5) {
        uint32_t reloc_rva = nt->OptionalHeader.DataDirectory[5].VirtualAddress;
        uint32_t reloc_size = nt->OptionalHeader.DataDirectory[5].Size;
        if (reloc_rva && reloc_size) {
            size_t offset = 0;
            while (offset + 8 <= reloc_size) {
                uint32_t page_rva = *(const uint32_t*)(mod->base + reloc_rva + offset);
                uint32_t block_size = *(const uint32_t*)(mod->base + reloc_rva + offset + 4);
                if (!block_size || offset + block_size > reloc_size) break;

                size_t entry_count = (block_size - 8) / 2;
                const auto* entries = (const uint16_t*)(mod->base + reloc_rva + offset + 8);
                for (size_t j = 0; j < entry_count; ++j) {
                    uint16_t type = entries[j] >> 12;
                    uint16_t off = entries[j] & 0xFFF;
                    if (type == 10) { // IMAGE_REL_BASED_DIR64
                        *(uint64_t*)(mod->base + page_rva + off) += delta;
                    }
                }
                offset += block_size;
            }
        }
    }

    // Process TLS Directory
    if (nt->OptionalHeader.NumberOfRvaAndSizes > 9) {
        uint32_t tls_rva = nt->OptionalHeader.DataDirectory[9].VirtualAddress;
        if (tls_rva) {
            auto* tls = (IMAGE_TLS_DIRECTORY64*)(mod->base + tls_rva);
            mod->tls_index = g_next_tls_index++;
            *(uint32_t*)tls->AddressOfIndex = mod->tls_index;
            mod->tls_template = (void*)tls->StartAddressOfRawData;
            mod->tls_template_size = (size_t)(tls->EndAddressOfRawData - tls->StartAddressOfRawData);
            mod->tls_zero_size = tls->SizeOfZeroFill;
        }
    }

    // Initialize TLS for calling thread
    init_thread_pe_tls();

    // Create per-module gs: stubs in stub arena
    auto create_gs_stub = [](LoadedModule* m, void* target_fn) -> uint8_t* {
        uint8_t* stub = m->stub_arena + m->stub_arena_used;
        m->stub_arena_used += 48;
        uint8_t seq[] = {
            0x9c,                   // pushfq
            0x51,                   // push rcx
            0x52,                   // push rdx
            0x41, 0x50,             // push r8
            0x41, 0x51,             // push r9
            0x41, 0x52,             // push r10
            0x41, 0x53,             // push r11
            0x57,                   // push rdi
            0x56,                   // push rsi
            0x48, 0xb8, 0,0,0,0,0,0,0,0, // movabs rax, target_fn
            0xff, 0xd0,             // call rax
            0x5e,                   // pop rsi
            0x5f,                   // pop rdi
            0x41, 0x5b,             // pop r11
            0x41, 0x5a,             // pop r10
            0x41, 0x59,             // pop r9
            0x41, 0x58,             // pop r8
            0x5a,                   // pop rdx
            0x59,                   // pop rcx
            0x9d,                   // popfq
            0xc3                    // ret
        };
        *(uint64_t*)&seq[15] = (uint64_t)target_fn;
        memcpy(stub, seq, sizeof(seq));
        return stub;
    };
    uint8_t* mod_stub_gs58 = create_gs_stub(mod, (void*)&get_pe_tls_array_c);
    uint8_t* mod_stub_gs30 = create_gs_stub(mod, (void*)&get_pe_teb_c);

    // Resolve Imports
    if (nt->OptionalHeader.NumberOfRvaAndSizes > 1) {
        uint32_t import_rva = nt->OptionalHeader.DataDirectory[1].VirtualAddress;
        if (import_rva) {
            auto* desc = (IMAGE_IMPORT_DESCRIPTOR*)(mod->base + import_rva);
            while (desc->Name && (desc->FirstThunk || desc->OriginalFirstThunk)) {
                const char* import_dll_name = (const char*)(mod->base + desc->Name);
                uint32_t thunk_rva = desc->OriginalFirstThunk ? desc->OriginalFirstThunk : desc->FirstThunk;
                auto* thunk = (uint64_t*)(mod->base + thunk_rva);
                auto* iat = (uint64_t*)(mod->base + desc->FirstThunk);

                while (*thunk) {
                    void* resolved_func = nullptr;
                    const char* sym_name = nullptr;
                    if (*thunk & 0x8000000000000000ULL) {
                        uint16_t ord = (uint16_t)(*thunk & 0xFFFF);
                        resolved_func = resolve_import(import_dll_name, nullptr, ord);
                    } else {
                        auto* by_name = (const uint8_t*)(mod->base + (*thunk & 0x7FFFFFFF));
                        sym_name = (const char*)(by_name + 2);
                        resolved_func = resolve_import(import_dll_name, sym_name, 0);
                    }

                    if (!resolved_func) {
                        host_log("pe_loader: missing symbol '%s' in %s imported by %s", sym_name ? sym_name : "<ord>", import_dll_name, mod->name.c_str());
                    }
                    *iat = (uint64_t)resolved_func;
                    ++thunk;
                    ++iat;
                }
                ++desc;
            }
        }
    }

    // Patch gs: instructions in executable sections
    for (uint16_t i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& sec = sec_headers[i];
        if (!(sec.Characteristics & 0x20000000)) continue; // IMAGE_SCN_MEM_EXECUTE
        uint8_t* code = mod->base + sec.VirtualAddress;
        size_t code_size = sec.VirtualSize ? sec.VirtualSize : sec.SizeOfRawData;

        for (size_t off = 0; off + 9 <= code_size; ++off) {
            // Pattern 1: 65 48 8b 04 25 58 00 00 00 (mov rax, gs:0x58)
            if (code[off] == 0x65 && code[off+1] == 0x48 && code[off+2] == 0x8b &&
                code[off+3] == 0x04 && code[off+4] == 0x25 && code[off+5] == 0x58 &&
                code[off+6] == 0x00 && code[off+7] == 0x00 && code[off+8] == 0x00) {
                int64_t rel = (int64_t)(uintptr_t)mod_stub_gs58 - (int64_t)(uintptr_t)&code[off + 5];
                code[off] = 0xE8; // call rel32
                *(int32_t*)&code[off + 1] = (int32_t)rel;
                code[off + 5] = 0x90; // nop
                code[off + 6] = 0x90;
                code[off + 7] = 0x90;
                code[off + 8] = 0x90;
                off += 8;
            }
            // Pattern 2: 65 48 8b 04 25 30 00 00 00 (mov rax, gs:0x30)
            else if (code[off] == 0x65 && code[off+1] == 0x48 && code[off+2] == 0x8b &&
                     code[off+3] == 0x04 && code[off+4] == 0x25 && code[off+5] == 0x30 &&
                     code[off+6] == 0x00 && code[off+7] == 0x00 && code[off+8] == 0x00) {
                int64_t rel = (int64_t)(uintptr_t)mod_stub_gs30 - (int64_t)(uintptr_t)&code[off + 5];
                code[off] = 0xE8;
                *(int32_t*)&code[off + 1] = (int32_t)rel;
                code[off + 5] = 0x90;
                code[off + 6] = 0x90;
                code[off + 7] = 0x90;
                code[off + 8] = 0x90;
                off += 8;
            }
            // Pattern 3: 65 4c 8b 1c 25 10 00 00 00 (mov r11, gs:0x10)
            else if (code[off] == 0x65 && code[off+1] == 0x4C && code[off+2] == 0x8B &&
                     code[off+3] == 0x1C && code[off+4] == 0x25 && code[off+5] == 0x10 &&
                     code[off+6] == 0x00 && code[off+7] == 0x00 && code[off+8] == 0x00) {
                // Inline replacement: mov $0x1000, %r11 (49 c7 c3 00 10 00 00) + 2 NOPs
                code[off]   = 0x49;
                code[off+1] = 0xC7;
                code[off+2] = 0xC3;
                code[off+3] = 0x00;
                code[off+4] = 0x10;
                code[off+5] = 0x00;
                code[off+6] = 0x00;
                code[off+7] = 0x90;
                code[off+8] = 0x90;
                off += 8;
            }
        }
    }

    // Call TLS Callbacks
    if (nt->OptionalHeader.NumberOfRvaAndSizes > 9) {
        uint32_t tls_rva = nt->OptionalHeader.DataDirectory[9].VirtualAddress;
        if (tls_rva) {
            auto* tls = (const IMAGE_TLS_DIRECTORY64*)(mod->base + tls_rva);
            if (tls->AddressOfCallbacks) {
                typedef void (BBHOST_MS_ABI *PIMAGE_TLS_CALLBACK)(void*, uint32_t, void*);
                auto** cb = (PIMAGE_TLS_CALLBACK*)tls->AddressOfCallbacks;
                while (cb && *cb) {
                    (*cb)(mod->base, 1 /* DLL_PROCESS_ATTACH */, nullptr);
                    ++cb;
                }
            }
        }
    }

    // Call DllMain
    if (nt->OptionalHeader.AddressOfEntryPoint) {
        typedef uint32_t (BBHOST_MS_ABI *DllMain_t)(void*, uint32_t, void*);
        auto entry = (DllMain_t)(mod->base + nt->OptionalHeader.AddressOfEntryPoint);
        uint32_t ok = entry(mod->base, 1 /* DLL_PROCESS_ATTACH */, nullptr);
        if (!ok) {
            host_log("pe_loader: DllMain returned FALSE for %s", mod->name.c_str());
        }
    }

    host_log("pe_loader: loaded %s at %p (size %zu)", mod->name.c_str(), mod->base, mod->image_size);
    return mod->base;
}

void* pe_get_proc_address(void* module_base, const char* name) {
    if (!module_base || !name) return nullptr;
    auto* dos = (const IMAGE_DOS_HEADER*)module_base;
    auto* nt = (const IMAGE_NT_HEADERS64*)((const uint8_t*)module_base + dos->e_lfanew);

    if (nt->OptionalHeader.NumberOfRvaAndSizes == 0) return nullptr;
    uint32_t export_rva = nt->OptionalHeader.DataDirectory[0].VirtualAddress;
    uint32_t export_size = nt->OptionalHeader.DataDirectory[0].Size;
    if (!export_rva || !export_size) return nullptr;

    auto* exp = (const IMAGE_EXPORT_DIRECTORY*)((const uint8_t*)module_base + export_rva);
    const auto* names = (const uint32_t*)((const uint8_t*)module_base + exp->AddressOfNames);
    const auto* ordinals = (const uint16_t*)((const uint8_t*)module_base + exp->AddressOfNameOrdinals);
    const auto* functions = (const uint32_t*)((const uint8_t*)module_base + exp->AddressOfFunctions);

    for (uint32_t i = 0; i < exp->NumberOfNames; ++i) {
        const char* func_name = (const char*)((const uint8_t*)module_base + names[i]);
        if (strcmp(func_name, name) == 0) {
            uint16_t ord = ordinals[i];
            uint32_t func_rva = functions[ord];
            return (void*)((uint8_t*)module_base + func_rva);
        }
    }

    return nullptr;
}

void pe_free_library(void* module_base) {
    if (!module_base) return;
    std::lock_guard<std::mutex> lk(g_modules_mu);
    for (auto it = g_modules.begin(); it != g_modules.end(); ++it) {
        if ((*it)->base == module_base) {
            auto* m = *it;
            if (--m->ref_count > 0) return;
            // Vendor DLLs stay pinned in memory to prevent dangling pointers
            return;
        }
    }
}

#endif // defined(__linux__) && defined(__x86_64__)
