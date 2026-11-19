/**
 *
 *  Copyright 2026 Michael Fabian 'Xaymar' Dirks
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "log.h"
#include "vulkan/common.h"
#include "vulkan/shaders.h"

/* bytes of a block of device memory that buffers are made in */
#define VMAF_VK_BLOCK ((VkDeviceSize)64 << 20)

static int vk_err(VkResult r, const char *what)
{
    if (r == VK_SUCCESS) return 0;
    vmaf_log(VMAF_LOG_LEVEL_ERROR, "vulkan: %s failed (%d)\n", what, (int)r);
    switch (r) {
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return -ENOMEM;
    default:
        return -EIO;
    }
}

static void *load_loader(PFN_vkGetInstanceProcAddr *gipa)
{
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) return NULL;
    *gipa = (PFN_vkGetInstanceProcAddr)(void (*)(void))
        GetProcAddress(lib, "vkGetInstanceProcAddr");
#elif defined(__APPLE__)
    void *lib = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return NULL;
    *gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
#else
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return NULL;
    *gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
#endif
    return lib;
}

static void unload_loader(void *lib)
{
    if (!lib) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

static bool device_usable(VmafVulkanState *s, VkPhysicalDevice p)
{
    VkPhysicalDeviceFeatures f;
    s->f.vkGetPhysicalDeviceFeatures(p, &f);
    if (!f.shaderInt64 || !f.shaderFloat64) return false;
    uint32_t qn = 0;
    s->f.vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, NULL);
    VkQueueFamilyProperties qf[32];
    if (qn > 32) qn = 32;
    s->f.vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, qf);
    for (uint32_t i = 0; i < qn; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return true;
    return false;
}

void vmaf_vulkan_state_release(VmafVulkanState *s);

double vmaf_vk_now(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

int vmaf_vulkan_state_init(VmafVulkanState **vk_state,
                           VmafVulkanConfiguration cfg)
{
    const char *tm = getenv("VMAF_VULKAN_TIMING");
    const bool timing = tm && atoi(tm) > 0;
    double t0 = timing ? vmaf_vk_now() : 0, t_inst = 0, t_dev0 = 0, t_dev = 0;

    if (!vk_state) return -EINVAL;
    *vk_state = NULL;

    VmafVulkanState *s = calloc(1, sizeof(*s));
    if (!s) return -ENOMEM;
    s->timing = timing;
    pthread_mutex_init(&s->queue_lock, NULL);
    pthread_mutex_init(&s->xqueue_lock, NULL);
    pthread_mutex_init(&s->pipe_lock, NULL);

    int err = 0;
    PFN_vkGetInstanceProcAddr gipa = NULL;
    s->lib = load_loader(&gipa);
    if (!s->lib || !gipa) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "vulkan: cannot load the Vulkan loader\n");
        err = -ENODEV;
        goto fail;
    }

    PFN_vkCreateInstance create_instance =
        (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    if (!create_instance) { err = -ENODEV; goto fail; }
    VkApplicationInfo ai = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "libvmaf",
        .pEngineName = "libvmaf",
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai,
    };
    err = vk_err(create_instance(&ici, NULL, &s->inst), "vkCreateInstance");
    if (err) goto fail;
    if (timing) t_inst = vmaf_vk_now();

#define VMAF_VK_LOAD(n) \
    s->f.n = (PFN_##n)gipa(s->inst, #n); \
    if (!s->f.n) { \
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "vulkan: no " #n "\n"); \
        err = -ENODEV; \
        goto fail; \
    }
    VMAF_VK_FUNCS(VMAF_VK_LOAD)
#undef VMAF_VK_LOAD

    uint32_t n = 0;
    s->f.vkEnumeratePhysicalDevices(s->inst, &n, NULL);
    if (!n) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "vulkan: there is no Vulkan device\n");
        err = -ENODEV;
        goto fail;
    }
    VkPhysicalDevice *phys = malloc(sizeof(*phys) * n);
    if (!phys) { err = -ENOMEM; goto fail; }
    s->f.vkEnumeratePhysicalDevices(s->inst, &n, phys);

    int pick = -1;
    if (cfg.device_index >= 0) {
        if ((uint32_t)cfg.device_index >= n) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "vulkan: there is no device %d (%u devices)\n",
                     cfg.device_index, n);
        } else if (!device_usable(s, phys[cfg.device_index])) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "vulkan: device %d lacks 64 bit integers and floats in "
                     "shaders or a compute queue\n", cfg.device_index);
        } else {
            pick = cfg.device_index;
        }
    } else {
        for (uint32_t i = 0; i < n && pick < 0; i++) {
            VkPhysicalDeviceProperties p;
            s->f.vkGetPhysicalDeviceProperties(phys[i], &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
                device_usable(s, phys[i]))
                pick = i;
        }
        for (uint32_t i = 0; i < n && pick < 0; i++)
            if (device_usable(s, phys[i])) pick = i;
        if (pick < 0)
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "vulkan: no device has 64 bit integers and floats in "
                     "shaders\n");
    }
    if (pick >= 0) s->phys = phys[pick];
    free(phys);
    if (pick < 0) { err = -ENODEV; goto fail; }

    VkPhysicalDeviceProperties props;
    s->f.vkGetPhysicalDeviceProperties(s->phys, &props);
    memcpy(s->name, props.deviceName, sizeof(s->name));
    s->name[sizeof(s->name) - 1] = 0;
    s->vendor_id = props.vendorID;
    s->f.vkGetPhysicalDeviceMemoryProperties(s->phys, &s->mem);

    uint32_t qn = 0;
    s->f.vkGetPhysicalDeviceQueueFamilyProperties(s->phys, &qn, NULL);
    VkQueueFamilyProperties qf[32];
    if (qn > 32) qn = 32;
    s->f.vkGetPhysicalDeviceQueueFamilyProperties(s->phys, &qn, qf);
    bool found = false;
    for (uint32_t i = 0; i < qn && !found; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            s->family = i;
            found = true;
        }
    }
    if (!found) { err = -ENODEV; goto fail; }
    if (qf[s->family].timestampValidBits && props.limits.timestampPeriod > 0)
        s->timestamp_ns = props.limits.timestampPeriod;

    const char *no_x = getenv("VMAF_VULKAN_NO_ASYNC_UPLOAD");
    if (!(no_x && atoi(no_x) > 0)) {
        for (uint32_t i = 0; i < qn && !s->has_x; i++) {
            if ((qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                !(qf[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) &&
                qf[i].queueCount > 0) {
                s->xfamily = i;
                s->has_x = true;
            }
        }
        if (!s->has_x && qf[s->family].queueCount >= 2) {
            s->xfamily = s->family;
            s->has_x = true;
        }
    }

    const float prio[2] = { 1.0f, 1.0f };
    VkDeviceQueueCreateInfo qci[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = s->family,
            .queueCount = (s->has_x && s->xfamily == s->family) ? 2 : 1,
            .pQueuePriorities = prio,
        },
        {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = s->xfamily,
            .queueCount = 1,
            .pQueuePriorities = prio,
        },
    };
    VkPhysicalDeviceFeatures feats = {
        .shaderInt64 = VK_TRUE,
        .shaderFloat64 = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = (s->has_x && s->xfamily != s->family) ? 2 : 1,
        .pQueueCreateInfos = qci,
        .pEnabledFeatures = &feats,
    };
    if (timing) t_dev0 = vmaf_vk_now();
    err = vk_err(s->f.vkCreateDevice(s->phys, &dci, NULL, &s->dev),
                 "vkCreateDevice");
    if (err) goto fail;
    if (timing) t_dev = vmaf_vk_now();
    s->f.vkGetDeviceQueue(s->dev, s->family, 0, &s->queue);
    if (s->has_x)
        s->f.vkGetDeviceQueue(s->dev, s->xfamily,
                              s->xfamily == s->family ? 1 : 0, &s->xqueue);

    VkDescriptorSetLayoutBinding b[VMAF_VK_BINDINGS];
    for (int i = 0; i < VMAF_VK_BINDINGS; i++) {
        b[i] = (VkDescriptorSetLayoutBinding) {
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo dli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = VMAF_VK_BINDINGS,
        .pBindings = b,
    };
    err = vk_err(s->f.vkCreateDescriptorSetLayout(s->dev, &dli, NULL, &s->dsl),
                 "vkCreateDescriptorSetLayout");
    if (err) goto fail;
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = VMAF_VK_PUSH_BYTES,
    };
    VkPipelineLayoutCreateInfo pli = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &s->dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    err = vk_err(s->f.vkCreatePipelineLayout(s->dev, &pli, NULL, &s->layout),
                 "vkCreatePipelineLayout");
    if (err) goto fail;

    while (vmaf_vulkan_shaders[s->n_shaders].name) s->n_shaders++;
    s->pipelines = calloc(s->n_shaders ? s->n_shaders : 1, sizeof(*s->pipelines));
    if (!s->pipelines) { err = -ENOMEM; goto fail; }

    vmaf_log(VMAF_LOG_LEVEL_INFO, "vulkan: using %s\n", s->name);
    if (timing) {
        const double t1 = vmaf_vk_now();
        vmaf_log(VMAF_LOG_LEVEL_INFO,
                 "vulkan: init %.1f ms (loader+instance %.1f, devices %.1f, "
                 "vkCreateDevice %.1f, layouts %.1f)\n", (t1 - t0) * 1e3,
                 (t_inst - t0) * 1e3, (t_dev0 - t_inst) * 1e3,
                 (t_dev - t_dev0) * 1e3, (t1 - t_dev) * 1e3);
    }
    *vk_state = s;
    return 0;

fail:
    vmaf_vulkan_state_release(s);
    return err;
}

void vmaf_vulkan_state_release(VmafVulkanState *s)
{
    if (!s) return;
    if (s->dev) {
        s->f.vkDeviceWaitIdle(s->dev);
        for (unsigned i = 0; i < s->n_shaders; i++)
            if (s->pipelines && s->pipelines[i])
                s->f.vkDestroyPipeline(s->dev, s->pipelines[i], NULL);
        if (s->layout) s->f.vkDestroyPipelineLayout(s->dev, s->layout, NULL);
        if (s->dsl) s->f.vkDestroyDescriptorSetLayout(s->dev, s->dsl, NULL);
        s->f.vkDestroyDevice(s->dev, NULL);
    }
    free(s->pipelines);
    if (s->inst && s->f.vkDestroyInstance) s->f.vkDestroyInstance(s->inst, NULL);
    unload_loader(s->lib);
    pthread_mutex_destroy(&s->queue_lock);
    pthread_mutex_destroy(&s->xqueue_lock);
    pthread_mutex_destroy(&s->pipe_lock);
    free(s);
}

int vmaf_vulkan_state_free(VmafVulkanState **vk_state)
{
    if (!vk_state) return -EINVAL;
    vmaf_vulkan_state_release(*vk_state);
    *vk_state = NULL;
    return 0;
}

const char *vmaf_vulkan_device_name(VmafVulkanState *vk_state)
{
    return vk_state ? vk_state->name : NULL;
}

/* --- VmafVk --- */

int vmaf_vk_init(VmafVk *vk, VmafVulkanState *s, int slots)
{
    memset(vk, 0, sizeof(*vk));
    if (!s || slots < 1 || slots > VMAF_VK_MAX_SLOTS) return -EINVAL;
    vk->s = s;
    vk->nslot = slots;
    vk->cur_slot = -1;
    int err;
    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = s->family,
    };
    err = vk_err(s->f.vkCreateCommandPool(s->dev, &cpi, NULL, &vk->pool),
                 "vkCreateCommandPool");
    if (err) goto fail;
    if (s->has_x) {
        cpi.queueFamilyIndex = s->xfamily;
        err = vk_err(s->f.vkCreateCommandPool(s->dev, &cpi, NULL, &vk->xpool),
                     "vkCreateCommandPool");
        if (err) goto fail;
    }
    for (int i = 0; i < slots; i++) {
        VmafVkSlot *sl = &vk->slot[i];
        VkCommandBufferAllocateInfo cai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = vk->pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        err = vk_err(s->f.vkAllocateCommandBuffers(s->dev, &cai, &sl->cmd),
                     "vkAllocateCommandBuffers");
        if (err) goto fail;
        VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        err = vk_err(s->f.vkCreateFence(s->dev, &fi, NULL, &sl->fence),
                     "vkCreateFence");
        if (err) goto fail;
        if (s->has_x) {
            cai.commandPool = vk->xpool;
            err = vk_err(s->f.vkAllocateCommandBuffers(s->dev, &cai, &sl->xcmd),
                         "vkAllocateCommandBuffers");
            if (err) goto fail;
            VkSemaphoreCreateInfo smi = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            };
            err = vk_err(s->f.vkCreateSemaphore(s->dev, &smi, NULL, &sl->xsem),
                         "vkCreateSemaphore");
            if (err) goto fail;
        }
        if (s->timestamp_ns > 0) {
            VkQueryPoolCreateInfo qi = {
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .queryType = VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = VMAF_VK_STAMPS,
            };
            err = vk_err(s->f.vkCreateQueryPool(s->dev, &qi, NULL, &sl->stamps),
                         "vkCreateQueryPool");
            if (err) goto fail;
        }
    }
    return 0;

fail:
    vmaf_vk_close(vk);
    return err;
}

void vmaf_vk_close(VmafVk *vk)
{
    VmafVulkanState *s = vk->s;
    if (!s) return;
    for (int i = 0; i < vk->nslot; i++) {
        VmafVkSlot *sl = &vk->slot[i];
        if (sl->submitted)
            s->f.vkWaitForFences(s->dev, 1, &sl->fence, VK_TRUE, UINT64_MAX);
    }
    for (int i = 0; i < vk->nslot; i++) {
        VmafVkSlot *sl = &vk->slot[i];
        if (sl->fence) s->f.vkDestroyFence(s->dev, sl->fence, NULL);
        if (sl->xsem) s->f.vkDestroySemaphore(s->dev, sl->xsem, NULL);
        if (sl->stamps) s->f.vkDestroyQueryPool(s->dev, sl->stamps, NULL);
    }
    if (vk->pool) s->f.vkDestroyCommandPool(s->dev, vk->pool, NULL);
    if (vk->xpool) s->f.vkDestroyCommandPool(s->dev, vk->xpool, NULL);
    for (unsigned i = 0; i < vk->n_dpools; i++)
        s->f.vkDestroyDescriptorPool(s->dev, vk->dpools[i], NULL);
    free(vk->dpools);
    for (unsigned i = 0; i < vk->n_owned; i++)
        s->f.vkDestroyBuffer(s->dev, vk->owned[i].buf, NULL);
    for (unsigned i = 0; i < vk->n_blocks; i++)
        s->f.vkFreeMemory(s->dev, vk->blocks[i].mem, NULL);
    free(vk->owned);
    free(vk->blocks);
    free(vk->aliases);
    memset(vk, 0, sizeof(*vk));
}

static int vmaf_vk_buffer_(VmafVk *vk, VmafVkBuf *b, VkDeviceSize size,
                           enum VmafVkMemory kind);

int vmaf_vk_buffer(VmafVk *vk, VmafVkBuf *b, VkDeviceSize size,
                   enum VmafVkMemory kind)
{
    VmafVulkanState *s = vk->s;
    const double t0 = s->timing ? vmaf_vk_now() : 0;
    int err = vmaf_vk_buffer_(vk, b, size, kind);
    if (s->timing) vk->buf_seconds += vmaf_vk_now() - t0;
    return err;
}

static int vmaf_vk_buffer_(VmafVk *vk, VmafVkBuf *b, VkDeviceSize size,
                           enum VmafVkMemory kind)
{
    VmafVulkanState *s = vk->s;
    memset(b, 0, sizeof(*b));
    if (!size) size = 16;
    b->size = size;
    if (vk->n_owned == vk->cap_owned) {
        unsigned cap = vk->cap_owned ? vk->cap_owned * 2 : 64;
        VmafVkBuf *o = realloc(vk->owned, sizeof(*o) * cap);
        if (!o) return -ENOMEM;
        vk->owned = o;
        vk->cap_owned = cap;
    }
    const uint32_t families[2] = { s->family, s->xfamily };
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    /* With a second queue of another family, buffers are used by both (the
     * copies in and the compute work) without ownership transfers. */
    if (s->has_x && s->xfamily != s->family) {
        bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bi.queueFamilyIndexCount = 2;
        bi.pQueueFamilyIndices = families;
    }
    int err = vk_err(s->f.vkCreateBuffer(s->dev, &bi, NULL, &b->buf),
                     "vkCreateBuffer");
    if (err) return err;
    VkMemoryRequirements mr;
    s->f.vkGetBufferMemoryRequirements(s->dev, b->buf, &mr);

    const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                dl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                hc = VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                co = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    /* The memory types in order of preference; when one runs out (a small
     * window of device-local host-visible memory without resizable BAR), the
     * next one is tried. */
    const VkMemoryPropertyFlags want[4][3] = {
        { hv | co | dl, hv | co, 0 },
        { dl, 0, 0 },
        { hv | co | hc, hv | co, 0 },
        { hv | co, hv | co, 0 },
    };
    const VkMemoryPropertyFlags avoid[4][3] = {
        { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { dl, 0, 0 },
    };
    const bool host = kind != VMAF_VK_MEM_DEVICE;
    uint32_t types[3 * VK_MAX_MEMORY_TYPES];
    unsigned nt = 0;
    for (int w = 0; w < 3; w++) {
        if (!want[kind][w] && w != 2) continue;
        for (uint32_t i = 0; i < s->mem.memoryTypeCount; i++) {
            const VkMemoryPropertyFlags fl = s->mem.memoryTypes[i].propertyFlags;
            if (!(mr.memoryTypeBits & (1u << i))) continue;
            if ((fl & want[kind][w]) != want[kind][w]) continue;
            if (fl & avoid[kind][w]) continue;
            if (host && (fl & (hv | co)) != (hv | co)) continue;
            bool seen = false;
            for (unsigned k = 0; k < nt; k++) seen |= types[k] == i;
            if (!seen) types[nt++] = i;
        }
    }
    if (vk->n_blocks == vk->cap_blocks) {
        unsigned cap = vk->cap_blocks ? vk->cap_blocks * 2 : 16;
        struct VmafVkBlock *nb = realloc(vk->blocks, sizeof(*nb) * cap);
        if (!nb) {
            s->f.vkDestroyBuffer(s->dev, b->buf, NULL);
            b->buf = VK_NULL_HANDLE;
            return -ENOMEM;
        }
        vk->blocks = nb;
        vk->cap_blocks = cap;
    }
    /* Each memory type in order of preference: a piece of a block of it with
     * room, else a new block (of its own for a large buffer). */
    VkResult r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    int found = -1;
    struct VmafVkBlock *blk = NULL;
    VkDeviceSize offset = 0;
    const VkDeviceSize align = mr.alignment ? mr.alignment : 1;
    for (unsigned k = 0; k < nt && found < 0; k++) {
        const VkMemoryPropertyFlags fl = s->mem.memoryTypes[types[k]].propertyFlags;
        /* smaller blocks where host-visible VRAM can be a 256 MB window */
        const VkDeviceSize block = (fl & hv) && (fl & dl) ? VMAF_VK_BLOCK / 4
                                                          : VMAF_VK_BLOCK;
        for (unsigned i = 0; i < vk->n_blocks && found < 0; i++) {
            struct VmafVkBlock *c = &vk->blocks[i];
            if (c->type != types[k] || c->size < block) continue;
            const VkDeviceSize o = (c->used + align - 1) / align * align;
            if (o + mr.size > c->size) continue;
            blk = c;
            offset = o;
            found = types[k];
        }
        if (found >= 0) break;
        const VkDeviceSize want_size = mr.size * 2 > block ? mr.size : block;
        VkMemoryAllocateInfo mi = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = want_size,
            .memoryTypeIndex = types[k],
        };
        VkDeviceMemory mem;
        r = s->f.vkAllocateMemory(s->dev, &mi, NULL, &mem);
        if (r == VK_SUCCESS) {
            blk = &vk->blocks[vk->n_blocks++];
            memset(blk, 0, sizeof(*blk));
            blk->mem = mem;
            blk->type = types[k];
            blk->size = want_size;
            if (fl & hv) {
                r = s->f.vkMapMemory(s->dev, mem, 0, VK_WHOLE_SIZE, 0, &blk->map);
                if (r != VK_SUCCESS) {
                    s->f.vkDestroyBuffer(s->dev, b->buf, NULL);
                    b->buf = VK_NULL_HANDLE;
                    return vk_err(r, "vkMapMemory");
                }
            }
            if (fl & dl)
                vk->device_bytes += want_size;
            else
                vk->host_bytes += want_size;
            offset = 0;
            found = types[k];
            break;
        }
        if (r != VK_ERROR_OUT_OF_DEVICE_MEMORY && r != VK_ERROR_OUT_OF_HOST_MEMORY)
            break;
    }
    if (found < 0) {
        s->f.vkDestroyBuffer(s->dev, b->buf, NULL);
        b->buf = VK_NULL_HANDLE;
        return nt ? vk_err(r, "vkAllocateMemory") : -ENOMEM;
    }
    blk->used = offset + mr.size;
    b->mem = blk->mem;
    vk->owned[vk->n_owned++] = *b;
    err = vk_err(s->f.vkBindBufferMemory(s->dev, b->buf, b->mem, offset),
                 "vkBindBufferMemory");
    if (err) return err;
    if (blk->map) b->map = (uint8_t *)blk->map + offset;
    return 0;
}

static int shader_pipeline(VmafVulkanState *s, const char *name,
                           VkPipeline *out, const char **stored_name)
{
    int idx = -1;
    for (unsigned i = 0; i < s->n_shaders; i++) {
        if (!strcmp(vmaf_vulkan_shaders[i].name, name)) { idx = i; break; }
    }
    if (idx < 0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "vulkan: no shader %s\n", name);
        return -EINVAL;
    }
    *stored_name = vmaf_vulkan_shaders[idx].name;
    int err = 0;
    pthread_mutex_lock(&s->pipe_lock);
    if (!s->pipelines[idx]) {
        const double t0 = s->timing ? vmaf_vk_now() : 0;
        VkShaderModuleCreateInfo si = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = vmaf_vulkan_shaders[idx].size,
            .pCode = vmaf_vulkan_shaders[idx].code,
        };
        VkShaderModule sm;
        err = vk_err(s->f.vkCreateShaderModule(s->dev, &si, NULL, &sm),
                     "vkCreateShaderModule");
        if (!err) {
            VkComputePipelineCreateInfo pi = {
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage = {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = sm,
                    .pName = "main",
                },
                .layout = s->layout,
            };
            err = vk_err(s->f.vkCreateComputePipelines(s->dev, VK_NULL_HANDLE,
                         1, &pi, NULL, &s->pipelines[idx]), name);
            s->f.vkDestroyShaderModule(s->dev, sm, NULL);
        }
        if (s->timing) {
            s->pipe_seconds += vmaf_vk_now() - t0;
            s->n_pipes++;
        }
    }
    *out = s->pipelines[idx];
    pthread_mutex_unlock(&s->pipe_lock);
    return err;
}

static int alloc_set(VmafVk *vk, VkDescriptorSet *set)
{
    VmafVulkanState *s = vk->s;
    VkDescriptorSetAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorSetCount = 1,
        .pSetLayouts = &s->dsl,
    };
    if (vk->n_dpools) {
        ai.descriptorPool = vk->dpools[vk->n_dpools - 1];
        VkResult r = s->f.vkAllocateDescriptorSets(s->dev, &ai, set);
        if (r == VK_SUCCESS) return 0;
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL)
            return vk_err(r, "vkAllocateDescriptorSets");
    }
    VkDescriptorPool *p = realloc(vk->dpools, sizeof(*p) * (vk->n_dpools + 1));
    if (!p) return -ENOMEM;
    vk->dpools = p;
    VkDescriptorPoolSize ps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 256 * VMAF_VK_BINDINGS,
    };
    VkDescriptorPoolCreateInfo dpi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 256,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    int err = vk_err(s->f.vkCreateDescriptorPool(s->dev, &dpi, NULL,
                     &vk->dpools[vk->n_dpools]), "vkCreateDescriptorPool");
    if (err) return err;
    vk->n_dpools++;
    ai.descriptorPool = vk->dpools[vk->n_dpools - 1];
    return vk_err(s->f.vkAllocateDescriptorSets(s->dev, &ai, set),
                  "vkAllocateDescriptorSets");
}

int vmaf_vk_pipeline(VmafVk *vk, VmafVkPipe *p, const char *name,
                     VmafVkBuf *const *bindings, unsigned n)
{
    VmafVulkanState *s = vk->s;
    memset(p, 0, sizeof(*p));
    if (!n || !bindings[0]) return -EINVAL;
    int err = shader_pipeline(s, name, &p->pipe, &p->name);
    if (err) return err;
    err = alloc_set(vk, &p->set);
    if (err) return err;
    VkDescriptorBufferInfo info[VMAF_VK_BINDINGS];
    VkWriteDescriptorSet w[VMAF_VK_BINDINGS];
    for (unsigned i = 0; i < VMAF_VK_BINDINGS; i++) {
        const VmafVkBuf *b = (i < n && bindings[i]) ? bindings[i] : bindings[0];
        info[i] = (VkDescriptorBufferInfo) { b->buf, 0, VK_WHOLE_SIZE };
        w[i] = (VkWriteDescriptorSet) {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = p->set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &info[i],
        };
    }
    s->f.vkUpdateDescriptorSets(s->dev, VMAF_VK_BINDINGS, w, 0, NULL);
    bool aliased = false;
    for (unsigned i = 0; i < VMAF_VK_BINDINGS; i++)
        for (unsigned a = 0; a < vk->n_aliases; a++)
            aliased |= info[i].buffer == vk->aliases[a].from;
    if (!aliased) return 0;
    p->nslot = vk->nslot;
    for (int k = 0; k < vk->nslot; k++) {
        err = alloc_set(vk, &p->slot_sets[k]);
        if (err) return err;
        VkDescriptorBufferInfo si[VMAF_VK_BINDINGS];
        memcpy(si, info, sizeof(si));
        for (unsigned i = 0; i < VMAF_VK_BINDINGS; i++) {
            for (unsigned a = 0; a < vk->n_aliases; a++)
                if (si[i].buffer == vk->aliases[a].from)
                    si[i].buffer = vk->aliases[a].to[k];
            w[i].dstSet = p->slot_sets[k];
            w[i].pBufferInfo = &si[i];
        }
        s->f.vkUpdateDescriptorSets(s->dev, VMAF_VK_BINDINGS, w, 0, NULL);
    }
    return 0;
}

int vmaf_vk_alias(VmafVk *vk, const VmafVkBuf *placeholder,
                  const VmafVkBuf *per_slot)
{
    void *a = realloc(vk->aliases, sizeof(*vk->aliases) * (vk->n_aliases + 1));
    if (!a) return -ENOMEM;
    vk->aliases = a;
    vk->aliases[vk->n_aliases].from = placeholder->buf;
    for (int k = 0; k < vk->nslot; k++)
        vk->aliases[vk->n_aliases].to[k] = per_slot[k].buf;
    vk->n_aliases++;
    return 0;
}

int vmaf_vk_begin(VmafVk *vk, int slot)
{
    VmafVulkanState *s = vk->s;
    VmafVkSlot *sl = &vk->slot[slot];
    vk->cmd = sl->cmd;
    vk->cur_slot = slot;
    s->f.vkResetCommandBuffer(vk->cmd, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    int err = vk_err(s->f.vkBeginCommandBuffer(vk->cmd, &bi),
                     "vkBeginCommandBuffer");
    if (err) return err;
    if (sl->stamps) s->f.vkCmdResetQueryPool(vk->cmd, sl->stamps, 0, VMAF_VK_STAMPS);
    return 0;
}

void vmaf_vk_bind(VmafVk *vk, const VmafVkPipe *p)
{
    VmafVulkanState *s = vk->s;
    s->f.vkCmdBindPipeline(vk->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    const VkDescriptorSet set = (p->nslot > 0 && vk->cur_slot >= 0 &&
                                 vk->cur_slot < p->nslot)
                                ? p->slot_sets[vk->cur_slot] : p->set;
    s->f.vkCmdBindDescriptorSets(vk->cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 s->layout, 0, 1, &set, 0, NULL);
}

void vmaf_vk_push(VmafVk *vk, const void *data, uint32_t size)
{
    vk->s->f.vkCmdPushConstants(vk->cmd, vk->s->layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

void vmaf_vk_dispatch(VmafVk *vk, uint32_t x, uint32_t y)
{
    vk->s->f.vkCmdDispatch(vk->cmd, x, y, 1);
}

void vmaf_vk_copy(VmafVk *vk, const VmafVkBuf *src, const VmafVkBuf *dst,
                  VkDeviceSize size)
{
    VkBufferCopy c = {
        0, 0, size ? size : (src->size < dst->size ? src->size : dst->size)
    };
    vk->s->f.vkCmdCopyBuffer(vk->cmd, src->buf, dst->buf, 1, &c);
}

void vmaf_vk_copy_range(VmafVk *vk, const VmafVkBuf *src, VkDeviceSize src_off,
                        const VmafVkBuf *dst, VkDeviceSize dst_off,
                        VkDeviceSize size)
{
    VkBufferCopy c = { src_off, dst_off, size };
    vk->s->f.vkCmdCopyBuffer(vk->cmd, src->buf, dst->buf, 1, &c);
}

void vmaf_vk_barrier(VmafVk *vk)
{
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                         VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    vk->s->f.vkCmdPipelineBarrier(vk->cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

void vmaf_vk_stamp(VmafVk *vk, int which)
{
    VmafVkSlot *sl = &vk->slot[vk->cur_slot];
    if (sl->stamps && which >= 0 && which < VMAF_VK_STAMPS)
        vk->s->f.vkCmdWriteTimestamp(vk->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                     sl->stamps, which);
}

int vmaf_vk_submit(VmafVk *vk, int slot)
{
    VmafVulkanState *s = vk->s;
    VmafVkSlot *sl = &vk->slot[slot];
    int err = vk_err(s->f.vkEndCommandBuffer(sl->cmd), "vkEndCommandBuffer");
    if (err) return err;
    const VkPipelineStageFlags wait_stage =
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &sl->cmd,
    };
    if (sl->xpending) {
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &sl->xsem;
        si.pWaitDstStageMask = &wait_stage;
        sl->xpending = false;
    }
    s->f.vkResetFences(s->dev, 1, &sl->fence);
    pthread_mutex_lock(&s->queue_lock);
    VkResult r = s->f.vkQueueSubmit(s->queue, 1, &si, sl->fence);
    pthread_mutex_unlock(&s->queue_lock);
    err = vk_err(r, "vkQueueSubmit");
    if (!err) sl->submitted = true;
    vk->cur_slot = -1;
    return err;
}

int vmaf_vk_wait(VmafVk *vk, int slot)
{
    VmafVulkanState *s = vk->s;
    VmafVkSlot *sl = &vk->slot[slot];
    if (!sl->submitted) return 0;
    int err = vk_err(s->f.vkWaitForFences(s->dev, 1, &sl->fence, VK_TRUE,
                                          UINT64_MAX), "vkWaitForFences");
    sl->submitted = false;
    return err;
}

double vmaf_vk_gpu_seconds(VmafVk *vk, int slot, int from, int to)
{
    VmafVulkanState *s = vk->s;
    VmafVkSlot *sl = &vk->slot[slot];
    if (!sl->stamps || from < 0 || to < 0 || from >= VMAF_VK_STAMPS ||
        to >= VMAF_VK_STAMPS)
        return -1;
    uint64_t t[2] = { 0, 0 };
    if (s->f.vkGetQueryPoolResults(s->dev, sl->stamps, from, 1, 8, &t[0], 8,
                                   VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return -1;
    if (s->f.vkGetQueryPoolResults(s->dev, sl->stamps, to, 1, 8, &t[1], 8,
                                   VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return -1;
    if (t[1] < t[0]) return -1;
    return (double)(t[1] - t[0]) * s->timestamp_ns * 1e-9;
}

int vmaf_vk_upload_begin(VmafVk *vk, int slot)
{
    VmafVulkanState *s = vk->s;
    if (!s->has_x) return -EINVAL;
    vk->xrec = vk->slot[slot].xcmd;
    s->f.vkResetCommandBuffer(vk->xrec, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    return vk_err(s->f.vkBeginCommandBuffer(vk->xrec, &bi),
                  "vkBeginCommandBuffer");
}

void vmaf_vk_upload_copy(VmafVk *vk, const VmafVkBuf *src,
                         const VmafVkBuf *dst, VkDeviceSize size)
{
    VkBufferCopy c = {
        0, 0, size ? size : (src->size < dst->size ? src->size : dst->size)
    };
    vk->s->f.vkCmdCopyBuffer(vk->xrec, src->buf, dst->buf, 1, &c);
}

int vmaf_vk_upload_submit(VmafVk *vk, int slot)
{
    VmafVulkanState *s = vk->s;
    VmafVkSlot *sl = &vk->slot[slot];
    int err = vk_err(s->f.vkEndCommandBuffer(sl->xcmd), "vkEndCommandBuffer");
    if (err) return err;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &sl->xcmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &sl->xsem,
    };
    pthread_mutex_lock(&s->xqueue_lock);
    VkResult r = s->f.vkQueueSubmit(s->xqueue, 1, &si, VK_NULL_HANDLE);
    pthread_mutex_unlock(&s->xqueue_lock);
    err = vk_err(r, "vkQueueSubmit (copies)");
    if (!err) sl->xpending = true;
    return err;
}
