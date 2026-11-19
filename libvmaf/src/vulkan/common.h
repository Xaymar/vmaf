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

#ifndef __VMAF_SRC_VULKAN_COMMON_H__
#define __VMAF_SRC_VULKAN_COMMON_H__

/* A small Vulkan compute layer for the Vulkan feature extractors: the device
 * (VmafVulkanState, shared by everything that uses it), and VmafVk, one user of
 * it with its own buffers, descriptor sets and command buffers (one per slot:
 * a frame in flight). Every pipeline has the same layout: 16 storage buffers
 * and 128 bytes of push constants. */

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "libvmaf/libvmaf_vulkan.h"

#define VMAF_VK_BINDINGS 16
#define VMAF_VK_PUSH_BYTES 128
#define VMAF_VK_MAX_SLOTS 8
#define VMAF_VK_STAMPS 12

#define VMAF_VK_FUNCS(X) \
    X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceFeatures) \
    X(vkEnumeratePhysicalDevices) X(vkCreateDevice) X(vkDestroyDevice) \
    X(vkDestroyInstance) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkMapMemory) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) \
    X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) \
    X(vkCmdDispatch) X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) \
    X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) \
    X(vkQueueSubmit) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) \
    X(vkResetFences) X(vkCreateSemaphore) X(vkDestroySemaphore) \
    X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkGetQueryPoolResults)

typedef struct VmafVulkanFunctions {
#define VMAF_VK_DECLARE(n) PFN_##n n;
    VMAF_VK_FUNCS(VMAF_VK_DECLARE)
#undef VMAF_VK_DECLARE
} VmafVulkanFunctions;

struct VmafVulkanState {
    void *lib;
    VmafVulkanFunctions f;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint32_t vendor_id;
    VkPhysicalDeviceMemoryProperties mem;
    double timestamp_ns; ///< per tick; 0 without timestamps
    uint32_t family; ///< the compute queue's
    VkQueue queue;
    pthread_mutex_t queue_lock;
    /* A second queue for copying frames in while the compute queue works on
     * others: a transfer-only family where there is one, else a second queue
     * of the compute family; none if neither exists. */
    bool has_x;
    uint32_t xfamily;
    VkQueue xqueue;
    pthread_mutex_t xqueue_lock;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    pthread_mutex_t pipe_lock;
    VkPipeline *pipelines; ///< per shader in vmaf_vulkan_shaders, made on first use
    unsigned n_shaders;
    /* VMAF_VULKAN_TIMING: host seconds spent making pipelines (pipe_lock) */
    bool timing;
    double pipe_seconds;
    unsigned n_pipes;
};

typedef struct VmafVkBuf {
    VkBuffer buf;
    VkDeviceMemory mem;
    void *map; ///< for host visible memory
    VkDeviceSize size;
} VmafVkBuf;

typedef struct VmafVkPipe {
    VkPipeline pipe;
    VkDescriptorSet set;
    /* With an aliased buffer bound (vmaf_vk_alias()): a set per slot, with
     * that slot's buffer in its place. */
    VkDescriptorSet slot_sets[VMAF_VK_MAX_SLOTS];
    int nslot;
    const char *name;
} VmafVkPipe;

enum VmafVkMemory {
    /* host visible, device local where there is a window for it (written by
     * the CPU straight into VRAM) */
    VMAF_VK_MEM_UPLOAD = 0,
    VMAF_VK_MEM_DEVICE = 1,   ///< device local
    VMAF_VK_MEM_READBACK = 2, ///< host cached (read back by the CPU)
    /* host visible in system memory where there is such (written by the CPU,
     * copied by the GPU) */
    VMAF_VK_MEM_STAGING = 3,
};

typedef struct VmafVkSlot {
    VkCommandBuffer cmd;
    VkFence fence;
    bool submitted;
    VkQueryPool stamps;
    VkCommandBuffer xcmd; ///< copies on the second queue
    VkSemaphore xsem;     ///< ... which the slot's next submit waits for
    bool xpending;
} VmafVkSlot;

typedef struct VmafVk {
    VmafVulkanState *s;
    VkCommandPool pool, xpool;
    VkDescriptorPool *dpools;
    unsigned n_dpools;
    int nslot;
    VmafVkSlot slot[VMAF_VK_MAX_SLOTS];
    VkCommandBuffer cmd; ///< the one being recorded
    int cur_slot;
    VkCommandBuffer xrec; ///< the copies' one being recorded
    struct {
        VkBuffer from;
        VkBuffer to[VMAF_VK_MAX_SLOTS];
    } *aliases;
    unsigned n_aliases;
    VmafVkBuf *owned;
    unsigned n_owned, cap_owned;
    uint64_t device_bytes, host_bytes;
    double buf_seconds; ///< host seconds in vmaf_vk_buffer()
} VmafVk;

/* A monotonic clock in seconds (for VMAF_VULKAN_TIMING). */
double vmaf_vk_now(void);

int vmaf_vk_init(VmafVk *vk, VmafVulkanState *s, int slots);
void vmaf_vk_close(VmafVk *vk);

/* Buffers live as long as the VmafVk. */
int vmaf_vk_buffer(VmafVk *vk, VmafVkBuf *b, VkDeviceSize size,
                   enum VmafVkMemory kind);
/* A pipeline of the embedded shader `name`, with a descriptor set that binds
 * bindings[i] to binding i (unused ones get bindings[0]). */
int vmaf_vk_pipeline(VmafVk *vk, VmafVkPipe *p, const char *name,
                     VmafVkBuf *const *bindings, unsigned n);
/* Makes `placeholder`, wherever a pipeline made after this binds it, stand for
 * per_slot[k] in the work of slot k. */
int vmaf_vk_alias(VmafVk *vk, const VmafVkBuf *placeholder,
                  const VmafVkBuf *per_slot);

int vmaf_vk_begin(VmafVk *vk, int slot);
void vmaf_vk_bind(VmafVk *vk, const VmafVkPipe *p);
void vmaf_vk_push(VmafVk *vk, const void *data, uint32_t size);
void vmaf_vk_dispatch(VmafVk *vk, uint32_t x, uint32_t y);
void vmaf_vk_copy(VmafVk *vk, const VmafVkBuf *src, const VmafVkBuf *dst,
                  VkDeviceSize size);
void vmaf_vk_copy_range(VmafVk *vk, const VmafVkBuf *src, VkDeviceSize src_off,
                        const VmafVkBuf *dst, VkDeviceSize dst_off,
                        VkDeviceSize size);
/* Orders everything recorded (and submitted to the queue) before against
 * everything after: shader and transfer writes before host reads. */
void vmaf_vk_barrier(VmafVk *vk);
void vmaf_vk_stamp(VmafVk *vk, int which);
int vmaf_vk_submit(VmafVk *vk, int slot);
int vmaf_vk_wait(VmafVk *vk, int slot);
/* GPU seconds between two stamps of the slot's last finished work, -1 if the
 * device has no timestamps. */
double vmaf_vk_gpu_seconds(VmafVk *vk, int slot, int from, int to);

/* Copies on the second queue (vk->s->has_x): recorded with upload_begin() and
 * upload_copy(), submitted with upload_submit(); the slot's next submit waits
 * for them. */
int vmaf_vk_upload_begin(VmafVk *vk, int slot);
void vmaf_vk_upload_copy(VmafVk *vk, const VmafVkBuf *src,
                         const VmafVkBuf *dst, VkDeviceSize size);
int vmaf_vk_upload_submit(VmafVk *vk, int slot);

#endif /* __VMAF_SRC_VULKAN_COMMON_H__ */
