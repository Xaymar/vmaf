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

#ifndef __VMAF_SRC_VULKAN_ENGINE_H__
#define __VMAF_SRC_VULKAN_ENGINE_H__

/* The GPU side of the Vulkan feature extractors of one VmafContext.
 *
 * Every Vulkan feature extractor of a context uses the same engine: each frame
 * pair is copied to the GPU once, and the work of all of them goes into one
 * command buffer per frame. The work is split by feature into modules
 * (motion, VIF, ADM, CAMBI, SpEED), one of each per engine; each feature
 * extractor instance is a unit of its module (models that use a feature with
 * different options), so that what the units compute alike (the ADM wavelets,
 * the SpEED planes, ...) is computed once.
 *
 * Several frames are in flight (one per slot): extract() copies the pictures
 * into the slot's staging buffers and returns; the frame's work is recorded and
 * submitted when the next frame comes (by then every feature extractor has
 * seen the frame), and a thread per slot waits for the GPU and writes the
 * features into the feature collector. The feature extractors are temporal
 * (VMAF_FEATURE_EXTRACTOR_TEMPORAL), so they are called in frame order, from
 * one thread. */

#include <pthread.h>
#include <stdbool.h>

#include "feature/feature_collector.h"
#include "libvmaf/picture.h"
#include "vulkan/common.h"

#define VMAF_VK_ENGINE_SLOTS 4
#define VMAF_VK_ENGINE_MODULES 8

typedef struct VmafVkEngine VmafVkEngine;

typedef struct VmafVkModule {
    const char *name;
    bool chroma; ///< reads the chroma planes
    /* Called once, before the first frame is recorded, when every unit has
     * joined: buffers and pipelines. */
    int (*setup)(struct VmafVkModule *m, VmafVkEngine *e);
    /* Records the module's work on frame `index` into the slot's command
     * buffer, including the copies of what it reads back into the slot's own
     * host buffers (after a barrier). */
    int (*record)(struct VmafVkModule *m, VmafVkEngine *e, int slot,
                  unsigned index);
    /* Once the GPU is done with the slot (on the slot's thread): the features
     * of frame `index`, into the collector. */
    int (*results)(struct VmafVkModule *m, VmafVkEngine *e, int slot,
                   unsigned index, VmafFeatureCollector *fc);
    void (*close)(struct VmafVkModule *m, VmafVkEngine *e);
    void *priv;
} VmafVkModule;

typedef struct VmafVkEngineSlot {
    VmafVkBuf in[6];  ///< staging: Y ref, Y dis, U ref, V ref, U dis, V dis
    VmafVkBuf dev[6]; ///< with a copies' queue: the frame in VRAM
    bool chroma_in;   ///< the chroma planes of this frame were copied
    unsigned index;
    VmafFeatureCollector *fc;
    enum { SLOT_FREE, SLOT_FILLED, SLOT_BUSY } state;
    pthread_t thread;
    bool thread_up;
} VmafVkEngineSlot;

struct VmafVkEngine {
    VmafVk vk;
    VmafVulkanState *s;
    unsigned w, h, bpc, bps;
    enum VmafPixelFormat pix_fmt;
    unsigned cw, ch; ///< chroma planes, 0 for YUV400
    size_t luma_size, chroma_size; ///< bytes of a buffer holding a plane (padded)

    /* What the shaders read: the frame pair. With a copies' queue these stand
     * for each slot's own buffers (vmaf_vk_alias), else they are copied in. */
    VmafVkBuf raw[6];
    bool alias_raw;

    VmafVkEngineSlot slot[VMAF_VK_ENGINE_SLOTS];
    VmafVkModule module[VMAF_VK_ENGINE_MODULES];
    unsigned n_modules;
    bool set_up;
    bool want_chroma;

    bool pending; ///< a frame was copied in and waits to be recorded
    int pending_slot;
    unsigned pending_index;
    unsigned next_slot;

    pthread_mutex_t lock;
    pthread_cond_t changed;
    bool stop;
    int error; ///< the first error of a slot's thread
    unsigned users;
    bool timing;
    double gpu_seconds;
    /* VMAF_VULKAN_TIMING=2: modules one after another, GPU time of each */
    bool timing_modules;
    double module_gpu[VMAF_VK_ENGINE_MODULES];
    unsigned frames;
    /* VMAF_VULKAN_TIMING: host seconds */
    double t_created;     ///< when the engine was made
    double t_engine;      ///< vmaf_vk_engine_get (buffers, threads)
    double t_engine_buf;  ///< ... of it in vmaf_vk_buffer
    double t_setup;       ///< setup() before the first frame (modules)
    double t_first_done;  ///< when the first frame's results were in
    double t_wait;        ///< waiting for a free slot
    double t_copy;        ///< copying planes into staging
    double t_record;      ///< recording and submitting
    double t_results;     ///< modules' results (slot threads, summed)
    double t_flush;       ///< waiting in vmaf_vk_engine_flush
};

/* The buffer of a frame plane that slot `slot` reads: for copies (which do not
 * go through the aliases). plane: 0 Y ref, 1 Y dis, 2 U ref, 3 V ref, 4 U dis,
 * 5 V dis. */
const VmafVkBuf *vmaf_vk_engine_plane(VmafVkEngine *e, int slot, int plane);

/* The module `name` of the engine, made with `create` (which fills in the
 * callbacks and priv) the first time; NULL on error. */
VmafVkModule *vmaf_vk_engine_module(VmafVkEngine *e, const char *name,
                                    int (*create)(VmafVkModule *m,
                                                  VmafVkEngine *e));

typedef struct VmafVulkanContext VmafVulkanContext;

/* The engine of a context, made by the first feature extractor that asks; each
 * call takes a reference that vmaf_vk_engine_release() gives back. */
int vmaf_vk_engine_get(VmafVulkanContext *ctx, enum VmafPixelFormat pix_fmt,
                       unsigned bpc, unsigned w, unsigned h,
                       VmafVkEngine **engine);
void vmaf_vk_engine_release(VmafVulkanContext *ctx, VmafVkEngine **engine);

int vmaf_vk_engine_extract(VmafVkEngine *e, VmafPicture *ref,
                           VmafPicture *dist, unsigned index,
                           VmafFeatureCollector *fc);
/* Records the last frame and waits until every frame's features are in the
 * collector. */
int vmaf_vk_engine_flush(VmafVkEngine *e);

/* What the framework keeps per VmafContext (vmaf_vulkan_import_state). */
struct VmafVulkanContext {
    VmafVulkanState *state;
    VmafVkEngine *engine;
    pthread_mutex_t lock;
};

int vmaf_vulkan_context_create(VmafVulkanContext **ctx, VmafVulkanState *s);
void vmaf_vulkan_context_destroy(VmafVulkanContext **ctx);

#endif /* __VMAF_SRC_VULKAN_ENGINE_H__ */
