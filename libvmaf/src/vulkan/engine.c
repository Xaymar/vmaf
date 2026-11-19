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

#include "log.h"
#include "vulkan/engine.h"

static size_t padded(size_t bytes) { return (bytes + 3) / 4 * 4 + 4; }

const VmafVkBuf *vmaf_vk_engine_plane(VmafVkEngine *e, int slot, int plane)
{
    return e->alias_raw ? &e->slot[slot].dev[plane] : &e->raw[plane];
}

VmafVkModule *vmaf_vk_engine_module(VmafVkEngine *e, const char *name,
                                    int (*create)(VmafVkModule *m,
                                                  VmafVkEngine *e))
{
    for (unsigned i = 0; i < e->n_modules; i++)
        if (!strcmp(e->module[i].name, name)) return &e->module[i];
    if (e->set_up) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vulkan: feature %s joined after the first frame\n", name);
        return NULL;
    }
    if (e->n_modules == VMAF_VK_ENGINE_MODULES) return NULL;
    VmafVkModule *m = &e->module[e->n_modules];
    memset(m, 0, sizeof(*m));
    m->name = name;
    if (create(m, e)) {
        if (m->close) m->close(m, e);
        memset(m, 0, sizeof(*m));
        return NULL;
    }
    e->n_modules++;
    if (m->chroma) e->want_chroma = true;
    return m;
}

static void *slot_thread(void *arg);

typedef struct SlotThreadArg {
    VmafVkEngine *e;
    int slot;
} SlotThreadArg;

static void engine_destroy(VmafVkEngine *e)
{
    pthread_mutex_lock(&e->lock);
    e->stop = true;
    pthread_cond_broadcast(&e->changed);
    pthread_mutex_unlock(&e->lock);
    for (int k = 0; k < VMAF_VK_ENGINE_SLOTS; k++)
        if (e->slot[k].thread_up) pthread_join(e->slot[k].thread, NULL);
    if (e->timing && e->frames) {
        char names[128] = "";
        for (unsigned i = 0; i < e->n_modules; i++) {
            size_t n = strlen(names);
            snprintf(names + n, sizeof(names) - n, "%s%s", i ? " " : "",
                     e->module[i].name);
        }
        vmaf_log(VMAF_LOG_LEVEL_INFO,
                 "vulkan: %u frames, %.3f ms of GPU time per frame (%s)\n",
                 e->frames, e->gpu_seconds * 1000.0 / e->frames, names);
    }
    for (unsigned i = 0; i < e->n_modules; i++)
        if (e->module[i].close) e->module[i].close(&e->module[i], e);
    vmaf_vk_close(&e->vk);
    pthread_mutex_destroy(&e->lock);
    pthread_cond_destroy(&e->changed);
    free(e);
}

int vmaf_vk_engine_get(VmafVulkanContext *ctx, enum VmafPixelFormat pix_fmt,
                       unsigned bpc, unsigned w, unsigned h,
                       VmafVkEngine **engine)
{
    if (!ctx || !ctx->state || !engine) return -EINVAL;
    int err = 0;
    pthread_mutex_lock(&ctx->lock);
    VmafVkEngine *e = ctx->engine;
    if (e) {
        if (e->w != w || e->h != h || e->bpc != bpc || e->pix_fmt != pix_fmt) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "vulkan: the feature extractors of a context must see "
                     "pictures of one format\n");
            err = -EINVAL;
        } else {
            e->users++;
            *engine = e;
        }
        pthread_mutex_unlock(&ctx->lock);
        return err;
    }
    if (bpc < 8 || bpc > 16 || w < 16 || h < 16) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vulkan: pictures of %ux%u at %u bits are not supported\n",
                 w, h, bpc);
        pthread_mutex_unlock(&ctx->lock);
        return -EINVAL;
    }

    e = calloc(1, sizeof(*e));
    if (!e) { pthread_mutex_unlock(&ctx->lock); return -ENOMEM; }
    pthread_mutex_init(&e->lock, NULL);
    pthread_cond_init(&e->changed, NULL);
    e->s = ctx->state;
    e->w = w;
    e->h = h;
    e->bpc = bpc;
    e->bps = bpc > 8 ? 2 : 1;
    e->pix_fmt = pix_fmt;
    switch (pix_fmt) {
    case VMAF_PIX_FMT_YUV420P: e->cw = (w + 1) / 2; e->ch = (h + 1) / 2; break;
    case VMAF_PIX_FMT_YUV422P: e->cw = (w + 1) / 2; e->ch = h; break;
    case VMAF_PIX_FMT_YUV444P: e->cw = w; e->ch = h; break;
    default: e->cw = e->ch = 0; break;
    }
    e->luma_size = padded((size_t)w * h * e->bps);
    e->chroma_size = padded((size_t)(e->cw ? e->cw : 1) * (e->ch ? e->ch : 1) * e->bps);
    const char *t = getenv("VMAF_VULKAN_TIMING");
    e->timing = t && atoi(t) > 0;

    err = vmaf_vk_init(&e->vk, e->s, VMAF_VK_ENGINE_SLOTS);
    if (err) goto fail;

    /* With a queue for copies, each slot's frame stays in its own buffers,
     * which the pipelines read in place of raw[] (no copy on the compute
     * queue). */
    e->alias_raw = e->s->has_x;
    for (int p = 0; p < 6; p++) {
        const size_t size = p < 2 ? e->luma_size : e->chroma_size;
        if (p >= 2 && !e->cw) break;
        for (int k = 0; k < VMAF_VK_ENGINE_SLOTS; k++) {
            err = vmaf_vk_buffer(&e->vk, &e->slot[k].in[p], size,
                                 VMAF_VK_MEM_STAGING);
            if (err) goto fail;
        }
    }
    for (int p = 0; p < 2; p++) {
        err = vmaf_vk_buffer(&e->vk, &e->raw[p], e->alias_raw ? 16 : e->luma_size,
                             VMAF_VK_MEM_DEVICE);
        if (err) goto fail;
        if (e->alias_raw) {
            VmafVkBuf per_slot[VMAF_VK_ENGINE_SLOTS];
            for (int k = 0; k < VMAF_VK_ENGINE_SLOTS; k++) {
                err = vmaf_vk_buffer(&e->vk, &e->slot[k].dev[p], e->luma_size,
                                     VMAF_VK_MEM_DEVICE);
                if (err) goto fail;
                per_slot[k] = e->slot[k].dev[p];
            }
            err = vmaf_vk_alias(&e->vk, &e->raw[p], per_slot);
            if (err) goto fail;
        }
    }

    for (int k = 0; k < VMAF_VK_ENGINE_SLOTS; k++) {
        SlotThreadArg *a = malloc(sizeof(*a));
        if (!a) { err = -ENOMEM; goto fail; }
        a->e = e;
        a->slot = k;
        if (pthread_create(&e->slot[k].thread, NULL, slot_thread, a)) {
            free(a);
            err = -ENOMEM;
            goto fail;
        }
        e->slot[k].thread_up = true;
    }

    e->users = 1;
    ctx->engine = e;
    *engine = e;
    pthread_mutex_unlock(&ctx->lock);
    return 0;

fail:
    engine_destroy(e);
    pthread_mutex_unlock(&ctx->lock);
    return err;
}

void vmaf_vk_engine_release(VmafVulkanContext *ctx, VmafVkEngine **engine)
{
    if (!engine || !*engine) return;
    VmafVkEngine *e = *engine;
    *engine = NULL;
    pthread_mutex_lock(&ctx->lock);
    if (--e->users == 0) {
        vmaf_vk_engine_flush(e);
        engine_destroy(e);
        if (ctx->engine == e) ctx->engine = NULL;
    }
    pthread_mutex_unlock(&ctx->lock);
}

static void set_error(VmafVkEngine *e, int err)
{
    if (!err) return;
    pthread_mutex_lock(&e->lock);
    if (!e->error) e->error = err;
    pthread_mutex_unlock(&e->lock);
}

static void *slot_thread(void *arg)
{
    SlotThreadArg a = *(SlotThreadArg *)arg;
    free(arg);
    VmafVkEngine *e = a.e;
    VmafVkEngineSlot *sl = &e->slot[a.slot];
    for (;;) {
        pthread_mutex_lock(&e->lock);
        while (sl->state != SLOT_BUSY && !e->stop)
            pthread_cond_wait(&e->changed, &e->lock);
        if (sl->state != SLOT_BUSY) {
            pthread_mutex_unlock(&e->lock);
            return NULL;
        }
        pthread_mutex_unlock(&e->lock);

        int err = vmaf_vk_wait(&e->vk, a.slot);
        double gpu = 0;
        if (!err && e->timing) gpu = vmaf_vk_gpu_seconds(&e->vk, a.slot, 0, 1);
        for (unsigned i = 0; i < e->n_modules && !err; i++) {
            VmafVkModule *m = &e->module[i];
            if (m->results) err = m->results(m, e, a.slot, sl->index, sl->fc);
        }

        pthread_mutex_lock(&e->lock);
        if (err && !e->error) e->error = err;
        if (gpu > 0) e->gpu_seconds += gpu;
        e->frames++;
        sl->state = SLOT_FREE;
        pthread_cond_broadcast(&e->changed);
        pthread_mutex_unlock(&e->lock);
    }
}

static void copy_plane(void *dst, const void *src, ptrdiff_t stride,
                       unsigned w, unsigned h, unsigned bps)
{
    const size_t row = (size_t)w * bps;
    if (stride == (ptrdiff_t)row) {
        memcpy(dst, src, row * h);
        return;
    }
    for (unsigned y = 0; y < h; y++)
        memcpy((uint8_t *)dst + y * row, (const uint8_t *)src + y * stride, row);
}

static int setup(VmafVkEngine *e)
{
    int err = 0;
    if (e->want_chroma && !e->cw) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "vulkan: a feature needs the chroma planes, which YUV400 "
                 "pictures do not have\n");
        return -EINVAL;
    }
    if (e->want_chroma) {
        for (int p = 2; p < 6; p++) {
            err = vmaf_vk_buffer(&e->vk, &e->raw[p],
                                 e->alias_raw ? 16 : e->chroma_size,
                                 VMAF_VK_MEM_DEVICE);
            if (err) return err;
            if (!e->alias_raw) continue;
            VmafVkBuf per_slot[VMAF_VK_ENGINE_SLOTS];
            for (int k = 0; k < VMAF_VK_ENGINE_SLOTS; k++) {
                err = vmaf_vk_buffer(&e->vk, &e->slot[k].dev[p], e->chroma_size,
                                     VMAF_VK_MEM_DEVICE);
                if (err) return err;
                per_slot[k] = e->slot[k].dev[p];
            }
            err = vmaf_vk_alias(&e->vk, &e->raw[p], per_slot);
            if (err) return err;
        }
    }
    for (unsigned i = 0; i < e->n_modules && !err; i++) {
        VmafVkModule *m = &e->module[i];
        if (m->setup) err = m->setup(m, e);
    }
    if (!err) e->set_up = true;
    return err;
}

/* Records and submits the frame that waits in pending_slot. */
static int submit_pending(VmafVkEngine *e)
{
    if (!e->pending) return 0;
    e->pending = false;
    int err = 0;
    if (!e->set_up) {
        err = setup(e);
        if (err) return err;
    }
    const int k = e->pending_slot;
    VmafVkEngineSlot *sl = &e->slot[k];
    VmafVk *vk = &e->vk;
    const int planes = (e->want_chroma && sl->chroma_in) ? 6 : 2;

    if (e->alias_raw) {
        err = vmaf_vk_upload_begin(vk, k);
        if (err) return err;
        for (int p = 0; p < planes; p++)
            vmaf_vk_upload_copy(vk, &sl->in[p], &sl->dev[p], 0);
        err = vmaf_vk_upload_submit(vk, k);
        if (err) return err;
    }

    err = vmaf_vk_begin(vk, k);
    if (err) return err;
    /* the frames before on the queue are done with the shared buffers before
     * this one writes them */
    vmaf_vk_barrier(vk);
    vmaf_vk_stamp(vk, 0);
    if (!e->alias_raw) {
        for (int p = 0; p < planes; p++)
            vmaf_vk_copy(vk, &sl->in[p], &e->raw[p], 0);
        vmaf_vk_barrier(vk);
    }
    for (unsigned i = 0; i < e->n_modules && !err; i++) {
        VmafVkModule *m = &e->module[i];
        if (m->record) err = m->record(m, e, k, sl->index);
    }
    if (err) return err;
    vmaf_vk_barrier(vk);
    vmaf_vk_stamp(vk, 1);
    err = vmaf_vk_submit(vk, k);
    if (err) return err;

    pthread_mutex_lock(&e->lock);
    sl->state = SLOT_BUSY;
    pthread_cond_broadcast(&e->changed);
    pthread_mutex_unlock(&e->lock);
    return 0;
}

int vmaf_vk_engine_extract(VmafVkEngine *e, VmafPicture *ref,
                           VmafPicture *dist, unsigned index,
                           VmafFeatureCollector *fc)
{
    int err = 0;
    pthread_mutex_lock(&e->lock);
    err = e->error;
    pthread_mutex_unlock(&e->lock);
    if (err) return err;

    if (!e->pending || e->pending_index != index) {
        err = submit_pending(e);
        if (err) { set_error(e, err); return err; }

        const int k = e->next_slot;
        e->next_slot = (e->next_slot + 1) % VMAF_VK_ENGINE_SLOTS;
        VmafVkEngineSlot *sl = &e->slot[k];
        pthread_mutex_lock(&e->lock);
        while (sl->state != SLOT_FREE)
            pthread_cond_wait(&e->changed, &e->lock);
        err = e->error;
        sl->state = SLOT_FILLED;
        pthread_mutex_unlock(&e->lock);
        if (err) return err;

        copy_plane(sl->in[0].map, ref->data[0], ref->stride[0], e->w, e->h, e->bps);
        copy_plane(sl->in[1].map, dist->data[0], dist->stride[0], e->w, e->h, e->bps);
        sl->chroma_in = false;
        sl->index = index;
        sl->fc = fc;
        e->pending = true;
        e->pending_slot = k;
        e->pending_index = index;
    }

    VmafVkEngineSlot *sl = &e->slot[e->pending_slot];
    if (e->want_chroma && e->cw && !sl->chroma_in) {
        copy_plane(sl->in[2].map, ref->data[1], ref->stride[1], e->cw, e->ch, e->bps);
        copy_plane(sl->in[3].map, ref->data[2], ref->stride[2], e->cw, e->ch, e->bps);
        copy_plane(sl->in[4].map, dist->data[1], dist->stride[1], e->cw, e->ch, e->bps);
        copy_plane(sl->in[5].map, dist->data[2], dist->stride[2], e->cw, e->ch, e->bps);
        sl->chroma_in = true;
    }
    return 0;
}

int vmaf_vk_engine_flush(VmafVkEngine *e)
{
    int err = submit_pending(e);
    if (err) set_error(e, err);
    pthread_mutex_lock(&e->lock);
    for (int k = 0; k < VMAF_VK_ENGINE_SLOTS; k++)
        while (e->slot[k].state == SLOT_BUSY)
            pthread_cond_wait(&e->changed, &e->lock);
    err = e->error;
    pthread_mutex_unlock(&e->lock);
    return err;
}

int vmaf_vulkan_context_create(VmafVulkanContext **ctx, VmafVulkanState *s)
{
    if (!ctx || !s) return -EINVAL;
    VmafVulkanContext *c = calloc(1, sizeof(*c));
    if (!c) return -ENOMEM;
    c->state = s;
    pthread_mutex_init(&c->lock, NULL);
    *ctx = c;
    return 0;
}

void vmaf_vulkan_context_destroy(VmafVulkanContext **ctx)
{
    if (!ctx || !*ctx) return;
    VmafVulkanContext *c = *ctx;
    if (c->engine) {
        vmaf_vk_engine_flush(c->engine);
        engine_destroy(c->engine);
    }
    pthread_mutex_destroy(&c->lock);
    free(c);
    *ctx = NULL;
}
