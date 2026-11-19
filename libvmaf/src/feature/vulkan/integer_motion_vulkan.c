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

/* integer_motion.c on Vulkan: the blur of each reference frame and the sum of
 * absolute differences to the one before (or two before, with the five-frame
 * window) on the GPU (shaders motion_xy_wide, or motion_y and motion_x), the
 * scores as integer_motion.c works them out. */

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "motion_blend_tools.h"
#include "vulkan/engine.h"

#define DEFAULT_MOTION_MAX_VAL (10000.0)
#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

typedef struct MotionVkState {
    double motion_max_val;
    double motion_blend_factor;
    double motion_blend_offset;
    double motion_fps_weight;
    bool motion_five_frame_window;
    bool motion_moving_average;
    bool motion_force_zero;
    bool debug;
    VmafDictionary *feature_name_dict;
    VmafVkEngine *engine;
    VmafVkModule *module;
    bool joined;
} MotionVkState;

static const VmafOption options[] = {
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "forcing motion score to zero",
        .offset = offsetof(MotionVkState, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_factor",
        .alias = "mbf",
        .help = "blend motion score given an offset",
        .offset = offsetof(MotionVkState, motion_blend_factor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_offset",
        .alias = "mbo",
        .help = "blend motion score starting from this offset",
        .offset = offsetof(MotionVkState, motion_blend_offset),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 40.0,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(MotionVkState, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_max_val",
        .alias = "mmxv",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(MotionVkState, motion_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_MOTION_MAX_VAL,
        .min = 0.0,
        .max = 10000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_five_frame_window",
        .alias = "mffw",
        .help = "use five-frame temporal window",
        .offset = offsetof(MotionVkState, motion_five_frame_window),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_moving_average",
        .alias = "mma",
        .help = "use moving average for motion scores after first frame",
        .offset = offsetof(MotionVkState, motion_moving_average),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionVkState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    { 0 }
};

/* --- the module: the GPU work, shared by every motion feature extractor --- */

typedef struct MotionModule {
    MotionVkState **unit;
    unsigned n_unit;
    bool need_stride[2]; ///< stride 1 (the frame before), stride 2 (five-frame window)
    bool fused;          ///< both convolutions in one pass (motion_xy_wide)
    uint32_t gx, gy, mgx, mgy;
    VmafVkBuf mfr[3], mtmp, mpart[2];
    VmafVkBuf readback[VMAF_VK_ENGINE_SLOTS][2];
    bool with_motion[VMAF_VK_ENGINE_SLOTS][2];
    VmafVkPipe mxy[2][3][3], my[2][3][3], mx[2][3][3];
} MotionModule;

static int module_setup(VmafVkModule *m, VmafVkEngine *e)
{
    MotionModule *mm = m->priv;
    VmafVk *vk = &e->vk;
    for (unsigned u = 0; u < mm->n_unit; u++) {
        if (!mm->unit[u] || mm->unit[u]->motion_force_zero) continue;
        mm->need_stride[mm->unit[u]->motion_five_frame_window ? 1 : 0] = true;
    }
    if (!mm->need_stride[0] && !mm->need_stride[1]) return 0;

    /* motion_xy_wide: faster on NVIDIA, 15% slower on an RDNA 2 iGPU than the
     * two passes (2026-11-18); up to 15 bits per sample */
    mm->fused = e->s->vendor_id != 0x1002;
    const char *env = getenv("VMAF_VULKAN_MOTION_SPLIT");
    if (env && *env) mm->fused = atoi(env) == 0;
    if (e->bpc > 15) mm->fused = false;
    mm->gx = (e->w + 15) / 16;
    mm->gy = (e->h + 15) / 16;
    mm->mgx = mm->fused ? (e->w + 63) / 64 : mm->gx;
    mm->mgy = mm->gy;

    int err = 0;
    for (int i = 0; i < 3 && !err; i++)
        err = vmaf_vk_buffer(vk, &mm->mfr[i], e->luma_size, VMAF_VK_MEM_DEVICE);
    if (!err)
        err = vmaf_vk_buffer(vk, &mm->mtmp,
                             mm->fused ? 16 : (VkDeviceSize)e->w * e->h * 4,
                             VMAF_VK_MEM_DEVICE);
    for (int k = 0; k < 2 && !err; k++) {
        if (!mm->need_stride[k]) continue;
        const VkDeviceSize size = (VkDeviceSize)mm->mgx * mm->mgy * 4;
        err = vmaf_vk_buffer(vk, &mm->mpart[k], size, VMAF_VK_MEM_DEVICE);
        for (int s = 0; s < VMAF_VK_ENGINE_SLOTS && !err; s++)
            err = vmaf_vk_buffer(vk, &mm->readback[s][k], size,
                                 VMAF_VK_MEM_READBACK);
        for (int a = 0; a < 3 && !err; a++) {
            for (int b = 0; b < 3 && !err; b++) {
                if (a == b) continue;
                VmafVkBuf *bind[4] = { &mm->mfr[a], &mm->mfr[b], &mm->mtmp,
                                       &mm->mpart[k] };
                if (mm->fused) {
                    err = vmaf_vk_pipeline(vk, &mm->mxy[k][a][b],
                                           "motion_xy_wide", bind, 4);
                } else {
                    err = vmaf_vk_pipeline(vk, &mm->my[k][a][b], "motion_y",
                                           bind, 4);
                    if (!err)
                        err = vmaf_vk_pipeline(vk, &mm->mx[k][a][b],
                                               "motion_x", bind, 4);
                }
            }
        }
    }
    return err;
}

static int module_record(VmafVkModule *m, VmafVkEngine *e, int slot,
                         unsigned index)
{
    MotionModule *mm = m->priv;
    VmafVk *vk = &e->vk;
    mm->with_motion[slot][0] = mm->with_motion[slot][1] = false;
    if (!mm->need_stride[0] && !mm->need_stride[1]) return 0;
    const int cur = index % 3;
    /* kept: the frames after compare with it, while its slot takes new
     * frames meanwhile */
    vmaf_vk_copy(vk, vmaf_vk_engine_plane(e, slot, 0), &mm->mfr[cur], 0);
    vmaf_vk_barrier(vk);
    bool any = false;
    for (int k = 0; k < 2; k++) {
        const unsigned stride = k + 1;
        if (!mm->need_stride[k] || index < stride) continue;
        mm->with_motion[slot][k] = any = true;
        const int prev = (index - stride) % 3;
        const int32_t pc[3] = { (int32_t)e->w, (int32_t)e->h, (int32_t)e->bpc };
        vmaf_vk_push(vk, pc, sizeof(pc));
        if (mm->fused) { /* the two strides' passes side by side */
            vmaf_vk_bind(vk, &mm->mxy[k][prev][cur]);
            vmaf_vk_dispatch(vk, mm->mgx, mm->mgy);
        } else {
            vmaf_vk_bind(vk, &mm->my[k][prev][cur]);
            vmaf_vk_dispatch(vk, mm->gx, mm->gy);
            vmaf_vk_barrier(vk);
            vmaf_vk_bind(vk, &mm->mx[k][prev][cur]);
            vmaf_vk_dispatch(vk, mm->gx, mm->gy);
            vmaf_vk_barrier(vk);
        }
    }
    if (!any) return 0;
    if (mm->fused) vmaf_vk_barrier(vk);
    for (int k = 0; k < 2; k++)
        if (mm->with_motion[slot][k])
            vmaf_vk_copy(vk, &mm->mpart[k], &mm->readback[slot][k], 0);
    return 0;
}

static int module_results(VmafVkModule *m, VmafVkEngine *e, int slot,
                          unsigned index, VmafFeatureCollector *fc)
{
    MotionModule *mm = m->priv;
    double sad[2] = { 0., 0. };
    for (int k = 0; k < 2; k++) {
        if (!mm->with_motion[slot][k]) continue;
        const uint32_t *p = mm->readback[slot][k].map;
        uint64_t sum = 0;
        for (uint32_t q = 0; q < mm->mgx * mm->mgy; q++) sum += p[q];
        sad[k] = (double)sum / 256. / (e->w * e->h);
    }
    int err = 0;
    for (unsigned u = 0; u < mm->n_unit; u++) {
        MotionVkState *s = mm->unit[u];
        if (!s) continue;
        double score = 0.;
        if (!s->motion_force_zero) {
            const int k = s->motion_five_frame_window ? 1 : 0;
            if (index >= (unsigned)k + 1)
                score = MIN(sad[k] * s->motion_fps_weight, s->motion_max_val);
        }
        err |= vmaf_feature_collector_append_with_dict(fc,
                s->feature_name_dict, "VMAF_integer_feature_motion_sad_score",
                score, index);
        if (s->debug)
            err |= vmaf_feature_collector_append_with_dict(fc,
                    s->feature_name_dict, "VMAF_integer_feature_motion_score",
                    score, index);
    }
    return err;
}

static void module_close(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    MotionModule *mm = m->priv;
    if (!mm) return;
    free(mm->unit);
    free(mm);
    m->priv = NULL;
}

static int module_create(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    MotionModule *mm = calloc(1, sizeof(*mm));
    if (!mm) return -ENOMEM;
    m->priv = mm;
    m->setup = module_setup;
    m->record = module_record;
    m->results = module_results;
    m->close = module_close;
    return 0;
}

/* --- the feature extractor --- */

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    MotionVkState *s = fex->priv;
    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) return -ENOMEM;

    int err = vmaf_vk_engine_get(fex->vk_ctx, pix_fmt, bpc, w, h, &s->engine);
    if (err) return err;
    s->module = vmaf_vk_engine_module(s->engine, "motion", module_create);
    if (!s->module) return -EINVAL;
    MotionModule *mm = s->module->priv;
    MotionVkState **u = realloc(mm->unit, sizeof(*u) * (mm->n_unit + 1));
    if (!u) return -ENOMEM;
    mm->unit = u;
    mm->unit[mm->n_unit++] = s;
    s->joined = true;
    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    MotionVkState *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;
    return vmaf_vk_engine_extract(s->engine, ref_pic, dist_pic, index,
                                  feature_collector);
}

/* integer_motion.c's flush(): motion2 and motion3 from the SAD scores. */
static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    MotionVkState *s = fex->priv;

    int err = vmaf_vk_engine_flush(s->engine);
    if (err) return err;
    if (!s->feature_name_dict) return 1;

    const VmafDictionaryEntry *sad_entry = vmaf_dictionary_get(
            &s->feature_name_dict, "VMAF_integer_feature_motion_sad_score", 0);
    if (!sad_entry) return -EINVAL;
    const char *sad_name = sad_entry->val;

    unsigned n = 0;
    double score;
    while (!vmaf_feature_collector_get_score(feature_collector, sad_name, &score, n))
        n++;
    const unsigned stride = s->motion_five_frame_window ? 2 : 1;
    const unsigned min_idx = s->motion_five_frame_window ? 2 : 1;
    if (!n) return 1;

    double stamp_value = 0.;
    if (n > min_idx) {
        double sad_at_min_idx;
        if (!vmaf_feature_collector_get_score(feature_collector, sad_name,
                                              &sad_at_min_idx, min_idx)) {
            stamp_value = MIN(motion_blend(sad_at_min_idx,
                                           s->motion_blend_factor,
                                           s->motion_blend_offset),
                              s->motion_max_val);
        }
    }

    double prev_processed = 0.;
    for (unsigned i = 0; i < n; i++) {
        double sad_i;
        vmaf_feature_collector_get_score(feature_collector, sad_name, &sad_i, i);

        double motion2;
        if (i < min_idx) {
            motion2 = 0.;
        } else {
            const int lo_idx = (int)i - (int)(stride - 1);
            const int hi_idx = (int)i + 1;
            double hi;
            const bool has_hi = !vmaf_feature_collector_get_score(
                    feature_collector, sad_name, &hi, hi_idx);
            if (!has_hi) {
                motion2 = sad_i;
            } else if (lo_idx >= (int)min_idx) {
                double lo;
                vmaf_feature_collector_get_score(
                        feature_collector, sad_name, &lo, lo_idx);
                motion2 = lo < hi ? lo : hi;
            } else {
                motion2 = hi;
            }
        }

        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion2_score", motion2, i);

        double motion3;
        if (i < min_idx) {
            motion3 = stamp_value;
            prev_processed = stamp_value;
        } else {
            double processed = MIN(motion_blend(motion2,
                                                s->motion_blend_factor,
                                                s->motion_blend_offset),
                                   s->motion_max_val);
            motion3 = s->motion_moving_average
                    ? (processed + prev_processed) / 2.0
                    : processed;
            prev_processed = processed;
        }

        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion3_score", motion3, i);
    }

    /* written once */
    vmaf_dictionary_free(&s->feature_name_dict);
    return 1;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    MotionVkState *s = fex->priv;
    if (s->engine) {
        vmaf_vk_engine_flush(s->engine);
        if (s->joined) {
            MotionModule *mm = s->module->priv;
            for (unsigned u = 0; u < mm->n_unit; u++)
                if (mm->unit[u] == s) mm->unit[u] = NULL;
        }
        vmaf_vk_engine_release(fex->vk_ctx, &s->engine);
    }
    return vmaf_dictionary_free(&s->feature_name_dict);
}

static const char *provided_features[] = {
    "VMAF_integer_feature_motion_sad_score",
    "VMAF_integer_feature_motion_score",
    "VMAF_integer_feature_motion2_score",
    "VMAF_integer_feature_motion3_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_motion_vulkan = {
    .name = "motion_vulkan",
    .options = options,
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .priv_size = sizeof(MotionVkState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_VULKAN,
};
