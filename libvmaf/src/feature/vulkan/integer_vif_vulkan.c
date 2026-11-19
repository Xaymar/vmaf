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

/* integer_vif.c on Vulkan: the subsampled scales (shaders vif_sub_v and
 * vif_sub_h) once, and the statistic of each scale (vif_stat_v and vif_stat_h)
 * with its vertical pass once and each horizontal pass for two
 * enhancement gain limits; each workgroup writes its partial sums of
 * the numerator and denominator, added up here as vif_statistic_8/16 do. */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "integer_vif.h"
#include "log.h"
#include "vulkan/engine.h"

typedef struct VifVkState {
    double vif_enhn_gain_limit;
    bool vif_skip_scale0;
    bool debug;
    VmafDictionary *feature_name_dict;
    VmafVkEngine *engine;
    VmafVkModule *module;
    bool joined;
} VifVkState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(VifVkState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on vif, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(VifVkState, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_skip_scale0",
        .alias = "ssclz",
        .help = "when set, skip scale 0 calculations",
        .offset = offsetof(VifVkState, vif_skip_scale0),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

/* --- the module: the GPU work, shared by every VIF feature extractor --- */

typedef struct VifPC {
    int32_t W, H, scale, srckind, bpc, egl_int;
    double egl;
    int32_t units, egl_int2;
    double egl2;
} VifPC;

/* The gain limit as the statistic's fast path takes it: a whole number from 1
 * to 1000, else -1 (the exact double path only). */
static int vif_egl_int(double egl)
{
    return egl >= 1.0 && egl <= 1000.0 && egl == (double)(int)egl ? (int)egl : -1;
}

typedef struct VifUnit {
    double egl;
    VmafVkBuf part[4];
    VmafVkBuf readback[VMAF_VK_ENGINE_SLOTS][4];
    VmafVkPipe stat_v[4], stat_h[4];
} VifUnit;

typedef struct VifModule {
    VifVkState **fex;
    unsigned n_fex;
    VifUnit *unit;
    unsigned n_unit;
    int vw[4], vh[4];
    uint32_t groups[4];
    VmafVkBuf dr[4], dd[4], tmp, log2tab;
    VmafVkPipe sub_v[4], sub_h[4];
} VifModule;

static int module_setup(VmafVkModule *m, VmafVkEngine *e)
{
    VifModule *vm = m->priv;
    VmafVk *vk = &e->vk;
    for (unsigned i = 0; i < vm->n_fex; i++) {
        if (!vm->fex[i]) continue;
        const double egl = vm->fex[i]->vif_enhn_gain_limit;
        unsigned u = 0;
        while (u < vm->n_unit && vm->unit[u].egl != egl) u++;
        if (u < vm->n_unit) continue;
        VifUnit *n = realloc(vm->unit, sizeof(*n) * (vm->n_unit + 1));
        if (!n) return -ENOMEM;
        vm->unit = n;
        memset(&vm->unit[vm->n_unit], 0, sizeof(*n));
        vm->unit[vm->n_unit++].egl = egl;
    }
    if (!vm->n_unit) return 0;

    for (int s = 0; s < 4; s++) {
        vm->vw[s] = s ? vm->vw[s - 1] / 2 : (int)e->w;
        vm->vh[s] = s ? vm->vh[s - 1] / 2 : (int)e->h;
        vm->groups[s] = ((vm->vw[s] + 15) / 16) * ((vm->vh[s] + 15) / 16);
    }
    int err = 0;
    for (int s = 1; s < 4 && !err; s++) {
        const VkDeviceSize size = (VkDeviceSize)vm->vw[s] * vm->vh[s] * 4 + 16;
        err = vmaf_vk_buffer(vk, &vm->dr[s], size, VMAF_VK_MEM_DEVICE);
        if (!err) err = vmaf_vk_buffer(vk, &vm->dd[s], size, VMAF_VK_MEM_DEVICE);
    }
    if (!err)
        err = vmaf_vk_buffer(vk, &vm->tmp, (VkDeviceSize)e->w * e->h * 5 * 4 + 16,
                             VMAF_VK_MEM_DEVICE);
    if (!err) err = vmaf_vk_buffer(vk, &vm->log2tab, 65536 * 4, VMAF_VK_MEM_UPLOAD);
    if (err) return err;
    {
        /* log2_table of integer_vif.c, from 32767 on */
        uint32_t *t = vm->log2tab.map;
        memset(t, 0, 65536 * 4);
        for (unsigned i = 32767; i < 65536; ++i)
            t[i] = (uint16_t)round(log2f((float)i) * 2048);
    }
    for (unsigned u = 0; u < vm->n_unit && !err; u++) {
        for (int s = 0; s < 4 && !err; s++) {
            const VkDeviceSize size = (VkDeviceSize)vm->groups[s] * 4 * 8;
            err = vmaf_vk_buffer(vk, &vm->unit[u].part[s], size, VMAF_VK_MEM_DEVICE);
            for (int k = 0; k < VMAF_VK_ENGINE_SLOTS && !err; k++)
                err = vmaf_vk_buffer(vk, &vm->unit[u].readback[k][s], size,
                                     VMAF_VK_MEM_READBACK);
        }
    }
    VmafVkBuf *sref[4] = { &e->raw[0], &vm->dr[1], &vm->dr[2], &vm->dr[3] };
    VmafVkBuf *sdis[4] = { &e->raw[1], &vm->dd[1], &vm->dd[2], &vm->dd[3] };
    for (int s = 1; s < 4 && !err; s++) {
        VmafVkBuf *b[7] = { sref[s - 1], sdis[s - 1], &vm->tmp, &vm->log2tab,
                            &vm->unit[0].part[s], &vm->dr[s], &vm->dd[s] };
        err = vmaf_vk_pipeline(vk, &vm->sub_v[s], "vif_sub_v", b, 7);
        if (!err) err = vmaf_vk_pipeline(vk, &vm->sub_h[s], "vif_sub_h", b, 7);
    }
    /* The filtered statistics do not depend on the gain limit: one vertical
     * pass per scale, and each horizontal pass scores two gain limits. */
    for (int s = 0; s < 4 && !err; s++) {
        VmafVkBuf *b[7] = { sref[s], sdis[s], &vm->tmp, &vm->log2tab,
                            &vm->unit[0].part[s], &vm->tmp, &vm->tmp };
        err = vmaf_vk_pipeline(vk, &vm->unit[0].stat_v[s], "vif_stat_v", b, 7);
    }
    for (unsigned u = 0; u < vm->n_unit && !err; u += 2) {
        const unsigned u2 = u + 1 < vm->n_unit ? u + 1 : u;
        for (int s = 0; s < 4 && !err; s++) {
            VmafVkBuf *b[7] = { sref[s], sdis[s], &vm->tmp, &vm->log2tab,
                                &vm->unit[u].part[s], &vm->unit[u2].part[s],
                                &vm->tmp };
            err = vmaf_vk_pipeline(vk, &vm->unit[u].stat_h[s], "vif_stat_h", b, 7);
        }
    }
    return err;
}

static int module_record(VmafVkModule *m, VmafVkEngine *e, int slot,
                         unsigned index)
{
    (void) index;
    VifModule *vm = m->priv;
    VmafVk *vk = &e->vk;
    if (!vm->n_unit) return 0;
    const int srckind0 = e->bps == 2 ? 1 : 0;
    for (int s = 0; s < 4; s++) {
        if (s > 0) {
            VifPC pc = { vm->vw[s - 1], vm->vh[s - 1], s,
                         s == 1 ? srckind0 : 2, (int32_t)e->bpc, -1,
                         vm->unit[0].egl, 1, -1, vm->unit[0].egl };
            vmaf_vk_push(vk, &pc, sizeof(pc));
            vmaf_vk_bind(vk, &vm->sub_v[s]);
            vmaf_vk_dispatch(vk, (vm->vw[s - 1] + 15) / 16,
                             (vm->vh[s - 1] / 2 + 15) / 16);
            vmaf_vk_barrier(vk);
            vmaf_vk_bind(vk, &vm->sub_h[s]);
            vmaf_vk_dispatch(vk, (vm->vw[s - 1] / 2 + 15) / 16,
                             (vm->vh[s - 1] / 2 + 15) / 16);
            vmaf_vk_barrier(vk);
        }
        const uint32_t gx = (vm->vw[s] + 15) / 16, gy = (vm->vh[s] + 15) / 16;
        VifPC pc = { vm->vw[s], vm->vh[s], s, s == 0 ? srckind0 : 2,
                     (int32_t)e->bpc, -1, vm->unit[0].egl, 1, -1, vm->unit[0].egl };
        vmaf_vk_push(vk, &pc, sizeof(pc));
        vmaf_vk_bind(vk, &vm->unit[0].stat_v[s]);
        vmaf_vk_dispatch(vk, gx, gy);
        vmaf_vk_barrier(vk);
        for (unsigned u = 0; u < vm->n_unit; u += 2) {
            const bool two = u + 1 < vm->n_unit;
            const VifUnit *a = &vm->unit[u], *b = &vm->unit[two ? u + 1 : u];
            pc.egl_int = vif_egl_int(a->egl);
            pc.egl = a->egl;
            pc.units = two ? 2 : 1;
            pc.egl_int2 = vif_egl_int(b->egl);
            pc.egl2 = b->egl;
            vmaf_vk_push(vk, &pc, sizeof(pc));
            vmaf_vk_bind(vk, &vm->unit[u].stat_h[s]);
            vmaf_vk_dispatch(vk, gx, gy);
        }
        vmaf_vk_barrier(vk);
    }
    for (unsigned u = 0; u < vm->n_unit; u++)
        for (int s = 0; s < 4; s++)
            vmaf_vk_copy(vk, &vm->unit[u].part[s], &vm->unit[u].readback[slot][s], 0);
    return 0;
}

typedef struct VifScore {
    struct {
        float num;
        float den;
    } scale[4];
} VifScore;

static int write_scores(VmafFeatureCollector *fc, unsigned index, VifScore vif,
                        VifVkState *s)
{
    VmafDictionary *d = s->feature_name_dict;
    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(fc, d,
            "VMAF_integer_feature_vif_scale0_score",
            s->vif_skip_scale0 ? 0.0f : vif.scale[0].num / vif.scale[0].den, index);
    err |= vmaf_feature_collector_append_with_dict(fc, d,
            "VMAF_integer_feature_vif_scale1_score",
            vif.scale[1].num / vif.scale[1].den, index);
    err |= vmaf_feature_collector_append_with_dict(fc, d,
            "VMAF_integer_feature_vif_scale2_score",
            vif.scale[2].num / vif.scale[2].den, index);
    err |= vmaf_feature_collector_append_with_dict(fc, d,
            "VMAF_integer_feature_vif_scale3_score",
            vif.scale[3].num / vif.scale[3].den, index);
    if (!s->debug) return err;

    double score_num, score_den;
    if (s->vif_skip_scale0) {
        score_num = (double)vif.scale[1].num + (double)vif.scale[2].num + (double)vif.scale[3].num;
        score_den = (double)vif.scale[1].den + (double)vif.scale[2].den + (double)vif.scale[3].den;
    } else {
        score_num = (double)vif.scale[0].num + (double)vif.scale[1].num +
                    (double)vif.scale[2].num + (double)vif.scale[3].num;
        score_den = (double)vif.scale[0].den + (double)vif.scale[1].den +
                    (double)vif.scale[2].den + (double)vif.scale[3].den;
    }
    const double score = score_den == 0.0 ? 1.0f : score_num / score_den;
    err |= vmaf_feature_collector_append_with_dict(fc, d, "integer_vif", score, index);
    err |= vmaf_feature_collector_append_with_dict(fc, d, "integer_vif_num", score_num, index);
    err |= vmaf_feature_collector_append_with_dict(fc, d, "integer_vif_den", score_den, index);
    static const char *const nn[4] = { "integer_vif_num_scale0", "integer_vif_num_scale1",
                                       "integer_vif_num_scale2", "integer_vif_num_scale3" };
    static const char *const dn[4] = { "integer_vif_den_scale0", "integer_vif_den_scale1",
                                       "integer_vif_den_scale2", "integer_vif_den_scale3" };
    for (int k = 0; k < 4; k++) {
        const bool skip = k == 0 && s->vif_skip_scale0;
        err |= vmaf_feature_collector_append_with_dict(fc, d, nn[k],
                skip ? 0.0f : vif.scale[k].num, index);
        err |= vmaf_feature_collector_append_with_dict(fc, d, dn[k],
                skip ? -1.0f : vif.scale[k].den, index);
    }
    return err;
}

static int module_results(VmafVkModule *m, VmafVkEngine *e, int slot,
                          unsigned index, VmafFeatureCollector *fc)
{
    (void) e;
    VifModule *vm = m->priv;
    int err = 0;
    for (unsigned u = 0; u < vm->n_unit; u++) {
        VifScore score;
        for (int s = 0; s < 4; s++) {
            const int64_t *p = vm->unit[u].readback[slot][s].map;
            int64_t sum[4] = { 0, 0, 0, 0 };
            for (uint32_t k = 0; k < vm->groups[s]; k++)
                for (int q = 0; q < 4; q++) sum[q] += p[k * 4 + q];
            /* as vif_statistic_8/16 */
            score.scale[s].num =
                (float)(sum[0] / 2048.0 + (sum[3] - ((sum[2]) / 16384.0) / (65025.0)));
            score.scale[s].den = (float)(sum[1] / 2048.0 + sum[3]);
        }
        for (unsigned i = 0; i < vm->n_fex; i++) {
            VifVkState *s = vm->fex[i];
            if (s && s->vif_enhn_gain_limit == vm->unit[u].egl)
                err |= write_scores(fc, index, score, s);
        }
    }
    return err;
}

static void module_close(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    VifModule *vm = m->priv;
    if (!vm) return;
    free(vm->unit);
    free(vm->fex);
    free(vm);
    m->priv = NULL;
}

static int module_create(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    VifModule *vm = calloc(1, sizeof(*vm));
    if (!vm) return -ENOMEM;
    m->priv = vm;
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
    VifVkState *s = fex->priv;
    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) return -ENOMEM;

    int err = vmaf_vk_engine_get(fex->vk_ctx, pix_fmt, bpc, w, h, &s->engine);
    if (err) return err;
    s->module = vmaf_vk_engine_module(s->engine, "vif", module_create);
    if (!s->module) return -EINVAL;
    VifModule *vm = s->module->priv;
    VifVkState **f = realloc(vm->fex, sizeof(*f) * (vm->n_fex + 1));
    if (!f) return -ENOMEM;
    vm->fex = f;
    vm->fex[vm->n_fex++] = s;
    s->joined = true;
    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    VifVkState *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;
    return vmaf_vk_engine_extract(s->engine, ref_pic, dist_pic, index,
                                  feature_collector);
}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    (void) feature_collector;
    VifVkState *s = fex->priv;
    int err = vmaf_vk_engine_flush(s->engine);
    return err ? err : 1;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    VifVkState *s = fex->priv;
    if (s->engine) {
        vmaf_vk_engine_flush(s->engine);
        if (s->joined) {
            VifModule *vm = s->module->priv;
            for (unsigned i = 0; i < vm->n_fex; i++)
                if (vm->fex[i] == s) vm->fex[i] = NULL;
        }
        vmaf_vk_engine_release(fex->vk_ctx, &s->engine);
    }
    return vmaf_dictionary_free(&s->feature_name_dict);
}

static const char *provided_features[] = {
    "VMAF_integer_feature_vif_scale0_score", "VMAF_integer_feature_vif_scale1_score",
    "VMAF_integer_feature_vif_scale2_score", "VMAF_integer_feature_vif_scale3_score",
    "integer_vif", "integer_vif_num", "integer_vif_den", "integer_vif_num_scale0",
    "integer_vif_den_scale0", "integer_vif_num_scale1", "integer_vif_den_scale1",
    "integer_vif_num_scale2", "integer_vif_den_scale2", "integer_vif_num_scale3",
    "integer_vif_den_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_vif_vulkan = {
    .name = "vif_vulkan",
    .options = options,
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .priv_size = sizeof(VifVkState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_VULKAN,
};
