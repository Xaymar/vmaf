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

/* integer_adm.c on Vulkan. The wavelets of the four scales, decouple, the
 * contrast sensitivity weighting and the contrast masking run on the GPU
 * (shaders adm_dwt, adm_den, adm_dcm or adm_dcm_multi, adm_cmsum), in the same
 * integer steps as integer_adm.c; their row sums come back and the scores are
 * worked out from them as integer_adm.c does (adm_cm(), adm_csf_den_*()).
 *
 * Every ADM feature extractor of a context is a unit of one module: the
 * wavelets and the denominator are computed once for all of them, decouple,
 * the weighting and the masking once for each set of the options they depend
 * on (the gain limit, the viewing distance, the display height and the CSF);
 * the noise weight, the DLM weight and the minimum are applied on the CPU. */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "barten_csf_tools.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "integer_adm.h"
#include "log.h"
#include "vulkan/engine.h"

/* As in integer_adm.c. */
static inline float
dwt_quant_step(const struct dwt_model_params *params, int lambda, int theta,
               double adm_norm_view_dist, int adm_ref_display_height)
{
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;
    float temp = log10(pow(2.0, lambda + 1) * params->f0 * params->g[theta] / r);
    float Q = 2.0 * params->a * pow(10.0, params->k * temp * temp) /
              dwt_7_9_basis_function_amplitudes[lambda][theta];
    return Q;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

typedef struct AdmVkState {
    bool debug;
    bool adm_skip_aim;
    bool adm_skip_scale0;
    double adm_csf_diag_scale;
    double adm_csf_scale;
    double adm_dlm_weight;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    double adm_norm_view_dist_extra;
    double adm_noise_weight;
    double adm_min_val;
    int adm_ref_display_height;
    int adm_csf_mode;
    VmafDictionary *feature_name_dict;
    VmafDictionary *feature_name_dict_extra;
    VmafVkEngine *engine;
    VmafVkModule *module;
    bool joined;
} AdmVkState;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(AdmVkState, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_csf_scale",
        .alias = "scf",
        .help = "scale coefficient for the horizontal & vertical direction terms of CSF",
        .offset = offsetof(AdmVkState, adm_csf_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_diag_scale",
        .alias = "scfd",
        .help = "scale coefficient for the diagonal direction term of CSF",
        .offset = offsetof(AdmVkState, adm_csf_diag_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_DIAG_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_dlm_weight",
        .alias = "dlmw",
        .help = "linear weighting between DLM and AIM; 1 corresponds to DLM-only",
        .offset = offsetof(AdmVkState, adm_dlm_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.5,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(AdmVkState, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmVkState, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist_extra",
        .alias = "nvde",
        .help = "extra normalized viewing distance; when > 0, ADM is also "
                "evaluated at this viewing distance reusing the shared "
                "DWT/decouple stages, exposed under the same feature names "
                "carrying this distance's nvd suffix",
        .offset = offsetof(AdmVkState, adm_norm_view_dist_extra),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.0,
        .min = 0.0,
        .max = 24.0,
    },
    {
        .name = "adm_ref_display_height",
        .alias = "rdh",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmVkState, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_mode",
        .alias = "csf",
        .help = "contrast sensitivity function",
        .offset = offsetof(AdmVkState, adm_csf_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_CSF_MODE,
        .min = 0,
        .max = 3,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_noise_weight",
        .alias = "nw",
        .help = "noise weight",
        .offset = offsetof(AdmVkState, adm_noise_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_skip_aim",
        .help = "skip the calculation of AIM",
        .offset = offsetof(AdmVkState, adm_skip_aim),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_skip_scale0",
        .alias = "ssz",
        .help = "skip the calculation of scale 0",
        .offset = offsetof(AdmVkState, adm_skip_scale0),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_min_val",
        .alias = "min",
        .help = "minimum value allowed; lower values will be clipped to this value",
        .offset = offsetof(AdmVkState, adm_min_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_MIN_VAL,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

/* --- the module --- */

/* the shaders' push constants (adm_common.glsl) */
typedef struct AdmPC {
    int32_t SW, SH, BW, BH, scale, prevN;
    double egl;
    float cos1sq;
    int32_t rf0, rf1, rf2;
    int32_t dl, dt, dr, db;
    int32_t cl, ct, cr, cb;
    int32_t srow, erow, scol, ecol;
    int32_t shc0, shc1, shc2;
    int32_t bpc;
    int32_t aim;
    int32_t egl_int;
    int32_t dbg;
} AdmPC;

typedef struct AdmHost {
    int left, top, right, bottom;
    uint32_t shift_inner;
    int shift_cub[3];
    float rfactor[3];
    uint32_t i_rfactor[3];
    int den_cub;
} AdmHost;

/* ADM with one set of the options the shaders depend on. */
typedef struct AdmUnit {
    double egl, nvd;
    int rdh, csf_mode;
    double csf_scale, csf_diag_scale;
    bool aim;
    AdmPC apc[4];
    AdmHost ah[4];
    VmafVkBuf cm[4], aimb[4];
    VmafVkPipe dcm[4], cmsum[4], aimsum[4];
    VmafVkBuf rb_cm[VMAF_VK_ENGINE_SLOTS][4], rb_aim[VMAF_VK_ENGINE_SLOTS][4];
} AdmUnit;

/* Units with the same gain limit: one pass per scale (adm_dcm_multi) reads and
 * decouples the bands once for all of them. */
typedef struct AdmGroup {
    int *units;
    unsigned n;
    VmafVkBuf tab[4];
    VmafVkPipe p[4];
} AdmGroup;

/* What a feature extractor writes: at its viewing distance, and at the extra
 * one (adm_norm_view_dist_extra) if it has one. */
typedef struct AdmView {
    AdmVkState *s;
    double nvd;
    bool extra;
    int unit;
} AdmView;

typedef struct AdmModule {
    AdmView *view;
    unsigned n_view;
    AdmUnit *unit;
    unsigned n_unit;
    AdmGroup *group;
    unsigned n_group;
    int sw[4], sh[4], bw[4], bh[4];
    int tiles_s[4], tile_base[4];
    VmafVkBuf acur[4], aden[4], atmp, adiv;
    VmafVkBuf rb_den[VMAF_VK_ENGINE_SLOTS][4];
    VmafVkPipe adwt[4], aden_p[4];
} AdmModule;

static void csf_factors(const AdmUnit *a, int scale, float *f1, float *f2)
{
    if (a->csf_mode == ADM_CSF_MODE_BARTEN) {
        *f1 = barten_csf(scale, a->nvd, a->rdh, DEFAULT_ADM_CSF_LUM, a->csf_scale);
        *f2 = barten_csf(scale, a->nvd, a->rdh, DEFAULT_ADM_CSF_LUM, a->csf_diag_scale);
    } else if (a->csf_mode == ADM_CSF_MODE_BARTEN_WATSON_BLEND) {
        *f1 = barten_watson_blend_csf(scale, 0, a->nvd, a->rdh);
        *f2 = barten_watson_blend_csf(scale, 1, a->nvd, a->rdh);
    } else if (a->csf_mode == ADM_CSF_MODE_BARTEN_WATSON_BLEND_MAE) {
        *f1 = barten_watson_blend_csf_mae(scale, 0, a->nvd, a->rdh);
        *f2 = barten_watson_blend_csf_mae(scale, 1, a->nvd, a->rdh);
    } else {
        *f1 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, a->nvd, a->rdh);
        *f2 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, a->nvd, a->rdh);
    }
}

static int unit_of(AdmModule *am, const AdmVkState *s, double nvd)
{
    for (unsigned u = 0; u < am->n_unit; u++) {
        const AdmUnit *a = &am->unit[u];
        if (a->egl == s->adm_enhn_gain_limit && a->nvd == nvd &&
            a->rdh == s->adm_ref_display_height && a->csf_mode == s->adm_csf_mode &&
            a->csf_scale == s->adm_csf_scale &&
            a->csf_diag_scale == s->adm_csf_diag_scale)
            return u;
    }
    AdmUnit *n = realloc(am->unit, sizeof(*n) * (am->n_unit + 1));
    if (!n) return -1;
    am->unit = n;
    AdmUnit *a = &am->unit[am->n_unit];
    memset(a, 0, sizeof(*a));
    a->egl = s->adm_enhn_gain_limit;
    a->nvd = nvd;
    a->rdh = s->adm_ref_display_height;
    a->csf_mode = s->adm_csf_mode;
    a->csf_scale = s->adm_csf_scale;
    a->csf_diag_scale = s->adm_csf_diag_scale;
    return am->n_unit++;
}

static int module_setup(VmafVkModule *m, VmafVkEngine *e)
{
    AdmModule *am = m->priv;
    VmafVk *vk = &e->vk;
    int err = 0;

    for (unsigned v = 0; v < am->n_view; v++) {
        AdmView *vw = &am->view[v];
        if (!vw->s) continue;
        vw->unit = unit_of(am, vw->s, vw->nvd);
        if (vw->unit < 0) return -ENOMEM;
        if (!vw->s->adm_skip_aim) am->unit[vw->unit].aim = true;
    }
    if (!am->n_unit) return 0;

    const int W = e->w, H = e->h;
    for (int s = 0; s < 4; s++) {
        am->sw[s] = s == 0 ? W : am->bw[s - 1];
        am->sh[s] = s == 0 ? H : am->bh[s - 1];
        am->bw[s] = (am->sw[s] + 1) / 2;
        am->bh[s] = (am->sh[s] + 1) / 2;
    }
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
    for (unsigned u = 0; u < am->n_unit; u++) {
        AdmUnit *a = &am->unit[u];
        for (int s = 0; s < 4; s++) {
            AdmPC *pc = &a->apc[s];
            AdmHost *hs = &a->ah[s];
            memset(pc, 0, sizeof(*pc));
            pc->SW = am->sw[s]; pc->SH = am->sh[s]; pc->BW = am->bw[s]; pc->BH = am->bh[s];
            pc->scale = s;
            pc->bpc = e->bpc;
            pc->prevN = s ? am->bw[s - 1] * am->bh[s - 1] : 0;
            pc->egl = a->egl;
            pc->cos1sq = cos_1deg_sq;
            pc->egl_int = a->egl >= 1.0 && a->egl <= 1000.0 && a->egl == (double)(int)a->egl
                        ? (int)a->egl : -1;
            const int w = am->bw[s], h = am->bh[s];
            int dl = w * 0.1 - 0.5 - 1, dt = h * 0.1 - 0.5 - 1, dr = w - dl + 2, db = h - dt + 2;
            if (dl < 0) dl = 0;
            if (dr > w) dr = w;
            if (dt < 0) dt = 0;
            if (db > h) db = h;
            pc->dl = dl; pc->dt = dt; pc->dr = dr; pc->db = db;
            hs->left = w * 0.1 - 0.5;
            hs->top = h * 0.1 - 0.5;
            hs->right = w - hs->left;
            hs->bottom = h - hs->top;
            if (hs->left <= 0 || hs->top <= 0) {
                vmaf_log(VMAF_LOG_LEVEL_ERROR, "vulkan: the picture is too small for ADM\n");
                return -EINVAL;
            }
            pc->cl = hs->left; pc->ct = hs->top; pc->cr = hs->right; pc->cb = hs->bottom;
            pc->scol = hs->left > 1 ? hs->left : 1;
            pc->ecol = hs->right < w - 1 ? hs->right : w - 1;
            pc->srow = hs->top > 1 ? hs->top : 1;
            pc->erow = hs->bottom < h - 1 ? hs->bottom : h - 1;
            float factor1, factor2;
            csf_factors(a, s, &factor1, &factor2);
            hs->rfactor[0] = factor1;
            hs->rfactor[1] = factor1;
            hs->rfactor[2] = factor2;
            if (s == 0) {
                if (fabs(a->nvd * a->rdh - DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8 &&
                    a->csf_mode == ADM_CSF_MODE_WATSON97) {
                    hs->i_rfactor[0] = 36453;
                    hs->i_rfactor[1] = 36453;
                    hs->i_rfactor[2] = 49417;
                } else {
                    const double pow2_21 = pow(2, 21), pow2_23 = pow(2, 23);
                    hs->i_rfactor[0] = (uint16_t)(hs->rfactor[0] * pow2_21);
                    hs->i_rfactor[1] = (uint16_t)(hs->rfactor[1] * pow2_21);
                    hs->i_rfactor[2] = (uint16_t)(hs->rfactor[2] * pow2_23);
                }
                hs->shift_cub[0] = (uint32_t)ceil(log2(w) - 4);
                hs->shift_cub[1] = (uint32_t)ceil(log2(w) - 4);
                hs->shift_cub[2] = (uint32_t)ceil(log2(w) - 3);
            } else {
                const double pow2_32 = pow(2, 32);
                for (int k = 0; k < 3; k++)
                    hs->i_rfactor[k] = (uint32_t)(hs->rfactor[k] * pow2_32);
                hs->shift_cub[0] = hs->shift_cub[1] = hs->shift_cub[2] = (uint32_t)ceil(log2(w));
            }
            hs->shift_inner = (uint32_t)ceil(log2(h));
            pc->rf0 = (int)hs->i_rfactor[0];
            pc->rf1 = (int)hs->i_rfactor[1];
            pc->rf2 = (int)hs->i_rfactor[2];
            pc->shc0 = hs->shift_cub[0];
            pc->shc1 = hs->shift_cub[1];
            pc->shc2 = hs->shift_cub[2];
            hs->den_cub = s ? (int)ceil(log2(hs->right - hs->left)) : 0;
        }
    }

    /* buffers: the bands of each scale (shared), the row sums of the
     * denominator (shared) and of each unit's masking */
    for (int s = 0; s < 4 && !err; s++) {
        const size_t N = (size_t)am->bw[s] * am->bh[s];
        const size_t rows = (size_t)am->bh[s] * 3 * 8 + 16;
        err = vmaf_vk_buffer(vk, &am->acur[s], 20 * N * 4 + 16, VMAF_VK_MEM_DEVICE);
        if (!err) err = vmaf_vk_buffer(vk, &am->aden[s], rows, VMAF_VK_MEM_DEVICE);
        for (int k = 0; k < VMAF_VK_ENGINE_SLOTS && !err; k++)
            err = vmaf_vk_buffer(vk, &am->rb_den[k][s], rows, VMAF_VK_MEM_READBACK);
        for (unsigned u = 0; u < am->n_unit && !err; u++) {
            AdmUnit *a = &am->unit[u];
            err = vmaf_vk_buffer(vk, &a->cm[s], rows, VMAF_VK_MEM_DEVICE);
            for (int k = 0; k < VMAF_VK_ENGINE_SLOTS && !err; k++)
                err = vmaf_vk_buffer(vk, &a->rb_cm[k][s], rows, VMAF_VK_MEM_READBACK);
            if (a->aim && !err) {
                err = vmaf_vk_buffer(vk, &a->aimb[s], rows, VMAF_VK_MEM_DEVICE);
                for (int k = 0; k < VMAF_VK_ENGINE_SLOTS && !err; k++)
                    err = vmaf_vk_buffer(vk, &a->rb_aim[k][s], rows, VMAF_VK_MEM_READBACK);
            }
        }
    }
    if (err) return err;

    /* tmp: each unit's tile sums of each scale in a part of their own, so that
     * the scales' and the units' masking passes run side by side */
    size_t at = 0;
    for (int s = 0; s < 4; s++) {
        am->tiles_s[s] = 2 * am->bh[s] * ((am->bw[s] + 15) / 16) * 3 * 2;
        am->tile_base[s] = (int)at;
        at += (size_t)am->tiles_s[s] * am->n_unit;
    }
    const size_t dwt_tmp = (size_t)4 * am->bh[0] * W;
    err = vmaf_vk_buffer(vk, &am->atmp, (dwt_tmp > at ? dwt_tmp : at) * 4 + 16,
                         VMAF_VK_MEM_DEVICE);
    if (!err) err = vmaf_vk_buffer(vk, &am->adiv, 65537 * 4, VMAF_VK_MEM_UPLOAD);
    if (err) return err;
    {
        int32_t *d = am->adiv.map;
        memset(d, 0, 65537 * 4);
        for (int i = 1; i <= 32768; ++i) {
            const int32_t recip = (int32_t)(1073741824 / i);
            d[32768 + i] = recip;
            d[32768 - i] = 0 - recip;
        }
    }

    VmafVkBuf *raw_ref = &e->raw[0], *raw_dis = &e->raw[1];
    for (int s = 0; s < 4 && !err; s++) {
        VmafVkBuf *prev = &am->acur[s ? s - 1 : 0];
        VmafVkBuf *b[8] = { &am->acur[s], prev, raw_ref, raw_dis, &am->atmp,
                            &am->adiv, &am->unit[0].cm[s], &am->aden[s] };
        err = vmaf_vk_pipeline(vk, &am->adwt[s], "adm_dwt", b, 8);
        if (!err) err = vmaf_vk_pipeline(vk, &am->aden_p[s], "adm_den", b, 8);
        for (unsigned u = 0; u < am->n_unit && !err; u++) {
            AdmUnit *a = &am->unit[u];
            b[6] = &a->cm[s];
            err = vmaf_vk_pipeline(vk, &a->dcm[s], "adm_dcm", b, 8);
            if (!err) err = vmaf_vk_pipeline(vk, &a->cmsum[s], "adm_cmsum", b, 8);
            if (!err && a->aim) {
                b[6] = &a->aimb[s];
                err = vmaf_vk_pipeline(vk, &a->aimsum[s], "adm_cmsum", b, 8);
            }
        }
    }
    if (err) return err;

    /* adm_dcm_multi: the units of one gain limit in one pass per scale */
    const char *env = getenv("VMAF_VULKAN_ADM_MULTI");
    const bool multi = !(env && *env && atoi(env) == 0);
    if (!multi) return 0;
    for (unsigned u = 0; u < am->n_unit; u++) {
        unsigned g = 0;
        while (g < am->n_group &&
               am->unit[am->group[g].units[0]].apc[0].egl != am->unit[u].apc[0].egl)
            g++;
        if (g == am->n_group) {
            AdmGroup *n = realloc(am->group, sizeof(*n) * (am->n_group + 1));
            if (!n) return -ENOMEM;
            am->group = n;
            memset(&am->group[g], 0, sizeof(am->group[g]));
            am->group[g].units = calloc(am->n_unit, sizeof(int));
            if (!am->group[g].units) return -ENOMEM;
            am->n_group++;
        }
        am->group[g].units[am->group[g].n++] = u;
    }
    for (unsigned g = 0; g < am->n_group && !err; g++) {
        AdmGroup *gr = &am->group[g];
        if (gr->n < 2) continue; /* one unit: adm_dcm */
        for (int s = 0; s < 4 && !err; s++) {
            err = vmaf_vk_buffer(vk, &gr->tab[s], 16 * gr->n + 16, VMAF_VK_MEM_UPLOAD);
            if (err) break;
            int32_t *t = gr->tab[s].map;
            for (unsigned k = 0; k < gr->n; k++) {
                const AdmUnit *a = &am->unit[gr->units[k]];
                t[4 * k] = a->apc[s].rf0;
                t[4 * k + 1] = a->apc[s].rf1;
                t[4 * k + 2] = a->apc[s].rf2;
                t[4 * k + 3] = (a->aim ? 1 : 0) |
                               ((am->tile_base[s] + gr->units[k] * am->tiles_s[s]) << 1);
            }
            VmafVkBuf *prev = &am->acur[s ? s - 1 : 0];
            VmafVkBuf *b[9] = { &am->acur[s], prev, raw_ref, raw_dis, &am->atmp,
                                &am->adiv, &am->unit[gr->units[0]].cm[s],
                                &am->aden[s], &gr->tab[s] };
            err = vmaf_vk_pipeline(vk, &gr->p[s], "adm_dcm_multi", b, 9);
        }
    }
    return err;
}

static uint32_t groups16(int n) { return (uint32_t)((n + 15) / 16); }

static int module_record(VmafVkModule *m, VmafVkEngine *e, int slot,
                         unsigned index)
{
    (void) index;
    AdmModule *am = m->priv;
    VmafVk *vk = &e->vk;
    if (!am->n_unit) return 0;

    /* the four scales' wavelets, then the denominator and the masking passes
     * of every scale and unit side by side, one barrier, all the row sums */
    for (int s = 0; s < 4; s++) {
        const AdmPC pc = am->unit[0].apc[s];
        vmaf_vk_push(vk, &pc, sizeof(pc));
        vmaf_vk_bind(vk, &am->adwt[s]);
        vmaf_vk_dispatch(vk, groups16(am->bw[s]), groups16(am->bh[s]));
        vmaf_vk_barrier(vk);
    }
    for (int s = 0; s < 4; s++) {
        AdmPC pd = am->unit[0].apc[s];
        if (s > 0) pd.shc0 = am->unit[0].ah[s].den_cub;
        vmaf_vk_push(vk, &pd, sizeof(pd));
        vmaf_vk_bind(vk, &am->aden_p[s]);
        vmaf_vk_dispatch(vk, 1, pd.cb - pd.ct);
        for (unsigned u = 0; u < am->n_unit; u++) {
            AdmUnit *a = &am->unit[u];
            AdmPC pu = a->apc[s];
            pu.dbg = am->tile_base[s] + (int)u * am->tiles_s[s];
            const uint32_t gx = groups16(pu.ecol - pu.scol), gy = groups16(pu.erow - pu.srow);
            bool grouped = false;
            for (unsigned g = 0; g < am->n_group; g++) {
                const AdmGroup *gr = &am->group[g];
                if (gr->n < 2) continue;
                bool in = false;
                for (unsigned k = 0; k < gr->n; k++) in |= gr->units[k] == (int)u;
                if (!in) continue;
                grouped = true;
                if (gr->units[0] == (int)u) {
                    pu.dbg = (int)gr->n;
                    vmaf_vk_push(vk, &pu, sizeof(pu));
                    vmaf_vk_bind(vk, &gr->p[s]);
                    vmaf_vk_dispatch(vk, gx, gy);
                }
            }
            if (grouped) continue;
            pu.aim = a->aim ? 1 : 0;
            vmaf_vk_push(vk, &pu, sizeof(pu));
            vmaf_vk_bind(vk, &a->dcm[s]);
            vmaf_vk_dispatch(vk, gx, gy);
        }
    }
    vmaf_vk_barrier(vk);
    for (int s = 0; s < 4; s++) {
        for (unsigned u = 0; u < am->n_unit; u++) {
            AdmUnit *a = &am->unit[u];
            AdmPC pu = a->apc[s];
            pu.dbg = am->tile_base[s] + (int)u * am->tiles_s[s];
            const uint32_t rg = (uint32_t)((pu.erow - pu.srow + 63) / 64);
            pu.aim = 0;
            vmaf_vk_push(vk, &pu, sizeof(pu));
            vmaf_vk_bind(vk, &a->cmsum[s]);
            vmaf_vk_dispatch(vk, rg, 1);
            if (a->aim) {
                pu.aim = 1;
                vmaf_vk_push(vk, &pu, sizeof(pu));
                vmaf_vk_bind(vk, &a->aimsum[s]);
                vmaf_vk_dispatch(vk, rg, 1);
            }
        }
    }
    vmaf_vk_barrier(vk);
    for (int s = 0; s < 4; s++) {
        vmaf_vk_copy(vk, &am->aden[s], &am->rb_den[slot][s], 0);
        for (unsigned u = 0; u < am->n_unit; u++) {
            AdmUnit *a = &am->unit[u];
            vmaf_vk_copy(vk, &a->cm[s], &a->rb_cm[slot][s], 0);
            if (a->aim) vmaf_vk_copy(vk, &a->aimb[s], &a->rb_aim[slot][s], 0);
        }
    }
    return 0;
}

/* The numerator of one scale from the masking row sums (integer_adm.c,
 * adm_cm() / i4_adm_cm()). */
static float adm_num(const AdmUnit *a, int s, const int64_t *cm,
                     double noise_weight)
{
    const AdmHost *hs = &a->ah[s];
    int64_t accum[3] = { 0, 0, 0 };
    const int rows = a->apc[s].erow - a->apc[s].srow;
    const int64_t add_inner = (int64_t)pow(2, (hs->shift_inner - 1));
    for (int r = 0; r < rows; r++)
        for (int q = 0; q < 3; q++)
            accum[q] += (cm[r * 3 + q] + add_inner) >> hs->shift_inner;
    float f_accum[3];
    if (s == 0) {
        f_accum[0] = (float)(accum[0] / pow(2, (52 - hs->shift_cub[0] - hs->shift_inner)));
        f_accum[1] = (float)(accum[1] / pow(2, (52 - hs->shift_cub[1] - hs->shift_inner)));
        f_accum[2] = (float)(accum[2] / pow(2, (57 - hs->shift_cub[2] - hs->shift_inner)));
    } else {
        const float final_shift[3] = {
            (float)pow(2, (45 - hs->shift_cub[0] - hs->shift_inner)),
            (float)pow(2, (39 - hs->shift_cub[0] - hs->shift_inner)),
            (float)pow(2, (36 - hs->shift_cub[0] - hs->shift_inner)),
        };
        for (int q = 0; q < 3; q++)
            f_accum[q] = (float)(accum[q] / final_shift[s - 1]);
    }
    const float add = powf((hs->bottom - hs->top) * (hs->right - hs->left) * noise_weight,
                           1.0f / 3.0f);
    float parts[3];
    for (int q = 0; q < 3; q++) parts[q] = powf(f_accum[q], 1.0f / 3.0f) + add;
    return (parts[0] + parts[1]) + parts[2];
}

typedef struct AdmOut {
    double score, aim, num, den;
    double scores[8];
} AdmOut;

/* integer_compute_adm()'s sums and scores (and adm_csf_den_*()'s
 * denominator), from the slot's row sums. */
static void adm_scores(const AdmModule *am, const VmafVkEngine *e, int slot,
                       const AdmUnit *a, double noise_weight, AdmOut *out)
{
    double num = 0, den = 0, aim_num = 0;
    for (int s = 0; s < 4; s++) {
        const AdmHost *hs = &a->ah[s];
        const int64_t *dn = am->rb_den[slot][s].map;
        const float num_scale = adm_num(a, s, a->rb_cm[slot][s].map, noise_weight);
        const float aim_scale = a->aim ? adm_num(a, s, a->rb_aim[slot][s].map, 0.0) : 0.0f;
        uint64_t acc_d[3] = { 0, 0, 0 };
        int32_t shift_accum, shift_cub = 0;
        if (s == 0) {
            shift_accum = (int32_t)ceil(log2((hs->bottom - hs->top) * (hs->right - hs->left)) - 20);
            shift_accum = shift_accum > 0 ? shift_accum : 0;
        } else {
            shift_cub = (int32_t)ceil(log2(hs->right - hs->left));
            shift_accum = (int32_t)ceil(log2(hs->bottom - hs->top));
        }
        const int32_t add_shift_accum =
            (s == 0 ? shift_accum > 0 : true) ? (1 << (shift_accum - 1)) : 0;
        const int drows = hs->bottom - hs->top;
        for (int r = 0; r < drows; r++)
            for (int q = 0; q < 3; q++)
                acc_d[q] += ((uint64_t)dn[r * 3 + q] + add_shift_accum) >> shift_accum;
        double shift_csf;
        if (s == 0) {
            shift_csf = pow(2, (18 - shift_accum));
        } else {
            static const uint32_t conv[3] = { 32, 27, 23 };
            shift_csf = pow(2, (conv[s - 1] - shift_accum - shift_cub));
        }
        double csf[3];
        for (int q = 0; q < 3; q++)
            csf[q] = (double)(acc_d[q] / shift_csf) * pow(hs->rfactor[q], 3);
        const float powf_add = powf((hs->bottom - hs->top) * (hs->right - hs->left) * noise_weight,
                                    1.0f / 3.0f);
        float dparts[3];
        for (int q = 0; q < 3; q++) dparts[q] = powf(csf[q], 1.0f / 3.0f) + powf_add;
        const float den_scale = (dparts[0] + dparts[1]) + dparts[2];
        out->scores[2 * s + 0] = num_scale;
        out->scores[2 * s + 1] = den_scale;
        num += num_scale;
        den += den_scale;
        aim_num += aim_scale;
    }
    const double numden_limit = 1e-10 * (e->w * e->h) / (1920.0 * 1080.0);
    const double n = num < numden_limit ? 0 : num, d = den < numden_limit ? 0 : den;
    out->aim = 0.0;
    if (d == 0.0) {
        out->score = 1.0f;
    } else {
        out->aim = aim_num / d;
        out->score = n / d;
    }
    out->num = n;
    out->den = d;
}

static int write_view(VmafFeatureCollector *fc, const AdmVkState *s,
                      VmafDictionary *dict, const AdmOut *o, bool debug,
                      unsigned index)
{
    int err = 0;
    err |= vmaf_feature_collector_append_with_dict(fc, dict,
            "VMAF_integer_feature_adm2_score", o->score, index);
    err |= vmaf_feature_collector_append_with_dict(fc, dict,
            "VMAF_integer_feature_aim_score", o->aim, index);
    err |= vmaf_feature_collector_append_with_dict(fc, dict,
            "VMAF_integer_feature_adm3_score",
            MAX(o->score * s->adm_dlm_weight + (1 - o->aim) * (1 - s->adm_dlm_weight), s->adm_min_val),
            index);
    err |= vmaf_feature_collector_append_with_dict(fc, dict,
            "integer_adm_scale0", o->scores[0] / o->scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(fc, dict,
            "integer_adm_scale1", o->scores[2] / o->scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(fc, dict,
            "integer_adm_scale2", o->scores[4] / o->scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(fc, dict,
            "integer_adm_scale3", o->scores[6] / o->scores[7], index);
    if (!debug) return err;
    err |= vmaf_feature_collector_append_with_dict(fc, dict, "integer_adm", o->score, index);
    err |= vmaf_feature_collector_append_with_dict(fc, dict, "integer_adm_num", o->num, index);
    err |= vmaf_feature_collector_append_with_dict(fc, dict, "integer_adm_den", o->den, index);
    static const char *const names[8] = {
        "integer_adm_num_scale0", "integer_adm_den_scale0",
        "integer_adm_num_scale1", "integer_adm_den_scale1",
        "integer_adm_num_scale2", "integer_adm_den_scale2",
        "integer_adm_num_scale3", "integer_adm_den_scale3",
    };
    for (int k = 0; k < 8; k++)
        err |= vmaf_feature_collector_append_with_dict(fc, dict, names[k], o->scores[k], index);
    return err;
}

static int module_results(VmafVkModule *m, VmafVkEngine *e, int slot,
                          unsigned index, VmafFeatureCollector *fc)
{
    AdmModule *am = m->priv;
    int err = 0;
    for (unsigned v = 0; v < am->n_view; v++) {
        const AdmView *vw = &am->view[v];
        const AdmVkState *s = vw->s;
        if (!s || vw->unit < 0) continue;
        AdmOut o;
        memset(&o, 0, sizeof(o));
        adm_scores(am, e, slot, &am->unit[vw->unit], s->adm_noise_weight, &o);
        if (s->adm_skip_aim) o.aim = 0.0;
        err |= write_view(fc, s, vw->extra ? s->feature_name_dict_extra
                                            : s->feature_name_dict,
                          &o, s->debug && !vw->extra, index);
    }
    return err;
}

static void module_close(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    AdmModule *am = m->priv;
    if (!am) return;
    for (unsigned g = 0; g < am->n_group; g++) free(am->group[g].units);
    free(am->group);
    free(am->unit);
    free(am->view);
    free(am);
    m->priv = NULL;
}

static int module_create(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    AdmModule *am = calloc(1, sizeof(*am));
    if (!am) return -ENOMEM;
    m->priv = am;
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
    AdmVkState *s = fex->priv;

    if (s->adm_skip_scale0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "adm_vulkan: adm_skip_scale0 is not supported\n");
        return -EINVAL;
    }
    if (s->adm_csf_mode == ADM_CSF_MODE_BARTEN || s->adm_csf_mode < 0 ||
        s->adm_csf_mode > ADM_CSF_MODE_BARTEN_WATSON_BLEND_MAE) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "adm_vulkan: adm_csf_mode %d is not supported\n", s->adm_csf_mode);
        return -EINVAL;
    }
    const int n_vd = s->adm_norm_view_dist_extra > 0.0 ? 2 : 1;
    const double nvd[2] = { s->adm_norm_view_dist, s->adm_norm_view_dist_extra };
    /* integer_adm.c: limited by the 16 bit data pipeline */
    for (int v = 0; v < n_vd; v++) {
        if (nvd[v] * s->adm_ref_display_height <
            DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT)
            return -EINVAL;
        if (s->adm_csf_mode >= ADM_CSF_MODE_BARTEN_WATSON_BLEND &&
            barten_watson_blend_csf(0, 0, nvd[v], s->adm_ref_display_height) < 0)
            return -EINVAL;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) return -ENOMEM;
    if (n_vd == 2) {
        const double primary = s->adm_norm_view_dist;
        s->adm_norm_view_dist = s->adm_norm_view_dist_extra;
        s->feature_name_dict_extra =
            vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                    fex->options, s);
        s->adm_norm_view_dist = primary;
        if (!s->feature_name_dict_extra) return -ENOMEM;
    }

    int err = vmaf_vk_engine_get(fex->vk_ctx, pix_fmt, bpc, w, h, &s->engine);
    if (err) return err;
    s->module = vmaf_vk_engine_module(s->engine, "adm", module_create);
    if (!s->module) return -EINVAL;
    AdmModule *am = s->module->priv;
    AdmView *vw = realloc(am->view, sizeof(*vw) * (am->n_view + n_vd));
    if (!vw) return -ENOMEM;
    am->view = vw;
    for (int v = 0; v < n_vd; v++) {
        am->view[am->n_view++] = (AdmView) {
            .s = s, .nvd = nvd[v], .extra = v == 1, .unit = -1,
        };
    }
    s->joined = true;
    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    AdmVkState *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;
    return vmaf_vk_engine_extract(s->engine, ref_pic, dist_pic, index,
                                  feature_collector);
}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    AdmVkState *s = fex->priv;
    (void) feature_collector;
    int err = vmaf_vk_engine_flush(s->engine);
    return err ? err : 1;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    AdmVkState *s = fex->priv;
    if (s->engine) {
        vmaf_vk_engine_flush(s->engine);
        if (s->joined) {
            AdmModule *am = s->module->priv;
            for (unsigned v = 0; v < am->n_view; v++)
                if (am->view[v].s == s) am->view[v].s = NULL;
        }
        vmaf_vk_engine_release(fex->vk_ctx, &s->engine);
    }
    vmaf_dictionary_free(&s->feature_name_dict);
    vmaf_dictionary_free(&s->feature_name_dict_extra);
    return 0;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_adm2_score", "VMAF_integer_feature_aim_score", "VMAF_integer_feature_adm3_score",
    "integer_adm_scale0", "integer_adm_scale1", "integer_adm_scale2", "integer_adm_scale3",
    "integer_adm", "integer_adm_num", "integer_adm_den",
    "integer_adm_num_scale0", "integer_adm_den_scale0", "integer_adm_num_scale1",
    "integer_adm_den_scale1", "integer_adm_num_scale2", "integer_adm_den_scale2",
    "integer_adm_num_scale3", "integer_adm_den_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_adm_vulkan = {
    .name = "adm_vulkan",
    .init = init,
    .extract = extract,
    .flush = flush,
    .options = options,
    .close = close_fex,
    .priv_size = sizeof(AdmVkState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_VULKAN,
};
