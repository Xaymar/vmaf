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

/* speed_chroma (speed.c) on Vulkan. On the GPU: each chroma plane as floats,
 * prescaled, filtered and decimated by 16 (speed_dec), then the 3x3 blur and
 * the plane minus its blur (speed_blur_v, speed_blur_h): filter_and_downscale().
 * The rest, est_params() and get_speed_score() on planes of 1/256th the size,
 * is serial work: by default it runs on the CPU, with speed.c's own code
 * (speed_tail_score()), on each slot's thread. VMAF_VULKAN_SPEED_GPU=1 does
 * it on the GPU as well (speed_mean, speed_center, speed_cov, speed_small,
 * speed_blocks, speed_score), which gives the same values.
 *
 * Feature extractors whose options differ only in speed_max_val share a
 * unit. */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "mem.h"
#include "speed.h"
#include "vif_tools.h"
#include "vulkan/engine.h"

#define DEFAULT_SPEED_SIGMA_NN (0.29)
#define DEFAULT_SPEED_MAX_VAL (1000.0)
#define DEFAULT_SPEED_NN_FLOOR (0.0)
#define DEFAULT_SPEED_KERNELSCALE (1.0)
#define DEFAULT_SPEED_PRESCALE (1.0)
#define DEFAULT_SPEED_PRESCALE_METHOD ("nearest")
#define N_SPEED_PAIRS (25 * 26 / 2)

typedef struct SpeedChromaVkState {
    double speed_chroma_kernelscale;
    double speed_chroma_prescale;
    char *speed_chroma_prescale_method;
    double speed_chroma_sigma_nn;
    double speed_chroma_nn_floor;
    double speed_chroma_max_val;
    int speed_weight_var_mode;

    VmafDictionary *feature_name_dict;
    VmafVkEngine *engine;
    VmafVkModule *module;
    bool joined;
    int unit;
} SpeedChromaVkState;

/* speed.c's options_chroma */
static const VmafOption options[] = {
    {
        .name = "speed_kernelscale",
        .help = "scaling factor for the gaussian kernel (2.0 means "
                "multiplying the standard deviation by 2 and enlarge "
                "the kernel size accordingly",
        .offset = offsetof(SpeedChromaVkState, speed_chroma_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_KERNELSCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ks",
    },
    {
        .name = "speed_prescale",
        .help = "scaling factor for the frame (2.0 means "
                "making the image twice as large on each dimension)",
        .offset = offsetof(SpeedChromaVkState, speed_chroma_prescale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_PRESCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ps",
    },
    {
        .name = "speed_prescale_method",
        .help = "scaling method for the frame, supported options: "
                 "[nearest, bilinear, bicubic, lanczos4]",
        .offset = offsetof(SpeedChromaVkState, speed_chroma_prescale_method),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_SPEED_PRESCALE_METHOD,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "psm",
    },
    {
        .name = "speed_sigma_nn",
        .help = "standard deviation of neural noise",
        .offset = offsetof(SpeedChromaVkState, speed_chroma_sigma_nn),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_SIGMA_NN,
        .min = 0.1,
        .max = 2.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "snn",
    },
    {
        .name = "speed_nn_floor",
        .help = "neural noise floor, expressed in percentage of sigma_nn",
        .offset = offsetof(SpeedChromaVkState, speed_chroma_nn_floor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_NN_FLOOR,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "nnf",
    },
    {
        .name = "speed_max_val",
        .help = "maximum value allowed; "
                "larger values will be clipped to this value",
        .offset = offsetof(SpeedChromaVkState, speed_chroma_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mxv",
    },
    {
        .name = "speed_weight_var_mode",
        .help = "different approaches to perform variance-absed weighting",
        .offset = offsetof(SpeedChromaVkState, speed_weight_var_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.d = 0,
        .min = 0,
        .max = 6,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "wvm",
    },
    { 0 }
};

/* --- the module: the GPU work, shared by every speed_chroma feature extractor --- */

typedef struct SpeedPC {
    int32_t W, H, SW, SH, DW, DH, bpc, fw, scaled, out_off;
    float scaler;
} SpeedPC;

typedef struct SpeedStatPC {
    int32_t DW, DH, fw, subw, subh, nbh, nbv, nb, mode;
    float sigma_nn, nn_floor, base_entropy, one;
    double log2_2pie;
} SpeedStatPC;

typedef struct SpeedUnit {
    /* the key */
    double kernelscale, prescale, sigma_nn, nn_floor;
    char *method;
    int mode;

    SpeedPC pc;
    SpeedStatPC st;
    VmafVkBuf tab, filt, dec, tmp, prep, filt3, covm, cen, small_buf, blk, fin;
    VmafVkPipe p_dec[4], p_blur_v, p_blur_h, p_mean, p_center, p_cov, p_small,
               p_blocks, p_score;
    /* what comes back: the planes minus their blur (CPU tail), or the scores
     * (GPU) */
    VmafVkBuf readback[VMAF_VK_ENGINE_SLOTS];
    /* CPU tail: speed.c's state and the planes as it takes them, per slot */
    SpeedTail *tail[VMAF_VK_ENGINE_SLOTS];
    float *plane[VMAF_VK_ENGINE_SLOTS][2];
    float score[VMAF_VK_ENGINE_SLOTS][4]; ///< U score, U failed, V score, V failed
} SpeedUnit;

typedef struct SpeedModule {
    SpeedChromaVkState **fex;
    unsigned n_fex;
    SpeedUnit *unit;
    unsigned n_unit;
    bool gpu; ///< all of SpEED on the GPU
} SpeedModule;

static bool same_unit(const SpeedUnit *u, const SpeedChromaVkState *s)
{
    return u->kernelscale == s->speed_chroma_kernelscale &&
           u->prescale == s->speed_chroma_prescale &&
           !strcmp(u->method, s->speed_chroma_prescale_method) &&
           (float)u->sigma_nn == (float)s->speed_chroma_sigma_nn &&
           (float)u->nn_floor == (float)s->speed_chroma_nn_floor &&
           u->mode == s->speed_weight_var_mode;
}

static float mirrorf(float i, float left, float right)
{
    return i < left ? -i : i > right ? 2 * right - i : i;
}

static int unit_setup(SpeedUnit *u, SpeedModule *sm, VmafVkEngine *e)
{
    VmafVk *vk = &e->vk;
    /* the chroma planes, as speed.c's init_chroma() */
    const int cw = e->cw, ch = e->ch;
    if (!cw || !ch) return -EINVAL;
    float scaler;
    switch (e->bpc) {
    case 8: scaler = 1.0f; break;
    case 10: scaler = 4.0f; break;
    case 12: scaler = 16.0f; break;
    case 16: scaler = 256.0f; break;
    default:
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "speed_chroma_vulkan: %u bits per sample are not supported\n", e->bpc);
        return -EINVAL;
    }
    const int SW = (int)(cw * u->prescale + 0.5), SH = (int)(ch * u->prescale + 0.5);
    const int DW = SW >> 4, DH = SH >> 4;
    const int bs = 5;
    const int tw = DW / bs * bs, th = DH / bs * bs;
    if (!tw || !th) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "speed_chroma_vulkan: the chroma planes are too small\n");
        return -EINVAL;
    }
    const bool scaled = !(SW == cw && SH == ch) && !(fabs(u->prescale - 1.0) < 1.0e-3);

    int err = vmaf_vk_buffer(vk, &u->tab, (VkDeviceSize)(2 * SW + 2 * SH) * 4 * 2 + 16,
                             VMAF_VK_MEM_UPLOAD);
    if (err) return err;
    {
        int32_t *t = u->tab.map;
        float *fx = (float *)(t + 2 * SW + 2 * SH);
        float *fy = fx + SW;
        if (!strcmp(u->method, "bilinear")) {
            /* vif_scale_frame_bilinear_precompute_columns_s() and its rows */
            const float ratio_x = (float)cw / SW;
            for (int x = 0; x < SW; x++) {
                const float xx = (x + 0.5) * ratio_x - 0.5;
                t[x] = (int)mirrorf(floor(xx), 0, cw - 1);
                t[SW + x] = (int)mirrorf(ceil(xx), 0, cw - 1);
                fx[x] = xx - t[x];
            }
            const float ratio_y = (float)ch / SH;
            for (int y = 0; y < SH; y++) {
                const float yy = (y + 0.5) * ratio_y - 0.5;
                t[2 * SW + y] = (int)mirrorf(floor(yy), 0, ch - 1);
                t[2 * SW + SH + y] = (int)mirrorf(ceil(yy), 0, ch - 1);
                fy[y] = yy - t[2 * SW + y];
            }
        } else { /* vif_scale_frame_nearest_s() */
            const float ratio_x = (float)cw / SW, ratio_y = (float)ch / SH;
            for (int x = 0; x < SW; x++) {
                t[x] = t[SW + x] = (int)(x * ratio_x);
                fx[x] = 0;
            }
            for (int y = 0; y < SH; y++) {
                t[2 * SW + y] = t[2 * SW + SH + y] = (int)(y * ratio_y);
                fy[y] = 0;
            }
        }
    }
    err = vmaf_vk_buffer(vk, &u->filt, 128 * 4, VMAF_VK_MEM_UPLOAD);
    if (!err) err = vmaf_vk_buffer(vk, &u->filt3, 128 * 4, VMAF_VK_MEM_UPLOAD);
    if (err) return err;
    memset(u->filt.map, 0, 128 * 4);
    memset(u->filt3.map, 0, 128 * 4);
    speed_get_antialias_filter(u->filt.map, 4, (float)u->kernelscale);
    vif_get_filter(u->filt3.map, 4, (float)u->kernelscale);

    u->pc = (SpeedPC) {
        .W = cw, .H = ch, .SW = SW, .SH = SH, .DW = DW, .DH = DH,
        .bpc = e->bpc, .fw = vif_get_filter_size(1, (float)u->kernelscale),
        .scaled = scaled, .scaler = scaler,
    };
    SpeedStatPC *st = &u->st;
    st->DW = DW;
    st->DH = DH;
    st->fw = vif_get_filter_size(4, (float)u->kernelscale);
    st->subw = tw - bs + 1;
    st->subh = th - bs + 1;
    st->nbh = tw / bs;
    st->nbv = th / bs;
    st->nb = st->nbh * st->nbv;
    st->mode = u->mode;
    st->sigma_nn = (float)u->sigma_nn;
    st->nn_floor = (float)u->nn_floor;
    st->one = 1.0f;
    /* get_speed_score()'s base entropy and est_params()'s log2(2 pi e), with
     * the CPU's log2 */
    st->log2_2pie = log2(2 * M_PI * M_E);
    st->base_entropy = bs * bs * (log2((1 + st->nn_floor) * st->sigma_nn) +
                                  log2(2 * M_PI * M_E));

    const VkDeviceSize n = (VkDeviceSize)DW * DH;
    err = vmaf_vk_buffer(vk, &u->dec, 4 * n * 4 + 16, VMAF_VK_MEM_DEVICE);
    if (!err) err = vmaf_vk_buffer(vk, &u->tmp, 4 * n * 4 + 16, VMAF_VK_MEM_DEVICE);
    if (!err) err = vmaf_vk_buffer(vk, &u->prep, 4 * n * 4 + 16, VMAF_VK_MEM_DEVICE);
    if (sm->gpu) {
        if (!err) err = vmaf_vk_buffer(vk, &u->covm, (4 * 625 + 4 * 25) * 4 + 16, VMAF_VK_MEM_DEVICE);
        if (!err)
            err = vmaf_vk_buffer(vk, &u->cen,
                                 (VkDeviceSize)4 * 25 * st->subw * st->subh * 8 + 16,
                                 VMAF_VK_MEM_DEVICE);
        if (!err) err = vmaf_vk_buffer(vk, &u->small_buf, 4 * 1280 * 4 + 16, VMAF_VK_MEM_DEVICE);
        if (!err)
            err = vmaf_vk_buffer(vk, &u->blk, (VkDeviceSize)4 * 2 * st->nb * 4 + 16,
                                 VMAF_VK_MEM_DEVICE);
        if (!err) err = vmaf_vk_buffer(vk, &u->fin, 4 * 4 + 16, VMAF_VK_MEM_DEVICE);
    }
    for (int k = 0; k < VMAF_VK_ENGINE_SLOTS && !err; k++)
        err = vmaf_vk_buffer(vk, &u->readback[k], sm->gpu ? 4 * 4 + 16 : 4 * n * 4 + 16,
                             VMAF_VK_MEM_READBACK);
    if (err) return err;

    for (int p = 0; p < 4 && !err; p++) {
        VmafVkBuf *b[4] = { &e->raw[2 + p], &u->tab, &u->filt, &u->dec };
        err = vmaf_vk_pipeline(vk, &u->p_dec[p], "speed_dec", b, 4);
    }
    VmafVkBuf *sb[9] = { &u->dec, &u->tmp, &u->prep, &u->filt3,
                         sm->gpu ? &u->covm : &u->dec, sm->gpu ? &u->small_buf : &u->dec,
                         sm->gpu ? &u->blk : &u->dec, sm->gpu ? &u->fin : &u->dec,
                         sm->gpu ? &u->cen : &u->dec };
    if (!err) err = vmaf_vk_pipeline(vk, &u->p_blur_v, "speed_blur_v", sb, 9);
    if (!err) err = vmaf_vk_pipeline(vk, &u->p_blur_h, "speed_blur_h", sb, 9);
    if (sm->gpu) {
        if (!err) err = vmaf_vk_pipeline(vk, &u->p_mean, "speed_mean", sb, 9);
        if (!err) err = vmaf_vk_pipeline(vk, &u->p_center, "speed_center", sb, 9);
        if (!err) err = vmaf_vk_pipeline(vk, &u->p_cov, "speed_cov", sb, 9);
        if (!err) err = vmaf_vk_pipeline(vk, &u->p_small, "speed_small", sb, 9);
        if (!err) err = vmaf_vk_pipeline(vk, &u->p_blocks, "speed_blocks", sb, 9);
        if (!err) err = vmaf_vk_pipeline(vk, &u->p_score, "speed_score", sb, 9);
        return err;
    }

    for (int k = 0; k < VMAF_VK_ENGINE_SLOTS && !err; k++) {
        err = speed_tail_init(&u->tail[k], cw, ch, u->kernelscale, u->prescale,
                              u->method, u->sigma_nn, u->nn_floor, u->mode);
        if (err) break;
        const size_t stride = speed_tail_float_stride(u->tail[k]);
        for (int q = 0; q < 2 && !err; q++) {
            u->plane[k][q] = aligned_malloc(stride * DH, 32);
            if (!u->plane[k][q]) err = -ENOMEM;
            else memset(u->plane[k][q], 0, stride * DH);
        }
    }
    return err;
}

static int module_setup(VmafVkModule *m, VmafVkEngine *e)
{
    SpeedModule *sm = m->priv;
    const char *env = getenv("VMAF_VULKAN_SPEED_GPU");
    sm->gpu = env && atoi(env) > 0;
    for (unsigned i = 0; i < sm->n_fex; i++) {
        SpeedChromaVkState *s = sm->fex[i];
        if (!s) continue;
        unsigned u = 0;
        while (u < sm->n_unit && !same_unit(&sm->unit[u], s)) u++;
        s->unit = u;
        if (u < sm->n_unit) continue;
        SpeedUnit *n = realloc(sm->unit, sizeof(*n) * (sm->n_unit + 1));
        if (!n) return -ENOMEM;
        sm->unit = n;
        SpeedUnit *nu = &sm->unit[sm->n_unit++];
        memset(nu, 0, sizeof(*nu));
        nu->kernelscale = s->speed_chroma_kernelscale;
        nu->prescale = s->speed_chroma_prescale;
        nu->method = s->speed_chroma_prescale_method;
        nu->sigma_nn = s->speed_chroma_sigma_nn;
        nu->nn_floor = s->speed_chroma_nn_floor;
        nu->mode = s->speed_weight_var_mode;
    }
    int err = 0;
    for (unsigned u = 0; u < sm->n_unit && !err; u++)
        err = unit_setup(&sm->unit[u], sm, e);
    return err;
}

/* every unit's pass `pipe`, side by side */
static void step(VmafVk *vk, SpeedModule *sm, size_t pipe, int what, uint32_t gy)
{
    vmaf_vk_barrier(vk);
    for (unsigned k = 0; k < sm->n_unit; k++) {
        SpeedUnit *u = &sm->unit[k];
        uint32_t gx = 1;
        switch (what) {
        case 0: gx = (u->st.DW * u->st.DH + 63) / 64; break; /* pixels */
        case 1: gx = 1; break;
        case 2: gx = (u->st.subw * u->st.subh + 63) / 64; break;
        case 3: gx = (N_SPEED_PAIRS + 31) / 32; break;
        case 4: gx = 2; break; /* eigenvalues and QR side by side */
        case 5: gx = (u->st.nb + 63) / 64; break;
        }
        vmaf_vk_push(vk, &u->st, sizeof(u->st));
        vmaf_vk_bind(vk, (const VmafVkPipe *)((const char *)u + pipe));
        vmaf_vk_dispatch(vk, gx, gy);
    }
}

static int module_record(VmafVkModule *m, VmafVkEngine *e, int slot,
                         unsigned index)
{
    (void) index;
    SpeedModule *sm = m->priv;
    VmafVk *vk = &e->vk;
    if (!sm->n_unit) return 0;
    for (unsigned k = 0; k < sm->n_unit; k++) {
        SpeedUnit *u = &sm->unit[k];
        for (int p = 0; p < 4; p++) {
            SpeedPC pc = u->pc;
            pc.out_off = p * u->pc.DW * u->pc.DH;
            vmaf_vk_push(vk, &pc, sizeof(pc));
            vmaf_vk_bind(vk, &u->p_dec[p]);
            vmaf_vk_dispatch(vk, (u->pc.DW + 7) / 8, (u->pc.DH + 7) / 8);
        }
    }
    step(vk, sm, offsetof(SpeedUnit, p_blur_v), 0, 4);
    step(vk, sm, offsetof(SpeedUnit, p_blur_h), 0, 4);
    if (sm->gpu) {
        step(vk, sm, offsetof(SpeedUnit, p_mean), 1, 4);
        step(vk, sm, offsetof(SpeedUnit, p_center), 2, 4);
        step(vk, sm, offsetof(SpeedUnit, p_cov), 3, 4);
        step(vk, sm, offsetof(SpeedUnit, p_small), 4, 4);
        step(vk, sm, offsetof(SpeedUnit, p_blocks), 5, 4);
        step(vk, sm, offsetof(SpeedUnit, p_score), 1, 2);
    }
    vmaf_vk_barrier(vk);
    for (unsigned k = 0; k < sm->n_unit; k++) {
        SpeedUnit *u = &sm->unit[k];
        vmaf_vk_copy(vk, sm->gpu ? &u->fin : &u->prep, &u->readback[slot], 0);
    }
    return 0;
}

/* speed.c's speed_extract_score() after filter_and_downscale(), for each
 * chroma plane: U score, U failed, V score, V failed */
static void unit_cpu_scores(SpeedUnit *u, int slot)
{
    const int DW = u->pc.DW, DH = u->pc.DH;
    const size_t n = (size_t)DW * DH;
    const size_t stride = speed_tail_float_stride(u->tail[slot]) / sizeof(float);
    const float *prep = u->readback[slot].map;
    for (int q = 0; q < 2; q++) {
        for (int r = 0; r < 2; r++) { /* reference, then distorted */
            const float *src = prep + (size_t)(q + 2 * r) * n;
            float *dst = u->plane[slot][r];
            for (int i = 0; i < DH; i++)
                memcpy(dst + i * stride, src + (size_t)i * DW, DW * sizeof(float));
        }
        float score;
        const int err = speed_tail_score(u->tail[slot], u->plane[slot][0],
                                         u->plane[slot][1], &score);
        u->score[slot][2 * q] = score;
        u->score[slot][2 * q + 1] = err ? 1.0f : 0.0f;
    }
}

static int module_results(VmafVkModule *m, VmafVkEngine *e, int slot,
                          unsigned index, VmafFeatureCollector *fc)
{
    (void) e;
    SpeedModule *sm = m->priv;
    for (unsigned k = 0; k < sm->n_unit; k++) {
        SpeedUnit *u = &sm->unit[k];
        if (sm->gpu) memcpy(u->score[slot], u->readback[slot].map, sizeof(u->score[slot]));
        else unit_cpu_scores(u, slot);
    }
    int err = 0;
    for (unsigned i = 0; i < sm->n_fex; i++) {
        SpeedChromaVkState *s = sm->fex[i];
        if (!s) continue;
        const float *fin = sm->unit[s->unit].score[slot];
        const float score_u = fin[0], score_v = fin[2];
        const bool err_u = fin[1] != 0.0f, err_v = fin[3] != 0.0f;
        float score_uv;
        if (err_u && !err_v) score_uv = score_v;
        else if (err_v && !err_u) score_uv = score_u;
        else score_uv = (score_u + score_v) / 2.0;
        const double mx = s->speed_chroma_max_val;
        err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                "Speed_chroma_feature_speed_chroma_u_score",
                score_u < mx ? score_u : mx, index);
        err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                "Speed_chroma_feature_speed_chroma_v_score",
                score_v < mx ? score_v : mx, index);
        err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                "Speed_chroma_feature_speed_chroma_uv_score",
                score_uv < mx ? score_uv : mx, index);
    }
    return err;
}

static void module_close(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    SpeedModule *sm = m->priv;
    if (!sm) return;
    for (unsigned k = 0; k < sm->n_unit; k++) {
        for (int s = 0; s < VMAF_VK_ENGINE_SLOTS; s++) {
            speed_tail_free(&sm->unit[k].tail[s]);
            for (int q = 0; q < 2; q++)
                if (sm->unit[k].plane[s][q]) aligned_free(sm->unit[k].plane[s][q]);
        }
    }
    free(sm->unit);
    free(sm->fex);
    free(sm);
    m->priv = NULL;
}

static int module_create(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    SpeedModule *sm = calloc(1, sizeof(*sm));
    if (!sm) return -ENOMEM;
    m->priv = sm;
    m->chroma = true;
    m->setup = module_setup;
    m->record = module_record;
    m->results = module_results;
    m->close = module_close;
    return 0;
}

/* --- the feature extractor --- */

/* What the GPU does not compute; the CPU's speed_chroma does those. */
static bool usable(VmafFeatureExtractor *fex)
{
    SpeedChromaVkState *s = fex->priv;
    const char *method = s->speed_chroma_prescale_method;
    if (!method || (strcmp(method, "nearest") && strcmp(method, "bilinear")))
        return false;
    const float ks = (float)s->speed_chroma_kernelscale;
    if (!vif_validate_kernelscale(ks)) return false;
    return vif_get_filter_size(1, ks) <= 64 && vif_get_filter_size(4, ks) <= 128 &&
           s->speed_weight_var_mode >= 0 && s->speed_weight_var_mode <= 6;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    SpeedChromaVkState *s = fex->priv;
    if (pix_fmt == VMAF_PIX_FMT_UNKNOWN || pix_fmt == VMAF_PIX_FMT_YUV400P)
        return -EINVAL;
    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) return -ENOMEM;

    int err = vmaf_vk_engine_get(fex->vk_ctx, pix_fmt, bpc, w, h, &s->engine);
    if (err) return err;
    s->module = vmaf_vk_engine_module(s->engine, "speed", module_create);
    if (!s->module) return -EINVAL;
    SpeedModule *sm = s->module->priv;
    SpeedChromaVkState **f = realloc(sm->fex, sizeof(*f) * (sm->n_fex + 1));
    if (!f) return -ENOMEM;
    sm->fex = f;
    sm->fex[sm->n_fex++] = s;
    s->joined = true;
    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    SpeedChromaVkState *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;
    return vmaf_vk_engine_extract(s->engine, ref_pic, dist_pic, index,
                                  feature_collector);
}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    (void) feature_collector;
    SpeedChromaVkState *s = fex->priv;
    int err = vmaf_vk_engine_flush(s->engine);
    return err ? err : 1;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    SpeedChromaVkState *s = fex->priv;
    if (s->engine) {
        vmaf_vk_engine_flush(s->engine);
        if (s->joined) {
            SpeedModule *sm = s->module->priv;
            for (unsigned i = 0; i < sm->n_fex; i++)
                if (sm->fex[i] == s) sm->fex[i] = NULL;
        }
        vmaf_vk_engine_release(fex->vk_ctx, &s->engine);
    }
    return vmaf_dictionary_free(&s->feature_name_dict);
}

static const char *provided_features[] = {
    "Speed_chroma_feature_speed_chroma_u_score",
    "Speed_chroma_feature_speed_chroma_v_score",
    "Speed_chroma_feature_speed_chroma_uv_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_speed_chroma_vulkan = {
    .name = "speed_chroma_vulkan",
    .options = options,
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .priv_size = sizeof(SpeedChromaVkState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_VULKAN,
    .vulkan_usable = usable,
};
