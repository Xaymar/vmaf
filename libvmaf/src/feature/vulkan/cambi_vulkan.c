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

/* cambi.c on Vulkan, for the distorted frame (full_ref and heatmaps_path are
 * left to the CPU's cambi): the conversion to 10 bits at the encoded size,
 * anti-dithering, the spatial mask, the decimation and mode filter of each
 * scale and the c values on the GPU; each workgroup also counts the c values
 * that are not 0 and adds them up. The pooling is cambi.c's: when the top k
 * values of a scale take in every value that is not 0, their mean comes from
 * those sums, else from the c values themselves (quick_select, as cambi.c).
 *
 * Feature extractors whose options change the GPU work (window, contrast,
 * thresholds, EOTF, encoded size and depth, speed-up) each get a unit of
 * their own; those that differ in topk or cambi_max_val share one. */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cambi.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "luminance_tools.h"
#include "vulkan/engine.h"

#define DEFAULT_CAMBI_TOPK_POOLING (0.6)
#define DEFAULT_CAMBI_WINDOW_SIZE (65)
#define DEFAULT_CAMBI_TVI (0.019)
#define DEFAULT_CAMBI_VLT (0.0)
#define DEFAULT_CAMBI_MAX_LOG_CONTRAST (2)
#define DEFAULT_CAMBI_FULL_REF_FLAG (false)
#define DEFAULT_CAMBI_EOTF ("bt1886")
#define DEFAULT_CAMBI_HIGH_RES_SPEEDUP (0)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p (1920 * 1080)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p (2560 * 1440)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p (3840 * 2160)
#define CAMBI_MIN_WIDTH_HEIGHT (216)
#define CAMBI_4K_WIDTH (3840)
#define CAMBI_4K_HEIGHT (2160)
#define DEFAULT_CAMBI_MAX_VAL (1000.0)
#define NUM_SCALES 5
#define MASK_FILTER_SIZE 7

static const int g_scale_weights[NUM_SCALES] = { 16, 8, 4, 2, 1 };
static const int g_contrast_weights[32] = { 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8 };

typedef struct CambiVkState {
    unsigned enc_width;
    unsigned enc_height;
    unsigned enc_bitdepth;
    unsigned src_width;
    unsigned src_height;
    uint16_t window_size;
    double topk;
    double cambi_topk;
    double tvi_threshold;
    double cambi_max_val;
    double cambi_vis_lum_threshold;
    uint16_t max_log_contrast;
    char *heatmaps_path;
    char *eotf;
    char *cambi_eotf;
    bool full_ref;
    int cambi_high_res_speedup;

    VmafDictionary *feature_name_dict;
    VmafVkEngine *engine;
    VmafVkModule *module;
    bool joined;
    int unit; ///< of the module, once set up
} CambiVkState;

/* cambi.c's options */
static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(CambiVkState, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiVkState, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 144,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encw",
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(CambiVkState, enc_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 144,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ench",
    },
    {
        .name = "enc_bitdepth",
        .help = "Encoding bitdepth",
        .offset = offsetof(CambiVkState, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 6,
        .max = 16,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encbd",
    },
    {
        .name = "src_width",
        .help = "Source width. Only used when full_ref=true.",
        .offset = offsetof(CambiVkState, src_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 320,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "srcw",
    },
    {
        .name = "src_height",
        .help = "Source height. Only used when full_ref=true.",
        .offset = offsetof(CambiVkState, src_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 200,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "srch",
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI: 65 corresponds to ~1 degree at 4k",
        .offset = offsetof(CambiVkState, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation, must be 0 < topk <= 1.0",
        .offset = offsetof(CambiVkState, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for the spatial pooling computation, must be 0 < cambi_topk <= 1.0",
        .offset = offsetof(CambiVkState, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibilty threshold for luminance ΔL < tvi_threshold*L_mean",
        .offset = offsetof(CambiVkState, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TVI,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which we assume any banding is not visible",
        .offset = offsetof(CambiVkState, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_VLT,
        .min = 0.0,
        .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum contrast in log luma level (2^max_log_contrast) at 10-bits, "
                "e.g., 2 is equivalent to 4 luma levels at 10-bit and 1 luma level at 8-bit. "
                "From 0 to 5: default 2 is recommended for banding from compression.",
        .offset = offsetof(CambiVkState, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
    },
    {
        .name = "heatmaps_path",
        .help = "Path where heatmaps will be dumped.",
        .offset = offsetof(CambiVkState, heatmaps_path),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = NULL,
    },
    {
        .name = "full_ref",
        .help = "If true, CAMBI will be run in full-reference mode and will be computed on both the reference and distorted inputs",
        .offset = offsetof(CambiVkState, full_ref),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = DEFAULT_CAMBI_FULL_REF_FLAG,
    },
    {
        .name = "eotf",
        .help = "Determines the EOTF used to compute the visibility thresholds. Possible values: ['bt1886', 'pq']. Default: 'bt1886'",
        .offset = offsetof(CambiVkState, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help = "Determines the EOTF used to compute the visibility thresholds. Possible values: ['bt1886', 'pq']. Default: 'bt1886'. If both eotf and cambi_eotf are set, cambi_eotf takes precedence.",
        .offset = offsetof(CambiVkState, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ceot",
    },
    {
        .name = "cambi_high_res_speedup",
        .help = "Speed up the processing by downsampling post spatial mask for resolutions >= 1080p. "
                "Min speed-up resolution possible values: [1080, 1440, 2160, 0]. Default: 0 (not applied)"
                "Note some loss of accuracy is expected with this speedup.",
        .offset = offsetof(CambiVkState, cambi_high_res_speedup),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_HIGH_RES_SPEEDUP,
        .min = 0,
        .max = CAMBI_4K_HEIGHT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "hrs",
    },
    { 0 }
};

/* --- the module: the GPU work, shared by every CAMBI feature extractor --- */

typedef struct CambiPC {
    int32_t W, H, OW, OH, bpc, anti_dither, mask_index, pad, nd, vlt_luma,
            band_base, band_size, lanes;
} CambiPC;

/* The work of feature extractors that agree on everything but topk and
 * cambi_max_val. */
typedef struct CambiUnit {
    /* the key */
    unsigned ew, eh, enc_bitdepth;
    uint16_t window_option;
    int speedup_option;
    double tvi, vlt;
    unsigned mlc;
    char eotf[16];

    /* the geometry */
    bool speedup, anti_dither, front, dmode;
    int window, pad;
    int sw[NUM_SCALES], sh[NUM_SCALES];
    uint32_t groups[NUM_SCALES];
    CambiPC pc;

    VmafVkBuf idx, full, dith, mask0, simg[NUM_SCALES], smask[NUM_SCALES], tmp, tab;
    VmafVkBuf *img[NUM_SCALES], *mask[NUM_SCALES];
    /* c values and the workgroups' sums: written by the shaders into each
     * slot's own host buffers */
    VmafVkBuf cval[VMAF_VK_ENGINE_SLOTS][NUM_SCALES];
    VmafVkBuf part[VMAF_VK_ENGINE_SLOTS][NUM_SCALES];
    VmafVkPipe p_front, p_pre, p_dither, p_mask;
    VmafVkPipe p_dec[NUM_SCALES], p_dmode[NUM_SCALES], p_mode[NUM_SCALES];
    VmafVkPipe p_cval[VMAF_VK_ENGINE_SLOTS][NUM_SCALES];
    float *sel; ///< quick_select's copy of a scale's c values
} CambiUnit;

typedef struct CambiModule {
    CambiVkState **fex;
    unsigned n_fex;
    CambiUnit *unit;
    unsigned n_unit;
    double *score[VMAF_VK_ENGINE_SLOTS]; ///< per feature extractor, before the clip
} CambiModule;

static const char *effective_eotf(const CambiVkState *s)
{
    return strcmp(s->cambi_eotf, DEFAULT_CAMBI_EOTF) ? s->cambi_eotf : s->eotf;
}

static double effective_topk(const CambiVkState *s)
{
    return s->topk != DEFAULT_CAMBI_TOPK_POOLING ? s->topk : s->cambi_topk;
}

static void adjust_window_size(uint16_t *window_size, unsigned input_width,
                               unsigned input_height, bool speedup)
{
    *window_size = ((*window_size * (input_width + input_height)) / 375) >> 4;
    if (speedup) *window_size = (*window_size + 1) >> 1;
    *window_size |= 1;
}

static uint16_t get_mask_index(unsigned input_width, unsigned input_height,
                               uint16_t filter_size)
{
    const uint32_t shifted_wh = (input_width >> 6) * (input_height >> 6);
    uint16_t lg = 0;
    if (shifted_wh) {
        uint32_t tmp = shifted_wh - 1;
        while (tmp > 0) { tmp >>= 1; lg++; }
    }
    return (filter_size * filter_size + 3 * (lg - 11) - 1) >> 1;
}

static bool same_unit(const CambiUnit *u, const CambiVkState *s)
{
    return u->ew == s->enc_width && u->eh == s->enc_height &&
           u->enc_bitdepth == s->enc_bitdepth &&
           u->window_option == s->window_size &&
           u->speedup_option == s->cambi_high_res_speedup &&
           u->tvi == s->tvi_threshold && u->vlt == s->cambi_vis_lum_threshold &&
           u->mlc == s->max_log_contrast && !strcmp(u->eotf, effective_eotf(s));
}

static int env_flag(const char *name, int dflt)
{
    const char *e = getenv(name);
    return e && *e ? atoi(e) > 0 : dflt;
}

static int unit_setup(CambiUnit *u, VmafVkEngine *e)
{
    VmafVk *vk = &e->vk;
    const unsigned ew = u->ew, eh = u->eh;
    const bool amd = e->s->vendor_id == 0x1002;
    int err = 0;

    uint16_t ws = u->window_option;
    adjust_window_size(&ws, ew, eh, u->speedup);
    u->window = ws;
    u->pad = ws >> 1;
    int w = ew, h = eh;
    for (int s = 0; s < NUM_SCALES; s++) {
        if (s > 0 || u->speedup) { w = (w + 1) >> 1; h = (h + 1) >> 1; }
        u->sw[s] = w;
        u->sh[s] = h;
        u->groups[s] = ((w + 15) / 16) * ((h + 15) / 16);
        if (w < u->pad + 1 || h < u->pad + 1) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "cambi_vulkan: the frame is too small for the window\n");
            return -EINVAL;
        }
    }
    if (u->window * u->window >= CAMBI_RECIPROCAL_LUT_SIZE) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "cambi_vulkan: window_size %d too large for reciprocal LUT\n",
                 u->window);
        return -EINVAL;
    }
    u->anti_dither = u->enc_bitdepth < 10;

    /* cambi.c's nearest neighbour positions (decimate_generic_*_and_convert_to_10b) */
    err = vmaf_vk_buffer(vk, &u->idx, (VkDeviceSize)(ew + eh) * 4 + 16, VMAF_VK_MEM_UPLOAD);
    if (err) return err;
    {
        int32_t *idx = u->idx.map;
        if (ew == e->w && eh == e->h) {
            for (unsigned j = 0; j < ew; j++) idx[j] = j;
            for (unsigned i = 0; i < eh; i++) idx[ew + i] = i;
        } else {
            const float ratio_x = (float)e->w / ew, ratio_y = (float)e->h / eh;
            float x = ratio_x / 2 - 0.5, y = ratio_y / 2 - 0.5;
            for (unsigned j = 0; j < ew; j++) { idx[j] = (int)(x + 0.5); x += ratio_x; }
            for (unsigned i = 0; i < eh; i++) { idx[ew + i] = (int)(y + 0.5); y += ratio_y; }
        }
    }
    const VkDeviceSize full = (VkDeviceSize)ew * eh * 4 + 16;
    err = vmaf_vk_buffer(vk, &u->full, full, VMAF_VK_MEM_DEVICE);
    if (!err && u->anti_dither) err = vmaf_vk_buffer(vk, &u->dith, full, VMAF_VK_MEM_DEVICE);
    if (!err) err = vmaf_vk_buffer(vk, &u->mask0, full, VMAF_VK_MEM_DEVICE);
    VmafVkBuf *src = u->anti_dither ? &u->dith : &u->full;
    for (int s = 0; s < NUM_SCALES && !err; s++) {
        const VkDeviceSize n = (VkDeviceSize)u->sw[s] * u->sh[s] * 4 + 16;
        if (s == 0 && !u->speedup) {
            u->img[0] = src;
            u->mask[0] = &u->mask0;
        } else {
            err = vmaf_vk_buffer(vk, &u->simg[s], n, VMAF_VK_MEM_DEVICE);
            if (!err) err = vmaf_vk_buffer(vk, &u->smask[s], n, VMAF_VK_MEM_DEVICE);
            u->img[s] = &u->simg[s];
            u->mask[s] = &u->smask[s];
        }
        for (int k = 0; k < VMAF_VK_ENGINE_SLOTS && !err; k++) {
            err = vmaf_vk_buffer(vk, &u->cval[k][s], n, VMAF_VK_MEM_READBACK);
            if (!err)
                err = vmaf_vk_buffer(vk, &u->part[k][s],
                                     (VkDeviceSize)u->groups[s] * 16 + 16,
                                     VMAF_VK_MEM_READBACK);
        }
    }
    if (!err)
        err = vmaf_vk_buffer(vk, &u->tmp, (VkDeviceSize)u->sw[0] * u->sh[0] * 4 + 16,
                             VMAF_VK_MEM_DEVICE);
    /* tvi_for_diff[32], diff weights[32], then the reciprocal table */
    if (!err)
        err = vmaf_vk_buffer(vk, &u->tab, (64 + CAMBI_RECIPROCAL_LUT_SIZE) * 4,
                             VMAF_VK_MEM_UPLOAD);
    if (err) return err;
    u->sel = malloc(sizeof(float) * u->sw[0] * u->sh[0]);
    if (!u->sel) return -ENOMEM;

    const int nd = 1 << u->mlc;
    VmafLumaRange luma_range;
    err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err) return err;
    VmafEOTF eotf;
    err = vmaf_luminance_init_eotf(&eotf, u->eotf);
    if (err) return err;
    int32_t *t = u->tab.map;
    memset(t, 0, 64 * 4);
    for (int d = 0; d < nd; d++) {
        t[d] = get_tvi_for_diff(d + 1, u->tvi, 10, luma_range, eotf) + nd;
        t[32 + d] = g_contrast_weights[d];
    }
    memcpy(t + 64, reciprocal_lut, CAMBI_RECIPROCAL_LUT_SIZE * 4);

    CambiPC *pc = &u->pc;
    pc->nd = nd;
    pc->vlt_luma = get_vlt_luma(u->vlt, luma_range, eotf);
    const int v_lo = pc->vlt_luma - 3 * nd + 1;
    pc->band_base = v_lo > 0 ? v_lo : 0;
    pc->band_size = (uint16_t)(t[nd - 1] + 1 - pc->band_base);
    pc->mask_index = get_mask_index(ew, eh, MASK_FILTER_SIZE);
    pc->pad = u->pad;
    pc->bpc = e->bpc;
    pc->anti_dither = u->anti_dither;
    /* cambi_cvalue's 8 bit row counts: 25% faster on an RTX 4070 Laptop, 12%
     * slower on a Radeon 780M (2026-11-18) */
    pc->lanes = env_flag("VMAF_VULKAN_CAMBI_LANES", !amd);
    /* cambi_cvalue_cols, column counts for tiles whose values lie close
     * together: CAMBI 30% cheaper on an RTX 4080 SUPER, the c values 1.9x
     * slower on an RDNA 2 iGPU (2026-11-18). cambi_cvalue_p32: half windows of
     * 17 .. 32 (65 at 4K) from tiles in shared memory. */
    const bool cols = env_flag("VMAF_VULKAN_CAMBI_COLS", !amd);
    const bool p32 = u->pad > 16 && u->pad <= 32 && env_flag("VMAF_VULKAN_CAMBI_P32", 1);
    const char *cval_shader = p32 ? "cambi_cvalue_p32" : cols ? "cambi_cvalue_cols" : "cambi_cvalue";
    /* cambi_front: conversion, mask and scale 0's decimation in one pass (not
     * with anti-dithering); cambi_dmode: a scale's decimation and mode filter
     * in one pass. VMAF_VULKAN_CAMBI_SPLIT=1 runs them as passes of their own. */
    const bool split = env_flag("VMAF_VULKAN_CAMBI_SPLIT", 0);
    u->front = !u->anti_dither && !split;
    u->dmode = !split;

    VmafVkBuf *raw_dis = &e->raw[1];
    if (u->front) {
        VmafVkBuf *b[4] = { raw_dis, &u->idx, u->speedup ? u->img[0] : &u->full,
                            u->speedup ? u->mask[0] : &u->mask0 };
        err = vmaf_vk_pipeline(vk, &u->p_front, "cambi_front", b, 4);
    } else {
        VmafVkBuf *b[3] = { raw_dis, &u->idx, &u->full };
        err = vmaf_vk_pipeline(vk, &u->p_pre, "cambi_pre", b, 3);
        if (!err && u->anti_dither) {
            VmafVkBuf *bd[2] = { &u->full, &u->dith };
            err = vmaf_vk_pipeline(vk, &u->p_dither, "cambi_dither", bd, 2);
        }
        if (!err) {
            VmafVkBuf *bm[2] = { src, &u->mask0 };
            err = vmaf_vk_pipeline(vk, &u->p_mask, "cambi_mask", bm, 2);
        }
    }
    for (int s = 0; s < NUM_SCALES && !err; s++) {
        if (s > 0 || u->speedup) {
            VmafVkBuf *b[4] = { s == 0 ? src : u->img[s - 1],
                                s == 0 ? &u->mask0 : u->mask[s - 1],
                                u->img[s], u->mask[s] };
            err = vmaf_vk_pipeline(vk, &u->p_dec[s], "cambi_decimate", b, 4);
            if (!err && u->dmode)
                err = vmaf_vk_pipeline(vk, &u->p_dmode[s], "cambi_dmode", b, 4);
        }
        if (!err) {
            VmafVkBuf *b[2] = { u->img[s], &u->tmp };
            err = vmaf_vk_pipeline(vk, &u->p_mode[s], "cambi_mode", b, 2);
        }
        for (int k = 0; k < VMAF_VK_ENGINE_SLOTS && !err; k++) {
            VmafVkBuf *b[5] = { u->img[s], u->mask[s], &u->cval[k][s], &u->tab,
                                &u->part[k][s] };
            err = vmaf_vk_pipeline(vk, &u->p_cval[k][s], cval_shader, b, 5);
        }
    }
    return err;
}

static int module_setup(VmafVkModule *m, VmafVkEngine *e)
{
    CambiModule *cm = m->priv;
    for (unsigned i = 0; i < cm->n_fex; i++) {
        CambiVkState *s = cm->fex[i];
        if (!s) continue;
        unsigned u = 0;
        while (u < cm->n_unit && !same_unit(&cm->unit[u], s)) u++;
        s->unit = u;
        if (u < cm->n_unit) continue;
        CambiUnit *n = realloc(cm->unit, sizeof(*n) * (cm->n_unit + 1));
        if (!n) return -ENOMEM;
        cm->unit = n;
        CambiUnit *nu = &cm->unit[cm->n_unit++];
        memset(nu, 0, sizeof(*nu));
        nu->ew = s->enc_width;
        nu->eh = s->enc_height;
        nu->enc_bitdepth = s->enc_bitdepth;
        nu->window_option = s->window_size;
        nu->speedup_option = s->cambi_high_res_speedup;
        nu->speedup = s->cambi_high_res_speedup != 0;
        nu->tvi = s->tvi_threshold;
        nu->vlt = s->cambi_vis_lum_threshold;
        nu->mlc = s->max_log_contrast;
        snprintf(nu->eotf, sizeof(nu->eotf), "%s", effective_eotf(s));
    }
    for (int k = 0; k < VMAF_VK_ENGINE_SLOTS; k++) {
        cm->score[k] = calloc(cm->n_fex ? cm->n_fex : 1, sizeof(double));
        if (!cm->score[k]) return -ENOMEM;
    }
    int err = 0;
    for (unsigned u = 0; u < cm->n_unit && !err; u++)
        err = unit_setup(&cm->unit[u], e);
    return err;
}

static void unit_record(CambiUnit *u, VmafVkEngine *e, int slot)
{
    VmafVk *vk = &e->vk;
    const int ew = u->ew, eh = u->eh;
    CambiPC pc = u->pc;
    pc.W = e->w;
    pc.H = e->h;
    pc.OW = ew;
    pc.OH = eh;
    const uint32_t gx = (ew + 15) / 16, gy = (eh + 15) / 16;
    if (u->front) {
        CambiPC pf = pc;
        pf.pad = u->speedup ? u->sw[0] : 0;
        pf.nd = u->speedup ? u->sh[0] : 0;
        vmaf_vk_push(vk, &pf, sizeof(pf));
        vmaf_vk_bind(vk, &u->p_front);
        vmaf_vk_dispatch(vk, gx, gy);
        vmaf_vk_barrier(vk);
    } else {
        vmaf_vk_push(vk, &pc, sizeof(pc));
        vmaf_vk_bind(vk, &u->p_pre);
        vmaf_vk_dispatch(vk, gx, gy);
        vmaf_vk_barrier(vk);
        pc.W = ew;
        pc.H = eh;
        vmaf_vk_push(vk, &pc, sizeof(pc));
        if (u->anti_dither) {
            vmaf_vk_bind(vk, &u->p_dither);
            vmaf_vk_dispatch(vk, gx, gy);
            vmaf_vk_barrier(vk);
        }
        vmaf_vk_bind(vk, &u->p_mask);
        vmaf_vk_dispatch(vk, gx, gy);
        vmaf_vk_barrier(vk);
    }
    int pw = ew, ph = eh;
    for (int s = 0; s < NUM_SCALES; s++) {
        const int w = u->sw[s], h = u->sh[s];
        const uint32_t sx = (w + 15) / 16, sy = (h + 15) / 16;
        const bool dec = s > 0 || (u->speedup && !u->front);
        if (dec && u->dmode) {
            pc.W = pw; pc.H = ph; pc.OW = w; pc.OH = h;
            vmaf_vk_push(vk, &pc, sizeof(pc));
            vmaf_vk_bind(vk, &u->p_dmode[s]);
            vmaf_vk_dispatch(vk, sx, sy);
            vmaf_vk_barrier(vk);
        } else {
            if (dec) {
                pc.W = pw; pc.H = ph; pc.OW = w; pc.OH = h;
                vmaf_vk_push(vk, &pc, sizeof(pc));
                vmaf_vk_bind(vk, &u->p_dec[s]);
                vmaf_vk_dispatch(vk, sx, sy);
                vmaf_vk_barrier(vk);
            }
            pc.W = w; pc.H = h; pc.OW = w; pc.OH = h;
            pc.anti_dither = 0; /* cambi_mode: the pass along the rows */
            vmaf_vk_push(vk, &pc, sizeof(pc));
            vmaf_vk_bind(vk, &u->p_mode[s]);
            vmaf_vk_dispatch(vk, sx, sy);
            vmaf_vk_barrier(vk);
            pc.anti_dither = 1; /* ... and along the columns */
            vmaf_vk_push(vk, &pc, sizeof(pc));
            vmaf_vk_dispatch(vk, sx, sy);
            vmaf_vk_barrier(vk);
            pc.anti_dither = u->pc.anti_dither;
        }
        pw = w;
        ph = h;
    }
    /* The c values of the scales need nothing from each other: their passes
     * go one after another with no barrier between them, so the GPU runs them
     * side by side (each pass of the small scales is mostly a workgroup's
     * latency). */
    for (int s = 0; s < NUM_SCALES; s++) {
        const int w = u->sw[s], h = u->sh[s];
        pc.W = w; pc.H = h; pc.OW = w; pc.OH = h;
        pc.anti_dither = u->pc.anti_dither;
        vmaf_vk_push(vk, &pc, sizeof(pc));
        vmaf_vk_bind(vk, &u->p_cval[slot][s]);
        vmaf_vk_dispatch(vk, (w + 15) / 16, (h + 15) / 16);
    }
}

static int module_record(VmafVkModule *m, VmafVkEngine *e, int slot,
                         unsigned index)
{
    (void) index;
    CambiModule *cm = m->priv;
    for (unsigned u = 0; u < cm->n_unit; u++)
        unit_record(&cm->unit[u], e, slot);
    return 0;
}

/* cambi.c's quick_select() */
#define SWAP_FLOATS(x, y) { float temp = x; x = y; y = temp; }
static void quick_select(float *arr, int n, int k)
{
    if (n == k) return;
    int left = 0;
    int right = n - 1;
    while (left < right) {
        float pivot = arr[k];
        int i = left;
        int j = right;
        do {
            while (arr[i] > pivot) i++;
            while (arr[j] < pivot) j--;
            if (i <= j) {
                SWAP_FLOATS(arr[i], arr[j]);
                i++;
                j--;
            }
        } while (i <= j);
        if (j < k) left = i;
        if (k < i) right = j;
    }
}

/* cambi.c's cambi_score() from the c values: the mean of the largest topk of
 * each scale, weighted by scale (before the clip to cambi_max_val). */
static double unit_score(CambiUnit *u, int slot, double topk)
{
    double score = 0.;
    for (int s = 0; s < NUM_SCALES; s++) {
        const int n = u->sw[s] * u->sh[s];
        int k = (int)(topk * n);
        k = k < 1 ? 1 : (k > n ? n : k);
        const double *p = u->part[slot][s].map;
        double nonzero = 0., sum = 0.;
        for (uint32_t g = 0; g < u->groups[s]; g++) {
            nonzero += p[2 * g];
            sum += p[2 * g + 1];
        }
        double v;
        if (nonzero <= k) {
            /* the top k: every value that is not 0, and zeros */
            v = sum / k;
        } else {
            memcpy(u->sel, u->cval[slot][s].map, sizeof(float) * n);
            quick_select(u->sel, n, k);
            double t = 0.;
            for (int i = 0; i < k; i++) t += u->sel[i];
            v = t / k;
        }
        score += v * g_scale_weights[s];
    }
    const int pixels = (2 * (u->window >> 1) + 1) * (2 * (u->window >> 1) + 1);
    return score / pixels;
}

static int module_results(VmafVkModule *m, VmafVkEngine *e, int slot,
                          unsigned index, VmafFeatureCollector *fc)
{
    (void) e;
    CambiModule *cm = m->priv;
    int err = 0;
    for (unsigned i = 0; i < cm->n_fex; i++) {
        CambiVkState *s = cm->fex[i];
        if (!s) continue;
        /* worked out once per unit and topk */
        const double topk = effective_topk(s);
        double score = -1.;
        for (unsigned j = 0; j < i; j++) {
            const CambiVkState *o = cm->fex[j];
            if (o && o->unit == s->unit && effective_topk(o) == topk) {
                score = cm->score[slot][j];
                break;
            }
        }
        if (score < 0.) score = unit_score(&cm->unit[s->unit], slot, topk);
        cm->score[slot][i] = score;
        err |= vmaf_feature_collector_append_with_dict(fc, s->feature_name_dict,
                "Cambi_feature_cambi_score",
                score < s->cambi_max_val ? score : s->cambi_max_val, index);
    }
    return err;
}

static void module_close(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    CambiModule *cm = m->priv;
    if (!cm) return;
    for (unsigned u = 0; u < cm->n_unit; u++) free(cm->unit[u].sel);
    for (int k = 0; k < VMAF_VK_ENGINE_SLOTS; k++) free(cm->score[k]);
    free(cm->unit);
    free(cm->fex);
    free(cm);
    m->priv = NULL;
}

static int module_create(VmafVkModule *m, VmafVkEngine *e)
{
    (void) e;
    CambiModule *cm = calloc(1, sizeof(*cm));
    if (!cm) return -ENOMEM;
    m->priv = cm;
    m->setup = module_setup;
    m->record = module_record;
    m->results = module_results;
    m->close = module_close;
    return 0;
}

/* --- the feature extractor --- */

/* What the GPU does not compute; the CPU's cambi does those. */
static bool usable(VmafFeatureExtractor *fex)
{
    CambiVkState *s = fex->priv;
    const char *eotf = effective_eotf(s);
    return !s->full_ref && !s->heatmaps_path && s->max_log_contrast <= 3 &&
           (!strcmp(eotf, "bt1886") || !strcmp(eotf, "pq"));
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    CambiVkState *s = fex->priv;
    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) return -ENOMEM;

    /* as cambi.c's init() */
    if (s->enc_bitdepth == 0) s->enc_bitdepth = bpc;
    if (s->enc_width == 0 || s->enc_height == 0) {
        s->enc_width = w;
        s->enc_height = h;
    }
    if (s->enc_height > h || s->enc_width > w) {
        s->enc_width = w;
        s->enc_height = h;
    }
    if (s->enc_width < CAMBI_MIN_WIDTH_HEIGHT && s->enc_height < CAMBI_MIN_WIDTH_HEIGHT)
        return -EINVAL;
    const unsigned enc_pix = s->enc_width * s->enc_height;
    switch (s->cambi_high_res_speedup) {
    case 1080:
        if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p) s->cambi_high_res_speedup = 0;
        break;
    case 1440:
        if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p) s->cambi_high_res_speedup = 0;
        break;
    case 2160:
        if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p) s->cambi_high_res_speedup = 0;
        break;
    default:
        s->cambi_high_res_speedup = 0;
    }

    int err = vmaf_vk_engine_get(fex->vk_ctx, pix_fmt, bpc, w, h, &s->engine);
    if (err) return err;
    s->module = vmaf_vk_engine_module(s->engine, "cambi", module_create);
    if (!s->module) return -EINVAL;
    CambiModule *cm = s->module->priv;
    CambiVkState **f = realloc(cm->fex, sizeof(*f) * (cm->n_fex + 1));
    if (!f) return -ENOMEM;
    cm->fex = f;
    cm->fex[cm->n_fex++] = s;
    s->joined = true;
    return 0;
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    CambiVkState *s = fex->priv;
    (void) ref_pic_90;
    (void) dist_pic_90;
    return vmaf_vk_engine_extract(s->engine, ref_pic, dist_pic, index,
                                  feature_collector);
}

static int flush(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    (void) feature_collector;
    CambiVkState *s = fex->priv;
    int err = vmaf_vk_engine_flush(s->engine);
    return err ? err : 1;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    CambiVkState *s = fex->priv;
    if (s->engine) {
        vmaf_vk_engine_flush(s->engine);
        if (s->joined) {
            CambiModule *cm = s->module->priv;
            for (unsigned i = 0; i < cm->n_fex; i++)
                if (cm->fex[i] == s) cm->fex[i] = NULL;
        }
        vmaf_vk_engine_release(fex->vk_ctx, &s->engine);
    }
    return vmaf_dictionary_free(&s->feature_name_dict);
}

static const char *provided_features[] = {
    "Cambi_feature_cambi_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_cambi_vulkan = {
    .name = "cambi_vulkan",
    .options = options,
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close_fex,
    .priv_size = sizeof(CambiVkState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_TEMPORAL | VMAF_FEATURE_EXTRACTOR_VULKAN,
    .vulkan_usable = usable,
};
