/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
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

#ifndef FEATURE_SPEED_H_
#define FEATURE_SPEED_H_

#include <stddef.h>

double compute_cov_kernel_scalar(const float *data_x, const float *data_y,
                                  size_t stride_px, size_t height,
                                  size_t width, double mean_x, double mean_y);

/* speed_extract_score() after filter_and_downscale(): for planes decimated
 * by 16, blurred and with their blur subtracted elsewhere (the Vulkan
 * speed_chroma), stored with speed_tail_float_stride() bytes per row. The
 * options as speed_init() takes them (prescale_method is kept, not copied);
 * returns non-zero if a covariance matrix was singular, as
 * speed_extract_score(). */
typedef struct SpeedTail SpeedTail;
int speed_tail_init(SpeedTail **tail, int w, int h, double kernelscale,
                    double prescale, char *prescale_method, double sigma_nn,
                    double nn_floor, int weight_var_mode);
size_t speed_tail_float_stride(const SpeedTail *t);
int speed_tail_score(SpeedTail *t, const float *ref, const float *dis,
                     float *score);
void speed_tail_free(SpeedTail **tail);

#endif /* FEATURE_SPEED_H_ */
