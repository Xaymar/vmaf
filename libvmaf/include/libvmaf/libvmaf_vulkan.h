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

#ifndef __VMAF_VULKAN_H__
#define __VMAF_VULKAN_H__

#include "libvmaf/libvmaf.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Vulkan compute feature extractors.
 *
 * With a VmafVulkanState imported into a VmafContext, the features of the
 * models used afterwards (vmaf_use_features_from_model()) are computed on the
 * GPU where a Vulkan feature extractor exists for them: integer ADM, VIF and
 * motion (VMAF v0.6.1), and CAMBI and SpEED chroma (VMAF v1). They give the
 * same values as the CPU feature extractors, bit for bit. Pictures are given
 * as usual, in host memory; they are copied to the GPU, and several frames are
 * in flight at once. The Vulkan loader is loaded at run time: without it (or
 * without a suitable GPU) vmaf_vulkan_state_init() fails and the CPU feature
 * extractors are used as before.
 *
 * The device needs 64 bit integers and floats in shaders (shaderInt64,
 * shaderFloat64), which desktop GPUs and llvmpipe have.
 */

typedef struct VmafVulkanState VmafVulkanState;

typedef struct VmafVulkanConfiguration {
    /**
     * The Vulkan physical device to use, in the order the loader lists them
     * (as vulkaninfo does); -1 picks one: the first discrete GPU that can be
     * used, else the first device that can.
     */
    int device_index;
} VmafVulkanConfiguration;

/**
 * Brings up a Vulkan device for the Vulkan feature extractors.
 *
 * @param vk_state The state, to be imported with vmaf_vulkan_import_state()
 *                 and released with vmaf_vulkan_state_free() once every
 *                 VmafContext that uses it is closed.
 *
 * @param cfg      Which device.
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_vulkan_state_init(VmafVulkanState **vk_state,
                           VmafVulkanConfiguration cfg);

/**
 * Makes `vmaf` use the Vulkan feature extractors for the models and features
 * registered after this call. Several VmafContexts may share one state.
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_vulkan_import_state(VmafContext *vmaf, VmafVulkanState *vk_state);

/**
 * Releases a state made by vmaf_vulkan_state_init().
 */
int vmaf_vulkan_state_free(VmafVulkanState **vk_state);

/**
 * The name of the device a state uses (as the driver reports it).
 */
const char *vmaf_vulkan_device_name(VmafVulkanState *vk_state);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_VULKAN_H__ */
