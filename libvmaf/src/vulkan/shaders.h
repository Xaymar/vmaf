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

#ifndef __VMAF_SRC_VULKAN_SHADERS_H__
#define __VMAF_SRC_VULKAN_SHADERS_H__

#include <stddef.h>
#include <stdint.h>

/* The compute shaders (the .comp files in src/vulkan/shaders), compiled to SPIR-V at build
 * time and embedded by embed_spirv.py. The table ends with a NULL name. */
typedef struct VmafVulkanShader {
    const char *name;
    const uint32_t *code;
    size_t size; ///< bytes
} VmafVulkanShader;

extern const VmafVulkanShader vmaf_vulkan_shaders[];

#endif /* __VMAF_SRC_VULKAN_SHADERS_H__ */
