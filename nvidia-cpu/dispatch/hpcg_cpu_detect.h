/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HPCG_CPU_DETECT_H
#define HPCG_CPU_DETECT_H

#include <stddef.h>
#include <stdint.h>

enum hpcg_cpu_variant
{
    HPCG_CPU_UNKNOWN = 0,
    HPCG_CPU_NEOVERSE_V2,
    HPCG_CPU_OLYMPUS
};

enum hpcg_cpu_variant hpcg_cpu_variant_from_midr(uint64_t midr);
enum hpcg_cpu_variant hpcg_detect_cpu_variant(char *reason,
                                              size_t reason_size);
const char *hpcg_cpu_variant_name(enum hpcg_cpu_variant variant);

#endif
