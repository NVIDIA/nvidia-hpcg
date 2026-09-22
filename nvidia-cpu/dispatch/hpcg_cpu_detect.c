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

#include "hpcg_cpu_detect.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>

#define HPCG_MIDR_IMPLEMENTER(midr) (((midr) >> 24) & UINT64_C(0xff))
#define HPCG_MIDR_PART(midr)        (((midr) >> 4) & UINT64_C(0x0fff))

#define HPCG_ARM_IMPLEMENTER      UINT64_C(0x41)
#define HPCG_NVIDIA_IMPLEMENTER   UINT64_C(0x4e)
#define HPCG_NEOVERSE_V2_PART     UINT64_C(0xd4f)
#define HPCG_OLYMPUS_PART         UINT64_C(0x010)

static void hpcg_set_reason(char *reason, size_t reason_size,
                            const char *format, ...)
{
    va_list args;

    if (reason == NULL || reason_size == 0) {
        return;
    }

    va_start(args, format);
    (void)vsnprintf(reason, reason_size, format, args);
    va_end(args);
}

/* Read cpu0's MIDR_EL1 identification register from Linux sysfs. The register
   and its sysfs entry exist only on AArch64 (ARMv8/v9); on any other host
   architecture there is nothing to read, so return 0 (unknown). */
static uint64_t hpcg_get_midr_el1(void)
{
#if defined(__aarch64__)
    static const char midr_path[] =
        "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1";
    FILE *file = fopen(midr_path, "r");

    if (file != NULL) {
        uint64_t value;
        const int fields_read = fscanf(file, "%" SCNx64, &value);
        (void)fclose(file);
        if (fields_read == 1) {
            return value;
        }
    }
#endif
    return 0;
}

enum hpcg_cpu_variant hpcg_cpu_variant_from_midr(uint64_t midr)
{
    const uint64_t implementer = HPCG_MIDR_IMPLEMENTER(midr);
    const uint64_t part = HPCG_MIDR_PART(midr);

    if (implementer == HPCG_ARM_IMPLEMENTER &&
        part == HPCG_NEOVERSE_V2_PART) {
        return HPCG_CPU_NEOVERSE_V2;
    }
    if (implementer == HPCG_NVIDIA_IMPLEMENTER &&
        part == HPCG_OLYMPUS_PART) {
        return HPCG_CPU_OLYMPUS;
    }
    return HPCG_CPU_UNKNOWN;
}

const char *hpcg_cpu_variant_name(enum hpcg_cpu_variant variant)
{
    switch (variant) {
    case HPCG_CPU_NEOVERSE_V2:
        return "neoverse-v2";
    case HPCG_CPU_OLYMPUS:
        return "olympus";
    default:
        return "unknown";
    }
}

enum hpcg_cpu_variant hpcg_detect_cpu_variant(char *reason,
                                              size_t reason_size)
{
    const uint64_t midr = hpcg_get_midr_el1();
    const enum hpcg_cpu_variant variant = hpcg_cpu_variant_from_midr(midr);

    if (midr == 0) {
        hpcg_set_reason(reason, reason_size,
                        "cannot read cpu0 MIDR_EL1 from Linux sysfs");
    } else if (variant == HPCG_CPU_UNKNOWN) {
        hpcg_set_reason(reason, reason_size,
                        "unsupported cpu0 MIDR 0x%08" PRIx64,
                        midr & UINT64_C(0xffffffff));
    } else {
        hpcg_set_reason(reason, reason_size,
                        "CPU 0 MIDR 0x%08" PRIx64,
                        midr & UINT64_C(0xffffffff));
    }
    return variant;
}
