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

#define _GNU_SOURCE

#include "hpcg_cpu_detect.h"

#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef HPCG_HAVE_NEOVERSE_V2
#define HPCG_HAVE_NEOVERSE_V2 1
#endif

#ifndef HPCG_HAVE_OLYMPUS
#define HPCG_HAVE_OLYMPUS 1
#endif

#if !HPCG_HAVE_NEOVERSE_V2 && !HPCG_HAVE_OLYMPUS
#error "The HPCG dispatcher requires at least one CPU implementation"
#endif

typedef int (*hpcg_entry_point)(int, char **);

/* Dispatcher-only flag: report the detected CPU and which library would be
 * loaded, then exit without running HPCG. Reference HPCG has no such option, so
 * claiming it here is safe. Handy for a one-second smoke test on a new CPU. */
static int hpcg_detect_only_requested(int argc, char **argv)
{
    int i;

    for (i = 1; i < argc; ++i) {
        if (argv[i] != NULL && strcmp(argv[i], "--detect") == 0) {
            return 1;
        }
    }
    return 0;
}

static const char *hpcg_library_name(enum hpcg_cpu_variant variant)
{
    switch (variant) {
#if HPCG_HAVE_NEOVERSE_V2
    case HPCG_CPU_NEOVERSE_V2:
        return "libhpcg_neoverse_v2.so";
#endif
#if HPCG_HAVE_OLYMPUS
    case HPCG_CPU_OLYMPUS:
        return "libhpcg_olympus.so";
#endif
    default:
        return NULL;
    }
}

static int hpcg_library_path(const char *library, char *path,
                             size_t path_size)
{
    char executable[PATH_MAX];
    char *separator;
    ssize_t length;
    int written;

    length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length < 0 || (size_t)length >= sizeof(executable) - 1) {
        return -1;
    }
    executable[length] = '\0';

    separator = strrchr(executable, '/');
    if (separator == NULL) {
        return -1;
    }
    *separator = '\0';

    written = snprintf(path, path_size, "%s/lib/%s", executable, library);
    if (written < 0 || (size_t)written >= path_size) {
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    char reason[256];
    char library_path[PATH_MAX];
    enum hpcg_cpu_variant variant;
    const char *library;
    void *handle;
    void *symbol;
    hpcg_entry_point entry;

    variant = hpcg_detect_cpu_variant(reason, sizeof(reason));
    library = hpcg_library_name(variant);

    if (hpcg_detect_only_requested(argc, argv)) {
        printf("NVIDIA HPCG dispatch: detected %s (%s)\n",
               hpcg_cpu_variant_name(variant), reason);
        if (library != NULL) {
            printf("NVIDIA HPCG dispatch: would load %s\n", library);
            return EXIT_SUCCESS;
        }
        if (variant == HPCG_CPU_UNKNOWN) {
            printf("NVIDIA HPCG dispatch: no matching implementation (unknown CPU)\n");
        } else {
            printf("NVIDIA HPCG dispatch: %s detected but not included in this build\n",
                   hpcg_cpu_variant_name(variant));
        }
        return EXIT_FAILURE;
    }

    if (library == NULL) {
        if (variant == HPCG_CPU_UNKNOWN) {
            fprintf(stderr, "NVIDIA HPCG dispatch failed: %s\n", reason);
        } else {
            fprintf(stderr,
                    "NVIDIA HPCG dispatch failed: detected %s (%s), but "
                    "this build does not include that implementation.\n",
                    hpcg_cpu_variant_name(variant), reason);
        }
        return EXIT_FAILURE;
    }

    if (hpcg_library_path(library, library_path, sizeof(library_path)) != 0) {
        fprintf(stderr,
                "NVIDIA HPCG dispatch failed: cannot locate the executable directory\n");
        return EXIT_FAILURE;
    }

    handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        fprintf(stderr, "NVIDIA HPCG dispatch failed while loading %s: %s\n",
                library_path, dlerror());
        return EXIT_FAILURE;
    }

    (void)dlerror();
    symbol = dlsym(handle, "main");
    {
        const char *error = dlerror();
        if (error != NULL) {
            fprintf(stderr,
                    "NVIDIA HPCG dispatch failed: entry point main is unavailable in %s: %s\n",
                    library_path, error);
            return EXIT_FAILURE;
        }
    }

    if (sizeof(entry) != sizeof(symbol)) {
        fprintf(stderr,
                "NVIDIA HPCG dispatch failed: incompatible function-pointer representation\n");
        return EXIT_FAILURE;
    }
    memcpy(&entry, &symbol, sizeof(entry));

    fprintf(stderr, "NVIDIA HPCG loaded: library=%s (%s)\n",
            library_path, library);

    return entry(argc, argv);
}
