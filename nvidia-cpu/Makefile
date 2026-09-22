# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL = /bin/sh

# Build the CPU-dispatched HPCG: one tuned shared object per NVIDIA Arm server
# CPU (Grace/Neoverse-V2, Vera/Olympus) plus a generic dispatcher that selects
# the right one at run time from cpu0's MIDR_EL1. Set an MPI C++ compiler and
# point NVPL_PATH at an NVPL Sparse install, then:
#
#   make auto        # GCC 16+ -> multiarch (Grace + Olympus), else Grace only
#   make multiarch   # both microarchitectures (needs GCC 16 for -mcpu=olympus)
#   make grace-only  # Grace/Neoverse-V2 only
#   make STRIP_SYMBOLS=1 ...   # strip the shipped libraries
#
# e.g.  make -j auto CXX=mpicxx DISPATCH_CC=gcc NVPL_PATH=/path/to/nvpl

# The implementations are C++ + MPI, so the "compiler" here is an MPI wrapper.
# The dispatcher is plain C and deliberately generic (armv8-a), so it loads on
# any Arm core before it has decided which tuned library to run.
CXX ?= mpicxx
DISPATCH_CC ?= gcc
HOSTCC ?= cc
STRIP ?= strip

TOPDIR := $(CURDIR)
HPCG_DIR := $(abspath $(TOPDIR)/..)
SRC_DIR := $(HPCG_DIR)/src
BIN_DIR := $(TOPDIR)/bin
LIB_DIR := $(BIN_DIR)/lib
OBJ_ROOT := $(TOPDIR)/obj

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

# NVPL Sparse provides the optimized CPU SpMV/SpSV kernels the
# USE_AARCH64 path calls. NVPL self-dispatches Grace vs Vera at run
# time, so the same library serves both per-uarch builds and -mcpu only tunes
# HPCG's own (VLA-SVE + integer) code around it. Point NVPL_PATH at the install
# (a directory with include/ and lib/libnvpl_sparse.*).
NVPL_PATH ?= /path/to/nvpl
NVPL_SPARSE_INC ?= $(NVPL_PATH)/include
NVPL_SPARSE_LIB ?= $(NVPL_PATH)/lib

# HPCG build-time options. Match the main makefile's optimized aarch64 build
# (setup/Make.CUDA_AARCH64) so the dispatched libraries are the real CPU
# benchmark -- NVPL kernels, contiguous SELL arrays -- not the reference kernels.
# Override HPCG_OPTS to customize (e.g. add -DHPCG_NO_MPI for a serial build).
HPCG_OPTS ?= -DHPCG_CUBIC_RADICAL_SEARCH -DHPCG_CONTIGUOUS_ARRAYS -DUSE_AARCH64

HPCG_COMMON_CXXFLAGS = -O3 -ffast-math -ftree-vectorize -fopenmp \
	-fPIC -ffunction-sections \
	$(HPCG_OPTS) -I$(SRC_DIR) -I$(NVPL_SPARSE_INC)

# Static C++/GCC runtime so a library built with the GCC 16 toolchain (needed
# for Olympus) does not require that toolchain's libstdc++ on LD_LIBRARY_PATH at
# run time. Keeps each .so self-contained and portable across the two CPUs.
HPCG_RUNTIME_LDFLAGS = -shared -fopenmp \
	-static-libstdc++ -static-libgcc \
	-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now

ifeq ($(STRIP_SYMBOLS),1)
HPCG_COMMON_CXXFLAGS += -g0
HPCG_STRIP_LDFLAGS = -s
else
HPCG_STRIP_LDFLAGS =
endif

NEOVERSE_V2_FLAGS = -mcpu=neoverse-v2 -mtune=neoverse-v2
OLYMPUS_FLAGS = -mcpu=olympus -mtune=olympus

DISPATCH_BIN := $(BIN_DIR)/xhpcg
DISPATCH_SOURCES := dispatch/hpcg_dispatch.c dispatch/hpcg_cpu_detect.c
DISPATCH_CFLAGS ?= -O2 -std=c11 -Wall -Wextra -march=armv8-a
DISPATCH_LDFLAGS ?= -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now
HPCG_HAVE_NEOVERSE_V2 ?= 1
HPCG_HAVE_OLYMPUS ?= 1

# Per-variant state. Only meaningful in the recursive build-impl invocation,
# where VARIANT is set on the command line.
OBJ_DIR := $(OBJ_ROOT)/$(VARIANT)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
DSO := $(LIB_DIR)/libhpcg_$(VARIANT).so

.DEFAULT_GOAL := auto

all: auto

# GCC 16+ can target Olympus (Vera); older GCC builds Grace only. Query the
# C++ compiler's backend version through the MPI wrapper so the gate reflects
# what actually compiles the kernels.
auto:
	@set -e; \
	version=`$(CXX) -dumpfullversion -dumpversion 2>/dev/null | sed -n '1p'`; \
	major=`printf '%s\n' "$$version" | sed 's/\..*//'`; \
	case "$$major" in ''|*[!0-9]*) \
		echo "ERROR: unexpected GCC version from $(CXX): $${version:-<empty>}" >&2; exit 1;; \
	esac; \
	if [ "$$major" -ge 16 ]; then \
		echo "INFO: GCC $$version detected; building Grace and Olympus implementations"; \
		$(MAKE) multiarch; \
	else \
		echo "INFO: GCC $$version detected; building Grace only (Olympus requires GCC 16 or newer)"; \
		$(MAKE) grace-only; \
	fi

multiarch: implementation-neoverse-v2 implementation-olympus
	$(MAKE) dispatcher HPCG_HAVE_NEOVERSE_V2=1 HPCG_HAVE_OLYMPUS=1

grace-only: implementation-neoverse-v2
	$(RM) $(LIB_DIR)/libhpcg_olympus.so
	$(MAKE) dispatcher HPCG_HAVE_NEOVERSE_V2=1 HPCG_HAVE_OLYMPUS=0

# VARIANT is the library token (underscored, matches the dispatcher's dlopen
# name); VARIANT_FLAGS carries the hyphenated -mcpu/-mtune the compiler wants.
implementation-neoverse-v2:
	$(MAKE) build-impl VARIANT=neoverse_v2 VARIANT_FLAGS="$(NEOVERSE_V2_FLAGS)"

implementation-olympus:
	$(MAKE) build-impl VARIANT=olympus VARIANT_FLAGS="$(OLYMPUS_FLAGS)"

build-impl: $(DSO)

# The benchmark entry stays named main: C++ leaves main unmangled (unlike any
# renamed C++ function, which the compiler would mangle and the version script
# would then fail to match). The version script exports only main and localizes
# everything else, so each library still exposes exactly one public symbol.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) -c $(HPCG_COMMON_CXXFLAGS) $(VARIANT_FLAGS) $< -o $@

$(DSO): $(OBJS) | $(LIB_DIR)
	$(CXX) $(VARIANT_FLAGS) $(OBJS) -o $@ \
		$(HPCG_RUNTIME_LDFLAGS) $(HPCG_STRIP_LDFLAGS) \
		-L$(NVPL_SPARSE_LIB) -lnvpl_sparse \
		-Wl,-soname,libhpcg_$(VARIANT).so
	@[ "$(STRIP_SYMBOLS)" != "1" ] || $(STRIP) --strip-all $(DSO)

dispatcher: $(DISPATCH_SOURCES) dispatch/hpcg_cpu_detect.h | $(BIN_DIR)
	$(DISPATCH_CC) $(DISPATCH_CFLAGS) \
		-DHPCG_HAVE_NEOVERSE_V2=$(HPCG_HAVE_NEOVERSE_V2) \
		-DHPCG_HAVE_OLYMPUS=$(HPCG_HAVE_OLYMPUS) \
		$(DISPATCH_LDFLAGS) $(HPCG_STRIP_LDFLAGS) \
		-I dispatch -o $(DISPATCH_BIN) $(DISPATCH_SOURCES) -ldl
	@[ "$(STRIP_SYMBOLS)" != "1" ] || $(STRIP) --strip-all $(DISPATCH_BIN)

$(OBJ_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@

# Compile-only sanity for the dispatch/detection layer on any Linux host.
check-dispatch-native:
	@dispatch_tmp=`mktemp /tmp/hpcg-cpu-dispatch-test.XXXXXX`; \
	trap 'rm -f "$$dispatch_tmp"' EXIT; \
	$(HOSTCC) -std=c11 -Wall -Wextra -Werror \
		-DHPCG_HAVE_NEOVERSE_V2=1 -DHPCG_HAVE_OLYMPUS=0 \
		-I dispatch -o "$$dispatch_tmp" $(DISPATCH_SOURCES) -ldl; \
	$(HOSTCC) -std=c11 -Wall -Wextra -Werror \
		-DHPCG_HAVE_NEOVERSE_V2=1 -DHPCG_HAVE_OLYMPUS=1 \
		-I dispatch -o "$$dispatch_tmp" $(DISPATCH_SOURCES) -ldl; \
	echo "dispatch layer compiles clean (grace-only and multiarch)"

clean:
	$(RM) -r $(OBJ_ROOT)
	$(RM) $(DISPATCH_BIN) $(BIN_DIR)/xhpcg \
		$(LIB_DIR)/libhpcg_neoverse_v2.so $(LIB_DIR)/libhpcg_olympus.so

.PHONY: all auto multiarch grace-only dispatcher \
	implementation-neoverse-v2 implementation-olympus build-impl \
	check-dispatch-native clean
