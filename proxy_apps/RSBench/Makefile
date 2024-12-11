#===============================================================================
# User Options
#===============================================================================

COMPILER  ?= llvm
OPTIMIZE  ?= yes
DEBUG     ?= no
PROFILE   ?= no
CUDA_ARCH ?= sm_70

#===============================================================================
# Program name & source code list
#===============================================================================

program = RSBench

source = \
main.c \
io.c \
init.c \
material.c \
utils.c

obj = $(source:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=gnu99 -Wall

# Linker Flags
LDFLAGS = -lm

# Intel Compiler
ifeq ($(COMPILER),intel)
  CC = icx 
  CFLAGS += -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ 
endif

# LLVM Compiler Targeting A100 -- Change SM Level to Target Other GPUs
ifeq ($(COMPILER),llvm)
  CC = clang
  CFLAGS += -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=$(CUDA_ARCH)
  # CFLAGS += -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu
endif

# IBM XL Compiler
ifeq ($(COMPILER),ibm)
  CC = xlc_r
  CFLAGS += -qsmp=omp -qoffload
endif

# NVIDIA Compiler Targeting A100 -- Change SM Level to Target Other GPUs
ifeq ($(COMPILER),nvidia)
  CC = nvc
  CFLAGS += -mp=gpu -Minfo=mp -gpu=cc80
endif

# AOMP Targeting MI100 -- Change march to Target Other GPUs
ifeq ($(COMPILER),amd)
  CC = clang
  CFLAGS += -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  # CFLAGS += -pg
  # LDFLAGS  += -pg
  CFLAGS += -fno-omit-frame-pointer
  LDFLAGS  += -fno-omit-frame-pointer
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

all: RSBench-Weak RSBench-Strong

RSBench-Weak: $(obj) simulation_weak.o rsbench.h Makefile
	$(CC) $(CFLAGS) $(obj) simulation_weak.o -o $@ $(LDFLAGS)

RSBench-Strong: $(obj) simulation_strong.o rsbench.h Makefile
	$(CC) $(CFLAGS) $(obj) simulation_strong.o -o $@ $(LDFLAGS)

# $(program): $(obj) rsbench.h Makefile
# 	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.c rsbench.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf rsbench RSBench-Weak RSBench-Strong *.o $(obj)

edit:
	vim -p $(source) rsbench.h

run:
	./rsbench
