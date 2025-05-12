#===============================================================================
# User Options
#===============================================================================

COMPILER    ?= llvm
OPTIMIZE    ?= yes
DEBUG       ?= no
PROFILE     ?= no
MPI         ?= no
CUDA_ARCH   ?= sm_70

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.c \
io.c \
GridInit.c \
XSutils.c \
Materials.c

obj = $(source:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=gnu99 -Wall

# Linker Flags
LDFLAGS = -lm #-fuse-ld=lld

# Regular gcc Compiler
ifeq ($(COMPILER),gnu)
  CC = gcc
  CFLAGS += -fopenmp -flto
endif

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
  LDFLAGS  += -g
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

# MPI
ifeq ($(MPI),yes)
  CC = mpicc
  CFLAGS += -DMPI
endif

#===============================================================================
# Targets to Build
#===============================================================================

all: XSBench-naive-weak XSBench-bcast-weak XSBench-naive-strong XSBench-bcast-strong

XSBench-naive-weak: $(obj) Simulation_weak.o XSbench_header.h Makefile
	$(CC) $(CFLAGS) $(obj) Simulation_weak.o -o $@ $(LDFLAGS)

XSBench-naive-strong: $(obj) Simulation_strong.o XSbench_header.h Makefile
	$(CC) $(CFLAGS) $(obj) Simulation_strong.o -o $@ $(LDFLAGS)

XSBench-bcast-weak: $(obj) Simulation_bcast_weak.o XSbench_header.h Makefile
	$(CC) $(CFLAGS) $(obj) Simulation_bcast_weak.o -o $@ $(LDFLAGS)

XSBench-bcast-strong: $(obj) Simulation_bcast_strong.o XSbench_header.h Makefile
	$(CC) $(CFLAGS) $(obj) Simulation_bcast_strong.o -o $@ $(LDFLAGS)

%.o: %.c XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) XSBench-naive-weak XSBench-bcast-weak XSBench-naive-strong XSBench-bcast-strong *.o $(obj)

edit:
	vim -p $(source) XSbench_header.h

run:
	./$(program)
