
CUDA_ARCH ?= sm_70
CXX ?= clang++
CXXFLAGS := -O3 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=$(CUDA_ARCH)

all: MPPBench

MPPBench: microbenchmark.cpp Makefile
	$(CXX) $(CXXFLAGS) microbenchmark.cpp -o $@

clean:
	rm -rf MPPBench