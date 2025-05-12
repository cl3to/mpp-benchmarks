#!/bin/bash

# Build Microbenchmark (MPPBench)
# cd ./microbenchmark
# CUDA_ARCH=$1 make
# cp -f ./MPPBench ../
# make clean

# Build RSBench
cd ../proxy_apps/RSBench
CUDA_ARCH=$1 make
cp -f ./RSBench-* ../../
make clean

# Build XSBench
cd ../XSBench
CUDA_ARCH=$1 make
cp -f ./XSBench-* ../../
make clean

cd ../../