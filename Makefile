#!/usr/bin/make -f

NVIDIA_SDK = $(HOME)/Desktop/cuda/sdk-4.0.17/sdk/

NVCC_ARCHES += -gencode arch=compute_20,code=sm_20
#NVCC_ARCHES += -gencode arch=compute_20,code=sm_21

CFLAGS = -O3 -Wall
NVCC_FLAGS =  -use_fast_math $(NVCC_ARCHES) --compiler-options="$(CFLAGS) -fno-strict-aliasing"

all: gpuLucas

gpuLucas.o: gpuLucas.cu IrrBaseBalanced.cu
	nvcc -c -o $@ gpuLucas.cu -O3 -I$(NVIDIA_SDK)/C/common/inc $(NVCC_FLAGS)

gpuLucas: gpuLucas.o
	g++ -fPIC $(CFLAGS) -o $@ $^  -Wl,-O1 -Wl,--as-needed -lcudart -lcufft -lqd $(NVIDIA_SDK)/C/lib/libcutil_x86_64.a

clean:
	-rm *.o *~
	-rm gpuLucas
