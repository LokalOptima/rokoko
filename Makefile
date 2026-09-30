MAKEFLAGS += -j$(shell nproc)
CUDA_HOME ?= /usr/local/cuda-13.1
CUTLASS   ?= third_party/cutlass/include

CXX      = g++
NVCC     = $(CUDA_HOME)/bin/nvcc
CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -I$(CUDA_HOME)/include -Isrc
NVFLAGS  = -std=c++17 -O3 -arch=native -I$(CUDA_HOME)/include -Isrc --expt-relaxed-constexpr
LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lpthread

SHARED_OBJS = src/kernels.o src/cutlass_conv.o src/cutlass_gemm.o \
              src/cutlass_gemm_f16.o src/cutlass_conv_f16.o

.PHONY: clean bench bench-fp16 test-frontend
.DEFAULT_GOAL := rokoko

src/kernels.o: src/kernels.cu src/kernels.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/cutlass_conv.o: src/cutlass_conv.cu
	$(NVCC) $(NVFLAGS) -I$(CUTLASS) -c $< -o $@

src/cutlass_gemm.o: src/cutlass_gemm.cu
	$(NVCC) $(NVFLAGS) -I$(CUTLASS) -c $< -o $@

src/cutlass_gemm_f16.o: src/cutlass_gemm_f16.cu
	$(NVCC) $(NVFLAGS) -I$(CUTLASS) -c $< -o $@

src/cutlass_conv_f16.o: src/cutlass_conv_f16.cu
	$(NVCC) $(NVFLAGS) -I$(CUTLASS) -c $< -o $@

src/main.o: src/main.cu src/g2p.h src/normalize.h src/weights.h src/rokoko_common.h \
            src/kernels.h src/server.h src/cpp-httplib/httplib.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

rokoko: src/main.o src/rokoko.cpp src/weights.cpp src/weights.h $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma \
		src/main.o src/rokoko.cpp src/weights.cpp \
		$(SHARED_OBJS) $(LDFLAGS) -o $@

rokoko.fp16: src/main.o src/rokoko_f16.cpp src/weights.cpp src/weights.h $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma \
		src/main.o src/rokoko_f16.cpp src/weights.cpp \
		$(SHARED_OBJS) $(LDFLAGS) -o $@

# --- Text frontend checks (see tests/frontend/README.md) ---

FRONTEND = tests/frontend
G2P ?= $(HOME)/.cache/rokoko/g2p.bin

$(FRONTEND)/normalize_cli: $(FRONTEND)/normalize_cli.cpp src/normalize.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Isrc $< -o $@

$(FRONTEND)/g2p_check: $(FRONTEND)/g2p_check.cu src/g2p.h src/normalize.h src/cutlass_gemm.o src/kernels.o
	$(NVCC) $(NVFLAGS) $< src/cutlass_gemm.o src/kernels.o -o $@

test-frontend: $(FRONTEND)/normalize_cli $(FRONTEND)/g2p_check
	uv run $(FRONTEND)/eval_frontend.py --g2p $(G2P)

bench: rokoko
	./bench.sh ./rokoko 3 10

bench-fp16: rokoko.fp16
	./bench.sh ./rokoko.fp16 3 10

clean:
	rm -f rokoko rokoko.fp16 src/kernels.o src/main.o src/cutlass_conv.o \
		src/cutlass_gemm.o src/cutlass_gemm_f16.o src/cutlass_conv_f16.o
