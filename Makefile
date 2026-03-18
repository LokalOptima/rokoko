MAKEFLAGS += -j$(shell nproc)
CUDA_HOME ?= /usr/local/cuda-13.1
CUTLASS   ?= third_party/cutlass/include

CXX      = g++
NVCC     = $(CUDA_HOME)/bin/nvcc
CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -I$(CUDA_HOME)/include -Isrc
NVFLAGS  = -std=c++17 -O3 -arch=native -I$(CUDA_HOME)/include -Isrc --expt-relaxed-constexpr
LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lpthread

FP16_WEIGHTS = $(HOME)/.cache/rokoko/weights_v2.koko
FP16_URL     = https://github.com/lfrati/rokoko/releases/download/v1.0.0/weights_v2.koko

.PHONY: clean rokoko.fp16
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

src/main.o: src/main.cu src/g2p.h src/normalize.h src/weights.h src/kernels.h \
            src/bundle.h src/server.h src/cpp-httplib/httplib.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

rokoko: src/main.o src/tts.cpp src/weights.cpp src/weights.h src/kernels.o \
        src/cutlass_conv.o src/cutlass_gemm.o src/cutlass_gemm_f16.o src/cutlass_conv_f16.o
	$(CXX) $(CXXFLAGS) -mavx2 -mfma \
		src/main.o src/tts.cpp src/weights.cpp \
		src/kernels.o src/cutlass_conv.o src/cutlass_gemm.o \
		src/cutlass_gemm_f16.o src/cutlass_conv_f16.o $(LDFLAGS) -o $@

rokoko.fp16: rokoko
	@if [ ! -f "$(FP16_WEIGHTS)" ]; then \
		mkdir -p "$$(dirname "$(FP16_WEIGHTS)")"; \
		echo "FP16 weights not found at $(FP16_WEIGHTS) — downloading..."; \
		curl -L -# -o "$(FP16_WEIGHTS).tmp" "$(FP16_URL)" && \
		mv "$(FP16_WEIGHTS).tmp" "$(FP16_WEIGHTS)" || \
		{ rm -f "$(FP16_WEIGHTS).tmp"; echo "Download failed"; exit 1; }; \
	fi
	@echo "FP16 weights ready: $(FP16_WEIGHTS)"
	@echo "Run: ./rokoko --weights $(FP16_WEIGHTS) \"your text\""

clean:
	rm -f rokoko src/kernels.o src/main.o src/cutlass_conv.o src/cutlass_gemm.o \
		src/cutlass_gemm_f16.o src/cutlass_conv_f16.o
