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

.PHONY: clean bench bench-fp16 test-frontend test test-cpu
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

src/main.o: src/main.cu src/rokoko.h src/phonemes.h src/request_json.h src/artifact_format.h src/model_schema.h src/g2p.h src/normalize.h src/weights.h src/rokoko_common.h src/audio.h \
            src/kernels.h src/server.h src/cpp-httplib/httplib.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

rokoko: src/main.o src/rokoko.cpp src/weights.cpp src/weights.h src/rokoko_common.h src/audio.h src/artifact_format.h src/model_schema.h $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma \
		src/main.o src/rokoko.cpp src/weights.cpp \
		$(SHARED_OBJS) $(LDFLAGS) -o $@

rokoko.fp16: src/main.o src/rokoko_f16.cpp src/weights.cpp src/weights.h src/rokoko_common.h src/audio.h src/artifact_format.h src/model_schema.h $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma \
		src/main.o src/rokoko_f16.cpp src/weights.cpp \
		$(SHARED_OBJS) $(LDFLAGS) -o $@

# --- Text frontend checks (see tests/frontend/README.md) ---

FRONTEND = tests/frontend
FRONTEND_PYTHON ?= uv run
G2P ?= $(HOME)/.cache/rokoko/g2p.bin

$(FRONTEND)/normalize_cli: $(FRONTEND)/normalize_cli.cpp src/normalize.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Isrc $< -o $@

$(FRONTEND)/g2p_check: $(FRONTEND)/g2p_check.cu src/g2p.h src/normalize.h src/cutlass_gemm.o src/kernels.o
	$(NVCC) $(NVFLAGS) $< src/cutlass_gemm.o src/kernels.o -o $@

test-frontend: $(FRONTEND)/normalize_cli $(FRONTEND)/g2p_check
	$(FRONTEND_PYTHON) $(FRONTEND)/eval_frontend.py --g2p $(G2P)

bench: rokoko rokoko.fp16
	python3 tests/bench.py --models $(MODELS)

bench-fp16: rokoko.fp16
	python3 tests/bench.py --binary ./rokoko.fp16 --models $(MODELS)

clean:
	rm -f rokoko rokoko.fp16 src/kernels.o src/main.o src/cutlass_conv.o \
		src/cutlass_gemm.o src/cutlass_gemm_f16.o src/cutlass_conv_f16.o \
		tests/helpers tests/audio tests/runtime.o tests/runtime tests/runtime.fp16 \
		$(FRONTEND)/normalize_cli $(FRONTEND)/g2p_check

# Fast, offline checks deliberately use no CUDA flags or Python dependencies.
tests/helpers: tests/helpers.cpp src/phonemes.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Isrc $< -o $@

test: test-cpu
test-cpu: tests/helpers $(FRONTEND)/normalize_cli
	python3 tests/test_cpu.py

tests/runtime.o: tests/runtime.cu src/rokoko.h src/phonemes.h src/weights.h src/g2p.h src/rokoko_common.h src/audio.h src/artifact_format.h src/model_schema.h
	$(NVCC) $(NVFLAGS) -c $< -o $@
tests/runtime: tests/runtime.o src/rokoko.cpp src/weights.cpp src/weights.h $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma tests/runtime.o src/rokoko.cpp src/weights.cpp $(SHARED_OBJS) $(LDFLAGS) -o $@
tests/runtime.fp16: tests/runtime.o src/rokoko_f16.cpp src/weights.cpp src/weights.h $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma tests/runtime.o src/rokoko_f16.cpp src/weights.cpp $(SHARED_OBJS) $(LDFLAGS) -o $@

REFERENCE_PYTHON ?= .venv-tests/bin/python
OFFICIAL ?= tests/models/Kokoro-82M
MODELS ?= $(HOME)/.cache/rokoko
.PHONY: test-gpu
test-gpu: rokoko rokoko.fp16 tests/runtime tests/runtime.fp16
	$(REFERENCE_PYTHON) tests/gpu.py --official $(OFFICIAL) --models $(MODELS)
	$(REFERENCE_PYTHON) tests/reference.py --official $(OFFICIAL)
	CUDA_HOME="$(CUDA_HOME)" python3 tests/mutations.py --models "$(MODELS)"

tests/audio: tests/audio.cpp src/audio.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Isrc $< -o $@
test-cpu: tests/audio
