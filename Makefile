MAKEFLAGS += -j$(shell nproc)
CUDA_HOME ?= /usr/local/cuda-13.1
CUTLASS   ?= third_party/cutlass/include
ASSET_DIR ?= build/assets
ASSET_SOURCE ?=
OFFLINE ?= 0
ASSET_FLAGS = $(if $(ASSET_SOURCE),--source "$(ASSET_SOURCE)") $(if $(filter 1,$(OFFLINE)),--offline)

.PHONY: FORCE
FORCE:

$(ASSET_DIR)/embedded.S: scripts/bundle_assets.py assets/manifest.json FORCE
	python3 scripts/bundle_assets.py --directory "$(ASSET_DIR)" $(ASSET_FLAGS)

$(ASSET_DIR)/embedded.o: $(ASSET_DIR)/embedded.S scripts/bundle_assets.py assets/manifest.json
	$(CXX) -c $< -o $@

.PRECIOUS: $(ASSET_DIR)/embedded.S

CXX      = g++
NVCC     = $(CUDA_HOME)/bin/nvcc
CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -I$(CUDA_HOME)/include -Isrc
NVFLAGS  = -std=c++17 -O3 -arch=native -I$(CUDA_HOME)/include -Isrc --expt-relaxed-constexpr
LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lpthread

SHARED_OBJS = src/kernels.o src/cutlass_gemm.o \
              src/cutlass_gemm_f16.o src/cutlass_conv_f16.o

.PHONY: clean bench test-frontend test test-cpu
.DEFAULT_GOAL := rokoko

src/kernels.o: src/kernels.cu src/kernels.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/cutlass_gemm.o: src/cutlass_gemm.cu
	$(NVCC) $(NVFLAGS) -I$(CUTLASS) -c $< -o $@

src/cutlass_gemm_f16.o: src/cutlass_gemm_f16.cu
	$(NVCC) $(NVFLAGS) -I$(CUTLASS) -c $< -o $@

src/cutlass_conv_f16.o: src/cutlass_conv_f16.cu
	$(NVCC) $(NVFLAGS) -I$(CUTLASS) -c $< -o $@

src/main.o: src/main.cu src/embedded.h src/device.h src/byte_reader.h src/rokoko.h src/phonemes.h src/request_json.h src/artifact_format.h src/model_schema.h src/g2p.h src/normalize.h src/weights.h src/rokoko_common.h src/audio.h \
            src/kernels.h src/server.h src/cpp-httplib/httplib.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

rokoko: Makefile $(ASSET_DIR)/embedded.o src/embedded.cpp src/embedded.h src/main.o src/rokoko.cpp src/weights.cpp src/weights.h src/device.h src/rokoko_common.h src/audio.h src/artifact_format.h src/model_schema.h $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma \
		src/main.o src/rokoko.cpp src/weights.cpp src/embedded.cpp \
		$(ASSET_DIR)/embedded.o $(SHARED_OBJS) $(LDFLAGS) -o $@

# --- Text frontend checks (see tests/frontend/README.md) ---

FRONTEND = tests/frontend
FRONTEND_PYTHON ?= uv run
G2P ?= $(ASSET_DIR)/g2p.bin

$(FRONTEND)/normalize_cli: $(FRONTEND)/normalize_cli.cpp src/normalize.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Isrc $< -o $@

$(FRONTEND)/g2p_check: $(FRONTEND)/g2p_check.cu src/g2p.h src/byte_reader.h src/normalize.h src/cutlass_gemm.o src/kernels.o
	$(NVCC) $(NVFLAGS) $< src/cutlass_gemm.o src/kernels.o -o $@

test-frontend: $(FRONTEND)/normalize_cli $(FRONTEND)/g2p_check
	$(FRONTEND_PYTHON) $(FRONTEND)/eval_frontend.py --g2p $(G2P)

bench: rokoko
	python3 tests/bench.py --models $(MODELS)

clean:
	rm -f "$(ASSET_DIR)/embedded.o"
	rm -f rokoko rokoko.cpu src/kernels.o src/main.o \
		src/cutlass_gemm.o src/cutlass_gemm_f16.o src/cutlass_conv_f16.o \
		tests/helpers tests/audio tests/runtime.o tests/runtime \
		$(FRONTEND)/normalize_cli $(FRONTEND)/g2p_check

# Fast, offline checks deliberately use no CUDA flags or Python dependencies.
tests/helpers: tests/helpers.cpp src/phonemes.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Isrc $< -o $@

test: test-cpu
test-cpu: tests/helpers $(FRONTEND)/normalize_cli build/byte_reader
	python3 tests/test_cpu.py
	./build/byte_reader
	python3 tests/test_bundle.py

tests/runtime.o: tests/runtime.cu src/embedded.h src/device.h src/byte_reader.h src/rokoko.h src/phonemes.h src/weights.h src/g2p.h src/rokoko_common.h src/audio.h src/artifact_format.h src/model_schema.h
	$(NVCC) $(NVFLAGS) -c $< -o $@
tests/runtime: Makefile $(ASSET_DIR)/embedded.o src/embedded.cpp tests/runtime.o src/rokoko.cpp src/weights.cpp src/weights.h $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma tests/runtime.o src/rokoko.cpp src/weights.cpp src/embedded.cpp $(ASSET_DIR)/embedded.o $(SHARED_OBJS) $(LDFLAGS) -o $@
REFERENCE_PYTHON ?= .venv-tests/bin/python
OFFICIAL ?= tests/models/Kokoro-82M
MODELS ?= $(ASSET_DIR)
.PHONY: test-gpu
test-gpu: rokoko tests/runtime
	$(REFERENCE_PYTHON) tests/gpu.py --official $(OFFICIAL) --models $(MODELS)
	$(REFERENCE_PYTHON) tests/reference.py --official $(OFFICIAL)
	CUDA_HOME="$(CUDA_HOME)" python3 tests/mutations.py --asset-dir "$(ASSET_DIR)"

tests/audio: tests/audio.cpp src/audio.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Isrc $< -o $@
test-cpu: tests/audio

# Training dependencies are optional and shared with the reference-test environment.
TRAINING_PYTHON ?= $(REFERENCE_PYTHON)
.PHONY: test-training
test-training: tests/frontend/normalize_cli tests/frontend/g2p_check
	@test -n "$(G2P_CHECKPOINT)" || { echo 'Set G2P_CHECKPOINT to the V11 best_exact.pt file'; exit 1; }
	$(TRAINING_PYTHON) tests/training.py --checkpoint "$(G2P_CHECKPOINT)"

build/byte_reader: tests/byte_reader.cpp src/byte_reader.h
	mkdir -p build
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Isrc $< -o $@


.PHONY: test-bundle
test-bundle: rokoko
	python3 tests/bundle_smoke.py

# CPU inference is isolated from CUDA object files and uses a static SIMD BLAS.
CPU_BUILD ?= build/cpu
.PHONY: cpu
cpu: rokoko.cpu
rokoko.cpu: FORCE
	cmake -S . -B "$(CPU_BUILD)" -DCMAKE_BUILD_TYPE=Release -DROKOKO_BACKEND=CPU -DROKOKO_BUILD_DIAGNOSTICS=ON -DROKOKO_ASSET_DIR="$(abspath $(ASSET_DIR))" -DROKOKO_ASSET_SOURCE="$(ASSET_SOURCE)" -DROKOKO_OFFLINE=$(OFFLINE)
	cmake --build "$(CPU_BUILD)" -j4
	cp "$(CPU_BUILD)/rokoko" $@

src/main.o tests/runtime.o: src/backend_ops.h src/device.h
src/kernels.o: src/device.h
rokoko tests/runtime: src/backend_ops.h src/device.h

.PHONY: test-cpu-inference test-cpu-parity
# CPU-only inference checks (requires the reference environment's numpy).
test-cpu-inference: rokoko.cpu
	ctest --test-dir "$(CPU_BUILD)" --output-on-failure
	$(REFERENCE_PYTHON) tests/cpu_inference.py --binary ./rokoko.cpu --runtime "$(CPU_BUILD)/rokoko_runtime" --models "$(MODELS)"
	python3 tests/bundle_smoke.py --binary ./rokoko.cpu

# Requires CUDA solely for the independent comparison executable.
test-cpu-parity: rokoko.cpu rokoko tests/runtime $(FRONTEND)/g2p_check
	$(REFERENCE_PYTHON) tests/cpu_parity.py --cpu ./rokoko.cpu --cpu-runtime "$(CPU_BUILD)/rokoko_runtime" --cpu-g2p "$(CPU_BUILD)/rokoko_g2p_check" --g2p-asset "$(G2P)"
