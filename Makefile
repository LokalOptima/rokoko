CUDA_HOME ?= /usr/local/cuda-13.1
CACHE_DIR  = $(HOME)/.cache/rokoko

CXX      = g++
NVCC     = $(CUDA_HOME)/bin/nvcc
CXXFLAGS = -std=c++17 -O3 -march=native -flto=auto -I$(CUDA_HOME)/include -Isrc
NVFLAGS  = -std=c++17 -O3 -I$(CUDA_HOME)/include -Isrc --expt-relaxed-constexpr
LDFLAGS  = -flto=auto -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt -lpthread

.PHONY: clean install bundle

src/kernels.o: src/kernels.cu src/kernels.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/main.o: src/main.cu src/g2p.h src/normalize.h src/weights.h src/kernels.h \
            src/bundle.h src/server.h src/cpp-httplib/httplib.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

rokoko: src/main.o src/tts.cpp src/weights.cpp src/weights.h src/kernels.o
	$(CXX) $(CXXFLAGS) -mavx2 -mfma \
		src/main.o src/tts.cpp src/weights.cpp \
		src/kernels.o $(LDFLAGS) -o $@

bundle: rokoko.bundle

rokoko.bundle: weights/weights.bin weights/g2p_v8_model.bin voices/*.bin
	uv run scripts/pack.py -o $@

install: rokoko.bundle
	mkdir -p $(CACHE_DIR)
	cp -u rokoko.bundle $(CACHE_DIR)/

clean:
	rm -f rokoko rokoko.bundle src/kernels.o src/main.o
