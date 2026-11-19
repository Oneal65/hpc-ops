# CuTe practice labs. Each lab is one .cu file -> one binary in bin/.
#   make            build every lab
#   make run        build and run every lab
#   make bin/lab11_eval && ./bin/lab11_eval
ROOT  := ../../..
NVCC  := nvcc
FLAGS := -O2 -std=c++17 --expt-relaxed-constexpr -lineinfo \
         -arch=sm_90a -DHPC_TARGET_ARCH=90 \
         -I$(ROOT) -I$(ROOT)/3rd/cutlass/include -I. -lcuda

SRCS := $(wildcard *.cu)
BINS := $(patsubst %.cu,bin/%,$(SRCS))

all: $(BINS)

bin/%: %.cu check.hpp | bin
	$(NVCC) -o $@ $< $(FLAGS)

bin:
	mkdir -p bin

run: all
	@for b in $(BINS); do echo "===== $$b"; ./$$b || exit 1; done

clean:
	rm -rf bin

.PHONY: all run clean
