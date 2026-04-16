##############################################################
#  TSMM Benchmark Makefile
#
#  Usage:
#    make                        # build with OpenBLAS (default)
#    make BLAS=mkl               # build with Intel MKL
#    make BLAS=none              # build with built-in reference (no BLAS)
#    make run                    # build + run locally
#    make web                    # start web dashboard server
#    make clean
##############################################################

BLAS     ?= openblas
CXX      ?= g++
TARGET    = benchmark

# Common flags
CXXFLAGS  = -O3 -march=native -std=c++17 -ffast-math
CXXFLAGS += -fopenmp
CXXFLAGS += -Wall -Wextra -Wno-unused-parameter
LDFLAGS   = -fopenmp -lm -lpthread

# BLAS backend
ifeq ($(BLAS), mkl)
  ifndef MKLROOT
    $(error MKLROOT not set. Run: source /opt/intel/mkl/bin/mklvars.sh intel64)
  endif
  CXXFLAGS += -DHAVE_MKL -I$(MKLROOT)/include
  LDFLAGS  += -L$(MKLROOT)/lib/intel64 \
              -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
              -liomp5 -ldl
else ifeq ($(BLAS), openblas)
  CXXFLAGS += -DHAVE_OPENBLAS
  LDFLAGS  += -lopenblas
else
  # BLAS=none: built-in reference, no external library needed
  CXXFLAGS += -UHAVE_MKL -UHAVE_OPENBLAS
endif

# AVX-512 (enabled automatically via -march=native on supported CPUs)
# Force enable for cross-compilation:  make AVX512=1
ifeq ($(AVX512), 1)
  CXXFLAGS += -mavx512f -mavx512dq -mavx512bw -mavx512vl
endif

SRC = src/benchmark.cpp

.PHONY: all run web clean help

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $(TARGET)  [BLAS=$(BLAS)]"

run: $(TARGET)
	@mkdir -p web
	./$(TARGET) --output web/results.json

run-required: $(TARGET)
	@mkdir -p web
	./$(TARGET) --output web/results.json --required-only

web:
	@echo "Starting web dashboard on http://localhost:8080"
	python3 web/server.py

clean:
	rm -f $(TARGET) web/results.json

help:
	@echo "Targets: all run run-required web clean"
	@echo "BLAS=mkl|openblas|none  (default: openblas)"
	@echo "AVX512=1                (force AVX-512 flags)"
