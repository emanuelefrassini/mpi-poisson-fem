MPICC ?= mpicc
CC ?= gcc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDLIBS = -lm

.PHONY: all clean test smoke
all: bin/poisson_mpi

bin/poisson_mpi: src/poisson_mpi.c
	mkdir -p bin
	$(MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

# This builds a true single-rank smoke-test binary without requiring MPI.
# It does NOT test the distributed communication path.
bin/poisson_serial_smoke: src/poisson_mpi.c tests/mpi_shim.c tests/mpi_shim.h
	mkdir -p bin
	$(CC) $(CFLAGS) -DSERIAL_MPI_SHIM -o $@ src/poisson_mpi.c tests/mpi_shim.c $(LDLIBS)

smoke: bin/poisson_serial_smoke
	python3 tests/test_solver.py --serial-only

test: bin/poisson_mpi
	python3 tests/test_solver.py

clean:
	rm -rf bin
