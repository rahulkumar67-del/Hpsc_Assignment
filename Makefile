CXX      = g++
CXXFLAGS = -Iinclude -O3 -Wall -std=c++17
SRC      = src/main.cpp src/physics.cpp
OBJ      = $(SRC:.cpp=.o)

# Three build targets:
#   particle_sim        – optimised + OpenMP  (parallel)
#   particle_sim_serial – optimised, no OpenMP (serial O3)
#   particle_sim_noopt  – no optimisation, no OpenMP (serial O0, baseline)

all: results particle_sim particle_sim_serial particle_sim_noopt

results:
	mkdir -p results

# ── Parallel build (OpenMP + O3) ─────────────────────────────────
particle_sim: $(SRC) include/Simulation.h
	$(CXX) $(CXXFLAGS) -fopenmp $(SRC) -o $@

# ── Serial optimised build (O3, no OpenMP) ───────────────────────
particle_sim_serial: $(SRC) include/Simulation.h
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

# ── Serial unoptimised build (O0, no OpenMP) – compiler baseline ─
particle_sim_noopt: $(SRC) include/Simulation.h
	$(CXX) -Iinclude -O0 -Wall -std=c++17 $(SRC) -o $@

# ── Compiler optimisation comparison ─────────────────────────────
#    Runs N=200 with both builds, shows timing difference
opt_compare: particle_sim_serial particle_sim_noopt
	@echo ""
	@echo "=== UNOPTIMISED (-O0, serial) ==="
	./particle_sim_noopt  runtable 2>&1 | grep "N=5000"
	@echo ""
	@echo "=== OPTIMISED (-O3, serial) ==="
	./particle_sim_serial runtable 2>&1 | grep "N=5000"
	@echo ""

# ── Verification tests ───────────────────────────────────────────
tests: particle_sim
	./particle_sim freefall
	./particle_sim constvel
	./particle_sim bounce
	./particle_sim errordt

# ── Serial profiling (runtime table) ─────────────────────────────
runtable: particle_sim_serial
	./particle_sim_serial runtable

# ── Simulation experiments ───────────────────────────────────────
multi200: particle_sim
	./particle_sim multi200

multi1000: particle_sim
	./particle_sim multi1000

multi5000: particle_sim
	./particle_sim multi5000

# ── Parallel correctness check ───────────────────────────────────
verify: particle_sim
	./particle_sim verify200
	./particle_sim verify1000

# ── Scaling / performance study ──────────────────────────────────
scaling200: particle_sim
	./particle_sim scaling200

scaling1000: particle_sim
	./particle_sim scaling1000

scaling5000: particle_sim
	./particle_sim scaling5000

# ── Run everything ───────────────────────────────────────────────
run_all: particle_sim particle_sim_serial particle_sim_noopt
	./particle_sim all

# ── Cleanup ──────────────────────────────────────────────────────
clean:
	rm -f particle_sim particle_sim_serial particle_sim_noopt src/*.o