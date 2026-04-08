#include "simulation.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#ifdef _OPENMP
  #include <omp.h>
#endif

// ── Zero forces  [PARALLEL: embarrassingly parallel – no dependencies] ────────
void zero_forces(std::vector<Particle>& p) {
    int n = static_cast<int>(p.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        p[i].force = {0.0, 0.0, 0.0};
}

// ── Gravity  [PARALLEL: each particle independent] ────────────────────────────
void add_gravity(std::vector<Particle>& p, double g) {
    int n = static_cast<int>(p.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        p[i].force.z += p[i].mass * g;
}

// ── Particle–particle contact  [PARALLEL: private-array reduction] ────────────
//
//  RACE CONDITION PROBLEM:
//    For each pair (i,j), the loop writes to force[i] AND force[j].
//    Two threads touching the same particle from different pairs = data race.
//
//  SOLUTION – Thread-private force arrays + manual reduction:
//    Each thread accumulates contacts into its OWN private copy of the force
//    array (zero-initialised).  After the double loop, a parallel reduction
//    sums every thread's contribution into the global particle force.
//    Zero contention during pair evaluation → maximum parallelism.
//
void compute_particle_contacts(std::vector<Particle>& p, double kn, double gamma_n) {
    int n = static_cast<int>(p.size());

    // How many threads will actually run?
    #ifdef _OPENMP
      int n_threads = omp_get_max_threads();
    #else
      int n_threads = 1;
    #endif

    // lf[thread_id][particle_id]  – all zeros
    std::vector<std::vector<Vec3>> lf(n_threads,
                                      std::vector<Vec3>(n, {0.0, 0.0, 0.0}));

    #pragma omp parallel
    {
        #ifdef _OPENMP
          int tid = omp_get_thread_num();
        #else
          int tid = 0;
        #endif
        std::vector<Vec3>& my_f = lf[tid];

        // schedule(dynamic,4): rows near i=0 have (N-1) pairs,
        // rows near i=N have ~0 pairs – unequal work per row.
        // Dynamic scheduling gives idle threads the next chunk → better balance.
        #pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                Vec3   r_ij  = p[j].pos - p[i].pos;
                double d_ij  = r_ij.norm();
                if (d_ij < 1.0e-12) continue;

                double delta = p[i].radius + p[j].radius - d_ij;
                if (delta > 0.0) {
                    Vec3   n_ij   = r_ij * (1.0 / d_ij);
                    double vn     = (p[j].vel - p[i].vel).dot(n_ij);
                    double fn_mag = std::max(0.0, kn * delta - gamma_n * vn);
                    Vec3   fvec   = n_ij * fn_mag;
                    // Only writes to THIS thread's private arrays → no race
                    my_f[i] -= fvec;
                    my_f[j] += fvec;
                }
            }
        }
    } // implicit barrier – all pairs processed

    // Reduction: each particle i summed across all thread arrays.
    // Different particles go to different threads → still no race.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        for (int t = 0; t < n_threads; ++t)
            p[i].force += lf[t][i];
}


// ── Wall contacts  [PARALLEL: per-particle, no cross-particle writes] ─────────
void compute_wall_contacts(std::vector<Particle>& p, Vec3 box, double kn, double gamma_n) {
    int n = static_cast<int>(p.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        { double d = p[i].radius - p[i].pos.z;
          if (d > 0) { double fn=std::max(0.0,kn*d - gamma_n*p[i].vel.z); p[i].force.z+=fn; } }
        { double d = p[i].radius - (box.z - p[i].pos.z);
          if (d > 0) { double fn=std::max(0.0,kn*d + gamma_n*p[i].vel.z); p[i].force.z-=fn; } }
        { double d = p[i].radius - p[i].pos.x;
          if (d > 0) { double fn=std::max(0.0,kn*d - gamma_n*p[i].vel.x); p[i].force.x+=fn; } }
        { double d = p[i].radius - (box.x - p[i].pos.x);
          if (d > 0) { double fn=std::max(0.0,kn*d + gamma_n*p[i].vel.x); p[i].force.x-=fn; } }
        { double d = p[i].radius - p[i].pos.y;
          if (d > 0) { double fn=std::max(0.0,kn*d - gamma_n*p[i].vel.y); p[i].force.y+=fn; } }
        { double d = p[i].radius - (box.y - p[i].pos.y);
          if (d > 0) { double fn=std::max(0.0,kn*d + gamma_n*p[i].vel.y); p[i].force.y-=fn; } }
    }
}

// ── Semi-implicit Euler  [PARALLEL: per-particle, fully independent] ──────────
void integrate_particles(std::vector<Particle>& p, double dt) {
    int n = static_cast<int>(p.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double inv_m = 1.0 / p[i].mass;
        p[i].vel.x += p[i].force.x * inv_m * dt;
        p[i].vel.y += p[i].force.y * inv_m * dt;
        p[i].vel.z += p[i].force.z * inv_m * dt;
        p[i].pos.x += p[i].vel.x * dt;
        p[i].pos.y += p[i].vel.y * dt;
        p[i].pos.z += p[i].vel.z * dt;
    }
}

// ── Kinetic energy  [PARALLEL: scalar reduction – built-in OpenMP clause] ─────
double compute_kinetic_energy(const std::vector<Particle>& p) {
    double ke = 0.0;
    int n = static_cast<int>(p.size());
    #pragma omp parallel for reduction(+:ke) schedule(static)
    for (int i = 0; i < n; ++i)
        ke += 0.5 * p[i].mass * p[i].vel.dot(p[i].vel);
    return ke;
}

// ── Profiling report ───────────────────────────────────────────────────────────
void print_timing_report(const TimingInfo& t, int total_steps) {
    double total = t.t_zero_forces + t.t_gravity + t.t_p_contacts
                 + t.t_w_contacts  + t.t_integrate + t.t_ke + t.t_io;
    auto pct = [&](double v) { return total > 0 ? 100.0*v/total : 0.0; };

    #ifdef _OPENMP
      int nthreads = omp_get_max_threads();
    #else
      int nthreads = 1;
    #endif

    std::cout << "\n============================================================\n";
    std::cout << "  PROFILING REPORT  (" << total_steps << " steps)"
              << "  threads=" << nthreads << "\n";
    std::cout << "============================================================\n";
    std::cout << std::fixed << std::setprecision(6);
    auto row = [&](const char* name, double val) {
        std::cout << std::left  << std::setw(28) << name
                  << std::right << std::setw(12) << val
                  << std::setw(10) << std::fixed << std::setprecision(2)
                  << pct(val) << " %\n";
    };
    std::cout << std::left  << std::setw(28) << "Function"
              << std::right << std::setw(12) << "Time (s)"
              << std::setw(10) << "Pct" << "\n";
    std::cout << std::string(50, '-') << "\n";
    row("zero_forces",            t.t_zero_forces);
    row("add_gravity",            t.t_gravity);
    row("compute_p_contacts",     t.t_p_contacts);
    row("compute_w_contacts",     t.t_w_contacts);
    row("integrate_particles",    t.t_integrate);
    row("compute_kinetic_energy", t.t_ke);
    row("I/O",                    t.t_io);
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::left << std::setw(28) << "TOTAL"
              << std::right << std::setw(12) << std::setprecision(6) << total << "\n";
    std::cout << "============================================================\n\n";
}
