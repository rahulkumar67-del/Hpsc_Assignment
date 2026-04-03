#include "simulation.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

// ── Zero forces ───────────────────────────────────────────────────────────────
void zero_forces(std::vector<Particle>& p) {
    for (auto& part : p)
        part.force = {0.0, 0.0, 0.0};
}

// ── Gravity ───────────────────────────────────────────────────────────────────
void add_gravity(std::vector<Particle>& p, double g) {
    for (auto& part : p)
        part.force.z += part.mass * g;
}

// ── Particle–particle contact (spring-dashpot) ────────────────────────────────
void compute_particle_contacts(std::vector<Particle>& p, double kn, double gamma_n) {
    int n = static_cast<int>(p.size());
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
                Vec3   force  = n_ij * fn_mag;
                p[i].force -= force;
                p[j].force += force;
            }
        }
    }
}



// ── Wall contacts (all 6 faces) ────────────────────────────────────────────────
void compute_wall_contacts(std::vector<Particle>& p, Vec3 box, double kn, double gamma_n) {
    for (auto& part : p) {
        // Floor  z=0, ceiling z=Lz
        { double d = part.radius - part.pos.z;
          if (d > 0) { double fn = std::max(0.0, kn*d - gamma_n*part.vel.z);   part.force.z += fn; } }
        { double d = part.radius - (box.z - part.pos.z);
          if (d > 0) { double fn = std::max(0.0, kn*d + gamma_n*part.vel.z);   part.force.z -= fn; } }
        // Walls  x=0, x=Lx
        { double d = part.radius - part.pos.x;
          if (d > 0) { double fn = std::max(0.0, kn*d - gamma_n*part.vel.x);   part.force.x += fn; } }
        { double d = part.radius - (box.x - part.pos.x);
          if (d > 0) { double fn = std::max(0.0, kn*d + gamma_n*part.vel.x);   part.force.x -= fn; } }
        // Walls  y=0, y=Ly
        { double d = part.radius - part.pos.y;
          if (d > 0) { double fn = std::max(0.0, kn*d - gamma_n*part.vel.y);   part.force.y += fn; } }
        { double d = part.radius - (box.y - part.pos.y);
          if (d > 0) { double fn = std::max(0.0, kn*d + gamma_n*part.vel.y);   part.force.y -= fn; } }
    }
}

// ── Semi-implicit Euler ────────────────────────────────────────────────────────
void integrate_particles(std::vector<Particle>& p, double dt) {
    for (auto& part : p) {
        double inv_m = 1.0 / part.mass;
        part.vel.x += part.force.x * inv_m * dt;
        part.vel.y += part.force.y * inv_m * dt;
        part.vel.z += part.force.z * inv_m * dt;
        part.pos.x += part.vel.x * dt;
        part.pos.y += part.vel.y * dt;
        part.pos.z += part.vel.z * dt;
    }
}

// ── Kinetic energy ─────────────────────────────────────────────────────────────
double compute_kinetic_energy(const std::vector<Particle>& p) {
    double ke = 0.0;
    for (const auto& part : p)
        ke += 0.5 * part.mass * part.vel.dot(part.vel);
    return ke;
}

// ── Profiling report ───────────────────────────────────────────────────────────
void print_timing_report(const TimingInfo& t, int total_steps) {
    double total = t.t_zero_forces + t.t_gravity + t.t_p_contacts
                 + t.t_w_contacts  + t.t_integrate + t.t_ke + t.t_io;
    auto pct = [&](double v) { return total > 0 ? 100.0*v/total : 0.0; };

    std::cout << "\n============================================================\n";
    std::cout << "  PROFILING REPORT  (" << total_steps << " timesteps)\n";
    std::cout << "============================================================\n";
    std::cout << std::fixed << std::setprecision(6);
    auto row = [&](const char* name, double val) {
        std::cout << std::left  << std::setw(28) << name
                  << std::right << std::setw(12) << val
                  << std::setw(10) << std::fixed << std::setprecision(2) << pct(val) << " %\n";
    };
    std::cout << std::left  << std::setw(28) << "Function"
              << std::right << std::setw(12) << "Time (s)"
              << std::setw(10) << "Pct"      << "\n";
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