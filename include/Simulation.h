#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <cmath>
#include <fstream>

// ── Vec3 ─────────────────────────────────────────────────────────────────────
struct Vec3 {
    double x, y, z;
    Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3 operator*(double s)      const { return {x*s,   y*s,   z*s};   }
    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x-=v.x; y-=v.y; z-=v.z; return *this; }
    double dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    double norm()             const { return std::sqrt(x*x + y*y + z*z); }
};

// ── Particle ─────────────────────────────────────────────────────────────────
struct Particle {
    Vec3   pos;
    Vec3   vel;
    Vec3   force;
    double mass;
    double radius;
};

// ── Profiling ────────────────────────────────────────────────────────────────
struct TimingInfo {
    double t_zero_forces = 0.0;
    double t_gravity     = 0.0;
    double t_p_contacts  = 0.0;
    double t_w_contacts  = 0.0;
    double t_integrate   = 0.0;
    double t_ke          = 0.0;
    double t_io          = 0.0;
};

// ── Physics prototypes ────────────────────────────────────────────────────────
void   zero_forces              (std::vector<Particle>& p);
void   add_gravity              (std::vector<Particle>& p, double g);
void   compute_particle_contacts(std::vector<Particle>& p, double kn, double gamma_n);
void   compute_wall_contacts    (std::vector<Particle>& p, Vec3 box, double kn, double gamma_n);
void   integrate_particles      (std::vector<Particle>& p, double dt);
double compute_kinetic_energy   (const std::vector<Particle>& p);
void   print_timing_report      (const TimingInfo& t, int total_steps);

#endif // SIMULATION_H