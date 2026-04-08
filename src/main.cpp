#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <string>
#include <iomanip>
#ifdef _OPENMP
  #include <omp.h>
#endif
#include "simulation.h"

// ── Wall-clock timer ──────────────────────────────────────────────────────────
static double wall_time() {
    using C = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(C::now().time_since_epoch()).count();
}
#define TIMED(acc, expr) { double _t0=wall_time(); (expr); acc += wall_time()-_t0; }

// ── Helper: get hardware thread count ────────────────────────────────────────
static int hw_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
static void set_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

// ── Non-overlapping random particle initialisation ────────────────────────────
static void init_particles(std::vector<Particle>& P, int N, double R, Vec3 box,
                            unsigned seed = 42) {
    P.clear(); P.reserve(N);
    srand(seed);
    const double margin = R * 1.1;
    const int    max_try = 2000;
    for (int i = 0; i < N; ++i) {
        bool placed = false;
        for (int t = 0; t < max_try && !placed; ++t) {
            double rx = margin + (double)rand()/RAND_MAX*(box.x - 2*margin);
            double ry = margin + (double)rand()/RAND_MAX*(box.y - 2*margin);
            double rz = margin + (double)rand()/RAND_MAX*(box.z - 2*margin);
            bool ok = true;
            for (const auto& q : P) {
                double dx=q.pos.x-rx, dy=q.pos.y-ry, dz=q.pos.z-rz;
                if (std::sqrt(dx*dx+dy*dy+dz*dz) < 2.1*R) { ok=false; break; }
            }
            if (ok) { P.push_back({{rx,ry,rz},{0,0,0},{0,0,0},1.0,R}); placed=true; }
        }
        if (!placed) {
            double rx=margin+(double)rand()/RAND_MAX*(box.x-2*margin);
            double ry=margin+(double)rand()/RAND_MAX*(box.y-2*margin);
            double rz=margin+(double)rand()/RAND_MAX*(box.z-2*margin);
            P.push_back({{rx,ry,rz},{0,0,0},{0,0,0},1.0,R});
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 1 – Free Fall
// ─────────────────────────────────────────────────────────────────────────────
void run_freefall() {
    const double g=-9.81, dt=0.001, total=0.6, z0=5.0;
    std::vector<Particle> P = {{{5,5,z0},{0,0,0},{0,0,0},1.0,0.1}};
    std::ofstream out("results/freefall.csv");
    out << "step,time,z_num,vz_num,z_exact,vz_exact,error_z\n";
    out << std::scientific << std::setprecision(10);
    double t=0.0; int step=0; double max_err=0;
    while (t <= total+0.5*dt) {
        zero_forces(P); add_gravity(P, g); integrate_particles(P, dt);
        double ze=z0+0.5*g*t*t, vze=g*t;
        double err=std::abs(P[0].pos.z-ze);
        if (err>max_err) max_err=err;
        out<<step<<','<<t<<','<<P[0].pos.z<<','<<P[0].vel.z<<','<<ze<<','<<vze<<','<<err<<'\n';
        t+=dt; step++;
    }
    out.close();
    std::cout<<"[Free Fall   ] Max |err| = "<<max_err<<" m  -> results/freefall.csv\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Error vs dt study
// ─────────────────────────────────────────────────────────────────────────────
void run_error_vs_dt() {
    const double g=-9.81, z0=5.0, t_end=0.5;
    std::vector<double> dts={0.1,0.05,0.02,0.01,0.005,0.001,0.0005,0.0001};
    std::ofstream out("results/error_vs_dt.csv");
    out<<"dt,max_error_z,max_error_vz\n";
    out<<std::scientific<<std::setprecision(8);
    std::cout<<"[Error vs dt ] Computing...\n";
    for (double dt : dts) {
        std::vector<Particle> P={{{5,5,z0},{0,0,0},{0,0,0},1.0,0.1}};
        double t=0,ez=0,evz=0;
        while (t<=t_end+0.5*dt) {
            zero_forces(P); add_gravity(P,g); integrate_particles(P,dt);
            double err_z =std::abs(P[0].pos.z-(z0+0.5*g*t*t));
            double err_vz=std::abs(P[0].vel.z- g*t);
            if (err_z >ez ) ez =err_z;
            if (err_vz>evz) evz=err_vz;
            t+=dt;
        }
        out<<dt<<','<<ez<<','<<evz<<'\n';
        std::cout<<"  dt="<<dt<<"  max_err_z="<<ez<<"  max_err_vz="<<evz<<'\n';
    }
    out.close();
    std::cout<<"  -> results/error_vs_dt.csv\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 2 – Constant Velocity
// ─────────────────────────────────────────────────────────────────────────────
void run_constvel() {
    const double dt=0.001,total=2.0,vx0=1.5,x0=0.5;
    std::vector<Particle> P={{{x0,500,500},{vx0,0,0},{0,0,0},1.0,0.1}};
    std::ofstream out("results/constvel.csv");
    out<<"step,time,x_num,vx_num,x_exact,error_x\n";
    out<<std::scientific<<std::setprecision(10);
    double t=0; int step=0; double max_e=0;
    while (t<=total+0.5*dt) {
        zero_forces(P); integrate_particles(P,dt);
        double xe=x0+vx0*t, err=std::abs(P[0].pos.x-xe);
        if (err>max_e) max_e=err;
        out<<step<<','<<t<<','<<P[0].pos.x<<','<<P[0].vel.x<<','<<xe<<','<<err<<'\n';
        t+=dt; step++;
    }
    out.close();
    std::cout<<"[Const Vel   ] Max |err| = "<<max_e<<" m  -> results/constvel.csv\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 3 – Bounce
// ─────────────────────────────────────────────────────────────────────────────
void run_bounce() {
    const double g=-9.81,dt=1e-4,total=4.0;
    const double kn=1e5,gamma_n=50.0,R=0.1,z0=2.0;
    Vec3 box={10.0,10.0,10.0};
    std::vector<Particle> P={{{5,5,z0},{0,0,0},{0,0,0},1.0,R}};
    std::ofstream out("results/bounce.csv");
    out<<"step,time,z,vz,KE\n";
    out<<std::scientific<<std::setprecision(8);
    std::ofstream outPk("results/bounce_peaks.csv");
    outPk<<"bounce_num,time,peak_z\n";
    double t=0; int step=0,bounceN=0; double prev_vz=0.0;
    while (t<=total+0.5*dt) {
        zero_forces(P); add_gravity(P,g);
        compute_wall_contacts(P,box,kn,gamma_n);
        integrate_particles(P,dt);
        double ke=compute_kinetic_energy(P);
        out<<step<<','<<t<<','<<P[0].pos.z<<','<<P[0].vel.z<<','<<ke<<'\n';
        if (prev_vz>0.0&&P[0].vel.z<=0.0&&P[0].pos.z>R+0.01) {
            bounceN++;
            outPk<<bounceN<<','<<t<<','<<P[0].pos.z<<'\n';
        }
        prev_vz=P[0].vel.z; t+=dt; step++;
    }
    out.close(); outPk.close();
    std::cout<<"[Bounce      ] "<<bounceN<<" bounces detected  -> results/bounce.csv\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-particle simulation with full profiling
// ─────────────────────────────────────────────────────────────────────────────
void run_multi(int N) {
    double R   =(N<=200)?0.05 :(N<=1000)?0.035:0.025;
    double Lbox=(N<=200)?2.0  :(N<=1000)?3.5  :5.0;
    double T   =(N<=200)?5.0  :(N<=1000)?2.0  :3.0;
    const double dt=0.001,kn=1e4,gamma_n=100.0,g=-9.81;
    Vec3 box={Lbox,Lbox,Lbox};

    std::cout<<"\n[Multi-"<<N<<"] Initialising "<<N<<" particles...\n";
    std::vector<Particle> P;
    init_particles(P,N,R,box,42);
    std::cout<<"[Multi-"<<N<<"] "<<P.size()<<" particles in "
             <<Lbox<<"^3 box.  T="<<T<<"s\n";

    std::string suf="N"+std::to_string(N);
    std::ofstream outE("results/energy_"+suf+".csv");
    outE<<"step,time,KE\n";
    outE<<std::scientific<<std::setprecision(8);

    int snap_every=std::max(1,(int)(T/dt)/10);
    std::ofstream outS("results/snapshot_"+suf+".csv");
    outS<<"snap,time,x,y,z\n"; 
    outS<<std::scientific<<std::setprecision(6);

    TimingInfo TM;
    double t=0.0; int step=0, total_steps=(int)(T/dt+0.5);
    while (t<=T+0.5*dt) {
        TIMED(TM.t_zero_forces, zero_forces(P));
        TIMED(TM.t_gravity,     add_gravity(P,g));
        TIMED(TM.t_p_contacts,  compute_particle_contacts(P,kn,gamma_n));
        TIMED(TM.t_w_contacts,  compute_wall_contacts(P,box,kn,gamma_n));
        TIMED(TM.t_integrate,   integrate_particles(P,dt));
        
        if (step%10==0) {
            double _t0=wall_time(); double ke=compute_kinetic_energy(P); TM.t_ke+=wall_time()-_t0;
            _t0=wall_time(); outE<<step<<','<<t<<','<<ke<<'\n'; TM.t_io+=wall_time()-_t0;
        }
        if (step%snap_every==0) {
            double _t0=wall_time(); int sid=step/snap_every;
            for (const auto& pp:P)
                outS<<sid<<','<<t<<','<<pp.pos.x<<','<<pp.pos.y<<','<<pp.pos.z<<'\n';
            TM.t_io+=wall_time()-_t0;
        }
        t+=dt; step++;
    }
    outE.close(); outS.close();
    // Write timing CSV
    {
        std::ofstream outT("results/timing_"+suf+".csv");
        outT<<"function,time_s,pct\n";
        double tot=TM.t_zero_forces+TM.t_gravity+TM.t_p_contacts
                  +TM.t_w_contacts+TM.t_integrate+TM.t_ke+TM.t_io;
        auto wr=[&](const char* nm,double v){ outT<<nm<<','<<v<<','<<(tot>0?100*v/tot:0)<<'\n'; };
        wr("zero_forces",TM.t_zero_forces); wr("add_gravity",TM.t_gravity);
        wr("compute_p_contacts",TM.t_p_contacts); wr("compute_w_contacts",TM.t_w_contacts);
        wr("integrate_particles",TM.t_integrate); wr("compute_kinetic_energy",TM.t_ke);
        wr("io",TM.t_io); outT.close();
    }
    print_timing_report(TM,total_steps);
    std::cout<<"[Multi-"<<N<<"] -> results/energy_"<<suf<<".csv, snapshot_"<<suf<<".csv\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Serial runtime table  – benchmark N=200,1000,5000 with 1 thread
// Produces results/runtime_table.csv for the report table
// ─────────────────────────────────────────────────────────────────────────────
void run_runtime_table() {
    std::cout<<"\n[RuntimeTable] Serial profiling: N=200, 1000, 5000 (T=0.5s each)\n";
    std::ofstream out("results/runtime_table.csv");
    out<<"N,total_s,contact_s,contact_pct,wall_s,integrate_s,other_s,steps\n";
    out<<std::fixed<<std::setprecision(4);

    auto bench=[&](int N){
        set_threads(1);  // force serial
        double R   =(N<=200)?0.05 :(N<=1000)?0.035:0.025;
        double Lbox=(N<=200)?2.0  :(N<=1000)?3.5  :5.0;
        const double T=0.5, dt=0.001, kn=1e4, gamma_n=100.0, g=-9.81;
        Vec3 box={Lbox,Lbox,Lbox};

        std::cout<<"  Initialising N="<<N<<"...\n";
        std::vector<Particle> P; init_particles(P,N,R,box,42);

        TimingInfo TM; double t=0.0; int step=0, nsteps=(int)(T/dt);
        while (t<=T+0.5*dt) {
            TIMED(TM.t_zero_forces, zero_forces(P));
            TIMED(TM.t_gravity,     add_gravity(P,g));
            TIMED(TM.t_p_contacts,  compute_particle_contacts(P,kn,gamma_n));
            TIMED(TM.t_w_contacts,  compute_wall_contacts(P,box,kn,gamma_n));
            TIMED(TM.t_integrate,   integrate_particles(P,dt));
            if (step%50==0) { double _t=wall_time(); compute_kinetic_energy(P); TM.t_ke+=wall_time()-_t; }
            t+=dt; step++;
        }
        double total=TM.t_zero_forces+TM.t_gravity+TM.t_p_contacts
                    +TM.t_w_contacts +TM.t_integrate+TM.t_ke;
        double other=TM.t_zero_forces+TM.t_gravity+TM.t_w_contacts+TM.t_integrate+TM.t_ke;
        double pct  =(total>0)?100.0*TM.t_p_contacts/total:0;
        out<<N<<','<<total<<','<<TM.t_p_contacts<<','<<pct<<','
           <<TM.t_w_contacts<<','<<TM.t_integrate<<','<<other<<','<<nsteps<<'\n';
        std::cout<<"  N="<<N<<"  total="<<total<<"s  contacts="
                 <<TM.t_p_contacts<<"s ("<<std::setprecision(1)<<pct<<"%)\n";
        print_timing_report(TM,nsteps);
    };
    bench(200); bench(1000); bench(5000);
    out.close();
    std::cout<<"[RuntimeTable] -> results/runtime_table.csv\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Verification: serial (1 thread) vs parallel (max threads)
// Compare KE time series to detect correctness issues
// ─────────────────────────────────────────────────────────────────────────────
void run_verification(int N) {
    double R   =(N<=200)?0.05 :(N<=1000)?0.035:0.025;
    double Lbox=(N<=200)?2.0  :(N<=1000)?3.5  :5.0;
    const double T=0.5, dt=0.001, kn=1e4, gamma_n=100.0, g=-9.81;
    Vec3 box={Lbox,Lbox,Lbox};

    auto run_sim=[&](int nthreads)->std::vector<double>{
        set_threads(nthreads);
        std::vector<Particle> P; init_particles(P,N,R,box,42);
        std::vector<double> ke_hist;
        double t=0.0; int step=0;
        while (t<=T+0.5*dt) {
            zero_forces(P); add_gravity(P,g);
            compute_particle_contacts(P,kn,gamma_n);
            compute_wall_contacts(P,box,kn,gamma_n);
            integrate_particles(P,dt);
            if (step%10==0) ke_hist.push_back(compute_kinetic_energy(P));
            t+=dt; step++;
        }
        return ke_hist;
    };

    std::cout<<"\n[Verify N="<<N<<"] Running serial (1 thread)...\n";
    auto ke1=run_sim(1);

    int hw=hw_threads();
    std::cout<<"[Verify N="<<N<<"] Running parallel ("<<hw<<" threads)...\n";
    auto ke2=run_sim(hw);

    double max_diff=0.0, max_rel=0.0;
    for (size_t i=0; i<ke1.size()&&i<ke2.size(); ++i) {
        double diff=std::abs(ke1[i]-ke2[i]);
        double rel =diff/(std::abs(ke1[i])+1e-30);
        if (diff>max_diff) max_diff=diff;
        if (rel >max_rel ) max_rel =rel;
    }
    // Write comparison CSV
    std::ofstream outV("results/verify_N"+std::to_string(N)+".csv");
    outV<<"step_idx,KE_serial,KE_parallel,abs_diff\n";
    outV<<std::scientific<<std::setprecision(10);
    for (size_t i=0; i<ke1.size()&&i<ke2.size(); ++i)
        outV<<i<<','<<ke1[i]<<','<<ke2[i]<<','<<std::abs(ke1[i]-ke2[i])<<'\n';
    outV.close();

    std::cout<<"[Verify N="<<N<<"] Max |ΔKE|   = "<<max_diff<<" J\n";
    std::cout<<"[Verify N="<<N<<"] Max rel |ΔKE| = "<<max_rel<<"\n";
    if (max_rel < 1e-8)
        std::cout<<"[Verify N="<<N<<"] ✓ PASS: parallel matches serial (within floating-point)\n\n";
    else
        std::cout<<"[Verify N="<<N<<"] ✗ WARN: difference "<<max_rel<<" — check implementation\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Scaling study – sweep 1,2,4,...,max threads and record timing
// ─────────────────────────────────────────────────────────────────────────────
void run_scaling(int N) {
    double R   =(N<=200)?0.05 :(N<=1000)?0.035:0.025;
    double Lbox=(N<=200)?2.0  :(N<=1000)?3.5  :5.0;
    const double T=0.5, dt=0.001, kn=1e4, gamma_n=100.0, g=-9.81;
    Vec3 box={Lbox,Lbox,Lbox};

    int hw=hw_threads();
    std::vector<int> tlist;
    for (int t=1; t<=hw; t*=2) tlist.push_back(t);
    if (tlist.back()!=hw) tlist.push_back(hw);

    std::string suf="N"+std::to_string(N);
    std::ofstream scl("results/scaling_"+suf+".csv");
    scl<<"threads,total_time_s,contact_time_s,speedup,efficiency\n";
    scl<<std::fixed<<std::setprecision(6);

    std::cout<<"\n[Scaling "<<suf<<"] N="<<N<<"  T="<<T<<"s  hw="<<hw<<" threads\n";

    double t1=-1, c1=-1;
    for (int nt : tlist) {
        set_threads(nt);
        std::vector<Particle> P; init_particles(P,N,R,box,42);
        double t_contact=0.0, ws=wall_time();
        double t=0.0; int step=0;
        while (t<=T+0.5*dt) {
            zero_forces(P); add_gravity(P,g);
            double c0=wall_time();
            compute_particle_contacts(P,kn,gamma_n);
            t_contact+=wall_time()-c0;
            compute_wall_contacts(P,box,kn,gamma_n);
            integrate_particles(P,dt);
            t+=dt; step++;
        }
        double t_total=wall_time()-ws;
        if (t1<0) { t1=t_total; c1=t_contact; }
        double S=t1/t_total, E=S/nt;
        scl<<nt<<','<<t_total<<','<<t_contact<<','<<S<<','<<E<<'\n';
        std::cout<<"  p="<<std::setw(3)<<nt
                 <<"  total="<<std::setw(9)<<std::setprecision(4)<<t_total<<"s"
                 <<"  contact="<<std::setw(9)<<t_contact<<"s"
                 <<"  S="<<std::setprecision(3)<<S
                 <<"  E="<<std::setprecision(3)<<E<<'\n';
    }
    scl.close();
    std::cout<<"[Scaling "<<suf<<"] -> results/scaling_"<<suf<<".csv\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout<<"Usage: particle_sim <mode>\n"
                 <<"Modes:\n"
                 <<"  freefall   constvel   bounce   errordt     (verification tests)\n"
                 <<"  multi200   multi1000  multi5000             (simulation experiments)\n"
                 <<"  runtable                                    (serial profiling N=200,1000,5000)\n"
                 <<"  verify200  verify1000  verify5000           (serial vs parallel correctness)\n"
                 <<"  scaling200 scaling1000 scaling5000          (speedup & efficiency study)\n"
                 <<"  all                                         (run everything)\n";
        return 1;
    }
    std::string mode(argv[1]);
    system("mkdir -p results");

    if      (mode=="freefall")   { run_freefall(); }
    else if (mode=="constvel")   { run_constvel(); }
    else if (mode=="bounce")     { run_bounce(); }
    else if (mode=="errordt")    { run_error_vs_dt(); }
    else if (mode=="multi200")   { run_multi(200); }
    else if (mode=="multi1000")  { run_multi(1000); }
    else if (mode=="multi5000")  { run_multi(5000); }
    else if (mode=="runtable")   { run_runtime_table(); }
    else if (mode=="verify200")  { run_verification(200); }
    else if (mode=="verify1000") { run_verification(1000); }
    else if (mode=="verify5000") { run_verification(5000); }
    else if (mode=="scaling200") { run_scaling(200); }
    else if (mode=="scaling1000"){ run_scaling(1000); }
    else if (mode=="scaling5000"){ run_scaling(5000); }
    else if (mode=="all") {
        run_freefall();
        run_error_vs_dt();
        run_constvel();
        run_bounce();
        run_multi(200);
        run_multi(1000);
        run_runtime_table();
        run_verification(200);
        run_verification(1000);
        run_scaling(200);
        run_scaling(1000);
        run_scaling(5000);
    }
    else { std::cerr<<"Unknown mode: "<<mode<<"\n"; return 1; }
    return 0;
}