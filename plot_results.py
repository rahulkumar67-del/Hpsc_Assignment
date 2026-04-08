"""
plot_results.py  –  One graph per file. Publication-quality.
Run after:  ./particle_sim all
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family":       "DejaVu Serif",
    "font.size":         11,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
C = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548"]
os.makedirs("plots", exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"plots/{name}.pdf", bbox_inches="tight")
    plt.savefig(f"plots/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved  plots/{name}.pdf  +  .png")

def skip(fname):
    print(f"  [skip] {fname} not found")

def v(s):
    return np.asarray(s, dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
# 01a – Free fall trajectory
# ─────────────────────────────────────────────────────────────────────────────
def plot_freefall_trajectory():
    f = "results/freefall.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(v(df.time), v(df.z_exact), "k--", lw=1.5, label="Analytical  $z_0 + \\frac{1}{2}gt^2$")
    ax.plot(v(df.time), v(df.z_num),   color=C[0], lw=1.0, alpha=0.85, label="Numerical (Euler)")
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("Height  $z$  (m)")
    ax.set_title("Free Fall – Trajectory")
    ax.legend()
    savefig("01a_freefall_trajectory")

# ─────────────────────────────────────────────────────────────────────────────
# 01b – Free fall absolute error vs time
# ─────────────────────────────────────────────────────────────────────────────
def plot_freefall_error():
    f = "results/freefall.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(v(df.time), v(df.error_z), color=C[1], lw=1.2)
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("$|z_{\\rm num} - z_{\\rm exact}|$  (m)")
    ax.set_title("Free Fall – Absolute Error vs Time")
    savefig("01b_freefall_error")

# ─────────────────────────────────────────────────────────────────────────────
# 02 – Error vs Δt (convergence)
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_vs_dt():
    f = "results/error_vs_dt.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(v(df.dt), v(df.max_error_z),  "o-",  color=C[0], label="Max error $z$")
    ax.loglog(v(df.dt), v(df.max_error_vz), "s--", color=C[1], label="Max error $v_z$")
    dt_ref = np.array([v(df.dt).min(), v(df.dt).max()])
    ax.loglog(dt_ref, 0.5*dt_ref, "k:", lw=1, label=r"$\mathcal{O}(\Delta t)$ reference")
    ax.set_xlabel(r"Timestep  $\Delta t$  (s)")
    ax.set_ylabel("Maximum absolute error")
    ax.set_title(r"Convergence: Error vs $\Delta t$  (Free Fall, $t=0.5$ s)")
    ax.legend()
    savefig("02_error_vs_dt")

# ─────────────────────────────────────────────────────────────────────────────
# 03a – Constant velocity position
# ─────────────────────────────────────────────────────────────────────────────
def plot_constvel_position():
    f = "results/constvel.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(v(df.time), v(df.x_exact), "k--", lw=1.5, label="Analytical  $x_0 + v_0 t$")
    ax.plot(v(df.time), v(df.x_num),   color=C[2], lw=1.0, alpha=0.85, label="Numerical")
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("Position  $x$  (m)")
    ax.set_title("Constant Velocity – Position vs Time")
    ax.legend()
    savefig("03a_constvel_position")

# ─────────────────────────────────────────────────────────────────────────────
# 03b – Constant velocity absolute error
# ─────────────────────────────────────────────────────────────────────────────
def plot_constvel_error():
    f = "results/constvel.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(v(df.time), v(df.error_x) + 1e-17, color=C[3], lw=1.2)
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("$|x_{\\rm num} - x_{\\rm exact}|$  (m)")
    ax.set_title("Constant Velocity – Absolute Error vs Time")
    savefig("03b_constvel_error")

# ─────────────────────────────────────────────────────────────────────────────
# 04a – Bounce height vs time
# ─────────────────────────────────────────────────────────────────────────────
def plot_bounce_height():
    fb = "results/bounce.csv"; fp = "results/bounce_peaks.csv"
    if not os.path.exists(fb): return skip(fb)
    df = pd.read_csv(fb)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(v(df.time), v(df.z), color=C[0], lw=0.8)
    if os.path.exists(fp):
        pk = pd.read_csv(fp)
        ax.scatter(v(pk.time), v(pk.peak_z), color=C[1], zorder=5, s=45, label="Peak heights")
        ax.legend()
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("Height  $z$  (m)")
    ax.set_title("Bouncing Particle – Height vs Time")
    savefig("04a_bounce_height")

# ─────────────────────────────────────────────────────────────────────────────
# 04b – Bounce kinetic energy
# ─────────────────────────────────────────────────────────────────────────────
def plot_bounce_ke():
    f = "results/bounce.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(v(df.time), v(df.KE), color=C[4], lw=0.9)
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("Kinetic Energy  (J)")
    ax.set_title("Bouncing Particle – Kinetic Energy vs Time")
    savefig("04b_bounce_ke")

# ─────────────────────────────────────────────────────────────────────────────
# 04c – Rebound peak heights
# ─────────────────────────────────────────────────────────────────────────────
def plot_bounce_peaks():
    f = "results/bounce_peaks.csv"
    if not os.path.exists(f): return skip(f)
    pk = pd.read_csv(f)
    if len(pk) < 2: return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(v(pk.bounce_num), v(pk.peak_z), "o-", color=C[2], lw=1.2)
    ax.set_xlabel("Bounce number")
    ax.set_ylabel("Peak height  (m)")
    ax.set_title("Rebound Height vs Bounce Number")
    savefig("04c_bounce_peaks")

# ─────────────────────────────────────────────────────────────────────────────
# 05 – Kinetic energy multi-particle
# ─────────────────────────────────────────────────────────────────────────────
def plot_energy_multi():
    cases = [("N200","N=200",C[0]), ("N1000","N=1000",C[1]), ("N5000","N=5000",C[2])]
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for suf, lbl, col in cases:
        fpath = f"results/energy_{suf}.csv"
        if not os.path.exists(fpath): continue
        df = pd.read_csv(fpath)
        ax.plot(v(df.time), v(df.KE), lw=0.9, color=col, label=lbl)
        plotted = True
    if not plotted: plt.close(); return skip("energy_N*.csv")
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("Total Kinetic Energy  (J)")
    ax.set_title("Multi-particle Experiments – Kinetic Energy vs Time")
    ax.legend()
    savefig("05_energy_multi")

# ─────────────────────────────────────────────────────────────────────────────
# 06 – Profiling pie chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_profiling_pie():
    f = "results/timing_N5000.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    df = df[df.time_s > 0].copy()
    labels = [nm.replace("_", "\n") for nm in df.function]
    sizes  = v(df.time_s)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140,
           explode=[0.05]*len(sizes), colors=C[:len(sizes)],
           textprops={"fontsize": 9})
    ax.set_title("Serial Runtime Distribution – N=5000 (T=0.5 s, -O3)")
    savefig("06_profiling_pie")

# ─────────────────────────────────────────────────────────────────────────────
# 07a – Total runtime vs N (bar)
# ─────────────────────────────────────────────────────────────────────────────
def plot_runtime_bar():
    f = "results/runtime_table.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    labels = [str(int(n)) for n in df.N]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, v(df.total_s), color=C[0], alpha=0.85)
    for bar, val in zip(bars, v(df.total_s)):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.02,
                f"{val:.2f} s", ha="center", fontsize=9)
    ax.set_xlabel("Number of particles  $N$")
    ax.set_ylabel("Total runtime  (s)")
    ax.set_title("Serial Total Runtime vs Particle Count (T=0.5 s, -O3)")
    savefig("07a_runtime_total")

# ─────────────────────────────────────────────────────────────────────────────
# 07b – Stacked runtime breakdown
# ─────────────────────────────────────────────────────────────────────────────
def plot_runtime_stacked():
    f = "results/runtime_table.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    labels  = [str(int(n)) for n in df.N]
    contact = v(df.contact_s)
    rest    = v(df.total_s) - contact
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, contact, label="compute\_p\_contacts", color=C[1])
    ax.bar(labels, rest, bottom=contact,
           label="All other functions", color=C[0], alpha=0.6)
    ax.set_xlabel("Number of particles  $N$")
    ax.set_ylabel("Runtime  (s)")
    ax.set_title("Runtime Breakdown: Contact Search vs Other (T=0.5 s)")
    ax.legend()
    savefig("07b_runtime_stacked")

    # Print table
    print("\n  Serial Runtime Table (T=0.5 s, -O3):")
    print(f"  {'N':>6}  {'Total (s)':>10}  {'Contact (s)':>12}  {'Contact %':>10}  {'Steps':>7}")
    print("  " + "-"*52)
    for _, row in df.iterrows():
        print(f"  {int(row.N):>6}  {row.total_s:>10.3f}  {row.contact_s:>12.3f}  "
              f"{row.contact_pct:>9.1f}%  {int(row.steps):>7}")

# ─────────────────────────────────────────────────────────────────────────────
# 08a/b/c – Particle configuration snapshots (one file each)
# ─────────────────────────────────────────────────────────────────────────────
def plot_snapshots():
    f = "results/snapshot_N200.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    snaps  = df.snap.unique()
    chosen = [snaps[0], snaps[len(snaps)//2], snaps[-1]]
    names  = ["08a_snap_initial", "08b_snap_mid", "08c_snap_final"]
    titles = ["Initial (t=0)", "Mid-simulation", "Final (settled)"]
    for sid, name, title in zip(chosen, names, titles):
        sub  = df[df.snap == sid]
        tval = float(sub.time.iloc[0])
        fig  = plt.figure(figsize=(5, 4))
        ax   = fig.add_subplot(111, projection="3d")
        ax.scatter(v(sub.x), v(sub.y), v(sub.z), s=8, alpha=0.65, color=C[0])
        ax.set_title(f"{title}  (t = {tval:.2f} s)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        savefig(name)

# ─────────────────────────────────────────────────────────────────────────────
# 09a – Speedup vs threads
# ─────────────────────────────────────────────────────────────────────────────
def plot_speedup_curve():
    cases = [("N200","N=200",C[0]), ("N1000","N=1000",C[1]),("N5000","N=5000",C[2])]
    found = [(suf,lbl,col) for suf,lbl,col in cases
             if os.path.exists(f"results/scaling_{suf}.csv")]
    if not found: return skip("scaling_N*.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    max_p = 1
    for suf, lbl, col in found:
        df = pd.read_csv(f"results/scaling_{suf}.csv")
        ax.plot(v(df.threads), v(df.speedup), "o-", color=col, lw=1.4, label=lbl)
        max_p = max(max_p, int(v(df.threads).max()))
    t_range = np.linspace(1, max_p, 200)
    ax.plot(t_range, t_range, "k--", lw=1, label="Ideal  $S = p$")
    ax.set_xlabel("Number of threads  $p$")
    ax.set_ylabel(r"Speedup  $S(p) = T_1 / T_p$")
    ax.set_title("OpenMP Speedup vs Thread Count")
    ax.legend()
    savefig("09a_speedup")

# ─────────────────────────────────────────────────────────────────────────────
# 09b – Efficiency vs threads
# ─────────────────────────────────────────────────────────────────────────────
def plot_efficiency_curve():
    cases = [("N200","N=200",C[0]), ("N1000","N=1000",C[1]),("N5000","N=5000",C[2])]
    found = [(suf,lbl,col) for suf,lbl,col in cases
             if os.path.exists(f"results/scaling_{suf}.csv")]
    if not found: return skip("scaling_N*.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    for suf, lbl, col in found:
        df = pd.read_csv(f"results/scaling_{suf}.csv")
        ax.plot(v(df.threads), v(df.efficiency), "s-", color=col, lw=1.4, label=lbl)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="Ideal  $E = 1$")
    ax.set_xlabel("Number of threads  $p$")
    ax.set_ylabel(r"Efficiency  $E(p) = S(p)/p$")
    ax.set_title("OpenMP Parallel Efficiency vs Thread Count")
    ax.set_ylim(0, 1.15)
    ax.legend()
    savefig("09b_efficiency")

    for suf, lbl, _ in found:
        df = pd.read_csv(f"results/scaling_{suf}.csv")
        print(f"\n  Scaling Table – {lbl}:")
        print(f"  {'p':>4}  {'T_p (s)':>10}  {'Speedup':>9}  {'Efficiency':>11}")
        print("  " + "-"*40)
        for _, row in df.iterrows():
            print(f"  {int(row.threads):>4}  {row.total_time_s:>10.4f}  "
                  f"{row.speedup:>9.3f}  {row.efficiency:>10.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10a/b – Serial vs parallel KE verification (one per N)
# ─────────────────────────────────────────────────────────────────────────────
def plot_verification():
    cases = [("200", C[0], "10a"), ("1000", C[1], "10b"), ("5000", C[2], "10c")]
    for n, col, prefix in cases:
        fpath = f"results/verify_N{n}.csv"
        if not os.path.exists(fpath): skip(fpath); continue
        df = pd.read_csv(fpath)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(v(df.step_idx), v(df.abs_diff) + 1e-20, color=col, lw=1.0)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("$|KE_{\\rm serial} - KE_{\\rm parallel}|$  (J)")
        ax.set_title(f"Verification N={n}: Serial vs Parallel KE Difference")
        savefig(f"{prefix}_verify_N{n}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Generating plots (one graph per file) ===\n")
    plot_freefall_trajectory()
    plot_freefall_error()
    plot_error_vs_dt()
    plot_constvel_position()
    plot_constvel_error()
    plot_bounce_height()
    plot_bounce_ke()
    plot_bounce_peaks()
    plot_energy_multi()
    plot_profiling_pie()
    plot_runtime_bar()
    plot_runtime_stacked()
    plot_snapshots()
    plot_speedup_curve()
    plot_efficiency_curve()
    plot_verification()
    print("\nDone!  All plots saved to  plots/\n")
