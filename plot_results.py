"""
plot_results.py  –  Publication-quality plots for the DEM assignment.
Usage:
    python3 plot_results.py
Run after:  ./particle_sim all
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ── Global style ──────────────────────────────────────────────────────────────
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

# helper: always return numpy array from a pandas column
def v(series):
    return np.asarray(series, dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
# 01 – Free fall: trajectory + error
# ─────────────────────────────────────────────────────────────────────────────
def plot_freefall():
    f = "results/freefall.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(v(df.time), v(df.z_exact), "k--", lw=1.5, label="Analytical  z₀ − ½gt²")
    ax.plot(v(df.time), v(df.z_num),   color=C[0], lw=1.0, alpha=0.85, label="Numerical (Euler)")
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Height  z  (m)")
    ax.set_title("Test 1 – Free Fall: Trajectory"); ax.legend()
    ax = axes[1]
    ax.semilogy(v(df.time), v(df.error_z), color=C[1], lw=1.2)
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("|z_num − z_exact|  (m)")
    ax.set_title("Test 1 – Free Fall: Absolute Error")
    savefig("01_freefall")

# ─────────────────────────────────────────────────────────────────────────────
# 02 – Error vs Δt  (convergence)
# ─────────────────────────────────────────────────────────────────────────────
def plot_error_vs_dt():
    f = "results/error_vs_dt.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(v(df.dt), v(df.max_error_z),  "o-",  color=C[0], label="Max error z")
    ax.loglog(v(df.dt), v(df.max_error_vz), "s--", color=C[1], label="Max error vz")
    dt_ref = np.array([v(df.dt).min(), v(df.dt).max()])
    ax.loglog(dt_ref, 0.5*dt_ref, "k:", lw=1, label=r"$\mathcal{O}(\Delta t)$  reference")
    ax.set_xlabel(r"Timestep  $\Delta t$  (s)")
    ax.set_ylabel("Maximum absolute error")
    ax.set_title(r"Convergence Study: Error vs $\Delta t$  (Free Fall, $t=0.5$ s)")
    ax.legend(); savefig("02_error_vs_dt")

# ─────────────────────────────────────────────────────────────────────────────
# 03 – Constant velocity
# ─────────────────────────────────────────────────────────────────────────────
def plot_constvel():
    f = "results/constvel.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(v(df.time), v(df.x_exact), "k--", lw=1.5, label="Analytical  x₀ + v₀t")
    ax.plot(v(df.time), v(df.x_num),   color=C[2], lw=1.0, alpha=0.85, label="Numerical")
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Position  x  (m)")
    ax.set_title("Test 2 – Constant Velocity: Position"); ax.legend()
    ax = axes[1]
    ax.semilogy(v(df.time), v(df.error_x) + 1e-17, color=C[3], lw=1.2)
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("|x_num − x_exact|  (m)")
    ax.set_title("Test 2 – Constant Velocity: Absolute Error")
    savefig("03_constvel")

# ─────────────────────────────────────────────────────────────────────────────
# 04 – Bouncing particle
# ─────────────────────────────────────────────────────────────────────────────
def plot_bounce():
    fb = "results/bounce.csv"; fp = "results/bounce_peaks.csv"
    if not os.path.exists(fb): return skip(fb)
    df = pd.read_csv(fb)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(v(df.time), v(df.z), color=C[0], lw=0.8)
    if os.path.exists(fp):
        pk = pd.read_csv(fp)
        ax.scatter(v(pk.time), v(pk.peak_z), color=C[1], zorder=5, s=45, label="Peak heights")
        ax.legend()
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Height  z  (m)")
    ax.set_title("Test 3 – Bounce: Height vs Time")
    ax = axes[1]
    ax.plot(v(df.time), v(df.KE), color=C[4], lw=0.9)
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Kinetic Energy  (J)")
    ax.set_title("Test 3 – Bounce: Kinetic Energy")
    savefig("04_bounce")
    if os.path.exists(fp):
        pk = pd.read_csv(fp)
        if len(pk) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(v(pk.bounce_num), v(pk.peak_z), "o-", color=C[2])
            ax.set_xlabel("Bounce number"); ax.set_ylabel("Peak height  (m)")
            ax.set_title("Test 3 – Rebound Height vs Bounce Number")
            savefig("04b_bounce_peaks")

# ─────────────────────────────────────────────────────────────────────────────
# 05 – Kinetic energy – multi-particle experiments
# ─────────────────────────────────────────────────────────────────────────────
def plot_energy_multi():
    cases = [("N200",  C[0], "N=200"),
             ("N1000", C[1], "N=1000"),
             ("N5000", C[2], "N=5000")]
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for suf, col, lbl in cases:
        f = f"results/energy_{suf}.csv"
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        ax.plot(v(df.time), v(df.KE), lw=0.9, color=col, label=lbl)
        plotted = True
    if not plotted: plt.close(); return skip("energy_N*.csv")
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Total Kinetic Energy  (J)")
    ax.set_title("Multi-particle Experiments – Kinetic Energy vs Time")
    ax.legend(); savefig("05_energy_multi")

# ─────────────────────────────────────────────────────────────────────────────
# 06 – Profiling pie chart (N=200)
# ─────────────────────────────────────────────────────────────────────────────
def plot_profiling_pie():
    f = "results/timing_N200.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    df = df[df.time_s > 0].copy()
    labels = [nm.replace("_", "\n") for nm in df.function]
    sizes  = v(df.time_s)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140,
           explode=[0.05]*len(sizes), colors=C[:len(sizes)],
           textprops={"fontsize": 9})
    ax.set_title("Serial Runtime Distribution – N=200 (T=5 s)")
    savefig("06_profiling_pie")

# ─────────────────────────────────────────────────────────────────────────────
# 07 – Runtime table: bar chart + stacked bar
# ─────────────────────────────────────────────────────────────────────────────
def plot_runtime_table():
    f = "results/runtime_table.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    labels = [str(int(n)) for n in df.N]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bars = ax.bar(labels, v(df.total_s), color=C[0], alpha=0.85)
    ax.set_xlabel("Number of particles  N"); ax.set_ylabel("Total runtime  (s)")
    ax.set_title("Serial Total Runtime vs Particle Count")
    for bar, val in zip(bars, v(df.total_s)):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.02,
                f"{val:.2f}s", ha="center", fontsize=9)

    ax = axes[1]
    contact = v(df.contact_s); rest = v(df.total_s) - contact
    ax.bar(labels, contact, label="compute_p_contacts", color=C[1])
    ax.bar(labels, rest, bottom=contact, label="All other functions", color=C[0], alpha=0.6)
    ax.set_xlabel("Number of particles  N"); ax.set_ylabel("Runtime  (s)")
    ax.set_title("Runtime Breakdown: Contact Search vs Rest")
    ax.legend()
    savefig("07_runtime_table")

    print("\n  Serial Runtime Table (T=0.5 s, -O3):")
    print(f"  {'N':>6}  {'Total (s)':>10}  {'Contact (s)':>12}  {'Contact %':>10}  {'Steps':>7}")
    print("  " + "-"*52)
    for _, row in df.iterrows():
        print(f"  {int(row.N):>6}  {row.total_s:>10.3f}  {row.contact_s:>12.3f}  "
              f"{row.contact_pct:>9.1f}%  {int(row.steps):>7}")

# ─────────────────────────────────────────────────────────────────────────────
# 08 – Particle configuration snapshots (N=200)
# ─────────────────────────────────────────────────────────────────────────────
def plot_snapshots():
    f = "results/snapshot_N200.csv"
    if not os.path.exists(f): return skip(f)
    df = pd.read_csv(f)
    snaps = df.snap.unique()
    chosen = [snaps[0], snaps[len(snaps)//2], snaps[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4),
                             subplot_kw={"projection": "3d"})
    titles = ["Initial (t=0)", "Mid-simulation", "Final"]
    for ax, sid, title in zip(axes, chosen, titles):
        sub = df[df.snap == sid]
        tval = float(sub.time.iloc[0])
        ax.scatter(v(sub.x), v(sub.y), v(sub.z), s=6, alpha=0.65, color=C[0])
        ax.set_title(f"{title}\nt = {tval:.2f} s")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle("Particle Configuration Snapshots – N=200", fontsize=12)
    savefig("08_snapshots_N200")

# ─────────────────────────────────────────────────────────────────────────────
# 09 – Speedup and efficiency
# ─────────────────────────────────────────────────────────────────────────────
def plot_speedup():
    cases = [("N200",  C[0], "N=200"),
             ("N1000", C[1], "N=1000")]
    found = [(suf, col, lbl) for suf, col, lbl in cases
             if os.path.exists(f"results/scaling_{suf}.csv")]
    if not found: return skip("scaling_N*.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_s, ax_e = axes[0], axes[1]
    max_p = 1
    for suf, col, lbl in found:
        df = pd.read_csv(f"results/scaling_{suf}.csv")
        ax_s.plot(v(df.threads), v(df.speedup),    "o-", color=col, lw=1.4, label=lbl)
        ax_e.plot(v(df.threads), v(df.efficiency), "s-", color=col, lw=1.4, label=lbl)
        max_p = max(max_p, int(v(df.threads).max()))

    t_range = np.linspace(1, max_p, 200)
    ax_s.plot(t_range, t_range, "k--", lw=1, label="Ideal  S = p")
    ax_e.axhline(1.0, color="k", ls="--", lw=1, label="Ideal  E = 1")

    ax_s.set_xlabel("Number of threads  p")
    ax_s.set_ylabel(r"Speedup  $S(p) = T_1 / T_p$")
    ax_s.set_title("OpenMP Speedup vs Thread Count"); ax_s.legend()

    ax_e.set_xlabel("Number of threads  p")
    ax_e.set_ylabel(r"Parallel Efficiency  $E(p) = S(p)/p$")
    ax_e.set_title("OpenMP Parallel Efficiency"); ax_e.set_ylim(0, 1.15); ax_e.legend()
    savefig("09_omp_speedup")

    for suf, _, lbl in found:
        df = pd.read_csv(f"results/scaling_{suf}.csv")
        print(f"\n  Scaling Table – {lbl}:")
        print(f"  {'p':>4}  {'T_p (s)':>10}  {'Speedup':>9}  {'Efficiency':>11}")
        print("  " + "-"*40)
        for _, row in df.iterrows():
            print(f"  {int(row.threads):>4}  {row.total_time_s:>10.4f}  "
                  f"{row.speedup:>9.3f}  {row.efficiency:>10.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10 – Parallel vs serial KE verification
# ─────────────────────────────────────────────────────────────────────────────
def plot_verification():
    cases = [("200", C[0]), ("1000", C[1])]
    found = [(n, col) for n, col in cases
             if os.path.exists(f"results/verify_N{n}.csv")]
    if not found: return skip("verify_N*.csv")

    fig, axes = plt.subplots(1, len(found), figsize=(6*len(found), 4))
    if len(found) == 1: axes = [axes]
    for ax, (n, col) in zip(axes, found):
        df = pd.read_csv(f"results/verify_N{n}.csv")
        ax.semilogy(v(df.step_idx), v(df.abs_diff) + 1e-20, color=col, lw=1.0)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("|KE_serial − KE_parallel|  (J)")
        ax.set_title(f"Verification N={n}: Serial vs Parallel KE Difference")
    savefig("10_verification")

# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Generating plots ===\n")
    plot_freefall()
    plot_error_vs_dt()
    plot_constvel()
    plot_bounce()
    plot_energy_multi()
    plot_profiling_pie()
    plot_runtime_table()
    plot_snapshots()
    plot_speedup()
    plot_verification()
    print("\nDone!  All plots saved to  plots/\n")
