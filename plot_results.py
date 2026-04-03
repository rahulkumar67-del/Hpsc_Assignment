"""
plot_results.py  –  Publication-quality plots for the DEM assignment.
Run after executing all simulation modes:
    ./particle_sim all
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Global style ────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        11,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
os.makedirs("plots", exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"plots/{name}.pdf", bbox_inches="tight")
    plt.savefig(f"plots/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plots/{name}.pdf  +  .png")


# ── 1. Free-fall: numerical vs analytical ─────────────────────────────────────
def plot_freefall():
    f = "results/freefall.csv"
    if not os.path.exists(f):
        print("  [skip] freefall.csv not found"); return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(df.time, df.z_exact, "k--", lw=1.5, label="Analytical")
    ax.plot(df.time, df.z_num,   color=COLORS[0], lw=1, alpha=0.85, label="Numerical")
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("Height  z  (m)")
    ax.set_title("Free Fall – Trajectory")
    ax.legend()

    ax = axes[1]
    ax.semilogy(df.time, df.error_z, color=COLORS[1], lw=1.2)
    ax.set_xlabel("Time  (s)")
    ax.set_ylabel("|z_num − z_exact|  (m)")
    ax.set_title("Free Fall – Absolute Error vs Time")

    savefig("01_freefall")

# ── 2. Error vs timestep ───────────────────────────────────────────────────────
def plot_error_vs_dt():
    f = "results/error_vs_dt.csv"
    if not os.path.exists(f):
        print("  [skip] error_vs_dt.csv not found"); return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(df.dt, df.max_error_z,  "o-", color=COLORS[0], label="Max error z")
    ax.loglog(df.dt, df.max_error_vz, "s--", color=COLORS[1], label="Max error vz")

    # Reference O(dt) line
    dt_ref = np.array([df.dt.min(), df.dt.max()])
    ax.loglog(dt_ref, 0.5*dt_ref, "k:", lw=1, label=r"$\mathcal{O}(\Delta t)$")

    ax.set_xlabel(r"Timestep $\Delta t$  (s)")
    ax.set_ylabel("Maximum absolute error")
    ax.set_title(r"Convergence: Error vs $\Delta t$  (Free Fall, $t=0.5$ s)")
    ax.legend()
    savefig("02_error_vs_dt")

# ── 3. Constant velocity ───────────────────────────────────────────────────────
def plot_constvel():
    f = "results/constvel.csv"
    if not os.path.exists(f):
        print("  [skip] constvel.csv not found"); return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(df.time, df.x_exact, "k--", lw=1.5, label="Analytical")
    ax.plot(df.time, df.x_num,   color=COLORS[2], lw=1, alpha=0.85, label="Numerical")
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Position  x  (m)")
    ax.set_title("Constant Velocity – Position"); ax.legend()

    ax = axes[1]
    ax.semilogy(df.time, df.error_x + 1e-17, color=COLORS[3], lw=1.2)
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("|x_num − x_exact|  (m)")
    ax.set_title("Constant Velocity – Absolute Error")
    savefig("03_constvel")

# ── 4. Bounce ──────────────────────────────────────────────────────────────────
def plot_bounce():
    fb = "results/bounce.csv"
    fp = "results/bounce_peaks.csv"
    if not os.path.exists(fb):
        print("  [skip] bounce.csv not found"); return
    df = pd.read_csv(fb)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(df.time, df.z, color=COLORS[0], lw=0.8, alpha=0.9)
    if os.path.exists(fp):
        pk = pd.read_csv(fp)
        ax.scatter(pk.time, pk.peak_z, color=COLORS[1], zorder=5, s=40, label="Peak heights")
        ax.legend()
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Height  z  (m)")
    ax.set_title("Bouncing Particle – Height vs Time")

    ax = axes[1]
    ax.plot(df.time, df.KE, color=COLORS[4], lw=0.9)
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Kinetic Energy  (J)")
    ax.set_title("Bouncing Particle – Kinetic Energy")
    savefig("04_bounce")

    # Peak heights vs bounce number
    if os.path.exists(fp):
        pk = pd.read_csv(fp)
        if len(pk) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(pk.bounce_num, pk.peak_z, "o-", color=COLORS[2])
            ax.set_xlabel("Bounce number"); ax.set_ylabel("Peak height  (m)")
            ax.set_title("Rebound Height vs Bounce Number")
            savefig("04b_bounce_peaks")

# ── 5. Kinetic energy – multi-particle ────────────────────────────────────────
def plot_energy_multi():
    cases = [("N200", COLORS[0]), ("N1000", COLORS[1]), ("N5000", COLORS[2])]
    plotted = False
    fig, ax = plt.subplots(figsize=(7, 4))
    for suf, col in cases:
        f = f"results/energy_{suf}.csv"
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        ax.plot(df.time, df.KE, lw=0.9, color=col, label=suf.replace("N","N="))
        plotted = True
    if not plotted:
        print("  [skip] no energy_NXxx.csv files found"); plt.close(); return
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Total Kinetic Energy  (J)")
    ax.set_title("Kinetic Energy Evolution – Multi-particle")
    ax.legend()
    savefig("05_energy_multi")

# ── 6. Profiling pie chart (N=200) ────────────────────────────────────────────
def plot_profiling():
    f = "results/timing_N200.csv"
    if not os.path.exists(f):
        print("  [skip] timing_N200.csv not found"); return
    df = pd.read_csv(f)
    df = df[df.time_s > 0]
    labels = df.function.str.replace("_", "\n")
    sizes  = df.time_s.values
    explode = [0.05] * len(sizes)
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        startangle=140, explode=explode,
        colors=COLORS[:len(sizes)],
        textprops={"fontsize": 9}
    )
    ax.set_title("Runtime Distribution – N=200 (Serial)")
    savefig("06_profiling_pie")

# ── 7. Snapshot scatter (N=200, last snap) ────────────────────────────────────
def plot_snapshot():
    f = "results/snapshot_N200.csv"
    if not os.path.exists(f):
        print("  [skip] snapshot_N200.csv not found"); return
    df = pd.read_csv(f)
    snaps = df.snap.unique()
    chosen = [snaps[0], snaps[len(snaps)//2], snaps[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"projection":"3d"})
    titles = ["Initial", "Mid-simulation", "Final"]
    for ax, sid, title in zip(axes, chosen, titles):
        sub = df[df.snap == sid]
        tval = sub.time.iloc[0]
        ax.scatter(sub.x, sub.y, sub.z, s=5, alpha=0.6, color=COLORS[0])
        ax.set_title(f"{title}\nt = {tval:.2f} s")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle("Particle Configurations – N=200", fontsize=12)
    savefig("07_snapshots_N200")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Generating plots ===\n")
    plot_freefall()
    plot_error_vs_dt()
    plot_constvel()
    plot_bounce()
    plot_energy_multi()
    plot_profiling()
    plot_snapshot()
    print("\nDone! All plots saved to plots/\n")
