#!/usr/bin/env python3
"""plot_couette.py — Visualize the Couette-flow startup transient.

Reads couette_profiles.csv (written by test_couette.cpp) and renders two
panels:
  1. Velocity profiles u(y)/U0 vs y/H at several snapshot times, simulated
     (markers) vs. the analytic time-dependent expansion (lines), showing
     the profile relaxing from rest onto the steady linear solution.
  2. log|A1(t)|, the amplitude of the dominant (n=1) Fourier mode, vs. time,
     with a fitted straight line -- its slope is the measured decay rate,
     compared against the analytic rate pi^2*nu/H^2 printed in the legend.

Usage:
    cd tests/regression/couette
    <build & run test_couette so couette_profiles.csv exists>
    python3 plot_couette.py                 # writes couette_decay.png

Requires numpy + matplotlib. If you don't have them on the system Python,
create a venv first:
    python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
    .venv/bin/python plot_couette.py
"""

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


def load(path="couette_profiles.csv"):
    with open(path) as f:
        header = f.readline()
        reader = csv.DictReader(f)
        rows = [
            (float(r["time"]), float(r["y"]), float(r["u_sim"]), float(r["u_analytic"]))
            for r in reader
        ]

    params = {}
    for tok in header.lstrip("#").split():
        key, val = tok.split("=")
        params[key] = float(val)

    data = np.array(rows)
    return params, data


def first_mode_amplitude(y, u_sim, H, U0):
    """Projection of (u_sim - steady) onto sin(pi*y/H); see test_couette.cpp."""
    deviation = u_sim - U0 * y / H
    return np.sum(deviation * np.sin(np.pi * y / H)) * 2.0 / len(y)


def main():
    params, data = load()
    H, U0, nu, Ny = params["H"], params["U0"], params["nu"], int(params["Ny"])
    times = np.unique(data[:, 0])

    fig, (ax_profiles, ax_decay) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel 1: velocity profiles over time ────────────────────────────
    # Geometric spacing, not linear: the transient decays exponentially, so
    # evenly-spaced times mostly show already-converged profiles. Always
    # include the last (steady) snapshot.
    n_show = min(len(times), 7)
    show_idx = np.unique(np.round(
        np.geomspace(1, len(times) - 1, n_show)
    ).astype(int))
    cmap = plt.get_cmap("viridis")

    y_fine = np.linspace(0.5, H - 0.5, 200)
    for k, idx in enumerate(show_idx):
        t = times[idx]
        rows = data[data[:, 0] == t]
        y, u_sim = rows[:, 1], rows[:, 2]
        color = cmap(k / max(n_show - 1, 1))

        ax_profiles.plot(u_sim / U0, y / H, "o", color=color, ms=4,
                         label=f"t={t:.0f}")
        u_an = np.array([
            # same series as analyticCouette() in test_couette.cpp
            yy / H + sum(
                2 * (-1) ** n / (n * np.pi) * np.sin(n * np.pi * yy / H) *
                np.exp(-(n ** 2) * np.pi ** 2 * nu * t / H ** 2)
                for n in range(1, 1001)
            )
            for yy in y_fine
        ])
        ax_profiles.plot(u_an, y_fine / H, "-", color=color, lw=1)

    ax_profiles.plot([0, 1], [0, 1], "k--", lw=1, label="steady (U0*y/H)")
    ax_profiles.set_xlabel("u / U0")
    ax_profiles.set_ylabel("y / H")
    ax_profiles.set_title("Couette startup: profile vs. time")
    ax_profiles.legend(fontsize=7, loc="lower right")

    # ── Panel 2: dominant-mode decay rate ────────────────────────────────
    log_amps = []
    for t in times:
        rows = data[data[:, 0] == t]
        y, u_sim = rows[:, 1], rows[:, 2]
        a1 = first_mode_amplitude(y, u_sim, H, U0)
        log_amps.append(np.log(abs(a1)) if abs(a1) > 1e-12 else np.nan)
    log_amps = np.array(log_amps)

    valid = ~np.isnan(log_amps)
    skip = min(3, valid.sum() // 4)
    fit_idx = np.where(valid)[0][skip:]
    slope, intercept = np.polyfit(times[fit_idx], log_amps[fit_idx], 1)
    analytic_rate = -(np.pi ** 2) * nu / H ** 2

    ax_decay.plot(times[valid], log_amps[valid], "o", ms=4, label="simulation")
    ax_decay.plot(times[fit_idx], slope * times[fit_idx] + intercept, "r-",
                  label=f"fit: rate={slope:.3e}")
    ax_decay.set_xlabel("t")
    ax_decay.set_ylabel("ln|A1(t)|")
    ax_decay.set_title(f"Mode decay: fit={slope:.4e}  analytic={analytic_rate:.4e}")
    ax_decay.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("couette_decay.png", dpi=150)
    print("Wrote couette_decay.png")


if __name__ == "__main__":
    sys.exit(main())
