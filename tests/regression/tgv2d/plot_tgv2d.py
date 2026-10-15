#!/usr/bin/env python3
"""plot_tgv2d.py — Visualize the 2-D Taylor-Green vortex regression test.

Reads the two CSVs written by test_tgv2d.cpp:
  tgv2d_convergence.csv — (N, dx, L2 velocity error) at three resolutions,
                          holding tau fixed and keeping one vortex
                          wavelength per domain (diffusive-scaling
                          refinement). Rendered as a log-log plot against
                          a reference dx^2 line -> tgv2d_convergence.png.
  tgv2d_snapshots.csv   — the velocity field at a single resolution (N=64)
                          sampled periodically as it decays. Rendered as a
                          row of vorticity-contour snapshots over time ->
                          tgv2d_decay.png.

Usage:
    cd tests/regression/tgv2d
    <build & run test_tgv2d so the CSVs exist>
    python3 plot_tgv2d.py

Requires numpy + matplotlib (see tests/regression/couette/plot_couette.py
for how to set up a venv with both if the system Python doesn't have them).
"""

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence():
    with open("tgv2d_convergence.csv") as f:
        f.readline()  # header comment line (tau/nu/U0), not needed here
        reader = csv.DictReader(f)
        rows = [(int(r["N"]), float(r["dx"]), float(r["error"])) for r in reader]

    N, dx, err = (np.array(x) for x in zip(*rows))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.loglog(dx, err, "o-", label="measured L2 error")

    # Reference 2nd-order line through the first data point.
    ref = err[0] * (dx / dx[0]) ** 2
    ax.loglog(dx, ref, "k--", label="O(dx^2) reference")

    orders = np.log(err[:-1] / err[1:]) / np.log(dx[:-1] / dx[1:])
    title = "TGV2D convergence: observed order " + ", ".join(f"{o:.2f}" for o in orders)
    ax.set_xlabel("dx (= 1/N)")
    ax.set_ylabel("L2 velocity error")
    ax.set_title(title, fontsize=10)
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig("tgv2d_convergence.png", dpi=150)
    print("Wrote tgv2d_convergence.png")


def plot_decay():
    with open("tgv2d_snapshots.csv") as f:
        header = f.readline()
        reader = csv.DictReader(f)
        rows = [
            (float(r["time"]), int(r["i"]), int(r["j"]), float(r["u_sim"]), float(r["v_sim"]))
            for r in reader
        ]

    params = {}
    for tok in header.lstrip("#").split():
        key, val = tok.split("=")
        params[key] = val
    N = int(params["N"])

    data = np.array(rows)
    times = np.unique(data[:, 0])

    def vorticity(t):
        rows_t = data[data[:, 0] == t]
        u = np.zeros((N, N))
        v = np.zeros((N, N))
        for _, i, j, ui, vi in rows_t:
            u[int(j), int(i)] = ui
            v[int(j), int(i)] = vi
        # omega_z = dv/dx - du/dy, central differences, periodic.
        dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / 2.0
        dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / 2.0
        return dvdx - dudy

    n_show = min(len(times), 6)
    show_idx = np.linspace(0, len(times) - 1, n_show).round().astype(int)
    vorts = [vorticity(times[idx]) for idx in show_idx]

    # Shared color scale across all panels, set by the initial (largest
    # magnitude) field -- otherwise each panel autoscales to its own max
    # and the decay becomes invisible.
    vmax = max(abs(v).max() for v in vorts)

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3.2), sharex=True, sharey=True)
    for ax, idx, vort in zip(axes, show_idx, vorts):
        t = times[idx]
        im = ax.imshow(vort, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"t={t:.0f}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.8, label="vorticity")
    fig.suptitle("2D Taylor-Green vortex: vorticity decay")
    fig.savefig("tgv2d_decay.png", dpi=150)
    print("Wrote tgv2d_decay.png")


def main():
    plot_convergence()
    plot_decay()


if __name__ == "__main__":
    sys.exit(main())
