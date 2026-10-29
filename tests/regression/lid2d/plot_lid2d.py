#!/usr/bin/env python3
"""plot_lid2d.py — Run the lid-driven cavity and plot how it approaches steady state.

Runs test_lid2d (one run per --ncells value) with dense diagnostic sampling
and plots, against convective time T = step * u_lid / N:

  1. KE / (N^2 u_lid^2)            normalized kinetic energy (intensive, so
                                   different N overlay)
  2. |dKE/dT| / KE                 steady-state indicator; should fall
                                   monotonically toward zero
  3. |KE - KE_end| / KE_end        approach to the final value (log). A clean
                                   decay is a straight line; a floor or a
                                   wiggle means KE never settles
  4. P_lid, eps, dKE/dt            kinetic-energy budget, dKE/dt = P_lid - eps
                                   (lattice units, per step, each divided by
                                   the final P_lid; P_lid and eps must meet at
                                   steady state)
  5. u_min, v_max, v_min / u_lid   centerline extrema vs. Marchi et al. 2009 (dotted)
  6. |m - m0| / m0                 mass drift

and a second figure with the final centerline profiles u(y) at x=0.5 and
v(x) at y=0.5. A text summary per run reports the final steady-state rate and
whether KE is still drifting or oscillating.

Usage:
    cd tests/regression/lid2d
    make -j DIM=2                     # or pass --exe <cmake-built test_lid2d>
    python3 plot_lid2d.py             # N=64, 40 convective times
    python3 plot_lid2d.py --ncells 32 64 128 --conv-times 60
    python3 plot_lid2d.py --ncells 64 --set lbm.tau=0.6   # any inputs override
    python3 plot_lid2d.py --reuse     # re-plot existing runs without re-running

Requires numpy + matplotlib.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from helper import (
    BENCH,
    LID2D_DIR,
    find_solver,
    input_float,
    lattice_lid_velocity,
    parse_check_lines,
    parse_lid_log,
    read_centerline,
    read_history,
    run_solver,
)

# Largest |dKE/dT|/KE (per convective time) we call "settled"
SETTLED_RATE = 1e-4


def run_or_load(N: int, args, solver_exe, inputs: Path, base_overrides: list) -> dict:
    run_dir = args.run_dir / f"N{N}"
    tau = input_float(inputs, "lbm.tau", base_overrides)
    Re = input_float(inputs, "units.Re", base_overrides)
    u_lat = lattice_lid_velocity(Re, tau, N)

    if args.max_step is not None:
        steps = args.max_step
    else:
        steps = max(1, math.ceil(args.conv_times * N / u_lat))
    sample_int = max(1, steps // args.samples)

    if not args.reuse:
        overrides = [
            *base_overrides,
            f"grid.ncells={N} {N}",
            f"grid.max_grid_size={N}",
            f"sim.max_step={steps - 1}",  # test_lid2d advances max_step + 1 times
            f"lid2d.sample_int={sample_int}",
            "lid2d.csv_prefix=lid",
            "amr.plot_int=100000000",
            "amr.chk_int=100000000",
        ]
        print(f"N={N:4d}  steps={steps:7d}  sample_int={sample_int:5d}  "
              f"(u_lid={u_lat:.4g}, tau={tau}) -> {run_dir}")
        rc = run_solver(solver_exe, inputs, run_dir, overrides)
        if rc != 0:
            print(f"  note: test_lid2d exited {rc} (some of its own checks failed; "
                  f"plotting anyway)")

    log_text = (run_dir / "run.log").read_text()
    info = parse_lid_log(log_text)
    return {
        "N": N,
        "tau": tau,
        "u_lid": info["u_lid"],
        "nu": info["nu"],
        "hist": read_history(run_dir / "lid_history.csv"),
        "center": read_centerline(run_dir / "lid_centerline.csv"),
        "checks": parse_check_lines(log_text),
    }


def derived(run: dict) -> dict:
    """Plot quantities from one run's history."""
    h, N, U = run["hist"], run["N"], run["u_lid"]
    T = h["step"] * U / N                       # convective time
    ke_n = h["ke"] / (N * N * U * U)            # intensive KE
    # per-interval KE rate, at interval midpoints
    dstep = np.diff(h["step"])
    dke = np.diff(h["ke"])
    ke_mid = 0.5 * (h["ke"][1:] + h["ke"][:-1])
    T_mid = 0.5 * (T[1:] + T[:-1])
    dke_dt = dke / dstep                        # lattice units, per step
    rate = np.abs(dke) / ke_mid / (np.diff(T))  # relative, per convective time
    power_mid = 0.5 * (h["power"][1:] + h["power"][:-1])
    eps_mid = 0.5 * (h["eps"][1:] + h["eps"][:-1])
    return {
        "T": T, "ke_n": ke_n, "T_mid": T_mid, "dke_dt": dke_dt, "rate": rate,
        "power_mid": power_mid, "eps_mid": eps_mid,
        "approach": np.abs(h["ke"] - h["ke"][-1]) / h["ke"][-1],
        "mass_drift": np.abs(h["mass"] - h["mass"][0]) / h["mass"][0],
    }


def verdict(run: dict, d: dict) -> str:
    """One-paragraph read on whether KE reached steady state."""
    h = run["hist"]
    n = len(d["rate"])
    tail = d["rate"][max(0, n - max(5, n // 5)):]      # last ~20% of the run
    final_rate = float(np.mean(tail))
    # Late-time sign flips of dKE/dt mean KE is oscillating, not relaxing
    # (changes below 1e-9 of KE are round-off, not signal)
    late = np.diff(h["ke"])[n // 2:]
    late = late[np.abs(late) > 1e-9 * h["ke"][-1]]
    flips = int(np.sum(np.sign(late[1:]) * np.sign(late[:-1]) < 0))
    ke_peak = float(np.max(h["ke"]))
    overshoot = (ke_peak - h["ke"][-1]) / h["ke"][-1]
    state = "SETTLED" if final_rate < SETTLED_RATE else "NOT SETTLED"
    msg = (f"N={run['N']}: {state}: late |dKE/dT|/KE = {final_rate:.2e} per convective time "
           f"(settled < {SETTLED_RATE:.0e}); T_end = {d['T'][-1]:.1f}; "
           f"KE_end/(N^2 u^2) = {d['ke_n'][-1]:.5f}; "
           f"KE overshoot over final = {overshoot:.1e}; late dKE/dt sign flips = {flips}")
    if flips > 2 and final_rate >= SETTLED_RATE:
        msg += "  <-- oscillating"
    return msg


def plot_history(runs: list, derivs: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    (a_ke, a_rate), (a_app, a_bud), (a_ext, a_mass) = axs
    colors = [f"C{i}" for i in range(10)]

    for i, (run, d) in enumerate(zip(runs, derivs)):
        c, lab, h = colors[i % 10], f"N={run['N']}", run["hist"]
        a_ke.plot(d["T"], d["ke_n"], color=c, label=lab)
        a_rate.semilogy(d["T_mid"], np.maximum(d["rate"], 1e-18), color=c, label=lab)
        a_app.semilogy(d["T"][:-1], np.maximum(d["approach"][:-1], 1e-18), color=c, label=lab)

        scale = 1.0 / max(abs(h["power"][-1]), 1e-300)   # normalize budget by final P_lid
        a_bud.plot(d["T_mid"], d["power_mid"] * scale, color=c, ls="-", label=f"{lab} P_lid")
        a_bud.plot(d["T_mid"], d["eps_mid"] * scale, color=c, ls="--", label=f"{lab} eps")
        a_bud.plot(d["T_mid"], d["dke_dt"] * scale, color=c, ls=":", label=f"{lab} dKE/dt")

        for key, ls in (("umin", "-"), ("vmax", "--"), ("vmin", ":")):
            a_ext.plot(d["T"], h[key], color=c, ls=ls, label=f"{lab} {key}")
        a_mass.semilogy(d["T"], np.maximum(d["mass_drift"], 1e-17), color=c, label=lab)

    for key, ref in BENCH.items():
        a_ext.axhline(ref, color="k", lw=0.8, ls=(0, (1, 3)))

    a_ke.set_ylabel(r"KE / ($N^2 u_{lid}^2$)")
    a_ke.set_title("normalized kinetic energy")
    a_rate.axhline(SETTLED_RATE, color="k", lw=0.8, ls="--")
    a_rate.set_ylabel(r"|dKE/dT| / KE")
    a_rate.set_title(f"steady-state indicator (dashed: {SETTLED_RATE:.0e})")
    a_app.set_ylabel(r"|KE - KE$_{end}$| / KE$_{end}$")
    a_app.set_title("approach to final KE")
    a_bud.set_ylabel(r"power / final $P_{lid}$")
    a_bud.set_title(r"energy budget: dKE/dt = $P_{lid}$ - $\varepsilon$ (spin-up spike clipped)")
    a_bud.set_ylim(-0.2, 2.0)
    a_ext.set_ylabel(r"centerline extrema / $u_{lid}$")
    a_ext.set_title("u_min, v_max, v_min (dotted black: Marchi et al. 2009)")
    a_mass.set_ylabel(r"|m - m$_0$| / m$_0$")
    a_mass.set_title("mass drift")
    for ax in (a_mass, a_ext):
        ax.set_xlabel(r"convective time $T = t\,u_{lid}/N$")
    for ax in axs.flat:
        ax.grid(True, which="both", ls=":")
    for ax in (a_ke, a_rate, a_app):
        ax.legend()
    a_bud.legend(fontsize=7, ncol=2)
    a_ext.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_centerlines(runs: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a_u, a_v) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, run in enumerate(runs):
        c, lab, cl = f"C{i % 10}", f"N={run['N']}", run["center"]
        a_u.plot(cl["u_of_y"], cl["coord"], color=c, label=lab)
        a_v.plot(cl["coord"], cl["v_of_x"], color=c, label=lab)
    a_u.axvline(BENCH["umin"], color="k", lw=0.8, ls=":", label="Marchi u_min")
    a_v.axhline(BENCH["vmax"], color="k", lw=0.8, ls=":", label="Marchi v_max/v_min")
    a_v.axhline(BENCH["vmin"], color="k", lw=0.8, ls=":")
    a_u.set(xlabel=r"u / $u_{lid}$", ylabel="y", title="u(y) at x = 0.5")
    a_v.set(xlabel="x", ylabel=r"v / $u_{lid}$", title="v(x) at y = 0.5")
    for ax in (a_u, a_v):
        ax.grid(True, ls=":")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=None,
                        help="2-D test_lid2d executable (default: auto-detect, see helper.py)")
    parser.add_argument("--inputs", type=Path, default=LID2D_DIR / "inputs",
                        help="base inputs file (default: ./inputs)")
    parser.add_argument("--ncells", type=int, nargs="+", default=[64],
                        help="resolutions N (square N x N); several overlay (default: 64)")
    parser.add_argument("--conv-times", type=float, default=40.0,
                        help="run length in convective times N/u_lid (default: 40; "
                             "the README's steady state is ~20)")
    parser.add_argument("--max-step", type=int, default=None,
                        help="override: same number of steps for every N")
    parser.add_argument("--samples", type=int, default=400,
                        help="diagnostic samples per run (default: 400)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="extra inputs override, e.g. --set lbm.tau=0.6 (repeatable)")
    parser.add_argument("--run-dir", type=Path, default=LID2D_DIR / "plot_runs",
                        help="where runs go, one N<n>/ subdirectory each (default: ./plot_runs)")
    parser.add_argument("--reuse", action="store_true",
                        help="don't run the solver; plot the CSVs already in --run-dir")
    parser.add_argument("--out", type=Path, default=LID2D_DIR / "lid2d",
                        help="output prefix -> <out>_ke.png, <out>_centerline.png")
    args = parser.parse_args()

    solver_exe = None if args.reuse else find_solver(args.exe)
    runs = [run_or_load(N, args, solver_exe, args.inputs.resolve(), args.overrides)
            for N in sorted(args.ncells)]
    derivs = [derived(r) for r in runs]

    print()
    for run, d in zip(runs, derivs):
        print(verdict(run, d))
        for line in run["checks"]:
            print("    " + line)

    plot_history(runs, derivs, Path(str(args.out) + "_ke.png"))
    plot_centerlines(runs, Path(str(args.out) + "_centerline.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
