#!/usr/bin/env python3
"""convergence.py — Grid-convergence study for the 2-D lid-driven cavity (Re=100).

Unlike Couette/TGV there is no closed-form solution, so the study reports two
things:

  * PASS/FAIL -- self-convergence: differences between consecutive resolutions
    (N, 2N) of u_min, v_max, v_min and the full centerline profiles u(y), v(x)
    (the fine profile is restricted to the coarse cells by pair-averaging; cell
    centers nest exactly). The observed order is
    p = log2(d(N/2 -> N) / d(N -> 2N)); the exit code is nonzero when the worst
    p falls below --min-order (default 1.8; second order is expected). Needs at
    least three resolutions.
  * PASS/FAIL (finest grid, --max-bench-err, default 1%) and plotted -- distance to the benchmark: relative deviation
    of the extrema from the grid-converged values of Marchi, Suero & Araki
    (2009), doi:10.1590/S1678-58782009000300004, per resolution and for the
    Richardson (p=2) extrapolation from the two finest grids. A converged value
    that still sits off the benchmark indicates a modeling error (e.g. BC
    treatment at the lid corners, collision model) rather than resolution.
    The intensive KE = KE/(N^2 u_lid^2) is also tabulated.

Resolution scaling: lbm.tau is pinned to the same value at every N (diffusive
scaling), so nu_lat is constant and u_lat = nu_lat * Re / N falls as 1/N. Mach
number then falls with N (Ma^2 ~ dx^2) and the compressibility error converges
at the same second order as the discretization error. Every run lasts the same
number of convective times N/u_lid (default 40, well past the ~20 needed to
settle), so all resolutions are compared at steady state.

Usage:
    cd tests/regression/lid2d
    make -j DIM=2                              # or pass --exe <cmake-built test_lid2d>
    python3 convergence.py                     # N = 32 64 128
    python3 convergence.py --ncells 32 64 128 256 --conv-times 30
    python3 convergence.py --reuse             # re-analyze existing runs

Requires numpy (and matplotlib for the plot).
"""

import argparse
import csv
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

SCALARS = ("umin", "vmax", "vmin", "ke_norm")


def run_one(N: int, args, solver_exe: Path, inputs: Path, base_overrides: list) -> dict:
    tau = input_float(inputs, "lbm.tau", base_overrides)
    Re = input_float(inputs, "units.Re", base_overrides)
    u_lat = lattice_lid_velocity(Re, tau, N)
    steps = max(1, math.ceil(args.conv_times * N / u_lat))
    sample_int = max(1, steps // args.samples)

    run_dir = args.run_dir / f"N{N}"
    if not args.reuse:
        overrides = [
            *base_overrides,
            f"grid.ncells={N} {N}",
            f"grid.max_grid_size={N}",
            f"lbm.tau={tau}",
            f"sim.max_step={steps - 1}",  # test_lid2d advances max_step + 1 times
            f"lid2d.sample_int={sample_int}",
            "lid2d.csv_prefix=lid",
            "amr.plot_int=100000000",
            "amr.chk_int=100000000",
        ]
        print(f"N={N:4d}  steps={steps:7d}  u_lid={u_lat:.4g}  Ma={u_lat * math.sqrt(3):.3f}"
              f"  -> {run_dir.relative_to(args.run_dir.parent) if run_dir.is_relative_to(args.run_dir.parent) else run_dir}",
              flush=True)
        rc = run_solver(solver_exe, inputs, run_dir, overrides)
        if rc != 0:
            print(f"  note: test_lid2d exited {rc} (some of its own checks failed; "
                  f"recording results anyway)")

    log_text = (run_dir / "run.log").read_text()
    info = parse_lid_log(log_text)
    hist = read_history(run_dir / "lid_history.csv")
    center = read_centerline(run_dir / "lid_centerline.csv")
    U = info["u_lid"]

    n = len(hist["ke"])
    t_conv = hist["step"] * U / N
    rate = abs(hist["ke"][-1] - hist["ke"][-2]) / hist["ke"][-1] / (t_conv[-1] - t_conv[-2])
    row = {
        "N": N,
        "dx": 1.0 / N,
        "steps": int(hist["step"][-1]),
        "u_lid": U,
        "Ma": U * math.sqrt(3.0),
        "ke_norm": hist["ke"][-1] / (N * N * U * U),
        "umin": hist["umin"][-1],
        "vmax": hist["vmax"][-1],
        "vmin": hist["vmin"][-1],
        "balance": abs(hist["power"][-1] - hist["eps"][-1]) / abs(hist["power"][-1]),
        "ke_rate": rate,
        "mass_drift": float(np.max(np.abs(hist["mass"] - hist["mass"][0])) / hist["mass"][0]),
        "checks_passed": all(ln.startswith("PASS") for ln in parse_check_lines(log_text)),
    }
    for key in ("umin", "vmax", "vmin"):
        row[f"bench_err_{key}"] = abs(row[key] - BENCH[key]) / abs(BENCH[key])
    row["_center"] = center
    return row


def restrict(fine: np.ndarray, factor: int) -> np.ndarray:
    """Average `factor` consecutive fine cells onto each coarse cell."""
    return fine.reshape(-1, factor).mean(axis=1)


def self_convergence(rows: list) -> None:
    """Fill d_<q> (difference to the next-coarser grid), order_<q>, and profile norms."""
    for i, row in enumerate(rows):
        if i == 0:
            continue
        coarse = rows[i - 1]
        f = row["N"] // coarse["N"]
        for q in SCALARS:
            row[f"d_{q}"] = abs(row[q] - coarse[q])
        for prof in ("u_of_y", "v_of_x"):
            diff = restrict(row["_center"][prof], f) - coarse["_center"][prof]
            ref = np.max(np.abs(coarse["_center"][prof]))
            row[f"d_{prof}_L2"] = float(np.sqrt(np.mean(diff**2)) / ref)
            row[f"d_{prof}_Linf"] = float(np.max(np.abs(diff)) / ref)
    for i, row in enumerate(rows):
        if i < 2:
            continue
        prev = rows[i - 1]
        ratio = row["N"] / prev["N"]
        for key in list(row):
            if key.startswith("d_") and prev.get(key, 0) > 0 and row[key] > 0:
                row["order_" + key[2:]] = math.log(prev[key] / row[key]) / math.log(ratio)


def richardson(rows: list) -> dict:
    """p=2 extrapolation of each extremum from the two finest grids."""
    if len(rows) < 2:
        return {}
    fine, coarse = rows[-1], rows[-2]
    out = {}
    for q in ("umin", "vmax", "vmin"):
        star = fine[q] + (fine[q] - coarse[q]) / 3.0
        out[q] = {"value": star, "bench_err": abs(star - BENCH[q]) / abs(BENCH[q])}
    return out


def print_tables(rows: list, extrap: dict) -> None:
    head = (f"{'N':>5}{'steps':>9}{'Ma':>7}{'KE/(N²u²)':>11}"
            f"{'u_min':>10}{'v_max':>10}{'v_min':>10}"
            f"{'err_u':>8}{'err_vmax':>9}{'err_vmin':>9}{'|P-e|/P':>9}{'dKE/KE/T':>10}")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['N']:>5}{r['steps']:>9}{r['Ma']:>7.3f}{r['ke_norm']:>11.5f}"
              f"{r['umin']:>10.5f}{r['vmax']:>10.5f}{r['vmin']:>10.5f}"
              f"{100 * r['bench_err_umin']:>7.2f}%{100 * r['bench_err_vmax']:>8.2f}%"
              f"{100 * r['bench_err_vmin']:>8.2f}%{r['balance']:>9.3f}{r['ke_rate']:>10.1e}")
    print(f"{'bench':>5}{'':>9}{'':>7}{'':>11}"
          f"{BENCH['umin']:>10.5f}{BENCH['vmax']:>10.5f}{BENCH['vmin']:>10.5f}")

    if len(rows) < 2:
        return
    cols = [*(f"d_{q}" for q in SCALARS), "d_u_of_y_L2", "d_v_of_x_L2"]
    names = ["u_min", "v_max", "v_min", "KE_norm", "u(y) L2", "v(x) L2"]
    print("\nSelf-convergence: difference to the next-coarser grid  [observed order]")
    head = f"{'N':>5}" + "".join(f"{n:>18}" for n in names)
    print(head)
    print("-" * len(head))
    for r in rows[1:]:
        cells = []
        for c in cols:
            s = f"{r[c]:.2e}"
            o = r.get("order_" + c[2:])
            s += f" [{o:4.2f}]" if o is not None else "        "
            cells.append(f"{s:>18}")
        print(f"{r['N']:>5}" + "".join(cells))

    print(f"\nRichardson (p=2) extrapolation from N={rows[-2]['N']}, {rows[-1]['N']}:")
    for q, e in extrap.items():
        print(f"  {q}: {e['value']:.5f}  (benchmark {BENCH[q]:.5f}, error {100 * e['bench_err']:.2f}%)")


def plot_convergence(rows: list, extrap: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a_self, a_val) = plt.subplots(1, 2, figsize=(12, 4.8))

    if len(rows) >= 2:
        dx_c = [r["dx"] * 2 for r in rows[1:]]       # coarse-grid spacing of each difference
        d0 = None
        for key, label in (("d_umin", "u_min"), ("d_vmax", "v_max"), ("d_vmin", "v_min"),
                           ("d_ke_norm", "KE norm"), ("d_u_of_y_L2", "u(y) L2"),
                           ("d_v_of_x_L2", "v(x) L2")):
            vals = [r[key] for r in rows[1:]]
            a_self.loglog(dx_c, vals, "o-", label=label)
            d0 = d0 if d0 else (dx_c[0], vals[0])
        ref = [d0[1] * (d / d0[0]) ** 2 for d in dx_c]
        a_self.loglog(dx_c, ref, "k--", label="2nd-order reference")
        a_self.invert_xaxis()
    a_self.set_xlabel("coarse-grid dx of the pair")
    a_self.set_ylabel("|q(N) - q(N/2)|")
    a_self.set_title("self-convergence")
    a_self.legend(fontsize=8)
    a_self.grid(True, which="both", ls=":")

    dx = [r["dx"] for r in rows]
    for q, c in (("umin", "C0"), ("vmax", "C1"), ("vmin", "C2")):
        a_val.loglog(dx, [r[f"bench_err_{q}"] for r in rows], "o-", color=c, label=q)
        if q in extrap:
            a_val.axhline(max(extrap[q]["bench_err"], 1e-6), color=c, ls=":", lw=1)
    ref0 = rows[0]["bench_err_vmin"]
    a_val.loglog(dx, [ref0 * (d / dx[0]) ** 2 for d in dx], "k--", label="2nd-order reference")
    a_val.invert_xaxis()
    a_val.set_xlabel("dx")
    a_val.set_ylabel("relative error vs benchmark")
    a_val.set_title("distance to Marchi et al. (2009); dotted: Richardson-extrapolated")
    a_val.legend(fontsize=8)
    a_val.grid(True, which="both", ls=":")

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
    parser.add_argument("--ncells", type=int, nargs="+", default=[32, 64, 128],
                        help="square resolutions N, each a power-of-two multiple of the first "
                             "(default: 32 64 128)")
    parser.add_argument("--tau", type=float, default=None,
                        help="lbm.tau pinned at every N (default: from inputs)")
    parser.add_argument("--conv-times", type=float, default=40.0,
                        help="run length in convective times N/u_lid (default: 40)")
    parser.add_argument("--samples", type=int, default=100,
                        help="history samples per run (default: 100)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="extra inputs override, e.g. --set lbm.tau=0.6 (repeatable)")
    parser.add_argument("--run-dir", type=Path, default=LID2D_DIR / "convergence_runs",
                        help="where runs go, one N<n>/ subdirectory each")
    parser.add_argument("--reuse", action="store_true",
                        help="don't run the solver; analyze the CSVs already in --run-dir")
    parser.add_argument("--min-order", type=float, default=1.8,
                        help="exit non-zero when the worst observed self-convergence order "
                             "(u_min, v_max, v_min, profiles) falls below this (default: 1.8)")
    parser.add_argument("--max-bench-err", type=float, default=0.01,
                        help="exit non-zero when the finest grid's relative error on any "
                             "extremum vs the benchmark exceeds this (default: 0.01)")
    parser.add_argument("--csv", type=Path, default=LID2D_DIR / "lid2d_convergence.csv",
                        help="where to write the results table")
    parser.add_argument("--plot", type=Path, default=LID2D_DIR / "lid2d_convergence.png",
                        help="where to write the convergence plot")
    parser.add_argument("--no-plot", action="store_true", help="skip the matplotlib plot")
    args = parser.parse_args()

    Ns = sorted(args.ncells)
    for a, b in zip(Ns, Ns[1:]):
        if b % a != 0 or (b // a) & (b // a - 1) != 0:
            parser.error(f"--ncells must be successive power-of-two multiples, got {a} -> {b}")
    if any(N % 2 for N in Ns):
        parser.error("--ncells must be even (centerline falls between cells)")

    overrides = list(args.overrides)
    if args.tau is not None:
        overrides.append(f"lbm.tau={args.tau}")

    solver_exe = None if args.reuse else find_solver(args.exe)
    inputs = args.inputs.resolve()
    args.run_dir = args.run_dir.resolve()

    rows = [run_one(N, args, solver_exe, inputs, overrides) for N in Ns]
    self_convergence(rows)
    extrap = richardson(rows)
    print_tables(rows, extrap)

    keys = [k for k in rows[-1] if not k.startswith("_")]
    for r in rows:
        keys += [k for k in r if not k.startswith("_") and k not in keys]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: v for k, v in r.items() if not k.startswith("_")})
    print(f"\nWrote {args.csv}")

    if not args.no_plot:
        plot_convergence(rows, extrap, args.plot)

    worst_bench = max(rows[-1][f"bench_err_{q}"] for q in ("umin", "vmax", "vmin"))
    bench_ok = worst_bench < args.max_bench_err
    print(f"\n{'PASS' if bench_ok else 'FAIL'}  benchmark: worst extremum error at "
          f"N={rows[-1]['N']} is {100 * worst_bench:.2f}% (require < {100 * args.max_bench_err:g}%)")

    orders = [v for r in rows for k, v in r.items()
              if k.startswith("order_") and k != "order_ke_norm" and not k.endswith("Linf")]
    if not orders:
        print("note: the order check needs at least three resolutions; nothing checked")
        return 0 if bench_ok else 1
    worst = min(orders)
    ok = worst >= args.min_order
    print(f"{'PASS' if ok else 'FAIL'}  self-convergence: worst observed order "
          f"{worst:.2f} (require >= {args.min_order})")
    return 0 if (ok and bench_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
