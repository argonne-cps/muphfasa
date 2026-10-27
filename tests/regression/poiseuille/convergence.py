#!/usr/bin/env python3
"""convergence.py — Grid-convergence study for body-force-driven plane Poiseuille flow.

Runs test_poiseuille at several wall-normal resolutions (N = 16, 32, 64 by
default) for the same *physical* problem (forcing.accel and units.* fixed),
pinning lbm.tau at every resolution so nu_lat stays constant as dx shrinks
(F_lat scales like dx^3 automatically). Every run goes to the same
dimensionless time nu*t/H^2 (default 2.5, i.e. steady to ~1e-11), then the
plt<step>/analytic_plt<step> pair written by test_poiseuille.cpp is diffed
with AMReX's fcompare and the observed order of accuracy of the inf-norm
error on vx is fitted across consecutive doublings of N.

Usage:
    cd tests/regression/poiseuille
    make DIM=2                  # builds test_poiseuille2d.gnu.ex
    python3 convergence.py      # 2-D, builds fcompare itself, runs the study
    make DIM=3
    python3 convergence.py --dim 3

Requires only the Python standard library (plus matplotlib for the plot).
"""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

# Reuse the Couette helpers (fcompare cache, parsing); they live next door.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "couette"))
from helper import CS2, ensure_fcompare, find_one, parse_fcompare  # noqa: E402

POISEUILLE_DIR = Path(__file__).resolve().parent


def run_one(n: int, dim: int, tau: float, time_factor: float, lateral: int, lattice,
            solver_exe: Path, base_inputs: Path, fcompare_exe: Path) -> dict:
    nu_lat = (tau - 0.5) * CS2
    target_step = max(1, round(time_factor * n * n / nu_lat))

    work_dir = POISEUILLE_DIR / "convergence_runs" / f"{dim}d_N{n}"
    work_dir.mkdir(parents=True, exist_ok=True)

    lateral_cells = " ".join([str(lateral)] * (dim - 1))
    overrides = [
        f"grid.ncells={lateral_cells} {n}",
        f"grid.max_grid_size={max(n, lateral)}",
        f"lbm.tau={tau}",
        f"sim.max_step={target_step}",
        f"amr.plot_int={target_step}",
        f"amr.chk_int={target_step + 1}",
        "test.max_rel_err=1e30",  # the study measures the error; no pass/fail per run
    ]
    if lattice:
        overrides.append(f"lbm.lattice_type={lattice}")

    log_path = work_dir / "run.log"
    print(f"N={n:4d}  target_step={target_step:7d}  (nu_lat={nu_lat:.4g}) "
          f"-> {work_dir.relative_to(POISEUILLE_DIR)}")
    with open(log_path, "w") as log_file:
        subprocess.run(
            [str(solver_exe), str(base_inputs), *overrides],
            cwd=work_dir, stdout=log_file, stderr=subprocess.STDOUT, check=True,
        )

    step_str = f"{target_step:07d}"
    plt_dir = work_dir / f"plt{step_str}"
    analytic_dir = work_dir / f"analytic_plt{step_str}"
    if not plt_dir.is_dir() or not analytic_dir.is_dir():
        raise FileNotFoundError(
            f"expected {plt_dir.name} and {analytic_dir.name} in {work_dir}; "
            f"see {log_path} for solver output")

    result = subprocess.run([str(fcompare_exe), str(plt_dir), str(analytic_dir)],
                            capture_output=True, text=True)
    errors = parse_fcompare(result.stdout)
    (work_dir / "fcompare.log").write_text(result.stdout)
    if "vx" not in errors:
        raise RuntimeError(f"fcompare output for N={n} had no vx row; see {work_dir / 'fcompare.log'}")

    return {"N": n, "dx": 1.0 / n, "target_step": target_step,
            "abs_err_vx": errors["vx"][0], "rel_err_vx": errors["vx"][1]}


def observed_order(rows: list) -> list:
    for i, row in enumerate(rows):
        if i == 0:
            row["order_vx"] = None
            continue
        prev = rows[i - 1]
        ratio = row["N"] / prev["N"]
        row["order_vx"] = (
            math.log(prev["abs_err_vx"] / row["abs_err_vx"]) / math.log(ratio)
            if row["abs_err_vx"] > 0 and prev["abs_err_vx"] > 0 else None)
    return rows


def plot_convergence(rows: list, plot_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx = [r["dx"] for r in rows]
    err = [r["abs_err_vx"] for r in rows]
    ref = [err[0] * (d / dx[0]) ** 2 for d in dx]
    fig, ax = plt.subplots()
    ax.loglog(dx, err, "o-", label="observed error (vx)")
    ax.loglog(dx, ref, "k--", label="2nd-order reference")
    ax.invert_xaxis()
    ax.set_xlabel("dx")
    ax.set_ylabel("abs_err_vx (inf-norm)")
    ax.set_title("Poiseuille grid convergence")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--n-list", type=int, nargs="+", default=[16, 32, 64],
                        help="wall-normal resolutions, increasing (default: 16 32 64)")
    parser.add_argument("--lateral", type=int, default=4,
                        help="periodic cells in each non-wall-normal direction (default: 4)")
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--time-factor", type=float, default=2.5,
                        help="run length in units of H^2/nu_lat (default: 2.5)")
    parser.add_argument("--base-inputs", type=Path, default=None,
                        help="inputs file (default: ./inputs for --dim 2, ./inputs3d for --dim 3)")
    parser.add_argument("--lattice", type=str, default=None,
                        help="lbm.lattice_type override (3-D: D3Q15/D3Q19/D3Q27)")
    parser.add_argument("--min-order", type=float, default=None,
                        help="exit non-zero if the worst observed order is below this")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    base_inputs = args.base_inputs or POISEUILLE_DIR / ("inputs" if args.dim == 2 else "inputs3d")
    csv_path = args.csv or POISEUILLE_DIR / f"poiseuille{args.dim}d_convergence.csv"
    plot_path = args.plot or POISEUILLE_DIR / f"poiseuille{args.dim}d_convergence.png"

    solver_exe = find_one(f"test_poiseuille*{args.dim}d*.ex", POISEUILLE_DIR)
    fcompare_exe = ensure_fcompare(dim=args.dim)

    rows = []
    for n in sorted(args.n_list):
        rows.append(run_one(n, args.dim, args.tau, args.time_factor, args.lateral,
                            args.lattice, solver_exe, base_inputs, fcompare_exe))
    observed_order(rows)

    header = f"{'N':>6}{'dx':>12}{'step':>9}{'abs_err_vx':>14}{'order_vx':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        ox = f"{r['order_vx']:.2f}" if r["order_vx"] is not None else "   -"
        print(f"{r['N']:>6}{r['dx']:>12.5f}{r['target_step']:>9}{r['abs_err_vx']:>14.6e}{ox:>10}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")
    if not args.no_plot:
        plot_convergence(rows, plot_path)

    if args.min_order is not None:
        worst = min(r["order_vx"] for r in rows if r["order_vx"] is not None)
        if worst < args.min_order:
            print(f"FAIL: worst observed order {worst:.2f} < --min-order {args.min_order}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
