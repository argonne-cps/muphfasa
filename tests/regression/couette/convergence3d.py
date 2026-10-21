#!/usr/bin/env python3
"""convergence3d.py — Grid-convergence study for the 3D plane Couette test.

Runs test_couette (built with DIM=3, wall-normal z) at several resolutions
(Nz = 16, 32, 64 by default), pinning lbm.tau to the same value at every
resolution so nu_lat (and hence the lattice wall Reynolds number) stays
constant as the grid is refined. Every run is stopped at the same
dimensionless time nu*t/H^2 (H=Nz, well before steady state) so the
comparison is apples-to-apples across resolutions.

Each run writes a matching plt<step>/analytic_plt<step> pair (already done
by test_couette.cpp with WALL_DIM=z); this script diffs that pair with
AMReX's fcompare tool and fits the observed order from the inf-norm error
on vx across consecutive doublings of Nz. (vy and vz are identically zero
for plane Couette flow, so their fcompare "error" is pure round-off.)

Usage:
    cd tests/regression/couette
    make -j6 DIM=3                      # build test_couette3d.gnu.ex
    python3 convergence3d.py            # builds fcompare itself, then runs the study

Requires only the Python standard library (except for plotting via matplotlib).
"""

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

from helper import (
    COUETTE_DIR,
    CS2,
    ensure_fcompare,
    find_one,
    parse_fcompare,
)


def _parse_ncells_3d(inputs_path: Path):
    text = inputs_path.read_text()
    m = re.search(r"^\s*grid\.ncells\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"could not find grid.ncells (3 ints) in {inputs_path}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def run_one(nz: int, nx: int, ny: int, tau: float, target_time: float,
            solver_exe: Path, base_inputs: Path, fcompare_exe: Path) -> dict:
    nu_lat = (tau - 0.5) * CS2
    target_step = max(1, round(target_time * nz * nz / nu_lat))

    work_dir = COUETTE_DIR / "convergence_runs3d" / f"Nz{nz}"
    work_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"grid.ncells={nx} {ny} {nz}",
        f"grid.max_grid_size={nz}",
        f"lbm.tau={tau}",
        f"sim.max_step={target_step}",
        f"amr.plot_int={target_step}",
        f"amr.chk_int={target_step + 1}",
    ]

    log_path = work_dir / "run.log"
    print(f"Nz={nz:4d}  Nx={nx:3d}  Ny={ny:3d}  target_step={target_step:6d}  "
          f"(nu_lat={nu_lat:.4g}) -> {work_dir.relative_to(COUETTE_DIR)}")
    with open(log_path, "w") as log_file:
        subprocess.run(
            [str(solver_exe), str(base_inputs), *overrides],
            cwd=work_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    step_str = f"{target_step:07d}"
    plt_dir = work_dir / f"plt{step_str}"
    analytic_dir = work_dir / f"analytic_plt{step_str}"
    if not plt_dir.is_dir() or not analytic_dir.is_dir():
        raise FileNotFoundError(
            f"expected {plt_dir.name} and {analytic_dir.name} in {work_dir}; "
            f"see {log_path} for solver output"
        )

    result = subprocess.run(
        [str(fcompare_exe), str(plt_dir), str(analytic_dir)],
        capture_output=True, text=True,
    )
    errors = parse_fcompare(result.stdout)
    (work_dir / "fcompare.log").write_text(result.stdout)

    if "vx" not in errors:
        raise RuntimeError(
            f"fcompare output for Nz={nz} did not contain a vx row; "
            f"see {work_dir / 'fcompare.log'}"
        )

    return {
        "Nz": nz,
        "dx": 1.0 / nz,
        "target_step": target_step,
        "abs_err_vx": errors["vx"][0],
        "rel_err_vx": errors["vx"][1],
    }


def observed_order(rows: list) -> list:
    for i, row in enumerate(rows):
        if i == 0:
            row["order_vx"] = None
            continue
        prev = rows[i - 1]
        ratio = row["Nz"] / prev["Nz"]
        row["order_vx"] = (
            math.log(prev["abs_err_vx"] / row["abs_err_vx"]) / math.log(ratio)
            if row["abs_err_vx"] > 0 and prev["abs_err_vx"] > 0 else None
        )
    return rows


def plot_convergence(rows: list, plot_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx = [row["dx"] for row in rows]
    err_vx = [row["abs_err_vx"] for row in rows]
    ref = [err_vx[0] * (d / dx[0]) ** 2 for d in dx]

    fig, ax = plt.subplots()
    ax.loglog(dx, err_vx, "o-", label="observed error (vx)")
    ax.loglog(dx, ref, "k--", label="2nd-order reference")
    ax.set_xlabel("dx")
    ax.set_ylabel("abs_err_vx (inf-norm)")
    ax.set_title("3D Couette grid convergence (z-wall)")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nz-list", type=int, nargs="+", default=[16, 32, 64],
                         help="resolutions to run, in increasing order (default: 16 32 64)")
    # Back-compat: --ny-list is alias for --nz-list for wall-normal Nz
    parser.add_argument("--ny-list", type=int, nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tau", type=float, default=0.8,
                         help="lbm.tau pinned at every resolution (default: 0.8)")
    parser.add_argument("--target-time", type=float, default=0.05,
                         help="dimensionless time nu*t/H^2 at which to compare (default: 0.05)")
    parser.add_argument("--base-inputs", type=Path, default=COUETTE_DIR / "inputs3d",
                         help="inputs file to derive bc/units/geometry from (default: ./inputs3d)")
    parser.add_argument("--min-order", type=float, default=None,
                         help="if set, exit non-zero when the worst observed order falls below this")
    parser.add_argument("--csv", type=Path, default=COUETTE_DIR / "couette3d_convergence.csv",
                         help="where to write the results table")
    parser.add_argument("--plot", type=Path, default=COUETTE_DIR / "couette3d_convergence.png",
                         help="where to write the convergence plot (default: ./couette3d_convergence.png)")
    parser.add_argument("--no-plot", action="store_true",
                         help="skip generating the matplotlib plot")
    args = parser.parse_args()
    if args.ny_list is not None:
        args.nz_list = args.ny_list

    solver_exe = find_one("test_couette3d*.ex", COUETTE_DIR)
    fcompare_exe = ensure_fcompare(dim=3)
    nx0, ny0, nz0 = _parse_ncells_3d(args.base_inputs)

    rows = []
    for nz in sorted(args.nz_list):
        nx = max(1, round(nz * nx0 / nz0))
        ny = max(1, round(nz * ny0 / nz0))
        rows.append(run_one(nz, nx, ny, args.tau, args.target_time,
                             solver_exe, args.base_inputs, fcompare_exe))

    observed_order(rows)

    header = f"{'Nz':>6}{'dx':>12}{'step':>8}{'abs_err_vx':>14}{'order_vx':>10}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        ox = f"{row['order_vx']:.2f}" if row["order_vx"] is not None else "   -"
        print(f"{row['Nz']:>6}{row['dx']:>12.5f}{row['target_step']:>8}"
              f"{row['abs_err_vx']:>14.6e}{ox:>10}")

    with open(args.csv, "w", newline="") as f:
        # Keep legacy column name Ny for backwards compat with existing CSVs, but also write Nz
        # New files use Nz; readers should handle either.
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.csv}")

    if not args.no_plot:
        plot_convergence(rows, args.plot)

    if args.min_order is not None:
        worst = min(row["order_vx"] for row in rows if row["order_vx"] is not None)
        if worst < args.min_order:
            print(f"FAIL: worst observed order {worst:.2f} < --min-order {args.min_order}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
