#!/usr/bin/env python3
"""convergence.py — Grid-convergence study for the 2-D Taylor-Green vortex.

Runs test_tgv2d at several resolutions (N = 32, 64, 128 by default),
pinning lbm.tau to the same value at every resolution so nu_lat stays
constant as the grid is refined (same approach as
tests/regression/couette/convergence.py). Each run keeps exactly one vortex
wavelength per domain (k=2*pi/N), so to compare apples-to-apples across
resolutions every run is stopped at the same dimensionless decay time
nu*k^2*t -- since k^2 ~ 1/N^2, this is the standard LBM "diffusive scaling"
refinement: the number of steps needed grows as N^2.

Each run writes a matching plt<step>/analytic_plt<step> pair (already done
by test_tgv2d.cpp); this script diffs that pair with AMReX's fcompare tool
and fits the observed order of accuracy from the inf-norm error on vx and vy
across consecutive doublings of N. (Unlike plane Couette, neither velocity
component is trivially zero for TGV, so both are tracked.)

Usage:
    cd tests/regression/tgv2d
    make                       # build test_tgv2d2d.gnu.ex, if not already
    python3 convergence.py     # builds fcompare itself, then runs the study

Requires only the Python standard library (except for plotting via matplotlib).
"""

import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
from pathlib import Path

CS2 = 1.0 / 3.0  # lattice sound speed squared (D2Q9 / D3Q19, see LatticeDescriptor)

TGV_DIR = Path(__file__).resolve().parent
REPO_ROOT = TGV_DIR.parents[2]
AMREX_HOME = Path(os.environ.get("AMREX_HOME", REPO_ROOT / "external" / "amrex"))
PLOTFILE_TOOLS_DIR = AMREX_HOME / "Tools" / "Plotfile"


def find_one(pattern: str, directory: Path) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {directory}")
    return matches[0]


def ensure_fcompare() -> Path:
    existing = sorted(PLOTFILE_TOOLS_DIR.glob("fcompare*.ex"))
    if existing:
        return existing[0]

    print("fcompare binary not found -- building it (DIM=2, one-time)...")
    subprocess.run(
        ["make", "DIM=2", "programs=fcompare"],
        cwd=PLOTFILE_TOOLS_DIR,
        check=True,
    )
    return find_one("fcompare*.ex", PLOTFILE_TOOLS_DIR)


FCOMPARE_ROW_RE = re.compile(
    r"^\s*(\S+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)


def parse_fcompare(output: str) -> dict:
    """Map variable name -> (abs_err, rel_err) from fcompare's stdout table."""
    errors = {}
    for line in output.splitlines():
        m = FCOMPARE_ROW_RE.match(line)
        if m and m.group(1) in ("density", "vx", "vy", "vz"):
            errors[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return errors


def run_one(n: int, tau: float, u0: float, target_time: float,
            solver_exe: Path, base_inputs: Path, fcompare_exe: Path) -> dict:
    nu_lat = (tau - 0.5) * CS2
    k = 2.0 * math.pi / n
    target_step = max(1, round(target_time / (nu_lat * k * k)))

    work_dir = TGV_DIR / "convergence_runs" / f"N{n}"
    work_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"grid.ncells={n} {n}",
        f"grid.max_grid_size={n}",
        f"lbm.tau={tau}",
        f"units.ref_velocity={u0}",
        f"sim.max_step={target_step}",
        f"amr.plot_int={target_step}",
        f"amr.chk_int={target_step + 1}",
    ]

    log_path = work_dir / "run.log"
    print(f"N={n:4d}  target_step={target_step:7d}  (nu_lat={nu_lat:.4g}) "
          f"-> {work_dir.relative_to(TGV_DIR)}")
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

    # fcompare exits non-zero whenever the (default zero) tolerances aren't
    # met, which is every run here -- the error, not the exit code, is what
    # we're after, so don't check=True.
    result = subprocess.run(
        [str(fcompare_exe), str(plt_dir), str(analytic_dir)],
        capture_output=True, text=True,
    )
    errors = parse_fcompare(result.stdout)
    (work_dir / "fcompare.log").write_text(result.stdout)

    if "vx" not in errors or "vy" not in errors:
        raise RuntimeError(
            f"fcompare output for N={n} did not contain vx/vy rows; "
            f"see {work_dir / 'fcompare.log'}"
        )

    return {
        "N": n,
        "dx": 1.0 / n,
        "target_step": target_step,
        "abs_err_vx": errors["vx"][0],
        "abs_err_vy": errors["vy"][0],
        "rel_err_vx": errors["vx"][1],
        "rel_err_vy": errors["vy"][1],
    }


def observed_order(rows: list) -> list:
    """Append an order-of-accuracy estimate between consecutive rows,
    assuming each N is double the previous one, using the combined (worse
    of vx/vy) error so one row's order reflects the slower-converging
    component."""
    for row in rows:
        row["err"] = max(row["abs_err_vx"], row["abs_err_vy"])
    for i, row in enumerate(rows):
        if i == 0:
            row["order"] = None
            continue
        prev = rows[i - 1]
        ratio = row["N"] / prev["N"]
        row["order"] = (
            math.log(prev["err"] / row["err"]) / math.log(ratio)
            if row["err"] > 0 and prev["err"] > 0 else None
        )
    return rows


def plot_convergence(rows: list, plot_path: Path) -> None:
    """Log-log plot of abs_err_vx/vy vs dx against a reference 2nd-order line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx = [row["dx"] for row in rows]
    err_vx = [row["abs_err_vx"] for row in rows]
    err_vy = [row["abs_err_vy"] for row in rows]

    ref = [err_vx[0] * (d / dx[0]) ** 2 for d in dx]

    fig, ax = plt.subplots()
    ax.loglog(dx, err_vx, "o-", label="observed error (vx)")
    ax.loglog(dx, err_vy, "s-", label="observed error (vy)")
    ax.loglog(dx, ref, "k--", label="2nd-order reference")
    ax.set_xlabel("dx")
    ax.set_ylabel("abs error (inf-norm)")
    ax.set_title("TGV2D grid convergence")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-list", type=int, nargs="+", default=[32, 64, 128],
                         help="resolutions to run, in increasing order (default: 32 64 128)")
    parser.add_argument("--tau", type=float, default=0.8,
                         help="lbm.tau pinned at every resolution (default: 0.8)")
    parser.add_argument("--u0", type=float, default=0.02,
                         help="units.ref_velocity vortex amplitude, lattice units (default: 0.02)")
    parser.add_argument("--target-time", type=float, default=0.1,
                         help="dimensionless decay time nu*k^2*t at which to compare (default: 0.1)")
    parser.add_argument("--base-inputs", type=Path, default=TGV_DIR / "inputs",
                         help="inputs file to derive geometry from (default: ./inputs)")
    parser.add_argument("--min-order", type=float, default=None,
                         help="if set, exit non-zero when the worst observed order falls below this")
    parser.add_argument("--csv", type=Path, default=TGV_DIR / "tgv2d_convergence.csv",
                         help="where to write the results table")
    parser.add_argument("--plot", type=Path, default=TGV_DIR / "tgv2d_convergence.png",
                         help="where to write the convergence plot (default: ./tgv2d_convergence.png)")
    parser.add_argument("--no-plot", action="store_true",
                         help="skip generating the matplotlib plot")
    args = parser.parse_args()

    solver_exe = find_one("test_tgv2d*.ex", TGV_DIR)
    fcompare_exe = ensure_fcompare()

    rows = []
    for n in sorted(args.n_list):
        rows.append(run_one(n, args.tau, args.u0, args.target_time,
                             solver_exe, args.base_inputs, fcompare_exe))

    observed_order(rows)

    header = f"{'N':>6}{'dx':>12}{'step':>8}{'abs_err_vx':>14}{'abs_err_vy':>14}{'order':>8}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        o = f"{row['order']:.2f}" if row["order"] is not None else "   -"
        print(f"{row['N']:>6}{row['dx']:>12.5f}{row['target_step']:>8}"
              f"{row['abs_err_vx']:>14.6e}{row['abs_err_vy']:>14.6e}{o:>8}")

    with open(args.csv, "w", newline="") as f:
        fieldnames = [k for k in rows[0].keys() if k != "err"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\nWrote {args.csv}")

    if not args.no_plot:
        plot_convergence(rows, args.plot)

    if args.min_order is not None:
        worst = min(row["order"] for row in rows if row["order"] is not None)
        if worst < args.min_order:
            print(f"FAIL: worst observed order {worst:.2f} < --min-order {args.min_order}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
