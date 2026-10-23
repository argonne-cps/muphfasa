#!/usr/bin/env python3
"""plot_tgv2d.py — Run the 2-D Taylor-Green vortex test and plot its decay.

End-to-end driver (same style as tests/regression/couette/plot_couette.py):
runs test_tgv2d with the base `inputs` file (writing plt<step> snapshots as
it already does) and reads the full 2-D density/velocity fields straight
from the solver's own plotfiles with yt (TGV's doubly-periodic field can't
be laterally averaged the way plot_couette.py averages the Couette profile
-- that would erase the vortex signal entirely, so this reads the full
field instead). Renders two figures:
  1. tgv2d_decay.png   — a row of vorticity heatmaps at several snapshot
                         times, visually demonstrating the vortex decaying.
  2. tgv2d_rate.png    — max|u|(t) vs. the analytic envelope
                         u0*exp(-2*nu*k^2*t), log scale, with a fitted decay
                         rate compared to the analytic one.

Usage:
    cd tests/regression/tgv2d
    make                     # build test_tgv2d2d.gnu.ex, if not already
    python3 plot_tgv2d.py    # runs the solver, then plots

Requires numpy + matplotlib + yt (`pip install yt`).
"""

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yt

yt.set_log_level(40)  # suppress yt's per-dataset INFO banner

TGV_DIR = Path(__file__).resolve().parent


def find_one(pattern: str, directory: Path) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {directory}")
    return matches[0]


def parse_inputs_value(text: str, key: str) -> list:
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\n#]+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"could not find {key!r} in inputs")
    return m.group(1).split()


UNITS_LOG_RE = {
    "nu_lat": re.compile(r"^nu_lat:\s*([-+0-9.eE]+)", re.MULTILINE),
    "C_x": re.compile(r"^C_x:\s*([-+0-9.eE]+)", re.MULTILINE),
    "C_t": re.compile(r"^C_t:\s*([-+0-9.eE]+)", re.MULTILINE),
}


def parse_units(log_text: str) -> dict:
    out = {}
    for key, pat in UNITS_LOG_RE.items():
        m = pat.search(log_text)
        if not m:
            raise ValueError(f"could not find '{key}:' in solver log")
        out[key] = float(m.group(1))
    return out


def parse_nu_lat(log_text: str) -> float:
    return parse_units(log_text)["nu_lat"]


def run_solver(solver_exe: Path, inputs_path: Path, run_dir: Path, overrides: list) -> str:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    log_path = run_dir / "run.log"
    with open(log_path, "w") as log_file:
        subprocess.run(
            [str(solver_exe), str(inputs_path), *overrides],
            cwd=run_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return log_path.read_text()


def vorticity(u: np.ndarray, v: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """omega_z = dv/dx - du/dy, central differences, periodic domain."""
    dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dx)
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dx)
    return dvdx - dudy


def read_plotfile_2d(pltdir: Path):
    """Read a 2-D single-level AMReX plotfile via yt, returning
    ({"density"/"vx"/"vy": 2-D array indexed [j, i]}, current_time)."""
    ds = yt.load(str(pltdir))
    dims = ds.domain_dimensions
    grid = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    fields = {
        name: np.asarray(grid["boxlib", name])[:, :, 0].T
        for name in ("density", "vx", "vy")
    }
    return fields, float(ds.current_time)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=TGV_DIR / "inputs",
                         help="inputs file to run (default: ./inputs)")
    parser.add_argument("--run-dir", type=Path, default=TGV_DIR / "profile_run",
                         help="scratch directory for the solver run (default: ./profile_run)")
    parser.add_argument("--override", nargs="*", default=[],
                         help="extra ParmParse overrides, e.g. --override sim.max_step=4000")
    parser.add_argument("--no-run", action="store_true",
                         help="reuse an existing --run-dir instead of rerunning the solver")
    parser.add_argument("--decay-out", type=Path, default=TGV_DIR / "tgv2d_decay.png",
                         help="where to write the vorticity-decay heatmaps")
    parser.add_argument("--rate-out", type=Path, default=TGV_DIR / "tgv2d_rate.png",
                         help="where to write the decay-rate plot")
    parser.add_argument("--n-panels", type=int, default=6,
                         help="number of heatmap snapshots to show (default: 6)")
    parser.add_argument("--max-rate-rel-err", type=float, default=None,
                         help="if set, exit non-zero when the fitted decay rate's "
                              "relative error against u0*exp(-2*nu*k^2*t) exceeds this")
    args = parser.parse_args()

    solver_exe = find_one("test_tgv2d*.ex", TGV_DIR)

    inputs_text = args.inputs.read_text()
    n = int(parse_inputs_value(inputs_text, "grid.ncells")[0])
    u0_phys = float(parse_inputs_value(inputs_text, "units.ref_velocity")[0])
    k = 2.0 * math.pi / n

    if args.no_run:
        log_text = (args.run_dir / "run.log").read_text()
    else:
        log_text = run_solver(solver_exe, args.inputs.resolve(), args.run_dir, args.override)

    units = parse_units(log_text)
    nu_lat = units["nu_lat"]
    # Plotfiles are now physical; convert analytic scales to physical for
    # comparison (vorticity/velocity now in SI units).
    nu_phys = nu_lat * units["C_x"] ** 2 / units["C_t"]
    # k_phys = 2*pi / Lx_phys, Lx_phys = n*C_x; k_lat = 2pi/n so k_phys = k / C_x
    k_phys = k / units["C_x"]

    plt_dirs = sorted(
        (p for p in args.run_dir.glob("plt*") if re.fullmatch(r"plt\d+", p.name)),
        key=lambda p: int(p.name[3:]),
    )
    if not plt_dirs:
        raise FileNotFoundError(f"no plt<step> directories found in {args.run_dir}")

    times, fields = [], []
    for pdir in plt_dirs:
        f, t = read_plotfile_2d(pdir)
        times.append(t)
        fields.append(f)
    times = np.array(times)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Figure 1: vorticity-decay heatmaps ──────────────────────────────
    n_show = min(len(times), args.n_panels)
    show_idx = np.unique(np.round(np.linspace(0, len(times) - 1, n_show)).astype(int))
    dx_phys = units["C_x"]
    vorts = [vorticity(fields[i]["vx"], fields[i]["vy"], dx=dx_phys) for i in show_idx]
    vmax = max(abs(v).max() for v in vorts)

    fig, axes = plt.subplots(1, len(show_idx), figsize=(3 * len(show_idx), 3.2),
                             sharex=True, sharey=True)
    if len(show_idx) == 1:
        axes = [axes]
    for ax, i, vort in zip(axes, show_idx, vorts):
        im = ax.imshow(vort, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"t={times[i]:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="vorticity")
    fig.suptitle("2D Taylor-Green vortex: vorticity decay")
    fig.savefig(args.decay_out, dpi=150)
    print(f"Wrote {args.decay_out}")

    # ── Figure 2: speed-decay rate vs. analytic envelope ────────────────
    max_speed = np.array([
        np.sqrt(f["vx"] ** 2 + f["vy"] ** 2).max() for f in fields
    ])
    analytic_rate = -2.0 * nu_phys * k_phys * k_phys

    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    ax2.semilogy(times, max_speed, "o", ms=4, label="simulation: max|u|")
    ax2.semilogy(times, u0_phys * np.exp(analytic_rate * times), "k--",
                label="analytic: u0*exp(-2*nu*k^2*t)")

    valid = max_speed > 0
    rate_rel_err = None
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(times[valid], np.log(max_speed[valid]), 1)
        rate_rel_err = abs(slope - analytic_rate) / abs(analytic_rate)
        title = f"fit rate={slope:.4e}  analytic={analytic_rate:.4e}  (rel.err={rate_rel_err:.2e})"
    else:
        title = f"analytic rate={analytic_rate:.4e} (too few points to fit)"

    ax2.set_xlabel("t")
    ax2.set_ylabel("max|u|")
    ax2.set_title(title, fontsize=9)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(args.rate_out, dpi=150)
    print(f"Wrote {args.rate_out}")

    if rate_rel_err is not None:
        print(f"decay rate relative error vs analytic: {rate_rel_err:.3e}")

    ok = True
    if args.max_rate_rel_err is not None:
        if rate_rel_err is None:
            print("FAIL: --max-rate-rel-err given but too few points to fit a decay rate")
            ok = False
        elif rate_rel_err > args.max_rate_rel_err:
            print(f"FAIL: decay rate relative error {rate_rel_err:.3e} > "
                  f"--max-rate-rel-err {args.max_rate_rel_err}")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
