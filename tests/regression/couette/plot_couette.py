#!/usr/bin/env python3
"""plot_couette.py — Run the Couette test and plot the startup transient.

End-to-end driver (same style as convergence.py): runs test_couette with the
base `inputs` file (writing matching plt<step>/analytic_plt<step> pairs as it
already does), laterally averages vx in each pair with AMReX's `faverage`
tool to get a vx(y) profile, and renders two panels:
  1. Velocity profiles u(y)/U0 vs y/H at several snapshot times, simulated
     (markers) vs. the analytic time-dependent solution (lines) -- both read
     straight from the solver's own plotfiles, so the analytic curve is
     exactly what the regression test compares against.
  2. log|A1(t)|, the amplitude of the dominant (n=1) Fourier mode of the
     simulated profile, vs. time, with a fitted straight line -- its slope is
     the measured decay rate, compared against the analytic rate
     pi^2*nu/H^2 printed in the legend.

Usage:
    cd tests/regression/couette
    make                      # build test_couette2d.gnu.ex, if not already
    python3 plot_couette.py   # builds faverage (DIM=2) on first run, then
                               # runs the solver and plots couette_decay.png

Requires numpy + matplotlib (standard library otherwise).
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

COUETTE_DIR = Path(__file__).resolve().parent
REPO_ROOT = COUETTE_DIR.parents[2]
# Match the GNUmakefile's "AMREX_HOME ?= .../external/amrex": respect an
# AMREX_HOME already set in the environment so faverage is built against the
# same AMReX tree test_couette2d.gnu.ex was linked against.
AMREX_HOME = Path(os.environ.get("AMREX_HOME", REPO_ROOT / "external" / "amrex"))
PLOTFILE_TOOLS_DIR = AMREX_HOME / "Tools" / "Plotfile"


def find_one(pattern: str, directory: Path) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {directory}")
    return matches[0]


def ensure_faverage() -> Path:
    existing = sorted(PLOTFILE_TOOLS_DIR.glob("faverage*.ex"))
    if existing:
        return existing[0]

    print("faverage binary not found -- building it (DIM=2, one-time)...")
    subprocess.run(
        ["make", "DIM=2", "programs=faverage"],
        cwd=PLOTFILE_TOOLS_DIR,
        check=True,
    )
    return find_one("faverage*.ex", PLOTFILE_TOOLS_DIR)


def parse_inputs_values(text: str, key: str) -> list:
    """Whitespace-separated numeric values following `key =` in an inputs
    file (ParmParse syntax), ignoring trailing '#' comments."""
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\n#]+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"could not find {key!r} in inputs")
    return [float(x) for x in m.group(1).split()]


def apply_overrides(values: dict, overrides: list) -> dict:
    """Overlay `key=value` CLI overrides (ParmParse syntax) onto a dict of
    already-parsed inputs-file values, for the keys this script cares about."""
    out = dict(values)
    for ov in overrides:
        key, _, val = ov.partition("=")
        if key in out:
            out[key] = [float(x) for x in val.split()]
    return out


UNITS_LOG_RE = {
    "nu_lat": re.compile(r"^nu_lat:\s*([-+0-9.eE]+)", re.MULTILINE),
    "C_x": re.compile(r"^C_x:\s*([-+0-9.eE]+)", re.MULTILINE),
    "C_t": re.compile(r"^C_t:\s*([-+0-9.eE]+)", re.MULTILINE),
}


def parse_units_log(log_text: str) -> dict:
    """Pull nu_lat/C_x/C_t straight from UnitConverter's own startup banner
    (see src/LBM/Units.cpp) instead of re-deriving them in Python."""
    out = {}
    for key, pat in UNITS_LOG_RE.items():
        m = pat.search(log_text)
        if not m:
            raise ValueError(f"could not find '{key}:' in solver log")
        out[key] = float(m.group(1))
    return out


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


def faverage_profile(faverage_exe: Path, pltdir: Path, varname: str = "vx"):
    """Run faverage on one plotfile and return (height, value) arrays."""
    subprocess.run(
        [str(faverage_exe), "-v", varname, str(pltdir)],
        capture_output=True, text=True, check=True,
    )
    slice_path = Path(str(pltdir) + ".slice")
    data = np.loadtxt(slice_path, comments="#")
    return data[:, 0], data[:, 1]


def first_mode_amplitude(y_norm: np.ndarray, u_sim: np.ndarray, U0: float) -> float:
    """Projection of (u_sim - steady) onto sin(pi*y_norm); see
    analyticCouette()'s n=1 term in test_couette.cpp."""
    deviation = u_sim - U0 * y_norm
    return np.sum(deviation * np.sin(np.pi * y_norm)) * 2.0 / len(y_norm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=COUETTE_DIR / "inputs",
                         help="inputs file to run (default: ./inputs)")
    parser.add_argument("--run-dir", type=Path, default=COUETTE_DIR / "profile_run",
                         help="scratch directory for the solver run (default: ./profile_run)")
    parser.add_argument("--override", nargs="*", default=[],
                         help="extra ParmParse overrides, e.g. --override sim.max_step=8000")
    parser.add_argument("--no-run", action="store_true",
                         help="reuse an existing --run-dir instead of rerunning the solver")
    parser.add_argument("--out", type=Path, default=COUETTE_DIR / "couette_decay.png",
                         help="where to write the plot (default: ./couette_decay.png)")
    parser.add_argument("--max-profile-err", type=float, default=None,
                         help="if set, exit non-zero when max|u_sim-u_analytic|/U0 "
                              "across all snapshots exceeds this")
    parser.add_argument("--max-rate-rel-err", type=float, default=None,
                         help="if set, exit non-zero when the fitted decay rate's "
                              "relative error against pi^2*nu/H^2 exceeds this")
    args = parser.parse_args()

    solver_exe = find_one("test_couette*.ex", COUETTE_DIR)
    faverage_exe = ensure_faverage()

    inputs_text = args.inputs.read_text()
    params = apply_overrides(
        {
            "grid.ncells": parse_inputs_values(inputs_text, "grid.ncells"),
            "bc.hiy_Uwall": parse_inputs_values(inputs_text, "bc.hiy_Uwall"),
            "geometry.prob_lo": parse_inputs_values(inputs_text, "geometry.prob_lo"),
            "geometry.prob_hi": parse_inputs_values(inputs_text, "geometry.prob_hi"),
        },
        args.override,
    )
    Ny = params["grid.ncells"][1]
    Uwall0 = params["bc.hiy_Uwall"][0]
    y_lo, y_hi = params["geometry.prob_lo"][1], params["geometry.prob_hi"][1]
    domain_height = y_hi - y_lo

    if args.no_run:
        log_text = (args.run_dir / "run.log").read_text()
    else:
        log_text = run_solver(solver_exe, args.inputs.resolve(), args.run_dir, args.override)

    units = parse_units_log(log_text)
    nu_lat = units["nu_lat"]
    # Same conversion as UnitConverter::toLatticeVelocity (Units.cpp): the
    # plotfiles store vx in lattice units, so U0 must be too.
    U0 = Uwall0 * units["C_t"] / units["C_x"]
    H = Ny

    plt_dirs = sorted(
        (p for p in args.run_dir.glob("plt*") if re.fullmatch(r"plt\d+", p.name)),
        key=lambda p: int(p.name[3:]),
    )
    if not plt_dirs:
        raise FileNotFoundError(f"no plt<step> directories found in {args.run_dir}")

    times, profiles = [], []
    for pdir in plt_dirs:
        step = int(pdir.name[3:])
        analytic_dir = pdir.with_name(f"analytic_{pdir.name}")
        if not analytic_dir.is_dir():
            continue

        y_phys, u_sim = faverage_profile(faverage_exe, pdir, "vx")
        _, u_analytic = faverage_profile(faverage_exe, analytic_dir, "vx")
        y_norm = (y_phys - y_lo) / domain_height

        # sim->time() is step+1 (m_time starts at 0 and advance() increments
        # it by 1.0 before output -- see SimulationSetup.cpp), matching the
        # time actually passed to calcAnalytic() for this plotfile.
        times.append(step + 1.0)
        profiles.append((y_norm, u_sim, u_analytic))

    times = np.array(times)
    # Skip the very first snapshot: right after the impulsive start, the
    # analytic Fourier series is still resolving a near step-function profile
    # and converges slowly there (Gibbs-like), which dwarfs any real solver
    # error and isn't indicative of solver accuracy.
    profile_max_err = max(
        (float(np.max(np.abs(u_sim - u_analytic)))
         for _, u_sim, u_analytic in (profiles[1:] or profiles)),
        default=0.0,
    ) / U0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_profiles, ax_decay) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel 1: velocity profiles over time ────────────────────────────
    # Geometric spacing, not linear: the transient decays exponentially, so
    # evenly-spaced times mostly show already-converged profiles. With only
    # a handful of snapshots (the common case -- one per amr.plot_int), just
    # show all of them, including the near-rest t=1 one.
    n_total = len(times)
    if n_total <= 8:
        show_idx = np.arange(n_total)
    else:
        show_idx = np.unique(np.round(
            np.geomspace(1, n_total, 7)
        ).astype(int) - 1)
    cmap = plt.get_cmap("viridis")

    for k, idx in enumerate(show_idx):
        t = times[idx]
        y_norm, u_sim, u_analytic = profiles[idx]
        order = np.argsort(y_norm)
        color = cmap(k / max(len(show_idx) - 1, 1))

        ax_profiles.plot(u_sim[order] / U0, y_norm[order], "o", color=color, ms=4,
                         label=f"t={t:.0f}")
        ax_profiles.plot(u_analytic[order] / U0, y_norm[order], "-", color=color, lw=1)

    ax_profiles.plot([0, 1], [0, 1], "k--", lw=1, label="steady (U0*y/H)")
    ax_profiles.set_xlabel("u / U0")
    ax_profiles.set_ylabel("y / H")
    ax_profiles.set_title("Couette startup: profile vs. time")
    ax_profiles.legend(fontsize=7, loc="lower right")

    # ── Panel 2: dominant-mode decay rate ────────────────────────────────
    log_amps = []
    for y_norm, u_sim, _ in profiles:
        a1 = first_mode_amplitude(y_norm, u_sim, U0)
        log_amps.append(np.log(abs(a1)) if abs(a1) > 1e-12 else np.nan)
    log_amps = np.array(log_amps)

    valid = ~np.isnan(log_amps)
    skip = min(3, valid.sum() // 4)
    fit_idx = np.where(valid)[0][skip:]
    analytic_rate = -(np.pi ** 2) * nu_lat / H ** 2

    ax_decay.plot(times[valid], log_amps[valid], "o", ms=4, label="simulation")
    if len(fit_idx) >= 2:
        slope, intercept = np.polyfit(times[fit_idx], log_amps[fit_idx], 1)
        ax_decay.plot(times[fit_idx], slope * times[fit_idx] + intercept, "r-",
                      label=f"fit: rate={slope:.3e}")
        title_rate = f"fit={slope:.4e}  analytic={analytic_rate:.4e}"
    else:
        title_rate = f"analytic={analytic_rate:.4e} (too few points to fit)"
    ax_decay.set_xlabel("t")
    ax_decay.set_ylabel("ln|A1(t)|")
    ax_decay.set_title(f"Mode decay: {title_rate}")
    ax_decay.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    print(f"max|u_sim-u_analytic|/U0 over all snapshots: {profile_max_err:.3e}")
    if len(fit_idx) >= 2:
        rate_rel_err = abs(slope - analytic_rate) / abs(analytic_rate)
        print(f"decay rate relative error vs analytic: {rate_rel_err:.3e}")
    else:
        rate_rel_err = None

    ok = True
    if args.max_profile_err is not None and profile_max_err > args.max_profile_err:
        print(f"FAIL: profile error {profile_max_err:.3e} > --max-profile-err {args.max_profile_err}")
        ok = False
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
