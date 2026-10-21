#!/usr/bin/env python3
"""plot_couette3d.py — Run the 3D Couette test and plot the startup transient.

End-to-end driver (same style as convergence3d.py): runs test_couette (built with DIM=3,
wall-normal z) with the base `inputs3d` file (writing matching plt<step>/analytic_plt<step>
pairs as it already does), laterally averages vx in each pair with AMReX's `faverage`
tool (which averages along the last dimension, z, matching the wall normal) to get a
vx(z) profile, and renders two panels:
  1. Velocity profiles u(z)/U0 vs z/H at several snapshot times, simulated
     (markers) vs. the analytic time-dependent solution (lines)
  2. log|A1(t)|, the amplitude of the dominant (n=1) Fourier mode vs. time,
     with fitted decay rate vs analytic pi^2*nu/H^2.

Usage:
    cd tests/regression/couette
    make -j6 DIM=3                          # build test_couette3d.gnu.ex, if not already
    python3 plot_couette3d.py               # builds faverage (DIM=3) on first run, then
                                            # runs the solver and plots couette3d_decay.png

Requires numpy + matplotlib (standard library otherwise).
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from helper import (
    COUETTE_DIR,
    apply_overrides,
    ensure_faverage,
    faverage_profile,
    find_one,
    first_mode_amplitude,
    parse_inputs_values,
    parse_units_log,
    run_solver,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=COUETTE_DIR / "inputs3d",
                         help="inputs file to run (default: ./inputs3d)")
    parser.add_argument("--run-dir", type=Path, default=COUETTE_DIR / "profile_run3d",
                         help="scratch directory for the solver run (default: ./profile_run3d)")
    parser.add_argument("--override", nargs="*", default=[],
                         help="extra ParmParse overrides, e.g. --override sim.max_step=8000")
    parser.add_argument("--no-run", action="store_true",
                         help="reuse an existing --run-dir instead of rerunning the solver")
    parser.add_argument("--out", type=Path, default=COUETTE_DIR / "couette3d_decay.png",
                         help="where to write the plot (default: ./couette3d_decay.png)")
    parser.add_argument("--max-profile-err", type=float, default=None,
                         help="if set, exit non-zero when max|u_sim-u_analytic|/U0 "
                              "across all snapshots exceeds this")
    parser.add_argument("--max-rate-rel-err", type=float, default=None,
                         help="if set, exit non-zero when the fitted decay rate's "
                              "relative error against pi^2*nu/H^2 exceeds this")
    args = parser.parse_args()

    solver_exe = find_one("test_couette3d*.ex", COUETTE_DIR)
    faverage_exe = ensure_faverage(dim=3)

    inputs_text = args.inputs.read_text()
    params = apply_overrides(
        {
            "grid.ncells": parse_inputs_values(inputs_text, "grid.ncells"),
            "bc.hiz_Uwall": parse_inputs_values(inputs_text, "bc.hiz_Uwall"),
            "geometry.prob_lo": parse_inputs_values(inputs_text, "geometry.prob_lo"),
            "geometry.prob_hi": parse_inputs_values(inputs_text, "geometry.prob_hi"),
        },
        args.override,
    )
    Nz = int(params["grid.ncells"][2])
    Uwall0 = params["bc.hiz_Uwall"][0]
    z_lo, z_hi = params["geometry.prob_lo"][2], params["geometry.prob_hi"][2]
    domain_height = z_hi - z_lo

    if args.no_run:
        log_text = (args.run_dir / "run.log").read_text()
    else:
        log_text = run_solver(solver_exe, args.inputs.resolve(), args.run_dir, args.override)

    units = parse_units_log(log_text)
    nu_lat = units["nu_lat"]
    U0 = Uwall0 * units["C_t"] / units["C_x"]
    H = Nz

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

        z_phys, u_sim = faverage_profile(faverage_exe, pdir, "vx")
        _, u_analytic = faverage_profile(faverage_exe, analytic_dir, "vx")
        if len(z_phys) != Nz:
            raise RuntimeError(
                f"faverage returned {len(z_phys)} bins but Nz={Nz}: expected wall-normal z "
                "with DIM=3 (faverage averages along z). Check inputs3d has walls in z "
                "and faverage was built with DIM=3."
            )
        z_norm = (z_phys - z_lo) / domain_height
        times.append(step + 1.0)
        profiles.append((z_norm, u_sim, u_analytic))

    times = np.array(times)
    profile_max_err = max(
        (float(np.max(np.abs(u_sim - u_analytic)))
         for _, u_sim, u_analytic in (profiles[1:] or profiles)),
        default=0.0,
    ) / U0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_profiles, ax_decay) = plt.subplots(1, 2, figsize=(11, 4.5))
    n_total = len(times)
    if n_total <= 8:
        show_idx = np.arange(n_total)
    else:
        show_idx = np.unique(np.round(np.geomspace(1, n_total, 7)).astype(int) - 1)
    cmap = plt.get_cmap("viridis")

    for k, idx in enumerate(show_idx):
        t = times[idx]
        z_norm, u_sim, u_analytic = profiles[idx]
        order = np.argsort(z_norm)
        color = cmap(k / max(len(show_idx) - 1, 1))
        ax_profiles.plot(u_sim[order] / U0, z_norm[order], "o", color=color, ms=4,
                         label=f"t={t:.0f}")
        ax_profiles.plot(u_analytic[order] / U0, z_norm[order], "-", color=color, lw=1)

    ax_profiles.plot([0, 1], [0, 1], "k--", lw=1, label="steady (U0*z/H)")
    ax_profiles.set_xlabel("u / U0")
    ax_profiles.set_ylabel("z / H")
    ax_profiles.set_title("3D Couette startup: profile vs. time (z-wall)")
    ax_profiles.legend(fontsize=7, loc="lower right")

    log_amps = []
    for z_norm, u_sim, _ in profiles:
        a1 = first_mode_amplitude(z_norm, u_sim, U0)
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
    ax_decay.set_title(f"3D Mode decay: {title_rate}")
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
