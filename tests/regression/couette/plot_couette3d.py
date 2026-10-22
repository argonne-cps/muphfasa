#!/usr/bin/env python3
"""plot_couette3d.py — Run the 3D Couette test and plot the startup transient."""

import argparse
import sys
from pathlib import Path

from helper import COUETTE_DIR, find_one, run_couette_transient_plot


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
    parser.add_argument("--lattice", type=str, default=None,
                        choices=["D3Q15", "D3Q19", "D3Q27"],
                        help="lbm.lattice_type override (default: from inputs, D3Q19)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the plot (default: ./couette3d[_<lattice>]_decay.png)")
    parser.add_argument("--max-profile-err", type=float, default=None,
                        help="if set, exit non-zero when max|u_sim-u_analytic|/U0 "
                             "across all snapshots exceeds this")
    parser.add_argument("--max-rate-rel-err", type=float, default=None,
                        help="if set, exit non-zero when the fitted decay rate's "
                             "relative error against pi^2*nu/H^2 exceeds this")
    args = parser.parse_args()
    if args.out is None:
        suffix = f"_{args.lattice}" if args.lattice else ""
        args.out = COUETTE_DIR / f"couette3d{suffix}_decay.png"
    if args.lattice and args.run_dir == COUETTE_DIR / "profile_run3d":
        args.run_dir = COUETTE_DIR / f"profile_run3d_{args.lattice}"
    if args.lattice:
        args.override = list(args.override) + [f"lbm.lattice_type={args.lattice}"]

    solver_exe = find_one("test_couette3d*.ex", COUETTE_DIR)

    return run_couette_transient_plot(
        dim=3,
        wall_axis=2,
        bc_key="bc.hiz_Uwall",
        inputs_path=args.inputs,
        run_dir=args.run_dir,
        out_path=args.out,
        solver_exe=solver_exe,
        overrides=args.override,
        no_run=args.no_run,
        max_profile_err=args.max_profile_err,
        max_rate_rel_err=args.max_rate_rel_err,
        steady_label="steady (U0*z/H)",
        profile_title="3D Couette startup: profile vs. time (z-wall)",
        decay_title_prefix="3D Mode decay",
        axis_label="z / H",
    )


if __name__ == "__main__":
    sys.exit(main())
