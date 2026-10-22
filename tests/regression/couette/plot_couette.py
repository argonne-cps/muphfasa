#!/usr/bin/env python3
"""plot_couette.py — Run the Couette test and plot the startup transient."""

import argparse
import sys
from pathlib import Path

from helper import COUETTE_DIR, find_one, run_couette_transient_plot


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

    solver_exe = find_one("test_couette*2d*.ex", COUETTE_DIR)

    return run_couette_transient_plot(
        dim=2,
        wall_axis=1,
        bc_key="bc.hiy_Uwall",
        inputs_path=args.inputs,
        run_dir=args.run_dir,
        out_path=args.out,
        solver_exe=solver_exe,
        overrides=args.override,
        no_run=args.no_run,
        max_profile_err=args.max_profile_err,
        max_rate_rel_err=args.max_rate_rel_err,
        steady_label="steady (U0*y/H)",
        profile_title="Couette startup: profile vs. time",
        decay_title_prefix="Mode decay",
        axis_label="y / H",
    )


if __name__ == "__main__":
    sys.exit(main())
