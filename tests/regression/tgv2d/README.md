# Taylor-Green vortex (2-D) regression test

Doubly-periodic NxN domain seeded with the classic 2-D Taylor-Green vortex:
one counter-rotating vortex pair per side, decaying exponentially under
viscous dissipation. `InitData.cpp` (this directory) overrides the solver's
default quiescent `initData()` to set the exact analytic solution at t=0;
`test_tgv2d.cpp` then compares the solver's evolution against the same
analytic solution at later times. See `TgvAnalytic.H` for the closed-form
density/velocity fields this relies on.

## Build and run

```sh
make                      # builds test_tgv2d2d.gnu.ex (DIM=2, no MPI)
./test_tgv2d2d.gnu.ex inputs
```

## Expected result

Saves `plt<step>` and `analytic_plt<step>` pairs (one every `amr.plot_int`
steps), which can be compared for accuracy. Also prints the max velocity
error at each writeout step and the final error.

## Plotting

`plot_tgv2d.py` runs the solver with the base `inputs` file (writing the
usual `plt<step>` snapshots), reads the full 2-D density/velocity fields
straight from the solver's own plotfiles with `yt` (TGV's doubly-periodic
field can't be laterally averaged the way `plot_couette.py` averages the
Couette profile -- that would erase the vortex signal entirely, so the full
field is read instead), and plots:

1. `tgv2d_decay.png` -- a row of vorticity heatmaps at several snapshot
   times, visually demonstrating the vortex pattern decaying toward zero.
2. `tgv2d_rate.png` -- max|u|(t) vs. the analytic envelope
   `u0*exp(-2*nu*k^2*t)` on a log scale, with a fitted decay rate compared
   against the analytic one.

```sh
make                   # build test_tgv2d2d.gnu.ex, if not already built
pip install yt         # one-time, if not already installed
python3 plot_tgv2d.py  # runs the solver, then plots
```

The solver runs in its own `profile_run/` subdirectory (log in `run.log`).
Useful flags: `--inputs <path>`, `--run-dir <path>`, `--override
sim.max_step=4000 ...` (extra ParmParse overrides), `--no-run` (re-plot from
an existing `--run-dir` without rerunning the solver), `--decay-out <path>`,
`--rate-out <path>`, `--n-panels <n>` (number of heatmap snapshots to show,
default 6), `--max-rate-rel-err <tol>` (exit non-zero if the fitted decay
rate's relative error against `u0*exp(-2*nu*k^2*t)` exceeds the given
tolerance -- used in CI).

## Convergence study

`convergence.py` runs the same problem at several resolutions (N = 32, 64,
128 by default -- the domain is square, one vortex wavelength per side) and
checks that the error against the analytic solution shrinks at the expected
rate, rather than just checking a single grid.

It pins `lbm.tau` to the same value at every resolution (via a command-line
`ParmParse` override), which keeps `nu_lat` constant as the grid is refined.
Since the vortex wavenumber scales as `k = 2*pi/N`, comparing runs
apples-to-apples across resolutions means holding the dimensionless decay
time `nu*k^2*t` fixed -- this is the standard LBM "diffusive scaling"
refinement, so the number of steps needed grows as `N^2`.

Unlike plane Couette, neither velocity component is trivially zero for TGV,
so both `vx` and `vy` errors are tracked.

```sh
make                     # build test_tgv2d2d.gnu.ex, if not already built
python3 convergence.py  # builds fcompare (DIM=2) on first run, then runs the study
```

Each resolution runs in its own `convergence_runs/N<N>/` subdirectory
(solver log in `run.log`, `fcompare` output in `fcompare.log`). Results are
printed as a table and written to `tgv2d_convergence.csv`
(`N, dx, target_step, abs_err_vx, abs_err_vy, rel_err_vx, rel_err_vy, order`),
plus a log-log plot of `abs_err_vx`/`abs_err_vy` vs. `dx` against a
reference 2nd-order line in `tgv2d_convergence.png` (requires `matplotlib`;
skip with `--no-plot`).

Useful flags: `--n-list 32 64 128`, `--tau 0.8`, `--u0 0.02` (overrides
`units.ref_velocity`),
`--target-time 0.1`, `--min-order 1.5` (exit non-zero if the worst observed
order falls below this), `--csv <path>`, `--plot <path>`, `--no-plot`.

## CI

`.github/workflows/tgv2d-regression.yml` builds `test_tgv2d2d.gnu.ex` and
runs the grid-convergence study on every push/PR touching `src/`,
`tests/regression/tgv2d/`, or the AMReX submodule:
`convergence.py --n-list 32 64 128 --min-order 1.2` (leaves roughly an order
of magnitude of margin over the ~1.55-1.95 order observed locally). The
convergence plot and CSV are uploaded as workflow artifacts regardless of
pass/fail. `plot_tgv2d.py` is not run in CI since it requires `yt`; run it
locally to check the decay-rate fit (see "Plotting" above).
