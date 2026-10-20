# Couette flow regression test

Plane Couette flow: periodic in x, stationary wall at y=lo, wall moving at
U0 at y=hi. Checks the simulated velocity profile against the analytic
time-dependent plane-Couette-startup solution, both during the transient and
at steady state. See the comment block at the top of `test_couette.cpp` for
the full derivation.

## Build and run

```sh
make                       # builds test_couette2d.gnu.ex (DIM=2, no MPI)
./test_couette2d.gnu.ex inputs
```

## Expected result

saves `pltXXXXXXX` and `analyticXXXXXXX` which can be compared for accuracy.

Additionally, outputs the error for each writeout step as well as the final steady-state error.

## Plotting

`plot_couette.py` runs the solver with the base `inputs` file (writing the
usual `plt<step>`/`analytic_plt<step>` pairs), laterally averages `vx` in
each pair with AMReX's `faverage` tool to get a `vx(y)` profile, and plots:

1. Velocity profiles `u/U0` vs. `y/H` at each snapshot time -- simulated
   (markers) against the analytic solution (lines), both read straight from
   the solver's own plotfiles.
2. The dominant (n=1) Fourier mode's log-amplitude vs. time, with a fitted
   decay rate compared against the analytic rate `pi^2*nu/H^2`.

```sh
make                    # build test_couette2d.gnu.ex, if not already built
python3 plot_couette.py # builds faverage (DIM=2) on first run, then plots
```

The solver runs in its own `profile_run/` subdirectory (log in `run.log`).
Output is written to `couette_decay.png`. Useful flags: `--inputs <path>`,
`--override sim.max_step=8000 ...` (extra ParmParse overrides), `--no-run`
(re-plot from an existing `profile_run/` without rerunning the solver),
`--out <path>`, `--max-profile-err <tol>` / `--max-rate-rel-err <tol>` (exit
non-zero if the max `|u_sim-u_analytic|/U0` across snapshots, or the fitted
decay rate's relative error against `pi^2*nu/H^2`, exceeds the given
tolerance -- used in CI; the very first snapshot right after the impulsive
start is excluded from the profile-error check since the analytic Fourier
series is still resolving a near step-function there).

## Convergence study

`convergence.py` runs the same problem at several resolutions (Ny = 16, 32,
64 by default) and checks that the error against the analytic solution
shrinks at the expected rate, rather than just checking a single grid.

It pins `lbm.tau` to the same value at every resolution (via a command-line
`ParmParse` override, not a separate inputs file) instead of scaling `Uwall`.
Under this codebase's unit system (`src/LBM/Units.cpp`), the lattice wall
Reynolds number is already resolution-independent as long as `tau` is
pinned; leaving `tau` to its unpinned default instead makes it drift upward
with `Ny` and risks instability at higher resolutions. All runs are also
stopped at the same dimensionless time `nu*t/H^2 = 0.05` (well before steady
state), so the comparison is apples-to-apples across resolutions.

```sh
make                       # build test_couette2d.gnu.ex, if not already built
python3 convergence.py     # builds fcompare (DIM=2) on first run, then runs the study
```

Each resolution runs in its own `convergence_runs/Ny<N>/` subdirectory
(solver log in `run.log`, `fcompare` output in `fcompare.log`). Results are
printed as a table and written to `couette_convergence.csv`
(`Ny, dx, target_step, abs_err_vx, order_vx`), plus a log-log plot of
`abs_err_vx` vs. `dx` against a reference 2nd-order line in
`couette_convergence.png` (requires `matplotlib`; skip with `--no-plot`).

Useful flags: `--ny-list 16 32 64 128`, `--tau 0.8`, `--target-time 0.05`,
`--min-order 1.8` (exit non-zero if the worst observed order falls below
this), `--plot <path>`, `--no-plot`.

`vy` is identically zero for this flow, so its `fcompare` "error" is pure
round-off noise (~1e-15) with no real convergence behavior -- this script
only tracks `vx`.

## CI

`.github/workflows/couette-regression.yml` builds `test_couette2d.gnu.ex`
and runs both scripts on every push/PR touching `src/`,
`tests/regression/couette/`, or the AMReX submodule:
`convergence.py --min-order 2.5` and
`plot_couette.py --max-profile-err 0.01 --max-rate-rel-err 0.01` (both
thresholds leave roughly an order of magnitude of margin over the values
observed locally). Plots and the convergence CSV are uploaded as workflow
artifacts regardless of pass/fail.