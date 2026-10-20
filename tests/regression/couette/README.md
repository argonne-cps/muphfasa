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

WIP: currently being reworked.

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
(`Ny, dx, target_step, abs_err_vx, order_vx, abs_err_vy, order_vy`), plus a
log-log plot of `abs_err_vx` vs. `dx` against a reference 2nd-order line in
`couette_convergence.png` (requires `matplotlib`; skip with `--no-plot`).

Useful flags: `--ny-list 16 32 64 128`, `--tau 0.8`, `--target-time 0.05`,
`--min-order 1.8` (exit non-zero if the worst observed order falls below
this), `--plot <path>`, `--no-plot`.