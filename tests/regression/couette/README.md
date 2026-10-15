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

`AMREX_HOME` defaults to `../../../external/amrex`; override it if your
AMReX checkout lives elsewhere. This is also wired into the CMake test
suite (`ctest -R regression_couette` from the build directory).

## Expected result

```
Max error vs. steady linear profile: ~1e-10
Max error vs. time-dependent expansion (all snapshots): ~3e-3
Fitted n=1 mode decay rate: ...  analytic: ...  relative error: < 0.2
All Couette regression checks passed.
```

A nonzero exit / thrown exception means one of the three checks failed
(final profile, transient profile, or decay rate) — the printed values above
say which.

## Plotting

```sh
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
./test_couette2d.gnu.ex inputs   # writes couette_profiles.csv
.venv/bin/python plot_couette.py # writes couette_decay.png
```
