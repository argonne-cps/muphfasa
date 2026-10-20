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