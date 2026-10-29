# lid2d validation

Correctness test for the 2-D lid-driven cavity (same physics as `lid_driven_cavity`:
closed square cavity, NoSlip x-lo/x-hi/y-lo, `MovingWall` lid at y-hi, Re=100,
D2Q9, N=64). The case has no closed-form solution, so instead of an
`fcompare`-vs-analytic test (like `beltrami3d`) `test_lid2d.cpp` runs the
cavity from rest to steady state and checks global invariants plus three centerline
extrema against a converged literature benchmark (the grid-convergence order is checked by
`convergence.py`). It prints PASS/FAIL per check and
**exits nonzero if any check fails**.

## Build and run

```sh
# GNUMake
make -j6 DIM=2          # -> test_lid2d2d.gnu.ex
./test_lid2d2d.gnu.ex inputs                      # ~3 s on one CPU core at N=64

# CMake (from repo root)
cmake -S . -B /tmp/build2d -Dmuphfasa_DIM=2 -Dmuphfasa_BUILD_TESTS=ON
cmake --build /tmp/build2d -j6 --target test_lid2d
mpirun -n 1 /tmp/build2d/tests/test_lid2d tests/regression/lid2d/inputs

# ctest (after CMake configure with BUILD_TESTS=ON)
ctest -R lid2d --output-on-failure
```

`sim.max_step=20000`; `amr.plot_int` (1000) doubles as the diagnostic sampling
interval unless `lid2d.sample_int` is given. Other sizes: override `grid.ncells`,
`grid.max_grid_size`, and run longer (steady state takes roughly `20 * N / u_lat` steps;
`u_lat = nu_lat*Re/N`, so N=128 needs ~100000).

Opt-in diagnostics for the analysis scripts below (default behavior is unchanged):
`lid2d.csv_prefix=<p>` writes `<p>_history.csv` (step, time, mass, ke, power, eps, umin,
vmax, vmin) and `<p>_centerline.csv` (the two centerline profiles); `lid2d.sample_int=<n>`
sets the sampling interval independently of `amr.plot_int`. The macroscopic fields
(`sim->vel()`, `sim->rho()`) are not refreshed by `advanceFused`; the test calls
`updateMacroscopic()` before every sample after t=0 (at t=0 that call would stream the lid
momentum in through the ghost cells, so KE(0)=0 is taken from the initial state).

## Analysis scripts

Both need numpy (matplotlib for plots) and a 2-D `test_lid2d` (`--exe`, `$LID2D_EXE`, or
auto-detected from a GNUmake/CMake build; see `helper.py`).

```sh
python3 plot_lid2d.py --ncells 32 64          # KE / steady-state / energy-budget plots
python3 convergence.py --ncells 32 64 128     # grid convergence (~4 min on one core)
```

* `plot_lid2d.py` runs each N for `--conv-times` convective times (`N/u_lid` steps each,
  default 40) with `lbm.tau` pinned, and plots KE, `|dKE/dT|/KE` (steady-state
  indicator), approach to the final KE, the energy budget `P_lid` vs `eps`, the centerline
  extrema, and mass drift. It prints a SETTLED / NOT SETTLED verdict per N. Use
  `--set KEY=VALUE` to vary any input (e.g. `--set lbm.tau=0.6`) when hunting a case
  where KE does not settle; `--reuse` re-plots existing runs.
* `convergence.py` runs N=32/64/128 (successive powers of two) and reports the
  self-convergence of the extrema, the intensive KE and the full centerline profiles with
  observed orders (**pass/fail**: worst order of the extrema and profiles >= `--min-order`,
  default 1.8) and that the finest grid is within `--max-bench-err` (default 1%) of the
  Marchi et al. benchmark. The distance to the benchmark is also plotted per resolution
  and for the Richardson (p=2) extrapolation. Writes
  `lid2d_convergence.csv/.png`.

Measured (tau=0.692, 40 convective times, steady at the end of every run); errors are
relative to the Marchi et al. values below:

| N | Ma | u_min | v_max | v_min | error (u_min, v_max, v_min) | `|P-eps|/P` |
|---|----|-------|-------|-------|------------------------------|-------------|
| 32 | 0.346 | -0.21362 | 0.17813 | -0.24887 | 0.20%, 0.80%, 1.95% | 0.172 |
| 64 | 0.173 | -0.21393 | 0.17919 | -0.25262 | 0.05%, 0.21%, 0.47% | 0.134 |
| 128 | 0.087 | -0.21401 | 0.17947 | -0.25351 | 0.02%, 0.06%, 0.12% | 0.110 |

Self-convergence order (N=64 to 128): 1.99 / 1.91 / 2.08 for u_min / v_max / v_min and
2.15 / 2.07 for the u(y) / v(x) profiles. The error to the benchmark drops about 4x per
refinement, and the extrapolated extrema (-0.21403, 0.17956, -0.25380) agree with the
benchmark to 4-5 digits, so the solver converges at second order to the true Re=100
solution.

## Checks

| # | Check | Pass criterion | Measured (N=64) |
|---|-------|----------------|-----------------|
| 1 | Mass: `sum(rho)` in a closed box | max relative drift < 1e-9 | 1.6e-12 |
| 2 | KE spin-up: `KE = 1/2 sum|u|^2` starts at 0, rises, no dips | KE(0)=0, KE(T)>0, worst relative dip between samples < 1e-3 | monotone (0 dip) |
| 3 | KE plateau (steady state reached) | relative KE change over the last sample interval < 1e-4 | 7.8e-08 at step 20000 |
| 4 | Energy balance `P_lid = eps` (below) | `abs(P - eps)/P < 0.20` | 0.134 |
| 5-7 | Centerline extrema u_min, v_max, v_min vs Marchi et al. | relative error < 1% each | 0.05%, 0.2%, 0.5% |

**Energy balance.** The kinetic-energy budget of the cavity is `dKE/dt = P_lid - eps`, where
`P_lid = U_lid * sum_x(nu * du/dy|lid)` is the work done by the lid on the fluid (the other
walls are at rest and do no work) and `eps = nu * sum |grad u|^2`. At steady state they must
be equal. Wall gradients use the halfway-bounce-back geometry (wall half a cell from the
first cell center: a quadratic through wall/first/second cell for the lid shear; ghost
`2*u_wall - u_cell` for the central differences). The tolerance is loose on purpose: the
discontinuous lid velocity at the two top corners makes `|grad u|^2 ~ 1/r^2` there, so both
integrals converge slowly with N. The measured mismatch falls 17.6% -> 13.4% -> 11.0% for
N = 32 / 64 / 128, consistent with that. The check catches gross errors (sign, factor 2,
missing/wrong wall coupling), not percent-level ones.

**Compressibility.** `u_lat = nu_lat*Re/N` with `lbm.tau` pinned, so u_lat=0.1 at N=64 (Ma ~
0.17, Ma^2 ~ 3%). Because Ma^2 ~ dx^2 under this scaling, the compressibility error converges
at the same second order as the discretization error, which is what `convergence.py` verifies.

## Benchmark reference values

Source: C. H. Marchi, R. Suero and L. K. Araki, "The lid-driven square cavity flow:
numerical solution with a 1024 x 1024 grid", *Journal of the Brazilian Society of Mechanical
Sciences and Engineering* **31**(3), 186-198 (2009). DOI:
[10.1590/S1678-58782009000300004](https://doi.org/10.1590/S1678-58782009000300004).

Re=100 centerline extrema, grid-converged (extrapolated), normalized by the lid velocity
(u along the vertical centerline x=0.5, v along the horizontal centerline y=0.5); values
supplied by the repository owner:

| Quantity | Value |
|----------|-------|
| `u_min` | -2.140417e-1 |
| `v_max` | +1.79572814e-1 |
| `v_min` | -2.53804e-1 |

These replace the Ghia, Ghia & Shin (1982) values (u_min -0.21090, v_max 0.17527, v_min
-0.24533) used earlier. Ghia's tabulated values are on a 129x129 grid and sit 1.5-3.5% from
the converged solution; the self-convergence study showed our solver converging to the
Marchi values, not Ghia's, with the test's 3% tolerance then failing at N=128.

In the test the extrema are taken from the cell-centered profiles along the centerline (the
two cells straddling x=0.5 / y=0.5 are averaged, N even) and refined with a 3-point
parabola; only the extremum *value* is compared.

## History

Before the diagonal-population bounce-back fix this test failed with mass drain and
divergent flow:

```
FAIL  mass conservation: max relative drift in sum(rho) = 3.6e-01 (tol 1e-9)
FAIL  Ghia u_min (x=0.5): -0.33757 at 0.4227 vs Ghia -0.21090 (rel err 6.0e-01, tol 3e-2)  [old Ghia check]
```

Root cause was the old kernel `f(g,n)=f(g-bound*e_dir,opp(n))` which leaked mass at wall
ends for diagonal `n`. The current `f(g,n)=f(g+c_n,opp(n))` plus edge/corner priority
resolution (`NoSlip > MovingWall > Inflow > Outflow`) fixes it.
