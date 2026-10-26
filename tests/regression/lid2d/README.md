# lid2d validation

Correctness test for the 2-D lid-driven cavity (`../inputs.bench` physics: closed square
cavity, NoSlip x-lo/x-hi/y-lo, `MovingWall` lid at y-hi, Re=100, D2Q9, N=64). The case has
no closed-form solution, so instead of an `fcompare`-vs-analytic test (like `beltrami3d`)
`test_lid2d.cpp` runs the cavity from rest to steady state and checks global invariants
plus three literature values. It prints PASS/FAIL per check and **exits nonzero if any
check fails**.

## Build and run

```sh
make -j6 DIM=2          # -> test_lid2d2d.gnu.ex
../test_lid2d2d.gnu.ex ../inputs                      # ~3 s on one CPU core at N=64
```

`sim.max_step=20000`; `amr.plot_int` (1000) doubles as the diagnostic sampling interval.
Other sizes: override `grid.ncells`, `grid.max_grid_size`, and run longer (steady state
takes roughly `20 * N / u_lat` steps; `u_lat = nu_lat*Re/N`, so N=128 needs ~100000).

## Checks

| # | Check | Pass criterion | Measured (N=64) |
|---|-------|----------------|-----------------|
| 1 | Mass: `sum(rho)` in a closed box | max relative drift < 1e-9 | 1.6e-12 |
| 2 | KE spin-up: `KE = 1/2 sum|u|^2` starts at 0, rises, no dips | KE(0)=0, KE(T)>0, worst relative dip between samples < 1e-3 | monotone |
| 3 | KE plateau (steady state reached) | relative KE change over the last sample interval < 1e-4 | ~1e-7 at step 20000 |
| 4 | Energy balance `P_lid = eps` (below) | `abs(P - eps)/P < 0.20` | 0.135 |
| 5-7 | Ghia centerline extrema u_min, v_max, v_min | relative error < 3% each | 1.4%, 0.4%, 0.6% |

**Energy balance.** The kinetic-energy budget of the cavity is `dKE/dt = P_lid - eps`, where
`P_lid = U_lid * sum_x(nu * du/dy|lid)` is the work done by the lid on the fluid (the other
walls are at rest and do no work) and `eps = nu * sum |grad u|^2`. At steady state they must
be equal. Wall gradients use the halfway-bounce-back geometry (wall half a cell from the
first cell center: a quadratic through wall/first/second cell for the lid shear; ghost
`2*u_wall - u_cell` for the central differences). The tolerance is loose on purpose: the
discontinuous lid velocity at the two top corners makes `|grad u|^2 ~ 1/r^2` there, so both
integrals converge slowly with N. The measured mismatch falls 17.6% -> 13.5% -> 11.0% for
N = 32 / 64 / 128, consistent with that. The check catches gross errors (sign, factor 2,
missing/wrong wall coupling), not percent-level ones.

**Compressibility.** `u_lat = nu_lat*Re/N` with `lbm.tau` pinned, so u_lat=0.1 at N=64 (Ma ~
0.17, Ma^2 ~ 3%). The Ghia comparison therefore carries an O(Ma^2) error of order a percent
on top of the discretization error; the 3% tolerance was set from the measured N=64 and
N=128 results (worst 1.4% and 2.1%), not derived.

## Ghia reference values

Source: U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for incompressible flow using
the Navier-Stokes equations and a multigrid method", *Journal of Computational Physics*
**48**(3), 387-411 (December 1982). DOI:
[10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4).

Re=100 centerline extrema from that paper (velocities normalized by the lid velocity;
Table I = u along the vertical centerline x=0.5, Table II = v along the horizontal
centerline y=0.5; tabulated on their 129x129 grid, so the location is a grid point):

| Quantity | Value | Location in paper |
|----------|-------|-------------------|
| `u_min` | -0.21090 | y = 0.4531 (Table I) |
| `v_max` | +0.17527 | x = 0.2344 (Table II) |
| `v_min` | -0.24533 | x = 0.8047 (Table II) |

The three values were confirmed against the paper's tables by the repository owner. The
bibliographic metadata above was checked against Crossref. The locations in the table are
quoted from memory and are not used by the test.

In the test the extrema are taken from the cell-centered profiles along the centerline (the
two cells straddling x=0.5 / y=0.5 are averaged, N even) and refined with a 3-point
parabola, so the reported location differs slightly from Ghia's grid point (e.g. 0.4587
vs 0.4531 for u_min); only the extremum *value* is compared.

## Expected failure on the current implementation

this test **fails** (mass drains, the flow diverges,
all Ghia checks fail). Root cause is in `muphfasa` halfway bounce-back, not the test: it
reflects diagonal populations from the wrong tangential cell, leaking mass at the ends of
every wall row. 

```
FAIL  mass conservation: max relative drift in sum(rho) = 3.6e-01 (tol 1e-9)
FAIL  Ghia u_min (x=0.5): -0.33757 at 0.4227 vs Ghia -0.21090 (rel err 6.0e-01, tol 3e-2)
```

Until the bounce-back is fixed, `lid2d` throughput
numbers are real but the physics they time is not correct.
