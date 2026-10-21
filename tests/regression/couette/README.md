# Couette flow regression test

Plane Couette flow: periodic in x (and z in 3D), stationary wall at y=lo, wall moving at
U0 at y=hi. Checks the simulated velocity profile against the analytic
time-dependent plane-Couette-startup solution, both during the transient and
at steady state. See the comment block at the top of `test_couette.cpp` for
the full derivation. The single source `test_couette.cpp` builds for both
2D and 3D via `AMREX_SPACEDIM`.

## Build and run

```sh
make DIM=2                  # builds test_couette2d.gnu.ex (default)
./test_couette2d.gnu.ex inputs

make DIM=3                  # builds test_couette3d.gnu.ex
./test_couette3d.gnu.ex inputs3d
```

`lbm.lattice_type` must match `DIM`: `D2Q9` for 2D (`inputs`), `D3Q19`/`D3Q27`/`D3Q15` for 3D (`inputs3d` defaults to `D3Q19`). Lattices follow Krüger 2017 Tables 3.4–3.6 (`src/LBM/LatticeDescriptor.H`).

## Expected result

saves `pltXXXXXXX` and `analytic_pltXXXXXXX` (same `rho`/`vel` layout, `vel` from `calcAnalytic()`) which can be compared with `fcompare`. The solver also prints per-snapshot `|u_sim-u_analytic|∞` and final steady-state error.

## Plotting

`plot_couette.py` (2D) and `plot_couette3d.py` (3D) run the solver with the
base inputs, laterally average `vx` with AMReX's `faverage` to get `vx(y)`, and plot:

1. Velocity profiles `u/U0` vs. `y/H` at each snapshot -- simulated (markers) vs analytic (lines).
2. Dominant (n=1) Fourier mode log-amplitude vs. time with fitted decay rate `pi^2*nu/H^2`.

```sh
make DIM=2                  # build test_couette2d.gnu.ex
python3 plot_couette.py     # builds faverage DIM=2, writes couette_decay.png

make DIM=3
python3 plot_couette3d.py   # builds faverage DIM=3, writes couette3d_decay.png
```

Runs in `profile_run/` (2D) / `profile_run3d/` (3D), log `run.log`. Flags: `--inputs <path>`,
`--override sim.max_step=8000 ...`, `--no-run`, `--out <path>`, `--max-profile-err` / `--max-rate-rel-err` (CI thresholds; first snapshot excluded from profile check).

## Convergence study

`convergence.py` (2D) and `convergence3d.py` (3D) run at Ny=16,32,64 and check error shrinks at expected rate. They pin `lbm.tau` (not `Uwall`) so `nu_lat=(tau-0.5)/3` and `Re` stay fixed; all runs stop at same `nu*t/H^2=0.05`.

```sh
make DIM=2
python3 convergence.py      # fcompare DIM=2, writes couette_convergence.csv/.png

make DIM=3
python3 convergence3d.py    # fcompare DIM=3, writes couette3d_convergence.csv/.png
```

Each resolution in `convergence_runs/Ny<N>/` (2D) / `convergence_runs3d/Ny<N>/` (3D) with `run.log`/`fcompare.log`. Table columns `Ny,dx,target_step,abs_err_vx,order_vx`; order ~2 expected. Flags: `--ny-list 16 32 64 128`, `--tau 0.8`, `--target-time 0.05`, `--min-order 1.8`, `--plot/--no-plot`. Only `vx` tracked (`vy`/`vz` ~0).

## 3D contribution options

- Lattice: `D3Q19` (default, 19 vel), `D3Q27` (full, more isotropic), `D3Q15` (light). Set `lbm.lattice_type` and `DIM=3`; ensure `grid.ncells` includes `Nz` and BCs have 3 entries (`Periodic NoSlip Periodic`).
- Extending: add new `D3Qxx` in `LatticeDescriptor.H` (Krueger ordering), update `SimulationFactory`, and add `inputs*` variant. Verify with `convergence3d.py`.

## CI

`.github/workflows/couette-regression.yml` builds `test_couette2d.gnu.ex` and runs `convergence.py --min-order 2.5` and `plot_couette.py --max-profile-err 0.01 --max-rate-rel-err 0.01`. Extend to 3D by adding `DIM=3` builds of `convergence3d.py`/`plot_couette3d.py`.
