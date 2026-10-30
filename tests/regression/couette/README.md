# Couette flow regression test

Plane Couette flow: periodic in x (and y in 3D), stationary wall at y=lo (2D) or
z=lo (3D) and moving wall at y=hi (2D) or z=hi (3D) at U0. Checks the simulated
velocity profile against the analytic time-dependent plane-Couette-startup
solution, both during the transient and at steady state. See the comment block
at the top of `test_couette.cpp` for the full derivation. The single source
`test_couette.cpp` builds for both 2D (`WALL_DIM=y`) and 3D (`WALL_DIM=z`, so
`faverage` with `DIM=3` averaging along z matches the wall normal).

## Build and run

```sh
# GNUMake (from this directory — defaults to DIM=2, no MPI for quick runs)
make DIM=2                  # -> test_couette2d.gnu.ex
./test_couette2d.gnu.ex inputs

make DIM=3                  # -> test_couette3d.gnu.ex
./test_couette3d.gnu.ex inputs3d

# CMake (from repo root)
cmake -S . -B build2d -Dmuphfasa_DIM=2 -Dmuphfasa_BUILD_TESTS=ON
cmake --build build2d -j$(nproc) --target test_couette
build2d/tests/test_couette tests/regression/couette/inputs
ctest --test-dir build2d -R couette --output-on-failure

# 3-D CMake
cmake -S . -B build -Dmuphfasa_DIM=3 -Dmuphfasa_BUILD_TESTS=ON
cmake --build build -j$(nproc) --target test_couette
build/tests/test_couette tests/regression/couette/inputs3d
```

## Expected result

saves `pltXXXXXXX` and `analytic_pltXXXXXXX` (same `rho`/`vel` layout, `vel` from `calcAnalytic()`) which can be compared with `fcompare`. The solver also prints per-snapshot `|u_sim-u_analytic|∞` and final steady-state error.

## Plotting

`plot_couette.py` (2D, wall-normal y) and `plot_couette3d.py` (3D, wall-normal z)
run the solver with the base inputs, laterally average `vx` with AMReX's `faverage`
to get `vx(y)` in 2D and `vx(z)` in 3D (faverage averages along the last dimension,
so 3D must vary in z), and plot:

1. Velocity profiles `u/U0` vs. `y/H` (2D) or `z/H` (3D) at each snapshot -- simulated (markers) vs analytic (lines).
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

`convergence.py` (2D, Ny) and `convergence3d.py` (3D, Nz wall-normal) run at Ny/Nz=16,32,64
and check error shrinks at expected rate. They pin `lbm.tau` (not `Uwall`) so `nu_lat=(tau-0.5)/3`
and `Re` stay fixed; all runs stop at same `nu*t/H^2=0.05`.

```sh
make DIM=2
python3 convergence.py      # fcompare DIM=2, writes couette_convergence.csv/.png

make DIM=3
python3 convergence3d.py    # fcompare DIM=3, writes couette3d_convergence.csv/.png
```

Each resolution in `convergence_runs/Ny<N>/` (2D) / `convergence_runs3d/Nz<N>/` (3D) with `run.log`/`fcompare.log`. Table columns `Ny/Nz,dx,target_step,abs_err_vx,order_vx`; order ~2 expected. Flags: `--ny-list/--nz-list 16 32 64 128`, `--tau 0.8`, `--target-time 0.05`, `--min-order 1.8`, `--plot/--no-plot`. Only `vx` tracked (`vy`/`vz` ~0).

## CI

`.github/workflows/couette-regression.yml` builds `test_couette2d.gnu.ex` and runs `convergence.py --min-order 1.8` and `plot_couette.py --max-profile-err 0.01 --max-rate-rel-err 0.01`. Extend to 3D by adding `DIM=3` builds of `convergence3d.py`/`plot_couette3d.py`.
