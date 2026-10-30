# MuPhFASa

The MultiPHase Flow Adaptive Simulator (MuPhFASa) is a Lattice Boltzmann solver for multiphase flows built on the [AMReX](https://amrex-codes.github.io/amrex/) framework for adaptive mesh refinement (AMR).

Currently focused on single-phase flow validation (Couette, Poiseuille, Taylor-Green, lid-driven cavity, uniform forcing).

## Requirements

- C++17 compiler (GCC ≥ 10, Clang, or Intel)
- MPI (OpenMPI or MPICH) — optional for single-rank runs
- `git` with submodule support
- CMake ≥ 3.20 **or** GNU Make (both build systems are supported)
- Python 3 + `matplotlib` / `yt` only for plotting and convergence scripts

Clone and initialize the vendored AMReX submodule:

```sh
git clone --recursive https://github.com/argonne-cps/muphfasa.git
# or, if already cloned:
git submodule update --init --recursive
```

## Building the solver

Two build systems are supported. Both default to **DIM=3, MPI on, EB off**.

### GNUMake (quick local builds)

```sh
# 3-D solver (default)
make -j$(nproc)

# 2-D solver
make -j$(nproc) DIM=2

# single-rank (no MPI), debug, or with EB
make -j$(nproc) DIM=2 USE_MPI=FALSE
make -j$(nproc) DIM=3 DEBUG=TRUE
make -j$(nproc) DIM=3 USE_EB=TRUE        # embedded-boundary support (off by default)

# clean
make clean
```

This produces `muphfasa3d.gnu.MPI.ex` (or `muphfasa2d.gnu.MPI.ex` for `DIM=2`). Top-level GNUMake enables MPI by default; regression tests (see below) default to no MPI for quick single-rank runs.

### CMake (recommended for tests and installs)

```sh
# 3-D, no tests
cmake -S . -B build -Dmuphfasa_DIM=3
cmake --build build -j$(nproc)

# 2-D with tests enabled
cmake -S . -B build2d -Dmuphfasa_DIM=2 -Dmuphfasa_BUILD_TESTS=ON
cmake --build build2d -j$(nproc)

# with EB or without MPI
cmake -S . -B build -Dmuphfasa_ENABLE_EB=ON -Dmuphfasa_ENABLE_MPI=OFF
cmake --build build -j$(nproc)
```

The CMake binary is `build/muphfasa` (or `build2d/muphfasa`).

#### Build options

| GNUMake (`make VAR=...`) | CMake (`-DVAR=...`) | Default | Description |
|---|---|---|---|
| `DIM=2\|3` | `muphfasa_DIM=2\|3` | `3` | Dimensionality (also selects lattice: `D2Q9` for 2D, `D3Q15/19/27` for 3D) |
| `USE_EB=TRUE/FALSE` | `muphfasa_ENABLE_EB=ON/OFF` | `FALSE/OFF` | AMReX embedded-boundary support |
| `USE_MPI=TRUE/FALSE` | `muphfasa_ENABLE_MPI=ON/OFF` | `TRUE/ON` | MPI parallelism |
| `USE_OMP=TRUE/FALSE` | `muphfasa_ENABLE_OMP=ON/OFF` | `FALSE/OFF` | OpenMP threading |
| `USE_CUDA/HIP=TRUE` | `muphfasa_ENABLE_CUDA/HIP=ON` | `FALSE/OFF` | GPU backends (mutually exclusive) |
| `DEBUG=TRUE/FALSE` | `CMAKE_BUILD_TYPE=Debug/Release` | `FALSE/Release` | Debug build |
| — | `muphfasa_BUILD_TESTS=ON/OFF` | `OFF` | Build unit + regression tests |

## Running the solver

```sh
# GNUMake build
./muphfasa3d.gnu.MPI.ex example_inputs/inputs.3d
mpirun -n 4 ./muphfasa3d.gnu.MPI.ex example_inputs/inputs.3d

# CMake build
./build/muphfasa example_inputs/inputs.3d
```

Example inputs are in `example_inputs/` (`inputs.2d`, `inputs.3d`). Key parameters are documented there.

## Tests

### Unit tests (`tests/unit/`)

Unit tests cover lattices, equilibria, macroscopic moments, BC corners/edges, mass conservation, and forcing. They are **CMake-only** (each `test_*.cpp` is its own executable sharing the solver sources).

```sh
# 3-D unit tests
cmake -S . -B build -Dmuphfasa_DIM=3 -Dmuphfasa_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -R "^test_"

# 2-D unit tests (includes D2Q9-specific checks)
cmake -S . -B build2d -Dmuphfasa_DIM=2 -Dmuphfasa_BUILD_TESTS=ON
cmake --build build2d -j$(nproc)
ctest --test-dir build2d --output-on-failure

# run a single test directly
build2d/tests/test_bc_corners
build/tests/test_mass_conservation
```

> **Note:** tests are only built when the build directory is configured with `-Dmuphfasa_BUILD_TESTS=ON`. If your existing build was configured without it, re-run `cmake` with this flag (or configure a new build directory).

### Regression tests (`tests/regression/`)

Each case has its own `GNUmakefile` (standalone, no MPI) and a CMake target. See per-case READMEs for physics and expected results:

- [`couette/`](tests/regression/couette/) — transient Couette startup vs analytic
- [`poiseuille/`](tests/regression/poiseuille/) — Poiseuille flow (body-force and inflow variants)
- [`tgv2d/`](tests/regression/tgv2d/) — 2-D Taylor-Green vortex decay (2D only)
- [`lid2d/`](tests/regression/lid2d/) — 2-D lid-driven cavity vs Ghia et al. (2D only)
- [`uniform_force/`](tests/regression/uniform_force/) — periodic box with uniform Guo forcing

GNUMake (from each case directory):

```sh
cd tests/regression/couette
make DIM=2                  # -> test_couette2d.gnu.ex
./test_couette2d.gnu.ex inputs

cd ../poiseuille
make DIM=3                  # -> test_poiseuille3d.gnu.ex
./test_poiseuille3d.gnu.ex inputs3d
```

CMake (from repo root):

```sh
cmake -S . -B build2d -Dmuphfasa_DIM=2 -Dmuphfasa_BUILD_TESTS=ON
cmake --build build2d -j$(nproc) --target test_couette
build2d/tests/test_couette tests/regression/couette/inputs
ctest --test-dir build2d -R regression --output-on-failure
```

Some convergence and plotting scripts (`convergence.py`, `plot_*.py`) require `fcompare` and `faverage` (AMReX tools) and `yt`/`matplotlib`; they are documented in each regression README.
