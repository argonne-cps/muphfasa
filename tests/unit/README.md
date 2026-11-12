# Unit tests

Lightweight, host-only tests for core LBM kernels and boundary conditions. Each `test_*.cpp` defines its own `main()` and links directly against the solver sources (excluding `main.cpp`), so tests stay in sync with the solver without a shared library.

| Test | What it checks |
|---|---|
| `test_lattice` | Lattice weights, discrete velocities, sound speed, opposite indices |
| `test_macroscopic` | Density/velocity moments from populations |
| `test_equilibrium` | Equilibrium distribution against analytic |
| `test_units` | Physical ↔ lattice unit conversion |
| `test_bc_corners` | BC coverage, source-cell correctness, corner/edge priority (`NoSlip > MovingWall > Inflow > Outflow`) |
| `test_mass_conservation` | Closed-box mass conservation |
| `test_forcing` | Guo forcing momentum update |
| `test_amr_convert` | AMR coarse/fine population conversion: round trip, ρ/u invariance, collide(pre) consistency |

## Build and run (CMake only)

Unit tests are not built by GNUMake — use CMake with `muphfasa_BUILD_TESTS=ON`.

```sh
# 2-D unit tests (covers D2Q9 corner/edge cases)
cmake -S . -B build2d -Dmuphfasa_DIM=2 -Dmuphfasa_BUILD_TESTS=ON
cmake --build build2d -j$(nproc)
ctest --test-dir build2d --output-on-failure -R "^test_"

# 3-D unit tests (covers D3Q15/D3Q19/D3Q27)
cmake -S . -B build -Dmuphfasa_DIM=3 -Dmuphfasa_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -R "^test_"

# run a single test directly
build2d/tests/test_bc_corners
build/tests/test_lattice
```

Build options:

- `muphfasa_DIM=2|3` — dimensionality (default `3`)
- `muphfasa_ENABLE_EB=ON/OFF` — embedded-boundary support (default `OFF`)
- `muphfasa_ENABLE_MPI=ON/OFF` — MPI (default `ON`; tests run single-rank regardless)

> **Note:** tests are only built when the build directory is configured with `-Dmuphfasa_BUILD_TESTS=ON`. If your existing build was configured without it, re-run `cmake` with this flag (or configure a new build directory).

## Adding a new unit test

1. Create `tests/unit/test_myfeature.cpp` with a `main()` that throws or returns non-zero on failure.
2. Add the name (without `.cpp`) to `UNIT_TEST_NAMES` in `tests/CMakeLists.txt`.
3. Reconfigure with `-Dmuphfasa_BUILD_TESTS=ON` and build — no other build-system changes needed.
