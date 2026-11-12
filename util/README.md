# Utilities

Standalone helpers that build without the full solver.

## input_info

Reads a muphfasa input file with the solver's own parser and prints what a run would start with, without running anything: grid and memory, unit conversion and lattice parameters (dx, dt, u_lat, tau), boundary conditions, estimated step count and end time, and disk usage of the plot/checkpoint output. Out-of-range `tau` / `u_lat` produce warnings with a suggested fix instead of an abort. If `amr.restart` is set, it also checks the checkpoint's Header against the input (grid size, number of distributions) and reports the steps that remain.

Use it to sanity-check an input file or a restart before submitting a job.

The executable is built for one dimensionality, which must match the input file (2 or 3 values in `grid.ncells`).

### Build

CMake (builds `muphfasa_inputinfo`):

```sh
cmake -S . -B build2d -Dmuphfasa_DIM=2 -Dmuphfasa_BUILD_UTIL=ON
cmake --build build2d --target muphfasa_inputinfo -j$(nproc)
./build2d/util/muphfasa_inputinfo <inputs file>
```

GNUMake (needs `AMREX_HOME`, or `external/amrex`):

```sh
cd util/input_info
make DIM=2 -j$(nproc)
./input_info2d.gnu.ex <inputs file>
```

### Run

```sh
muphfasa_inputinfo example_inputs/inputs.2d
muphfasa_inputinfo example_inputs/inputs.2d lbm.tau=0.8 amr.restart=chk0002000
```

Extra `key=value` arguments override the file, as for the solver. Run it from the directory the solver will run in so a relative `amr.restart` path resolves the same way.
