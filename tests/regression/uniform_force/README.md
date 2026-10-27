# Uniform body-force regression test

Fully periodic box, quiescent start (u = 0; the Guo-consistent initialisation keeps the reported initial velocity exactly 0), constant body force (`forcing.accel`, physical
m/s^2). With the Guo scheme the momentum `sum(c f)` grows by exactly `F` per step,
independent of `omega`, so after `k` `advance()` calls the (half-force corrected)
velocity must be exactly

    u_k = k * F / rho,     rho_k = rho_0,

and the field must stay spatially uniform. `test_uniform_force.cpp` checks this every
step to 1e-12 and exits non-zero on failure. It exercises input parsing -> unit
conversion -> forced collide -> forced macroscopic update -> `SimulationImpl` branch
together, independent of the wall treatment.

## Build and run
```sh
make DIM=2                  # test_uniform_force2d.gnu.ex
./test_uniform_force2d.gnu.ex inputs
make DIM=3                  # test_uniform_force3d.gnu.ex
./test_uniform_force3d.gnu.ex inputs3d lbm.lattice_type=D3Q27   # also D3Q15, D3Q19
```
Different relaxation times (the result must not depend on `tau`):
```sh
for tau in 0.6 1.0 1.8; do ./test_uniform_force2d.gnu.ex inputs lbm.tau=$tau || exit 1; done
```
