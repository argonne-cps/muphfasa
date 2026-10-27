# Poiseuille flow regression test

Plane Poiseuille flow driven by a uniform body force (Guo forcing, `forcing.accel`):
periodic in x (and y in 3D), `NoSlip` walls at y=lo/hi (2D) or z=lo/hi (3D). With
H = wall-normal cell count (halfway bounce-back, as in the Couette test) and lattice
force F, the steady state is

    u(y) = F/(2 nu) * y * (H - y)           (y = j + 1/2 at cell centres)
    U_max = F H^2 / (8 nu),  U_mean = F H^2 / (12 nu) = (2/3) U_max

`test_poiseuille.cpp` prints the predicted `U_max`/`U_mean`/Mach number, writes
`pltXXXXXXX`/`analytic_pltXXXXXXX` pairs for `fcompare`, and at the end asserts
(exit code 1 on failure) that the profile error, mean-velocity error (both relative to
`U_max`, tolerance `test.max_rel_err`, default 1e-2), transverse velocity and density
spread are small. `sim.max_step` must be long enough to reach steady state
(2.5 H^2/nu_lat in the shipped inputs).

## Inflow / PressureOutlet variant
`inputs_inflow` (2D) and `inputs3d_inflow` (3D, D3Q19) drive the same flow through a
uniform `Inflow` at x=lo and a fixed-density `PressureOutlet` at x=hi (density from
`bc.hix_rho`, default `units.ref_density`) together with the body force. The zero-gradient
`Outflow` cannot be used here: with `NoSlip` walls it has no density anchor and the mean
density drifts. The inlet is uniform, so the profile develops downstream and only
`test.window` (fractions of the length) is compared with the parabola. Where `NoSlip`
outranks `Inflow` at the inlet/wall corners the diagonal populations lose their inflow
correction, so the delivered flux is `Uin (1 - 1/(3H))`; the inputs set
`Uin = U_mean 3H/(3H-1)` and the test asserts this.

## Build and run
```sh
make DIM=2                  # test_poiseuille2d.gnu.ex
./test_poiseuille2d.gnu.ex inputs
./test_poiseuille2d.gnu.ex inputs_inflow   # Inflow / PressureOutlet
make DIM=3                  # test_poiseuille3d.gnu.ex
./test_poiseuille3d.gnu.ex inputs3d      # lbm.lattice_type=D3Q15/D3Q27 for the others
./test_poiseuille3d.gnu.ex inputs3d_inflow
```

## Convergence study
```sh
python3 convergence.py                   # 2-D, N = 16 32 64, writes poiseuille2d_convergence.csv/.png
python3 convergence.py --dim 3 --min-order 1.8
```
The physical problem is fixed and `lbm.tau` pinned, so `nu_lat` is constant and
`F_lat` scales like dx^3. Expected observed order ~2.

## Magic relaxation time
At `Lambda = (tau - 1/2)^2 = 3/16` (`lbm.tau=0.9330127`) BGK + halfway bounce-back is
expected to reproduce the parabola essentially to round-off; run with that override as
a diagnostic of the force term and wall treatment.
