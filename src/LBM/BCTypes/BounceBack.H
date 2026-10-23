// BounceBack.H — Halfway bounce-back, ghost-fill form, with a prescribed
// density and velocity at the wall/face. Shared by NoSlipKernel,
// MovingWallKernel, and InflowKernel, each of which is a thin wrapper that
// just fixes rho/U (see those files).
//
// The wall/face sits halfway between the last fluid node x and the first
// ghost node g = x + bound*e_dir (bound = -1 for a lo-`dir` face, +1 for a
// hi-`dir` face). This kernel is only ever invoked by AMReX for ghost cells
// so `iv` below is always such a `g`.
//
// We want the ghost cell to hold, for every population n whose velocity
// points into the face along `dir` (c_n[dir] == -bound, i.e. n would stream
// out of the domain), the opposite-direction population already sitting in
// the adjacent fluid cell x = g - bound*e_dir (pre-stream), plus a momentum
// correction for the prescribed velocity U:
//
//     f(g, n) = f(x, opposite(n)) + contrib(opposite(n))
//
// where,
//     contrib(m) = -2 * rho * weight(m) * sum_i (c_i(m) * U_i) / cs2
//
// (Kruger et al., "The Lattice Boltzmann Method", ch. 5.) The sum runs over
// every spatial component, including the one along `dir` — so U may carry a
// component normal to the face (as InflowKernel needs) as well as tangential
// components (as MovingWallKernel needs); passing U=0 reduces this to plain
// halfway bounce-back (NoSlipKernel).

#ifndef MUPHFASA_BOUNCEBACK_KERNEL_H
#define MUPHFASA_BOUNCEBACK_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

namespace muphfasa {

template <typename LatticeTag>
struct BounceBack
{
    // dir: face normal (0=x, 1=y, 2=z). bound: -1 for the lo face, +1 for
    // the hi face. iv is the ghost cell being filled.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       int dir, int bound,
                       amrex::Real rho,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> U
                       ) noexcept
    {
        for (int n = 0; n < LatticeTag::Q; ++n) {
#if (AMREX_SPACEDIM == 2)
            if (dir == 0) {
                if (LatticeTag::cx(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0]-bound, iv[1], opp_n) + contrib(opp_n, rho, U);
                }
            } else if (dir == 1) {
                if (LatticeTag::cy(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0], iv[1]-bound, opp_n) + contrib(opp_n, rho, U);
                }
            }
#else
            if (dir == 0) {
                if (LatticeTag::cx(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0]-bound, iv[1], iv[2], opp_n) + contrib(opp_n, rho, U);
                }
            } else if (dir == 1) {
                if (LatticeTag::cy(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0], iv[1]-bound, iv[2], opp_n) + contrib(opp_n, rho, U);
                }
            } else if (dir == 2) {
                if (LatticeTag::cz(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0], iv[1], iv[2]-bound, opp_n) + contrib(opp_n, rho, U);
                }
            }
#endif
        }
    }

    private:
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real contrib(int opp_n, amrex::Real rho,
                               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& U) {
        amrex::Real cdotu = AMREX_D_TERM(LatticeTag::cx(opp_n) * U[0],
                                        + LatticeTag::cy(opp_n) * U[1],
                                        + LatticeTag::cz(opp_n) * U[2]);
        return -2. * rho * LatticeTag::weights(opp_n) * cdotu * LatticeTag::cs2inv;
    }
};

}

#endif //MUPHFASA_BOUNCEBACK_KERNEL_H
