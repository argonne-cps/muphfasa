// MovingWallKernel.H — Halfway bounce-back, ghost-fill form, 
// with wall moving perpendicularly to face normal.
//
//
// The wall sits halfway between the last fluid node x and the first ghost
// node g = x + bound*e_dir (bound = -1 for a lo-`dir` wall, +1 for a hi-`dir`
// wall). This kernel is only ever invoked by AMReX for ghost cells so `iv` 
// below is always such a `g`.
//
// We want the ghost cell to hold, for every population n whose velocity
// points into the wall along `dir` (c_n[dir] == -bound, i.e. n would stream
// out of the domain), the opposite-direction population already sitting in
// the adjacent fluid cell x = g - bound*e_dir (pre-stream):
//
//     f(g, n) = f(x, opposite(n)) + wall_contrib(opposite(n))
//
// where, 
//     wall_contrib(m) = -2 * rho_wall * weight(m) * sum_i (c_i(m) * U_wall_i)
// 
// wall contribution depends on the **opposite** direction population and the
// velocity and density at the wall. We only consider walls moving perpendicular
// to the face in this implementation. (No expanding/compressing walls)
// 

#ifndef MUPHFASA_MOVINGWALL_KERNEL_H
#define MUPHFASA_MOVINGWALL_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

namespace muphfasa {

template <typename LatticeTag>
struct MovingWallKernel
{
    // dir: wall normal (0=x, 1=y, 2=z). bound: -1 for the lo face, +1 for
    // the hi face. iv is the ghost cell being filled.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       int dir, int bound,
                       amrex::Real rho_wall,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Uwall
                       ) noexcept
    {
        for (int n = 0; n < LatticeTag::Q; ++n) {
#if (AMREX_SPACEDIM == 2)
            if (dir == 0) {
                if (LatticeTag::cx(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0]-bound, iv[1], opp_n) 
                             + wallContrib(n, LatticeTag::cy(opp_n), rho_wall, Uwall[1]);
                }
            } else if (dir == 1) {
                if (LatticeTag::cy(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0], iv[1]-bound, opp_n)
                             + wallContrib(n, LatticeTag::cx(opp_n), rho_wall, Uwall[0]);
                }
            }
#else
            if (dir == 0) {
                if (LatticeTag::cx(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0]-bound, iv[1], iv[2], opp_n)
                             + wallContrib(n, LatticeTag::cy(opp_n), rho_wall, Uwall[1])
                             + wallContrib(n, LatticeTag::cz(opp_n), rho_wall, Uwall[2]);
                }
            } else if (dir == 1) {
                if (LatticeTag::cy(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0], iv[1]-bound, iv[2], opp_n)
                             + wallContrib(n, LatticeTag::cx(opp_n), rho_wall, Uwall[0])
                             + wallContrib(n, LatticeTag::cz(opp_n), rho_wall, Uwall[2]);
                }
            } else if (dir == 2) {
                if (LatticeTag::cz(n) == -bound) {
                    int opp_n = LatticeTag::opposite(n);
                    f(iv, n) = f(iv[0], iv[1], iv[2]-bound, opp_n)
                             + wallContrib(n, LatticeTag::cx(opp_n), rho_wall, Uwall[0])
                             + wallContrib(n, LatticeTag::cy(opp_n), rho_wall, Uwall[1]);
                }
            }
#endif
        }
    }

    private:
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real wallContrib(int n, int c_dir, amrex::Real rho_wall, amrex::Real u) {
        return -2. * rho_wall * LatticeTag::weights(n) * c_dir * u;
    }
};

}

#endif //MUPHFASA_MOVINGWALL_KERNEL_H 