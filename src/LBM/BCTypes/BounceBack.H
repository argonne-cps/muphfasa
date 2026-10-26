// BounceBack.H — Halfway bounce-back, ghost-fill form, with a prescribed
// density and velocity at the wall/face. Shared by NoSlipKernel,
// MovingWallKernel, and InflowKernel, each of which is a thin wrapper that
// just fixes rho/U (see those files).
//
// The wall/face sits halfway between the last fluid node x_f and the first
// ghost node g. For streaming (pull) the fluid cell x_f = g + c_n is the one
// that reads f(g, n), so the correct halfway bounce-back (Kruger et al.,
// ch. 5) is:
//
//     f(g, n) = f(g + c_n, opposite(n)) + contrib(opposite(n))
//
// where g + c_n is the interior source cell (inside geom.Domain()) and
//     contrib(m) = -2 * rho * weight(m) * sum_i (c_i(m) * U_i) / cs2
//
// This holds for all populations n "needed" by streaming, i.e. where
// g + c_n is inside the domain. The older face-neighbour form
// f(g - bound*e_dir, opp) was only correct for axis-aligned n; diagonal n
// require the tangential shift g + c_n as well.

#ifndef MUPHFASA_BOUNCEBACK_KERNEL_H
#define MUPHFASA_BOUNCEBACK_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include "BCCommon.H"

namespace muphfasa {

template <typename LatticeTag>
struct BounceBack
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       const amrex::Box& domain,
                       amrex::Real rho,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> U
                       ) noexcept
    {
        for (int n = 0; n < LatticeTag::Q; ++n) {
            amrex::IntVect c = cvec<LatticeTag>(n);
            if (!neededPop(iv, c, domain)) continue;
            int opp_n = LatticeTag::opposite(n);
            amrex::IntVect src = iv + c;
            f(iv, n) = f(src, opp_n) + contrib(opp_n, rho, U);
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

} // namespace muphfasa

#endif //MUPHFASA_BOUNCEBACK_KERNEL_H
