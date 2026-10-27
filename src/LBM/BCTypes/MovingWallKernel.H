// MovingWallKernel.H — Halfway bounce-back with a prescribed wall velocity,
// restricted to walls moving perpendicular to their own face normal (i.e.
// sliding/tangential motion, as in Couette/lid-driven-cavity flow — no
// expanding/compressing walls). Thin wrapper around BounceBack.H; see that
// file for the ghost-fill bounce-back + momentum-correction derivation.
//
// The "no normal component" restriction is enforced on the input side
// (InputParser::Uwall()), not in this kernel — BounceBack.H itself supports
// a normal velocity component (needed by InflowKernel).

#ifndef MUPHFASA_MOVINGWALL_KERNEL_H
#define MUPHFASA_MOVINGWALL_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include "BounceBack.H"

namespace muphfasa {

template <typename LatticeTag>
struct MovingWallKernel
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       const amrex::Box& domain,
                       amrex::Real rho_wall,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Uwall
                       ) noexcept
    {
        BounceBack<LatticeTag>::apply(iv, f, domain, rho_wall, Uwall);
    }
};

} // namespace muphfasa

#endif //MUPHFASA_MOVINGWALL_KERNEL_H
