// InflowKernel.H — Simplified Dirichlet-velocity inflow BC, ghost-fill form.
// Thin wrapper around BounceBack.H (see that file for the derivation).
//
// Unlike MovingWallKernel, Uin is expected to carry a component along the
// face normal (fluid entering the domain) in addition to any tangential
// components — BounceBack.H's momentum correction sums over all
// AMREX_SPACEDIM components, so that normal component is not dropped here.
//
// This is a *simplified* bounce-back-based Dirichlet-velocity BC, not full
// Zou-He: it reproduces the prescribed Uin to leading order by reusing the
// same halfway bounce-back + momentum-correction formula as NoSlip/
// MovingWall
//
#ifndef MUPHFASA_INFLOW_KERNEL_H
#define MUPHFASA_INFLOW_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include "BounceBack.H"

namespace muphfasa {

template <typename LatticeTag>
struct InflowKernel
{
    // dir: face normal (0=x, 1=y, 2=z). bound: -1 for the lo face, +1 for
    // the hi face. iv is the ghost cell being filled.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       int dir, int bound,
                       amrex::Real rho_in,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Uin
                       ) noexcept
    {
        BounceBack<LatticeTag>::apply(iv, f, dir, bound, rho_in, Uin);
    }
};

} // namespace muphfasa

#endif // MUPHFASA_INFLOW_KERNEL_H
