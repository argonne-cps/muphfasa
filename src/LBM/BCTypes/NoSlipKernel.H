// NoSlipKernel.H — Halfway bounce-back, ghost-fill form.
// Special case of MovingWall with U=0.

#ifndef MUPHFASA_NOSLIP_KERNEL_H
#define MUPHFASA_NOSLIP_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include "MovingWallKernel.H"

namespace muphfasa {

template <typename LatticeTag>
struct NoSlipKernel
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       const amrex::Box& domain) noexcept
    {
        amrex::Real dummy_rho = 1;
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dummy_Uwall{AMREX_D_DECL(0., 0., 0.)};
        MovingWallKernel<LatticeTag>::apply(iv, f, domain, dummy_rho, dummy_Uwall);
    }
};

} // namespace muphfasa

#endif // MUPHFASA_NOSLIP_KERNEL_H
