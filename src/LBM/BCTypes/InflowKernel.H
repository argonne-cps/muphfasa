// InflowKernel.H — Simplified Dirichlet-velocity inflow BC, ghost-fill form.
// Thin wrapper around BounceBack.H (see that file for the derivation).

#ifndef MUPHFASA_INFLOW_KERNEL_H
#define MUPHFASA_INFLOW_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include "BounceBack.H"

namespace muphfasa {

template <typename LatticeTag>
struct InflowKernel
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       const amrex::Box& domain,
                       amrex::Real rho_in,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Uin
                       ) noexcept
    {
        BounceBack<LatticeTag>::apply(iv, f, domain, rho_in, Uin);
    }
};

} // namespace muphfasa

#endif // MUPHFASA_INFLOW_KERNEL_H
