// InflowKernel.H — Zou-He velocity boundary condition, ghost-fill form.
//
// TODO (BC_implementation_plan.md Step 6): not yet implemented. Needs
// rho/u reconstructed at the face from the already-streamed-in populations
// at the boundary-adjacent fluid cell plus the prescribed u_in; derive and
// document that formula here the same way NoSlipKernel.H documents its own
// before implementing.

#ifndef MUPHFASA_INFLOW_KERNEL_H
#define MUPHFASA_INFLOW_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

namespace muphfasa {

template <typename LatticeTag>
struct InflowKernel
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& /*iv*/,
                       amrex::Array4<amrex::Real> const& /*f*/,
                       int /*dir*/, int /*bound*/) noexcept
    {
        amrex::Abort("InflowKernel::apply is not yet implemented (BC_implementation_plan.md Step 6)");
    }
};

} // namespace muphfasa

#endif // MUPHFASA_INFLOW_KERNEL_H
