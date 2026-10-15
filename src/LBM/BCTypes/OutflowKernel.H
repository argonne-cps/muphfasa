// OutflowKernel.H — Zero-gradient / extrapolation outflow BC, ghost-fill form.
//
// TODO (BC_implementation_plan.md Step 6): not yet implemented.

#ifndef MUPHFASA_OUTFLOW_KERNEL_H
#define MUPHFASA_OUTFLOW_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

namespace muphfasa {

template <typename LatticeTag>
struct OutflowKernel
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& /*iv*/,
                       amrex::Array4<amrex::Real> const& /*f*/,
                       int /*dir*/, int /*bound*/) noexcept
    {
        amrex::Abort("OutflowKernel::apply is not yet implemented (BC_implementation_plan.md Step 6)");
    }
};

} // namespace muphfasa

#endif // MUPHFASA_OUTFLOW_KERNEL_H
