// OutflowKernel.H — Zero-gradient / extrapolation outflow BC, ghost-fill form.
//
// For every "needed" population n (g + c_n inside domain) we copy that same
// population from the interior source cell x_f = g + c_n:
//
//     f(g, n) = f(g + c_n, n)
//
// so that stream()'s pull step reproduces f(x_f, n) unchanged at x_f, i.e.
// df/dn = 0 across the face. Diagonal n use the same rule as axis-aligned.

#ifndef MUPHFASA_OUTFLOW_KERNEL_H
#define MUPHFASA_OUTFLOW_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>


namespace muphfasa {

template <typename LatticeTag>
struct OutflowKernel
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       const amrex::Box& domain) noexcept
    {
        for (int n = 0; n < LatticeTag::Q; ++n) {
            amrex::IntVect c = amrex::IntVect{AMREX_D_DECL(LatticeTag::cx(n), LatticeTag::cy(n), LatticeTag::cz(n))};
            const amrex::IntVect src = iv + c;
            if (domain.contains(src) && f.contains(src)) {
                f(iv, n) = f(src, n);
            }
        }
    }
};

} // namespace muphfasa

#endif // MUPHFASA_OUTFLOW_KERNEL_H
