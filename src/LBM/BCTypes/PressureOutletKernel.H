// PressureOutletKernel.H — Fixed-density (pressure) outlet, ghost-fill form.
//
// Anti-bounce-back (Kruger et al., sec. 5.3.3.2). With the face halfway
// between the interior source cell x_f = g + c_n and the ghost g:
//
//     f(g, n) = -f(x_f, opp(n)) + 2 w_n rho_out [ 1 + (c_n.u_w)^2 / (2 cs^4)
//                                                   - u_w.u_w / (2 cs^2) ]
//
// which pins the density at the face to rho_out. The face velocity u_w is
// extrapolated (zero-gradient) from the source cell, i.e. the velocity of
// x_f computed from its own populations. Unlike the zero-gradient OutflowKernel
// this anchors the density level, so a velocity-driven inflow stays mass
// consistent.

#ifndef MUPHFASA_PRESSURE_OUTLET_KERNEL_H
#define MUPHFASA_PRESSURE_OUTLET_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

namespace muphfasa {

template <typename LatticeTag>
struct PressureOutletKernel
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       const amrex::Box& domain,
                       amrex::Real rho_out) noexcept
    {
        for (int n = 0; n < LatticeTag::Q; ++n) {
            const amrex::IntVect c = amrex::IntVect{AMREX_D_DECL(LatticeTag::cx(n), LatticeTag::cy(n), LatticeTag::cz(n))};
            const amrex::IntVect src = iv + c;
            if (!(domain.contains(src) && f.contains(src))) { continue; }

            // velocity of the source cell
            amrex::Real rho = 0.0;
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> j{AMREX_D_DECL(0., 0., 0.)};
            for (int m = 0; m < LatticeTag::Q; ++m) {
                const amrex::Real fm = f(src, m);
                rho += fm;
                AMREX_D_TERM(j[0] += LatticeTag::cx(m) * fm;,
                             j[1] += LatticeTag::cy(m) * fm;,
                             j[2] += LatticeTag::cz(m) * fm;)
            }
            const amrex::Real inv_rho = amrex::Real(1.0) / rho;
            const amrex::Real cu = (AMREX_D_TERM(LatticeTag::cx(n) * j[0],
                                                + LatticeTag::cy(n) * j[1],
                                                + LatticeTag::cz(n) * j[2])) * inv_rho;
            const amrex::Real uu = (AMREX_D_TERM(j[0] * j[0],
                                                + j[1] * j[1],
                                                + j[2] * j[2])) * inv_rho * inv_rho;

            const amrex::Real cs2inv = LatticeTag::cs2inv;
            const amrex::Real feq_even = LatticeTag::weights(n) * rho_out *
                (amrex::Real(1.0) + amrex::Real(0.5) * cu * cu * cs2inv * cs2inv
                                  - amrex::Real(0.5) * uu * cs2inv);
            f(iv, n) = -f(src, LatticeTag::opposite(n)) + amrex::Real(2.0) * feq_even;
        }
    }
};

} // namespace muphfasa

#endif // MUPHFASA_PRESSURE_OUTLET_KERNEL_H
