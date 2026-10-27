// Collision.H — Interface for LBM collision operators.
// Currently supports BGK (Bhatnagar-Gross-Krook) single-relaxation-time.
// Additional models (TRT, MRT, regularised) can be added as new policy types.

#ifndef MUPHFASA_COLLISION_H
#define MUPHFASA_COLLISION_H

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// ── Guo body-force term ───────────────────────────────────────────────────
// F_i = (1 - omega/2) w_i cs2inv [ c_i.F + cs2inv (c_i.u)(c_i.F) - u.F ]
// (lattice units, dt = 1), added to f_i after relaxation. u is the
// half-force-corrected velocity also used for feq (Guo, Zheng & Shi, PRE 65,
// 046308, 2002). Defined inline here, like the Equilibrium.H cell kernels,
// since always_inline needs the body at every call site.
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real guoForceTermCell2D(int n,
                               amrex::GpuArray<amrex::Real, 2> const& u,
                               amrex::GpuArray<amrex::Real, 2> const& F,
                               amrex::Real omega) noexcept
{
    const amrex::Real cdotF = LatticeTag::cx(n) * F[0] + LatticeTag::cy(n) * F[1];
    const amrex::Real cdotu = LatticeTag::cx(n) * u[0] + LatticeTag::cy(n) * u[1];
    const amrex::Real udotF = u[0] * F[0] + u[1] * F[1];
    return (amrex::Real(1.0) - amrex::Real(0.5) * omega) * LatticeTag::weights(n) * LatticeTag::cs2inv *
           (cdotF + LatticeTag::cs2inv * cdotu * cdotF - udotF);
}

template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real guoForceTermCell3D(int n,
                               amrex::GpuArray<amrex::Real, 3> const& u,
                               amrex::GpuArray<amrex::Real, 3> const& F,
                               amrex::Real omega) noexcept
{
    const amrex::Real cdotF = LatticeTag::cx(n) * F[0] + LatticeTag::cy(n) * F[1]
                            + LatticeTag::cz(n) * F[2];
    const amrex::Real cdotu = LatticeTag::cx(n) * u[0] + LatticeTag::cy(n) * u[1]
                            + LatticeTag::cz(n) * u[2];
    const amrex::Real udotF = u[0] * F[0] + u[1] * F[1] + u[2] * F[2];
    return (amrex::Real(1.0) - amrex::Real(0.5) * omega) * LatticeTag::weights(n) * LatticeTag::cs2inv *
           (cdotF + LatticeTag::cs2inv * cdotu * cdotF - udotF);
}

// Dimension-dispatching wrapper (D2Q9 has no cz, so the 2D/3D split is
// compile-time via AMREX_SPACEDIM, as in the rest of the LBM kernels).
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real guoForceTerm(int n,
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& u,
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F,
                         amrex::Real omega) noexcept
{
#if (AMREX_SPACEDIM == 2)
    return guoForceTermCell2D<LatticeTag>(n, u, F, omega);
#else
    return guoForceTermCell3D<LatticeTag>(n, u, F, omega);
#endif
}

// ── Collision model tag types ─────────────────────────────────────────────
struct BGK {
    template <typename LatticeTag>
    static void collideArr(const amrex::Box& box,
                           amrex::Array4<amrex::Real> const& f_arr,
                           amrex::Array4<const amrex::Real> const& feq_arr,
                           const amrex::Real omega);

    // Forced variant: relaxation plus the Guo term, using the (half-force
    // corrected) velocity in vel_arr. A separate overload so the force-free
    // kernel above does no extra reads or arithmetic.
    template <typename LatticeTag>
    static void collideArr(const amrex::Box& box,
                           amrex::Array4<amrex::Real> const& f_arr,
                           amrex::Array4<const amrex::Real> const& feq_arr,
                           amrex::Array4<const amrex::Real> const& vel_arr,
                           const amrex::Real omega,
                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F);

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real collide(const amrex::Real f,
                               const amrex::Real feq,
                               const amrex::Real omega) noexcept;
};   // single-relaxation-time
struct TRT {};   // two-relaxation-time     (not yet implemented)
struct MRT {};   // multiple-relaxation-time (not yet implemented)

// ── MultiFab-level collision (host-side) ─────────────────────────────────

// Apply the collision operator in-place to f_mf given feq_mf and tau.
// Dispatches on CollisionModel tag (currently only BGK is active).
template <typename LatticeTag, typename CollisionModel>
void collideMF(amrex::MultiFab&       f_mf,
               const amrex::MultiFab& feq_mf,
               amrex::Real            omega);

// Forced variant (body force F, lattice units): vel_mf is the half-force
// corrected velocity from computeMacroscopic(..., F).
template <typename LatticeTag, typename CollisionModel>
void collideMF(amrex::MultiFab&       f_mf,
               const amrex::MultiFab& feq_mf,
               const amrex::MultiFab& vel_mf,
               amrex::Real            omega,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F);

// Per-tile kernel: computes rho, u, and f^eq per cell and immediately
// relaxes f toward it into fstar_arr. rho/u/feq are local per-cell scratch,
// not written out to MultiFabs, since nothing downstream of this fused
// kernel needs them.
template <typename LatticeTag, typename CollisionModel>
void collideFullArr(const amrex::Box& box,
                    amrex::Array4<amrex::Real> const& fstar_arr,
                    amrex::Array4<const amrex::Real> const& f_arr,
                    amrex::Real omega);

// Full collision step: fused rho/u/feq/fstar computation in one MFIter pass.
template <typename LatticeTag, typename CollisionModel>
void collideFullMF(amrex::MultiFab&       fstar_mf,
                   const amrex::MultiFab& f_mf,
                   amrex::Real            omega);

} // namespace muphfasa

#endif // MUPHFASA_COLLISION_H
