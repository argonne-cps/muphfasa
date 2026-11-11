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
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real collide(const amrex::Real f,
                               const amrex::Real feq,
                               const amrex::Real omega) noexcept
    {                           
        return (1 - omega) * f + omega * feq;
    }
};   // single-relaxation-time
struct TRT {};   // two-relaxation-time     (not yet implemented)
struct MRT {};   // multiple-relaxation-time (not yet implemented)

// ── Array-level collision (host-side) ─────────────────────────────────
// Per-tile kernel fused stream/collide: pulls (streams)
// then computes rho, u, and f^eq per cell and immediately
// relaxes f toward it into fstar_out_arr. rho/u/feq are local per-cell scratch,
// not written out to MultiFabs.
template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedArr(const amrex::Box& box,
                    amrex::Array4<const amrex::Real> const& fstar_in_arr,
                    amrex::Array4<amrex::Real> const& fstar_out_arr,
                    amrex::Real omega);

// Forced variant: half-force-corrected velocity (Guo scheme) and Guo
// force term added after BGK relaxation. F is constant body force in
// lattice units; u used for feq and guoForceTerm is (sum c f + F/2)/rho.
template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedArr(const amrex::Box& box,
                    amrex::Array4<const amrex::Real> const& fstar_in_arr,
                    amrex::Array4<amrex::Real> const& fstar_out_arr,
                    amrex::Real omega,
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F);

// ── MultiFab-level collision (host-side) ─────────────────────────────────
// Full stream/collision step: Pulls from f then collides
// w/ fused rho/u/feq/fstar computation in one MFIter pass.
template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedMF(const amrex::MultiFab& fin_mf,
                       amrex::MultiFab& fout_mf,
                       amrex::Real omega);

// Forced variant (body force F, lattice units).
template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedMF(const amrex::MultiFab& fin_mf,
                       amrex::MultiFab& fout_mf,
                       amrex::Real omega,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F);

} // namespace muphfasa

#endif // MUPHFASA_COLLISION_H
