// Collision.H — Interface for LBM collision operators.
// Currently supports BGK (Bhatnagar-Gross-Krook) single-relaxation-time.
// Additional models (TRT, MRT, regularised) can be added as new policy types.

#ifndef MUPHFASA_COLLISION_H
#define MUPHFASA_COLLISION_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// ── Collision model tag types ─────────────────────────────────────────────
struct BGK {
    template <typename LatticeTag>
    static void collideArr(const amrex::Box& box,
                           amrex::Array4<amrex::Real> const& f_arr,
                           amrex::Array4<const amrex::Real> const& feq_arr,
                           const amrex::Real omega);

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
