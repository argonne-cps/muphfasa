// Equilibrium.H — Interface for the Maxwell-Boltzmann equilibrium distribution.
// Computes f^eq_q(rho, u) using the standard second-order expansion of the
// Maxwell-Boltzmann distribution, valid at low Mach number.
// Per-cell functions are AMREX_GPU_HOST_DEVICE callable.

#ifndef MUPHFASA_EQUILIBRIUM_H
#define MUPHFASA_EQUILIBRIUM_H

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// ── single-cell kernel ─────────────────────────────────────────────────
// Returns f^eq for a single (cell, direction n) pair given rho and u.
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell(const amrex::Real rho,
                                   amrex::Array4<amrex::Real const> const& vel,
                                   int i, int j, int k, int n) noexcept;

// GpuArray-based variants: same math, u passed as local scratch instead of
// an Array4 lookup. Used by fused kernels with no vel MultiFab. Defined
// inline here, not in Equilibrium.cpp, since always_inline needs the body
// visible at every call site (e.g. from Collision.cpp).
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell2D(const amrex::Real rho,
                                     amrex::GpuArray<amrex::Real, 2> const& u,
                                     int n) noexcept
{
    const amrex::Real u_sq = u[0] * u[0] + u[1] * u[1];
    const amrex::Real cdotu = LatticeTag::cx(n) * u[0] + LatticeTag::cy(n) * u[1];

    return LatticeTag::weights(n) * rho *
           (1 + LatticeTag::cs2inv * cdotu
              + 0.5 * LatticeTag::cs2inv * LatticeTag::cs2inv * cdotu * cdotu
              - 0.5 * LatticeTag::cs2inv * u_sq);
}

template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell3D(const amrex::Real rho,
                                     amrex::GpuArray<amrex::Real, 3> const& u,
                                     int n) noexcept
{
    const amrex::Real u_sq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const amrex::Real cdotu = LatticeTag::cx(n) * u[0] + LatticeTag::cy(n) * u[1]
                             + LatticeTag::cz(n) * u[2];

    return LatticeTag::weights(n) * rho *
           (1 + LatticeTag::cs2inv * cdotu
              + 0.5 * LatticeTag::cs2inv * LatticeTag::cs2inv * cdotu * cdotu
              - 0.5 * LatticeTag::cs2inv * u_sq);
}

// ── array-level kernel ─────────────────────────────────────────────────
// Fills feq_arr for all velocity directions given rho and u arrays.
template <typename LatticeTag>
void computeEquilibrium(const amrex::Box& box,
                               amrex::Array4<amrex::Real const> const& rho,
                               amrex::Array4<amrex::Real const> const& vel,
                               amrex::Array4<amrex::Real> const& feq_arr);


// ── MultiFab-level helpers (host-side) ───────────────────────────────────
// Fills feq_mf (Q components) over all valid cells using rho_mf and vel_mf.
template <typename LatticeTag>
void computeEquilibriumMF(const amrex::MultiFab& rho_mf,
                           const amrex::MultiFab& vel_mf,
                           amrex::MultiFab&       feq_mf);

} // namespace muphfasa

#endif // MUPHFASA_EQUILIBRIUM_H
