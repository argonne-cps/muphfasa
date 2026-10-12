// Equilibrium.H — Interface for the Maxwell-Boltzmann equilibrium distribution.
// Computes f^eq_q(rho, u) using the standard second-order expansion of the
// Maxwell-Boltzmann distribution, valid at low Mach number.
// Per-cell functions are AMREX_GPU_HOST_DEVICE callable.

#ifndef MUPHFASA_EQUILIBRIUM_H
#define MUPHFASA_EQUILIBRIUM_H

#include <AMReX_GpuArray.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// ── single-cell kernel ─────────────────────────────────────────────────
// Returns f^eq for a single (cell, direction n) pair given rho and u at
// that cell. Callable directly from GPU device code without an AMReX
// ParallelFor context (e.g. from unit tests on host).
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell(const amrex::Real rho,
                                   amrex::Array4<amrex::Real const> const& vel,
                                   int i, int j, int k, int n) noexcept;

// GpuArray-based variants: identical math to computeEquilibriumCell, but take
// u as local scratch (from e.g. computeVelocityCell2D/3D) instead of reading
// it out of an Array4. For use by fused kernels that have no vel MultiFab.
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell2D(const amrex::Real rho,
                                     amrex::GpuArray<amrex::Real, 2> const& u,
                                     int n) noexcept;

template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell3D(const amrex::Real rho,
                                     amrex::GpuArray<amrex::Real, 3> const& u,
                                     int n) noexcept;

// ── array-level kernel ─────────────────────────────────────────────────
// Returns f^eq for a all velocity directions given rho and u array
// LatticeTag must be one of the LatticeDescriptor specialisations.
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
