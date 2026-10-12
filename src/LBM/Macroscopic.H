// Macroscopic.H — Interfaces for computing macroscopic fluid variables.
// Derives density (rho) and velocity (u) from the distribution function f
// by taking zeroth and first moments over the discrete velocity set.
// Per-cell kernels are AMREX_GPU_HOST_DEVICE callable; their definitions live
// in Macroscopic.cpp (explicit instantiations for each supported lattice).

#ifndef MUPHFASA_MACROSCOPIC_H
#define MUPHFASA_MACROSCOPIC_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// ── Single-cell kernels ────────────────────────────────────────────────────
// Operate on one (i,j,k) cell; callable directly from GPU device code
// without an AMReX ParallelFor context (e.g. from unit tests on host).

// rho = sum_q f_q, for a single cell.
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeDensityCell(amrex::Array4<const amrex::Real> const& f_arr,
                               int i, int j, int k) noexcept;

// u_alpha = (1/rho) * sum_q c_{q,alpha} * f_q, for a single cell.
// Writes AMREX_SPACEDIM velocity components into vel_arr(i,j,k,*).
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void computeVelocityCell(amrex::Array4<const amrex::Real> const& f_arr,
                         const amrex::Real rho,
                         amrex::Array4<amrex::Real> const& vel_arr,
                         int i, int j, int k) noexcept;

// ── Array-level moment kernels ─────────────────────────────────────────────
// f[q] holds the q-th distribution value for a single cell.

// rho = sum_q f_q
template <typename LatticeTag>
void computeDensity(const amrex::Box& box,
                    amrex::Array4<const amrex::Real> const& f_arr,
                    amrex::Array4<amrex::Real> const& rho_arr);

// u_alpha = (1/rho) * sum_q c_{q,alpha} * f_q   (alpha = 0..D-1)
template <typename LatticeTag>
void computeVelocity(const amrex::Box& box,
                     amrex::Array4<const amrex::Real> const& f_arr,
                     amrex::Array4<const amrex::Real> const& rho_arr,
                     amrex::Array4<amrex::Real> const& vel_arr);

template <typename LatticeTag>
void computeVelocity(const amrex::Box& box,
                     amrex::Array4<const amrex::Real> const& f_arr,
                     amrex::Array4<amrex::Real> const& rho_arr,
                     amrex::Array4<amrex::Real> const& vel_arr);


// ── MultiFab-level helpers (host-side) ───────────────────────────────────
template <typename LatticeTag>
void computeMacroscopic(const amrex::MultiFab& f_mf,
                        amrex::MultiFab&       rho_mf,
                        amrex::MultiFab&       vel_mf);

} // namespace muphfasa

#endif // MUPHFASA_MACROSCOPIC_H
