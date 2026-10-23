// Macroscopic.H — Interfaces for computing macroscopic fluid variables.
// Derives density (rho) and velocity (u) from the distribution function f
// by taking zeroth and first moments over the discrete velocity set.
// computeDensityCell/computeVelocityCell2D/3D are defined inline here since
// they're always_inline and called across translation units; computeVelocityCell
// stays in Macroscopic.cpp since it's only called from within that file.

#ifndef MUPHFASA_MACROSCOPIC_H
#define MUPHFASA_MACROSCOPIC_H

#include <AMReX_Array.H>
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
                               int i, int j, int k) noexcept
{
    amrex::Real rho = 0.;
    for (int n = 0; n < LatticeTag::Q; ++n) {
        rho += f_arr(i, j, k, n);
    }
    return rho;
}

// rho = sum_q f_q, for a single cell.
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeDensityCell(amrex::GpuArray<amrex::Real, LatticeTag::Q> const& f_arr) noexcept
{
    amrex::Real rho = 0.;
    for (int n = 0; n < LatticeTag::Q; ++n) {
        rho += f_arr[n];
    }
    return rho;
}

// u_alpha = (1/rho) * sum_q c_{q,alpha} * f_q, for a single cell.
// Writes AMREX_SPACEDIM velocity components into vel_arr(i,j,k,*).
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void computeVelocityCell(amrex::Array4<const amrex::Real> const& f_arr,
                         const amrex::Real rho,
                         amrex::Array4<amrex::Real> const& vel_arr,
                         int i, int j, int k) noexcept;

// u_alpha = (1/rho) * sum_q c_{q,alpha} * f_q, for a single cell.
// Writes AMREX_SPACEDIM velocity components into vel_arr(i,j,k,*).
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void computeVelocityCell(amrex::GpuArray<amrex::Real, LatticeTag::Q> const& f_arr,
                         const amrex::Real rho,
                         amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& vel_vec) noexcept;

// GpuArray-returning variants: same math, returned as local scratch instead
// of written into an Array4. Used by fused kernels with no MultiFab for u.
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 2> computeVelocityCell2D(amrex::Array4<const amrex::Real> const& f_arr,
                                                       const amrex::Real rho,
                                                       int i, int j, int k) noexcept
{
    amrex::Real ux = 0.;
    amrex::Real uy = 0.;
    for (int n = 0; n < LatticeTag::Q; ++n) {
        ux += LatticeTag::cx(n) * f_arr(i, j, k, n);
        uy += LatticeTag::cy(n) * f_arr(i, j, k, n);
    }
    return {ux / rho, uy / rho};
}

template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 2> computeVelocityCell2D(amrex::GpuArray<amrex::Real, LatticeTag::Q> const& f_arr,
                                                       const amrex::Real rho) noexcept
{
    amrex::Real ux = 0.;
    amrex::Real uy = 0.;
    for (int n = 0; n < LatticeTag::Q; ++n) {
        ux += LatticeTag::cx(n) * f_arr[n];
        uy += LatticeTag::cy(n) * f_arr[n];
    }
    return {ux / rho, uy / rho};
}


template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> computeVelocityCell3D(amrex::Array4<const amrex::Real> const& f_arr,
                                                       const amrex::Real rho,
                                                       int i, int j, int k) noexcept
{
    amrex::Real ux = 0.;
    amrex::Real uy = 0.;
    amrex::Real uz = 0.;
    for (int n = 0; n < LatticeTag::Q; ++n) {
        ux += LatticeTag::cx(n) * f_arr(i, j, k, n);
        uy += LatticeTag::cy(n) * f_arr(i, j, k, n);
        uz += LatticeTag::cz(n) * f_arr(i, j, k, n);
    }
    return {ux / rho, uy / rho, uz / rho};
}

template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> computeVelocityCell3D(amrex::GpuArray<amrex::Real, LatticeTag::Q> const& f_arr,
                                                       const amrex::Real rho) noexcept
{
    amrex::Real ux = 0.;
    amrex::Real uy = 0.;
    amrex::Real uz = 0.;
    for (int n = 0; n < LatticeTag::Q; ++n) {
        ux += LatticeTag::cx(n) * f_arr[n];
        uy += LatticeTag::cy(n) * f_arr[n];
        uz += LatticeTag::cz(n) * f_arr[n];
    }
    return {ux / rho, uy / rho, uz / rho};
}

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


// ── MultiFab-level helpers (host-side) ───────────────────────────────────
template <typename LatticeTag>
void computeMacroscopic(const amrex::MultiFab& f_mf,
                        amrex::MultiFab&       rho_mf,
                        amrex::MultiFab&       vel_mf);

} // namespace muphfasa

#endif // MUPHFASA_MACROSCOPIC_H
