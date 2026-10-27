// Macroscopic.cpp — Implementations of density and velocity moment calculations.
// computeDensityCell/computeVelocityCell2D/3D live in Macroscopic.H (see
// header comment); computeVelocityCell stays here since it's only called
// from within this file.

#include "Macroscopic.H"
#include "LatticeDescriptor.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_GpuLaunch.H>

namespace muphfasa {

// ── computeVelocityCell ───────────────────────────────────────────────────
// Writes the computeVelocityCell2D/3D result into vel_arr(i,j,k,*).
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void computeVelocityCell(amrex::Array4<const amrex::Real> const& f_arr,
                         const amrex::Real rho,
                         amrex::Array4<amrex::Real> const& vel_arr,
                         int i, int j, int k) noexcept
{
#if (AMREX_SPACEDIM == 2)
    const auto u = computeVelocityCell2D<LatticeTag>(f_arr, rho, i, j, k);
    vel_arr(i, j, k, 0) = u[0];
    vel_arr(i, j, k, 1) = u[1];
#else
    const auto u = computeVelocityCell3D<LatticeTag>(f_arr, rho, i, j, k);
    vel_arr(i, j, k, 0) = u[0];
    vel_arr(i, j, k, 1) = u[1];
    vel_arr(i, j, k, 2) = u[2];
#endif
}

// ── computeDensity ────────────────────────────────────────────────────────
template <typename LatticeTag>
void computeDensity(const amrex::Box& box,
                    amrex::Array4<const amrex::Real> const& f_arr,
                    amrex::Array4<amrex::Real> const& rho_arr)

{
    BL_PROFILE("computeDensity()");

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        rho_arr(i, j, k) = computeDensityCell<LatticeTag>(f_arr, i, j, k);
    });
}

// ── computeVelocity ───────────────────────────────────────────────────────

template <typename LatticeTag>
void computeVelocity(const amrex::Box& box,
                     amrex::Array4<const amrex::Real> const& f_arr,
                     amrex::Array4<const amrex::Real> const& rho_arr,
                     amrex::Array4<amrex::Real> const& vel_arr)
{
    BL_PROFILE("computeVelocity()");

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        computeVelocityCell<LatticeTag>(f_arr, rho_arr(i, j, k), vel_arr, i, j, k);
    });
}


// ── computeMacroscopic (MultiFab) ─────────────────────────────────────────
template <typename LatticeTag>
void computeMacroscopic(const amrex::MultiFab& f_mf,
                        amrex::MultiFab&       rho_mf,
                        amrex::MultiFab&       vel_mf)
{
    BL_PROFILE("computeMacroscopic()");

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (amrex::MFIter mfi(f_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto tileBox = mfi.tilebox();

        amrex::Array4<amrex::Real const> const& f_arr = f_mf[mfi].const_array();
        amrex::Array4<amrex::Real> const& rho = rho_mf[mfi].array();
        amrex::Array4<amrex::Real> const& vel = vel_mf[mfi].array();
        computeDensity<LatticeTag>(tileBox, f_arr, rho);
        computeVelocity<LatticeTag>(tileBox, f_arr, rho, vel);
    }

    // sync for profiling
    amrex::Gpu::streamSynchronize();
}
void initVelCorrection(const amrex::MultiFab& rho_mf, amrex::MultiFab& vel_mf, 
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F) {

    BL_PROFILE("initVelCorrection");

    for (amrex::MFIter mfi(vel_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& rho_arr = rho_mf.const_array(mfi);
        auto const& vel_arr = vel_mf.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                vel_arr(i, j, k, d) -= amrex::Real(0.5) * F[d] / rho_arr(i, j, k);
            }
        });
    }
    // sync for profiling
    amrex::Gpu::streamSynchronize();
}

// ── Explicit instantiations ───────────────────────────────────────────────
// Dimension-matched only: computeVelocityCell branches on AMREX_SPACEDIM.
#if (AMREX_SPACEDIM == 2)
template void computeVelocityCell<D2Q9>(amrex::Array4<const amrex::Real> const&, const amrex::Real, amrex::Array4<amrex::Real> const&, int, int, int);
#else
template void computeVelocityCell<D3Q15>(amrex::Array4<const amrex::Real> const&, const amrex::Real, amrex::Array4<amrex::Real> const&, int, int, int);
template void computeVelocityCell<D3Q19>(amrex::Array4<const amrex::Real> const&, const amrex::Real, amrex::Array4<amrex::Real> const&, int, int, int);
template void computeVelocityCell<D3Q27>(amrex::Array4<const amrex::Real> const&, const amrex::Real, amrex::Array4<amrex::Real> const&, int, int, int);
#endif

template void computeDensity<D2Q9>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeDensity<D3Q15>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeDensity<D3Q19>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeDensity<D3Q27>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);

#if (AMREX_SPACEDIM == 2)
template void computeVelocity<D2Q9>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
#else
template void computeVelocity<D3Q15>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeVelocity<D3Q19>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeVelocity<D3Q27>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
#endif

#if (AMREX_SPACEDIM == 2)
template void computeMacroscopic<D2Q9>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&);
#else
template void computeMacroscopic<D3Q15>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&);
template void computeMacroscopic<D3Q19>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&);
template void computeMacroscopic<D3Q27>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&);
#endif

} // namespace muphfasa
