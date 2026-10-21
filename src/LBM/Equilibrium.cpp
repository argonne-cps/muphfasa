// Equilibrium.cpp — Implementations of the Maxwell-Boltzmann equilibrium.
// The second-order expansion reads:
//   f^eq_q = w_q * rho * [1 + (c_q·u)/cs2 + (c_q·u)^2/(2*cs2^2) - u^2/(2*cs2)]
// This is valid for Ma << 1; higher-order corrections can be added later.

#include "Equilibrium.H"
#include "LatticeDescriptor.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_GpuLaunch.H>

namespace muphfasa {

// ── computeEquilibriumCell ────────────────────────────────────────────────
// Reads u out of vel(i,j,k,*) and delegates to computeEquilibriumCell2D/3D.
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell(const amrex::Real rho,
                                   amrex::Array4<amrex::Real const> const& vel,
                                   int i, int j, int k, int n) noexcept
{
#if (AMREX_SPACEDIM == 2)
    const amrex::GpuArray<amrex::Real, 2> u = {vel(i, j, k, 0), vel(i, j, k, 1)};
    return computeEquilibriumCell2D<LatticeTag>(rho, u, n);
#else
    const amrex::GpuArray<amrex::Real, 3> u = {vel(i, j, k, 0), vel(i, j, k, 1), vel(i, j, k, 2)};
    return computeEquilibriumCell3D<LatticeTag>(rho, u, n);
#endif
}

// ── compute kernel ─────────────────────────────────────────────────────
template <typename LatticeTag>
void computeEquilibrium(const amrex::Box& box,
                               amrex::Array4<amrex::Real const> const& rho,
                               amrex::Array4<amrex::Real const> const& vel,
                               amrex::Array4<amrex::Real> const& feq_arr) {

        BL_PROFILE("computeEquilibrium()");

        amrex::ParallelFor(box, LatticeTag::Q, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
            feq_arr(i, j, k, n) = computeEquilibriumCell<LatticeTag>(rho(i, j, k), vel, i, j, k, n);
        });
}

// ── MultiFab-level helper ─────────────────────────────────────────────────
template <typename LatticeTag>
void computeEquilibriumMF(const amrex::MultiFab& rho_mf,
                           const amrex::MultiFab& vel_mf,
                           amrex::MultiFab&       feq_mf)
{
    BL_PROFILE("computeEquilibriumMF()");

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (amrex::MFIter mfi(feq_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto tileBox = mfi.tilebox();

        amrex::Array4<amrex::Real> const& feq_arr = feq_mf[mfi].array();
        amrex::Array4<amrex::Real const> const& rho = rho_mf[mfi].const_array();
        amrex::Array4<amrex::Real const> const& vel = vel_mf[mfi].const_array();
        computeEquilibrium<LatticeTag>(tileBox, rho, vel, feq_arr);
    }

    // sync for profiling
    amrex::Gpu::streamSynchronize();
}

// ── Explicit instantiations ───────────────────────────────────────────────
// Dimension-matched only: computeEquilibriumCell branches on AMREX_SPACEDIM.
#if (AMREX_SPACEDIM == 2)
template amrex::Real computeEquilibriumCell<D2Q9>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
#else
template amrex::Real computeEquilibriumCell<D3Q15>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
template amrex::Real computeEquilibriumCell<D3Q19>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
template amrex::Real computeEquilibriumCell<D3Q27>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
#endif

#if (AMREX_SPACEDIM == 2)
template void computeEquilibrium<D2Q9>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
#else
template void computeEquilibrium<D3Q15>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeEquilibrium<D3Q19>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeEquilibrium<D3Q27>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
#endif

#if (AMREX_SPACEDIM == 2)
template void computeEquilibriumMF<D2Q9>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
#else
template void computeEquilibriumMF<D3Q15>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q19>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q27>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
#endif

} // namespace muphfasa
