// Collision.cpp — Implementations of LBM collision operators.
// Currently only BGK is active; the TRT and MRT stubs are placeholders
// for future extension.

#include "Collision.H"
#include "Streaming.H"
#include "Equilibrium.H"
#include "LatticeDescriptor.H"
#include "Macroscopic.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_GpuLaunch.H>

namespace muphfasa {


// ── BGK::collide (single-cell kernel) ─────────────────────────────────────
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real BGK::collide(const amrex::Real f,
                         const amrex::Real feq,
                         const amrex::Real omega) noexcept
{
    return (1 - omega) * f + omega * feq;
}

// ── pullCollideFusedArr (per-tile kernel) ──────────────────────────────────────
// Pull (stream) then Fuses rho/u/feq into per-cell scratch (no intermediate MultiFabs).
template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedArr(const amrex::Box& box,
                    amrex::Array4<const amrex::Real> const& fstar_in_arr,
                    amrex::Array4<amrex::Real> const& fstar_out_arr,
                    amrex::Real omega)
{
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

        // scratch array
        amrex::GpuArray<amrex::Real, LatticeTag::Q> f;
        // stream f*_in --> f (scratch)
        for (int n=0; n < LatticeTag::Q; ++n){
            f[n] = streamCell<LatticeTag>(fstar_in_arr, i, j, k, n);
        }

        // calc rho
        const amrex::Real rho = computeDensityCell<LatticeTag>(f);

        // calc u 
#if (AMREX_SPACEDIM == 2)
        const auto u = computeVelocityCell2D<LatticeTag>(f, rho);
#else
        const auto u = computeVelocityCell3D<LatticeTag>(f, rho);
#endif

        // calc fstar for each comp
        for (int n = 0; n < LatticeTag::Q; ++n) {
#if (AMREX_SPACEDIM == 2)
            const amrex::Real feq_n = computeEquilibriumCell2D<LatticeTag>(rho, u, n);
#else
            const amrex::Real feq_n = computeEquilibriumCell3D<LatticeTag>(rho, u, n);
#endif
            fstar_out_arr(i, j, k, n) = CollisionModel::collide(f[n], feq_n, omega);
        }
    });
}

template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedArr(const amrex::Box& box,
                    amrex::Array4<const amrex::Real> const& fstar_in_arr,
                    amrex::Array4<amrex::Real> const& fstar_out_arr,
                    amrex::Real omega,
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F)
{
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

        // scratch array
        amrex::GpuArray<amrex::Real, LatticeTag::Q> f;
        // stream f*_in --> f (scratch)
        for (int n=0; n < LatticeTag::Q; ++n){
            f[n] = streamCell<LatticeTag>(fstar_in_arr, i, j, k, n);
        }

        // calc rho
        const amrex::Real rho = computeDensityCell<LatticeTag>(f);

        // calc u (half-force corrected, Guo scheme)
#if (AMREX_SPACEDIM == 2)
        const auto u = computeVelocityCell2D<LatticeTag>(f, rho, F);
#else
        const auto u = computeVelocityCell3D<LatticeTag>(f, rho, F);
#endif

        // calc fstar for each comp
        for (int n = 0; n < LatticeTag::Q; ++n) {
#if (AMREX_SPACEDIM == 2)
            const amrex::Real feq_n = computeEquilibriumCell2D<LatticeTag>(rho, u, n);
#else
            const amrex::Real feq_n = computeEquilibriumCell3D<LatticeTag>(rho, u, n);
#endif
            fstar_out_arr(i, j, k, n) = CollisionModel::collide(f[n], feq_n, omega)
                                      + guoForceTerm<LatticeTag>(n, u, F, omega);
        }
    });
}

// ── MultiFab-level Pull - Full Collide dispatch ──────────────────────────────────
template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedMF(const amrex::MultiFab&       fin_mf,
                       amrex::MultiFab& fout_mf,
                       amrex::Real            omega)
{

    BL_PROFILE("pullCollideFusedMF()");

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (amrex::MFIter mfi(fin_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box tileBox = mfi.tilebox();

        amrex::Array4<amrex::Real const> const& fstar_in_arr = fin_mf[mfi].const_array();
        amrex::Array4<amrex::Real> const& fstar_out_arr = fout_mf[mfi].array();

        pullCollideFusedArr<LatticeTag, CollisionModel>(tileBox, fstar_in_arr, fstar_out_arr, omega);
    }

    // sync for profiling and safety
    amrex::Gpu::streamSynchronize();
}

// ── MultiFab-level Pull - Full Collide dispatch ──────────────────────────────────
template <typename LatticeTag, typename CollisionModel>
void pullCollideFusedMF(const amrex::MultiFab&       fin_mf,
                       amrex::MultiFab& fout_mf,
                       amrex::Real     omega,
                       amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& F)
{

    BL_PROFILE("pullCollideFusedMF(forced)");

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (amrex::MFIter mfi(fin_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box tileBox = mfi.tilebox();

        amrex::Array4<amrex::Real const> const& fstar_in_arr = fin_mf[mfi].const_array();
        amrex::Array4<amrex::Real> const& fstar_out_arr = fout_mf[mfi].array();

        pullCollideFusedArr<LatticeTag, CollisionModel>(tileBox, fstar_in_arr, fstar_out_arr, omega, F);
    }

    // sync for profiling and safety
    amrex::Gpu::streamSynchronize();
}

// Dimension-matched only: collideFullArr/MF branch on AMREX_SPACEDIM.
#if (AMREX_SPACEDIM == 2)
template void pullCollideFusedArr<D2Q9, BGK>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Real);
template void pullCollideFusedArr<D2Q9, BGK>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&);
#else
template void pullCollideFusedArr<D3Q15,BGK>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Real);
template void pullCollideFusedArr<D3Q19,BGK>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Real);
template void pullCollideFusedArr<D3Q27,BGK>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Real);
template void pullCollideFusedArr<D3Q15,BGK>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&);
template void pullCollideFusedArr<D3Q19,BGK>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&);
template void pullCollideFusedArr<D3Q27,BGK>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&);
#endif

#if (AMREX_SPACEDIM == 2)
template void pullCollideFusedMF<D2Q9,BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::Real);
template void pullCollideFusedMF<D2Q9,BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&);
#else
template void pullCollideFusedMF<D3Q15,BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::Real);
template void pullCollideFusedMF<D3Q19,BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::Real);
template void pullCollideFusedMF<D3Q27,BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::Real);
template void pullCollideFusedMF<D3Q15,BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&);
template void pullCollideFusedMF<D3Q19,BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&);
template void pullCollideFusedMF<D3Q27,BGK>(const amrex::MultiFab&, amrex::MultiFab&, amrex::Real, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&);
#endif

} // namespace muphfasa
