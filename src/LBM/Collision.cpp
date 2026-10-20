// Collision.cpp — Implementations of LBM collision operators.
// Currently only BGK is active; the TRT and MRT stubs are placeholders
// for future extension.

#include "Collision.H"
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

// ── BGK::collideArr (array-level kernel) ──────────────────────────────────
template <typename LatticeTag>
void BGK::collideArr(const amrex::Box& box,
                     amrex::Array4<amrex::Real> const& f_arr,
                     amrex::Array4<const amrex::Real> const& feq_arr,
                     const amrex::Real omega)
{
    BL_PROFILE("BGK::collideArr()");

    amrex::ParallelFor(box, LatticeTag::Q,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
        f_arr(i, j, k, n) = BGK::collide(f_arr(i, j, k, n), feq_arr(i, j, k, n), omega);
    });
}

// ── MultiFab-level dispatch ───────────────────────────────────────────────
template <typename LatticeTag, typename CollisionModel>
void collideMF(amrex::MultiFab&       f_mf,
             const amrex::MultiFab& feq_mf,
             amrex::Real            omega)
{
    BL_PROFILE("collideMF()");

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (amrex::MFIter mfi(f_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto tileBox = mfi.tilebox();

        amrex::Array4<amrex::Real> const& f_arr = f_mf[mfi].array();
        amrex::Array4<amrex::Real const> const& feq_arr = feq_mf[mfi].const_array();
        CollisionModel::template collideArr<LatticeTag>(tileBox, f_arr, feq_arr, omega);
    }
}

// ── collideFullArr (per-tile kernel) ──────────────────────────────────────
// Fuses rho/u/feq into per-cell scratch (no intermediate MultiFabs).
template <typename LatticeTag, typename CollisionModel>
void collideFullArr(const amrex::Box& box,
                    amrex::Array4<amrex::Real> const& fstar_arr,
                    amrex::Array4<const amrex::Real> const& f_arr,
                    amrex::Real omega)
{
        BL_PROFILE("collideFullArr()");

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

        // calc rho
        const amrex::Real rho = computeDensityCell<LatticeTag>(f_arr, i, j, k);

        // calc u 
#if (AMREX_SPACEDIM == 2)
        const auto u = computeVelocityCell2D<LatticeTag>(f_arr, rho, i, j, k);
#else
        const auto u = computeVelocityCell3D<LatticeTag>(f_arr, rho, i, j, k);
#endif

        // calc fstar for each comp
        for (int n = 0; n < LatticeTag::Q; ++n) {
#if (AMREX_SPACEDIM == 2)
            const amrex::Real feq = computeEquilibriumCell2D<LatticeTag>(rho, u, n);
#else
            const amrex::Real feq = computeEquilibriumCell3D<LatticeTag>(rho, u, n);
#endif
            fstar_arr(i, j, k, n) = CollisionModel::collide(f_arr(i, j, k, n), feq, omega);
        }
    });
}

// ── MultiFab-level Full Collide dispatch ──────────────────────────────────
template <typename LatticeTag, typename CollisionModel>
void collideFullMF(amrex::MultiFab&       fstar_mf,
                   const amrex::MultiFab& f_mf,
                   amrex::Real            omega)
{
    BL_PROFILE("collideFullMF()");

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (amrex::MFIter mfi(f_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box tileBox = mfi.tilebox();

        amrex::Array4<amrex::Real> const& fstar_arr = fstar_mf[mfi].array();
        amrex::Array4<amrex::Real const> const& f_arr = f_mf[mfi].const_array();

        collideFullArr<LatticeTag, CollisionModel>(tileBox, fstar_arr, f_arr, omega);
    }
}


// ── Explicit instantiations — BGK ────────────────────────────────────────
template void BGK::collideArr<D2Q9>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, const amrex::Real);
template void BGK::collideArr<D3Q15>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, const amrex::Real);
template void BGK::collideArr<D3Q19>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, const amrex::Real);
template void BGK::collideArr<D3Q27>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, const amrex::Real);

template void collideMF<D2Q9, BGK> (amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
template void collideMF<D3Q15,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
template void collideMF<D3Q19,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
template void collideMF<D3Q27,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);

// Dimension-matched only: collideFullArr/MF branch on AMREX_SPACEDIM.
#if (AMREX_SPACEDIM == 2)
template void collideFullArr<D2Q9, BGK>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Real);
#else
template void collideFullArr<D3Q15,BGK>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Real);
template void collideFullArr<D3Q19,BGK>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Real);
template void collideFullArr<D3Q27,BGK>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Real);
#endif

#if (AMREX_SPACEDIM == 2)
template void collideFullMF<D2Q9, BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
#else
template void collideFullMF<D3Q15,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
template void collideFullMF<D3Q19,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
template void collideFullMF<D3Q27,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::Real);
#endif

} // namespace muphfasa
