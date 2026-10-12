// Collision.cpp — Implementations of LBM collision operators.
// Currently only BGK is active; the TRT and MRT stubs are placeholders
// for future extension.

#include "Collision.H"
#include "LatticeDescriptor.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelFor.H>

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

        amrex::Array4<amrex::Real> const& f_arr= f_mf[mfi].array();
        amrex::Array4<amrex::Real> const& f_arr= f_mf[mfi].array();
        amrex::Array4<amrex::Real const> const& feq_arr = feq_mf[mfi].const_array();
        CollisionModel::template collideArr<LatticeTag>(tileBox, f_arr, feq_arr, omega);
    }
}

// ── Full Collide ParallelFor ───────────────────────────────────────────────
template <typename LatticeTag, typename CollisionModel>
void collideFullArr(const amrex::Box box,
                    amrex::Array4<amrex::Real> const& fstar_arr,
                    amrex::Array4<amrex::Real const> const& f_arr,
                    amrex::Array4<amrex::Real> const& feq_arr,
                    amrex::Array4<amrex::Real> const& rho_arr,
                    amrex::Array4<amrex::Real> const& vel_arr,
                    amrex::Real omega)
{
        BL_PROFILE("collideFullArr()");

        // launch kernel
        amrex::ParallelFor(box,[=] AMREX_GPU_DEVICE(int i, int j, int k){

            //calc rho
            rho_arr(i, j, k) = computeDensityCell<LatticTag>(f_arr, i, j, k);

            // calc u
            computeVelocityCell<LatticeTag>(f_arr, rho(i, j, k), vel_arr, i, j, k);

            // calc feq for each comp
            for (int n=0; n < LatticeTag::Q; ++n){

                feq_arr(i, j, k, n) = computeEquilibriumCell(rho(i, j, k),
                                                              vel_arr,
                                                              i, j, k, n)
            }
                         
            // calc fe for each comp
            for (int n=0; n < LatticeTag::Q; ++n){
                fstar_arr(i, j, k, n) = CollisionModel::collide(f_arr(i, j, k, n), feq_arr(i, j, k, n), omega);
            }
 
        });
}

// ── MultiFab-level Full Collide dispatch ───────────────────────────────────────────────
template <typename LatticeTag, typename CollisionModel>
void collideFullMF(amrex::MultiFab& fstar_mf,
             const amrex::MultiFab& f_mf,
             amrex::MultiFab& feq_mf,
             amrex::MultiFab& rho_mf,
             amrex::MultiFab& vel_mf,
             amrex::Real omega)
{
    BL_PROFILE("collideFullMF()");

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (amrex::MFIter mfi(f_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box tileBox = mfi.tilebox();

        // load arrays
        amrex::Array4<amrex::Real> const& fstar_arr = fstar_mf[mfi].array();
        amrex::Array4<amrex::Real const> const& f_arr = f_mf[mfi].const_array();
        amrex::Array4<amrex::Real> const& feq_arr = feq_mf[mfi].array();
        amrex::Array4<amrex::Real> const& rho_arr= rho_mf[mfi].array();
        amrex::Array4<amrex::Real> const& vel_arr = vel_mf[mfi].array();

        collideFullArr<LatticeTag, CollisionModel>(tileBox, fstar_arr, f_arr, feq_arr, rho_arr, vel_arr, omega);
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

template void BGK::collideFullArr<D2Q9>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real const> const&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real> const&, const amrex::Real);
template void BGK::collideFullArr<D3Q15>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real const> const&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real> const&, const amrex::Real);
template void BGK::collideFullArr<D3Q19>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real const> const&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real> const&, const amrex::Real);
template void BGK::collideFullArr<D3Q27>(const amrex::Box&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real const> const&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real> const&, amrex::Array4<amrex::Real> const&, const amrex::Real);

template void collideFullMF<D2Q9, BGK> (amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real);
template void collideFullMF<D3Q15,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real);
template void collideFullMF<D3Q19,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real);
template void collideFullMF<D3Q27,BGK>(amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&, amrex::Real);

} // namespace muphfasa
