// Collision.cpp — Implementations of LBM collision operators.
// Currently only BGK is active; the TRT and MRT stubs are placeholders
// for future extension.

#include "Collision.H"
#include "LatticeDescriptor.H"

#include <AMReX_ParallelFor.H>

namespace muphfasa {


// ── BGK ParallelFor Launch kernel ─────────────────────────────────────────────────
template <typename LatticeTag>
static void BGK::collideArr(const amrex::Box& box,
                       amrex::Array4<amrex::Real> const& f_arr,
                       amrex::Array4<const amrex::Real> const& feq_arr,
                       const amrex::Real omega)
{
    amrex::ParallelFor(box, LatticeTag::Q,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {

        // collide
        f_arr(i, j, k, n) = collide(f_arr(i, j, k, n), feq_arr(i, j, k, n), omega);
    });
}

// single cell compute kernel
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static amrex::Real BGK::collide(const amrex::Real f,
                    const amrex::Real feq,  
                    const amrex::Real omega) noexcept
{
    return (1 - omega) * f + omega * feq;
}

// ── MultiFab-level dispatch ───────────────────────────────────────────────
template <typename LatticeTag, typename CollisionModel>
void collideMF(amrex::MultiFab&       f_mf,
             const amrex::MultiFab& feq_mf,
             amrex::Real            omega)
{
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (amrex::MFIter mfi(f_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto tileBox = mfi.tilebox();

        amrex::Array4<amrex::Real> const& f_arr= f_mf[mfi].array();
        amrex::Array4<amrex::Real const> const& feq_arr = feq_mf[mfi].const_array();
        CollisionModel::template collideArr<LatticeTag>(tileBox, f_arr, feq_arr, omega);
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

} // namespace muphfasa
