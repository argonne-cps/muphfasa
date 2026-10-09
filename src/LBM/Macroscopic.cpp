// Macroscopic.cpp — Implementations of density and velocity moment calculations.
// Per-cell kernels are defined here as explicit template instantiations so that
// the GPU compiler can see their bodies via the header.  MultiFab-level helpers
// drive AMReX parallel loops over the MFIter / ParallelFor infrastructure.

#include "Macroscopic.H"
#include "LatticeDescriptor.H"

#include <AMReX_ParallelFor.H>

namespace muphfasa {

// ── computeDensity ────────────────────────────────────────────────────────
template <typename LatticeTag>
void computeDensity(const amrex::Box& box,
                    amrex::Array4<const amrex::Real> const& f_arr,
                    amrex::Array4<amrex::Real> const& rho_arr)

{
    // TODO: return sum_q f[q]
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        rho_arr(i, j, k) = 0.;
        for (int n=0; n < LatticeTag::Q; ++n){
            rho_arr(i, j, k) += f_arr(i, j, k, n);
        }
    });
}

// ── computeVelocity ───────────────────────────────────────────────────────

template <typename LatticeTag>
void computeVelocity(const amrex::Box& box,
                     amrex::Array4<const amrex::Real> const& f_arr,
                     amrex::Array4<const amrex::Real> const& rho_arr,
                     amrex::Array4<amrex::Real> const& vel_arr)
{
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        // set vel to zero
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim){
            vel_arr(i, j, k, idim) = 0.;
        }

        // calc momentum 
        for (int n=0; n < LatticeTag::Q; ++n){
            vel_arr(i, j, k, 0) += LatticeTag::cx(n) * f_arr(i, j, k, n);
            vel_arr(i, j, k, 1) += LatticeTag::cy(n) * f_arr(i, j, k, n);
#if (AMREX_SPACEDIM == 3)
            vel_arr(i, j, k, 2) += LatticeTag::cz(n) * f_arr(i, j, k, n);
#endif  
        }

        // calc velocity
        for (int idim=0; idim < AMREX_SPACEDIM; ++idim){
            vel_arr(i, j, k, idim) /= rho_arr(i, j, k);
        }
    });
}


// ── computeMacroscopic (MultiFab) ─────────────────────────────────────────
template <typename LatticeTag>
void computeMacroscopic(const amrex::MultiFab& f_mf,
                        amrex::MultiFab&       rho_mf,
                        amrex::MultiFab&       vel_mf)
{
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
}

// ── Explicit instantiations ───────────────────────────────────────────────
template void computeDensity<D2Q9>(const amrex::Box, amrex::Array4<const amrex::Real>, amrex::Array4<amrex::Real> const);
template void computeDensity<D3Q15>(const amrex::Box, amrex::Array4<const amrex::Real>, amrex::Array4<amrex::Real> const);
template void computeDensity<D3Q19>(const amrex::Box, amrex::Array4<const amrex::Real>, amrex::Array4<amrex::Real> const);
template void computeDensity<D3Q27>(const amrex::Box, amrex::Array4<const amrex::Real>, amrex::Array4<amrex::Real> const);

template void computeVelocity<D2Q9>(const amrex::Box, amrex::Array4<const amrex::Real>, amrex::Array4<const amrex::Real>, amrex::Array4<amrex::Real>);
template void computeVelocity<D3Q15>(const amrex::Box, amrex::Array4<const amrex::Real>, amrex::Array4<const amrex::Real>, amrex::Array4<amrex::Real>);
template void computeVelocity<D3Q19>(const amrex::Box, amrex::Array4<const amrex::Real>, amrex::Array4<const amrex::Real>, amrex::Array4<amrex::Real>);
template void computeVelocity<D3Q27>(const amrex::Box, amrex::Array4<const amrex::Real>, amrex::Array4<const amrex::Real>, amrex::Array4<amrex::Real>);

template void computeMacroscopic<D2Q9>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&);
template void computeMacroscopic<D3Q15>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&);
template void computeMacroscopic<D3Q19>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&);
template void computeMacroscopic<D3Q27>(const amrex::MultiFab&, amrex::MultiFab&, amrex::MultiFab&);

} // namespace muphfasa
