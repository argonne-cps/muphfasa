// Equilibrium.cpp — Implementations of the Maxwell-Boltzmann equilibrium.
// The second-order expansion reads:
//   f^eq_q = w_q * rho * [1 + (c_q·u)/cs2 + (c_q·u)^2/(2*cs2^2) - u^2/(2*cs2)]
// This is valid for Ma << 1; higher-order corrections can be added later.

#include "Equilibrium.H"
#include "LatticeDescriptor.H"

#include <AMReX_ParallelFor.H>

namespace muphfasa {

// ── computeEquilibriumCell ────────────────────────────────────────────────
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell(const amrex::Real rho,
                                   amrex::Array4<amrex::Real const> const& vel,
                                   int i, int j, int k, int n) noexcept
{
#if (AMREX_SPACEDIM == 2)
    const amrex::Real u_sq = vel(i, j, k, 0) * vel(i, j, k, 0)
                            + vel(i, j, k, 1) * vel(i, j, k, 1);

    const amrex::Real cdotu = LatticeTag::cx(n) * vel(i, j, k, 0)
                             + LatticeTag::cy(n) * vel(i, j, k, 1);
#else
    const amrex::Real u_sq = vel(i, j, k, 0) * vel(i, j, k, 0)
                            + vel(i, j, k, 1) * vel(i, j, k, 1)
                            + vel(i, j, k, 2) * vel(i, j, k, 2);

    const amrex::Real cdotu = LatticeTag::cx(n) * vel(i, j, k, 0)
                             + LatticeTag::cy(n) * vel(i, j, k, 1)
                             + LatticeTag::cz(n) * vel(i, j, k, 2);
#endif

    return LatticeTag::weights(n) * rho *
           (1 + LatticeTag::cs2inv * cdotu
              + 0.5 * LatticeTag::cs2inv * LatticeTag::cs2inv * cdotu * cdotu
              - 0.5 * LatticeTag::cs2inv * u_sq);
}

// ── computeEquilibriumCell2D / computeEquilibriumCell3D ──────────────────
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell2D(const amrex::Real rho,
                                     amrex::GpuArray<amrex::Real, 2> const& u,
                                     int n) noexcept
{
    const amrex::Real u_sq = u[0] * u[0] + u[1] * u[1];
    const amrex::Real cdotu = LatticeTag::cx(n) * u[0] + LatticeTag::cy(n) * u[1];

    return LatticeTag::weights(n) * rho *
           (1 + LatticeTag::cs2inv * cdotu
              + 0.5 * LatticeTag::cs2inv * LatticeTag::cs2inv * cdotu * cdotu
              - 0.5 * LatticeTag::cs2inv * u_sq);
}

template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real computeEquilibriumCell3D(const amrex::Real rho,
                                     amrex::GpuArray<amrex::Real, 3> const& u,
                                     int n) noexcept
{
    const amrex::Real u_sq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const amrex::Real cdotu = LatticeTag::cx(n) * u[0] + LatticeTag::cy(n) * u[1]
                             + LatticeTag::cz(n) * u[2];

    return LatticeTag::weights(n) * rho *
           (1 + LatticeTag::cs2inv * cdotu
              + 0.5 * LatticeTag::cs2inv * LatticeTag::cs2inv * cdotu * cdotu
              - 0.5 * LatticeTag::cs2inv * u_sq);
}

// ── compute kernel ─────────────────────────────────────────────────────
template <typename LatticeTag>
void computeEquilibrium(const amrex::Box& box,
                               amrex::Array4<amrex::Real const> const& rho,
                               amrex::Array4<amrex::Real const> const& vel,
                               amrex::Array4<amrex::Real> const& feq_arr) {

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
}

// ── Explicit instantiations ───────────────────────────────────────────────
template amrex::Real computeEquilibriumCell<D2Q9> (const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
template amrex::Real computeEquilibriumCell<D3Q15>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
template amrex::Real computeEquilibriumCell<D3Q19>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);
template amrex::Real computeEquilibriumCell<D3Q27>(const amrex::Real, amrex::Array4<amrex::Real const> const&, int, int, int, int);

template amrex::Real computeEquilibriumCell2D<D2Q9>(const amrex::Real, amrex::GpuArray<amrex::Real, 2> const&, int);

template amrex::Real computeEquilibriumCell3D<D3Q15>(const amrex::Real, amrex::GpuArray<amrex::Real, 3> const&, int);
template amrex::Real computeEquilibriumCell3D<D3Q19>(const amrex::Real, amrex::GpuArray<amrex::Real, 3> const&, int);
template amrex::Real computeEquilibriumCell3D<D3Q27>(const amrex::Real, amrex::GpuArray<amrex::Real, 3> const&, int);

template void computeEquilibrium<D2Q9> (const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeEquilibrium<D3Q15>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeEquilibrium<D3Q19>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void computeEquilibrium<D3Q27>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);

template void computeEquilibriumMF<D2Q9> (const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q15>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q19>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);
template void computeEquilibriumMF<D3Q27>(const amrex::MultiFab&, const amrex::MultiFab&, amrex::MultiFab&);

} // namespace muphfasa
