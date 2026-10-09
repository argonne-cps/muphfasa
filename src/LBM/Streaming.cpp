// Streaming.cpp — Implementation of the LBM streaming (propagation) step.
// Performs: f_out(x, q) = f_in(x - c_q, q) for every cell and direction.
// Relies on ghost cells being pre-filled by the caller (FillBoundary or
// FillPatch), so that cells adjacent to MPI boundaries are available.

#include "Streaming.H"
#include "LatticeDescriptor.H"

#include <AMReX_ParallelFor.H>

namespace muphfasa {

template <typename LatticeTag>
void default_stream(const amrex::Box& box,
                    amrex::Array4<const amrex::Real> const& f_in_arr,
                    amrex::Array4<amrex::Real> const& f_out_arr) {

    amrex::ParallelFor(box, LatticeTag::Q, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
        // find node to stream from
        int i_old = i - LatticeTag::cx(n);
        int j_old = j - LatticeTag::cy(n);
#if (AMREX_SPACEDIM == 3)
        int k_old = k - LatticeTag::cz(n);
#else
        int k_old = k;
#endif

        // stream
        f_out_arr(i, j, k, n) = f_in_arr(i_old, j_old, k_old, n);
    });
}

// ── Two-MultiFab streaming ────────────────────────────────────────────────
template <typename LatticeTag>
void stream(const amrex::MultiFab& f_in,
            amrex::MultiFab&       f_out,
            const amrex::Geometry& geom)
{
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    for (amrex::MFIter mfi(f_in, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto tileBox = mfi.tilebox();

        amrex::Array4<amrex::Real> const& fo_arr = f_out[mfi].array();
        amrex::Array4<amrex::Real const> const& fi_arr = f_in[mfi].const_array();

        default_stream<LatticeTag>(tileBox, fi_arr, fo_arr);
    }
}

// ── In-place streaming (using scratch space) ──────────────────────────────
template <typename LatticeTag>
void streamInPlace(amrex::MultiFab&       f,
                   const amrex::Geometry& geom)
{
    amrex::MultiFab f_scratch(f.boxArray(), f.DistributionMap(),
                               f.nComp(), f.nGrow());
    stream<LatticeTag>(f, f_scratch, geom);
    amrex::MultiFab::Copy(f, f_scratch, 0, 0, f.nComp(), 0);
}

// ── Explicit instantiations ───────────────────────────────────────────────
template void default_stream<D2Q9>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void default_stream<D3Q15>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void default_stream<D3Q19>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);
template void default_stream<D3Q27>(const amrex::Box&, amrex::Array4<const amrex::Real> const&, amrex::Array4<amrex::Real> const&);

template void stream<D2Q9> (const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
template void stream<D3Q15>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
template void stream<D3Q19>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);
template void stream<D3Q27>(const amrex::MultiFab&, amrex::MultiFab&, const amrex::Geometry&);

template void streamInPlace<D2Q9> (amrex::MultiFab&, const amrex::Geometry&);
template void streamInPlace<D3Q15>(amrex::MultiFab&, const amrex::Geometry&);
template void streamInPlace<D3Q19>(amrex::MultiFab&, const amrex::Geometry&);
template void streamInPlace<D3Q27>(amrex::MultiFab&, const amrex::Geometry&);

} // namespace muphfasa
