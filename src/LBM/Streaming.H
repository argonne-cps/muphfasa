// Streaming.H — Interface for the LBM streaming (propagation) step.
// Shifts each distribution f_q by one lattice site in the direction c_q.
// Requires ghost-cell-filled MultiFabs; the caller is responsible for
// calling FillBoundary / FillPatch before streaming.

#ifndef MUPHFASA_STREAMING_H
#define MUPHFASA_STREAMING_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// ── single-cell kernel ─────────────────────────────────────────────────
// Returns f_in at the upstream node for direction n: f_in(x - c_n, n).
// Callable directly from GPU device code without an AMReX ParallelFor
// context (e.g. from unit tests on host).
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real streamCell(amrex::Array4<const amrex::Real> const& f_in_arr,
                       int i, int j, int k, int n) noexcept;

// simple exact streaming in a ParallelFor
template <typename LatticeTag>
void default_stream(const amrex::Box& box,
                    amrex::Array4<const amrex::Real> const& f_in_arr,
                    amrex::Array4<amrex::Real> const& f_out_arr);

// Perform the streaming step: f_out(x) <- f_in(x - c_q) for each q.
// f_in  must have at least one layer of valid ghost cells filled.
// f_out may alias f_in only if an intermediate buffer is used internally.
template <typename LatticeTag>
void stream(const amrex::MultiFab& f_in,
            amrex::MultiFab&       f_out,
            const amrex::Geometry& geom);

// In-place streaming variant using a temporary scratch MultiFab.
template <typename LatticeTag>
void streamInPlace(amrex::MultiFab&       f,
                   const amrex::Geometry& geom);

} // namespace muphfasa

#endif // MUPHFASA_STREAMING_H
