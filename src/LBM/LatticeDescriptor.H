// LatticeDescriptor.H — Compile-time descriptors for LBM velocity sets.
// Provides D2Q9, D3Q15, D3Q19, and D3Q27 lattices: discrete velocities,
// quadrature weights, opposite-direction indices, and the lattice sound speed.
// All data is accessible from both host and GPU device code via indexed
// accessors:  LatticeTag::cx(q),  LatticeTag::weights(q),  etc.

#ifndef MUPHFASA_LATTICE_DESCRIPTOR_H
#define MUPHFASA_LATTICE_DESCRIPTOR_H

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace muphfasa {

// ── Primary template (specialisations below) ───────────────────────────────
template <int NDIM, int NQ>
struct LatticeDescriptor;

// ── D2Q9 ──────────────────────────────────────────────────────────────────
//   q:  0   1   2   3   4   5   6   7   8
// (cx): 0   1   0  -1   0   1  -1  -1   1
// (cy): 0   0   1   0  -1   1   1  -1  -1
template <>
struct LatticeDescriptor<2, 9>
{
    static constexpr int D = 2;
    static constexpr int Q = 9;
    static constexpr amrex::Real cs2    = amrex::Real(1.0) / amrex::Real(3.0);
    static constexpr amrex::Real cs2inv = amrex::Real(3.0);

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int q) noexcept {
        constexpr amrex::GpuArray<int, 9> d = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int q) noexcept {
        constexpr amrex::GpuArray<int, 9> d = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real weights(int q) noexcept {
        constexpr amrex::GpuArray<amrex::Real, 9> d = {
            amrex::Real(4.0/9.0),
            amrex::Real(1.0/9.0),  amrex::Real(1.0/9.0),
            amrex::Real(1.0/9.0),  amrex::Real(1.0/9.0),
            amrex::Real(1.0/36.0), amrex::Real(1.0/36.0),
            amrex::Real(1.0/36.0), amrex::Real(1.0/36.0)
        };
        return d[q];
    }

    // opposite[q]: index r such that c_r = -c_q  (used in bounce-back BCs)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int opposite(int q) noexcept {
        constexpr amrex::GpuArray<int, 9> d = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
        return d[q];
    }
};

// ── D3Q15 ─────────────────────────────────────────────────────────────────
template <>
struct LatticeDescriptor<3, 15>
{
    static constexpr int D = 3;
    static constexpr int Q = 15;
    static constexpr amrex::Real cs2    = amrex::Real(1.0) / amrex::Real(3.0);
    static constexpr amrex::Real cs2inv = amrex::Real(3.0);

    // TODO: fill in D3Q15 velocity data
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int /*q*/) noexcept { return 0; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int /*q*/) noexcept { return 0; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cz(int /*q*/) noexcept { return 0; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real weights(int /*q*/) noexcept { return amrex::Real(0.0); }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int opposite(int q) noexcept { return q; }
};

// ── D3Q19 ─────────────────────────────────────────────────────────────────
template <>
struct LatticeDescriptor<3, 19>
{
    static constexpr int D = 3;
    static constexpr int Q = 19;
    static constexpr amrex::Real cs2    = amrex::Real(1.0) / amrex::Real(3.0);
    static constexpr amrex::Real cs2inv = amrex::Real(3.0);

    // TODO: fill in D3Q19 velocity data
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int /*q*/) noexcept { return 0; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int /*q*/) noexcept { return 0; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cz(int /*q*/) noexcept { return 0; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real weights(int /*q*/) noexcept { return amrex::Real(0.0); }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int opposite(int q) noexcept { return q; }
};

// ── D3Q27 ─────────────────────────────────────────────────────────────────
template <>
struct LatticeDescriptor<3, 27>
{
    static constexpr int D = 3;
    static constexpr int Q = 27;
    static constexpr amrex::Real cs2    = amrex::Real(1.0) / amrex::Real(3.0);
    static constexpr amrex::Real cs2inv = amrex::Real(3.0);

    // TODO: fill in D3Q27 velocity data
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int /*q*/) noexcept { return 0; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int /*q*/) noexcept { return 0; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cz(int /*q*/) noexcept { return 0; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real weights(int /*q*/) noexcept { return amrex::Real(0.0); }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int opposite(int q) noexcept { return q; }
};

// ── Convenience aliases ────────────────────────────────────────────────────
using D2Q9  = LatticeDescriptor<2,  9>;
using D3Q15 = LatticeDescriptor<3, 15>;
using D3Q19 = LatticeDescriptor<3, 19>;
using D3Q27 = LatticeDescriptor<3, 27>;

} // namespace muphfasa

#endif // MUPHFASA_LATTICE_DESCRIPTOR_H
