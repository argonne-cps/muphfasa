// LatticeDescriptor.H — Compile-time descriptors for LBM velocity sets.
// Provides D2Q9, D3Q15, D3Q19, and D3Q27 lattices: discrete velocities,
// quadrature weights, opposite-direction indices, and the lattice sound speed.
// All data is accessible from both host and GPU device code via indexed
// accessors:  LatticeTag::cx(q),  LatticeTag::weights(q),  etc.

#ifndef MUPHFASA_LATTICE_DESCRIPTOR_H
#define MUPHFASA_LATTICE_DESCRIPTOR_H

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
    static constexpr int nGhost = 1;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int q) noexcept {
        constexpr int d[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int q) noexcept {
        constexpr int d[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real weights(int q) noexcept {
        constexpr amrex::Real d[9] = {
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
        constexpr int d[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
        return d[q];
    }
};

// ── D3Q15 ─────────────────────────────────────────────────────────────────
//   q:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
// (cx): 0   1  -1   0   0   0   0   1  -1   1  -1   1  -1  -1   1
// (cy): 0   0   0   1  -1   0   0   1  -1   1  -1  -1   1   1  -1
// (cz): 0   0   0   0   0   1  -1   1  -1  -1   1   1  -1   1  -1
//  Velocities: 1 rest, 6 face (±1,0,0),(0,±1,0),(0,0,±1), 8 corners (±1,±1,±1)
//  Weights: w0=2/9, w1-6=1/9, w7-14=1/72
template <>
struct LatticeDescriptor<3, 15>
{
    static constexpr int D = 3;
    static constexpr int Q = 15;
    static constexpr amrex::Real cs2    = amrex::Real(1.0) / amrex::Real(3.0);
    static constexpr amrex::Real cs2inv = amrex::Real(3.0);
    static constexpr int nGhost = 1;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int q) noexcept {
        constexpr int d[15] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int q) noexcept {
        constexpr int d[15] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, -1, 1, 1, -1 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cz(int q) noexcept {
        constexpr int d[15] = { 0, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real weights(int q) noexcept {
        constexpr amrex::Real d[15] = {
            amrex::Real(2.0/9.0),                    // q=0: rest
            amrex::Real(1.0/9.0),  amrex::Real(1.0/9.0),  // q=1,2: ±x
            amrex::Real(1.0/9.0),  amrex::Real(1.0/9.0),  // q=3,4: ±y
            amrex::Real(1.0/9.0),  amrex::Real(1.0/9.0),  // q=5,6: ±z
            amrex::Real(1.0/72.0), amrex::Real(1.0/72.0), // q=7,8: (1,1,1), (-1,-1,-1)
            amrex::Real(1.0/72.0), amrex::Real(1.0/72.0), // q=9,10: (1,1,-1), (-1,-1,1)
            amrex::Real(1.0/72.0), amrex::Real(1.0/72.0), // q=11,12: (1,-1,1), (-1,1,-1)
            amrex::Real(1.0/72.0), amrex::Real(1.0/72.0)  // q=13,14: (-1,1,1), (1,-1,-1)
        };
        return d[q];
    }

    // opposite[q]: index r such that c_r = -c_q  (used in bounce-back BCs)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int opposite(int q) noexcept {
        constexpr int d[15] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13 };
        return d[q];
    }
};

// ── D3Q19 ─────────────────────────────────────────────────────────────────
//   q:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
// (cx): 0   1  -1   0   0   0   0   1  -1   1  -1   0   0   0   0   1  -1   1  -1
// (cy): 0   0   0   1  -1   0   0   1  -1   0   0   1  -1   1  -1   0   0   0   0
// (cz): 0   0   0   0   0   1  -1   0   0   1  -1   1  -1  -1   1   1  -1  -1   1
//  Velocities: 1 rest, 6 face (±1,0,0),(0,±1,0),(0,0,±1), 12 edge (±1,±1,0),(±1,0,±1),(0,±1,±1)
//  Weights: w0=1/3, w1-6=1/18, w7-18=1/36
template <>
struct LatticeDescriptor<3, 19>
{
    static constexpr int D = 3;
    static constexpr int Q = 19;
    static constexpr amrex::Real cs2    = amrex::Real(1.0) / amrex::Real(3.0);
    static constexpr amrex::Real cs2inv = amrex::Real(3.0);
    static constexpr int nGhost = 1;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int q) noexcept {
        constexpr int d[19] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int q) noexcept {
        constexpr int d[19] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cz(int q) noexcept {
        constexpr int d[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1 };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real weights(int q) noexcept {
        constexpr amrex::Real d[19] = {
            amrex::Real(1.0/3.0),                        // q=0: rest
            amrex::Real(1.0/18.0), amrex::Real(1.0/18.0), // q=1,2: ±x
            amrex::Real(1.0/18.0), amrex::Real(1.0/18.0), // q=3,4: ±y
            amrex::Real(1.0/18.0), amrex::Real(1.0/18.0), // q=5,6: ±z
            amrex::Real(1.0/36.0), amrex::Real(1.0/36.0), // q=7,8: (1,1,0), (-1,-1,0)
            amrex::Real(1.0/36.0), amrex::Real(1.0/36.0), // q=9,10: (1,-1,0), (-1,1,0)
            amrex::Real(1.0/36.0), amrex::Real(1.0/36.0), // q=11,12: (0,1,1), (0,-1,-1)
            amrex::Real(1.0/36.0), amrex::Real(1.0/36.0), // q=13,14: (0,1,-1), (0,-1,1)
            amrex::Real(1.0/36.0), amrex::Real(1.0/36.0), // q=15,16: (1,0,1), (-1,0,-1)
            amrex::Real(1.0/36.0), amrex::Real(1.0/36.0)  // q=17,18: (1,0,-1), (-1,0,1)
        };
        return d[q];
    }

    // opposite[q]: index r such that c_r = -c_q  (used in bounce-back BCs)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int opposite(int q) noexcept {
        constexpr int d[19] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17 };
        return d[q];
    }
};

// ── D3Q27 ─────────────────────────────────────────────────────────────────
// Krueger Table 3.6 ordering (C1 = -1):
// i:     0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
// cix:   0 -1  1  0  0  0  0 -1  1 -1  1  0  0 -1  1 -1  1  0  0 -1  1 -1  1 -1  1  1 -1
// ciy:   0  0  0 -1  1  0  0 -1  1  0  0 -1  1  1 -1  0  0 -1  1  1  1  1  1 -1 -1  1
// ciz:   0  0  0  0  0 -1  1  0  0 -1  1 -1  1  0  0  1 -1  1  1  1  1 -1 -1  1 -1  1
//  Velocities: 1 rest, 6 face (±x,±y,±z), 12 edge, 8 corners (full 3x3x3 cube)
//  Weights: w0=8/27, w1-6=2/27, w7-18=1/54, w19-26=1/216
template <>
struct LatticeDescriptor<3, 27>
{
    static constexpr int D = 3;
    static constexpr int Q = 27;
    static constexpr amrex::Real cs2    = amrex::Real(1.0) / amrex::Real(3.0);
    static constexpr amrex::Real cs2inv = amrex::Real(3.0);
    static constexpr int nGhost = 1;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int q) noexcept {
        constexpr int d[27] = {
            0,                                  // q=0: rest
            -1, 1, 0, 0, 0, 0,                  // q=1-6: face (-x,+x,-y,+y,-z,+z)
            -1, 1, -1, 1, 0, 0, 0, 0, -1, 1, -1, 1,  // q=7-18: edge
            -1, 1, -1, 1, -1, 1, 1, -1          // q=19-26: corners
        };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int q) noexcept {
        constexpr int d[27] = {
            0,                                  // q=0: rest
            0, 0, -1, 1, 0, 0,                  // q=1-6: face
            -1, 1, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0,  // q=7-18: edge
            1, 1, 1, 1, -1, -1, -1, 1           // q=19-26: corners
        };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cz(int q) noexcept {
        constexpr int d[27] = {
            0,                                  // q=0: rest
            0, 0, 0, 0, -1, 1,                  // q=1-6: face
            0, 0, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1,  // q=7-18: edge
            -1, -1, 1, -1, 1, 1, -1, 1          // q=19-26: corners
        };
        return d[q];
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real weights(int q) noexcept {
        constexpr amrex::Real d[27] = {
            amrex::Real(8.0/27.0),                        // q=0: rest
            amrex::Real(2.0/27.0), amrex::Real(2.0/27.0), // q=1,2: -x,+x
            amrex::Real(2.0/27.0), amrex::Real(2.0/27.0), // q=3,4: -y,+y
            amrex::Real(2.0/27.0), amrex::Real(2.0/27.0), // q=5,6: -z,+z
            amrex::Real(1.0/54.0), amrex::Real(1.0/54.0), // q=7,8: (-1,-1,0), (1,1,0)
            amrex::Real(1.0/54.0), amrex::Real(1.0/54.0), // q=9,10: (-1,0,-1), (1,0,1)
            amrex::Real(1.0/54.0), amrex::Real(1.0/54.0), // q=11,12: (0,-1,-1), (0,1,1)
            amrex::Real(1.0/54.0), amrex::Real(1.0/54.0), // q=13,14: (-1,1,0), (1,-1,0)
            amrex::Real(1.0/54.0), amrex::Real(1.0/54.0), // q=15,16: (-1,0,1), (1,0,-1)
            amrex::Real(1.0/54.0), amrex::Real(1.0/54.0), // q=17,18: (0,-1,1), (0,1,-1)
            amrex::Real(1.0/216.0), amrex::Real(1.0/216.0), // q=19,20: (-1,-1,-1), (1,1,-1)
            amrex::Real(1.0/216.0), amrex::Real(1.0/216.0), // q=21,22: (-1,1,-1), (1,-1,-1)
            amrex::Real(1.0/216.0), amrex::Real(1.0/216.0), // q=23,24: (1,1,1), (-1,-1,1)
            amrex::Real(1.0/216.0), amrex::Real(1.0/216.0)  // q=25,26: (1,-1,1), (-1,1,1)
        };
        return d[q];
    }

    // opposite[q]: index r such that c_r = -c_q  (used in bounce-back BCs)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int opposite(int q) noexcept {
        constexpr int d[27] = {
            0,  2,  1,  4,  3,  6,  5,  // q=0-6: rest, (-x↔+x), (-y↔+y), (-z↔+z)
            8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17,  // q=7-18: edge opposites
            23, 24, 25, 26, 19, 20, 21, 22                   // q=19-26: corner opposites
        };
        return d[q];
    }
};

// ── Convenience aliases ────────────────────────────────────────────────────
using D2Q9  = LatticeDescriptor<2,  9>;
using D3Q15 = LatticeDescriptor<3, 15>;
using D3Q19 = LatticeDescriptor<3, 19>;
using D3Q27 = LatticeDescriptor<3, 27>;

} // namespace muphfasa

#endif // MUPHFASA_LATTICE_DESCRIPTOR_H
