// NoSlipKernel.H — Halfway bounce-back, ghost-fill form.
//
// Derivation (BC_implementation_plan.md Step 1):
//
// The wall sits halfway between the last fluid node x and the first ghost
// node g = x + bound*e_dir (bound = -1 for a lo-`dir` wall, +1 for a hi-`dir`
// wall). This kernel is only ever invoked by AMReX for ghost cells (see the
// `BCManager` contract), so `iv` below is always such a `g`.
//
// We want the ghost cell to hold, for every population n whose velocity
// points into the wall along `dir` (c_n[dir] == -bound, i.e. n would stream
// out of the domain), the opposite-direction population already sitting in
// the adjacent fluid cell x = g - bound*e_dir (pre-stream):
//
//     f(g, n) = f(x, opposite(n))
//
// so that stream()'s pull step, f_out(x, n) = f_in(x - c_n, n) = f_in(g, n),
// reproduces f(x, opposite(n)) arriving back into x along n — the
// bounce-back. See Kruger et al., "The Lattice Boltzmann Method", ch. 5.

#ifndef MUPHFASA_NOSLIP_KERNEL_H
#define MUPHFASA_NOSLIP_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

namespace muphfasa {

template <typename LatticeTag>
struct NoSlipKernel
{
    // dir: wall normal (0=x, 1=y, 2=z). bound: -1 for the lo face, +1 for
    // the hi face. iv is the ghost cell being filled.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(const amrex::IntVect& iv,
                       amrex::Array4<amrex::Real> const& f,
                       int dir, int bound) noexcept
    {
        for (int n = 0; n < LatticeTag::Q; ++n) {
#if (AMREX_SPACEDIM == 2)
            if (dir == 0) {
                if (LatticeTag::cx(n) == -bound) {
                    f(iv, n) = f(iv[0]-bound, iv[1], LatticeTag::opposite(n));
                }
            } else if (dir == 1) {
                if (LatticeTag::cy(n) == -bound) {
                    f(iv, n) = f(iv[0], iv[1]-bound, LatticeTag::opposite(n));
                }
            }
#else
            if (dir == 0) {
                if (LatticeTag::cx(n) == -bound) {
                    f(iv, n) = f(iv[0]-bound, iv[1], iv[2], LatticeTag::opposite(n));
                }
            } else if (dir == 1) {
                if (LatticeTag::cy(n) == -bound) {
                    f(iv, n) = f(iv[0], iv[1]-bound, iv[2], LatticeTag::opposite(n));
                }
            } else if (dir == 2) {
                if (LatticeTag::cz(n) == -bound) {
                    f(iv, n) = f(iv[0], iv[1], iv[2]-bound, LatticeTag::opposite(n));
                }
            }
#endif
        }
    }
};

} // namespace muphfasa

#endif // MUPHFASA_NOSLIP_KERNEL_H
