// OutflowKernel.H — Zero-gradient / extrapolation outflow BC, ghost-fill
// form, masked to the unknown-direction populations only.
//
// The face sits halfway between the last fluid node x and the first ghost
// node g = x + bound*e_dir (bound = -1 for a lo-`dir` face, +1 for a
// hi-`dir` face). This kernel is only ever invoked by AMReX for ghost cells
// so `iv` below is always such a `g`.
//
// For every population n whose velocity points into the face along `dir`
// (c_n[dir] == -bound, i.e. n would stream out of the domain and so is
// otherwise unknown), we assume zero gradient at the boundary and copy that
// same population directly from the adjacent fluid cell x = g - bound*e_dir
// (no reflection/opposite(), unlike NoSlip/MovingWall/Inflow):
//
//     f(g, n) = f(x, n)
//
// so that stream()'s pull step reproduces f(x, n) unchanged at x, i.e.
// df/dn = 0 across the face.

#ifndef MUPHFASA_OUTFLOW_KERNEL_H
#define MUPHFASA_OUTFLOW_KERNEL_H

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

namespace muphfasa {

template <typename LatticeTag>
struct OutflowKernel
{
    // dir: face normal (0=x, 1=y, 2=z). bound: -1 for the lo face, +1 for
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
                    f(iv, n) = f(iv[0]-bound, iv[1], n);
                }
            } else if (dir == 1) {
                if (LatticeTag::cy(n) == -bound) {
                    f(iv, n) = f(iv[0], iv[1]-bound, n);
                }
            }
#else
            if (dir == 0) {
                if (LatticeTag::cx(n) == -bound) {
                    f(iv, n) = f(iv[0]-bound, iv[1], iv[2], n);
                }
            } else if (dir == 1) {
                if (LatticeTag::cy(n) == -bound) {
                    f(iv, n) = f(iv[0], iv[1]-bound, iv[2], n);
                }
            } else if (dir == 2) {
                if (LatticeTag::cz(n) == -bound) {
                    f(iv, n) = f(iv[0], iv[1], iv[2]-bound, n);
                }
            }
#endif
        }
    }
};

} // namespace muphfasa

#endif // MUPHFASA_OUTFLOW_KERNEL_H
