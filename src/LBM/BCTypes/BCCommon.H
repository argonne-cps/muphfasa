#ifndef MUPHFASA_BC_COMMON_H
#define MUPHFASA_BC_COMMON_H

#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>

namespace muphfasa {

// Lattice velocity as IntVect: cvec<LatticeTag>(n) = IntVect(cx,cy,cz)
template <typename LatticeTag>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::IntVect cvec(int n) noexcept
{
#if (AMREX_SPACEDIM == 2)
    return amrex::IntVect{AMREX_D_DECL(LatticeTag::cx(n), LatticeTag::cy(n), 0)};
#else
    return amrex::IntVect{AMREX_D_DECL(LatticeTag::cx(n), LatticeTag::cy(n), LatticeTag::cz(n))};
#endif
}

// True iff g + c lies inside domain (population n is "needed" for streaming)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool neededPop(const amrex::IntVect& iv, const amrex::IntVect& c,
               const amrex::Box& domain) noexcept
{
    return domain.contains(iv + c);
}

} // namespace muphfasa

#endif // MUPHFASA_BC_COMMON_H
