// BoundaryConditions.cpp — Explicit instantiations for BCManager<LatticeTag>.
//
// BCManager is a header-only template (all of operator()/setFaceBC must be
// visible to AMReX's GpuBndryFuncFab at the call site, and operator() must be
// inlineable for GPU builds), so there is nothing to implement out-of-line
// here. These instantiations exist purely to catch lattice-specific compile
// errors early, mirroring the explicit-instantiation pattern used elsewhere
// in src/LBM/.

#include "BoundaryConditions.H"
#include "LatticeDescriptor.H"

namespace muphfasa {

#if (AMREX_SPACEDIM == 2)
template class BCManager<D2Q9>;
#else
template class BCManager<D3Q15>;
template class BCManager<D3Q19>;
template class BCManager<D3Q27>;
#endif

} // namespace muphfasa
