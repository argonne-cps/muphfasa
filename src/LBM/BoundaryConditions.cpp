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

lbmBCtype parselbmBCtype(const std::string& name)
{
    if (name == "MovingWall")  { return lbmBCtype::MovingWall;  }
    if (name == "NoSlip")      { return lbmBCtype::NoSlip;  }
    if (name == "Inflow")      { return lbmBCtype::Inflow;  }
    if (name == "Outflow")     { return lbmBCtype::Outflow; }
    amrex::Abort("parselbmBCtype: unrecognized BC name \"" + name +
                 "\" (expected MovingWall, NoSlip, Inflow, or Outflow; Periodic faces "
                 "must be filtered out before calling this)");
    return lbmBCtype::NoSlip; // unreachable
}

amrex::Vector<amrex::BCRec> makeBCRecVector(
    const std::array<std::string, 2 * AMREX_SPACEDIM>& names, int ncomp)
{
    amrex::BCRec bcr;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const std::string& lo_name = names[faceIndex(dir, -1)];
        const std::string& hi_name = names[faceIndex(dir, 1)];

        bcr.setLo(dir, (lo_name == "Periodic") ? amrex::BCType::int_dir
                                                : kCustomBCSentinel);
        bcr.setHi(dir, (hi_name == "Periodic") ? amrex::BCType::int_dir
                                                : kCustomBCSentinel);
    }
    return amrex::Vector<amrex::BCRec>(ncomp, bcr);
}

#if (AMREX_SPACEDIM == 2)
template class BCManager<D2Q9>;
#else
template class BCManager<D3Q15>;
template class BCManager<D3Q19>;
template class BCManager<D3Q27>;
#endif

} // namespace muphfasa
