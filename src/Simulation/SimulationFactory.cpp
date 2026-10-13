// SimulationFactory.cpp — Runtime -> compile-time lattice dispatch.

#include "SimulationFactory.H"

#include "ISimulation.H"
#include "SimulationSetup.H"
#include "IO/InputParser.H"
#include "LBM/Collision.H"
#include "LBM/LatticeDescriptor.H"

#include <AMReX.H>

namespace muphfasa {

// Second stage of the dispatch: LatticeTag is now fixed, so only the
// collision model (read from lbm.collision_model) remains to be resolved.
// Only BGK has an implemented kernel today (see Collision.cpp); TRT/MRT are
// declared tags but abort until their collide()/collideArr() are written.
template <typename LatticeTag>
static std::unique_ptr<ISimulation> makeSimulationForLattice(const InputParser& params)
{
    const std::string collision = params.collisionModel();

    if (collision == "BGK") {
        return std::make_unique<SimulationImpl<LatticeTag, BGK>>(params);
    }
    if (collision == "TRT" || collision == "MRT") {
        amrex::Abort("lbm.collision_model '" + collision +
                     "' is a recognised tag but not yet implemented.");
    }
    amrex::Abort("Unknown lbm.collision_model '" + collision +
                 "' (expected BGK | TRT | MRT).");
    return nullptr; // unreachable: amrex::Abort does not return
}

std::unique_ptr<ISimulation> makeSimulation(const InputParser& params)
{
    const std::string lattice = params.latticeType();

#if (AMREX_SPACEDIM == 2)
    if (lattice == "D2Q9") {
        return makeSimulationForLattice<D2Q9>(params);
    }
    amrex::Abort("Unknown or unsupported lbm.lattice_type '" + lattice +
                 "' for a 2D build (only D2Q9 is available).");
#else
    if (lattice == "D3Q15") {
        return makeSimulationForLattice<D3Q15>(params);
    }
    if (lattice == "D3Q19") {
        return makeSimulationForLattice<D3Q19>(params);
    }
    if (lattice == "D3Q27") {
        return makeSimulationForLattice<D3Q27>(params);
    }
    amrex::Abort("Unknown or unsupported lbm.lattice_type '" + lattice +
                 "' for a 3D build (expected D3Q15 | D3Q19 | D3Q27).");
#endif
    return nullptr; // unreachable: amrex::Abort does not return
}

} // namespace muphfasa
