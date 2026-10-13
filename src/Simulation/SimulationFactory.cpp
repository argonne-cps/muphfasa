// SimulationFactory.cpp — Runtime -> compile-time lattice dispatch.

#include "SimulationFactory.H"

#include "ISimulation.H"
#include "SimulationSetup.H"
#include "IO/InputParser.H"
#include "LBM/LatticeDescriptor.H"

#include <AMReX.H>

namespace muphfasa {

std::unique_ptr<ISimulation> makeSimulation(const InputParser& params)
{
    const std::string lattice = params.latticeType();

#if (AMREX_SPACEDIM == 2)
    if (lattice == "D2Q9") {
        return std::make_unique<SimulationImpl<D2Q9>>(params);
    }
    amrex::Abort("Unknown or unsupported lbm.lattice_type '" + lattice +
                 "' for a 2D build (only D2Q9 is available).");
#else
    if (lattice == "D3Q15") {
        return std::make_unique<SimulationImpl<D3Q15>>(params);
    }
    if (lattice == "D3Q19") {
        return std::make_unique<SimulationImpl<D3Q19>>(params);
    }
    if (lattice == "D3Q27") {
        return std::make_unique<SimulationImpl<D3Q27>>(params);
    }
    amrex::Abort("Unknown or unsupported lbm.lattice_type '" + lattice +
                 "' for a 3D build (expected D3Q15 | D3Q19 | D3Q27).");
#endif
    return nullptr; // unreachable: amrex::Abort does not return
}

} // namespace muphfasa
