// SimulationSetup.cpp — Top-level simulation orchestration.
// Constructs and owns all AMReX data structures, and implements the
// collision -> streaming -> BC -> macroscopic update cycle.

#include "SimulationSetup.H"

#include "IO/InputParser.H"
#include "IO/Checkpoint.H"
#include "IO/Plotfile.H"
#include "Geometry/GeometryHandler.H"
#include "LBM/BoundaryConditions.H"
#include "LBM/Collision.H"
#include "LBM/Equilibrium.H"
#include "LBM/Macroscopic.H"
#include "LBM/Streaming.H"
#include "LBM/Units.H"

#include <AMReX_Print.H>

namespace muphfasa {

Simulation::Simulation(const InputParser& params)
    : m_params(params)
{
    // TODO: construct UnitConverter from params, then GeometryHandler,
    //       then allocate MultiFabs, then buildBCManager.
}

Simulation::~Simulation() = default;

void Simulation::initialize()
{
    // TODO: setInitialConditions(), fillGhostCells()
    //       Optionally restart from a checkpoint if one exists.
    amrex::Print() << "Simulation::initialize() -- stub\n";
}

void Simulation::advance(int step)
{
    // TODO:
    //   1. computeEquilibriumMF (rho, vel -> feq)
    //   2. collide              (f, feq, tau)
    //   3. fillGhostCells       (MPI halo exchange)
    //   4. streamInPlace        (f)
    //   5. m_bc_manager->applyAll(f, geom, step)
    //   6. computeMacroscopic   (f -> rho, vel)
    //   7. m_time += 1.0        (lattice time unit)
    amrex::ignore_unused(step);
}

void Simulation::writeOutput(int step) const
{
    if (step % m_params.plotInterval() == 0) {
        const std::string fname = plotfileName(m_params.plotfilePrefix(), step);
        writeMacroPlotfile(fname, m_rho, m_vel, m_geom, m_time, step);
    }

    if (step % m_params.checkpointInterval() == 0) {
        const std::string cname = checkpointName(m_params.checkpointPrefix(), step);
        writeCheckpoint(cname, m_f, m_geom, step, m_time);
    }
}

void Simulation::setInitialConditions()
{
    // TODO: fill m_f with the equilibrium for the prescribed initial rho and u
}

void Simulation::fillGhostCells()
{
    // TODO: m_f.FillBoundary(m_geom.periodicity())
    //       and FillPatch for non-periodic / EB boundaries
}

void Simulation::buildBCManager()
{
    // TODO: read BC types from m_params and register them with m_bc_manager
}

} // namespace muphfasa
