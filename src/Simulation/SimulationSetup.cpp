// SimulationSetup.cpp — Top-level simulation orchestration.
// Constructs and owns all AMReX data structures, and implements the
// collision -> streaming -> BC -> macroscopic update cycle.
//
// Templated on LatticeTag (see SimulationSetup.H); explicitly instantiated
// below for every lattice matching this build's AMREX_SPACEDIM, mirroring
// the explicit-instantiation pattern used throughout src/LBM/.

#include "SimulationSetup.H"

#include "IO/InputParser.H"
#include "IO/Checkpoint.H"
#include "IO/Plotfile.H"
#include "Geometry/GeometryHandler.H"
#include "LBM/BoundaryConditions.H"
#include "LBM/Collision.H"
#include "LBM/Equilibrium.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Macroscopic.H"
#include "LBM/Streaming.H"
#include "LBM/Units.H"

#include <AMReX_Print.H>

namespace muphfasa {

template <typename LatticeTag, typename CollisionModel>
SimulationImpl<LatticeTag, CollisionModel>::SimulationImpl(const InputParser& params)
    : m_params(params)
{
    // TODO: construct UnitConverter from params, then GeometryHandler,
    //       then allocate MultiFabs (m_f/m_feq with LatticeTag::Q components),
    //       then buildBCManager.
}

template <typename LatticeTag, typename CollisionModel>
SimulationImpl<LatticeTag, CollisionModel>::~SimulationImpl() = default;

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::initialize()
{
    // TODO: setInitialConditions(), fillGhostCells()
    //       Optionally restart from a checkpoint if one exists.
    amrex::Print() << "Simulation::initialize() -- stub\n";
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::advance(int step)
{
    // TODO:
    //   1. computeEquilibriumMF<LatticeTag>            (rho, vel -> feq)
    //   2. collideMF<LatticeTag, CollisionModel>        (f, feq, tau)
    //   3. fillGhostCells                   (MPI halo exchange)
    //   4. streamInPlace<LatticeTag>        (f)
    //   5. m_bc_manager->applyAll(f, geom, step)
    //   6. computeMacroscopic<LatticeTag>   (f -> rho, vel)
    //   7. m_time += 1.0                    (lattice time unit)
    amrex::ignore_unused(step);
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::writeOutput(int step) const
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

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::setInitialConditions()
{
    // TODO: fill m_f with the equilibrium for the prescribed initial rho and u
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::fillGhostCells()
{
    // TODO: m_f.FillBoundary(m_geom.periodicity())
    //       and FillPatch for non-periodic / EB boundaries
    m_f.FillBoundary(m_geom.periodicity())

    m_phys_bc.FillDomainBoundary(m_f, 0, m_f.nComp(), m_geom);

}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::buildBCManager()
{
    // TODO: read BC types from m_params and register them with m_bc_manager
}

// ── Explicit instantiations ─────────────────────────────────────────────────
// Dimension-matched only: a 2D build only ever has D2Q9; a 3D build offers a
// runtime choice among D3Q15/D3Q19/D3Q27 (see SimulationFactory.cpp).
//
// Only BGK is instantiated for now, since TRT/MRT (Collision.H) don't yet
// implement collide()/collideArr(). Adding a new collision model means
// implementing it in Collision.cpp, then adding its instantiation lines here
// and wiring it into SimulationFactory.cpp's collision-model dispatch.
#if (AMREX_SPACEDIM == 2)
template class SimulationImpl<D2Q9, BGK>;
#else
template class SimulationImpl<D3Q15, BGK>;
template class SimulationImpl<D3Q19, BGK>;
template class SimulationImpl<D3Q27, BGK>;
#endif

} // namespace muphfasa
