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
    // get Ncells in largest direction
    auto Ncells = params.ncells();
    int max_N = Ncells[0];
    for (auto n : Ncells) { max_N = n ? n > max_N : max_N; }

    // Set phys/lattice units
    m_units = std::make_unique<UnitConverter>(params.refLength(),
                                              params.refVelocity(),
                                              params.refViscosity(),
                                              params.refDensity(),
                                              max_N);

    // Setup Geometry of Problem
    m_geom_handler = std::make_unique<GeometryHandler>(params);
    if (params.useEB()) {
        m_geom_handler->initEB(params.geometryType());
    }
    m_geom_handler->buildBoxArray(params.maxGridSize());

    m_geom = m_geom_handler->geometry();
    m_ba   = m_geom_handler->boxArray();
    m_dm   = m_geom_handler->distMap();

    // init multifabs
    m_f.define  (m_ba, m_dm, LatticeTag::Q,  LatticeTag::nGhost);
    m_feq.define(m_ba, m_dm, LatticeTag::Q,  0);
    m_rho.define(m_ba, m_dm, 1,              0);
    m_vel.define(m_ba, m_dm, AMREX_SPACEDIM, 0);

    // setup boundary conditions
    buildBCManager();
}

template <typename LatticeTag, typename CollisionModel>
SimulationImpl<LatticeTag, CollisionModel>::~SimulationImpl() = default;

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::initialize()
{
    // Restart support needs a restart-file key on InputParser, which doesn't
    // exist yet; always start fresh for now.
    setInitialConditions();
    fillGhostCells();
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::advance(int step)
{
    // TODO:
    //   1. computeEquilibriumMF<LatticeTag>            (rho, vel -> feq)
    //   2. collideMF<LatticeTag, CollisionModel>        (f, feq, tau)
    //   3. fillGhostCells                   (MPI halo exchange)
    //   4. streamInPlace<LatticeTag>        (f)
    //   5. computeMacroscopic<LatticeTag>   (f -> rho, vel)
    //   6. m_time += 1.0                    (lattice time unit)
    amrex::ignore_unused(step);

    // compute collision

    // fill ghost cells

    // do streaming

    // optional plotting
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
    // Quiescent start: rho = 1 (lattice density unit), u = 0 everywhere.
    // InputParser has no per-case IC parameters yet (e.g. lid velocity).
    m_rho.setVal(amrex::Real(1.0));
    m_vel.setVal(amrex::Real(0.0));
    computeEquilibriumMF<LatticeTag>(m_rho, m_vel, m_f);
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::fillGhostCells()
{
    // fill internal bounds/periodic
    m_f.FillBoundary(m_geom.periodicity());

    // fill physical bounds via BCManager<LatticeTag> (see buildBCManager())
    (*m_phys_bc)(m_f, 0, m_f.nComp(), m_f.nGrowVect(), m_time, 0);
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::buildBCManager()
{
    const auto names = m_params.bcNames();
    const auto rhoWall = m_params.refDensity();
    const auto Uwall = m_params.Uwall();

    BCManager<LatticeTag> bc_manager{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int bound : {-1, 1}) {
            const std::string& name = names[faceIndex(dir, bound)];
            if (name == "Periodic") {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    m_geom.isPeriodic(dir),
                    "buildBCManager: bc input requests \"Periodic\" on a "
                    "face whose Geometry is not periodic in that direction");
                continue; // no lbmBCtype to set; FillBoundary handles it
            }
            bc_manager.setFaceBC(dir, bound, parselbmBCtype(name));
            if (name == "MovingWall") {
                bc_manager.setFaceDensity(dir, bound, rhoWall);
                if (bound == -1)
                    bc_manager.setFaceUwall(dir, bound, Uwall[2*dir]);
                else if (bound == 1)
                    bc_manager.setFaceUwall(dir, bound, Uwall[2*dir+1]);
            }
        }
    }

    amrex::Vector<amrex::BCRec> bcs = makeBCRecVector(names, LatticeTag::Q);

    amrex::GpuBndryFuncFab<BCManager<LatticeTag>> gpu_bndry_func(bc_manager);
    m_phys_bc = std::make_unique<PhysBCFunctType>(m_geom, bcs, gpu_bndry_func);
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
