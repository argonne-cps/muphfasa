// SimulationSetup.cpp — Top-level simulation orchestration.
// Constructs and owns all AMReX data structures, and implements the
// collision -> streaming -> BC -> macroscopic update cycle.
//
// Templated on LatticeTag (see SimulationSetup.H); explicitly instantiated
// below for every lattice matching this build's AMREX_SPACEDIM, mirroring
// the explicit-instantiation pattern used throughout src/LBM/.

#include "SimulationSetup.H"

#include "InitData.H"
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

#include <AMReX_BLProfiler.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

namespace muphfasa {

template <typename LatticeTag, typename CollisionModel>
SimulationImpl<LatticeTag, CollisionModel>::SimulationImpl(const InputParser& params)
    : m_params(params)
{
    // get Ncells in largest direction
    auto Ncells = params.ncells();
    int max_N = Ncells[0];
    for (auto n : Ncells) { max_N = (n > max_N) ? n : max_N; }

    // Set phys/lattice units. lbm.u_lat / lbm.tau (both optional) let the
    // user pin the lattice spacing directly; UnitConverter checks they're
    // mutually consistent (and with the other physical scales) if both are
    // given, and falls back to its own default u_lat if neither is.
    m_units = std::make_unique<UnitConverter>(params.refLength(),
                                              params.refVelocity(),
                                              params.reynoldsNumber(),
                                              params.refDensity(),
                                              max_N,
                                              LatticeTag::cs2,
                                              params.latticeVelocity(),
                                              params.relaxationTime());

    // Set collision param: omega = 1/tau.
    m_omega = amrex::Real(1.0) / m_units->relaxationTime();

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
    m_fin.define  (m_ba, m_dm, LatticeTag::Q,  LatticeTag::nGhost);
    m_fout.define  (m_ba, m_dm, LatticeTag::Q,  0);
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
void SimulationImpl<LatticeTag, CollisionModel>::advanceFused(int step)
{
    BL_PROFILE("SimulationImpl::advanceFused()");

    amrex::ignore_unused(step);

    // fused stream then collide
    pullCollideFullMF<LatticeTag, CollisionModel>(m_fin, m_fout, m_omega);

    // swap pointers
    std::swap(m_fin, m_fout);

    // fill ghost and stream
    fillGhostCells();
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::advanceSep(int step)
{
    BL_PROFILE("SimulationImpl::advance()");

    amrex::ignore_unused(step);

    // collide
    amrex::Real t0 = amrex::second();
    computeEquilibriumMF<LatticeTag>(m_rho, m_vel, m_feq);
    collideMF<LatticeTag, CollisionModel>(m_fin, m_feq, m_omega);
    amrex::Real t1 = amrex::second();
    m_t_collide += t1 - t0;

    // fill ghost and stream
    fillGhostCells();
    amrex::Real t2 = amrex::second();
    m_t_ghost += t2 - t1;

    streamInPlace<LatticeTag>(m_fin, m_geom);
    amrex::Real t3 = amrex::second();
    m_t_stream += t3 - t2;

    // compute macro
    computeMacroscopic<LatticeTag>(m_fin, m_rho, m_vel);
    amrex::Real t4 = amrex::second();
    m_t_macro += t4 - t3;

    ++m_steps_since_report;

    m_time += amrex::Real(1.0);
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::writeOutput(int step)
{
    BL_PROFILE("SimulationImpl::writeOutput()");

    if (step % m_params.plotInterval() == 0) {

        // compute macroscopic NOTE: need to stream fstar_in first.
        // This is kind of wasted computation
        stream<LatticeTag>(m_fin, m_fout, m_geom);
        computeMacroscopic<LatticeTag>(m_fout, m_rho, m_vel);

        const std::string fname = plotfileName(m_params.plotfilePrefix(), step);
        // Plotfiles are written in physical (SI) units; solver state stays lattice.
        const amrex::Real t_phys = m_units->toPhysicalTime(m_time);

        // copy rho/vel
        amrex::MultiFab rho_phys(m_rho.boxArray(), m_rho.DistributionMap(), m_rho.nComp(), 0);
        amrex::MultiFab vel_phys(m_vel.boxArray(), m_vel.DistributionMap(), m_vel.nComp(), 0);
        amrex::MultiFab::Copy(rho_phys, m_rho, 0, 0, m_rho.nComp(), 0);
        amrex::MultiFab::Copy(vel_phys, m_vel, 0, 0, m_vel.nComp(), 0);

        // convert units
        m_units->toPhysical(rho_phys, vel_phys);

        // writeout
        writeMacroPlotfile(fname, rho_phys, vel_phys, m_geom, t_phys, step);
        reportStageTiming(step);
    }

    if (step % m_params.checkpointInterval() == 0) {
        const std::string cname = checkpointName(m_params.checkpointPrefix(), step);
        writeCheckpoint(cname, m_fin, m_geom, step, m_time);
    }
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::reportStageTiming(int step) const
{
    // Report average per-step stage timing since the last report, then
    // reset so the next report covers only the steps since this one.
    if (m_steps_since_report == 0) { return; }

    amrex::Real t_collide = m_t_collide;
    amrex::Real t_ghost   = m_t_ghost;
    amrex::Real t_stream  = m_t_stream;
    amrex::Real t_macro   = m_t_macro;
    amrex::ParallelDescriptor::ReduceRealMax({t_collide, t_ghost, t_stream, t_macro});

    const amrex::Real n = amrex::Real(m_steps_since_report);
    amrex::Print() << "[step " << step << "] avg stage time over "
                    << m_steps_since_report << " step(s): "
                    << "collide=" << (t_collide / n) * 1e3 << " ms, "
                    << "ghost=" << (t_ghost / n) * 1e3 << " ms, "
                    << "stream=" << (t_stream / n) * 1e3 << " ms, "
                    << "macro=" << (t_macro / n) * 1e3 << " ms, "
                    << "total=" << ((t_collide + t_ghost + t_stream + t_macro) / n) * 1e3
                    << " ms\n";

    m_t_collide = amrex::Real(0.0);
    m_t_ghost   = amrex::Real(0.0);
    m_t_stream  = amrex::Real(0.0);
    m_t_macro   = amrex::Real(0.0);
    m_steps_since_report = 0;
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::setInitialConditions()
{
    initData(m_rho, m_vel, *m_units);
    computeEquilibriumMF<LatticeTag>(m_rho, m_vel, m_fin);
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::fillGhostCells()
{
    BL_PROFILE("SimulationImpl::fillGhostCells()");

    // fill internal bounds/periodic
    m_fin.FillBoundary(m_geom.periodicity());

    // fill physical bounds via BCManager<LatticeTag> (see buildBCManager())
    (*m_phys_bc)(m_fin, 0, m_fin.nComp(), m_fin.nGrowVect(), m_time, 0);

    // sync for profiling
    amrex::Gpu::streamSynchronize();
}

template <typename LatticeTag, typename CollisionModel>
void SimulationImpl<LatticeTag, CollisionModel>::buildBCManager()
{
    const auto names = m_params.bcNames();

    // wall/inflow params in lattice units
    const auto rhoWall = m_units->toLatticeDensity(m_params.refDensity());
    auto Uwall = m_params.Uwall();
    auto Uin = m_params.Uinflow();
    for (int wall_idx=0; wall_idx < 2 * AMREX_SPACEDIM; ++wall_idx) {
        for (int dim=0; dim < AMREX_SPACEDIM; ++dim) {
            Uwall[wall_idx][dim] = m_units->toLatticeVelocity(Uwall[wall_idx][dim]);
            Uin[wall_idx][dim]   = m_units->toLatticeVelocity(Uin[wall_idx][dim]);
        }
    }
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
                    bc_manager.setFaceVel(dir, bound, Uwall[2*dir]);
                else if (bound == 1)
                    bc_manager.setFaceVel(dir, bound, Uwall[2*dir+1]);
            } else if (name == "Inflow") {
                // reuses the reference density as rho_in; see InputParser::Uinflow()
                bc_manager.setFaceDensity(dir, bound, rhoWall);
                if (bound == -1)
                    bc_manager.setFaceVel(dir, bound, Uin[2*dir]);
                else if (bound == 1)
                    bc_manager.setFaceVel(dir, bound, Uin[2*dir+1]);
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
