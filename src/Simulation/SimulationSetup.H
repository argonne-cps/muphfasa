// SimulationSetup.H — Top-level simulation object for the muphfasa LBM solver.
// Owns all persistent AMReX data (Geometry, BoxArray, MultiFabs) and
// orchestrates initialisation, time-advance, and output.  Physics details
// are delegated to the LBM sub-modules.

#ifndef MUPHFASA_SIMULATION_SETUP_H
#define MUPHFASA_SIMULATION_SETUP_H

#include "ISimulation.H"
#include "LBM/BoundaryConditions.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_REAL.H>

#include <memory>

namespace muphfasa {

class InputParser;
class GeometryHandler;
class UnitConverter;

// LatticeTag fixes the velocity set (D2Q9, D3Q15, D3Q19, or D3Q27), and
// CollisionModel fixes the collision operator (BGK, TRT, MRT), both at
// compile time -- this is what lets the LBM kernels use compile-time-known
// component counts and constexpr lattice data on the GPU. The *runtime*
// choice of lattice and collision model (read from the input file) happens
// once, in makeSimulation() (SimulationFactory.H), which picks which
// SimulationImpl<LatticeTag, CollisionModel> to instantiate.
template <typename LatticeTag, typename CollisionModel>
class SimulationImpl : public ISimulation
{
public:
    explicit SimulationImpl(const InputParser& params);
    ~SimulationImpl() override;

    // ── Lifecycle ─────────────────────────────────────────────────────────
    // Allocate MultiFabs, set initial conditions, and fill ghost cells.
    void initialize() override;

    // Advance by one time step: collision -> streaming -> boundary conditions
    // -> macroscopic update.
    void advance(int step) override;

    // Write plotfile and/or checkpoint when the output intervals are hit.
    void writeOutput(int step) const override;

    // ── State accessors ───────────────────────────────────────────────────
    const amrex::MultiFab& f()   const noexcept override { return m_f;   }
    const amrex::MultiFab& rho() const noexcept override { return m_rho; }
    const amrex::MultiFab& vel() const noexcept override { return m_vel; }

    amrex::Real time() const noexcept override { return m_time; }

private:
    // ── Input and unit conversion ─────────────────────────────────────────
    const InputParser&           m_params;
    std::unique_ptr<UnitConverter> m_units;

    // ── AMReX grid objects ────────────────────────────────────────────────
    std::unique_ptr<GeometryHandler> m_geom_handler;
    amrex::Geometry                  m_geom;
    amrex::BoxArray                  m_ba;
    amrex::DistributionMapping       m_dm;

    // ── Distribution function and macroscopic fields ──────────────────────
    amrex::MultiFab m_f;    // distribution: LatticeTag::Q components
    amrex::MultiFab m_feq;  // equilibrium:  LatticeTag::Q components (scratch)
    amrex::MultiFab m_rho;  // density:      1 component
    amrex::MultiFab m_vel;  // velocity:     AMREX_SPACEDIM components

    // ── Boundary conditions ───────────────────────────────────────────────
    // BCManager set once at setup, used to set m_phys_bc
    using PhysBCFunctType = amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BCManager<LatticeTag>>>;
    std::unique_ptr<PhysBCFunctType> m_phys_bc;

    // ── Time tracking ─────────────────────────────────────────────────────
    amrex::Real m_time = amrex::Real(0.0);

    // ── Private helpers ───────────────────────────────────────────────────
    void setInitialConditions();
    void fillGhostCells();
    void buildBCManager();
};

} // namespace muphfasa

#endif // MUPHFASA_SIMULATION_SETUP_H
