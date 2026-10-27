// test_mass_conservation.cpp — Closed-box mass conservation via ISimulation.
// All walls NoSlip (bounce-back), quiescent init (rho=1, u=0), N advance() steps.
// Total mass (sum rho over valid cells) must be conserved to ~1e-12.
//
// This mirrors the regression drivers in tests/regression/tgv2d/test_tgv2d.cpp
// and tests/regression/couette/test_couette.cpp:
//   InputParser params; auto sim = makeSimulation(params); sim->initialize();
//   for (step) { sim->advance(step); sim->writeOutput(step); }
// Here we exercise the same ISimulation::advance() path (collide -> FillBoundary
// -> BC -> stream -> macroscopic) instead of hand-rolling collide/stream in
// the test. Both single-box and multi-box (max_grid_size < domain) layouts are
// exercised: multi-box guards against wall BCs being skipped for boxes that
// touch the domain boundary on one side and a neighbour box on the other.

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "IO/InputParser.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>

using namespace muphfasa;

// Sum of density over valid cells (global, via ReduceRealSum).
static amrex::Real totalMassRho(const amrex::MultiFab& rho)
{
    amrex::Real sum = 0.0;
    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        auto const arr = rho.const_array(mfi);
        const amrex::Box& vbx = mfi.validbox();
        const auto lo = vbx.smallEnd();
        const auto hi = vbx.bigEnd();
#if AMREX_SPACEDIM == 2
        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                sum += arr(i, j, 0, 0);
            }
        }
#else
        for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    sum += arr(i, j, k, 0);
                }
            }
        }
#endif
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}

// Populate ParmParse with a closed-box (all NoSlip) configuration.
// Each call appends a new definition; ParmParse query picks the last, so this
// shadows any previous mass-conservation setup and lets us run several
// back-to-back simulations in one executable (same pattern as overriding inputs
// on the command line).
static void setupClosedBoxInputs(const std::vector<int>& ncells,
                                 int max_grid_size,
                                 const std::string& lattice,
                                 amrex::Real tau = 1.0)
{
    // Grid
    {
        amrex::ParmParse pp("grid");
        pp.addarr("ncells", ncells);
        pp.add("max_grid_size", max_grid_size);
        pp.add("n_ghost", 1);
    }
    // Time stepping — InputParser::maxStep() reads this, but the test loop
    // uses its own Nsteps; keep them consistent.
    {
        amrex::ParmParse pp("sim");
        pp.add("max_step", 50);
        pp.add("stop_time", -1.0);
    }
    // LBM: pin tau directly so omega = 1/tau = 1.0 (mass-conserving BGK).
    {
        amrex::ParmParse pp("lbm");
        pp.add("lattice_type", lattice);
        pp.add("tau", tau);
        // collision_model defaults to BGK via InputParser
    }
    // Units: L_ref, U_ref, Re, rho_ref. With tau pinned, Re only sets the
    // derived u_lat via UnitConverter but must be non-zero.
    {
        amrex::ParmParse pp("units");
        pp.add("ref_length", 1.0);
        pp.add("ref_velocity", 1.0);
        pp.add("Re", 1.0);
        pp.add("ref_density", 1.0);
    }
    // BC: all faces NoSlip (closed box).
    {
        amrex::ParmParse pp("bc");
        std::vector<std::string> lo(AMREX_SPACEDIM, "NoSlip");
        std::vector<std::string> hi(AMREX_SPACEDIM, "NoSlip");
        pp.addarr("lo_type", lo);
        pp.addarr("hi_type", hi);
    }
    // Geometry: unit cube / square.
    {
        amrex::ParmParse pp("geometry");
        std::vector<amrex::Real> prob_lo(AMREX_SPACEDIM, 0.0);
        std::vector<amrex::Real> prob_hi(AMREX_SPACEDIM, 1.0);
        pp.addarr("prob_lo", prob_lo);
        pp.addarr("prob_hi", prob_hi);
        pp.add("type", std::string("box"));
    }
    // Output: suppress plot/chk writes during the unit test (writeOutput
    // checks step % plot_int; with 1M the only hit would be step 0).
    {
        amrex::ParmParse pp("amr");
        pp.add("plot_int", 1000000);
        pp.add("chk_int", 1000000);
        pp.add("plot_file", std::string("plt"));
        pp.add("check_file", std::string("chk"));
    }
}

static void runMassConservationTest(const char* tag,
                                    const std::vector<int>& ncells,
                                    int max_grid_size,
                                    const std::string& lattice)
{
    setupClosedBoxInputs(ncells, max_grid_size, lattice, 1.0);

    // Mimic regression driver: InputParser -> makeSimulation -> initialize
    InputParser params;
    std::unique_ptr<ISimulation> sim = makeSimulation(params);
    sim->initialize();

    const int Nsteps = 50;
    const amrex::Real m0 = totalMassRho(sim->rho());

    // Advance exactly like src/main.cpp / regression tests
    for (int step = 0; step < Nsteps; ++step) {
        sim->advance(step);
    }

    const amrex::Real m1 = totalMassRho(sim->rho());
    const amrex::Real rel = std::abs(m1 - m0) / std::abs(m0);

    // Report (tag, domain, masses)
    amrex::Print() << "  " << tag << " ncells ";
    for (auto n : ncells) { amrex::Print() << n << " "; }
    amrex::Print() << "max_grid " << max_grid_size
                   << " lattice " << lattice
                   << " m0=" << m0 << " m1=" << m1
                   << " rel drift=" << rel << "\n";

    const amrex::Real tol = 1e-12;
    if (rel > tol) {
        throw std::runtime_error(std::string(tag) + ": mass drift " +
                                 std::to_string(rel) + " exceeds tol " +
                                 std::to_string(tol));
    }
    amrex::Print() << "    [PASS]\n";
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Closed-box mass conservation (DIM="
                       << AMREX_SPACEDIM << ") via ISimulation::advance() ===\n";

#if (AMREX_SPACEDIM == 2)
        // 2-D
        runMassConservationTest("D2Q9 single-box 8x8",
                                std::vector<int>{8, 8}, 64, "D2Q9");
        runMassConservationTest("D2Q9 rect 12x6",
                                std::vector<int>{12, 6}, 64, "D2Q9");
        runMassConservationTest("D2Q9 two-box 8x8",
                                std::vector<int>{8, 8}, 4, "D2Q9");
#else
        // 3-D: one test per lattice, single-box
        runMassConservationTest("D3Q15 single-box 6x6x6",
                                std::vector<int>{6, 6, 6}, 64, "D3Q15");
        runMassConservationTest("D3Q19 single-box 6x6x6",
                                std::vector<int>{6, 6, 6}, 64, "D3Q19");
        runMassConservationTest("D3Q27 single-box 6x6x6",
                                std::vector<int>{6, 6, 6}, 64, "D3Q27");
        runMassConservationTest("D3Q15 multi-box 8x8x8",
                                std::vector<int>{8, 8, 8}, 4, "D3Q15");
        runMassConservationTest("D3Q19 multi-box 8x8x8",
                                std::vector<int>{8, 8, 8}, 4, "D3Q19");
        runMassConservationTest("D3Q27 multi-box 8x8x8",
                                std::vector<int>{8, 8, 8}, 4, "D3Q27");
#endif
        amrex::Print() << "All mass conservation tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
