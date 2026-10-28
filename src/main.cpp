// main.cpp — Entry point for the muphfasa LBM solver.
// Initialises AMReX, constructs a Simulation from an input file, runs the
// main time-advance loop, and finalises AMReX before exit.

#include <AMReX.H>
#include <AMReX_BLProfiler.H>

#include "IO/InputParser.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <memory>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main()");

        muphfasa::InputParser params;
        std::unique_ptr<muphfasa::ISimulation> sim = muphfasa::makeSimulation(params);

        sim->initialize();

        const int max_step = params.maxStep();
        for (int step = 0; step <= max_step; ++step) {
            sim->advanceFused(step);
            sim->writeOutput(step);
        }
    }
    amrex::Finalize();
    return 0;
}
