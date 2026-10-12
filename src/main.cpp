// main.cpp — Entry point for the muphfasa LBM solver.
// Initialises AMReX, constructs a Simulation from an input file, runs the
// main time-advance loop, and finalises AMReX before exit.

#include <AMReX.H>

#include "IO/InputParser.H"
#include "Simulation/SimulationSetup.H"

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        muphfasa::InputParser params;
        muphfasa::Simulation  sim(params);

        sim.initialize();

        const int max_step = params.maxStep();
        for (int step = 0; step <= max_step; ++step) {
            sim.advance(step);
            sim.writeOutput(step);
        }
    }
    amrex::Finalize();
    return 0;
}
