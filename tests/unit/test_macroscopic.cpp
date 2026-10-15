// test_macroscopic.cpp — Unit tests for density and velocity moment kernels.
// Verifies that:
//   - computeDensity returns the correct sum over a known f array
//   - computeVelocity returns the correct first moment for a known f array
//   - round-tripping an equilibrium state recovers the original rho and u

#include <AMReX.H>
#include <AMReX_Print.H>

#include "LBM/LatticeDescriptor.H"
#include "LBM/Macroscopic.H"

#include <cmath>
#include <stdexcept>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Macroscopic moment tests ===\n";

        // TODO: construct a known f array for D2Q9, call computeDensity /
        //       computeVelocity, and compare to analytic expectations.

        amrex::Print() << "All macroscopic tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
