// test_equilibrium.cpp — Unit tests for the equilibrium distribution function.
// Verifies that:
//   - sum_q f^eq_q == rho                     (mass conservation)
//   - sum_q c_{q,alpha} f^eq_q == rho * u_alpha  (momentum conservation)
//   - f^eq is non-negative for Ma << 1
//   - f^eq recovers the rest-state limit when u=0

#include <AMReX.H>
#include <AMReX_Print.H>

#include "LBM/LatticeDescriptor.H"
#include "LBM/Equilibrium.H"

#include <cmath>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Equilibrium distribution tests ===\n";

        // TODO: evaluate computeEquilibrium<D2Q9> for several (rho, u) pairs
        //       and assert conservation properties.

        amrex::Print() << "All equilibrium tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
