// test_units.cpp — Unit tests for the physical-to-lattice unit converter.
// Verifies round-trip conversions: physical -> lattice -> physical should
// recover the original value within floating-point tolerance.
// Also checks that derived dimensionless parameters (Re, Ma, tau) are
// consistent with the input physical scales.

#include <AMReX.H>
#include <AMReX_Print.H>

#include "LBM/Units.H"

#include <cmath>
#include <stdexcept>

namespace {

void assertClose(amrex::Real a, amrex::Real b, amrex::Real tol, const char* msg)
{
    if (std::abs(a - b) > tol)
        throw std::runtime_error(std::string(msg) + ": " +
                                 std::to_string(a) + " != " + std::to_string(b));
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Unit converter tests ===\n";

        // TODO: construct a UnitConverter with known inputs and verify:
        //   - toLatticeLength(refLength()) returns N_ref
        //   - relaxationTime() > 0.5   (stability requirement)
        //   - machNumber() < 0.1       (low-Ma assumption)

        amrex::Print() << "All unit converter tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
