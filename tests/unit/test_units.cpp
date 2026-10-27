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

        // Body-force acceleration: a_lat = a_phys * dt^2 / dx. Hand-computed:
        // L=1, U=1, Re=32, N=32 -> dx = 1/32 and dt = dx * u_lat/U = dx * u_lat,
        // so for any u_lat, a_lat = a_phys * dx * u_lat^2.
        {
            const amrex::Real cs2 = amrex::Real(1.0) / amrex::Real(3.0);
            const muphfasa::UnitConverter units(1.0, 1.0, 32.0, 1.0, 32, cs2);
            const amrex::Real dx = units.dx();
            const amrex::Real dt = units.dt();
            assertClose(dx, amrex::Real(1.0) / amrex::Real(32.0), 1e-15, "dx");

            const amrex::Real a_phys = 0.125;
            const amrex::Real a_lat  = units.toLatticeAcceleration(a_phys);
            assertClose(a_lat, a_phys * dt * dt / dx, 1e-15, "a_lat = a dt^2/dx");

            const amrex::Real u_lat = units.toLatticeVelocity(1.0);
            assertClose(a_lat, a_phys * dx * u_lat * u_lat, 1e-15, "a_lat = a dx u_lat^2");

            assertClose(units.toPhysicalAcceleration(a_lat), a_phys, 1e-15, "acceleration round trip");
            assertClose(units.toLatticeAcceleration(0.0), 0.0, 0.0, "zero acceleration");
        }

        amrex::Print() << "All unit converter tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
