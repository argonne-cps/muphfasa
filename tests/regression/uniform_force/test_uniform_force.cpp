// test_uniform_force.cpp — Regression test: uniform body force, fully periodic.
// Supports both 2D and 3D builds.
//
// With a quiescent start (reported u = 0; f is built from feq(rho, -F/(2 rho))
// by setInitialConditions) in a spatially uniform, fully periodic domain, the Guo scheme adds exactly F to the momentum
// Sum(c f) every step regardless of omega (relaxation contributes omega*F/2,
// the force term (1 - omega/2)*F). After k advanceFused() calls the reported
// (half-force corrected) velocity is therefore exactly
//     u_k = k * F / rho,     rho_k = rho_0,
// and the field stays spatially uniform. Checked every step to round-off.
// Exits non-zero on failure.

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include "IO/InputParser.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <algorithm>
#include <cmath>
#include <memory>

using namespace muphfasa;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    {
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        const auto accel = params.forcingAccel();
        amrex::Real F[AMREX_SPACEDIM];
        bool any_force = false;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            F[d] = sim->units().toLatticeAcceleration(accel[d]);
            any_force = any_force || (accel[d] != amrex::Real(0.0));
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(any_force,
            "test_uniform_force: forcing.accel must be nonzero");

        const amrex::Real rho0 = sim->rho().max(0);
        const amrex::Real tol = amrex::Real(1e-12);
        amrex::Print() << "tau=" << sim->units().relaxationTime() << " rho0=" << rho0
                       << " F_lat=(";
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { amrex::Print() << F[d] << (d + 1 < AMREX_SPACEDIM ? ", " : ")\n"); }

        amrex::Real worst_u = 0, worst_spread = 0, worst_rho = 0;
        const int max_step = params.maxStep();
        for (int step = 0; step <= max_step; ++step) {
            sim->advanceFused(step);
            const int k = step + 1;  // advanceFused() calls so far

            const amrex::MultiFab& vel = sim->vel();
            const amrex::MultiFab& rho = sim->rho();
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (step % params.plotInterval() == 0 ) {
                    sim->writeOutput(step);
                    const amrex::Real expected = amrex::Real(k) * F[d] / rho0;
                    const amrex::Real vmax = vel.max(d);
                    const amrex::Real vmin = vel.min(d);
                    worst_u = std::max({worst_u, std::abs(vmax - expected), std::abs(vmin - expected)});
                    worst_spread = std::max(worst_spread, vmax - vmin);
                }
            }
            worst_rho = std::max({worst_rho, std::abs(rho.max(0) - rho0), std::abs(rho.min(0) - rho0)});
        }

        amrex::Print() << "max |u - kF/rho| = " << worst_u << "\n"
                       << "max u spread           = " << worst_spread << "\n"
                       << "max |rho - rho0|       = " << worst_rho << "\n";

        const bool ok = worst_u < tol && worst_spread < tol && worst_rho < tol;
        amrex::Print() << (ok ? "uniform_force PASSED\n" : "uniform_force FAILED\n");
        status = ok ? 0 : 1;
    }
    amrex::Finalize();
    return status;
}
