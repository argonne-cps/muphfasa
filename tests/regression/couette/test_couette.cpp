// test_couette.cpp — Regression test: plane Couette flow startup + steady state.
//
// Geometry: periodic in x, NoSlip wall at y=lo, MovingWall at y=hi moving at
// U0 (see ./inputs). With H = grid.ncells[1] (wall-to-wall distance in
// lattice units, per the halfway-bounce-back convention: the wall sits half
// a cell beyond the last fluid node on each side, so H == Ny exactly), the
// velocity field obeys the 1-D unsteady diffusion equation
//   du/dt = nu * d^2u/dy^2,   u(0,t) = 0,  u(H,t) = U0 (t>0),  u(y,0) = 0
// whose solution (classic plane-Couette-startup / "doubly-bounded Rayleigh"
// result) is
//   u(y,t)/U0 = y/H + sum_{n=1}^inf [2*(-1)^n/(n*pi)] * sin(n*pi*y/H)
//                     * exp(-n^2*pi^2*nu*t/H^2)
// which reduces to the steady linear profile u(y) = U0*y/H as t -> inf.
//
// This test produces two sets of plotfiles from the sim (pltxxxxx/) and 
// from the analytic soln (analytic_pltxxxxx/)
// These can then be compared via amrex/Tools/Plotfiles/fcompare
//
//

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include "IO/InputParser.H"
#include "IO/Plotfile.H"
#include "LBM/BoundaryConditions.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace muphfasa;

namespace {

constexpr amrex::Real kPi  = amrex::Real(3.14159265358979323846);
constexpr amrex::Real kCs2 = amrex::Real(1.0) / amrex::Real(3.0);


// Time-dependent plane-Couette analytic solution (see file header).
amrex::Real analyticCouette(amrex::Real y, amrex::Real H, amrex::Real U0,
                            amrex::Real nu, amrex::Real t, int n_terms = 200)
{
    amrex::Real u = U0 * y / H;
    for (int n = 1; n <= n_terms; ++n) {
        const amrex::Real npi  = n * kPi;
        const amrex::Real sign = (n % 2 == 0) ? amrex::Real(1.0) : amrex::Real(-1.0);
        u += (amrex::Real(2.0) * U0 * sign / npi) * std::sin(npi * y / H) *
             std::exp(-amrex::Real(n * n) * kPi * kPi * nu * t / (H * H));
    }
    return u;
}

// calc analytic soln on whole grid
void calcAnalytic(amrex::MultiFab& vel, amrex::Real dy, amrex::Real H, amrex::Real U0,
                            amrex::Real nu, amrex::Real t)
{

   for(amrex::MFIter mfi(vel, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) 
   {
    const auto box = mfi.tilebox();
    const auto velx_arr = vel.array(mfi, 0);
    const auto vely_arr = vel.array(mfi, 1);

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        amrex::Real y = dy * (amrex::Real(j) + 0.5);
        velx_arr(i, j, k) = analyticCouette(y, H, U0, nu, t);
        vely_arr(i, j, k) = 0.;
    });
    }
}

} //namspace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // Input file comes from the command line (e.g. `./test_couette
        // inputs`) 
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        // Get fundamental params
        const int Ny = params.ncells()[1];
        const amrex::Real H = amrex::Real(Ny);
        const amrex::Real dy = amrex::Real(1);

        const amrex::Real tau = sim->units().relaxationTime();
        const amrex::Real nu  = (tau - 0.5) / 3.0;

        // y-hi face, x-component (tangential velocity of the moving wall).
        const amrex::Real U0 = sim->units().toLatticeVelocity(params.Uwall()[faceIndex(1, 1)][0]);

        const int max_step   = params.maxStep();
        std::vector<amrex::Real> max_err;
        std::vector<amrex::Real> avg_err;

        // create copy of velocity mf for analytic soln
        const std::string analytic_file="analytic_plt";

        // run simulation
        for (int step = 0; step <= max_step; ++step) {
            sim->advance(step);
            sim->writeOutput(step);

            // write out analytic soln
            if (step % params.plotInterval() == 0) {
                // calc_analytic vel
                const amrex::MultiFab& sim_vel = sim->vel();
                amrex::MultiFab analytic_vel(sim_vel.boxArray(), sim_vel.DistributionMap(),
                               sim_vel.nComp(), sim_vel.nGrow());

                calcAnalytic(analytic_vel, dy, H, U0, nu, sim->time());

                // convert to physical units
                //amrex::Real lat_vel_fact = sim->units().dx() /sim->units().dt();
                //analytic_vel.mult(lat_vel_fact, 0, 2);

                // write out file
                muphfasa::writeMacroPlotfile(plotfileName(analytic_file, step),
                        sim->rho(), analytic_vel, sim->geom(), sim->time(), step);

                // calc avg and max error
                analytic_vel.minus(sim_vel, 0, 1, 0);
                analytic_vel.abs(0, 1, 0);
                amrex::Real err = analytic_vel.max(0);
                max_err.push_back(err);
                amrex::Print() << "At Time=" << step << ": Error=" << err << "\n";
            }
        }

    // compute total max and avg error
    amrex::Real final_error = max_err.back();

    //printout results
    amrex::Print() << "Final Steady state error: " <<  final_error << '\n';

    amrex::Finalize();
    return 0;
    }
}
