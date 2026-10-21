// test_couette.cpp — Regression test: plane Couette flow startup + steady state.
// Supports both 2D (AMREX_SPACEDIM=2) and 3D (AMREX_SPACEDIM=3) builds.
//
// Geometry: periodic in x (and z for 3D), NoSlip wall at y=lo, MovingWall at y=hi
// moving at U0. With H = grid.ncells[1] (wall-to-wall distance in lattice units,
// per the halfway-bounce-back convention: the wall sits half a cell beyond the
// last fluid node on each side, so H == Ny exactly), the velocity field obeys
// the 1-D unsteady diffusion equation
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
        amrex::Real exp_arg = -amrex::Real(n * n) * kPi * kPi * nu * t / (H * H);
        if (exp_arg < -50) { // exp(-50) ~ 2e-22, remaining terms negligible
            break;
        }
        amrex::Real term = (amrex::Real(2.0) * U0 * sign / npi) * std::sin(npi * y / H) *
             std::exp(exp_arg);
        u += term;
    }
    return u;
}

// calc analytic soln on whole grid
// Note: AMReX's ParallelFor accepts (i,j,k) in both 2D and 3D; k is ignored in 2D.
void calcAnalytic(amrex::MultiFab& vel, amrex::Real dy, amrex::Real H, amrex::Real U0,
                  amrex::Real nu, amrex::Real t)
{
    for (amrex::MFIter mfi(vel, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto box = mfi.tilebox();
        const auto velx_arr = vel.array(mfi, 0);
        const auto vely_arr = vel.array(mfi, 1);
#if (AMREX_SPACEDIM == 3)
        const auto velz_arr = vel.array(mfi, 2);
#endif

        amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real y = dy * (amrex::Real(j) + 0.5);
                velx_arr(i, j, k) = analyticCouette(y, H, U0, nu, t);
                vely_arr(i, j, k) = 0.;
#if (AMREX_SPACEDIM == 3)
                velz_arr(i, j, k) = 0.;
#endif
            });
    }
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // Input file comes from the command line (e.g. `./test_couette inputs`)
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
        const std::string analytic_file = "analytic_plt";

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

                // write out file
                muphfasa::writeMacroPlotfile(plotfileName(analytic_file, step),
                                             sim->rho(), analytic_vel, sim->geom(), sim->time(), step);

                // calc avg and max error (only on valid cells, not ghost cells)
                amrex::MultiFab diff(sim_vel.boxArray(), sim_vel.DistributionMap(),
                                     sim_vel.nComp(), 0);
                amrex::MultiFab::Copy(diff, analytic_vel, 0, 0, sim_vel.nComp(), 0);
                diff.minus(sim_vel, 0, sim_vel.nComp(), 0);
                diff.abs(0, sim_vel.nComp(), 0);

                amrex::Real diff_max = diff.max(0);
                if (std::isnan(diff_max)) {
                    amrex::Print() << "WARNING: NaN in diff at step " << step << "\n";
                    max_err.push_back(-1.0);
                } else {
                    max_err.push_back(diff_max);
                    amrex::Print() << "At Time=" << step << ": Error=" << diff_max << "\n";
                }
            }
        }

        // compute final error
        amrex::Real final_error = max_err.back();

        // printout results
        amrex::Print() << "Final Steady state error: " << final_error << '\n';

        amrex::Finalize();
        return 0;
    }
}