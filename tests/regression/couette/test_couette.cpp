// test_couette.cpp — Regression test: plane Couette flow startup + steady state.
// Supports both 2D (AMREX_SPACEDIM=2) and 3D (AMREX_SPACEDIM=3) builds.
//
// Geometry: periodic in x (and y in 3D), NoSlip wall at y=lo (2D) or z=lo (3D),
// MovingWall at y=hi (2D) or z=hi (3D) moving at U0. With H = wall-normal
// cells (Ny in 2D, Nz in 3D, wall-to-wall distance in lattice units per the
// halfway-bounce-back convention: wall sits half a cell beyond last fluid node
// on each side, so H == Nwall exactly), the velocity field obeys the 1-D
// unsteady diffusion equation
//   du/dt = nu * d^2u/dy^2,   u(0,t) = 0,  u(H,t) = U0 (t>0),  u(y,0) = 0
// whose solution (classic plane-Couette-startup / "doubly-bounded Rayleigh"
// result) is
//   u(y,t)/U0 = y/H + sum_{n=1}^inf [2*(-1)^n/(n*pi)] * sin(n*pi*y/H)
//                     * exp(-n^2*pi^2*nu*t/H^2)
// which reduces to the steady linear profile u(y) = U0*y/H as t -> inf.
// For 3D the same holds with y→z and H=Nz; faverage (DIM=3) averages over
// x-y and returns the z-profile, matching the wall normal.
//
// This test produces two sets of plotfiles from the sim (pltxxxxxxx/) and
// from the analytic soln (analytic_pltxxxxxxx/)
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

// Wall-normal direction: y in 2D (dir=1), z in 3D (dir=2) so faverage
// (which averages along the last dimension) returns the wall-normal profile.
constexpr int WALL_DIM = (AMREX_SPACEDIM == 2 ? 1 : 2);


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
// For 3D the wall normal is z (WALL_DIM=2) so we vary with k and dz.
void calcAnalytic(amrex::MultiFab& vel, amrex::Real H, amrex::Real U0,
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
#if (AMREX_SPACEDIM == 3)
                amrex::Real wall_coord = amrex::Real(k) + 0.5;
#else
                amrex::Real wall_coord = amrex::Real(j) + 0.5;
#endif
                velx_arr(i, j, k) = analyticCouette(wall_coord, H, U0, nu, t);
                vely_arr(i, j, k) = 0.;
#if (AMREX_SPACEDIM == 3)
                velz_arr(i, j, k) = 0.;
#endif
            });
    }
}

// calculate the error between current sim state and the analytic soln
void calcError(const std::unique_ptr<ISimulation>& sim,
               std::vector<amrex::Real>& max_err,
               const std::string& analytic_file, 
               amrex::Real H, amrex::Real U0, amrex::Real nu,
               int step)
{
    // calc_analytic vel in lattice then convert to physical
    const amrex::MultiFab& sim_vel = sim->vel();
    const amrex::MultiFab& sim_rho = sim->rho();

    amrex::MultiFab analytic_vel(sim_vel.boxArray(), sim_vel.DistributionMap(),
                                 sim_vel.nComp(), sim_vel.nGrow());
    amrex::MultiFab analytic_rho(sim_rho.boxArray(), sim_rho.DistributionMap(),
                                 sim_rho.nComp(), 0);
    amrex::MultiFab::Copy(analytic_rho, sim_rho, 0, 0, sim_rho.nComp(), 0);

    calcAnalytic(analytic_vel, H, U0, nu, sim->time());
    sim->units().toPhysical(analytic_rho, analytic_vel);
    const amrex::Real t_phys = sim->units().toPhysicalTime(sim->time());

    // write out analytic file
    muphfasa::writeMacroPlotfile(plotfileName(analytic_file, step),
                                 analytic_rho, analytic_vel, sim->geom(), t_phys, step);

    // calc error in physical units
    amrex::MultiFab sim_vel_phys(sim_vel.boxArray(), sim_vel.DistributionMap(),
                                 sim_vel.nComp(), 0);
    amrex::MultiFab::Copy(sim_vel_phys, sim_vel, 0, 0, sim_vel.nComp(), 0);
    sim->units().toPhysicalVelocity(sim_vel_phys);
    amrex::MultiFab diff(sim_vel.boxArray(), sim_vel.DistributionMap(),
                         sim_vel.nComp(), 0);
    amrex::MultiFab::Copy(diff, analytic_vel, 0, 0, sim_vel.nComp(), 0);
    diff.minus(sim_vel_phys, 0, sim_vel.nComp(), 0);
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

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // Input file comes from the command line (e.g. `./test_couette inputs`)
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        // Get fundamental params (wall-normal is y in 2D, z in 3D)
        const int Nwall = params.ncells()[WALL_DIM];
        const amrex::Real H = amrex::Real(Nwall);

        const amrex::Real tau = sim->units().relaxationTime();
        const amrex::Real nu  = (tau - 0.5) / 3.0;

        // Wall-hi face, x-component (tangential velocity of the moving wall).
        const amrex::Real U0 = sim->units().toLatticeVelocity(params.Uwall()[faceIndex(WALL_DIM, 1)][0]);

        const int max_step   = params.maxStep();
        std::vector<amrex::Real> max_err;

        // file prefix to plot out analytic soln
        const std::string analytic_file = "analytic_plt";

        // run simulation
        for (int step = 0; step <= max_step; ++step) {
            sim->advanceFused(step);
            sim->writeOutput(step);

            // write out analytic soln (both plt and analytic_plt are physical)
            if (step % params.plotInterval() == 0 && params.plotInterval() > 0) {
                // calc error
                calcError(sim, max_err, analytic_file, H, U0, nu, step);
            }
        }

        // calc final error if no writes
        if (params.plotInterval() < 0 ){
                sim->updateMacroscopic();
                calcError(sim, max_err, analytic_file, H, U0, nu, max_step);
        }

        // compute final error
        amrex::Real final_error = max_err.back();

        // printout results
        amrex::Print() << "Final Steady state error: " << final_error << '\n';
    }
    amrex::Finalize();
    return 0;
}
