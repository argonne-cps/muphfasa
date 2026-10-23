// test_tgv2d.cpp — Regression test: 2-D Taylor-Green vortex decay.
//
// Uses the normal InputParser/ISimulation/advance() machinery (see
// tests/regression/couette/test_couette.cpp, which this mirrors) with the
// non-quiescent initial condition supplied by this directory's own
// InitData.cpp (see TgvAnalytic.H for the analytic solution it and this
// file both use).
//
// At every amr.plot_int step, writes a plt<step>/analytic_plt<step> pair
// (density + velocity, matching BoxArray/variable names) so they can be
// diffed with AMReX's fcompare, and prints the max velocity error against
// the analytic solution. See convergence.py for a spatial grid-convergence
// study built on top of these plotfile pairs, and plot_tgv2d.py for
// vorticity-decay heatmaps read directly from the plt<step> plotfiles.

#include "IO/InputParser.H"
#include "IO/Plotfile.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"
#include "TgvAnalytic.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace muphfasa;

namespace {

// Fills analytic_vel/analytic_rho with the TGV solution at lattice time t.
void calcAnalytic(amrex::MultiFab& analytic_rho, amrex::MultiFab& analytic_vel,
                  amrex::Real k, amrex::Real u0, amrex::Real nu, amrex::Real t)
{
    for (amrex::MFIter mfi(analytic_rho); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& r = analytic_rho.array(mfi);
        amrex::Array4<amrex::Real> const& u = analytic_vel.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int kc) {
            const amrex::Real x = amrex::Real(i) + amrex::Real(0.5);
            const amrex::Real y = amrex::Real(j) + amrex::Real(0.5);
            amrex::Real ux, uy;
            tgv::exactUV(x, y, k, u0, nu, t, ux, uy);
            r(i, j, kc, 0) = tgv::exactRho(x, y, k, u0, nu, t);
            u(i, j, kc, 0) = ux;
            u(i, j, kc, 1) = uy;
        });
    }
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // read in params and init sim
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        // grab sim params for analytic soln
        const auto ncells = params.ncells();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ncells[0] == ncells[1],
            "test_tgv2d: domain must be square (one vortex wavelength per side)");
        const int N = ncells[0];
        const amrex::Real k = tgv::waveNumber(N);

        amrex::Real u0_phys = params.refVelocity();
        amrex::Real u0_lat = sim->units().toLatticeVelocity(u0_phys);

        const amrex::Real tau = sim->units().relaxationTime();
        const amrex::Real nu  = (tau - amrex::Real(0.5)) * tgv::kCs2;

        const int max_step = params.maxStep();
        const std::string analytic_file = "analytic_plt";

        std::vector<amrex::Real> errors_vel;
        std::vector<amrex::Real> errors_rho;

        // run sim
        for (int step = 0; step <= max_step; ++step) {
            sim->advance(step);
            sim->writeOutput(step);

            // calc error in rho and vel
            if (step % params.plotInterval() == 0) {
                const amrex::MultiFab& sim_rho = sim->rho();
                const amrex::MultiFab& sim_vel = sim->vel();

                // calc and write analytic soln
                amrex::MultiFab analytic_rho(sim_rho.boxArray(), sim_rho.DistributionMap(),
                                             sim_rho.nComp(), sim_rho.nGrow());
                amrex::MultiFab analytic_vel(sim_vel.boxArray(), sim_vel.DistributionMap(),
                                             sim_vel.nComp(), sim_vel.nGrow());

                // calc analytic soln in lattice units
                calcAnalytic(analytic_rho, analytic_vel, k, u0_lat, nu, sim->time());
                // convert analytic (and sim copies) to physical for fcompare and error
                sim->units().toPhysical(analytic_rho, analytic_vel);
                const amrex::Real t_phys = sim->units().toPhysicalTime(sim->time());

                muphfasa::writeMacroPlotfile(plotfileName(analytic_file, step),
                        analytic_rho, analytic_vel, sim->geom(), t_phys, step);

                // calc error in physical units
                amrex::MultiFab sim_rho_phys(sim_rho.boxArray(), sim_rho.DistributionMap(),
                                             sim_rho.nComp(), 0);
                amrex::MultiFab sim_vel_phys(sim_vel.boxArray(), sim_vel.DistributionMap(),
                                             sim_vel.nComp(), 0);
                amrex::MultiFab::Copy(sim_rho_phys, sim_rho, 0, 0, sim_rho.nComp(), 0);
                amrex::MultiFab::Copy(sim_vel_phys, sim_vel, 0, 0, sim_vel.nComp(), 0);
                sim->units().toPhysical(sim_rho_phys, sim_vel_phys);

                amrex::MultiFab diff_rho(sim_rho.boxArray(), sim_rho.DistributionMap(),
                                             sim_rho.nComp(), 0);
                amrex::MultiFab diff_vel(sim_vel.boxArray(), sim_vel.DistributionMap(),
                                             sim_vel.nComp(), 0);

                amrex::MultiFab::Copy(diff_vel, analytic_vel, 0, 0, sim_vel.nComp(), 0);
                diff_vel.minus(sim_vel_phys, 0, sim_vel.nComp(), 0);
                diff_vel.abs(0, sim_vel.nComp(), 0);

                amrex::MultiFab::Copy(diff_rho, analytic_rho, 0, 0, sim_rho.nComp(), 0);
                diff_rho.minus(sim_rho_phys, 0, sim_rho.nComp(), 0);
                diff_rho.abs(0, sim_rho.nComp(), 0);

                amrex::Real err_vel = amrex::Real(0.0);
                for (int c = 0; c < sim_vel.nComp(); ++c) {
                    err_vel = std::max(err_vel, diff_vel.max(c));
                }
                errors_vel.push_back(err_vel);
                amrex::Print() << "At Time=" << sim->time() << ": Error=" << err_vel << "\n";

                errors_rho.push_back(diff_rho.max(0));
            }
        }

        amrex::Real final_vel_error = errors_vel.back();
        amrex::Real final_rho_error = errors_rho.back();
        amrex::Print() << "Final velocity error: " << final_vel_error << '\n';
        amrex::Print() << "Final rho error: " << final_rho_error << '\n';
    }
    amrex::Finalize();
    return 0;
}
