// test_poiseuille.cpp — Regression test: body-force-driven plane Poiseuille flow.
// Supports both 2D (AMREX_SPACEDIM=2) and 3D (AMREX_SPACEDIM=3) builds.
//
// Geometry: periodic in x (and y in 3D), NoSlip walls at y=lo/hi (2D) or
// z=lo/hi (3D), uniform body force (forcing.accel, along x) driving the flow.
// With H = wall-normal cell count (halfway-bounce-back convention, same as
// test_couette.cpp) the steady Stokes balance nu u'' + F = 0 gives
//     u(y) = F/(2 nu) * y * (H - y),   y = j + 1/2 at cell centres,
//     U_max = F H^2 / (8 nu),          U_mean = F H^2 / (12 nu) = (2/3) U_max
// (all in lattice units, F = toLatticeAcceleration(forcing.accel[0])).
//
// Two streamwise configurations share this test:
//   periodic:      periodic in x; the whole domain is checked.
//   inflow/outflow: bc.lo_type x = Inflow, bc.hi_type x = PressureOutlet; the inflow
//     velocity (bc.lox_Uin) must give the analytic U_mean mass flux so it is
//     compatible with the body-force-driven profile. At the two inlet/wall
//     corners NoSlip outranks Inflow, so the (1,0,+-1) diagonal populations
//     lose their inflow correction and only a fraction 1 - 1/(3H) of the
//     nominal Uin flux enters (w = 1/36; D2Q9, D3Q19, D3Q27); the test
//     therefore requires Uin = U_mean * 3H/(3H-1). The flow develops from
//     the uniform inlet, so only the streamwise window test.window = <lo> <hi>
//     (fractions of the domain length, default 0 1) is checked, chosen clear
//     of the entrance region and of the outlet.
//
// At every amr.plot_int step this writes a plt<step>/analytic_plt<step> pair
// (physical units, for AMReX's fcompare, see convergence.py) and prints the
// max error relative to U_max. At the final step it asserts (exit code 1 on
// failure), over the checked window:
//   max|u_x - u_analytic| / U_max   < test.max_rel_err     (default 1e-2)
//   |U_mean - (2/3) U_max| / U_max  < test.max_rel_err
//   max|u_transverse| / U_max       < test.max_transverse  (default 1e-6)
//   max rho - min rho               < test.max_rho_spread  (default 1e-8)
// (only the spatial profile is checked, i.e. steady state is assumed to have
// been reached by sim.max_step; see README.md.)

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParReduce.H>
#include <AMReX_Print.H>

#include "IO/InputParser.H"
#include "IO/Plotfile.H"
#include "LBM/BoundaryConditions.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <algorithm>
#include <cmath>
#include <memory>

using namespace muphfasa;

namespace {

// Wall-normal direction: y in 2D (dir=1), z in 3D (dir=2). Flow is along x.
constexpr int WALL_DIM = (AMREX_SPACEDIM == 2 ? 1 : 2);

void calcAnalytic(amrex::MultiFab& vel, amrex::Real H, amrex::Real F, amrex::Real nu)
{
    for (amrex::MFIter mfi(vel, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto box = mfi.tilebox();
        const auto vel_arr = vel.array(mfi);
        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
#if (AMREX_SPACEDIM == 3)
            const amrex::Real y = amrex::Real(k) + amrex::Real(0.5);
#else
            const amrex::Real y = amrex::Real(j) + amrex::Real(0.5);
#endif
            vel_arr(i, j, k, 0) = F / (amrex::Real(2.0) * nu) * y * (H - y);
            for (int c = 1; c < AMREX_SPACEDIM; ++c) { vel_arr(i, j, k, c) = amrex::Real(0.0); }
        });
    }
}

struct WindowStats {
    amrex::Real max_err;     // max |u_x - u_analytic|
    amrex::Real mean_u;      // mean u_x over the window
    amrex::Real max_trans;   // max |u_transverse|
    amrex::Real rho_spread;  // max rho - min rho
};

// Profile/transverse/density statistics (lattice units) over the cells with
// i_lo <= i < i_hi (streamwise index), against u(y) = F/(2 nu) y (H - y).
WindowStats measureWindow(const amrex::MultiFab& vel, const amrex::MultiFab& rho,
                          int i_lo, int i_hi, amrex::Real H, amrex::Real F, amrex::Real nu)
{
    using namespace amrex;
    const auto vel_arrs = vel.const_arrays();
    const auto rho_arrs = rho.const_arrays();
    constexpr Real big = Real(1e300);
    const auto r = ParReduce(
        TypeList<ReduceOpMax, ReduceOpSum, ReduceOpMax, ReduceOpMax, ReduceOpMin, ReduceOpSum>{},
        TypeList<Real, Real, Real, Real, Real, Real>{}, vel, IntVect(0),
        [=] AMREX_GPU_DEVICE(int b, int i, int j, int k) noexcept
            -> GpuTuple<Real, Real, Real, Real, Real, Real> {
            if (i < i_lo || i >= i_hi) { return {Real(0), Real(0), Real(0), -big, big, Real(0)}; }
#if (AMREX_SPACEDIM == 3)
            const Real y = Real(k) + Real(0.5);
#else
            const Real y = Real(j) + Real(0.5);
#endif
            const Real ux = vel_arrs[b](i, j, k, 0);
            const Real ua = F / (Real(2.0) * nu) * y * (H - y);
            Real trans = Real(0);
            for (int c = 1; c < AMREX_SPACEDIM; ++c) {
                trans = amrex::max(trans, amrex::Math::abs(vel_arrs[b](i, j, k, c)));
            }
            const Real rh = rho_arrs[b](i, j, k);
            return {amrex::Math::abs(ux - ua), ux, trans, rh, rh, Real(1)};
        });
    Real max_err = get<0>(r), sum_u = get<1>(r), max_trans = get<2>(r);
    Real rho_max = get<3>(r), rho_min = get<4>(r), count = get<5>(r);
    ParallelDescriptor::ReduceRealMax(max_err);
    ParallelDescriptor::ReduceRealSum(sum_u);
    ParallelDescriptor::ReduceRealMax(max_trans);
    ParallelDescriptor::ReduceRealMax(rho_max);
    ParallelDescriptor::ReduceRealMin(rho_min);
    ParallelDescriptor::ReduceRealSum(count);
    return {max_err, sum_u / count, max_trans, rho_max - rho_min};
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    {
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        amrex::Real max_rel_err = amrex::Real(1e-2);
        amrex::Real max_transverse = amrex::Real(1e-6);
        amrex::Real max_rho_spread = amrex::Real(1e-8);
        std::vector<amrex::Real> window{amrex::Real(0.0), amrex::Real(1.0)};
        {
            amrex::ParmParse pp("test");
            pp.query("max_rel_err", max_rel_err);
            pp.query("max_transverse", max_transverse);
            pp.query("max_rho_spread", max_rho_spread);
            pp.queryarr("window", window);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(window.size() == 2 && window[0] < window[1],
                "test_poiseuille: test.window must be two increasing fractions");
        }
        const int Nx = params.ncells()[0];
        const int i_lo = static_cast<int>(std::lround(window[0] * Nx));
        const int i_hi = static_cast<int>(std::lround(window[1] * Nx));
        const auto bcs = params.bcNames();
        const bool inflow_mode = (bcs[faceIndex(0, -1)] == "Inflow");

        const int Nwall = params.ncells()[WALL_DIM];
        const amrex::Real H = amrex::Real(Nwall);
        const amrex::Real tau = sim->units().relaxationTime();
        const amrex::Real nu  = (tau - amrex::Real(0.5)) / amrex::Real(3.0);
        const amrex::Real F   = sim->units().toLatticeAcceleration(params.forcingAccel()[0]);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(F != amrex::Real(0.0),
            "test_poiseuille: forcing.accel x-component must be nonzero");
        for (int d = 1; d < AMREX_SPACEDIM; ++d) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(params.forcingAccel()[d] == amrex::Real(0.0),
                "test_poiseuille: forcing.accel must point along x only");
        }

        const amrex::Real U_max = F * H * H / (amrex::Real(8.0) * nu);
        const amrex::Real U_mean = amrex::Real(2.0) / amrex::Real(3.0) * U_max;
        amrex::Print() << "tau=" << tau << " nu_lat=" << nu << " F_lat=" << F << " H=" << H << "\n"
                       << "Predicted U_max_lat=" << U_max << " (Mach=" << U_max * std::sqrt(amrex::Real(3.0))
                       << ")  U_mean_lat=" << U_mean << "\n"
                       << "Predicted U_max_phys=" << sim->units().toPhysicalVelocity(U_max)
                       << "  U_mean_phys=" << sim->units().toPhysicalVelocity(U_mean) << "\n";
        if (inflow_mode) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bcs[faceIndex(0, 1)] == "PressureOutlet",
                "test_poiseuille: Inflow at x=lo needs PressureOutlet at x=hi (zero-gradient Outflow has no density anchor)");
            const amrex::Real U_in = sim->units().toLatticeVelocity(params.Uinflow()[faceIndex(0, -1)][0]);
            const amrex::Real U_in_expected = U_mean * (amrex::Real(3.0) * H) / (amrex::Real(3.0) * H - amrex::Real(1.0));
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(U_in - U_in_expected) < amrex::Real(1e-6) * U_max,
                "test_poiseuille: bc.lox_Uin must equal (2/3) F H^2/(8 nu) * 3H/(3H-1) (corner-corrected mean velocity)");
            amrex::Print() << "Inflow/outflow mode: checking cells " << i_lo << " <= i < " << i_hi
                           << " of " << Nx << "\n";
        }

        const int max_step = params.maxStep();
        const std::string analytic_file = "analytic_plt";
        amrex::Real final_err = 0.0;

        for (int step = 0; step <= max_step; ++step) {
            sim->advanceSep(step);
            sim->writeOutput(step);

            if (step % params.plotInterval() == 0 || step == max_step) {
                const amrex::MultiFab& sim_vel = sim->vel();
                const amrex::MultiFab& sim_rho = sim->rho();

                amrex::MultiFab analytic_vel(sim_vel.boxArray(), sim_vel.DistributionMap(),
                                             sim_vel.nComp(), 0);
                amrex::MultiFab analytic_rho(sim_rho.boxArray(), sim_rho.DistributionMap(),
                                             sim_rho.nComp(), 0);
                amrex::MultiFab::Copy(analytic_rho, sim_rho, 0, 0, sim_rho.nComp(), 0);
                calcAnalytic(analytic_vel, H, F, nu);

                // error in lattice units over the checked window, relative to U_max
                final_err = measureWindow(sim_vel, sim_rho, i_lo, i_hi, H, F, nu).max_err / U_max;
                amrex::Print() << "At step=" << step << ": max|u-u_analytic|/U_max=" << final_err << "\n";

                if (step % params.plotInterval() == 0) {
                    sim->units().toPhysical(analytic_rho, analytic_vel);
                    const amrex::Real t_phys = sim->units().toPhysicalTime(sim->time());
                    writeMacroPlotfile(plotfileName(analytic_file, step),
                                       analytic_rho, analytic_vel, sim->geom(), t_phys, step);
                }
            }
        }

        // ── final assertions ────────────────────────────────────────────────
        const WindowStats st = measureWindow(sim->vel(), sim->rho(), i_lo, i_hi, H, F, nu);
        const amrex::Real mean_err = std::abs(st.mean_u - U_mean) / U_max;
        const amrex::Real transverse = st.max_trans / U_max;
        const amrex::Real rho_spread = st.rho_spread;

        amrex::Print() << "Final profile error (rel. to U_max): " << final_err << "\n"
                       << "Final mean-velocity error (rel. to U_max): " << mean_err << "\n"
                       << "Final transverse velocity (rel. to U_max): " << transverse << "\n"
                       << "Final rho spread: " << rho_spread << "\n";

        const bool ok = final_err < max_rel_err && mean_err < max_rel_err
                     && transverse < max_transverse && rho_spread < max_rho_spread;
        amrex::Print() << (ok ? "poiseuille PASSED\n" : "poiseuille FAILED\n");
        status = ok ? 0 : 1;
    }
    amrex::Finalize();
    return status;
}
