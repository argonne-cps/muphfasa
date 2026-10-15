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
// This test checks three things against that analytic solution:
//   1. The final-time profile matches the steady linear profile.
//   2. The profile matches the full time-dependent expansion at every
//      sampled time during the startup transient, not just at the end.
//   3. The *rate* at which the simulation approaches steady state matches
//      theory: the amplitude of the dominant (n=1) Fourier mode should decay
//      as exp(-pi^2*nu*t/H^2); we extract that mode from the simulated
//      profile over time and check its fitted decay rate against the
//      analytic one.
//
#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "IO/InputParser.H"
#include "LBM/BoundaryConditions.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace muphfasa;

namespace {

constexpr amrex::Real kPi  = amrex::Real(3.14159265358979323846);
constexpr amrex::Real kCs2 = amrex::Real(1.0) / amrex::Real(3.0);

void require(bool cond, const std::string& msg)
{
    if (!cond) { throw std::runtime_error(msg); }
}

// x-average of the vx component (component 0) of vel_mf, one value per
// y-index (0..Ny-1). Host-side reduction; fine for this small test grid.
std::vector<amrex::Real> averageVxOverX(const amrex::MultiFab& vel_mf, int Ny)
{
    std::vector<amrex::Real> sum(Ny, amrex::Real(0.0));
    std::vector<amrex::Long> count(Ny, 0);

    for (amrex::MFIter mfi(vel_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real const> const& v = vel_mf.const_array(mfi);
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
        for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                sum[j]   += v(i, j, lo.z, 0);
                count[j] += 1;
            }
        }
    }

    amrex::ParallelDescriptor::ReduceRealSum(sum.data(), Ny);
    amrex::ParallelDescriptor::ReduceLongSum(count.data(), Ny);

    std::vector<amrex::Real> avg(Ny);
    for (int j = 0; j < Ny; ++j) { avg[j] = sum[j] / amrex::Real(count[j]); }
    return avg;
}

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

// Projection of (profile - steady profile) onto the first sine mode
// sin(pi*y/H); analytically this is -2*U0/pi * exp(-pi^2*nu*t/H^2).
amrex::Real firstModeAmplitude(const std::vector<amrex::Real>& profile,
                               amrex::Real H, amrex::Real U0, int Ny)
{
    amrex::Real a1 = amrex::Real(0.0);
    for (int j = 0; j < Ny; ++j) {
        const amrex::Real y = j + amrex::Real(0.5);
        const amrex::Real deviation = profile[j] - U0 * y / H;
        a1 += deviation * std::sin(kPi * y / H);
    }
    return a1 * amrex::Real(2.0) / Ny;
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse::addfile("inputs");

        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        const int Ny = params.ncells()[1];
        const amrex::Real H   = amrex::Real(Ny);
        const amrex::Real tau = params.tau();
        const amrex::Real nu  = (tau - amrex::Real(0.5)) * kCs2;

        // y-hi face, x-component (tangential velocity of the moving wall).
        const amrex::Real U0 = params.Uwall()[faceIndex(1, 1)][0];

        const int max_step           = params.maxStep();
        const int snapshot_interval  = 200;

        std::vector<amrex::Real> snapshot_times;
        std::vector<std::vector<amrex::Real>> snapshot_profiles;

        for (int step = 0; step <= max_step; ++step) {
            sim->advance(step);
            if (step % snapshot_interval == 0) {
                snapshot_times.push_back(sim->time());
                snapshot_profiles.push_back(averageVxOverX(sim->vel(), Ny));
            }
        }

        require(!snapshot_profiles.empty(), "no snapshots recorded");

        // ── Check 1: steady state ────────────────────────────────────────
        const auto& final_profile = snapshot_profiles.back();
        amrex::Real max_err_steady = amrex::Real(0.0);
        for (int j = 0; j < Ny; ++j) {
            const amrex::Real y = j + amrex::Real(0.5);
            const amrex::Real u_exact = U0 * y / H;
            max_err_steady = std::max(max_err_steady, std::abs(final_profile[j] - u_exact));
        }
        amrex::Print() << "Max error vs. steady linear profile: " << max_err_steady << "\n";
        require(max_err_steady < amrex::Real(1.0e-3),
                "final profile does not match the steady Couette solution");

        // ── Check 2: time-dependent startup, tracked throughout ──────────
        amrex::Real max_err_transient = amrex::Real(0.0);
        for (std::size_t k = 0; k < snapshot_profiles.size(); ++k) {
            const amrex::Real t = snapshot_times[k];
            for (int j = 0; j < Ny; ++j) {
                const amrex::Real y = j + amrex::Real(0.5);
                const amrex::Real u_exact = analyticCouette(y, H, U0, nu, t);
                max_err_transient = std::max(max_err_transient,
                                             std::abs(snapshot_profiles[k][j] - u_exact));
            }
        }
        amrex::Print() << "Max error vs. time-dependent expansion (all snapshots): "
                       << max_err_transient << "\n";
        require(max_err_transient < amrex::Real(5.0e-3),
                "simulated profile departs from the time-dependent analytic "
                "expansion at some point during the startup transient");

        // ── Check 3: the dominant mode decays at the theoretical rate ────
        // Fit ln|A1(t)| ~ -rate*t + const by least squares, skipping the
        // first few snapshots where higher harmonics still contaminate A1.
        const std::size_t skip = std::min<std::size_t>(3, snapshot_profiles.size() / 4);
        std::vector<amrex::Real> ts, log_amps;
        for (std::size_t k = skip; k < snapshot_profiles.size(); ++k) {
            const amrex::Real a1 = firstModeAmplitude(snapshot_profiles[k], H, U0, Ny);
            if (std::abs(a1) < amrex::Real(1.0e-12)) { continue; } // below float noise floor
            ts.push_back(snapshot_times[k]);
            log_amps.push_back(std::log(std::abs(a1)));
        }
        require(ts.size() >= 2, "not enough usable samples to fit the mode decay rate");

        const std::size_t n = ts.size();
        amrex::Real sum_t = 0, sum_l = 0, sum_tt = 0, sum_tl = 0;
        for (std::size_t k = 0; k < n; ++k) {
            sum_t  += ts[k];
            sum_l  += log_amps[k];
            sum_tt += ts[k] * ts[k];
            sum_tl += ts[k] * log_amps[k];
        }
        const amrex::Real denom = amrex::Real(n) * sum_tt - sum_t * sum_t;
        require(std::abs(denom) > amrex::Real(1.0e-12), "degenerate fit (all sample times equal)");
        const amrex::Real fitted_slope = (amrex::Real(n) * sum_tl - sum_t * sum_l) / denom;

        const amrex::Real analytic_rate = -kPi * kPi * nu / (H * H);
        const amrex::Real rel_err = std::abs((fitted_slope - analytic_rate) / analytic_rate);
        amrex::Print() << "Fitted n=1 mode decay rate: " << fitted_slope
                       << "  analytic: " << analytic_rate
                       << "  relative error: " << rel_err << "\n";
        require(rel_err < amrex::Real(0.2),
                "measured decay rate of the startup transient's dominant mode "
                "does not match the analytic rate pi^2*nu/H^2 within 20%");

        amrex::Print() << "All Couette regression checks passed.\n";
    }
    amrex::Finalize();
    return 0;
}
