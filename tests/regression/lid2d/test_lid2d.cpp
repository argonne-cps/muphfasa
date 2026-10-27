// test_lid2d.cpp — Correctness test: 2-D lid-driven cavity, global checks.
//
// Runs the closed square cavity of ../inputs.bench from rest to steady state
// and checks, in lattice units straight off sim->vel()/sim->rho():
//
//   1. Mass:     sum(rho) is conserved (closed box, bounce-back walls).
//   2. Spin-up:  KE(t) = (1/2) sum|u|^2 rises from 0 and plateaus.
//   3. Balance:  at steady state, power in = viscous dissipation,
//                  P_lid = U_lid * sum_lid( nu * du/dy|wall )
//                  eps   = nu * sum|grad u|^2
//                from dKE/dt = P_lid - eps (boundary work by the lid minus
//                dissipation; the other walls do no work since u=0 there).
//   4. Ghia:     centerline extrema u_min (x=0.5), v_max / v_min (y=0.5)
//                against Ghia, Ghia & Shin (1982), Re=100 -- see README.md.
//
// Walls/lid are hardwired to the ../inputs.bench layout (NoSlip x-lo/x-hi/y-lo,
// lid at y-hi moving along +x). Exits nonzero if any check fails.

#include "IO/InputParser.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace muphfasa;

namespace {

// Ghia, Ghia & Shin (1982), J. Comput. Phys. 48(3), 387-411,
// doi:10.1016/0021-9991(82)90058-4, Re=100 (Tables I and II), u/U_lid, v/U_lid.
constexpr amrex::Real kGhiaUmin = amrex::Real(-0.21090);  // at x=0.5,  y=0.4531
constexpr amrex::Real kGhiaVmax = amrex::Real( 0.17527);  // at y=0.5,  x=0.2344
constexpr amrex::Real kGhiaVmin = amrex::Real(-0.24533);  // at y=0.5,  x=0.8047

// Pass/fail tolerances (see README.md for how they were chosen).
constexpr amrex::Real kTolMass    = amrex::Real(1e-9);  // relative drift in sum(rho)
constexpr amrex::Real kTolMono    = amrex::Real(1e-3);  // allowed relative KE dip between samples
constexpr amrex::Real kTolPlateau = amrex::Real(1e-4);  // relative KE change over last sample interval
constexpr amrex::Real kTolBalance = amrex::Real(0.20);  // |P_lid - eps| / P_lid at steady state
constexpr amrex::Real kTolGhia    = amrex::Real(0.03);  // relative error on each centerline extremum

constexpr amrex::Real kCs2 = amrex::Real(1.0) / amrex::Real(3.0);

struct Sample {
    amrex::Real time, mass, ke, power, eps;
};

// Lid power P_lid and dissipation eps (lattice units, dx=dt=1, per unit depth).
// Wall-adjacent gradients use the wall at distance 0.5 from the boundary cell
// centers: ghost = 2*u_wall - u_cell for central differences, and a quadratic
// fit through (wall, first cell, second cell) for the lid shear.
void computePowerAndDissipation(const amrex::MultiFab& vel, const amrex::Geometry& geom,
                                amrex::Real u_lid, amrex::Real nu,
                                amrex::Real& power, amrex::Real& eps)
{
    amrex::MultiFab vel_g(vel.boxArray(), vel.DistributionMap(), vel.nComp(), 1);
    amrex::MultiFab::Copy(vel_g, vel, 0, 0, vel.nComp(), 0);
    vel_g.FillBoundary(geom.periodicity());

    amrex::MultiFab grad_sq(vel.boxArray(), vel.DistributionMap(), 1, 0);
    amrex::MultiFab lid_shear(vel.boxArray(), vel.DistributionMap(), 1, 0);
    lid_shear.setVal(amrex::Real(0.0));

    const amrex::Box domain = geom.Domain();
    const int ilo = domain.smallEnd(0), ihi = domain.bigEnd(0);
    const int jlo = domain.smallEnd(1), jhi = domain.bigEnd(1);

    for (amrex::MFIter mfi(vel_g); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real const> const& u = vel_g.const_array(mfi);
        amrex::Array4<amrex::Real> const& g2 = grad_sq.array(mfi);
        amrex::Array4<amrex::Real> const& sh = lid_shear.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // component c of the neighbor at (i+di, j+dj); outside the domain
            // it is the wall-reflected ghost (only the y-hi wall moves, along x)
            auto nbr = [&](int di, int dj, int c) -> amrex::Real {
                const int ni = i + di, nj = j + dj;
                if (ni < ilo || ni > ihi || nj < jlo || nj > jhi) {
                    const amrex::Real uw = (nj > jhi && c == 0) ? u_lid : amrex::Real(0.0);
                    return amrex::Real(2.0) * uw - u(i, j, k, c);
                }
                return u(ni, nj, k, c);
            };
            const amrex::Real half = amrex::Real(0.5);
            const amrex::Real dudx = half * (nbr( 1, 0, 0) - nbr(-1, 0, 0));
            const amrex::Real dudy = half * (nbr( 0, 1, 0) - nbr( 0,-1, 0));
            const amrex::Real dvdx = half * (nbr( 1, 0, 1) - nbr(-1, 0, 1));
            const amrex::Real dvdy = half * (nbr( 0, 1, 1) - nbr( 0,-1, 1));
            g2(i, j, k) = dudx*dudx + dudy*dudy + dvdx*dvdx + dvdy*dvdy;

            if (j == jhi) {
                // u(s) = U + a*s + b*s^2, s = distance below the lid; du/dy|wall = -a
                const amrex::Real d1 = u(i, j,   k, 0) - u_lid;   // s = 0.5
                const amrex::Real d2 = u(i, j-1, k, 0) - u_lid;   // s = 1.5
                const amrex::Real b = (d2 - amrex::Real(3.0) * d1) / amrex::Real(1.5);
                const amrex::Real a = amrex::Real(2.0) * d1 - half * b;
                sh(i, j, k) = -a;
            }
        });
    }

    eps   = nu * grad_sq.sum(0);
    power = u_lid * nu * lid_shear.sum(0);
}

// Parabolic refinement of the extremum of f[0..n-1] sampled at (m+0.5)/n.
// sign=+1 for a maximum, -1 for a minimum.
void extremum(const std::vector<amrex::Real>& f, int sign, amrex::Real& val, amrex::Real& pos)
{
    const int n = static_cast<int>(f.size());
    int m = 0;
    for (int j = 1; j < n; ++j) {
        if (sign * f[j] > sign * f[m]) { m = j; }
    }
    val = f[m];
    pos = (amrex::Real(m) + amrex::Real(0.5)) / amrex::Real(n);
    if (m > 0 && m < n - 1) {
        const amrex::Real den = f[m-1] - amrex::Real(2.0) * f[m] + f[m+1];
        if (den != amrex::Real(0.0)) {
            const amrex::Real delta = amrex::Real(0.5) * (f[m-1] - f[m+1]) / den;
            val = f[m] - amrex::Real(0.25) * (f[m-1] - f[m+1]) * delta;
            pos = (amrex::Real(m) + amrex::Real(0.5) + delta) / amrex::Real(n);
        }
    }
}

// u(y) along x=0.5 and v(x) along y=0.5 (average of the two cells straddling
// the centerline; N even), normalized by the lid velocity.
void centerlineProfiles(const amrex::MultiFab& vel, int N, amrex::Real u_lid,
                        std::vector<amrex::Real>& u_of_y, std::vector<amrex::Real>& v_of_x)
{
    amrex::MultiFab vel_h(vel.boxArray(), vel.DistributionMap(), vel.nComp(), 0,
                          amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
    amrex::MultiFab::Copy(vel_h, vel, 0, 0, vel.nComp(), 0);

    u_of_y.assign(N, amrex::Real(0.0));
    v_of_x.assign(N, amrex::Real(0.0));
    const int c0 = N / 2 - 1, c1 = N / 2;
    for (amrex::MFIter mfi(vel_h); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real const> const& u = vel_h.const_array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (i == c0 || i == c1) { u_of_y[j] += amrex::Real(0.5) * u(i, j, 0, 0) / u_lid; }
                if (j == c0 || j == c1) { v_of_x[i] += amrex::Real(0.5) * u(i, j, 0, 1) / u_lid; }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(u_of_y.data(), N);
    amrex::ParallelDescriptor::ReduceRealSum(v_of_x.data(), N);
}

bool report(const char* name, bool ok, const char* detail)
{
    amrex::Print() << (ok ? "PASS  " : "FAIL  ") << name << ": " << detail << "\n";
    return ok;
}

std::string fmt(const char* f, amrex::Real a, amrex::Real b = 0, amrex::Real c = 0)
{
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, static_cast<double>(a), static_cast<double>(b),
                  static_cast<double>(c));
    return buf;
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    bool all_ok = true;
    {
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        const auto ncells = params.ncells();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            ncells.size() == 2 && ncells[0] == ncells[1] && ncells[0] % 2 == 0,
            "test_lid2d: domain must be a 2-D square with an even number of cells");
        const int N = ncells[0];

        // lid (y-hi wall) velocity, converted from physical to lattice units
        const amrex::Real u_lid = sim->units().toLatticeVelocity(params.Uwall()[3][0]);
        const amrex::Real nu = (sim->units().relaxationTime() - amrex::Real(0.5)) * kCs2;

        const int max_step = params.maxStep();
        const int every = params.plotInterval();
        const amrex::Geometry& geom = sim->geom();

        std::vector<Sample> hist;
        auto sample = [&]() {
            Sample s;
            s.time = sim->time();
            s.mass = sim->rho().sum(0);
            s.ke   = amrex::Real(0.5) * amrex::MultiFab::Dot(sim->vel(), 0, sim->vel().nComp(), 0);
            computePowerAndDissipation(sim->vel(), geom, u_lid, nu, s.power, s.eps);
            hist.push_back(s);
            amrex::Print() << "At Time=" << s.time << ": KE=" << s.ke << " P_lid=" << s.power
                           << " eps=" << s.eps << " mass=" << s.mass << "\n";
        };

        sample(); // t=0, at rest
        for (int step = 0; step <= max_step; ++step) {
            sim->advanceFused(step);
            sim->writeOutput(step);
            if (step % every == 0) { sample(); }
        }

        amrex::Print() << "\n--- lid2d validation (N=" << N << ", U_lid=" << u_lid
                       << " lattice, nu=" << nu << ") ---\n";

        // 1. mass conservation
        amrex::Real mass_drift = amrex::Real(0.0);
        for (const auto& s : hist) {
            mass_drift = std::max(mass_drift, std::abs(s.mass - hist.front().mass) / hist.front().mass);
        }
        all_ok &= report("mass conservation", mass_drift < kTolMass,
                         fmt("max relative drift in sum(rho) = %.3e (tol 1e-9)", mass_drift).c_str());

        // 2. KE spin-up: starts at 0, rises (up to small overshoot), plateaus
        amrex::Real worst_dip = amrex::Real(0.0);
        for (std::size_t n = 1; n < hist.size(); ++n) {
            worst_dip = std::max(worst_dip, (hist[n-1].ke - hist[n].ke) / hist[n-1].ke);
        }
        const Sample& last = hist.back();
        const amrex::Real plateau = std::abs(last.ke - hist[hist.size()-2].ke) / last.ke;
        const bool spinup_ok = (hist.front().ke == amrex::Real(0.0)) && (last.ke > amrex::Real(0.0))
                               && (worst_dip < kTolMono);
        all_ok &= report("KE spin-up", spinup_ok,
                         fmt("KE 0 -> %.4e, worst relative dip between samples %.2e (tol 1e-3)",
                             last.ke, worst_dip).c_str());
        all_ok &= report("KE plateau", plateau < kTolPlateau,
                         fmt("relative KE change over last sample interval = %.2e (tol 1e-4)",
                             plateau).c_str());

        // 3. steady-state energy balance
        const amrex::Real bal = std::abs(last.power - last.eps) / std::abs(last.power);
        all_ok &= report("energy balance", bal < kTolBalance,
                         fmt("P_lid=%.4e eps=%.4e |P-eps|/P=%.2e (tol 2e-1)",
                             last.power, last.eps, bal).c_str());

        // 4. Ghia centerline extrema
        std::vector<amrex::Real> u_of_y, v_of_x;
        centerlineProfiles(sim->vel(), N, u_lid, u_of_y, v_of_x);
        amrex::Real umin, ypos, vmax, xmax, vmin, xmin;
        extremum(u_of_y, -1, umin, ypos);
        extremum(v_of_x, +1, vmax, xmax);
        extremum(v_of_x, -1, vmin, xmin);
        auto ghia = [&](const char* name, amrex::Real val, amrex::Real pos, amrex::Real ref) {
            const amrex::Real err = std::abs(val - ref) / std::abs(ref);
            return report(name, err < kTolGhia,
                          fmt("%.5f at %.4f vs Ghia %.5f", val, pos, ref).append(
                              fmt(" (rel err %.2e, tol 3e-2)", err)).c_str());
        };
        all_ok &= ghia("Ghia u_min (x=0.5)", umin, ypos, kGhiaUmin);
        all_ok &= ghia("Ghia v_max (y=0.5)", vmax, xmax, kGhiaVmax);
        all_ok &= ghia("Ghia v_min (y=0.5)", vmin, xmin, kGhiaVmin);

        amrex::Print() << (all_ok ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    }
    amrex::Finalize();
    return all_ok ? 0 : 1;
}
