// test_tgv2d.cpp — Regression test: 2-D Taylor-Green vortex decay.
//
// Doubly-periodic domain, one vortex pair per domain (kx=ky=k=2*pi/N, N
// cells per side). Exact solution of the 2-D incompressible Navier-Stokes
// equations (see e.g. Kruger et al., "The Lattice Boltzmann Method", ch.
// 10.4), in lattice units (dx=dt=1):
//
//   u(x,y,t) = -U0 * cos(k*x) * sin(k*y) * exp(-2*nu*k^2*t)
//   v(x,y,t) =  U0 * sin(k*x) * cos(k*y) * exp(-2*nu*k^2*t)
//   rho(x,y,t) = 1 - (U0^2/(4*cs2)) * (cos(2*k*x) + cos(2*k*y)) * exp(-4*nu*k^2*t)
//
// with cell centers at x=i+0.5, y=j+0.5. This file bypasses the
// InputParser/ISimulation layer entirely and drives Geometry + the raw
// LBM building blocks (computeEquilibriumMF/collideMF/streamInPlace/
// computeMacroscopic) directly, mirroring SimulationImpl::advance() but
// with a prescribed non-quiescent initial condition -- there is currently
// no InputParser-driven way to set an arbitrary IC, and this test needs
// the IC to actually be the TGV solution, not the solver's quiescent
// default.
//
// Two things get checked/plotted (see plot_tgv2d.py):
//   1. Spatial convergence: run at N=32,64,128, holding tau (hence nu_lat)
//      fixed and always keeping exactly one vortex wavelength per domain
//      (k=2*pi/N) -- the standard LBM "diffusive scaling" refinement: the
//      number of steps needed to reach a fixed non-dimensional decay time
//      grows as N^2 (nu*k^2*t fixed => t ~ 1/k^2 ~ N^2), so each run is
//      compared to the analytic solution at the same physical instant.
//      Written to tgv2d_convergence.csv; expect the velocity L2 error to
//      drop by ~4x (2nd order) each time N doubles.
//   2. A visual decay demonstration at a single resolution (N=64),
//      snapshotting the velocity field periodically. Written to
//      tgv2d_snapshots.csv.

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include "LBM/Collision.H"
#include "LBM/Equilibrium.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Macroscopic.H"
#include "LBM/Streaming.H"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace muphfasa;

namespace {

constexpr amrex::Real kPi   = amrex::Real(3.14159265358979323846);
constexpr amrex::Real kCs2  = amrex::Real(1.0) / amrex::Real(3.0);
constexpr amrex::Real kTau  = amrex::Real(0.8);             // fixed across resolutions
constexpr amrex::Real kNu   = (kTau - amrex::Real(0.5)) * kCs2;  // nu_lat = 0.1
constexpr amrex::Real kU0   = amrex::Real(0.02);
constexpr int          kNref = 32;   // reference resolution
constexpr int          kTref = 100;  // steps at kNref (~0.77 e-folds of decay)

void require(bool cond, const std::string& msg)
{
    if (!cond) { throw std::runtime_error(msg); }
}

amrex::Real kOf(int N) { return amrex::Real(2.0) * kPi / amrex::Real(N); }

void exactUV(amrex::Real x, amrex::Real y, amrex::Real k, amrex::Real t,
            amrex::Real& u, amrex::Real& v)
{
    const amrex::Real decay = std::exp(-amrex::Real(2.0) * kNu * k * k * t);
    u = -kU0 * std::cos(k * x) * std::sin(k * y) * decay;
    v =  kU0 * std::sin(k * x) * std::cos(k * y) * decay;
}

amrex::Real exactRho(amrex::Real x, amrex::Real y, amrex::Real k, amrex::Real t)
{
    const amrex::Real decay2 = std::exp(-amrex::Real(4.0) * kNu * k * k * t);
    return amrex::Real(1.0) -
           (kU0 * kU0 / (amrex::Real(4.0) * kCs2)) *
           (std::cos(amrex::Real(2.0) * k * x) + std::cos(amrex::Real(2.0) * k * y)) * decay2;
}

// Builds a periodic N x N single-box setup and the LBM state MultiFabs.
struct TgvState
{
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    amrex::MultiFab f, feq, rho, vel;
    int N;

    explicit TgvState(int N_) : N(N_)
    {
        const amrex::Box domain(amrex::IntVect(0, 0), amrex::IntVect(N - 1, N - 1));
        const amrex::RealBox real_box({0.0, 0.0}, {1.0, 1.0});
        const amrex::Array<int, 2> is_periodic{1, 1};
        geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);

        ba = amrex::BoxArray(domain);
        dm = amrex::DistributionMapping(ba);

        f.define  (ba, dm, D2Q9::Q, D2Q9::nGhost);
        feq.define(ba, dm, D2Q9::Q, 0);
        rho.define(ba, dm, 1, 0);
        vel.define(ba, dm, 2, 0);
    }

    void setInitialCondition()
    {
        const amrex::Real k = kOf(N);
        for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            amrex::Array4<amrex::Real> const& r = rho.array(mfi);
            amrex::Array4<amrex::Real> const& u = vel.array(mfi);
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    const amrex::Real x = i + amrex::Real(0.5);
                    const amrex::Real y = j + amrex::Real(0.5);
                    amrex::Real ux, uy;
                    exactUV(x, y, k, amrex::Real(0.0), ux, uy);
                    r(i, j, lo.z, 0) = exactRho(x, y, k, amrex::Real(0.0));
                    u(i, j, lo.z, 0) = ux;
                    u(i, j, lo.z, 1) = uy;
                }
            }
        }
        computeEquilibriumMF<D2Q9>(rho, vel, f);
    }

    void advance()
    {
        const amrex::Real omega = amrex::Real(1.0) / kTau;
        computeEquilibriumMF<D2Q9>(rho, vel, feq);
        collideMF<D2Q9, BGK>(f, feq, omega);
        f.FillBoundary(geom.periodicity());
        streamInPlace<D2Q9>(f, geom);
        computeMacroscopic<D2Q9>(f, rho, vel);
    }

    // L2 error of (u,v) against the analytic solution at lattice time t.
    amrex::Real velocityL2Error(amrex::Real t) const
    {
        const amrex::Real k = kOf(N);
        amrex::Real sum_sq = amrex::Real(0.0);
        amrex::Long count = 0;
        for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            amrex::Array4<amrex::Real const> const& u = vel.const_array(mfi);
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    const amrex::Real x = i + amrex::Real(0.5);
                    const amrex::Real y = j + amrex::Real(0.5);
                    amrex::Real ux, uy;
                    exactUV(x, y, k, t, ux, uy);
                    const amrex::Real du = u(i, j, lo.z, 0) - ux;
                    const amrex::Real dv = u(i, j, lo.z, 1) - uy;
                    sum_sq += du * du + dv * dv;
                    ++count;
                }
            }
        }
        return std::sqrt(sum_sq / amrex::Real(count));
    }
};

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        // ── Part 1: spatial convergence ──────────────────────────────────
        const std::vector<int> resolutions = {kNref, 2 * kNref, 4 * kNref};
        std::vector<amrex::Real> dxs, errors;

        {
            std::ofstream csv("tgv2d_convergence.csv");
            csv.precision(17);
            csv << "# tau=" << kTau << " nu=" << kNu << " U0=" << kU0 << "\n";
            csv << "N,dx,error\n";
            for (int N : resolutions) {
                const int nsteps = kTref * (N / kNref) * (N / kNref); // diffusive scaling
                TgvState state(N);
                state.setInitialCondition();
                for (int s = 0; s < nsteps; ++s) { state.advance(); }

                const amrex::Real dx = amrex::Real(1.0) / amrex::Real(N);
                const amrex::Real err = state.velocityL2Error(amrex::Real(nsteps));
                dxs.push_back(dx);
                errors.push_back(err);
                csv << N << ',' << dx << ',' << err << '\n';
                amrex::Print() << "N=" << N << " nsteps=" << nsteps
                               << " L2 velocity error=" << err << "\n";
            }
        }
        amrex::Print() << "Wrote tgv2d_convergence.csv\n";

        require(errors.size() == resolutions.size(), "convergence loop did not run");
        for (std::size_t k = 0; k + 1 < errors.size(); ++k) {
            // observed order p from error ~ C*dx^p between consecutive refinements
            const amrex::Real p = std::log(errors[k] / errors[k + 1]) /
                                  std::log(dxs[k] / dxs[k + 1]);
            amrex::Print() << "Observed order between N=" << resolutions[k]
                           << " and N=" << resolutions[k + 1] << ": " << p << "\n";
            require(p > amrex::Real(1.5),
                    "TGV velocity error is not converging at ~2nd order");
        }

        // ── Part 2: visual decay demonstration at a single resolution ────
        const int N_vis = 64;
        const int max_step_vis = 2000;
        const int snapshot_interval = 200;

        TgvState vis(N_vis);
        vis.setInitialCondition();

        std::ofstream csv("tgv2d_snapshots.csv");
        csv.precision(17);
        csv << "# N=" << N_vis << " tau=" << kTau << " nu=" << kNu << " U0=" << kU0 << "\n";
        csv << "time,i,j,u_sim,v_sim\n";

        auto dumpSnapshot = [&](amrex::Real t) {
            for (amrex::MFIter mfi(vis.vel); mfi.isValid(); ++mfi) {
                const amrex::Box& bx = mfi.validbox();
                amrex::Array4<amrex::Real const> const& u = vis.vel.const_array(mfi);
                const auto lo = amrex::lbound(bx);
                const auto hi = amrex::ubound(bx);
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        csv << t << ',' << i << ',' << j << ','
                            << u(i, j, lo.z, 0) << ',' << u(i, j, lo.z, 1) << '\n';
                    }
                }
            }
        };

        dumpSnapshot(amrex::Real(0.0));
        for (int step = 0; step < max_step_vis; ++step) {
            vis.advance();
            const int t = step + 1;
            if (t % snapshot_interval == 0) { dumpSnapshot(amrex::Real(t)); }
        }
        amrex::Print() << "Wrote tgv2d_snapshots.csv\n";

        amrex::Print() << "All TGV2D regression checks passed.\n";
    }
    amrex::Finalize();
    return 0;
}
