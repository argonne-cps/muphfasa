// TgvAnalytic.H — Shared analytic solution for the 2-D Taylor-Green vortex.
//
// Doubly-periodic domain, one vortex pair per domain (kx=ky=k=2*pi/N, N
// cells per side). Exact solution of the 2-D incompressible Navier-Stokes
// equations (see e.g. Kruger et al., "The Lattice Boltzmann Method", ch.
// 10.4), in lattice units (dx=dt=1), with cell centers at x=i+0.5, y=j+0.5:
//
//   u(x,y,t) = -u0 * cos(k*x) * sin(k*y) * exp(-2*nu*k^2*t)
//   v(x,y,t) =  u0 * sin(k*x) * cos(k*y) * exp(-2*nu*k^2*t)
//   rho(x,y,t) = 1 - (u0^2/(4*cs2)) * (cos(2*k*x) + cos(2*k*y)) * exp(-4*nu*k^2*t)
//
// Used by both InitData.cpp (t=0 initial condition, where nu drops out of
// the decay factor entirely) and test_tgv2d.cpp (comparison against the
// solver's output at any later time).

#ifndef MUPHFASA_TGV_ANALYTIC_H
#define MUPHFASA_TGV_ANALYTIC_H

#include <AMReX_REAL.H>

#include <cmath>

namespace muphfasa { namespace tgv {

constexpr amrex::Real kPi  = amrex::Real(3.14159265358979323846);
constexpr amrex::Real kCs2 = amrex::Real(1.0) / amrex::Real(3.0);

// One vortex wavelength per N-cell domain.
inline amrex::Real waveNumber(int N) noexcept
{
    return amrex::Real(2.0) * kPi / amrex::Real(N);
}

inline void exactUV(amrex::Real x, amrex::Real y, amrex::Real k, amrex::Real u0,
                    amrex::Real nu, amrex::Real t, amrex::Real& u, amrex::Real& v) noexcept
{
    const amrex::Real decay = std::exp(-amrex::Real(2.0) * nu * k * k * t);
    u = -u0 * std::cos(k * x) * std::sin(k * y) * decay;
    v =  u0 * std::sin(k * x) * std::cos(k * y) * decay;
}

inline amrex::Real exactRho(amrex::Real x, amrex::Real y, amrex::Real k, amrex::Real u0,
                            amrex::Real nu, amrex::Real t) noexcept
{
    const amrex::Real decay2 = std::exp(-amrex::Real(4.0) * nu * k * k * t);
    return amrex::Real(1.0) -
           (u0 * u0 / (amrex::Real(4.0) * kCs2)) *
           (std::cos(amrex::Real(2.0) * k * x) + std::cos(amrex::Real(2.0) * k * y)) * decay2;
}

}} // namespace muphfasa::tgv

#endif // MUPHFASA_TGV_ANALYTIC_H
