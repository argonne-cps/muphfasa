// InitData.cpp — Taylor-Green vortex initial condition.
//
// Overrides the default quiescent src/Simulation/InitData.cpp: this file's
// GNUmakefile/CMake target lists it ahead of the solver's own InitData.cpp
// (GNUmake VPATH searches the current directory first; CMake passes this
// path explicitly -- see tests/CMakeLists.txt), so SimulationImpl's
// setInitialConditions() picks this one up instead.
//
// Reads the vortex amplitude from `units.ref_velocity` (via
// InputParser::refVelocity()); the domain is assumed square with one vortex
// wavelength per side, k=2*pi/N with N taken from the MultiFab's own
// BoxArray so this works regardless of how grid.max_grid_size tiles the
// domain. The decay-rate (nu) dependence of the analytic solution only
// matters for t>0, so it's irrelevant here.

#include "InitData.H"
#include "TgvAnalytic.H"

#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include "IO/InputParser.H"

namespace muphfasa {

void initData(amrex::MultiFab& rho, amrex::MultiFab& vel)
{
    InputParser params;
    amrex::Real u0 = params.refVelocity();

    const amrex::Box domain = rho.boxArray().minimalBox();
    const int N = domain.length(0);
    const amrex::Real k = tgv::waveNumber(N);

    for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        amrex::Array4<amrex::Real> const& r = rho.array(mfi);
        amrex::Array4<amrex::Real> const& u = vel.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int kc) {
            const amrex::Real x = amrex::Real(i) + amrex::Real(0.5);
            const amrex::Real y = amrex::Real(j) + amrex::Real(0.5);
            amrex::Real ux, uy;
            tgv::exactUV(x, y, k, u0, amrex::Real(0.0), amrex::Real(0.0), ux, uy);
            r(i, j, kc, 0) = tgv::exactRho(x, y, k, u0, amrex::Real(0.0), amrex::Real(0.0));
            u(i, j, kc, 0) = ux;
            u(i, j, kc, 1) = uy;
        });
    }
}

} // namespace muphfasa
