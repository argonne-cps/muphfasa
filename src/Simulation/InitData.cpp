
#include "InitData.H"

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

void initData(amrex::MultiFab& rho, amrex::MultiFab& vel,
              const UnitConverter& units)
{
    amrex::ignore_unused(units);
    // default behavior is quiescent start
    rho.setVal(amrex::Real(1.0));
    vel.setVal(amrex::Real(0.0));
}
} // namespace muphfasa
