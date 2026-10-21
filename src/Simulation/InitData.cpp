
#include "InitData.H"

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

void initData(amrex::MultiFab& rho, amrex::MultiFab& vel)
{
    // default behavior is quiescent start
    rho.setVal(amrex::Real(1.0));
    vel.setVal(amrex::Real(0.0));
}
} // namespace muphfasa
