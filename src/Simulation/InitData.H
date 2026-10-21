#ifndef MUPHFASA_INITDATA_H
#define MUPHFASA_INITDATA_H

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace muphfasa {

void initData(amrex::MultiFab& rho, amrex::MultiFab& vel);

} // namespace muphfasa

#endif // MUPHFASA_INITDATA_H