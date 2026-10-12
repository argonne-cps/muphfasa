// Units.cpp — Implementation of physical-to-lattice unit conversion.
// Derives dx and dt from the reference scales and lattice resolution,
// then expresses all conversions as multiplicative scale factors.

#include "Units.H"

#include <AMReX_Print.H>
#include <cmath>

namespace muphfasa {

UnitConverter::UnitConverter(amrex::Real L_ref,
                              amrex::Real U_ref,
                              amrex::Real nu_ref,
                              amrex::Real rho_ref,
                              int         N_ref)
    : m_L_ref(L_ref), m_U_ref(U_ref), m_nu_ref(nu_ref), m_rho_ref(rho_ref)
{
    // TODO: compute m_dx, m_dt, m_u_lat, m_nu_lat from the input scales
    m_dx     = amrex::Real(0.0);
    m_dt     = amrex::Real(0.0);
    m_u_lat  = amrex::Real(0.0);
    m_nu_lat = amrex::Real(0.0);

    amrex::ignore_unused(N_ref);
}

amrex::Real UnitConverter::toLatticeLength   (amrex::Real /*x*/)   const noexcept { return amrex::Real(0.0); }
amrex::Real UnitConverter::toLatticeVelocity (amrex::Real /*u*/)   const noexcept { return amrex::Real(0.0); }
amrex::Real UnitConverter::toLatticeTime     (amrex::Real /*t*/)   const noexcept { return amrex::Real(0.0); }
amrex::Real UnitConverter::toLatticeViscosity(amrex::Real /*nu*/)  const noexcept { return amrex::Real(0.0); }
amrex::Real UnitConverter::LatticeDensity    (amrex::Real /*rho*/) const noexcept { return amrex::Real(0.0); }

amrex::Real UnitConverter::toPhysicalLength   (amrex::Real /*x*/)   const noexcept { return amrex::Real(0.0); }
amrex::Real UnitConverter::toPhysicalVelocity (amrex::Real /*u*/)   const noexcept { return amrex::Real(0.0); }
amrex::Real UnitConverter::toPhysicalTime     (amrex::Real /*t*/)   const noexcept { return amrex::Real(0.0); }
amrex::Real UnitConverter::toPhysicalViscosity(amrex::Real /*nu*/)  const noexcept { return amrex::Real(0.0); }
amrex::Real UnitConverter::toPhysicalDensity  (amrex::Real /*rho*/) const noexcept { return amrex::Real(0.0); }

amrex::Real UnitConverter::reynoldsNumber() const noexcept
{
    // TODO: Re = U_ref * L_ref / nu_ref
    return amrex::Real(0.0);
}

amrex::Real UnitConverter::machNumber() const noexcept
{
    // TODO: Ma = u_lat * sqrt(3)   (cs = 1/sqrt(3))
    return amrex::Real(0.0);
}

amrex::Real UnitConverter::relaxationTime() const noexcept
{
    // TODO: tau = nu_lat / cs2 + 0.5,  cs2 = 1/3
    return amrex::Real(0.0);
}

} // namespace muphfasa
