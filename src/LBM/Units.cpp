// Units.cpp — Implementation of physical-to-lattice unit conversion.
// Derives dx and dt from the reference scales and lattice resolution,
// then expresses all conversions as multiplicative scale factors.

#include "Units.H"

#include <cmath>

namespace muphfasa {

UnitConverter::UnitConverter(amrex::Real L_ref,
                              amrex::Real U_ref,
                              amrex::Real nu_ref,
                              amrex::Real rho_ref,
                              int         N_ref,
                              amrex::Real U_lat)
    : m_L_ref(L_ref), m_U_ref(U_ref), m_nu_ref(nu_ref), m_rho_ref(rho_ref), m_u_lat(U_lat)
{
    // set conversion factors (length, time)
    m_dx     = m_L_ref / amrex::Real(N_ref);
    m_dt     = m_u_lat * m_dx / m_U_ref;

    // calc lattice viscosity
    m_nu_lat = m_nu_ref * m_dt / (m_dx * m_dx);
}

amrex::Real UnitConverter::toLatticeLength   (amrex::Real x_phys)   const noexcept { return x_phys / m_dx; }
amrex::Real UnitConverter::toLatticeVelocity (amrex::Real u_phys)   const noexcept { return u_phys * m_dt / m_dx; }
amrex::Real UnitConverter::toLatticeTime     (amrex::Real t_phys)   const noexcept { return t_phys / m_dt; }
amrex::Real UnitConverter::toLatticeViscosity(amrex::Real nu_phys)  const noexcept { return nu_phys * m_dt / (m_dx * m_dx); }
amrex::Real UnitConverter::LatticeDensity    (amrex::Real rho_phys) const noexcept { return rho_phys / m_rho_ref; }

amrex::Real UnitConverter::toPhysicalLength   (amrex::Real x_lat)   const noexcept { return x_lat * m_dx; }
amrex::Real UnitConverter::toPhysicalVelocity (amrex::Real u_lat)   const noexcept { return u_lat * m_dx / m_dt; }
amrex::Real UnitConverter::toPhysicalTime     (amrex::Real t_lat)   const noexcept { return t_lat * m_dt; }
amrex::Real UnitConverter::toPhysicalViscosity(amrex::Real nu_lat)  const noexcept { return nu_lat * m_dx * m_dx / m_dt; }
amrex::Real UnitConverter::toPhysicalDensity  (amrex::Real rho_lat) const noexcept { return rho_lat * m_rho_ref; }

amrex::Real UnitConverter::reynoldsNumber() const noexcept
{
    return m_U_ref * m_L_ref / m_nu_ref;
}

template <typename LatticeTag>
amrex::Real UnitConverter::machNumber() const noexcept
{
    return m_u_lat * std::sqrt(LatticeTag::cs2);
}

template <typename LatticeTag>
amrex::Real UnitConverter::relaxationTime() const noexcept
{
    return m_nu_lat / LatticeTag::cs2 + amrex::Real(0.5);
}

} // namespace muphfasa
