// Units.cpp — Implementation of physical-to-lattice unit conversion.
// Derives dx and dt from the reference scales and lattice resolution,
// then expresses all conversions as multiplicative scale factors.

#include "Units.H"

#include <AMReX_BLassert.H>

#include <cmath>

namespace muphfasa {

namespace {
// Lattice reference velocity used when the user pins neither u_lat nor tau.
// Kept well below the lattice sound speed (cs = 1/sqrt(3) ~ 0.577) to keep
// compressibility error low while staying comfortably inside the BGK
// stability region; this is a standard convention (see e.g. Kruger et al.,
// "The Lattice Boltzmann Method", Ch. 7, unit conversion).
constexpr amrex::Real kDefaultLatticeVelocity = amrex::Real(0.1);

// Relative tolerance for the u_lat/tau cross-check below.
constexpr amrex::Real kConsistencyTol = amrex::Real(1.0e-6);
}

UnitConverter::UnitConverter(amrex::Real L_ref,
                              amrex::Real U_ref,
                              amrex::Real nu_ref,
                              amrex::Real rho_ref,
                              int         N_ref,
                              amrex::Real cs2,
                              std::optional<amrex::Real> u_lat,
                              std::optional<amrex::Real> tau)
    : m_L_ref(L_ref), m_U_ref(U_ref), m_nu_ref(nu_ref), m_rho_ref(rho_ref), m_cs2(cs2)
{
    m_dx = m_L_ref / amrex::Real(N_ref);

    if (tau) {
        // Solve for the u_lat implied by the requested relaxation time:
        // nu_lat = (tau-0.5)*cs2, and nu_lat = nu_ref*u_lat/(U_ref*dx).
        const amrex::Real nu_lat_from_tau = (*tau - amrex::Real(0.5)) * cs2;
        const amrex::Real u_lat_from_tau  = nu_lat_from_tau * m_U_ref * m_dx / m_nu_ref;

        if (u_lat) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(*u_lat - u_lat_from_tau) <
                    kConsistencyTol * std::max(amrex::Real(1.0), std::abs(u_lat_from_tau)),
                "UnitConverter: lbm.u_lat and lbm.tau were both specified but "
                "are inconsistent with each other given the other physical "
                "reference scales (ref_length/ref_velocity/ref_viscosity) and "
                "the grid resolution -- specify only one and let the other "
                "be derived.");
        }
        m_u_lat = u_lat_from_tau;
    } else if (u_lat) {
        m_u_lat = *u_lat;
    } else {
        m_u_lat = kDefaultLatticeVelocity;
    }

    m_dt     = m_u_lat * m_dx / m_U_ref;
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

amrex::Real UnitConverter::machNumber() const noexcept
{
    return m_u_lat / std::sqrt(m_cs2);
}

amrex::Real UnitConverter::relaxationTime() const noexcept
{
    return m_nu_lat / m_cs2 + amrex::Real(0.5);
}

} // namespace muphfasa
