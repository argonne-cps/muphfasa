// Units.cpp — Implementation of physical-to-lattice unit conversion.
// Derives dx and dt from the reference scales and lattice resolution,
// then expresses all conversions as multiplicative scale factors.

#include "Units.H"

#include <AMReX_Print.H>
#include <AMReX_Array.H>
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
constexpr amrex::Real kConsistencyTol = amrex::Real(1.0e-12);
}

UnitConverter::UnitConverter(amrex::Real L_ref,
                              amrex::Real U_ref,
                              amrex::Real Re,
                              amrex::Real rho_ref,
                              int         N_ref,
                              amrex::Real cs2,
                              std::optional<amrex::Real> u_lat,
                              std::optional<amrex::Real> tau)
    : m_L_ref(L_ref), m_U_ref(U_ref), m_Re(Re), m_rho_ref(rho_ref), m_cs2(cs2)
{
    // set length conversion
    m_dx = m_L_ref / amrex::Real(N_ref);

    // set nu ref
    m_nu_ref = m_U_ref * m_L_ref / m_Re;

    // Define lattice velocity from inputs or default
    if (tau) {
        // Solve for the u_lat implied by the requested relaxation time:
        // nu_lat = (tau-0.5)*cs2, and nu_lat = nu_ref*u_lat/(U_ref*dx).
        const amrex::Real nu_lat_from_tau = (*tau - amrex::Real(0.5)) * cs2;
        const amrex::Real u_lat_from_tau  = nu_lat_from_tau * m_U_ref * m_dx / m_nu_ref;

        if (u_lat) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                "UnitConverter: lbm.u_lat and lbm.tau were both specified."
                "specify only one and let the other be derived.");
        }
        m_u_lat = u_lat_from_tau;
    } else if (u_lat) {
        m_u_lat = *u_lat;
    } else {
        m_u_lat = kDefaultLatticeVelocity;
    }

    // set time converstion
    m_dt     = m_u_lat * m_dx / m_U_ref;

    // set lattice viscosity
    m_nu_lat = m_nu_ref * m_dt / (m_dx * m_dx);

    // set tau
    m_tau = 0.5 + m_nu_lat / m_cs2;

    // warn if tau or u_lat close to stability bounds
    if ((m_tau > 1.5) || (m_tau < 0.6)){
    amrex::Print() << "Warning: the value of tau: " << m_tau <<
                      " is outside ideal range of 0.6 < tau < 1.5\n";

    }
    if (m_u_lat > 0.15) {
    amrex::Print() << "Warning: the value of u_lat: " << m_u_lat <<
                      " is close to violating low Mach number constraint\n";

    }

    // confirm stability
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE((m_tau > 0.5) && (m_tau < 2.0),
                "tau is not within the stability range 0.5 < tau < 2.0."
                "Alter lbm.tau or lbm.u_lat to fix.");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE((m_u_lat < 0.3),
                "u_lat does not satisfy low Mach number limit"
                "Alter lbm.u_lat to fix.");


    
    // Print Unit settings
    amrex::Print() << "Physical Inputs:" << std::endl
                   << "Re: " << reynoldsNumber() << std::endl 
                   << "U_phys: " << m_U_ref << std::endl
                   << "L_phys: " << m_L_ref << std::endl
                   << "nu_phys: " << m_nu_ref << std::endl
                   << "Ngrid: " << N_ref << std::endl
    
                   << "Starting Simulation with Lattice Units:" << std::endl
                   << "U_lat: " << m_u_lat << std::endl
                   << "L_lat: " << toLatticeLength(m_L_ref) << std::endl
                   << "nu_lat: " << m_nu_lat << std::endl
                   << "tau: " << relaxationTime() << std::endl
                   << "Ma: " << machNumber() << std::endl
                   << "C_x: " << dx() << std::endl
                   << "C_t: " << dt() << std::endl;
}

amrex::Real UnitConverter::toLatticeLength   (amrex::Real x_phys)   const noexcept { return x_phys / m_dx; }
amrex::Real UnitConverter::toLatticeVelocity (amrex::Real u_phys)   const noexcept { return u_phys * m_dt / m_dx; }
amrex::Real UnitConverter::toLatticeTime     (amrex::Real t_phys)   const noexcept { return t_phys / m_dt; }
amrex::Real UnitConverter::toLatticeViscosity(amrex::Real nu_phys)  const noexcept { return nu_phys * m_dt / (m_dx * m_dx); }
amrex::Real UnitConverter::toLatticeAcceleration(amrex::Real a_phys) const noexcept { return a_phys * m_dt * m_dt / m_dx; }
amrex::Real UnitConverter::toLatticeDensity    (amrex::Real rho_phys) const noexcept { return rho_phys / m_rho_ref; }

amrex::Real UnitConverter::toPhysicalLength   (amrex::Real x_lat)   const noexcept { return x_lat * m_dx; }
amrex::Real UnitConverter::toPhysicalVelocity (amrex::Real u_lat)   const noexcept { return u_lat * m_dx / m_dt; }
amrex::Real UnitConverter::toPhysicalTime     (amrex::Real t_lat)   const noexcept { return t_lat * m_dt; }
amrex::Real UnitConverter::toPhysicalViscosity(amrex::Real nu_lat)  const noexcept { return nu_lat * m_dx * m_dx / m_dt; }
amrex::Real UnitConverter::toPhysicalAcceleration(amrex::Real a_lat) const noexcept { return a_lat * m_dx / (m_dt * m_dt); }
amrex::Real UnitConverter::toPhysicalDensity  (amrex::Real rho_lat) const noexcept { return rho_lat * m_rho_ref; }

void UnitConverter::toPhysical(amrex::MultiFab& rho, amrex::MultiFab& vel) const noexcept
{
    toPhysicalDensity(rho);
    toPhysicalVelocity(vel);
}

void UnitConverter::toPhysicalDensity(amrex::MultiFab& rho) const noexcept
{
    rho.mult(m_rho_ref);
}

void UnitConverter::toPhysicalVelocity(amrex::MultiFab& vel) const noexcept
{
    vel.mult(m_dx / m_dt);
}

amrex::Real UnitConverter::machNumber() const noexcept
{
    return m_u_lat / std::sqrt(m_cs2);
}

} // namespace muphfasa
