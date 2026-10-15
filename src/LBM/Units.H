// Units.H — Conversion between physical (SI) units and lattice units.
// The lattice unit system fixes: dx_lat = dt_lat = rho_lat = 1.
// Physical scales are set at construction time; conversion factors are
// derived automatically so both directions are available at any call site.

#ifndef MUPHFASA_UNITS_H
#define MUPHFASA_UNITS_H

#include <AMReX_REAL.H>

#include <cmath>

namespace muphfasa {

class UnitConverter
{
public:
    // Construct from physical reference quantities.
    // L_ref  [m]       : reference length (e.g. channel half-width)
    // U_ref  [m/s]     : reference velocity
    // nu_ref [m^2/s]   : kinematic viscosity
    // rho_ref [kg/m^3] : reference density
    // N_ref            : number of lattice cells spanning L_ref (resolution)
    // U_lat [lu/lt]     : Lattice velocity in dimensionless lattice units (defaults to 0.1)
    UnitConverter(amrex::Real L_ref,
                  amrex::Real U_ref,
                  amrex::Real nu_ref,
                  amrex::Real rho_ref,
                  int         N_ref,
                  amrex::Real U_lat=amrex::Real(0.1));

    // ── Physical → lattice ───────────────────────────────────────────────
    amrex::Real toLatticeLength   (amrex::Real x_phys)   const noexcept;
    amrex::Real toLatticeVelocity (amrex::Real u_phys)   const noexcept;
    amrex::Real toLatticeTime     (amrex::Real t_phys)   const noexcept;
    amrex::Real toLatticeViscosity(amrex::Real nu_phys)  const noexcept;
    amrex::Real LatticeDensity    (amrex::Real rho_phys) const noexcept;

    // ── Lattice → physical ───────────────────────────────────────────────
    amrex::Real toPhysicalLength   (amrex::Real x_lat)   const noexcept;
    amrex::Real toPhysicalVelocity (amrex::Real u_lat)   const noexcept;
    amrex::Real toPhysicalTime     (amrex::Real t_lat)   const noexcept;
    amrex::Real toPhysicalViscosity(amrex::Real nu_lat)  const noexcept;
    amrex::Real toPhysicalDensity  (amrex::Real rho_lat) const noexcept;

    // ── Dimensionless parameters ─────────────────────────────────────────
    amrex::Real reynoldsNumber()    const noexcept;  // Re = U_ref * L_ref / nu_ref

    template <typename LatticeTag>
    amrex::Real machNumber() const noexcept  // Ma = U_lat / cs
    {
        return m_u_lat / std::sqrt(LatticeTag::cs2);
    }

    template <typename LatticeTag>
    amrex::Real relaxationTime() const noexcept  // tau = nu_lat/cs2 + 0.5
    {
        return m_nu_lat / LatticeTag::cs2 + amrex::Real(0.5);
    }

    // ── Lattice spacings ────────────────────────────────────────────────
    amrex::Real dx() const noexcept { return m_dx; }  // lattice spacing [m]
    amrex::Real dt() const noexcept { return m_dt; }  // time step       [s]

private:
    amrex::Real m_L_ref, m_U_ref, m_nu_ref, m_rho_ref;
    amrex::Real m_dx, m_dt;
    amrex::Real m_u_lat;   // lattice reference velocity (Ma * cs)
    amrex::Real m_nu_lat;  // lattice kinematic viscosity
};

} // namespace muphfasa

#endif // MUPHFASA_UNITS_H
