// Units.H — Conversion between physical (SI) units and lattice units.
// The lattice unit system fixes: dx_lat = dt_lat = rho_lat = 1.
// Physical scales are set at construction time; conversion factors are
// derived automatically so both directions are available at any call site.

#ifndef MUPHFASA_UNITS_H
#define MUPHFASA_UNITS_H

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <optional>

namespace muphfasa {

// Stability / accuracy limits for the lattice parameters. The constructor
// aborts outside the hard limits and warns outside the ideal ones;
// tools/inputinfo.cpp reuses them to report (not abort) and suggest fixes.
namespace unit_limits {
constexpr amrex::Real tau_hard_lo     = amrex::Real(0.5);   // tau must exceed this
constexpr amrex::Real tau_hard_hi     = amrex::Real(2.0);   // tau must stay below this
constexpr amrex::Real tau_ideal_lo    = amrex::Real(0.6);
constexpr amrex::Real tau_ideal_hi    = amrex::Real(1.5);
constexpr amrex::Real u_lat_hard_hi   = amrex::Real(0.3);   // low-Mach hard limit
constexpr amrex::Real u_lat_ideal_hi  = amrex::Real(0.15);
// Used when the user pins neither u_lat nor tau; well below cs ~ 0.577 and
// inside the BGK stability region (Kruger et al., Ch. 7, unit conversion).
constexpr amrex::Real u_lat_default   = amrex::Real(0.1);
} // namespace unit_limits

class UnitConverter
{
public:
    // Construct from physical reference quantities.
    // L_ref  [m]       : reference length (e.g. channel half-width)
    // U_ref  [m/s]     : reference velocity
    // rho_ref [kg/m^3] : reference density
    // N                : number of lattice cells spanning L_dom (resolution)
    // L_dom  [m]       : length of domain
    // cs2              : lattice sound speed squared (LatticeTag::cs2).
    //
    // nuphys is either set by the user or determined by using the 
    // Re number as set by the user. nu_phys take priority and user is warned.
    //      Re               : Reynolds number (dimensionless scale invariant)
    //      nu_phys [m^2/s]  : kinematic viscosity
    //
    // dt is pinned by exactly one of two user choices:
    //   u_lat : the dimensionless lattice reference velocity directly.
    //   tau   : the BGK relaxation time directly.
    // Supplying neither falls back to the conventional default lattice
    // velocity (0.1).
    //
    // inspect_only: skip the hard stability asserts (tau, u_lat) and the
    // warning/summary printing, so a diagnostic tool can inspect out-of-range
    // settings and report on them itself. The solver leaves this false.
    //
    UnitConverter(amrex::Real L_ref,
                  amrex::Real U_ref,
                  amrex::Real rho_ref,
                  std::vector<int> N,
                  std::vector<amrex::Real> domLo,
                  std::vector<amrex::Real> domHi,
                  amrex::Real cs2,
                  std::optional<amrex::Real> Re = std::nullopt,
                  std::optional<amrex::Real> nu_phys = std::nullopt,
                  std::optional<amrex::Real> u_lat = std::nullopt,
                  std::optional<amrex::Real> tau   = std::nullopt,
                  bool inspect_only = false);

    // ── Physical → lattice ───────────────────────────────────────────────
    amrex::Real toLatticeLength   (amrex::Real x_phys)   const noexcept;
    amrex::Real toLatticeVelocity (amrex::Real u_phys)   const noexcept;
    amrex::Real toLatticeTime     (amrex::Real t_phys)   const noexcept;
    amrex::Real toLatticeViscosity(amrex::Real nu_phys)  const noexcept;
    amrex::Real toLatticeAcceleration(amrex::Real a_phys) const noexcept;  // body force per unit mass
    amrex::Real toLatticeEnergy(amrex::Real a_phys) const noexcept;  
    amrex::Real toLatticeDensity    (amrex::Real rho_phys) const noexcept;

    // ── Lattice → physical ───────────────────────────────────────────────
    amrex::Real toPhysicalLength   (amrex::Real x_lat)   const noexcept;
    amrex::Real toPhysicalVelocity (amrex::Real u_lat)   const noexcept;
    amrex::Real toPhysicalTime     (amrex::Real t_lat)   const noexcept;
    amrex::Real toPhysicalViscosity(amrex::Real nu_lat)  const noexcept;
    amrex::Real toPhysicalAcceleration(amrex::Real a_lat) const noexcept;
    amrex::Real toPhysicalEnergy(amrex::Real a_phys) const noexcept;  
    amrex::Real toPhysicalDensity  (amrex::Real rho_lat) const noexcept;

    // ── Dimensionless parameters ─────────────────────────────────────────
    amrex::Real machNumber()     const noexcept;  // Ma = u_lat / cs
    amrex::Real reynoldsNumber() const noexcept { return m_Re; } // Re = U_ref * L_ref / nu_ref
    amrex::Real relaxationTime() const noexcept { return m_tau; } // tau = nu_lat/cs2 + 0.5

    // ── MultiFab lattice → physical (in-place) ─────────────────────────
    void toPhysical(amrex::MultiFab& rho, amrex::MultiFab& vel) const noexcept;
    void toPhysicalDensity(amrex::MultiFab& rho) const noexcept;
    void toPhysicalVelocity(amrex::MultiFab& vel) const noexcept;

    // ── Stored scales ───────────────────────────────────────────────────
    amrex::Real refLength()       const noexcept { return m_L_ref; }    // [m]
    amrex::Real refVelocity()     const noexcept { return m_U_ref; }    // [m/s]
    amrex::Real refDensity()      const noexcept { return m_rho_ref; }  // [kg/m^3]
    amrex::Real physViscosity()   const noexcept { return m_nu_phys; }  // [m^2/s]
    amrex::Real latticeVelocity() const noexcept { return m_u_lat; }    // u_lat
    amrex::Real latticeViscosity()const noexcept { return m_nu_lat; }   // nu_lat
    amrex::Real domainLength()    const noexcept { return m_L_dom; }    // longest side [m]
    amrex::Real gridCells()       const noexcept { return m_N_grid; }   // cells along longest side

    // ── Lattice spacings ────────────────────────────────────────────────
    amrex::Real dx() const noexcept { return m_dx; }  // lattice spacing [m]
    amrex::Real dt() const noexcept { return m_dt; }  // time step       [s]

private:
    amrex::Real m_L_ref, m_U_ref, m_Re, m_nu_phys, m_rho_ref;
    amrex::Real m_L_dom, m_N_grid;
    amrex::Real m_cs2;
    amrex::Real m_dx, m_dt;
    amrex::Real m_u_lat;   // lattice reference velocity (Ma * cs)
    amrex::Real m_nu_lat;  // lattice kinematic viscosity
    amrex::Real m_tau;  // relaxation time

    void setGridResolution(std::vector<int> Ncells,
                           std::vector<amrex::Real> probLo,
                           std::vector<amrex::Real> probHi);

};

} // namespace muphfasa

#endif // MUPHFASA_UNITS_H
