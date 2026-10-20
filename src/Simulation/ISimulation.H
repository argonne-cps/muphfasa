// ISimulation.H — Non-templated interface implemented by SimulationImpl<LatticeTag>.
// Lets callers (main.cpp) drive the simulation without knowing which
// LatticeTag was selected at runtime; see SimulationFactory.H for how that
// selection happens.

#ifndef MUPHFASA_ISIMULATION_H
#define MUPHFASA_ISIMULATION_H

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <LBM/Units.H>

namespace muphfasa {

class ISimulation
{
public:
    virtual ~ISimulation() = default;

    // ── Lifecycle ─────────────────────────────────────────────────────────
    virtual void initialize() = 0;
    virtual void advance(int step) = 0;
    virtual void writeOutput(int step) const = 0;

    // ── State accessors ───────────────────────────────────────────────────
    virtual const amrex::MultiFab& f()   const noexcept = 0;
    virtual const amrex::MultiFab& rho() const noexcept = 0;
    virtual const amrex::MultiFab& vel() const noexcept = 0;

    virtual amrex::Real time() const noexcept = 0;

    // BGK relaxation parameter actually in use (omega = 1/tau), derived from
    // UnitConverter::relaxationTime<LatticeTag>() -- see SimulationImpl's
    // constructor.
    virtual amrex::Real omega() const noexcept = 0;
    virtual const UnitConverter& units() const noexcept = 0;
    virtual const amrex::Geometry& geom() const noexcept = 0;
};

} // namespace muphfasa

#endif // MUPHFASA_ISIMULATION_H
