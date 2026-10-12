// BoundaryConditions.H — Interfaces for LBM boundary condition types.
// Provides a polymorphic base class and concrete implementations for common
// boundary treatments.  Each BC object applies its rule to ghost/boundary
// cells in a MultiFab after streaming and before collision.

#ifndef MUPHFASA_BOUNDARY_CONDITIONS_H
#define MUPHFASA_BOUNDARY_CONDITIONS_H

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <memory>
#include <string>
#include <vector>

namespace muphfasa {

// ── Abstract base ─────────────────────────────────────────────────────────
class BoundaryConditionBase
{
public:
    virtual ~BoundaryConditionBase() = default;

    // Apply this BC to the distribution function MultiFab.
    virtual void apply(amrex::MultiFab&       f_mf,
                       const amrex::Geometry& geom,
                       int                    step) const = 0;

    virtual std::string name() const = 0;
};

// ── No-slip (full-way bounce-back) ────────────────────────────────────────
// Reflects all populations in the opposite direction; enforces zero velocity
// at solid walls.
template <typename LatticeTag>
class NoSlipBC : public BoundaryConditionBase
{
public:
    void apply(amrex::MultiFab&       f_mf,
               const amrex::Geometry& geom,
               int                    step) const override;

    std::string name() const override { return "NoSlip"; }
};

// ── Inflow (prescribed velocity / density) ────────────────────────────────
// Sets incoming populations using the Zou-He pressure / velocity scheme.
template <typename LatticeTag>
class InflowBC : public BoundaryConditionBase
{
public:
    explicit InflowBC(amrex::Real rho_in,
                      amrex::Array<amrex::Real, AMREX_SPACEDIM> u_in);

    void apply(amrex::MultiFab&       f_mf,
               const amrex::Geometry& geom,
               int                    step) const override;

    std::string name() const override { return "Inflow"; }

private:
    amrex::Real m_rho;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_u;
};

// ── Outflow (zero-gradient / extrapolation) ───────────────────────────────
template <typename LatticeTag>
class OutflowBC : public BoundaryConditionBase
{
public:
    void apply(amrex::MultiFab&       f_mf,
               const amrex::Geometry& geom,
               int                    step) const override;

    std::string name() const override { return "Outflow"; }
};

// ── Periodic (handled by AMReX FillBoundary) ─────────────────────────────
// This class exists for bookkeeping; periodicity is enforced by AMReX
// automatically when the Geometry is flagged periodic.
template <typename LatticeTag>
class PeriodicBC : public BoundaryConditionBase
{
public:
    void apply(amrex::MultiFab&       f_mf,
               const amrex::Geometry& geom,
               int                    step) const override;

    std::string name() const override { return "Periodic"; }
};

// ── BC manager ────────────────────────────────────────────────────────────
class BCManager
{
public:
    void addBC(std::unique_ptr<BoundaryConditionBase> bc);

    // Apply all registered BCs in insertion order.
    void applyAll(amrex::MultiFab&       f_mf,
                  const amrex::Geometry& geom,
                  int                    step) const;

private:
    std::vector<std::unique_ptr<BoundaryConditionBase>> m_bcs;
};

} // namespace muphfasa

#endif // MUPHFASA_BOUNDARY_CONDITIONS_H
