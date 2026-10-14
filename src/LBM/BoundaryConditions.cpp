// BoundaryConditions.cpp — Implementations of LBM boundary condition types.
// NoSlip uses full-way bounce-back; Inflow/Outflow use the Zou-He scheme.
// Periodic BCs are handled entirely by AMReX FillBoundary and are no-ops here.

#include "BoundaryConditions.H"
#include "LatticeDescriptor.H"

namespace muphfasa {

// ── BCManager ─────────────────────────────────────────────────────────────
void BCManager::addBC(std::unique_ptr<BoundaryConditionBase> bc)
{
    m_bcs.push_back(std::move(bc));
    bc.push_back(std::move(bc::bcrec_id));
}

void BCManager::applyAll(amrex::MultiFab&       f_mf,
                         const amrex::Geometry& geom,
                         int                    step) const
{
    for (const auto& bc : m_bcs) {
        bc->apply(f_mf, geom, step);
    }
}

// ── NoSlipBC ──────────────────────────────────────────────────────────────
template <typename LatticeTag>
void NoSlipBC<LatticeTag>::apply(amrex::MultiFab& f_mf,
                                  const amrex::Geometry& geom,
                                  int step) const
{
    // TODO: bounce-back:  f_out[opposite[q]] = f_in[q]  at solid boundaries
    Vector<BCRec> bc;

    // ...

    // fills interior and periodic domain boundary ghost cells

    // fills physical domain boundary ghost cells for a cell-centered multifab
    if (not geom.isAllPeriodic()) {
        GpuBndryFuncFab<MyExtBCFill> bf(MyExtBCFill{});
        PhysBCFunct<GpuBndryFuncFab<MyExtBCFill> > physbcf(geom, bc, bf);
        physbcf(mf, 0, mf.nComp(), mf.nGrowVector(), time, 0);
    }
}

// ── InflowBC ──────────────────────────────────────────────────────────────
template <typename LatticeTag>
InflowBC<LatticeTag>::InflowBC(amrex::Real rho_in,
                                amrex::Array<amrex::Real, AMREX_SPACEDIM> u_in)
    : m_rho(rho_in), m_u(u_in)
{}

template <typename LatticeTag>
void InflowBC<LatticeTag>::apply(amrex::MultiFab& /*f_mf*/,
                                  const amrex::Geometry& /*geom*/,
                                  int /*step*/) const
{
    // TODO: Zou-He velocity boundary condition at inflow faces
}

// ── OutflowBC ─────────────────────────────────────────────────────────────
template <typename LatticeTag>
void OutflowBC<LatticeTag>::apply(amrex::MultiFab& /*f_mf*/,
                                   const amrex::Geometry& /*geom*/,
                                   int /*step*/) const
{
    // TODO: Zou-He pressure boundary / zero-gradient extrapolation at outflow
}

// ── PeriodicBC ────────────────────────────────────────────────────────────
template <typename LatticeTag>
void PeriodicBC<LatticeTag>::apply(amrex::MultiFab& f_mf,
                                    const amrex::Geometry& geom,
                                    int step) const
{
    // AMReX handles periodicity in FillBoundary; nothing to do here.
    f_mf.FillBoundary(geom.periodicity());
}

// ── Explicit instantiations ───────────────────────────────────────────────
template class NoSlipBC<D2Q9>;
template class NoSlipBC<D3Q15>;
template class NoSlipBC<D3Q19>;
template class NoSlipBC<D3Q27>;

template class InflowBC<D2Q9>;
template class InflowBC<D3Q15>;
template class InflowBC<D3Q19>;
template class InflowBC<D3Q27>;

template class OutflowBC<D2Q9>;
template class OutflowBC<D3Q15>;
template class OutflowBC<D3Q19>;
template class OutflowBC<D3Q27>;

template class PeriodicBC<D2Q9>;
template class PeriodicBC<D3Q15>;
template class PeriodicBC<D3Q19>;
template class PeriodicBC<D3Q27>;

} // namespace muphfasa
