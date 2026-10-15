// GeometryHandler.cpp — AMReX geometry and optional EB setup.
// Constructs the Geometry from ParmParse inputs, creates the BoxArray
// by chopping the domain into tiles, and builds the DistributionMapping.

#include "GeometryHandler.H"
#include "IO/InputParser.H"

#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

namespace muphfasa {

GeometryHandler::GeometryHandler(const InputParser& params)
{
    const auto n_cells = params.ncells();
    amrex::IntVect dom_hi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        dom_hi[d] = n_cells[d] - 1;
    }
    const amrex::Box domain(amrex::IntVect(0), dom_hi);

    amrex::ParmParse pp("geometry");

    amrex::Vector<amrex::Real> prob_lo(AMREX_SPACEDIM), prob_hi(AMREX_SPACEDIM);
    pp.getarr("prob_lo", prob_lo, 0, AMREX_SPACEDIM);
    pp.getarr("prob_hi", prob_hi, 0, AMREX_SPACEDIM);
    const amrex::RealBox real_box(prob_lo.data(), prob_hi.data());

    amrex::Vector<int> is_periodic_in(AMREX_SPACEDIM, 0);
    pp.getarr("is_periodic", is_periodic_in, 0, AMREX_SPACEDIM);
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { is_periodic[d] = is_periodic_in[d]; }

    m_geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);
}

void GeometryHandler::initEB(const std::string& /*geom_type*/)
{
    // TODO: call AMReX EB2::Build with the requested implicit-function geometry
    m_has_eb = true;
}

void GeometryHandler::buildBoxArray(int max_grid_size)
{
    m_ba = amrex::BoxArray(m_geom.Domain());
    m_ba.maxSize(max_grid_size);
    m_dm = amrex::DistributionMapping(m_ba);
}

amrex::RealBox GeometryHandler::physDomain() const noexcept
{
    return m_geom.ProbDomain();
}

} // namespace muphfasa
