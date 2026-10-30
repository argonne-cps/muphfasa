// GeometryHandler.cpp — AMReX geometry and optional EB setup.
// Constructs the Geometry from ParmParse inputs, creates the BoxArray
// by chopping the domain into tiles, and builds the DistributionMapping.

#include "GeometryHandler.H"
#include "IO/InputParser.H"

#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EBFabFactory.H>
#endif
#include <AMReX_Print.H>

namespace muphfasa {

GeometryHandler::GeometryHandler(const InputParser& params)
{
    // create comp boundary
    const auto n_cells = params.ncells();
    amrex::IntVect dom_hi;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        dom_hi[d] = n_cells[d] - 1;
    }
    const amrex::Box domain(amrex::IntVect(0), dom_hi);

    // init physical boundary
    std::vector<amrex::Real> prob_lo_in = params.geometryProbLo();
    std::vector<amrex::Real> prob_hi_in = params.geometryProbHi();
    amrex::Array<amrex::Real, AMREX_SPACEDIM> prob_lo{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> prob_hi{};
    for (int dim=0; dim < AMREX_SPACEDIM; ++dim) {
        prob_lo[dim] = prob_lo_in[dim];
        prob_hi[dim] = prob_hi_in[dim];
    }

    const amrex::RealBox real_box(prob_lo, prob_hi);

    // check for periodicity
    amrex::Array<int, AMREX_SPACEDIM> is_periodic{};
    auto const bc_names = params.bcNames();
    for (int dim=0; dim < AMREX_SPACEDIM; ++dim){
        if (bc_names[2*dim] == "Periodic"){
            is_periodic[dim] = 1;
        } else {
            is_periodic[dim] = 0;
        }
        
    }

    m_geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);
}

void GeometryHandler::initEB(const std::string& /*geom_type*/)
{
#ifdef AMREX_USE_EB
    // TODO: call AMReX EB2::Build with the requested implicit-function geometry
    m_has_eb = true;
#else
    amrex::Abort(
        "GeometryHandler::initEB: muphfasa was built without EB support; "
        "rebuild with USE_EB=TRUE (GNUMake) or -Dmuphfasa_ENABLE_EB=ON (CMake)");
#endif
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
