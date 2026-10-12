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

GeometryHandler::GeometryHandler(const InputParser& /*params*/)
{
    // TODO: read domain extents and periodicity flags via params, construct m_geom
}

void GeometryHandler::initEB(const std::string& /*geom_type*/)
{
    // TODO: call AMReX EB2::Build with the requested implicit-function geometry
    m_has_eb = true;
}

void GeometryHandler::buildBoxArray(int /*max_grid_size*/)
{
    // TODO: create BoxArray from m_geom.Domain(), chop to max_grid_size,
    //       and build DistributionMapping with the default SFC strategy.
}

amrex::RealBox GeometryHandler::physDomain() const noexcept
{
    return m_geom.ProbDomain();
}

} // namespace muphfasa
