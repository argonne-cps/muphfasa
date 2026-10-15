// InputParser.cpp — ParmParse-backed implementation of InputParser.
// Each accessor queries the AMReX ParmParse database (populated from the
// input file) and returns a typed value, applying a default if the key
// is absent.  Required keys throw if missing.

#include "InputParser.H"

#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <vector>

namespace muphfasa {

InputParser::InputParser()
{
    // ParmParse reads from the file already loaded by AMReX::Initialize.
}

std::array<int, AMREX_SPACEDIM> InputParser::ncells() const
{
    amrex::ParmParse pp("grid");
    amrex::Vector<int> n(AMREX_SPACEDIM);
    pp.getarr("ncells", n, 0, AMREX_SPACEDIM);

    std::array<int, AMREX_SPACEDIM> out{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) { out[d] = n[d]; }
    return out;
}

int InputParser::maxGridSize() const
{
    amrex::ParmParse pp("grid");
    int val = 64;
    pp.query("max_grid_size", val);
    return val;
}

int InputParser::nGhost() const
{
    amrex::ParmParse pp("grid");
    int val = 1;
    pp.query("n_ghost", val);
    return val;
}

int InputParser::maxStep() const
{
    amrex::ParmParse pp("sim");
    int val = 0;
    pp.get("max_step", val);
    return val;
}

amrex::Real InputParser::stopTime() const
{
    amrex::ParmParse pp("sim");
    amrex::Real val = -amrex::Real(1.0);
    pp.query("stop_time", val);
    return val;
}

std::string InputParser::latticeType() const
{
    amrex::ParmParse pp("lbm");
    std::string val = "D3Q19";
    pp.query("lattice_type", val);
    return val;
}

std::string InputParser::collisionModel() const
{
    amrex::ParmParse pp("lbm");
    std::string val = "BGK";
    pp.query("collision_model", val);
    return val;
}

amrex::Real InputParser::tau() const
{
    amrex::ParmParse pp("lbm");
    amrex::Real val = amrex::Real(1.0);
    pp.query("tau", val);
    return val;
}

amrex::Real InputParser::reynoldsNumber() const
{
    amrex::ParmParse pp("units");
    amrex::Real val = amrex::Real(0.0);
    pp.query("Re", val);
    return val;
}

amrex::Real InputParser::machNumber() const
{
    amrex::ParmParse pp("units");
    amrex::Real val = amrex::Real(0.0);
    pp.query("Ma", val);
    return val;
}

amrex::Real InputParser::refLength() const
{
    amrex::ParmParse pp("units");
    amrex::Real val = amrex::Real(1.0);
    pp.query("ref_length", val);
    return val;
}

amrex::Real InputParser::refVelocity() const
{
    amrex::ParmParse pp("units");
    amrex::Real val = amrex::Real(1.0);
    pp.query("ref_velocity", val);
    return val;
}

amrex::Real InputParser::refViscosity() const
{
    amrex::ParmParse pp("units");
    amrex::Real val = amrex::Real(1.0e-6);
    pp.query("ref_viscosity", val);
    return val;
}

amrex::Real InputParser::refDensity() const
{
    amrex::ParmParse pp("units");
    amrex::Real val = amrex::Real(1.0);
    pp.query("ref_density", val);
    return val;
}

std::string InputParser::geometryType() const
{
    amrex::ParmParse pp("geometry");
    std::string val = "box";
    pp.query("type", val);
    return val;
}

bool InputParser::useEB() const
{
    amrex::ParmParse pp("geometry");
    int val = 0;
    pp.query("use_eb", val);
    return val != 0;
}

std::array<std::string, 2 * AMREX_SPACEDIM> InputParser::bcNames() const
{
    // bc.lo_type = NoSlip Inflow Periodic   (one name per direction: x [y [z]])
    // bc.hi_type = NoSlip Outflow Periodic
    amrex::ParmParse pp("bc");

    std::vector<std::string> lo, hi;
    pp.getarr("lo_type", lo);
    pp.getarr("hi_type", hi);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(lo.size()) == AMREX_SPACEDIM &&
        static_cast<int>(hi.size()) == AMREX_SPACEDIM,
        "InputParser::bcNames: bc.lo_type and bc.hi_type must each list "
        "exactly AMREX_SPACEDIM names (one per direction)");

    // Order matches muphfasa::faceIndex(dir, bound): [xlo, xhi, ylo, yhi,
    // zlo, zhi].
    std::array<std::string, 2 * AMREX_SPACEDIM> names{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        names[2 * dir]     = lo[dir];
        names[2 * dir + 1] = hi[dir];
    }
    return names;
}

int InputParser::plotInterval() const
{
    amrex::ParmParse pp("amr");
    int val = 100;
    pp.query("plot_int", val);
    return val;
}

int InputParser::checkpointInterval() const
{
    amrex::ParmParse pp("amr");
    int val = 1000;
    pp.query("chk_int", val);
    return val;
}

std::string InputParser::plotfilePrefix() const
{
    amrex::ParmParse pp("amr");
    std::string val = "plt";
    pp.query("plot_file", val);
    return val;
}

std::string InputParser::checkpointPrefix() const
{
    amrex::ParmParse pp("amr");
    std::string val = "chk";
    pp.query("check_file", val);
    return val;
}

} // namespace muphfasa
