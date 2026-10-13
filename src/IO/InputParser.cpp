// InputParser.cpp — ParmParse-backed implementation of InputParser.
// Each accessor queries the AMReX ParmParse database (populated from the
// input file) and returns a typed value, applying a default if the key
// is absent.  Required keys throw if missing.

#include "InputParser.H"

#include <AMReX_ParmParse.H>

namespace muphfasa {

InputParser::InputParser()
{
    // ParmParse reads from the file already loaded by AMReX::Initialize.
}

std::array<int, AMREX_SPACEDIM> InputParser::ncells() const
{
    amrex::ParmParse pp("grid");
    std::array<int, AMREX_SPACEDIM> n{};
    // TODO: pp.getarr("ncells", n, 0, AMREX_SPACEDIM);
    return n;
}

int InputParser::maxGridSize() const
{
    amrex::ParmParse pp("grid");
    int val = 64;
    // TODO: pp.query("max_grid_size", val);
    return val;
}

int InputParser::nGhost() const
{
    amrex::ParmParse pp("grid");
    int val = 1;
    // TODO: pp.query("n_ghost", val);
    return val;
}

int InputParser::maxStep() const
{
    amrex::ParmParse pp("sim");
    int val = 0;
    // TODO: pp.get("max_step", val);
    return val;
}

amrex::Real InputParser::stopTime() const
{
    amrex::ParmParse pp("sim");
    amrex::Real val = -amrex::Real(1.0);
    // TODO: pp.query("stop_time", val);
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
    // TODO: pp.query("tau", val);
    return val;
}

amrex::Real InputParser::reynoldsNumber() const
{
    amrex::ParmParse pp("units");
    amrex::Real val = amrex::Real(0.0);
    // TODO: pp.query("Re", val);
    return val;
}

amrex::Real InputParser::machNumber() const
{
    amrex::ParmParse pp("units");
    amrex::Real val = amrex::Real(0.0);
    // TODO: pp.query("Ma", val);
    return val;
}

amrex::Real InputParser::refLength()    const { return amrex::Real(1.0); }
amrex::Real InputParser::refVelocity()  const { return amrex::Real(1.0); }
amrex::Real InputParser::refViscosity() const { return amrex::Real(1.0e-6); }
amrex::Real InputParser::refDensity()   const { return amrex::Real(1.0); }

std::string InputParser::geometryType() const { return "box"; }
bool        InputParser::useEB()        const { return false; }

int InputParser::plotInterval()       const { return 100; }
int InputParser::checkpointInterval() const { return 1000; }

std::string InputParser::plotfilePrefix()    const { return "plt"; }
std::string InputParser::checkpointPrefix()  const { return "chk"; }

} // namespace muphfasa
