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

std::vector<int> InputParser::ncells() const
{
    amrex::ParmParse pp("grid");
    amrex::Vector<int> n(AMREX_SPACEDIM);
    pp.getarr("ncells", n, 0, AMREX_SPACEDIM);

    return n;
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

std::vector<amrex::Real> InputParser::geometryProbLo() const
{
    amrex::ParmParse pp("geometry");

    std::vector<amrex::Real> prob_lo(AMREX_SPACEDIM);
    pp.getarr("prob_lo", prob_lo, 0, AMREX_SPACEDIM);

    return prob_lo;
}

std::vector<amrex::Real> InputParser::geometryProbHi() const
{
    amrex::ParmParse pp("geometry");

    std::vector<amrex::Real> prob_hi(AMREX_SPACEDIM);
    pp.getarr("prob_hi", prob_hi, 0, AMREX_SPACEDIM);

    return prob_hi;
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

amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 2 * AMREX_SPACEDIM> InputParser::Uwall() const {
    amrex::ParmParse pp("bc");

    std::vector<amrex::Real> lox_Uwall, hix_Uwall,
                             loy_Uwall, hiy_Uwall,
                             loz_Uwall, hiz_Uwall;
    
    
    amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 2 * AMREX_SPACEDIM> Uwall{};

    std::vector<std::string> param_Uwall = {"lox_Uwall", "hix_Uwall",
                                            "loy_Uwall", "hiy_Uwall",
                                            "loz_Uwall", "hiz_Uwall"};
    // loop over each wall 
    for (int dir=0; dir < 2 * AMREX_SPACEDIM; ++dir) {
        std::vector<amrex::Real> curr_U{};
        pp.queryarr(param_Uwall[dir], curr_U);
        if (curr_U.empty()) continue; // wall velocity not specified for this face

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(curr_U.size()) == AMREX_SPACEDIM,
            ("InputParser::Uwall: " + param_Uwall[dir] + " must list exactly "
             "AMREX_SPACEDIM components").c_str());

        // fill our Uwall array
        const int normal_dim = dir / 2;
        for (int dim=0; dim < AMREX_SPACEDIM; ++dim){
            // don't set wall-direction velocity
            if (dim != normal_dim) {
                Uwall[dir][dim] = curr_U[dim];
            } else {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    curr_U[dim] == 0.,
                    ("InputParser::Uwall: " + param_Uwall[dir] + " has a nonzero "
                     "component along the wall normal (component " + std::to_string(dim) +
                     "), which implies an expanding/contracting wall. Only wall "
                     "velocities tangential to the face are supported.").c_str());
            }
        }
    }

    return Uwall;
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
