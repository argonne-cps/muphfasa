// InputParser.cpp — ParmParse-backed implementation of InputParser.
// Each accessor queries the AMReX ParmParse database (populated from the
// input file) and returns a typed value, applying a default if the key
// is absent.  Required keys throw if missing.

#include "InputParser.H"

#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>

#include <vector>

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

std::array<std::string, 2 * AMREX_SPACEDIM> InputParser::bcNames() const
{
    // bc.lo = NoSlip Inflow Periodic   (one name per direction: x [y [z]])
    // bc.hi = NoSlip Outflow Periodic
    amrex::ParmParse pp("bc");

    std::vector<std::string> lo, hi;
    pp.getarr("lo", lo);
    pp.getarr("hi", hi);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(lo.size()) == AMREX_SPACEDIM &&
        static_cast<int>(hi.size()) == AMREX_SPACEDIM,
        "InputParser::bcNames: bc.lo and bc.hi must each list exactly "
        "AMREX_SPACEDIM names (one per direction)");

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

int InputParser::plotInterval()       const { return 100; }
int InputParser::checkpointInterval() const { return 1000; }

std::string InputParser::plotfilePrefix()    const { return "plt"; }
std::string InputParser::checkpointPrefix()  const { return "chk"; }

} // namespace muphfasa
