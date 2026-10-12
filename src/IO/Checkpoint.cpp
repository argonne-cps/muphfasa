// Checkpoint.cpp — AMReX checkpoint write / read implementation.
// Uses VisMF to write MultiFab data and a plain-text Header file for
// scalar metadata (step, time, domain info), matching the AMReX convention
// used by IAMR, PeleLM, and other AMReX-based solvers.

#include "Checkpoint.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>

#include <fstream>
#include <sstream>
#include <iomanip>

namespace muphfasa {

void writeCheckpoint(const std::string&     dir_name,
                     const amrex::MultiFab& /*f_mf*/,
                     const amrex::Geometry& /*geom*/,
                     int                    step,
                     amrex::Real            time)
{
    // TODO:
    //   1. amrex::UtilCreateDirectory(dir_name)
    //   2. Write Header file (step, time, boxarray, periodicity)
    //   3. amrex::VisMF::Write(f_mf, dir_name + "/f")
    amrex::Print() << "Checkpoint: would write " << dir_name
                   << "  step=" << step << "  time=" << time << "\n";
}

void readCheckpoint(const std::string& dir_name,
                    amrex::MultiFab&   /*f_mf*/,
                    int&               step,
                    amrex::Real&       time)
{
    // TODO:
    //   1. Read Header file into step, time, and boxarray
    //   2. amrex::VisMF::Read(f_mf, dir_name + "/f")
    step = 0;
    time = amrex::Real(0.0);
    amrex::Print() << "Checkpoint: would read " << dir_name << "\n";
}

bool checkpointExists(const std::string& dir_name)
{
    std::ifstream hdr(dir_name + "/Header");
    return hdr.good();
}

std::string checkpointName(const std::string& prefix, int step)
{
    std::ostringstream ss;
    ss << prefix << std::setw(5) << std::setfill('0') << step;
    return ss.str();
}

} // namespace muphfasa
