// Checkpoint.cpp — AMReX checkpoint write / read implementation.
// Uses VisMF to write MultiFab data and a plain-text Header file for
// scalar metadata (step, time, domain info), matching the AMReX convention
// used by IAMR, PeleLM, and other AMReX-based solvers.

#include "Checkpoint.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>

#include <fstream>
#include <iomanip>
#include <sstream>

namespace muphfasa {

namespace {
constexpr const char* kCheckpointVersion = "muphfasa_checkpoint_v1";
}

void writeCheckpoint(const std::string&     dir_name,
                     const amrex::MultiFab& f_mf,
                     const amrex::Geometry& geom,
                     int                    step,
                     amrex::Real            time)
{
    amrex::UtilCreateCleanDirectory(dir_name, false);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream hdr(dir_name + "/Header");
        hdr << kCheckpointVersion << "\n";
        hdr << step << "\n";
        hdr << std::setprecision(17) << time << "\n";
        hdr << geom.Domain() << "\n";
        hdr << f_mf.nComp() << "\n";
        hdr << f_mf.nGrowVect() << "\n";
    }
    amrex::ParallelDescriptor::Barrier();

    amrex::VisMF::Write(f_mf, dir_name + "/f");

    amrex::Print() << "Checkpoint: wrote " << dir_name
                   << "  step=" << step << "  time=" << time << "\n";
}

void readCheckpoint(const std::string& dir_name,
                    amrex::MultiFab&   f_mf,
                    int&               step,
                    amrex::Real&       time)
{
    if (!checkpointExists(dir_name)) {
        amrex::Abort("readCheckpoint: no Header file found in " + dir_name);
    }

    std::ifstream hdr(dir_name + "/Header");
    std::string version;
    hdr >> version;
    if (version != kCheckpointVersion) {
        amrex::Abort("readCheckpoint: unrecognized checkpoint version in " + dir_name);
    }
    hdr >> step;
    hdr >> time;

    amrex::VisMF::Read(f_mf, dir_name + "/f");

    amrex::Print() << "Checkpoint: read " << dir_name
                   << "  step=" << step << "  time=" << time << "\n";
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
