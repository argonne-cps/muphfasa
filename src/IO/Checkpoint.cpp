// Checkpoint.cpp — AMReX checkpoint write / read implementation.
// Uses VisMF to write MultiFab data and a plain-text Header file for
// scalar metadata (step, time, domain info), matching the AMReX convention
// used by IAMR, PeleLM, and other AMReX-based solvers.

#include "Checkpoint.H"

#include <AMReX_BLProfiler.H>
#include <AMReX_Box.H>
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
    BL_PROFILE("writeCheckpoint()");

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

    // In checkpoint files, always write out FABs in NATIVE format
    amrex::FABio::Format thePrevFormat = amrex::FArrayBox::getFormat();

    amrex::FArrayBox::setFormat(amrex::FABio::FAB_NATIVE);

    amrex::VisMF::Write(f_mf, dir_name + "/f");

    // Restore the previous FAB format.
    amrex::FArrayBox::setFormat(thePrevFormat);

    amrex::Print() << "Checkpoint: wrote " << dir_name
                   << "  step=" << step << "  time=" << time << "\n";
}

void readCheckpoint(const std::string&    dir_name,
                    amrex::MultiFab&      f_mf,
                    const amrex::Geometry& geom,
                    int&                  step,
                    amrex::Real&          time)
{
    BL_PROFILE("readCheckpoint()");

    if (!checkpointExists(dir_name)) {
        amrex::Abort("readCheckpoint: no Header file found in " + dir_name);
    }

    std::ifstream hdr(dir_name + "/Header");
    std::string version;
    hdr >> version;
    if (version != kCheckpointVersion) {
        amrex::Abort("readCheckpoint: unrecognized checkpoint version in " + dir_name);
    }

    amrex::Box domain;
    int ncomp = 0;
    hdr >> step >> time >> domain >> ncomp;
    if (!hdr) {
        amrex::Abort("readCheckpoint: malformed Header in " + dir_name);
    }

    if (domain != geom.Domain()) {
        std::ostringstream msg;
        msg << "readCheckpoint: domain mismatch. " << dir_name << " was written for " << domain
            << " but this run has " << geom.Domain() << " (check grid.ncells)";
        amrex::Abort(msg.str());
    }
    if (ncomp != f_mf.nComp()) {
        std::ostringstream msg;
        msg << "readCheckpoint: " << dir_name << " holds " << ncomp
            << " distribution components but this run needs " << f_mf.nComp()
            << " (check lbm.lattice_type)";
        amrex::Abort(msg.str());
    }

    // Read with the checkpoint's own layout, then copy valid cells into f_mf.
    amrex::MultiFab tmp;
    amrex::VisMF::Read(tmp, dir_name + "/f");
    if (tmp.nComp() != ncomp || tmp.boxArray().minimalBox() != domain ||
        tmp.boxArray().numPts() != static_cast<amrex::Long>(domain.numPts())) {
        amrex::Abort("readCheckpoint: data in " + dir_name + "/f does not cover the domain (incomplete checkpoint?)");
    }
    f_mf.setVal(amrex::Real(0.0), 0, ncomp, f_mf.nGrowVect());
    f_mf.ParallelCopy(tmp, 0, 0, ncomp);

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
    ss << prefix << std::setw(7) << std::setfill('0') << step;
    return ss.str();
}

} // namespace muphfasa
