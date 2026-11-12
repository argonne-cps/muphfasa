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
#include <vector>

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

std::string readCheckpointHeader(const std::string& dir_name, CheckpointHeader& h)
{
    std::ifstream hdr(dir_name + "/Header");
    if (!hdr) {
        return "no Header file found in " + dir_name;
    }
    hdr >> h.version;
    if (h.version != kCheckpointVersion) {
        return "unrecognized checkpoint version \"" + h.version + "\" in " + dir_name +
               " (expected " + kCheckpointVersion + ")";
    }
    // The domain is read from its own line: amrex::Box's operator>> aborts on
    // malformed text instead of failing the stream.
    std::string domain_line;
    hdr >> h.step >> h.time >> std::ws;
    std::getline(hdr, domain_line);
    std::istringstream domain_stream(domain_line);
    if (!hdr || domain_line.rfind("((", 0) != 0 || !(domain_stream >> h.domain) || !(hdr >> h.ncomp)) {
        return "malformed Header in " + dir_name;
    }
    return {};
}

std::vector<std::string> checkpointMismatches(const std::string&      dir_name,
                                              const CheckpointHeader& h,
                                              const amrex::Box&       domain,
                                              int                     ncomp)
{
    std::vector<std::string> problems;
    if (h.domain != domain) {
        std::ostringstream msg;
        msg << "domain mismatch: " << dir_name << " was written for " << h.domain
            << " but this run has " << domain << " (check grid.ncells)";
        problems.push_back(msg.str());
    }
    if (h.ncomp != ncomp) {
        std::ostringstream msg;
        msg << dir_name << " holds " << h.ncomp << " distribution components but this run needs "
            << ncomp << " (check lbm.lattice_type)";
        problems.push_back(msg.str());
    }
    return problems;
}

void readCheckpoint(const std::string&    dir_name,
                    amrex::MultiFab&      f_mf,
                    const amrex::Geometry& geom,
                    int&                  step,
                    amrex::Real&          time)
{
    BL_PROFILE("readCheckpoint()");

    CheckpointHeader h;
    if (const std::string err = readCheckpointHeader(dir_name, h); !err.empty()) {
        amrex::Abort("readCheckpoint: " + err);
    }
    const std::vector<std::string> problems =
        checkpointMismatches(dir_name, h, geom.Domain(), f_mf.nComp());
    if (!problems.empty()) {
        std::string msg = "readCheckpoint: " + problems.front();
        for (std::size_t i = 1; i < problems.size(); ++i) { msg += "; " + problems[i]; }
        amrex::Abort(msg);
    }
    step = h.step;
    time = h.time;
    const amrex::Box& domain = h.domain;
    const int ncomp = h.ncomp;

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
