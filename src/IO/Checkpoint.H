// Checkpoint.H — Checkpoint and restart functionality.
// Writes the distribution function MultiFab and scalar simulation state to
// disk in AMReX checkpoint format.  A restart reads those files back and
// reconstructs the solver state, allowing seamless job continuation.

#ifndef MUPHFASA_CHECKPOINT_H
#define MUPHFASA_CHECKPOINT_H

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <string>

namespace muphfasa {

// Write a checkpoint directory containing f_mf and scalar metadata.
// dir_name : base directory name (e.g. "chk00100")
// step     : current time step (written to the header)
// time     : current simulation time in lattice units
void writeCheckpoint(const std::string&     dir_name,
                     const amrex::MultiFab& f_mf,
                     const amrex::Geometry& geom,
                     int                    step,
                     amrex::Real            time);

// Read a checkpoint written by writeCheckpoint.
// Fills f_mf and returns the step and time at which it was saved.
// The BoxArray / DistributionMapping in f_mf are used as-is; it is the
// caller's responsibility to ensure they match the checkpoint's grid.
void readCheckpoint(const std::string& dir_name,
                    amrex::MultiFab&   f_mf,
                    int&               step,
                    amrex::Real&       time);

// Returns true if the checkpoint directory looks valid (has a Header file).
bool checkpointExists(const std::string& dir_name);

// Build a checkpoint directory name from a prefix and step index.
// e.g. checkpointName("chk", 100) -> "chk00100"
std::string checkpointName(const std::string& prefix, int step);

} // namespace muphfasa

#endif // MUPHFASA_CHECKPOINT_H
