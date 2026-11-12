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
// dir_name : base directory name (e.g. "chk0000100")
// step     : current time step (written to the header)
// time     : current simulation time in lattice units
void writeCheckpoint(const std::string&     dir_name,
                     const amrex::MultiFab& f_mf,
                     const amrex::Geometry& geom,
                     int                    step,
                     amrex::Real            time);

// Read a checkpoint written by writeCheckpoint.
// Fills the valid cells of f_mf and returns the step and time at which it was
// saved. Ghost cells are zeroed, not restored: the caller must refill them (the
// solver's fillGhostCells(), the same call that ends every advance()).
// The file is read into its own BoxArray and copied across, so f_mf's
// BoxArray / DistributionMapping may differ from the writer's (different
// max_grid_size or rank count). Aborts with a message if the directory is
// missing or malformed, or if its domain or component count do not match
// geom / f_mf.
void readCheckpoint(const std::string&    dir_name,
                    amrex::MultiFab&      f_mf,
                    const amrex::Geometry& geom,
                    int&                  step,
                    amrex::Real&          time);

// Returns true if the checkpoint directory looks valid (has a Header file).
bool checkpointExists(const std::string& dir_name);

// Build a checkpoint directory name from a prefix and step index.
// e.g. checkpointName("chk", 100) -> "chk0000100"
std::string checkpointName(const std::string& prefix, int step);

} // namespace muphfasa

#endif // MUPHFASA_CHECKPOINT_H
