// Plotfile.H — Interface for writing AMReX plotfiles.
// Exports macroscopic fields (density, velocity, pressure, vorticity, …) in
// AMReX / VisIt / Paraview HDF5 or native plotfile format for post-processing.
// Distribution functions are not typically written to plotfiles; use
// Checkpoint.H if you need to save f directly.

#ifndef MUPHFASA_PLOTFILE_H
#define MUPHFASA_PLOTFILE_H

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

namespace muphfasa {

// Write a plotfile containing the supplied list of named MultiFabs.
// file_name  : output directory / file (e.g. "plt00100")
// mfs        : fields to write (all must share the same BoxArray / DM)
// var_names  : one name per component across all mfs (must total ncomp)
// geom       : domain geometry
// time       : simulation time (lattice or physical; recorded in the header)
// step       : time-step index (recorded in the header)
void writePlotfile(const std::string&                  file_name,
                   const amrex::Vector<const amrex::MultiFab*>& mfs,
                   const amrex::Vector<std::string>&   var_names,
                   const amrex::Geometry&              geom,
                   amrex::Real                         time,
                   int                                 step);

// Convenience overload: write rho_mf (1 comp) and vel_mf (D comps) with
// default variable names ("density", "vx", "vy"[, "vz"]).
void writeMacroPlotfile(const std::string&     file_name,
                        const amrex::MultiFab& rho_mf,
                        const amrex::MultiFab& vel_mf,
                        const amrex::Geometry& geom,
                        amrex::Real            time,
                        int                    step);

// Build a plotfile directory name from a prefix and step index.
// e.g. plotfileName("plt", 100) -> "plt00100"
std::string plotfileName(const std::string& prefix, int step);

} // namespace muphfasa

#endif // MUPHFASA_PLOTFILE_H
