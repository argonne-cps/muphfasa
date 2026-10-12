// Plotfile.cpp — AMReX plotfile writing.
// Delegates to amrex::WriteSingleLevelPlotfile which produces output
// readable by VisIt and Paraview via the AMReX BoxLib I/O layer.

#include "Plotfile.H"

#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <iomanip>
#include <sstream>

namespace muphfasa {

void writePlotfile(const std::string&                          file_name,
                   const amrex::Vector<const amrex::MultiFab*>& /*mfs*/,
                   const amrex::Vector<std::string>&           /*var_names*/,
                   const amrex::Geometry&                      /*geom*/,
                   amrex::Real                                 time,
                   int                                         step)
{
    // TODO: merge mfs into a single MultiFab (or use WriteSingleLevelPlotfile
    //       directly if they already share BoxArray/DM), then call
    //       amrex::WriteSingleLevelPlotfile(file_name, merged, var_names, geom, time, step).
    amrex::Print() << "Plotfile: would write " << file_name
                   << "  step=" << step << "  time=" << time << "\n";
}

void writeMacroPlotfile(const std::string&     file_name,
                        const amrex::MultiFab& /*rho_mf*/,
                        const amrex::MultiFab& /*vel_mf*/,
                        const amrex::Geometry& geom,
                        amrex::Real            time,
                        int                    step)
{
    amrex::Vector<std::string> names = {"density"};
#if   (AMREX_SPACEDIM == 2)
    names.push_back("vx"); names.push_back("vy");
#elif (AMREX_SPACEDIM == 3)
    names.push_back("vx"); names.push_back("vy"); names.push_back("vz");
#endif
    // TODO: build combined MultiFab and forward to writePlotfile
    writePlotfile(file_name, {}, names, geom, time, step);
}

std::string plotfileName(const std::string& prefix, int step)
{
    std::ostringstream ss;
    ss << prefix << std::setw(5) << std::setfill('0') << step;
    return ss.str();
}

} // namespace muphfasa
