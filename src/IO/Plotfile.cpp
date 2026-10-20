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
                   const amrex::Vector<const amrex::MultiFab*>& mfs,
                   const amrex::Vector<std::string>&           var_names,
                   const amrex::Geometry&                      geom,
                   amrex::Real                                 time,
                   int                                         step)
{
    if (mfs.empty()) {
        amrex::Print() << "Plotfile: nothing to write for " << file_name << "\n";
        return;
    }

    int ncomp_total = 0;
    for (const auto* mf : mfs) { ncomp_total += mf->nComp(); }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ncomp_total == static_cast<int>(var_names.size()),
        "writePlotfile: var_names size does not match total component count");

    amrex::MultiFab plot_mf(mfs[0]->boxArray(), mfs[0]->DistributionMap(), ncomp_total, 0);

    int dcomp = 0;
    for (const auto* mf : mfs) {
        amrex::MultiFab::Copy(plot_mf, *mf, 0, dcomp, mf->nComp(), 0);
        dcomp += mf->nComp();
    }

    amrex::WriteSingleLevelPlotfile(file_name, plot_mf, var_names, geom, time, step);

    amrex::Print() << "Plotfile: wrote " << file_name
                   << "  step=" << step << "  time=" << time << "\n";
}

void writeMacroPlotfile(const std::string&     file_name,
                        const amrex::MultiFab& rho_mf,
                        const amrex::MultiFab& vel_mf,
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

    const amrex::Vector<const amrex::MultiFab*> mfs = {&rho_mf, &vel_mf};
    writePlotfile(file_name, mfs, names, geom, time, step);
}

std::string plotfileName(const std::string& prefix, int step)
{
    std::ostringstream ss;
    ss << prefix << std::setw(7) << std::setfill('0') << step;
    return ss.str();
}

} // namespace muphfasa
