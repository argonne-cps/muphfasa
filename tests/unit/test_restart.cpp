// test_restart.cpp — Checkpoint / restart round trip through ISimulation.
//
// A lid-driven cavity (so the flow actually evolves) is run straight through
// to step kLast, writing checkpoints along the way. A second simulation is
// restarted from the step-kChk checkpoint (amr.restart) and run to kLast with
// the same loop as src/main.cpp. The restart is exact only if the state it
// resumes from is complete, so the final distribution functions must match
// the uninterrupted run bit for bit. Also checks:
//   - the restored state (valid cells, time, start step) equals what the
//     original run had right after writing the checkpoint,
//   - restarting on a different box layout (max_grid_size) gives the same
//     answer, since readCheckpoint reads into the file's layout and copies.

#include <AMReX.H>
#include <AMReX_FileSystem.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "IO/Checkpoint.H"
#include "IO/InputParser.H"
#include "Simulation/ISimulation.H"
#include "Simulation/SimulationFactory.H"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace muphfasa;

namespace {

constexpr int kChkInt = 10;   // checkpoint every kChkInt steps
constexpr int kChk    = 20;   // step of the checkpoint we restart from
constexpr int kLast   = 35;   // last step of the run
const std::string kPrefix = "test_restart_chk";

void expect(bool ok, const std::string& what)
{
    if (!ok) { throw std::runtime_error(what); }
}

// Re-add (shadow) every key the run needs. ParmParse queries return the last
// definition, so successive calls reconfigure back-to-back simulations.
void setupInputs(const std::vector<int>& ncells, int max_grid_size, const std::string& lattice,
                 int chk_int)
{
    {
        amrex::ParmParse pp("grid");
        pp.addarr("ncells", ncells);
        pp.add("max_grid_size", max_grid_size);
        pp.add("n_ghost", 1);
    }
    {
        amrex::ParmParse pp("sim");
        pp.add("max_step", kLast);
        pp.add("stop_time", -1.0);
    }
    {
        amrex::ParmParse pp("lbm");
        pp.add("lattice_type", lattice);
        pp.add("u_lat", 0.1);
    }
    {
        amrex::ParmParse pp("units");
        pp.add("ref_length", 1.0);
        pp.add("ref_velocity", 1.0);
        pp.add("ref_density", 1.0);
        pp.add("Re", 10.0);
    }
    {
        // Closed box, lid on the +y face moving in +x.
        amrex::ParmParse pp("bc");
        std::vector<std::string> lo(AMREX_SPACEDIM, "NoSlip");
        std::vector<std::string> hi(AMREX_SPACEDIM, "NoSlip");
        hi[1] = "MovingWall";
        pp.addarr("lo_type", lo);
        pp.addarr("hi_type", hi);
        std::vector<amrex::Real> lid(AMREX_SPACEDIM, 0.0);
        lid[0] = 1.0;
        pp.addarr("hiy_Uwall", lid);
    }
    {
        amrex::ParmParse pp("geometry");
        pp.addarr("prob_lo", std::vector<amrex::Real>(AMREX_SPACEDIM, 0.0));
        pp.addarr("prob_hi", std::vector<amrex::Real>(AMREX_SPACEDIM, 1.0));
        pp.add("type", std::string("box"));
    }
    {
        amrex::ParmParse pp("amr");
        pp.add("plot_int", 0);
        pp.add("chk_int", chk_int);
        pp.add("check_file", kPrefix);
    }
}

void removeCheckpoints()
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        for (int step = 0; step <= kLast; step += kChkInt) {
            amrex::FileSystem::RemoveAll(checkpointName(kPrefix, step));
        }
    }
    amrex::ParallelDescriptor::Barrier();
}

// max |a - b| over the first nghost layers of ghosts plus the valid cells;
// a and b must share a BoxArray and DistributionMapping.
amrex::Real maxAbsDiff(const amrex::MultiFab& a, const amrex::MultiFab& b, int nghost)
{
    amrex::MultiFab d(a.boxArray(), a.DistributionMap(), a.nComp(), nghost);
    amrex::MultiFab::Copy(d, a, 0, 0, a.nComp(), nghost);
    amrex::MultiFab::Subtract(d, b, 0, 0, a.nComp(), nghost);
    amrex::Real m = 0;
    for (int c = 0; c < d.nComp(); ++c) { m = std::max(m, d.norm0(c, nghost)); }
    return m;
}

// Valid-cell max |a - b| where b may live on a different layout than a.
amrex::Real maxAbsDiffAnyLayout(const amrex::MultiFab& a, const amrex::MultiFab& b)
{
    amrex::MultiFab a_on_b(b.boxArray(), b.DistributionMap(), b.nComp(), 0);
    a_on_b.ParallelCopy(a, 0, 0, b.nComp());
    return maxAbsDiff(a_on_b, b, 0);
}

amrex::MultiFab copyOf(const amrex::MultiFab& src, int nghost)
{
    amrex::MultiFab dst(src.boxArray(), src.DistributionMap(), src.nComp(), nghost);
    amrex::MultiFab::Copy(dst, src, 0, 0, src.nComp(), nghost);
    return dst;
}

void runRestartTest(const char* tag, const std::vector<int>& ncells, int mg_write, int mg_read,
                    const std::string& lattice)
{
    amrex::Print() << "  " << tag << "  (max_grid_size write=" << mg_write << " restart=" << mg_read
                   << ")\n";
    removeCheckpoints();

    // ---- Reference: straight run, checkpoints every kChkInt steps ----------
    amrex::MultiFab f_final_ref;
    amrex::MultiFab f_at_chk;   // valid cells right after writeOutput(kChk)
    amrex::Real     t_at_chk = 0;
    {
        {
            amrex::ParmParse pp("amr");
            pp.remove("restart");
        }
        setupInputs(ncells, mg_write, lattice, kChkInt);
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();
        expect(sim->startStep() == 0, "fresh start must begin at step 0");
        expect(sim->time() == amrex::Real(0.0), "fresh start must begin at time 0");

        for (int step = sim->startStep(); step <= kLast; ++step) {
            sim->advance();
            sim->writeOutput(step);
            if (step == kChk) {
                f_at_chk = copyOf(sim->f(), 0);
                t_at_chk = sim->time();
            }
        }
        f_final_ref = copyOf(sim->f(), 0);
    }
    expect(checkpointExists(checkpointName(kPrefix, kChk)), "reference run did not write the checkpoint");
    // Guard against a vacuous test: the flow must still be changing after the checkpoint.
    expect(maxAbsDiffAnyLayout(f_at_chk, f_final_ref) > amrex::Real(1e-6),
           "flow does not evolve between the checkpoint and the end of the run");

    // ---- Restart from the step-kChk checkpoint -----------------------------
    {
        setupInputs(ncells, mg_read, lattice, 0);   // chk_int = 0: don't rewrite checkpoints
        {
            amrex::ParmParse pp("amr");
            pp.add("restart", checkpointName(kPrefix, kChk));
        }
        InputParser params;
        std::unique_ptr<ISimulation> sim = makeSimulation(params);
        sim->initialize();

        expect(sim->startStep() == kChk + 1, "restart must resume at the step after the checkpoint");
        expect(sim->time() == t_at_chk, "restored time differs from the original run");

        // Valid cells must be exactly what the original run had. Ghost cells
        // are not compared: they are refilled from the valid cells, and the
        // outside-domain populations that the BCs never write are never read.
        // The final-state comparison below covers anything that matters.
        const amrex::Real d0 = maxAbsDiffAnyLayout(f_at_chk, sim->f());
        amrex::Print() << "    restored state (valid cells) max|diff| = " << d0 << "\n";
        expect(d0 == amrex::Real(0.0), "restored state differs from the checkpointed state");

        for (int step = sim->startStep(); step <= kLast; ++step) {
            sim->advance();
            sim->writeOutput(step);
        }

        const amrex::Real d = maxAbsDiffAnyLayout(f_final_ref, sim->f());
        amrex::Print() << "    final f vs uninterrupted run  max|diff| = " << d << "\n";
        expect(d == amrex::Real(0.0), "restarted run diverged from the uninterrupted run");
    }

    removeCheckpoints();
    amrex::Print() << "    [PASS]\n";
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Checkpoint / restart round trip (DIM=" << AMREX_SPACEDIM << ") ===\n";
#if (AMREX_SPACEDIM == 2)
        runRestartTest("D2Q9 single box", {16, 16}, 16, 16, "D2Q9");
        runRestartTest("D2Q9 multi box, same layout", {16, 16}, 8, 8, "D2Q9");
        runRestartTest("D2Q9 re-boxed 8 -> 16", {16, 16}, 8, 16, "D2Q9");
        runRestartTest("D2Q9 re-boxed 16 -> 4", {16, 16}, 16, 4, "D2Q9");
#else
        runRestartTest("D3Q19 single box", {12, 12, 12}, 12, 12, "D3Q19");
        runRestartTest("D3Q19 re-boxed 6 -> 12", {12, 12, 12}, 6, 12, "D3Q19");
        runRestartTest("D3Q27 multi box, same layout", {12, 12, 12}, 6, 6, "D3Q27");
#endif
        amrex::Print() << "All restart tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
