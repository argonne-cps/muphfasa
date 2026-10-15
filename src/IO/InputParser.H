// InputParser.H — Wrapper around AMReX ParmParse for muphfasa input files.
// Reads all solver parameters from the input file and exposes typed accessors.
// Unrecognised keys are silently ignored by ParmParse; this class enforces
// required keys and provides defaults for optional ones.

#ifndef MUPHFASA_INPUT_PARSER_H
#define MUPHFASA_INPUT_PARSER_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <array>
#include <string>

namespace muphfasa {

class InputParser
{
public:
    // Reads from the file already loaded by AMReX (specified on the command
    // line as the first positional argument, or via amrex.input_file).
    InputParser();

    // ── Grid ─────────────────────────────────────────────────────────────
    std::vector<int> ncells()       const;  // cells per direction
    int                              maxGridSize()  const;
    int                              nGhost()       const;  // ghost-cell layers

    // ── Time stepping ─────────────────────────────────────────────────────
    int         maxStep()      const;
    amrex::Real stopTime()     const;  // physical end time (optional)

    // ── LBM parameters ───────────────────────────────────────────────────
    std::string latticeType()    const;  // "D2Q9" | "D3Q15" | "D3Q19" | "D3Q27"
    std::string collisionModel() const;  // "BGK" | "TRT" | "MRT"
    // Relaxation time (tau) is derived from the physical reference scales
    // via UnitConverter, not read directly -- see SimulationImpl's
    // constructor and ISimulation::omega().

    // ── Physical / unit parameters ────────────────────────────────────────
    amrex::Real reynoldsNumber() const;
    amrex::Real machNumber()     const;
    amrex::Real refLength()      const;   // [m]
    amrex::Real refVelocity()    const;   // [m/s]
    amrex::Real refViscosity()   const;   // [m^2/s]
    amrex::Real refDensity()     const;   // [kg/m^3]

    // ── Geometry ─────────────────────────────────────────────────────────
    std::string geometryType()   const;   // "box" | "sphere" | "stl" | …
    std::vector<amrex::Real> geometryProbLo()   const;   // [m] AMREX_SPACEDIM entries
    std::vector<amrex::Real> geometryProbHi()   const;   // [m] AMREX_SPACEDIM entries
    bool        useEB()          const;

    // ── Boundary conditions ───────────────────────────────────────────────
    // bc.lo_type = <x> [<y> [<z>]], bc.hi_type = <x> [<y> [<z>]], one name
    // per direction, each "NoSlip" | "Inflow" | "Outflow" | "Periodic"
    // Returned array order matches muphfasa::faceIndex(dir, bound) in
    // BoundaryConditions.H: [xlo, xhi, ylo, yhi, zlo, zhi].
    std::array<std::string, 2 * AMREX_SPACEDIM> bcNames() const;
    amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 2 * AMREX_SPACEDIM> Uwall() const;

    // ── Output ───────────────────────────────────────────────────────────
    int         plotInterval()       const;
    int         checkpointInterval() const;
    std::string plotfilePrefix()     const;
    std::string checkpointPrefix()   const;
};

} // namespace muphfasa

#endif // MUPHFASA_INPUT_PARSER_H
