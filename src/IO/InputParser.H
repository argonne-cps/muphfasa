// InputParser.H — Wrapper around AMReX ParmParse for muphfasa input files.
// Reads all solver parameters from the input file and exposes typed accessors.
// Unrecognised keys are silently ignored by ParmParse; this class enforces
// required keys and provides defaults for optional ones.

#ifndef MUPHFASA_INPUT_PARSER_H
#define MUPHFASA_INPUT_PARSER_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <array>
#include <optional>
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

    // tau (relaxation time) and u_lat (dimensionless lattice reference
    // velocity) are two ways of pinning the same lattice-spacing choice --
    // see UnitConverter. At most one is normally given; nullopt means "not
    // specified in the input file", not zero. Supplying both is allowed
    // only if they're consistent with the other physical reference scales
    // (UnitConverter's constructor checks this and aborts otherwise).
    // Supplying neither falls back to UnitConverter's default u_lat.
    std::optional<amrex::Real> relaxationTime() const;  // lbm.tau
    std::optional<amrex::Real> latticeVelocity() const; // lbm.u_lat

    // ── Physical / unit parameters ────────────────────────────────────────
    amrex::Real reynoldsNumber() const;
    amrex::Real machNumber()     const;
    amrex::Real refLength()      const;   // [m]
    amrex::Real refVelocity()    const;   // [m/s]
    amrex::Real refDensity()     const;   // [kg/m^3]

    // ── Geometry ─────────────────────────────────────────────────────────
    std::string geometryType()   const;   // "box" | "sphere" | "stl" | …
    std::vector<amrex::Real> geometryProbLo()   const;   // [m] AMREX_SPACEDIM entries
    std::vector<amrex::Real> geometryProbHi()   const;   // [m] AMREX_SPACEDIM entries
    bool        useEB()          const;

    // ── Boundary conditions ───────────────────────────────────────────────
    // bc.lo_type = <x> [<y> [<z>]], bc.hi_type = <x> [<y> [<z>]], one name
    // per direction, each "NoSlip" | "MovingWall" | "Inflow" | "Outflow" | "PressureOutlet" |
    // "Periodic"
    // Returned array order matches muphfasa::faceIndex(dir, bound) in
    // BoundaryConditions.H: [xlo, xhi, ylo, yhi, zlo, zhi].
    std::array<std::string, 2 * AMREX_SPACEDIM> bcNames() const;
    amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 2 * AMREX_SPACEDIM> Uwall() const;

    // Per-face inflow velocity (bc.lox_Uin, bc.hix_Uin, ...), same shape as
    // Uwall() but with no restriction on the normal component (inflow
    // velocity is expected to point into the domain).
    amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 2 * AMREX_SPACEDIM> Uinflow() const;

    // Per-face density [kg/m^3] for PressureOutlet faces (bc.lox_rho,
    // bc.hix_rho, ...); faces without the key default to units.ref_density.
    std::array<amrex::Real, 2 * AMREX_SPACEDIM> faceDensity() const;

    // ── Body force ────────────────────────────────────────────────────────
    // forcing.accel = <ax> [<ay> [<az>]]: uniform body force per unit mass
    // [m/s^2], applied with the Guo scheme in the collision step. Absent key
    // -> zero vector, which keeps the force-free fast path (no forcing
    // kernels run). Validated configurations: periodic streamwise forcing, and
    // Inflow + PressureOutlet (tests/regression/poiseuille/inputs_inflow). The
    // zero-gradient Outflow has no density anchor and drifts with NoSlip walls,
    // so do not pair it with forcing.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> forcingAccel() const;

    // ── Output ───────────────────────────────────────────────────────────
    int         plotInterval()       const;
    int         checkpointInterval() const;
    std::string plotfilePrefix()     const;
    std::string checkpointPrefix()   const;
};

} // namespace muphfasa

#endif // MUPHFASA_INPUT_PARSER_H
