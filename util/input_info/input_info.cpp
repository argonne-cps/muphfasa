// input_info.cpp — Standalone inspector for muphfasa input files.
//
// Reads an input file with the solver's own InputParser, builds the solver's
// UnitConverter from it, and prints what the solver would start with: grid,
// physical scales, lattice units (dx, dt, u_lat, nu_lat, tau), boundary
// conditions, estimated step count / end time, and output counts.
//
// Unlike the solver, out-of-range tau / u_lat only produce warnings here, with
// a suggested fix (adjust lbm.u_lat / lbm.tau, or refine the grid, and by how
// much). Nothing is run or allocated.
//
// Usage:  muphfasa_inputinfo <inputs file> [key=value ...]
// Extra key=value arguments override the file (AMReX ParmParse convention),
// which makes "what if lbm.tau were 0.8?" a one-liner. The executable is
// built for one dimensionality (AMREX_SPACEDIM): use a build whose
// muphfasa_DIM matches the input file.

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "IO/InputParser.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Units.H"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Real = amrex::Real;
namespace lim = muphfasa::unit_limits;

// Tool-only target for the "recommended" refinement: u_lat at the solver
// default, and a tau that leaves some margin above the ideal lower bound.
constexpr Real kTauComfort = Real(0.7);

// VisMF/plotfile metadata overhead, fitted to measured output. Per box: a FAB
// header plus a min/max table row per component. Small next to the data.
constexpr double kBoxBytes = 130, kBoxBytesPerComp = 50;
constexpr double kPlotHeaderBytes = 600, kChkHeaderBytes = 100;

#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
std::string fmt(const char* f, ...)
{
    va_list a, b;
    va_start(a, f);
    va_copy(b, a);
    const int n = std::vsnprintf(nullptr, 0, f, a);
    va_end(a);
    std::string s(n > 0 ? static_cast<std::size_t>(n) : 0u, '\0');
    if (n > 0) { std::vsnprintf(s.data(), static_cast<std::size_t>(n) + 1, f, b); }
    va_end(b);
    return s;
}

template <typename T>
std::string join(const std::vector<T>& v, const char* sep, const char* spec)
{
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) { s += sep; }
        s += fmt(spec, static_cast<double>(v[i]));
    }
    return s;
}

template <typename Arr>
std::string joinArr(const Arr& v, const char* spec)
{
    std::string s = "(";
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        if (i) { s += ", "; }
        s += fmt(spec, static_cast<double>(v[i]));
    }
    return s + ")";
}

void row(const char* label, const std::string& value)
{
    amrex::Print() << fmt("  %-16s %s\n", label, value.c_str());
}

void section(const char* title) { amrex::Print() << "\n" << title << "\n"; }

struct Advisory {
    bool warning;                    // false => informational note
    std::string headline;
    std::vector<std::string> detail;
};

// ── Lattice lookup (mirrors SimulationFactory's dispatch) ───────────────────
struct LatticeInfo {
    bool valid;
    int Q, nGhost;
    Real cs2;
};

template <typename L>
LatticeInfo infoOf() { return {true, L::Q, L::nGhost, L::cs2}; }

LatticeInfo lookupLattice(const std::string& name)
{
#if (AMREX_SPACEDIM == 2)
    if (name == "D2Q9")  { return infoOf<muphfasa::D2Q9>(); }
#else
    if (name == "D3Q15") { return infoOf<muphfasa::D3Q15>(); }
    if (name == "D3Q19") { return infoOf<muphfasa::D3Q19>(); }
    if (name == "D3Q27") { return infoOf<muphfasa::D3Q27>(); }
#endif
    return {false, 0, 0, Real(1.0) / Real(3.0)};
}

// Scale ncells by (at least) factor f keeping Lx/Nx == Ly/Ny == ... : multiply
// the reduced integer ratio N_i/g by t = ceil(f*g).
std::vector<int> scaledCells(const std::vector<int>& N, Real f)
{
    int g = 0;
    for (int n : N) { g = std::gcd(g, n); }
    const int t = static_cast<int>(std::ceil(f * g * (1.0 - 1e-12)));
    std::vector<int> out;
    for (int n : N) { out.push_back(n / g * t); }
    return out;
}

std::string humanBytes(double b)
{
    const char* unit[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int i = 0;
    while (b >= 1024.0 && i < 4) { b /= 1024.0; ++i; }
    return fmt("%.3g %s", b, unit[i]);
}

int countOf(const char* prefix, const char* key)
{
    amrex::ParmParse pp(prefix);
    return pp.contains(key) ? pp.countval(key) : 0;
}

// ── tau / u_lat range analysis ──────────────────────────────────────────────
struct LatticeCheck {
    const muphfasa::UnitConverter& units;
    Real cs2;
    Real bc_ratio;        // max boundary speed / U_ref (0 if none)
    std::string pinned;   // "lbm.tau" | "lbm.u_lat" | "default"
    std::vector<int> ncells;
    int dim;
    Real n_adv;           // steps the run is estimated to take
};

void checkLatticeParams(const LatticeCheck& c, std::vector<Advisory>& out)
{
    const Real tau = c.units.relaxationTime();
    const Real u   = c.units.latticeVelocity();
    const Real Re  = c.units.reynoldsNumber();
    const Real Nref = c.units.toLatticeLength(c.units.refLength());

    // tau - 1/2 = s * u_lat   (nu_lat = u_lat * N_ref / Re,  tau = 1/2 + nu_lat/cs2)
    const Real s = Nref / (Re * c.cs2);
    const Real r = std::max(Real(1), c.bc_ratio);
    const Real u_cap = lim::u_lat_ideal_hi / r;
    const Real u_lo  = (lim::tau_ideal_lo - Real(0.5)) / s;
    const Real u_hi  = std::min(u_cap, (lim::tau_ideal_hi - Real(0.5)) / s);
    const Real u_pref = std::min(lim::u_lat_default, u_cap);

    const bool u_nonpos  = !(u > Real(0));
    const bool tau_hard  = !(tau > lim::tau_hard_lo && tau < lim::tau_hard_hi);
    const bool u_hard    = !(u < lim::u_lat_hard_hi);
    const bool tau_low   = tau < lim::tau_ideal_lo;
    const bool tau_high  = tau > lim::tau_ideal_hi;
    const bool u_high    = u > lim::u_lat_ideal_hi;
    const bool bc_high   = !u_high && (u * c.bc_ratio > lim::u_lat_ideal_hi);

    if (u_nonpos) {
        out.push_back({true,
            fmt("u_lat = %.4g is not positive, so dt is not positive.", u),
            {"This happens when lbm.tau <= 0.5 (negative viscosity) or lbm.u_lat <= 0."}});
    }
    if (tau_low || tau_high) {
        Advisory a{true,
            fmt("tau = %.4g is outside the ideal range %.2g < tau < %.2g.%s", tau,
                lim::tau_ideal_lo, lim::tau_ideal_hi,
                tau_hard ? " The solver aborts at startup (hard limit 0.5 < tau < 2.0)." : ""),
            {}};
        a.detail.push_back(tau_low
            ? "tau near 0.5 means very low lattice viscosity: BGK loses stability / accuracy at this Re and resolution."
            : "large tau means large lattice viscosity: the solution is over-damped and accuracy degrades.");
        out.push_back(a);
    }
    if (u_high) {
        out.push_back({true,
            fmt("u_lat = %.4g (Ma = %.3g) exceeds the low-Mach guideline u_lat <= %.2g.%s", u,
                c.units.machNumber(), lim::u_lat_ideal_hi,
                u_hard ? " The solver aborts at startup (hard limit u_lat < 0.3)." : ""),
            {"Compressibility error grows as Ma^2."}});
    }
    if (bc_high) {
        out.push_back({true,
            fmt("Boundary speed in lattice units = %.4g (%.3g x U_ref x u_lat) exceeds %.2g.",
                u * c.bc_ratio, c.bc_ratio, lim::u_lat_ideal_hi),
            {"Wall / inflow velocities are larger than units.ref_velocity, so they set the Mach number."}});
    }

    const bool problem = u_nonpos || tau_low || tau_high || u_high || bc_high;
    if (!problem) { return; }

    Advisory fix{false, "Suggested fix", {}};
    const bool feasible = u_lo <= u_hi;

    if (feasible) {
        const Real u_rec = (u_pref >= u_lo && u_pref <= u_hi) ? u_pref : Real(0.5) * (u_lo + u_hi);
        const Real tau_rec = Real(0.5) + s * u_rec;
        fix.detail.push_back(fmt(
            "At the current resolution (L_ref = %.4g cells, Re = %.4g) the ideal range is "
            "u_lat in [%.4g, %.4g]  <->  tau in [%.4g, %.4g].",
            Nref, Re, u_lo, u_hi, Real(0.5) + s * u_lo, Real(0.5) + s * u_hi));
        const std::string by_u = fmt("lbm.u_lat = %.4g   (-> tau = %.4g, Ma = %.3g)",
                                     u_rec, tau_rec, u_rec / std::sqrt(c.cs2));
        const std::string by_tau = fmt("lbm.tau = %.4g   (-> u_lat = %.4g, Ma = %.3g)",
                                       tau_rec, u_rec, u_rec / std::sqrt(c.cs2));
        fix.detail.push_back(std::string("Set  ") + (c.pinned == "lbm.tau" ? by_tau : by_u));
        fix.detail.push_back(std::string("or   ") + (c.pinned == "lbm.tau" ? by_u : by_tau) +
                             "   [specify only one of lbm.tau / lbm.u_lat]");
        if (u > Real(0)) {
            fix.detail.push_back(fmt(
                "Cost: dt changes by x%.3g, i.e. x%.3g steps for the same physical time "
                "(the same step count would cover t = %.4g s instead of %.4g s).",
                u_rec / u, u / u_rec, c.n_adv * c.units.dt() * u_rec / u,
                c.n_adv * c.units.dt()));
        }
    } else {
        // No (u_lat, tau) pair satisfies both limits: the mesh is too coarse for Re.
        const Real s_min  = (lim::tau_ideal_lo - Real(0.5)) / u_cap;
        const Real f_min  = s_min / s;
        const Real f_rec  = ((kTauComfort - Real(0.5)) / u_pref) / s;

        fix.detail.push_back(fmt(
            "No (u_lat, tau) pair meets both limits at this resolution: tau >= %.2g needs "
            "u_lat >= %.4g, above the u_lat limit %.4g. The grid is too coarse for Re = %.4g "
            "(only L_ref = %.4g cells across the reference length).",
            lim::tau_ideal_lo, u_lo, u_cap, Re, Nref));

        const auto describe = [&](const char* tag, Real f, Real u_at) {
            const std::vector<int> N2 = scaledCells(c.ncells, f);
            const int nmax_old = *std::max_element(c.ncells.begin(), c.ncells.end());
            const int nmax_new = *std::max_element(N2.begin(), N2.end());
            const Real fN = Real(nmax_new) / Real(nmax_old);
            fix.detail.push_back(fmt(
                "%s: refine ~x%.3g  ->  grid.ncells = %s   (then u_lat = %.4g gives tau = %.4g)",
                tag, f, join(N2, " ", "%.0f").c_str(), u_at, Real(0.5) + s * fN * u_at));
            const double cells = std::pow(static_cast<double>(fN), c.dim);
            const double steps = u > Real(0) ? static_cast<double>(fN) * u / u_at : 0.0;
            if (steps > 0.0) {
                fix.detail.push_back(fmt(
                    "    cost: cells x%.3g, steps per physical time x%.3g, total work x%.3g",
                    cells, steps, cells * steps));
            } else {
                fix.detail.push_back(fmt("    cost: cells x%.3g", cells));
            }
        };
        describe("Minimum    ", f_min, u_cap);
        describe("Recommended", f_rec, u_pref);
        fix.detail.push_back("Refine all directions together (the lattice must stay square). "
                             "Then pin lbm.u_lat or lbm.tau as shown.");
    }
    out.push_back(fix);
}

int reportErrors(const std::string& input_file, const std::vector<std::string>& errors)
{
    amrex::Print() << "\nERROR: cannot summarise " << input_file << ":\n";
    for (const auto& e : errors) { amrex::Print() << "  - " << e << "\n"; }
    const int nc = countOf("grid", "ncells");
    if (nc != 0 && nc != AMREX_SPACEDIM) {
        amrex::Print() << "\nHint: the input has " << nc << " grid.ncells values but this tool was built for "
                       << AMREX_SPACEDIM << "-D. Use a build configured with -Dmuphfasa_DIM=" << nc << ".\n";
    }
    return 1;
}

// ── main report ─────────────────────────────────────────────────────────────
int run(const std::string& input_file)
{
    using namespace muphfasa;
    const InputParser params;
    constexpr int D = AMREX_SPACEDIM;

    // ---- hard errors: things the solver would also abort on ----
    std::vector<std::string> errors;
    std::vector<Advisory> adv;

    const char* required[][2] = {{"grid", "ncells"},        {"geometry", "prob_lo"},
                                 {"geometry", "prob_hi"},   {"sim", "max_step"},
                                 {"bc", "lo_type"},         {"bc", "hi_type"}};
    for (const auto& r : required) {
        if (!countOf(r[0], r[1])) { errors.push_back(fmt("missing required key %s.%s", r[0], r[1])); }
    }
    {
        amrex::ParmParse pu("units");
        if (!pu.contains("Re") && !pu.contains("nu_phys")) {
            errors.push_back("neither units.Re nor units.nu_phys is set (one is required)");
        }
        if (pu.contains("Re") && pu.contains("nu_phys")) {
            errors.push_back("both units.Re and units.nu_phys are set (give only one)");
        }
        amrex::ParmParse pl("lbm");
        if (pl.contains("tau") && pl.contains("u_lat")) {
            errors.push_back("both lbm.tau and lbm.u_lat are set (give only one)");
        }
    }
    if (!errors.empty()) { return reportErrors(input_file, errors); }

    for (const char* key : {"ncells"}) {
        const int c = countOf("grid", key);
        if (c < D) { errors.push_back(fmt("grid.%s lists %d value(s); a %d-D build needs %d", key, c, D, D)); }
        else if (c > D) {
            adv.push_back({true, fmt("grid.%s lists %d values but this tool is built for %d-D.", key, c, D),
                           {fmt("The solver reads only the first %d; for a %d-D input rebuild with "
                                "-Dmuphfasa_DIM=%d.", D, c, c)}});
        }
    }
    for (const char* key : {"prob_lo", "prob_hi"}) {
        const int c = countOf("geometry", key);
        if (c < D) { errors.push_back(fmt("geometry.%s lists %d value(s); a %d-D build needs %d", key, c, D, D)); }
        else if (c > D) {
            adv.push_back({true, fmt("geometry.%s lists %d values but this tool is built for %d-D.", key, c, D),
                           {fmt("The solver reads only the first %d.", D)}});
        }
    }
    for (const char* key : {"lo_type", "hi_type"}) {
        const int c = countOf("bc", key);
        if (c != D) { errors.push_back(fmt("bc.%s lists %d name(s); a %d-D build needs exactly %d", key, c, D, D)); }
    }
    {
        const int c = countOf("forcing", "accel");
        if (c != 0 && c != D) { errors.push_back(fmt("forcing.accel lists %d value(s); needs exactly %d", c, D)); }
        static const char* faces[] = {"lox", "hix", "loy", "hiy", "loz", "hiz"};
        for (const char* f : faces) {
            for (const char* suffix : {"Uwall", "Uin"}) {
                const std::string key = std::string(f) + "_" + suffix;
                const int n = countOf("bc", key.c_str());
                if (n != 0 && n != D) { errors.push_back(fmt("bc.%s lists %d value(s); needs exactly %d", key.c_str(), n, D)); }
            }
        }
    }
    if (!errors.empty()) { return reportErrors(input_file, errors); }

    // ---- read everything the solver reads ----
    const std::vector<int> ncells = params.ncells();
    const std::vector<Real> lo = params.geometryProbLo();
    const std::vector<Real> hi = params.geometryProbHi();
    const int maxStep = params.maxStep();
    const Real stopTime = params.stopTime();
    const std::string latticeName = params.latticeType();
    const std::string collision = params.collisionModel();
    const auto tauIn = params.relaxationTime();
    const auto uIn = params.latticeVelocity();
    const auto ReIn = params.reynoldsNumber();
    const auto nuIn = params.physViscosity();
    const Real Lref = params.refLength();
    const Real Uref = params.refVelocity();
    const Real rhoRef = params.refDensity();
    const std::string geomType = params.geometryType();
    const bool useEB = params.useEB();
    const auto bcNames = params.bcNames();
    const auto Uwall = params.Uwall();
    const auto Uin = params.Uinflow();
    const auto faceRho = params.faceDensity();
    const auto accel = params.forcingAccel();
    const int plotInt = params.plotInterval();
    const int chkInt = params.checkpointInterval();
    const std::string plotPrefix = params.plotfilePrefix();
    const std::string chkPrefix = params.checkpointPrefix();
    const int maxGrid = params.maxGridSize();

    // ---- the UnitConverter's own asserts (positivity, square lattice) ----
    std::vector<Real> len(D), res(D);
    for (int i = 0; i < D; ++i) {
        len[i] = hi[i] - lo[i];
        res[i] = len[i] / Real(ncells[i]);
        if (ncells[i] <= 0) { errors.push_back(fmt("grid.ncells[%d] = %d must be positive", i, ncells[i])); }
        if (!(len[i] > 0)) { errors.push_back(fmt("geometry.prob_hi[%d] must exceed prob_lo[%d]", i, i)); }
    }
    if (!(Lref > 0) || !(Uref > 0) || !(rhoRef > 0)) {
        errors.push_back("units.ref_length, ref_velocity and ref_density must be positive");
    }
    if ((ReIn && !(*ReIn > 0)) || (nuIn && !(*nuIn > 0))) {
        errors.push_back("units.Re / units.nu_phys must be positive");
    }
    if (errors.empty()) {
        for (int i = 1; i < D; ++i) {
            if (!(std::abs(res[0] - res[i]) < 1e-12)) {
                errors.push_back(fmt("lattice is not square: L/N is %.10g in x but %.10g in direction %d "
                                     "(UnitConverter requires Lx/Nx == Ly/Ny == Lz/Nz)", res[0], res[i], i));
            }
        }
    }
    if (!errors.empty()) { return reportErrors(input_file, errors); }

    // ---- lattice / collision model ----
    const LatticeInfo lat = lookupLattice(latticeName);
    if (!lat.valid) {
        adv.push_back({true, fmt("lbm.lattice_type = \"%s\" is not available in this %d-D build; the solver aborts.",
                                 latticeName.c_str(), D),
                       {D == 2 ? "A 2-D build only supports D2Q9."
                               : "A 3-D build supports D3Q15 | D3Q19 | D3Q27.",
                        "Report below assumes cs^2 = 1/3 and skips lattice-dependent estimates."}});
    }
    if (collision != "BGK") {
        adv.push_back({true, fmt("lbm.collision_model = \"%s\": only BGK is implemented; the solver aborts.",
                                 collision.c_str()), {}});
    }

    UnitConverter units(Lref, Uref, rhoRef, ncells, lo, hi, lat.cs2, ReIn, nuIn, uIn, tauIn,
                        /*inspect_only=*/true);

    const Real dx = units.dx();
    const Real dt = units.dt();
    const Real tau = units.relaxationTime();
    const Real u = units.latticeVelocity();
    const bool dt_ok = dt > Real(0) && std::isfinite(dt);
    const std::string pinned = tauIn ? "lbm.tau" : (uIn ? "lbm.u_lat" : "default");

    // ---- boundary-condition scan ----
    static const char* axis[] = {"x", "y", "z"};
    static const char* known[] = {"NoSlip", "MovingWall", "Inflow", "Outflow", "PressureOutlet", "Periodic"};
    Real bc_speed_max = 0;
    bool has_outflow = false;
    std::vector<std::string> bc_rows;
    for (int d = 0; d < D; ++d) {
        for (int b = 0; b < 2; ++b) {
            const int idx = 2 * d + b;
            const std::string& name = bcNames[idx];
            const std::string face = fmt("%s-%s", axis[d], b ? "hi" : "lo");
            std::string detail;
            if (std::find(std::begin(known), std::end(known), name) == std::end(known)) {
                adv.push_back({true, fmt("bc face %s: unknown type \"%s\"; the solver aborts.", face.c_str(), name.c_str()),
                               {"Expected NoSlip | MovingWall | Inflow | Outflow | PressureOutlet | Periodic."}});
            }
            Real speed = 0;
            const auto& vel = (name == "Inflow") ? Uin[idx] : Uwall[idx];
            for (int k = 0; k < D; ++k) { speed += vel[k] * vel[k]; }
            speed = std::sqrt(speed);
            if (name == "MovingWall" || name == "Inflow") {
                amrex::GpuArray<Real, D> vl{};
                for (int k = 0; k < D; ++k) { vl[k] = units.toLatticeVelocity(vel[k]); }
                detail = fmt("U = %s m/s  ->  lattice %s", joinArr(vel, "%.6g").c_str(), joinArr(vl, "%.5g").c_str());
                bc_speed_max = std::max(bc_speed_max, speed);
            } else if (name == "PressureOutlet") {
                detail = fmt("rho = %.6g kg/m^3  ->  lattice %.6g", faceRho[idx], units.toLatticeDensity(faceRho[idx]));
            }
            if (name == "Outflow") { has_outflow = true; }
            // velocities given for a face whose type ignores them
            auto nonzero = [&](const amrex::GpuArray<Real, D>& v) {
                for (int k = 0; k < D; ++k) { if (v[k] != Real(0)) { return true; } }
                return false;
            };
            if (name != "MovingWall" && nonzero(Uwall[idx])) {
                adv.push_back({false, fmt("bc.%s%s_Uwall is set but face %s is \"%s\"; the value is ignored.",
                                          b ? "hi" : "lo", axis[d], face.c_str(), name.c_str()), {}});
            }
            if (name != "Inflow" && nonzero(Uin[idx])) {
                adv.push_back({false, fmt("bc.%s%s_Uin is set but face %s is \"%s\"; the value is ignored.",
                                          b ? "hi" : "lo", axis[d], face.c_str(), name.c_str()), {}});
            }
            bc_rows.push_back(fmt("%-5s %-15s %s", face.c_str(), name.c_str(), detail.c_str()));
        }
        const bool lo_p = bcNames[2 * d] == "Periodic";
        const bool hi_p = bcNames[2 * d + 1] == "Periodic";
        if (lo_p != hi_p) {
            adv.push_back({true, fmt("bc in %s: one side is Periodic and the other is \"%s\".", axis[d],
                                     (lo_p ? bcNames[2 * d + 1] : bcNames[2 * d]).c_str()),
                           {"The solver takes periodicity from bc.lo_type alone (GeometryHandler.cpp); both sides must be Periodic."}});
        }
    }
    bool has_forcing = false;
    for (int k = 0; k < D; ++k) { if (accel[k] != Real(0)) { has_forcing = true; } }
    if (has_forcing && has_outflow) {
        adv.push_back({true, "forcing.accel is set together with a zero-gradient Outflow face.",
                       {"Outflow has no density anchor and drifts under body force with NoSlip walls; use PressureOutlet."}});
    }
    const Real bc_ratio = bc_speed_max / Uref;

    // ---- time stepping (mirrors the loop in src/main.cpp) ----
    // for step = start..max_step: stop if 0 < stop_time_lat < t_lat, else advance
    // (t_lat -> step+1) and write output. A fresh run starts at step 0, so max_step
    // gives max_step + 1 advances, and stop_time ends the run after the first
    // advance with t_lat > stop_time_lat. max_step / stop_time are absolute, so a
    // restart (below) only changes where the loop starts.
    const std::int64_t adv_maxstep = static_cast<std::int64_t>(maxStep) + 1;
    const Real stop_lat = units.toLatticeTime(stopTime);
    const bool has_stop = stopTime > Real(0);
    const bool stop_active = dt_ok && std::isfinite(stop_lat) && stop_lat > Real(0);
    const std::int64_t adv_stop = stop_active ? static_cast<std::int64_t>(std::floor(stop_lat)) + 1 : 0;
    const bool stop_first = stop_active && adv_stop < adv_maxstep;
    const bool limits_tie = stop_active && adv_stop == adv_maxstep;
    const std::int64_t n_adv = stop_first ? adv_stop : adv_maxstep;
    const std::int64_t last_step = std::max<std::int64_t>(n_adv - 1, 0);

    // ---- restart (amr.restart): the loop resumes at the checkpoint's step + 1 ----
    // Only the Header's version/step/time lines are read; the solver itself
    // validates the domain and lattice against the checkpoint.
    const std::string restartDir = params.restartFile();
    bool restart_found = false;
    std::int64_t first_step = 0;
    double restart_time = 0.0;
    if (!restartDir.empty()) {
        std::ifstream hdr(restartDir + "/Header");
        std::string version;
        std::int64_t chk_step = 0;
        double chk_time = 0.0;
        if ((hdr >> version >> chk_step >> chk_time) && version.rfind("muphfasa_checkpoint_", 0) == 0) {
            restart_found = true;
            first_step = chk_step + 1;
            restart_time = chk_time;
        } else {
            adv.push_back({true, fmt("amr.restart = \"%s\": no readable checkpoint Header found; the solver aborts.",
                                     restartDir.c_str()),
                           {"The path is relative to the directory the solver is run from "
                            "(and this tool: run it from the same directory)."}});
        }
    }
    const std::int64_t n_run = std::max<std::int64_t>(n_adv - first_step, 0);

    // ---- memory estimate (two f copies + rho + vel, ghost cells per box) ----
    double cells = 1.0, cells_ghost = 1.0, boxes = 1.0;
    for (int i = 0; i < D; ++i) {
        const double nb = std::ceil(double(ncells[i]) / double(std::max(maxGrid, 1)));
        cells *= ncells[i];
        boxes *= nb;
        cells_ghost *= ncells[i] + 2.0 * lat.nGhost * nb;
    }

    // ================= print =================
    amrex::Print() << "muphfasa input summary\n";
    row("input file", input_file);
    row("build", fmt("%d-D (AMREX_SPACEDIM=%d)", D, D));

    section("Simulation");
    row("lattice", lat.valid ? fmt("%s  (Q = %d, cs^2 = %.6g)", latticeName.c_str(), lat.Q, lat.cs2)
                             : fmt("%s  (not available in this build)", latticeName.c_str()));
    row("collision", collision);
    row("geometry", fmt("%s  (EB %s)", geomType.c_str(), useEB ? "on" : "off"));
    {
        std::string dom;
        for (int i = 0; i < D; ++i) { dom += fmt("%s[%.6g, %.6g]", i ? " x " : "", lo[i], hi[i]); }
        row("domain [m]", dom);
    }
    row("cells", fmt("%s = %.0f", join(ncells, " x ", "%.0f").c_str(), cells));
    row("boxes", fmt("%.0f  (grid.max_grid_size = %d)", boxes, maxGrid));
    if (lat.valid) {
        const double bytes = sizeof(Real) * cells_ghost * (2.0 * lat.Q) + sizeof(Real) * cells * (1 + D);
        row("memory (est.)", fmt("~%s  (2 x %d distributions + rho, u; %d ghost)",
                                 humanBytes(bytes).c_str(), lat.Q, lat.nGhost));
    }

    section("Physical scales");
    row("L_ref", fmt("%.6g m", units.refLength()));
    row("U_ref", fmt("%.6g m/s", units.refVelocity()));
    row("rho_ref", fmt("%.6g kg/m^3", units.refDensity()));
    row("Re", fmt("%.6g   [%s]", units.reynoldsNumber(), ReIn ? "units.Re" : "derived from nu_phys"));
    row("nu_phys", fmt("%.6g m^2/s   [%s]", units.physViscosity(), nuIn ? "units.nu_phys" : "derived from Re"));
    row("L_domain", fmt("%.6g m  (longest side, %.0f cells)", units.domainLength(), units.gridCells()));

    section("Lattice units (UnitConverter)");
    row("dt pinned by", pinned == "default"
            ? fmt("default u_lat = %.3g (neither lbm.u_lat nor lbm.tau set)", lim::u_lat_default) : pinned);
    row("C_x  (dx)", fmt("%.6g m", dx));
    row("C_t  (dt)", fmt("%.6g s", dt));
    row("L_ref", fmt("%.6g cells", units.toLatticeLength(units.refLength())));
    row("u_lat", fmt("%.6g   (Ma = %.4g)%s", u, units.machNumber(),
                     u > lim::u_lat_ideal_hi || !(u > 0) ? "   [!]" : ""));
    row("nu_lat", fmt("%.6g", units.latticeViscosity()));
    row("tau", fmt("%.6g   (omega = %.6g)%s", tau, tau != Real(0) ? Real(1) / tau : Real(0),
                   tau < lim::tau_ideal_lo || tau > lim::tau_ideal_hi ? "   [!]" : ""));
    if (has_forcing) {
        amrex::GpuArray<Real, D> al{};
        for (int k = 0; k < D; ++k) { al[k] = units.toLatticeAcceleration(accel[k]); }
        row("forcing.accel", fmt("%s m/s^2  ->  lattice %s", joinArr(accel, "%.6g").c_str(), joinArr(al, "%.5g").c_str()));
    } else {
        row("forcing.accel", "none");
    }

    section("Boundary conditions");
    for (const auto& r : bc_rows) { amrex::Print() << "  " << r << "\n"; }

    section("Time stepping (estimate)");
    row("sim.max_step", fmt("%d  ->  loop runs steps 0..%d = %lld advances", maxStep, maxStep,
                            static_cast<long long>(adv_maxstep)));
    if (stop_active) {
        row("sim.stop_time", fmt("%.6g s = %.6g lattice  ->  loop breaks after %lld advances",
                                 stopTime, stop_lat, static_cast<long long>(adv_stop)));
    } else if (has_stop) {
        row("sim.stop_time", fmt("%.6g s  (dt not positive; cannot convert)", stopTime));
    } else {
        row("sim.stop_time", fmt("%.6g  (disabled; <= 0)", stopTime));
    }
    row("limit hit first", stop_first ? "sim.stop_time" : (limits_tie ? "both (same step)" : "sim.max_step"));
    row("total", fmt("%lld steps (last step index %lld)", static_cast<long long>(n_adv),
                     static_cast<long long>(last_step)));
    if (!restartDir.empty()) {
        row("amr.restart", restart_found
                ? fmt("%s  (step %lld, lattice t = %.6g)", restartDir.c_str(),
                      static_cast<long long>(first_step - 1), restart_time)
                : fmt("%s  (no readable checkpoint found)", restartDir.c_str()));
        if (restart_found) {
            row("this run", n_run > 0
                    ? fmt("resumes at step %lld: %lld of %lld advances remain", static_cast<long long>(first_step),
                          static_cast<long long>(n_run), static_cast<long long>(n_adv))
                    : std::string("nothing to do: the checkpoint is already at or past the end of the run"));
        }
    }
    if (dt_ok) {
        row("end time", fmt("t = %.6g s  (lattice t = %lld)", n_adv * dt, static_cast<long long>(n_adv)));
        row("flow-throughs", fmt("%.4g  (end time * U_ref / L_domain)", n_adv * dt * Uref / units.domainLength()));
    } else {
        row("end time", "unavailable (dt is not positive)");
    }
    // Outputs written by this run: steps first_step..last_step that are multiples of interval.
    auto outputCount = [&](int interval) -> std::int64_t {
        if (interval <= 0 || (first_step > 0 && last_step < first_step)) { return 0; }
        return first_step > 0 ? last_step / interval - (first_step - 1) / interval : last_step / interval + 1;
    };
    row("plotfiles", plotInt > 0 ? fmt("%lld  (every %d steps, prefix \"%s\")",
                                       static_cast<long long>(outputCount(plotInt)), plotInt, plotPrefix.c_str())
                                 : "disabled (amr.plot_int <= 0)");
    row("checkpoints", chkInt > 0 ? fmt("%lld  (every %d steps, prefix \"%s\")",
                                        static_cast<long long>(outputCount(chkInt)), chkInt, chkPrefix.c_str())
                                  : "disabled (amr.chk_int <= 0)");

    // ---- disk usage ----
    // Sizes of the file contents (not rounded up to filesystem blocks). Every
    // output gets a fresh directory (step number in the name), so they accumulate.
    //   plotfile  : rho + velocity (1 + D components), valid cells, no ghosts
    //   checkpoint: all Q distributions of f, including ghost cells of every box
    // plus VisMF per-box FAB headers / min-max tables and the small Header files.
    // fab.format (AMReX) sets the on-disk precision of plotfile FABs: NATIVE =
    // sizeof(Real), NATIVE_32 and IEEE32 = 4 bytes. IEEE/ASCII/8BIT pass AMReX's
    // startup check but VisMF aborts on the first plotfile write; any other
    // string aborts at startup. writeCheckpoint() overrides the format to NATIVE
    // for its own write, so checkpoints are always sizeof(Real) per value.
    std::string fabFmt = "NATIVE";
    {
        amrex::ParmParse ppfab("fab");
        ppfab.query("format", fabFmt);
    }
    double plot_elem_bytes = sizeof(Real);
    bool plot_writable = true;
    if (fabFmt == "NATIVE_32" || fabFmt == "IEEE32") {
        plot_elem_bytes = 4;
    } else if (fabFmt == "IEEE" || fabFmt == "ASCII" || fabFmt == "8BIT") {
        plot_writable = false;
    } else if (fabFmt != "NATIVE") {
        throw std::runtime_error(fmt("fab.format = \"%s\" is not a valid AMReX format "
                                     "(use NATIVE, NATIVE_32 or IEEE32).", fabFmt.c_str()));
    }
    const double chk_elem_bytes = sizeof(Real);
    if (!plot_writable && plotInt > 0) {
        adv.push_back({true, fmt("fab.format = %s cannot be written by VisMF; the solver aborts at the first "
                                 "plotfile write.", fabFmt.c_str()),
                       {"Use NATIVE (double) or NATIVE_32 / IEEE32 (single precision), or disable "
                        "plotfiles with amr.plot_int = 0. Checkpoints are unaffected: they are always "
                        "written in NATIVE."}});
    }

    const int plot_ncomp = 1 + D;
    const double plot_bytes = plot_elem_bytes * plot_ncomp * cells
                            + boxes * (kBoxBytes + kBoxBytesPerComp * plot_ncomp) + kPlotHeaderBytes;
    const double chk_bytes = chk_elem_bytes * lat.Q * cells_ghost
                           + boxes * (kBoxBytes + kBoxBytesPerComp * lat.Q) + kChkHeaderBytes;
    const std::int64_t n_plot = outputCount(plotInt);
    const std::int64_t n_chk = outputCount(chkInt);
    section("Disk usage (estimate)");
    row("fab.format", plot_writable
            ? fmt("%s  (plotfiles: %d bytes/value; checkpoints always NATIVE: %d)", fabFmt.c_str(),
                  static_cast<int>(plot_elem_bytes), static_cast<int>(chk_elem_bytes))
            : fmt("%s  (unsupported for plotfiles; checkpoints always NATIVE)", fabFmt.c_str()));
    row("per plotfile", !plot_writable ? (plotInt > 0 ? "n/a (fab.format cannot be written)" : "n/a (plotfiles disabled)")
            : plotInt > 0
            ? fmt("~%s  (%d comps: rho, u; valid cells only)", humanBytes(plot_bytes).c_str(), plot_ncomp)
            : "n/a (plotfiles disabled)");
    if (lat.valid) {
        row("per checkpoint", chkInt > 0
                ? fmt("~%s  (%d distributions incl. %d ghost layer)", humanBytes(chk_bytes).c_str(), lat.Q, lat.nGhost)
                : "n/a (checkpoints disabled)");
    } else {
        row("per checkpoint", "unknown (lattice not available in this build)");
    }
    const double plot_total = (plotInt > 0 && plot_writable) ? static_cast<double>(n_plot) * plot_bytes : 0.0;
    const double chk_total = (chkInt > 0 && lat.valid) ? static_cast<double>(n_chk) * chk_bytes : 0.0;
    row("plot total", fmt("~%s  (%lld files)", humanBytes(plot_total).c_str(), static_cast<long long>(plotInt > 0 ? n_plot : 0)));
    row("checkpoint total", fmt("~%s  (%lld files)", humanBytes(chk_total).c_str(), static_cast<long long>(chkInt > 0 ? n_chk : 0)));
    row("total", fmt("~%s", humanBytes(plot_total + chk_total).c_str()));

    // ---- tau / u_lat range analysis ----
    checkLatticeParams({units, lat.cs2, bc_ratio, pinned, ncells, D, static_cast<Real>(n_adv)}, adv);

    // ---- ignored keys ----
    std::vector<std::string> unused;
    for (const auto& k : amrex::ParmParse::getUnusedInputs()) {
        // amrex.* is AMReX's own; test.* is read by the regression-test executables
        if (k.rfind("amrex.", 0) != 0 && k.rfind("test.", 0) != 0) { unused.push_back(k); }
    }
    if (!unused.empty()) {
        adv.push_back({false, "Input keys the solver never reads (typo, or not implemented?):", unused});
    }

    // ================= warnings & suggestions =================
    int nwarn = 0;
    for (const auto& a : adv) { nwarn += a.warning ? 1 : 0; }
    section("Warnings & suggestions");
    if (adv.empty()) {
        amrex::Print() << "  none: tau and u_lat are inside the ideal ranges "
                       << fmt("(%.2g < tau < %.2g, u_lat <= %.2g).\n", lim::tau_ideal_lo, lim::tau_ideal_hi,
                              lim::u_lat_ideal_hi);
    }
    for (const auto& a : adv) {
        amrex::Print() << "  " << (a.warning ? "WARNING: " : (a.headline == "Suggested fix" ? "" : "NOTE: "))
                       << a.headline << "\n";
        for (const auto& d : a.detail) { amrex::Print() << "      " << d << "\n"; }
    }
    amrex::Print() << "\n" << nwarn << " warning(s).\n";
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help" ||
        std::string(argv[1]).find('=') != std::string::npos) {
        std::cout << "Usage: " << argv[0] << " <inputs file> [key=value ...]\n"
                  << "Prints the grid, unit conversion, lattice parameters and estimated run length for a\n"
                  << "muphfasa input file, and warns (with suggested fixes) about out-of-range tau / u_lat.\n"
                  << "Build dimension (" << AMREX_SPACEDIM << "-D) must match the input file.\n";
        return argc < 2 ? 1 : 0;
    }
    const std::string input_file = argv[1];
    if (!std::ifstream(input_file).good()) {
        std::cerr << "ERROR: cannot open input file \"" << input_file << "\"\n";
        return 1;
    }

    // Quiet AMReX banner, and turn Abort/Assert/ParmParse failures into
    // exceptions so bad input prints a message instead of a backtrace.
    try {
        amrex::Initialize(argc, argv, std::function<void()>([] {
            amrex::ParmParse pp("amrex");
            pp.add("verbose", 0);
            pp.add("throw_exception", 1);
        }));
    } catch (const std::exception&) {
        std::cerr << "ERROR: AMReX rejected the input file during startup (see the message above;\n"
                     "       valid fab.format values are NATIVE, NATIVE_32 and IEEE32).\n";
        return 1;
    }
    int rc = 0;
    try {
        rc = run(input_file);
    } catch (const std::exception& e) {
        amrex::Print() << "\nERROR: " << e.what() << "\n";
        rc = 1;
    }
    amrex::Finalize();
    return rc;
}
