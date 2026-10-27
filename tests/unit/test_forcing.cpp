// test_forcing.cpp — Unit tests for the Guo body-force term (Collision.H) and
// the half-force velocity correction (Macroscopic.H).
// Host-only, single cell, every lattice of the active DIM.
// Verifies, for random u and F and several omega:
//   - sum_i F_i = 0, sum_i c F_i = (1-omega/2) F,
//     sum_i c_a c_b F_i = (1-omega/2)(u_a F_b + u_b F_a)
//   - the force-aware velocity of f = feq(rho,u0) is u0 + F/(2 rho)
//   - one forced BGK collision changes sum_i c f by exactly F and keeps sum_i f
//   - with F = 0 the forcing kernels are bit-identical to the plain ones

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Print.H>

#include "LBM/Collision.H"
#include "LBM/Equilibrium.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Macroscopic.H"

#include <cmath>
#include <random>

using namespace muphfasa;

namespace {

using Vec = amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;

int g_failures = 0;

void check(bool ok, const char* lattice, const char* what, double err)
{
    if (!ok) {
        ++g_failures;
        amrex::Print() << "  FAIL [" << lattice << "] " << what << " (err=" << err << ")\n";
    }
}

void checkClose(double a, double b, double tol, const char* lattice, const char* what)
{
    check(std::abs(a - b) <= tol, lattice, what, std::abs(a - b));
}

template <typename LatticeTag>
amrex::Real cComp(int n, int d)
{
    if (d == 0) { return LatticeTag::cx(n); }
    if (d == 1) { return LatticeTag::cy(n); }
#if (AMREX_SPACEDIM == 3)
    return LatticeTag::cz(n);
#else
    return 0;
#endif
}

template <typename LatticeTag>
amrex::Real feqCell(amrex::Real rho, Vec const& u, int n)
{
#if (AMREX_SPACEDIM == 2)
    return computeEquilibriumCell2D<LatticeTag>(rho, u, n);
#else
    return computeEquilibriumCell3D<LatticeTag>(rho, u, n);
#endif
}

template <typename LatticeTag>
Vec velCell(amrex::Array4<const amrex::Real> const& f, amrex::Real rho)
{
#if (AMREX_SPACEDIM == 2)
    return computeVelocityCell2D<LatticeTag>(f, rho, 0, 0, 0);
#else
    return computeVelocityCell3D<LatticeTag>(f, rho, 0, 0, 0);
#endif
}

template <typename LatticeTag>
Vec velCell(amrex::Array4<const amrex::Real> const& f, amrex::Real rho, Vec const& F)
{
#if (AMREX_SPACEDIM == 2)
    return computeVelocityCell2D<LatticeTag>(f, rho, 0, 0, 0, F);
#else
    return computeVelocityCell3D<LatticeTag>(f, rho, 0, 0, 0, F);
#endif
}

template <typename LatticeTag>
void testLattice(const char* name, std::mt19937& rng)
{
    amrex::Print() << "  " << name << "\n";
    std::uniform_real_distribution<double> uni(-0.05, 0.05);
    constexpr double tol = 1e-14;
    const double omegas[3] = {0.8, 1.0, 1.5};
    constexpr int D = AMREX_SPACEDIM;

    // ── 1. Guo moments ────────────────────────────────────────────────────
    for (double omega : omegas) {
        Vec u{}, F{};
        for (int d = 0; d < D; ++d) { u[d] = uni(rng); F[d] = uni(rng) * 1e-2; }

        double m0 = 0;
        double m1[D] = {};
        double m2[D][D] = {};
        for (int n = 0; n < LatticeTag::Q; ++n) {
            const double Fi = guoForceTerm<LatticeTag>(n, u, F, omega);
            m0 += Fi;
            for (int a = 0; a < D; ++a) {
                m1[a] += cComp<LatticeTag>(n, a) * Fi;
                for (int b = 0; b < D; ++b) {
                    m2[a][b] += cComp<LatticeTag>(n, a) * cComp<LatticeTag>(n, b) * Fi;
                }
            }
        }
        const double pref = 1.0 - 0.5 * omega;
        checkClose(m0, 0.0, tol, name, "sum F_i = 0");
        for (int a = 0; a < D; ++a) {
            checkClose(m1[a], pref * F[a], tol, name, "sum c F_i = (1-w/2) F");
            for (int b = 0; b < D; ++b) {
                checkClose(m2[a][b], pref * (u[a] * F[b] + u[b] * F[a]), tol, name,
                           "sum c c F_i = (1-w/2)(uF+Fu)");
            }
        }
    }

    // One-cell fabs for the Array4-level checks.
    const amrex::Box bx(amrex::IntVect(0), amrex::IntVect(0));
    amrex::FArrayBox f_fab(bx, LatticeTag::Q), feq_fab(bx, LatticeTag::Q), vel_fab(bx, D);

    for (double omega : omegas) {
        const double rho = 1.0 + 0.1 * uni(rng);
        Vec u0{}, F{}, uc{};
        for (int d = 0; d < D; ++d) { u0[d] = uni(rng); F[d] = uni(rng) * 1e-2; }

        auto f = f_fab.array();
        for (int n = 0; n < LatticeTag::Q; ++n) { f(0, 0, 0, n) = feqCell<LatticeTag>(rho, u0, n); }

        // ── 2. Half-force velocity ────────────────────────────────────────
        const amrex::Array4<const amrex::Real> fc = f_fab.const_array();
        const amrex::Real rho_c = computeDensityCell<LatticeTag>(fc, 0, 0, 0);
        checkClose(rho_c, rho, tol, name, "density of feq");
        auto vel = vel_fab.array();
        uc = velCell<LatticeTag>(fc, rho_c, F);
        for (int d = 0; d < D; ++d) {
            vel(0, 0, 0, d) = uc[d];
            checkClose(uc[d], u0[d] + F[d] / (2 * rho), tol, name, "u = J/rho + F/(2 rho)");
        }

        // ── 3. Single-step momentum / mass ────────────────────────────────
        auto feq = feq_fab.array();
        for (int n = 0; n < LatticeTag::Q; ++n) { feq(0, 0, 0, n) = feqCell<LatticeTag>(rho, uc, n); }
        BGK::collideArr<LatticeTag>(bx, f, feq_fab.const_array(), vel_fab.const_array(), omega, F);

        double mass = 0;
        double J[D] = {};
        for (int n = 0; n < LatticeTag::Q; ++n) {
            mass += f(0, 0, 0, n);
            for (int d = 0; d < D; ++d) { J[d] += cComp<LatticeTag>(n, d) * f(0, 0, 0, n); }
        }
        checkClose(mass, rho, tol, name, "forced collide keeps mass");
        for (int d = 0; d < D; ++d) {
            checkClose(J[d], rho * u0[d] + F[d], tol, name, "forced collide: dJ = F");
        }
    }

    // ── 4. F = 0 overloads are bit-identical to the plain kernels ─────────
    {
        const double omega = 1.3;
        const double rho = 1.02;
        Vec u0{}, zero{};
        for (int d = 0; d < D; ++d) { u0[d] = uni(rng); zero[d] = 0; }
        amrex::FArrayBox fa(bx, LatticeTag::Q), fb(bx, LatticeTag::Q);
        auto a = fa.array();
        auto b = fb.array();
        for (int n = 0; n < LatticeTag::Q; ++n) {
            a(0, 0, 0, n) = b(0, 0, 0, n) = feqCell<LatticeTag>(rho, u0, n) * (1 + 0.01 * uni(rng));
        }

        const amrex::Array4<const amrex::Real> ac = fa.const_array();
        amrex::FArrayBox v_plain(bx, D);
        auto vp = v_plain.array();
        const amrex::Real r = computeDensityCell<LatticeTag>(ac, 0, 0, 0);
        const Vec u_plain = velCell<LatticeTag>(ac, r);
        const Vec u_force = velCell<LatticeTag>(ac, r, zero);
        for (int d = 0; d < D; ++d) {
            vp(0, 0, 0, d) = u_plain[d];
            check(u_plain[d] == u_force[d], name, "F=0 velocity bit-identical", 0);
        }

        amrex::FArrayBox feq2(bx, LatticeTag::Q);
        auto fe = feq2.array();
        for (int n = 0; n < LatticeTag::Q; ++n) { fe(0, 0, 0, n) = feqCell<LatticeTag>(r, u0, n); }
        BGK::collideArr<LatticeTag>(bx, a, feq2.const_array(), omega);
        BGK::collideArr<LatticeTag>(bx, b, feq2.const_array(), v_plain.const_array(), omega, zero);
        for (int n = 0; n < LatticeTag::Q; ++n) {
            check(a(0, 0, 0, n) == b(0, 0, 0, n), name, "F=0 collide bit-identical", 0);
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Guo forcing tests ===\n";
        std::mt19937 rng(12345);
#if (AMREX_SPACEDIM == 2)
        testLattice<D2Q9>("D2Q9", rng);
#else
        testLattice<D3Q15>("D3Q15", rng);
        testLattice<D3Q19>("D3Q19", rng);
        testLattice<D3Q27>("D3Q27", rng);
#endif
        if (g_failures == 0) {
            amrex::Print() << "All forcing tests passed.\n";
        } else {
            amrex::Print() << g_failures << " forcing check(s) FAILED.\n";
        }
    }
    amrex::Finalize();
    return g_failures == 0 ? 0 : 1;
}
