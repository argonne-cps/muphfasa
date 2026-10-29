// test_equilibrium.cpp — Unit tests for the equilibrium distribution.
//
// Verifies the discrete Maxwell–Boltzmann equilibrium
//   fEq_q = w_q rho [1 + (c_q·u)/cs2 + (c_q·u)^2/(2 cs4) - u^2/(2 cs2)]
// satisfies its defining moments and physical constraints:
//
//   - Zeroth moment:  sum_q fEq_q == rho                      (mass)
//   - First moment:   sum_q c_q fEq_q == rho u                (momentum)
//   - Symmetry:       fEq_q(rho,0) == w_q rho                 (rest state)
//   - Positivity:     fEq_q >= 0 for Ma << 1 (low-Mach regime)
//   - Round-trip:     fEq -> (rho,u) via macroscopic moments recovers input
//   - MultiFab path:  computeEquilibriumMF agrees with per-cell kernel
//
// Tests are run for every lattice of the active DIM (D2Q9 in 2-D,
// D3Q15/D3Q19/D3Q27 in 3-D) over a grid of (rho,u) pairs.

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include "LBM/Equilibrium.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Macroscopic.H"

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace muphfasa;

// Tolerance for moment conservation (floating-point round-off).
static constexpr amrex::Real kTol = 1e-12;

//------------------------------------------------------------------------------
// Small helper
//------------------------------------------------------------------------------

static void requireClose(amrex::Real got, amrex::Real expected,
                         amrex::Real tol, const std::string& what)
{
    const amrex::Real err = std::abs(got - expected);
    // Use relative tolerance for large expected, absolute for near-zero.
    const amrex::Real scale = std::max(amrex::Real(1.0), std::abs(expected));
    if (err > tol * scale && err > tol) {
        throw std::runtime_error(what + ": got " + std::to_string(got) +
                                 " expected " + std::to_string(expected) +
                                 " err " + std::to_string(err));
    }
}

//------------------------------------------------------------------------------
// Per-cell moment tests
//------------------------------------------------------------------------------

#if AMREX_SPACEDIM == 2

template <typename LatticeTag>
static void checkMassAndMomentum(const char* lattice_name,
                                 amrex::Real rho,
                                 amrex::GpuArray<amrex::Real, 2> const& u)
{
    // Compute all populations.
    std::vector<amrex::Real> feq(LatticeTag::Q);
    for (int q = 0; q < LatticeTag::Q; ++q) {
        feq[q] = computeEquilibriumCell2D<LatticeTag>(rho, u, q);
    }

    // Zeroth moment: sum fEq == rho
    amrex::Real mass = 0.0;
    for (auto v : feq) { mass += v; }
    requireClose(mass, rho, kTol,
                 std::string(lattice_name) + " zeroth moment (mass)");

    // First moment: sum c_q fEq == rho u
    amrex::Real mom_x = 0.0, mom_y = 0.0;
    for (int q = 0; q < LatticeTag::Q; ++q) {
        mom_x += LatticeTag::cx(q) * feq[q];
        mom_y += LatticeTag::cy(q) * feq[q];
    }
    requireClose(mom_x, rho * u[0], kTol,
                 std::string(lattice_name) + " first moment x");
    requireClose(mom_y, rho * u[1], kTol,
                 std::string(lattice_name) + " first moment y");

    // Rest state: when u==0, fEq == w_q rho exactly
    const bool at_rest = (u[0] == 0.0 && u[1] == 0.0);
    if (at_rest) {
        for (int q = 0; q < LatticeTag::Q; ++q) {
            const amrex::Real expected = LatticeTag::weights(q) * rho;
            requireClose(feq[q], expected, kTol,
                         std::string(lattice_name) + " rest state q=" + std::to_string(q));
        }
    }

    // Positivity for low Mach: u^2/(2 cs2) term dominates only at large Ma.
    // We only enforce non-negativity inside the tested low-Ma regime (|u| < 0.15).
    const amrex::Real u_mag2 = u[0]*u[0] + u[1]*u[1];
    if (u_mag2 < 0.15*0.15) {
        for (int q = 0; q < LatticeTag::Q; ++q) {
            if (feq[q] < -1e-14) {
                throw std::runtime_error(std::string(lattice_name) +
                    " negative fEq at rho=" + std::to_string(rho) +
                    " u=(" + std::to_string(u[0]) + "," + std::to_string(u[1]) + ")" +
                    " q=" + std::to_string(q) + " fEq=" + std::to_string(feq[q]));
            }
        }
    }
}

template <typename LatticeTag>
static void testRoundTripViaMacroscopic(const char* lattice_name,
                                        amrex::Real rho,
                                        amrex::GpuArray<amrex::Real, 2> const& u)
{
    // Build a 1-cell FArrayBox holding fEq, then recover (rho,u) via macroscopic kernels.
    const amrex::Box box(amrex::IntVect{0,0}, amrex::IntVect{0,0});
    amrex::FArrayBox f_fab(box, LatticeTag::Q);
    amrex::FArrayBox rho_fab(box, 1);
    amrex::FArrayBox vel_fab(box, 2);

    auto f_arr = f_fab.array();
    for (int q = 0; q < LatticeTag::Q; ++q) {
        f_arr(0,0,0,q) = computeEquilibriumCell2D<LatticeTag>(rho, u, q);
    }

    // Use the muphfasa macroscopic helpers to recompute moments.
    auto f_const = f_fab.const_array();
    amrex::Real rho_rec = computeDensityCell<LatticeTag>(f_const, 0,0,0);
    requireClose(rho_rec, rho, kTol,
                 std::string(lattice_name) + " round-trip rho");

    auto u_rec = computeVelocityCell2D<LatticeTag>(f_const, rho_rec, 0,0,0);
    requireClose(u_rec[0], u[0], kTol,
                 std::string(lattice_name) + " round-trip ux");
    requireClose(u_rec[1], u[1], kTol,
                 std::string(lattice_name) + " round-trip uy");
}

#else // AMREX_SPACEDIM == 3

template <typename LatticeTag>
static void checkMassAndMomentum(const char* lattice_name,
                                 amrex::Real rho,
                                 amrex::GpuArray<amrex::Real, 3> const& u)
{
    std::vector<amrex::Real> feq(LatticeTag::Q);
    for (int q = 0; q < LatticeTag::Q; ++q) {
        feq[q] = computeEquilibriumCell3D<LatticeTag>(rho, u, q);
    }

    amrex::Real mass = 0.0;
    for (auto v : feq) { mass += v; }
    requireClose(mass, rho, kTol,
                 std::string(lattice_name) + " zeroth moment (mass)");

    amrex::Real mom_x = 0.0, mom_y = 0.0, mom_z = 0.0;
    for (int q = 0; q < LatticeTag::Q; ++q) {
        mom_x += LatticeTag::cx(q) * feq[q];
        mom_y += LatticeTag::cy(q) * feq[q];
        mom_z += LatticeTag::cz(q) * feq[q];
    }
    requireClose(mom_x, rho * u[0], kTol,
                 std::string(lattice_name) + " first moment x");
    requireClose(mom_y, rho * u[1], kTol,
                 std::string(lattice_name) + " first moment y");
    requireClose(mom_z, rho * u[2], kTol,
                 std::string(lattice_name) + " first moment z");

    const bool at_rest = (u[0]==0.0 && u[1]==0.0 && u[2]==0.0);
    if (at_rest) {
        for (int q = 0; q < LatticeTag::Q; ++q) {
            const amrex::Real expected = LatticeTag::weights(q) * rho;
            requireClose(feq[q], expected, kTol,
                         std::string(lattice_name) + " rest state q=" + std::to_string(q));
        }
    }

    const amrex::Real u_mag2 = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    if (u_mag2 < 0.15*0.15) {
        for (int q = 0; q < LatticeTag::Q; ++q) {
            if (feq[q] < -1e-14) {
                throw std::runtime_error(std::string(lattice_name) +
                    " negative fEq at rho=" + std::to_string(rho) +
                    " u=(" + std::to_string(u[0]) + "," + std::to_string(u[1]) + "," + std::to_string(u[2]) + ")" +
                    " q=" + std::to_string(q));
            }
        }
    }
}

template <typename LatticeTag>
static void testRoundTripViaMacroscopic(const char* lattice_name,
                                        amrex::Real rho,
                                        amrex::GpuArray<amrex::Real, 3> const& u)
{
    const amrex::Box box(amrex::IntVect{0,0,0}, amrex::IntVect{0,0,0});
    amrex::FArrayBox f_fab(box, LatticeTag::Q);

    auto f_arr = f_fab.array();
    for (int q = 0; q < LatticeTag::Q; ++q) {
        f_arr(0,0,0,q) = computeEquilibriumCell3D<LatticeTag>(rho, u, q);
    }

    auto f_const = f_fab.const_array();
    amrex::Real rho_rec = computeDensityCell<LatticeTag>(f_const, 0,0,0);
    requireClose(rho_rec, rho, kTol,
                 std::string(lattice_name) + " round-trip rho");

    auto u_rec = computeVelocityCell3D<LatticeTag>(f_const, rho_rec, 0,0,0);
    requireClose(u_rec[0], u[0], kTol,
                 std::string(lattice_name) + " round-trip ux");
    requireClose(u_rec[1], u[1], kTol,
                 std::string(lattice_name) + " round-trip uy");
    requireClose(u_rec[2], u[2], kTol,
                 std::string(lattice_name) + " round-trip uz");
}

#endif

//------------------------------------------------------------------------------
// MultiFab path test — computeEquilibriumMF vs per-cell kernel
//------------------------------------------------------------------------------

#if AMREX_SPACEDIM == 2

template <typename LatticeTag>
static void testMultiFabPath(const char* lattice_name)
{
    // Small tiled domain so tiling logic is exercised.
    const amrex::Box domain(amrex::IntVect{0,0}, amrex::IntVect{7,7});
    amrex::BoxArray ba(domain);
    ba.maxSize(4);
    amrex::DistributionMapping dm(ba);

    amrex::MultiFab rho_mf(ba, dm, 1, 0);
    amrex::MultiFab vel_mf(ba, dm, 2, 0);
    amrex::MultiFab feq_mf(ba, dm, LatticeTag::Q, 0);
    amrex::MultiFab rho_rec_mf(ba, dm, 1, 0);
    amrex::MultiFab vel_rec_mf(ba, dm, 2, 0);

    // Fill rho/vel with a spatially varying pattern covering rest and sheared flow.
    for (amrex::MFIter mfi(rho_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto rho_arr = rho_mf.array(mfi);
        auto vel_arr = vel_mf.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            rho_arr(i,j,k,0) = 1.0 + 0.1 * std::sin(0.7*i) * std::cos(0.5*j);
            vel_arr(i,j,k,0) = 0.04 * std::sin(0.3*j);
            vel_arr(i,j,k,1) = 0.04 * std::cos(0.3*i);
        });
    }

    // Use the muphfasa MultiFab helper.
    computeEquilibriumMF<LatticeTag>(rho_mf, vel_mf, feq_mf);

    // Verify per-cell moments via macroscopic helpers recover inputs,
    // and that each feq matches the per-cell equilibrium kernel.
    for (amrex::MFIter mfi(feq_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const rho_arr = rho_mf.const_array(mfi);
        auto const vel_arr = vel_mf.const_array(mfi);
        auto const feq_arr = feq_mf.const_array(mfi);

        const auto lo = bx.smallEnd();
        const auto hi = bx.bigEnd();
        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                const amrex::Real rho = rho_arr(i,j,0,0);
                amrex::GpuArray<amrex::Real,2> u{vel_arr(i,j,0,0), vel_arr(i,j,0,1)};

                // Check feq values match the inline kernel.
                for (int q = 0; q < LatticeTag::Q; ++q) {
                    const amrex::Real expected = computeEquilibriumCell2D<LatticeTag>(rho, u, q);
                    const amrex::Real got = feq_arr(i,j,0,q);
                    requireClose(got, expected, kTol,
                        std::string(lattice_name) + " MultiFab feq vs kernel");
                }
            }
        }
    }

    // Round-trip through macroscopic moments should recover rho/vel.
    computeMacroscopic<LatticeTag>(feq_mf, rho_rec_mf, vel_rec_mf);

    for (amrex::MFIter mfi(rho_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const rho_arr = rho_mf.const_array(mfi);
        auto const vel_arr = vel_mf.const_array(mfi);
        auto const rho_rec_arr = rho_rec_mf.const_array(mfi);
        auto const vel_rec_arr = vel_rec_mf.const_array(mfi);

        const auto lo = bx.smallEnd();
        const auto hi = bx.bigEnd();
        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                requireClose(rho_rec_arr(i,j,0,0), rho_arr(i,j,0,0), kTol,
                             std::string(lattice_name) + " MultiFab round-trip rho");
                requireClose(vel_rec_arr(i,j,0,0), vel_arr(i,j,0,0), kTol,
                             std::string(lattice_name) + " MultiFab round-trip ux");
                requireClose(vel_rec_arr(i,j,0,1), vel_arr(i,j,0,1), kTol,
                             std::string(lattice_name) + " MultiFab round-trip uy");
            }
        }
    }
}

#else // 3-D

template <typename LatticeTag>
static void testMultiFabPath(const char* lattice_name)
{
    const amrex::Box domain(amrex::IntVect{0,0,0}, amrex::IntVect{3,3,3});
    amrex::BoxArray ba(domain);
    ba.maxSize(2);
    amrex::DistributionMapping dm(ba);

    amrex::MultiFab rho_mf(ba, dm, 1, 0);
    amrex::MultiFab vel_mf(ba, dm, 3, 0);
    amrex::MultiFab feq_mf(ba, dm, LatticeTag::Q, 0);
    amrex::MultiFab rho_rec_mf(ba, dm, 1, 0);
    amrex::MultiFab vel_rec_mf(ba, dm, 3, 0);

    for (amrex::MFIter mfi(rho_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto rho_arr = rho_mf.array(mfi);
        auto vel_arr = vel_mf.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            rho_arr(i,j,k,0) = 1.0 + 0.05 * (i - j + k);
            vel_arr(i,j,k,0) = 0.03 * std::sin(0.5*i);
            vel_arr(i,j,k,1) = 0.03 * std::cos(0.5*j);
            vel_arr(i,j,k,2) = 0.02 * std::sin(0.5*k);
        });
    }

    computeEquilibriumMF<LatticeTag>(rho_mf, vel_mf, feq_mf);

    for (amrex::MFIter mfi(feq_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const rho_arr = rho_mf.const_array(mfi);
        auto const vel_arr = vel_mf.const_array(mfi);
        auto const feq_arr = feq_mf.const_array(mfi);
        const auto lo = bx.smallEnd();
        const auto hi = bx.bigEnd();
        for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    const amrex::Real rho = rho_arr(i,j,k,0);
                    amrex::GpuArray<amrex::Real,3> u{
                        vel_arr(i,j,k,0), vel_arr(i,j,k,1), vel_arr(i,j,k,2)};
                    for (int q = 0; q < LatticeTag::Q; ++q) {
                        const amrex::Real expected = computeEquilibriumCell3D<LatticeTag>(rho, u, q);
                        const amrex::Real got = feq_arr(i,j,k,q);
                        requireClose(got, expected, kTol,
                            std::string(lattice_name) + " MultiFab feq vs kernel");
                    }
                }
            }
        }
    }

    computeMacroscopic<LatticeTag>(feq_mf, rho_rec_mf, vel_rec_mf);

    for (amrex::MFIter mfi(rho_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const rho_arr = rho_mf.const_array(mfi);
        auto const vel_arr = vel_mf.const_array(mfi);
        auto const rho_rec_arr = rho_rec_mf.const_array(mfi);
        auto const vel_rec_arr = vel_rec_mf.const_array(mfi);
        const auto lo = bx.smallEnd();
        const auto hi = bx.bigEnd();
        for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    requireClose(rho_rec_arr(i,j,k,0), rho_arr(i,j,k,0), kTol,
                                 std::string(lattice_name) + " MultiFab round-trip rho");
                    for (int d = 0; d < 3; ++d) {
                        requireClose(vel_rec_arr(i,j,k,d), vel_arr(i,j,k,d), kTol,
                                     std::string(lattice_name) + " MultiFab round-trip vel");
                    }
                }
            }
        }
    }
}

#endif

//------------------------------------------------------------------------------
// Lattice driver
//------------------------------------------------------------------------------

template <typename LatticeTag>
static void runForLattice(const char* name)
{
    amrex::Print() << "  Lattice " << name
                   << " (D=" << LatticeTag::D << " Q=" << LatticeTag::Q << ")\n";

    // Deterministic samples: rest, axial, diagonal, and varied density.
    const std::vector<amrex::Real> rho_samples{0.5, 1.0, 1.3, 2.0};

#if AMREX_SPACEDIM == 2
    const std::vector<amrex::GpuArray<amrex::Real,2>> vel_samples{
        {0.0, 0.0},
        {0.05, 0.0},
        {0.0, -0.07},
        {0.05, -0.02},
        {0.10, 0.05},
        {-0.08, 0.06},
        {0.02, 0.02}
    };

    for (auto rho : rho_samples) {
        for (auto u : vel_samples) {
            checkMassAndMomentum<LatticeTag>(name, rho, u);
            testRoundTripViaMacroscopic<LatticeTag>(name, rho, u);
        }
    }

    // Random low-Mach samples to catch accidental symmetry errors.
    {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<amrex::Real> rho_dist(0.8, 1.2);
        std::uniform_real_distribution<amrex::Real> vel_dist(-0.08, 0.08);
        for (int s = 0; s < 20; ++s) {
            amrex::Real rho = rho_dist(rng);
            amrex::GpuArray<amrex::Real,2> u{vel_dist(rng), vel_dist(rng)};
            checkMassAndMomentum<LatticeTag>(name, rho, u);
        }
    }
#else
    const std::vector<amrex::GpuArray<amrex::Real,3>> vel_samples{
        {0.0, 0.0, 0.0},
        {0.05, 0.0, 0.0},
        {0.0, -0.06, 0.0},
        {0.0, 0.0, 0.04},
        {0.04, -0.02, 0.03},
        {0.08, 0.05, -0.03},
        {-0.05, 0.04, 0.06}
    };

    for (auto rho : rho_samples) {
        for (auto u : vel_samples) {
            checkMassAndMomentum<LatticeTag>(name, rho, u);
            testRoundTripViaMacroscopic<LatticeTag>(name, rho, u);
        }
    }

    {
        std::mt19937 rng(12345);
        std::uniform_real_distribution<amrex::Real> rho_dist(0.8, 1.2);
        std::uniform_real_distribution<amrex::Real> vel_dist(-0.06, 0.06);
        for (int s = 0; s < 20; ++s) {
            amrex::Real rho = rho_dist(rng);
            amrex::GpuArray<amrex::Real,3> u{vel_dist(rng), vel_dist(rng), vel_dist(rng)};
            checkMassAndMomentum<LatticeTag>(name, rho, u);
        }
    }
#endif

    // MultiFab helper should agree with per-cell kernel and round-trip.
    testMultiFabPath<LatticeTag>(name);

    amrex::Print() << "    [PASS] " << name << "\n";
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Equilibrium distribution tests (DIM="
                       << AMREX_SPACEDIM << ") ===\n";

#if AMREX_SPACEDIM == 2
        runForLattice<D2Q9>("D2Q9");
#else
        runForLattice<D3Q15>("D3Q15");
        runForLattice<D3Q19>("D3Q19");
        runForLattice<D3Q27>("D3Q27");
#endif

        amrex::Print() << "All equilibrium tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
