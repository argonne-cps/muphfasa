// test_macroscopic.cpp — Unit tests for density and velocity moment kernels.
//
// Verifies that the muphfasa macroscopic helpers correctly recover
// hydrodynamic fields from the distribution function:
//
//   - Density:   rho = sum_q f_q
//                (computeDensityCell / computeDensity / computeMacroscopic)
//   - Velocity:  u = (1/rho) sum_q c_q f_q
//                (computeVelocityCell / computeVelocity / computeMacroscopic)
//   - Round-trip: equilibrium fEq(rho,u) -> (rho,u) recovers the inputs
//   - Box vs cell consistency and MultiFab tiling correctness
//
// Tests run for every lattice of the active DIM (D2Q9 in 2-D,
// D3Q15/D3Q19/D3Q27 in 3-D) using both hand-crafted f patterns and
// equilibrium-generated states.

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include "LBM/Equilibrium.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Macroscopic.H"

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace muphfasa;

static constexpr amrex::Real kTol = 1e-12;

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------

static void requireClose(amrex::Real got, amrex::Real expected,
                         amrex::Real tol, const std::string& what)
{
    const amrex::Real err = std::abs(got - expected);
    const amrex::Real scale = std::max(amrex::Real(1.0), std::abs(expected));
    if (err > tol * scale && err > tol) {
        throw std::runtime_error(what + ": got " + std::to_string(got) +
                                 " expected " + std::to_string(expected) +
                                 " err " + std::to_string(err));
    }
}

#if AMREX_SPACEDIM == 2
static amrex::Real equilibrium2D_proxy(
    amrex::Real rho, amrex::GpuArray<amrex::Real,2> const& u, int q, const std::string&)
{
    // Use the same inline kernel the solver uses — ensures the test tracks the
    // implementation rather than a duplicate formula.
    // Lattice type is templated at the call site; this non-template proxy is
    // unused — per-lattice wrappers below call the templated version directly.
    amrex::ignore_unused(rho, u, q);
    return 0;
}
#endif

//------------------------------------------------------------------------------
// 1) Known-f pattern — single cell
//------------------------------------------------------------------------------

template <typename LatticeTag>
static void testSingleCellKnownF(const char* lattice_name, std::mt19937& rng)
{
    std::uniform_real_distribution<amrex::Real> dist(0.5, 1.5);

#if AMREX_SPACEDIM == 2
    const amrex::Box box(amrex::IntVect{0,0}, amrex::IntVect{0,0});
#else
    const amrex::Box box(amrex::IntVect{0,0,0}, amrex::IntVect{0,0,0});
#endif
    amrex::FArrayBox f_fab(box, LatticeTag::Q);
    amrex::FArrayBox rho_fab(box, 1);
    amrex::FArrayBox vel_fab(box, AMREX_SPACEDIM);

    auto f_arr = f_fab.array();
    auto rho_arr = rho_fab.array();
    auto vel_arr = vel_fab.array();
    auto f_const = f_fab.const_array();

    // Fill f with a random but deterministic pattern.
    std::vector<amrex::Real> f_ref(LatticeTag::Q);
    for (int q = 0; q < LatticeTag::Q; ++q) {
        f_ref[q] = dist(rng);
        f_arr(0,0,0,q) = f_ref[q];
    }

    // Expected moments — hand-computed independent of the kernels under test.
    amrex::Real rho_expected = 0.0;
    for (auto v : f_ref) { rho_expected += v; }

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> u_expected{};
#if AMREX_SPACEDIM == 2
    {
        amrex::Real mx = 0.0, my = 0.0;
        for (int q = 0; q < LatticeTag::Q; ++q) {
            mx += LatticeTag::cx(q) * f_ref[q];
            my += LatticeTag::cy(q) * f_ref[q];
        }
        u_expected[0] = mx / rho_expected;
        u_expected[1] = my / rho_expected;
    }
#else
    {
        amrex::Real mx = 0.0, my = 0.0, mz = 0.0;
        for (int q = 0; q < LatticeTag::Q; ++q) {
            mx += LatticeTag::cx(q) * f_ref[q];
            my += LatticeTag::cy(q) * f_ref[q];
            mz += LatticeTag::cz(q) * f_ref[q];
        }
        u_expected[0] = mx / rho_expected;
        u_expected[1] = my / rho_expected;
        u_expected[2] = mz / rho_expected;
    }
#endif

    // --- Cell kernels ---
    const amrex::Real rho_cell = computeDensityCell<LatticeTag>(f_const, 0,0,0);
    requireClose(rho_cell, rho_expected, kTol,
                 std::string(lattice_name) + " single-cell rho");

    // Array-level density should agree.
    computeDensity<LatticeTag>(box, f_const, rho_arr);
    requireClose(rho_arr(0,0,0,0), rho_expected, kTol,
                 std::string(lattice_name) + " box density");

    // Velocity: cell-wise GpuArray variant + array helper that writes to MultiFab.
#if AMREX_SPACEDIM == 2
    {
        auto u_cell = computeVelocityCell2D<LatticeTag>(f_const, rho_cell, 0,0,0);
        requireClose(u_cell[0], u_expected[0], kTol,
                     std::string(lattice_name) + " single-cell ux");
        requireClose(u_cell[1], u_expected[1], kTol,
                     std::string(lattice_name) + " single-cell uy");
    }
    computeVelocity<LatticeTag>(box, f_const, rho_arr, vel_arr);
    requireClose(vel_arr(0,0,0,0), u_expected[0], kTol,
                 std::string(lattice_name) + " box ux");
    requireClose(vel_arr(0,0,0,1), u_expected[1], kTol,
                 std::string(lattice_name) + " box uy");
#else
    {
        auto u_cell = computeVelocityCell3D<LatticeTag>(f_const, rho_cell, 0,0,0);
        requireClose(u_cell[0], u_expected[0], kTol,
                     std::string(lattice_name) + " single-cell ux");
        requireClose(u_cell[1], u_expected[1], kTol,
                     std::string(lattice_name) + " single-cell uy");
        requireClose(u_cell[2], u_expected[2], kTol,
                     std::string(lattice_name) + " single-cell uz");
    }
    computeVelocity<LatticeTag>(box, f_const, rho_arr, vel_arr);
    requireClose(vel_arr(0,0,0,0), u_expected[0], kTol,
                 std::string(lattice_name) + " box ux");
    requireClose(vel_arr(0,0,0,1), u_expected[1], kTol,
                 std::string(lattice_name) + " box uy");
    requireClose(vel_arr(0,0,0,2), u_expected[2], kTol,
                 std::string(lattice_name) + " box uz");
#endif
}

//------------------------------------------------------------------------------
// 2) Known-f pattern — tiled Box (exercises ParallelFor tiling)
//------------------------------------------------------------------------------

template <typename LatticeTag>
static void testBoxKnownF(const char* lattice_name)
{
#if AMREX_SPACEDIM == 2
    const amrex::Box box(amrex::IntVect{0,0}, amrex::IntVect{7,7});
#else
    const amrex::Box box(amrex::IntVect{0,0,0}, amrex::IntVect{3,3,3});
#endif
    amrex::FArrayBox f_fab(box, LatticeTag::Q);
    amrex::FArrayBox rho_fab(box, 1);
    amrex::FArrayBox vel_fab(box, AMREX_SPACEDIM);

    auto f_arr = f_fab.array();

    // Deterministic spatial pattern: f(i,j,q) encodes position so we can
    // verify per-cell moments vary correctly across the box.
    {
        const auto lo = box.smallEnd();
        const auto hi = box.bigEnd();
#if AMREX_SPACEDIM == 2
        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                for (int q = 0; q < LatticeTag::Q; ++q) {
                    f_arr(i,j,0,q) = 0.7 + 0.1*q + 0.02*i - 0.015*j;
                }
            }
        }
#else
        for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    for (int q = 0; q < LatticeTag::Q; ++q) {
                        f_arr(i,j,k,q) = 0.7 + 0.1*q + 0.02*i - 0.015*j + 0.01*k;
                    }
                }
            }
        }
#endif
    }

    auto f_const = f_fab.const_array();
    auto rho_arr = rho_fab.array();
    auto vel_arr = vel_fab.array();

    // Use the array-level kernels under test.
    computeDensity<LatticeTag>(box, f_const, rho_arr);
    computeVelocity<LatticeTag>(box, f_const, rho_arr, vel_arr);

    // Verify each cell against a hand-computed reference.
    {
        const auto lo = box.smallEnd();
        const auto hi = box.bigEnd();
#if AMREX_SPACEDIM == 2
        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                amrex::Real rho_ref = 0.0;
                for (int q = 0; q < LatticeTag::Q; ++q) { rho_ref += f_const(i,j,0,q); }

                requireClose(rho_arr(i,j,0,0), rho_ref, kTol,
                             std::string(lattice_name) + " box rho");

                amrex::Real mx = 0.0, my = 0.0;
                for (int q = 0; q < LatticeTag::Q; ++q) {
                    mx += LatticeTag::cx(q) * f_const(i,j,0,q);
                    my += LatticeTag::cy(q) * f_const(i,j,0,q);
                }
                requireClose(vel_arr(i,j,0,0), mx / rho_ref, kTol,
                             std::string(lattice_name) + " box ux");
                requireClose(vel_arr(i,j,0,1), my / rho_ref, kTol,
                             std::string(lattice_name) + " box uy");
            }
        }
#else
        for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    amrex::Real rho_ref = 0.0;
                    for (int q = 0; q < LatticeTag::Q; ++q) { rho_ref += f_const(i,j,k,q); }
                    requireClose(rho_arr(i,j,k,0), rho_ref, kTol,
                                 std::string(lattice_name) + " box rho");

                    amrex::Real mx = 0.0, my = 0.0, mz = 0.0;
                    for (int q = 0; q < LatticeTag::Q; ++q) {
                        mx += LatticeTag::cx(q) * f_const(i,j,k,q);
                        my += LatticeTag::cy(q) * f_const(i,j,k,q);
                        mz += LatticeTag::cz(q) * f_const(i,j,k,q);
                    }
                    requireClose(vel_arr(i,j,k,0), mx / rho_ref, kTol,
                                 std::string(lattice_name) + " box ux");
                    requireClose(vel_arr(i,j,k,1), my / rho_ref, kTol,
                                 std::string(lattice_name) + " box uy");
                    requireClose(vel_arr(i,j,k,2), mz / rho_ref, kTol,
                                 std::string(lattice_name) + " box uz");
                }
            }
        }
#endif
    }
}

//------------------------------------------------------------------------------
// 3) Equilibrium round-trip — single cell
//------------------------------------------------------------------------------

#if AMREX_SPACEDIM == 2

template <typename LatticeTag>
static void testEquilibriumRoundTripCell(const char* lattice_name,
                                         amrex::Real rho,
                                         amrex::GpuArray<amrex::Real,2> const& u)
{
    const amrex::Box box(amrex::IntVect{0,0}, amrex::IntVect{0,0});
    amrex::FArrayBox f_fab(box, LatticeTag::Q);
    auto f_arr = f_fab.array();

    for (int q = 0; q < LatticeTag::Q; ++q) {
        f_arr(0,0,0,q) = computeEquilibriumCell2D<LatticeTag>(rho, u, q);
    }

    auto f_const = f_fab.const_array();
    const amrex::Real rho_rec = computeDensityCell<LatticeTag>(f_const, 0,0,0);
    requireClose(rho_rec, rho, kTol,
                 std::string(lattice_name) + " Eq round-trip rho");

    auto u_rec = computeVelocityCell2D<LatticeTag>(f_const, rho_rec, 0,0,0);
    requireClose(u_rec[0], u[0], kTol,
                 std::string(lattice_name) + " Eq round-trip ux");
    requireClose(u_rec[1], u[1], kTol,
                 std::string(lattice_name) + " Eq round-trip uy");
}

#else

template <typename LatticeTag>
static void testEquilibriumRoundTripCell(const char* lattice_name,
                                         amrex::Real rho,
                                         amrex::GpuArray<amrex::Real,3> const& u)
{
    const amrex::Box box(amrex::IntVect{0,0,0}, amrex::IntVect{0,0,0});
    amrex::FArrayBox f_fab(box, LatticeTag::Q);
    auto f_arr = f_fab.array();

    for (int q = 0; q < LatticeTag::Q; ++q) {
        f_arr(0,0,0,q) = computeEquilibriumCell3D<LatticeTag>(rho, u, q);
    }

    auto f_const = f_fab.const_array();
    const amrex::Real rho_rec = computeDensityCell<LatticeTag>(f_const, 0,0,0);
    requireClose(rho_rec, rho, kTol,
                 std::string(lattice_name) + " Eq round-trip rho");

    auto u_rec = computeVelocityCell3D<LatticeTag>(f_const, rho_rec, 0,0,0);
    requireClose(u_rec[0], u[0], kTol,
                 std::string(lattice_name) + " Eq round-trip ux");
    requireClose(u_rec[1], u[1], kTol,
                 std::string(lattice_name) + " Eq round-trip uy");
    requireClose(u_rec[2], u[2], kTol,
                 std::string(lattice_name) + " Eq round-trip uz");
}

#endif

//------------------------------------------------------------------------------
// 4) MultiFab tiling + equilibrium round-trip
//------------------------------------------------------------------------------

template <typename LatticeTag>
static void testMultiFabPath(const char* lattice_name, std::mt19937& rng)
{
#if AMREX_SPACEDIM == 2
    const amrex::Box domain(amrex::IntVect{0,0}, amrex::IntVect{7,7});
#else
    const amrex::Box domain(amrex::IntVect{0,0,0}, amrex::IntVect{3,3,3});
#endif
    amrex::BoxArray ba(domain);
    ba.maxSize(4); // force multiple boxes/tiles
    amrex::DistributionMapping dm(ba);

    amrex::MultiFab f_mf(ba, dm, LatticeTag::Q, 0);
    amrex::MultiFab rho_mf(ba, dm, 1, 0);
    amrex::MultiFab vel_mf(ba, dm, AMREX_SPACEDIM, 0);

    // ---- Part A: known-f MultiFab — verify computeMacroscopic ----
    {
        std::uniform_real_distribution<amrex::Real> dist(0.3, 1.2);
        for (amrex::MFIter mfi(f_mf); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto f_arr = f_mf.array(mfi);
            const auto lo = bx.smallEnd();
            const auto hi = bx.bigEnd();
#if AMREX_SPACEDIM == 2
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    for (int q = 0; q < LatticeTag::Q; ++q) {
                        f_arr(i,j,0,q) = dist(rng);
                    }
                }
            }
#else
            for (int k = lo[2]; k <= hi[2]; ++k) {
                for (int j = lo[1]; j <= hi[1]; ++j) {
                    for (int i = lo[0]; i <= hi[0]; ++i) {
                        for (int q = 0; q < LatticeTag::Q; ++q) {
                            f_arr(i,j,k,q) = dist(rng);
                        }
                    }
                }
            }
#endif
        }

        computeMacroscopic<LatticeTag>(f_mf, rho_mf, vel_mf);

        // Spot-check each cell against hand sums.
        for (amrex::MFIter mfi(f_mf); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const f_arr = f_mf.const_array(mfi);
            auto const rho_arr = rho_mf.const_array(mfi);
            auto const vel_arr = vel_mf.const_array(mfi);
            const auto lo = bx.smallEnd();
            const auto hi = bx.bigEnd();
#if AMREX_SPACEDIM == 2
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    amrex::Real rho_ref = 0.0;
                    for (int q = 0; q < LatticeTag::Q; ++q) { rho_ref += f_arr(i,j,0,q); }
                    requireClose(rho_arr(i,j,0,0), rho_ref, kTol,
                                 std::string(lattice_name) + " MF rho");

                    amrex::Real mx = 0.0, my = 0.0;
                    for (int q = 0; q < LatticeTag::Q; ++q) {
                        mx += LatticeTag::cx(q) * f_arr(i,j,0,q);
                        my += LatticeTag::cy(q) * f_arr(i,j,0,q);
                    }
                    requireClose(vel_arr(i,j,0,0), mx / rho_ref, kTol,
                                 std::string(lattice_name) + " MF ux");
                    requireClose(vel_arr(i,j,0,1), my / rho_ref, kTol,
                                 std::string(lattice_name) + " MF uy");
                }
            }
#else
            for (int k = lo[2]; k <= hi[2]; ++k) {
                for (int j = lo[1]; j <= hi[1]; ++j) {
                    for (int i = lo[0]; i <= hi[0]; ++i) {
                        amrex::Real rho_ref = 0.0;
                        for (int q = 0; q < LatticeTag::Q; ++q) { rho_ref += f_arr(i,j,k,q); }
                        requireClose(rho_arr(i,j,k,0), rho_ref, kTol,
                                     std::string(lattice_name) + " MF rho");

                        amrex::Real mx = 0.0, my = 0.0, mz = 0.0;
                        for (int q = 0; q < LatticeTag::Q; ++q) {
                            mx += LatticeTag::cx(q) * f_arr(i,j,k,q);
                            my += LatticeTag::cy(q) * f_arr(i,j,k,q);
                            mz += LatticeTag::cz(q) * f_arr(i,j,k,q);
                        }
                        requireClose(vel_arr(i,j,k,0), mx / rho_ref, kTol,
                                     std::string(lattice_name) + " MF ux");
                        requireClose(vel_arr(i,j,k,1), my / rho_ref, kTol,
                                     std::string(lattice_name) + " MF uy");
                        requireClose(vel_arr(i,j,k,2), mz / rho_ref, kTol,
                                     std::string(lattice_name) + " MF uz");
                    }
                }
            }
#endif
        }
    }

    // ---- Part B: equilibrium -> macroscopic round-trip via MultiFab ----
    {
        // Fill rho/vel with a smooth spatially varying field.
        amrex::MultiFab rho0_mf(ba, dm, 1, 0);
        amrex::MultiFab vel0_mf(ba, dm, AMREX_SPACEDIM, 0);
        amrex::MultiFab feq_mf(ba, dm, LatticeTag::Q, 0);

        for (amrex::MFIter mfi(rho0_mf); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto rho_arr = rho0_mf.array(mfi);
            auto vel_arr = vel0_mf.array(mfi);
#if AMREX_SPACEDIM == 2
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                rho_arr(i,j,k,0) = 1.0 + 0.08 * std::sin(0.5*i) * std::cos(0.4*j);
                vel_arr(i,j,k,0) = 0.03 * std::sin(0.6*j);
                vel_arr(i,j,k,1) = 0.03 * std::cos(0.6*i);
            });
#else
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                rho_arr(i,j,k,0) = 1.0 + 0.05 * (i - 0.5*j + 0.3*k);
                vel_arr(i,j,k,0) = 0.02 * std::sin(0.4*i);
                vel_arr(i,j,k,1) = 0.02 * std::cos(0.4*j);
                vel_arr(i,j,k,2) = 0.015 * std::sin(0.4*k);
            });
#endif
        }

        computeEquilibriumMF<LatticeTag>(rho0_mf, vel0_mf, feq_mf);
        computeMacroscopic<LatticeTag>(feq_mf, rho_mf, vel_mf);

        // Should recover the original fields within round-off.
        for (amrex::MFIter mfi(rho0_mf); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const rho0_arr = rho0_mf.const_array(mfi);
            auto const vel0_arr = vel0_mf.const_array(mfi);
            auto const rho_arr = rho_mf.const_array(mfi);
            auto const vel_arr = vel_mf.const_array(mfi);
            const auto lo = bx.smallEnd();
            const auto hi = bx.bigEnd();
#if AMREX_SPACEDIM == 2
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    requireClose(rho_arr(i,j,0,0), rho0_arr(i,j,0,0), kTol,
                                 std::string(lattice_name) + " Eq MF round-trip rho");
                    requireClose(vel_arr(i,j,0,0), vel0_arr(i,j,0,0), kTol,
                                 std::string(lattice_name) + " Eq MF round-trip ux");
                    requireClose(vel_arr(i,j,0,1), vel0_arr(i,j,0,1), kTol,
                                 std::string(lattice_name) + " Eq MF round-trip uy");
                }
            }
#else
            for (int k = lo[2]; k <= hi[2]; ++k) {
                for (int j = lo[1]; j <= hi[1]; ++j) {
                    for (int i = lo[0]; i <= hi[0]; ++i) {
                        requireClose(rho_arr(i,j,k,0), rho0_arr(i,j,k,0), kTol,
                                     std::string(lattice_name) + " Eq MF round-trip rho");
                        for (int d = 0; d < 3; ++d) {
                            requireClose(vel_arr(i,j,k,d), vel0_arr(i,j,k,d), kTol,
                                         std::string(lattice_name) + " Eq MF round-trip vel");
                        }
                    }
                }
            }
#endif
        }
    }
}

//------------------------------------------------------------------------------
// Lattice driver
//------------------------------------------------------------------------------

template <typename LatticeTag>
static void runForLattice(const char* name, std::mt19937& rng)
{
    amrex::Print() << "  Lattice " << name
                   << " (D=" << LatticeTag::D << " Q=" << LatticeTag::Q << ")\n";

    testSingleCellKnownF<LatticeTag>(name, rng);
    testBoxKnownF<LatticeTag>(name);

#if AMREX_SPACEDIM == 2
    const std::vector<std::pair<amrex::Real, amrex::GpuArray<amrex::Real,2>>> eq_samples{
        {1.0, {0.0, 0.0}},
        {1.2, {0.05, -0.03}},
        {0.9, {-0.04, 0.06}},
        {1.5, {0.08, 0.02}},
    };
    for (auto [rho, u] : eq_samples) {
        testEquilibriumRoundTripCell<LatticeTag>(name, rho, u);
    }
#else
    const std::vector<std::pair<amrex::Real, amrex::GpuArray<amrex::Real,3>>> eq_samples{
        {1.0, {0.0, 0.0, 0.0}},
        {1.2, {0.04, -0.02, 0.03}},
        {0.9, {-0.03, 0.05, -0.01}},
        {1.4, {0.06, 0.02, -0.04}},
    };
    for (auto [rho, u] : eq_samples) {
        testEquilibriumRoundTripCell<LatticeTag>(name, rho, u);
    }
#endif

    testMultiFabPath<LatticeTag>(name, rng);

    amrex::Print() << "    [PASS] " << name << "\n";
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Macroscopic moment tests (DIM="
                       << AMREX_SPACEDIM << ") ===\n";

        std::mt19937 rng(42);

#if AMREX_SPACEDIM == 2
        runForLattice<D2Q9>("D2Q9", rng);
#else
        runForLattice<D3Q15>("D3Q15", rng);
        runForLattice<D3Q19>("D3Q19", rng);
        runForLattice<D3Q27>("D3Q27", rng);
#endif

        amrex::Print() << "All macroscopic tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
