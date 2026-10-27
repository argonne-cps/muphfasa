// test_bc_corners.cpp — BCManager corner, edge, and priority tests.
//
// Covers three concerns:
//
//  1. Coverage     — every pull stencil has a finite source after FillBoundary + BC
//                   (testCoverageSingleBox, testCoveragePeriodicX, testCoverageTwoBoxes).
//  2. Source cell  — ghost populations come from the correct neighbour with the right
//                   value (NoSlip bounce-back and MovingWall correction).
//  3. Resolution   — edges/corners where multiple walls meet pick the highest-
//                   priority BC and reject incompatible combinations (testPriorityTable);
//                   plus the PressureOutlet anti-bounce-back kernel.
//
// Host-only, tiny domains, runs for every lattice of the active DIM.

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Loop.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include "LBM/BoundaryConditions.H"
#include "LBM/Equilibrium.H"
#include "LBM/LatticeDescriptor.H"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

using namespace muphfasa;

//==============================================================================
// Lattice helpers
//==============================================================================

// Lattice velocity as IntVect — mirrors the old BCTypes/BCCommon.H helper.
template <typename LatticeTag>
amrex::IntVect latticeVelocity(int n)
{
#if (AMREX_SPACEDIM == 2)
    return amrex::IntVect{AMREX_D_DECL(LatticeTag::cx(n), LatticeTag::cy(n), 0)};
#else
    return amrex::IntVect{AMREX_D_DECL(LatticeTag::cx(n), LatticeTag::cy(n), LatticeTag::cz(n))};
#endif
}

//==============================================================================
// Geometry helpers
//==============================================================================

static amrex::Geometry makeGeometry(const amrex::Box& domain,
                                    const amrex::IntVect& is_periodic_iv)
{
    amrex::RealBox real_box(AMREX_D_DECL(0., 0., 0.), AMREX_D_DECL(1., 1., 1.));

    amrex::Array<int, AMREX_SPACEDIM> is_periodic{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        is_periodic[d] = is_periodic_iv[d];
    }

    amrex::Geometry geom;
    geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);
    return geom;
}

static amrex::BCRec makeBCRecForTest(const amrex::IntVect& is_periodic)
{
    amrex::BCRec bcr;

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const int bc_type = is_periodic[d] ? amrex::BCType::int_dir : kCustomBCSentinel;
        bcr.setLo(d, bc_type);
        bcr.setHi(d, bc_type);
    }

    return bcr;
}

// Build a BCManager with NoSlip on every non-periodic face.
template <typename LatticeTag>
static BCManager<LatticeTag> makeNoSlipBCManager()
{
    BCManager<LatticeTag> bcm;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int side : {-1, 1}) {
            bcm.setFaceBC(dir, side, lbmBCtype::NoSlip);
            bcm.setFaceDensity(dir, side, 1.0);

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> zero_vel{AMREX_D_DECL(0., 0., 0.)};
            bcm.setFaceVel(dir, side, zero_vel);
        }
    }

    bcm.finalize();
    return bcm;
}

//==============================================================================
// MultiFab helpers
//==============================================================================

template <typename LatticeTag>
static void fillWithNaN(amrex::MultiFab& mf)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr = mf.array(mfi);
        const amrex::Box& fab_box = mfi.fabbox();

        amrex::ParallelFor(fab_box, LatticeTag::Q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                arr(i, j, k, n) = std::numeric_limits<amrex::Real>::quiet_NaN();
            });
    }
}

template <typename LatticeTag>
static void fillValidWithRandom(amrex::MultiFab& mf, std::mt19937& rng)
{
    std::uniform_real_distribution<double> uniform(0.5, 1.5);

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr = mf.array(mfi);
        const amrex::Box& valid_box = mfi.validbox();
        const auto lo = valid_box.smallEnd();
        const auto hi = valid_box.bigEnd();

#if AMREX_SPACEDIM == 2
        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                for (int n = 0; n < LatticeTag::Q; ++n) {
                    arr(i, j, 0, n) = static_cast<amrex::Real>(uniform(rng));
                }
            }
        }
#else
        for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    for (int n = 0; n < LatticeTag::Q; ++n) {
                        arr(i, j, k, n) = static_cast<amrex::Real>(uniform(rng));
                    }
                }
            }
        }
#endif
    }
}

// Fill ghost cells with a deterministic pattern: f(i,j,n) = i*1000 + j*10 + n*0.1 + 0.5.
// Ghost entries are left as NaN so we can verify the BC actually writes them.
template <typename LatticeTag>
static void fillDeterministicPattern(amrex::MultiFab& mf)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr = mf.array(mfi);
        const amrex::Box& valid_box = mfi.validbox();
        const amrex::Box& fab_box = mfi.fabbox();

        // Valid cells: deterministic values.
        {
            const auto lo = valid_box.smallEnd();
            const auto hi = valid_box.bigEnd();
#if AMREX_SPACEDIM == 2
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    for (int n = 0; n < LatticeTag::Q; ++n) {
                        arr(i, j, 0, n) = i * 1000 + j * 10 + n * 0.1 + 0.5;
                    }
                }
            }
#else
            const auto lo3 = valid_box.smallEnd();
            const auto hi3 = valid_box.bigEnd();
            for (int k = lo3[2]; k <= hi3[2]; ++k) {
                for (int j = lo3[1]; j <= hi3[1]; ++j) {
                    for (int i = lo3[0]; i <= hi3[0]; ++i) {
                        for (int n = 0; n < LatticeTag::Q; ++n) {
                            arr(i, j, k, n) = i * 1000 + j * 10 + k * 0.1 + n * 0.01 + 0.5;
                        }
                    }
                }
            }
#endif
        }

        // Ghost cells: NaN sentinel.
        {
            const auto lo = fab_box.smallEnd();
            const auto hi = fab_box.bigEnd();
#if AMREX_SPACEDIM == 2
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    amrex::IntVect iv{AMREX_D_DECL(i, j, 0)};
                    if (valid_box.contains(iv)) {
                        continue;
                    }
                    for (int n = 0; n < LatticeTag::Q; ++n) {
                        arr(i, j, 0, n) = std::numeric_limits<amrex::Real>::quiet_NaN();
                    }
                }
            }
#else
            for (int k = lo[2]; k <= hi[2]; ++k) {
                for (int j = lo[1]; j <= hi[1]; ++j) {
                    for (int i = lo[0]; i <= hi[0]; ++i) {
                        amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
                        if (valid_box.contains(iv)) {
                            continue;
                        }
                        for (int n = 0; n < LatticeTag::Q; ++n) {
                            arr(i, j, k, n) = std::numeric_limits<amrex::Real>::quiet_NaN();
                        }
                    }
                }
            }
#endif
        }
    }
}

//==============================================================================
// BC application
//==============================================================================

// True if iv is outside the domain on a non-periodic side.
static bool isWallGhost(const amrex::IntVect& iv,
                        const amrex::Box& domain,
                        const amrex::BCRec& bcr)
{
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (bcr.lo(d) == amrex::BCType::int_dir) {
            continue; // periodic direction — not a wall
        }
        if (iv[d] < domain.smallEnd(d) || iv[d] > domain.bigEnd(d)) {
            return true;
        }
    }
    return false;
}

template <typename LatticeTag>
static void applyBCManager(amrex::MultiFab& mf,
                           const amrex::Geometry& geom,
                           const BCManager<LatticeTag>& bcm,
                           const amrex::BCRec& bcr)
{
    const amrex::Box& domain = geom.Domain();
    const amrex::BCRec* bcr_ptr = &bcr;

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr = mf.array(mfi);
        const amrex::Box& fab_box = mfi.fabbox();
        const auto lo = fab_box.smallEnd();
        const auto hi = fab_box.bigEnd();

#if AMREX_SPACEDIM == 2
        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                amrex::IntVect iv{AMREX_D_DECL(i, j, 0)};
                if (domain.contains(iv)) {
                    continue;
                }
                if (!isWallGhost(iv, domain, bcr)) {
                    continue;
                }
                bcm(iv, arr, 0, LatticeTag::Q, geom.data(), 0.0, bcr_ptr, 0, 0);
            }
        }
#else
        for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    amrex::IntVect iv{AMREX_D_DECL(i, j, k)};
                    if (domain.contains(iv)) {
                        continue;
                    }
                    if (!isWallGhost(iv, domain, bcr)) {
                        continue;
                    }
                    bcm(iv, arr, 0, LatticeTag::Q, geom.data(), 0.0, bcr_ptr, 0, 0);
                }
            }
        }
#endif
    }
}

//==============================================================================
// Verification helpers
//==============================================================================

// Check that every pull source y - c_n is finite — i.e. ghost cells were
// filled for all directions that stream into the domain (faces, edges, corners).
template <typename LatticeTag>
static void checkCoverage(const amrex::MultiFab& mf, const amrex::Geometry& geom)
{
    const amrex::Box& domain = geom.Domain();
    amrex::ignore_unused(domain);

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const arr = mf.const_array(mfi);
        const amrex::Box& valid_box = mfi.validbox();
        const auto lo = valid_box.smallEnd();
        const auto hi = valid_box.bigEnd();

#if AMREX_SPACEDIM == 2
        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                amrex::IntVect cell{AMREX_D_DECL(i, j, 0)};
                for (int n = 0; n < LatticeTag::Q; ++n) {
                    amrex::IntVect src = cell - latticeVelocity<LatticeTag>(n);
                    amrex::Real value = arr(src[0], src[1], 0, n);
                    if (!std::isfinite(value)) {
                        throw std::runtime_error(
                            "coverage: non-finite source for cell (" +
                            std::to_string(i) + "," + std::to_string(j) +
                            ") population " + std::to_string(n));
                    }
                }
            }
        }
#else
        for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    amrex::IntVect cell{AMREX_D_DECL(i, j, k)};
                    for (int n = 0; n < LatticeTag::Q; ++n) {
                        amrex::IntVect src = cell - latticeVelocity<LatticeTag>(n);
                        amrex::Real value = arr(src[0], src[1], src[2], n);
                        if (!std::isfinite(value)) {
                            throw std::runtime_error("coverage: non-finite source");
                        }
                    }
                }
            }
        }
#endif
    }
}

//==============================================================================
// Test cases — coverage
//==============================================================================

template <typename LatticeTag>
static void testCoverageSingleBox()
{
    amrex::Print() << "    coverage: single non-periodic box...\n";

#if AMREX_SPACEDIM == 2
    amrex::Box domain(amrex::IntVect{0, 0}, amrex::IntVect{3, 3});
#else
    amrex::Box domain(amrex::IntVect{0, 0, 0}, amrex::IntVect{3, 3, 3});
#endif
    amrex::IntVect periodic{AMREX_D_DECL(0, 0, 0)};
    amrex::Geometry geom = makeGeometry(domain, periodic);

    amrex::BoxArray ba(domain);
    ba.maxSize(64); // single box

    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);

    fillWithNaN<LatticeTag>(mf);

    std::mt19937 rng(42);
    fillValidWithRandom<LatticeTag>(mf, rng);

    BCManager<LatticeTag> bcm = makeNoSlipBCManager<LatticeTag>();
    amrex::BCRec bcr = makeBCRecForTest(periodic);

    applyBCManager<LatticeTag>(mf, geom, bcm, bcr);
    checkCoverage<LatticeTag>(mf, geom);

    amrex::Print() << "      [PASS]\n";
}

template <typename LatticeTag>
static void testCoveragePeriodicX()
{
    amrex::Print() << "    coverage: x-periodic + FillBoundary...\n";

#if AMREX_SPACEDIM == 2
    amrex::Box domain(amrex::IntVect{0, 0}, amrex::IntVect{3, 3});
#else
    amrex::Box domain(amrex::IntVect{0, 0, 0}, amrex::IntVect{3, 3, 3});
#endif
    amrex::IntVect periodic{AMREX_D_DECL(1, 0, 0)}; // periodic in x only
    amrex::Geometry geom = makeGeometry(domain, periodic);

    amrex::BoxArray ba(domain);
    ba.maxSize(64);

    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);

    fillWithNaN<LatticeTag>(mf);

    std::mt19937 rng(43);
    fillValidWithRandom<LatticeTag>(mf, rng);

    // Fill periodic ghosts first; BCManager handles the remaining walls.
    mf.FillBoundary(geom.periodicity());

    BCManager<LatticeTag> bcm;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (periodic[dir]) {
            continue; // periodic — no BC to set
        }
        for (int side : {-1, 1}) {
            bcm.setFaceBC(dir, side, lbmBCtype::NoSlip);
            bcm.setFaceDensity(dir, side, 1.0);
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> zero{AMREX_D_DECL(0., 0., 0.)};
            bcm.setFaceVel(dir, side, zero);
        }
    }
    bcm.finalize();

    amrex::BCRec bcr = makeBCRecForTest(periodic);

    applyBCManager<LatticeTag>(mf, geom, bcm, bcr);
    checkCoverage<LatticeTag>(mf, geom);

    amrex::Print() << "      [PASS]\n";
}

template <typename LatticeTag>
static void testCoverageTwoBoxes()
{
    amrex::Print() << "    coverage: two boxes + FillBoundary...\n";

#if AMREX_SPACEDIM == 2
    amrex::Box domain(amrex::IntVect{0, 0}, amrex::IntVect{7, 3});
#else
    amrex::Box domain(amrex::IntVect{0, 0, 0}, amrex::IntVect{7, 3, 3});
#endif
    amrex::IntVect periodic{AMREX_D_DECL(0, 0, 0)};
    amrex::Geometry geom = makeGeometry(domain, periodic);

    amrex::BoxArray ba(domain);
    ba.maxSize(4); // force two boxes

    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);

    fillWithNaN<LatticeTag>(mf);

    std::mt19937 rng(44);
    fillValidWithRandom<LatticeTag>(mf, rng);

    // Interior box boundaries need FillBoundary before BC.
    mf.FillBoundary(geom.periodicity());

    BCManager<LatticeTag> bcm = makeNoSlipBCManager<LatticeTag>();
    amrex::BCRec bcr = makeBCRecForTest(periodic);

    applyBCManager<LatticeTag>(mf, geom, bcm, bcr);

    // Inter-box ghosts also need filling after BC for full coverage.
    mf.FillBoundary(geom.periodicity());

    checkCoverage<LatticeTag>(mf, geom);

    amrex::Print() << "      [PASS]\n";
}

#if AMREX_SPACEDIM == 2

template <typename LatticeTag>
static void testSourceCell2D()
{
    amrex::Box domain(amrex::IntVect{0, 0}, amrex::IntVect{5, 5});
    amrex::IntVect periodic{AMREX_D_DECL(0, 0, 0)};
    amrex::Geometry geom = makeGeometry(domain, periodic);

    amrex::BoxArray ba(domain);
    ba.maxSize(64);
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);

    // ------------------------------------------------------------------
    // NoSlip: ghost should be exact bounce-back from source cell.
    // ------------------------------------------------------------------
    fillDeterministicPattern<LatticeTag>(mf);

    BCManager<LatticeTag> no_slip_bcm = makeNoSlipBCManager<LatticeTag>();
    amrex::BCRec bcr = makeBCRecForTest(periodic);
    applyBCManager<LatticeTag>(mf, geom, no_slip_bcm, bcr);

    // Verify every ghost population that streams into the domain matches its source.
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const arr = mf.const_array(mfi);
        const amrex::Box& fab_box = mfi.fabbox();

        const auto lo = fab_box.smallEnd();
        const auto hi = fab_box.bigEnd();

        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int i = lo[0]; i <= hi[0]; ++i) {
                amrex::IntVect ghost{AMREX_D_DECL(i, j, 0)};
                if (domain.contains(ghost)) {
                    continue;
                }

                for (int n = 0; n < LatticeTag::Q; ++n) {
                    amrex::IntVect c = latticeVelocity<LatticeTag>(n);
                    if (!domain.contains(ghost + c)) {
                        continue; // not an incoming population at this ghost
                    }

                    const int opposite = LatticeTag::opposite(n);
                    amrex::IntVect src = ghost + c;

                    amrex::Real expected = arr(src[0], src[1], 0, opposite);
                    amrex::Real got = arr(ghost[0], ghost[1], 0, n);

                    if (std::abs(got - expected) > 1e-12) {
                        throw std::runtime_error("source-cell mismatch (NoSlip)");
                    }
                }
            }
        }
    }

    // Extra sanity: the south-west corner ghost (-1,-1) should have exactly
    // one incoming population — the diagonal (1,1). Catches wrong corner logic.
    {
        bool found_corner_population = false;
        for (int n = 0; n < LatticeTag::Q; ++n) {
            amrex::IntVect c = latticeVelocity<LatticeTag>(n);
            amrex::IntVect ghost{AMREX_D_DECL(-1, -1, 0)};
            if (domain.contains(ghost + c)) {
                if (c[0] != 1 || c[1] != 1) {
                    throw std::runtime_error("corner ghost has wrong incoming direction");
                }
                found_corner_population = true;
            }
        }
        if (!found_corner_population) {
            throw std::runtime_error("corner ghost has no incoming population");
        }
    }

    // ------------------------------------------------------------------
    // MovingWall: top wall (y-hi) moves at U = (0.1, 0), rest are NoSlip.
    // Ghost f = f(src, opp) - 2 * rho * w_opp * (c_opp · U) / cs2.
    // ------------------------------------------------------------------
    fillDeterministicPattern<LatticeTag>(mf);

    BCManager<LatticeTag> moving_wall_bcm;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int side : {-1, 1}) {
            if (dir == 1 && side == 1) {
                // Top wall: MovingWall at 0.1 in x.
                moving_wall_bcm.setFaceBC(dir, side, lbmBCtype::MovingWall);
                moving_wall_bcm.setFaceDensity(dir, side, 1.0);
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> wall_vel{AMREX_D_DECL(0.1, 0., 0.)};
                moving_wall_bcm.setFaceVel(dir, side, wall_vel);
            } else {
                moving_wall_bcm.setFaceBC(dir, side, lbmBCtype::NoSlip);
                moving_wall_bcm.setFaceDensity(dir, side, 1.0);
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> zero{AMREX_D_DECL(0., 0., 0.)};
                moving_wall_bcm.setFaceVel(dir, side, zero);
            }
        }
    }
    moving_wall_bcm.finalize();
    applyBCManager<LatticeTag>(mf, geom, moving_wall_bcm, bcr);

    // Check a ghost on the moving top wall: (2,6).
    {
        amrex::IntVect ghost{AMREX_D_DECL(2, 6, 0)};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> wall_vel{AMREX_D_DECL(0.1, 0., 0.)};

        for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
            auto const arr = mf.const_array(mfi);

            for (int n = 0; n < LatticeTag::Q; ++n) {
                amrex::IntVect c = latticeVelocity<LatticeTag>(n);
                if (!domain.contains(ghost + c)) {
                    continue;
                }

                const int opposite = LatticeTag::opposite(n);
                amrex::IntVect src = ghost + c;

                // Expected bounce-back value plus MovingWall correction.
                amrex::Real f_src = src[0] * 1000 + src[1] * 10 + opposite * 0.1 + 0.5;
                amrex::Real c_dot_u =
                    LatticeTag::cx(opposite) * wall_vel[0] +
                    LatticeTag::cy(opposite) * wall_vel[1];
                amrex::Real correction =
                    -2.0 * 1.0 * LatticeTag::weights(opposite) * c_dot_u * LatticeTag::cs2inv;

                amrex::Real expected = f_src + correction;
                amrex::Real got = arr(ghost[0], ghost[1], 0, n);

                if (std::abs(got - expected) > 1e-12) {
                    throw std::runtime_error("MovingWall correction mismatch");
                }
            }
        }
    }

    amrex::Print() << "      [PASS]\n";
}

#else // AMREX_SPACEDIM == 3

template <typename LatticeTag>
static void testSourceCell3D()
{
    amrex::Box domain(amrex::IntVect{0, 0, 0}, amrex::IntVect{3, 3, 3});
    amrex::IntVect periodic{AMREX_D_DECL(0, 0, 0)};
    amrex::Geometry geom = makeGeometry(domain, periodic);

    amrex::BoxArray ba(domain);
    ba.maxSize(64);
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);

    fillDeterministicPattern<LatticeTag>(mf);

    BCManager<LatticeTag> bcm = makeNoSlipBCManager<LatticeTag>();
    amrex::BCRec bcr = makeBCRecForTest(periodic);
    applyBCManager<LatticeTag>(mf, geom, bcm, bcr);

    // Verify bounce-back for every ghost.
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const arr = mf.const_array(mfi);
        const amrex::Box& fab_box = mfi.fabbox();
        const auto lo = fab_box.smallEnd();
        const auto hi = fab_box.bigEnd();

        for (int k = lo[2]; k <= hi[2]; ++k) {
            for (int j = lo[1]; j <= hi[1]; ++j) {
                for (int i = lo[0]; i <= hi[0]; ++i) {
                    amrex::IntVect ghost{AMREX_D_DECL(i, j, k)};
                    if (domain.contains(ghost)) {
                        continue;
                    }

                    for (int n = 0; n < LatticeTag::Q; ++n) {
                        amrex::IntVect c = latticeVelocity<LatticeTag>(n);
                        if (!domain.contains(ghost + c)) {
                            continue;
                        }

                        const int opposite = LatticeTag::opposite(n);
                        amrex::IntVect src = ghost + c;

                        amrex::Real expected = arr(src[0], src[1], src[2], opposite);
                        amrex::Real got = arr(ghost[0], ghost[1], ghost[2], n);

                        if (std::abs(got - expected) > 1e-12) {
                            throw std::runtime_error("3D source-cell mismatch");
                        }
                    }
                }
            }
        }
    }

    // D3Q19 has no body-diagonal — the (-1,-1,-1) corner needs zero populations.
    {
        amrex::IntVect corner{AMREX_D_DECL(-1, -1, -1)};
        int needed = 0;
        for (int n = 0; n < LatticeTag::Q; ++n) {
            if (domain.contains(corner + latticeVelocity<LatticeTag>(n))) {
                ++needed;
            }
        }
        if constexpr (std::is_same_v<LatticeTag, muphfasa::D3Q19>) {
            if (needed != 0) {
                throw std::runtime_error("D3Q19 corner should need 0 populations");
            }
        }
        amrex::Print() << "      [PASS] (corner needed=" << needed << ")\n";
    }
}

#endif // AMREX_SPACEDIM == 2 / 3

//==============================================================================
// Test cases — source cell dispatch
//==============================================================================

// Verify NoSlip bounce-back: ghost f(g,n) == f(g+c_n, opposite(n)).
// Also checks MovingWall adds the velocity correction at the top wall.
template <typename LatticeTag>
static void testSourceCell()
{
    amrex::Print() << "    source cell (face, edge, corner)...\n";

#if AMREX_SPACEDIM == 2
    testSourceCell2D<LatticeTag>();
#else
    testSourceCell3D<LatticeTag>();
#endif
}

//==============================================================================
// Test cases — priority table
//==============================================================================

static void testPriorityTable()
{
    amrex::Print() << "    priority table (resolveBC)...\n";

    auto makeVelocity = [](double vx, double vy) {
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> vel{AMREX_D_DECL(0., 0., 0.)};
        vel[0] = vx;
        if constexpr (AMREX_SPACEDIM >= 2) {
            vel[1] = vy;
        }
        return vel;
    };

    // NoSlip outranks MovingWall.
    {
        amrex::Vector<lbmBCtype> types{lbmBCtype::NoSlip, lbmBCtype::MovingWall};
        amrex::Vector<amrex::Real> rho{1.0, 1.0};
        amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> vel{
            makeVelocity(0.0, 0.0), makeVelocity(0.1, 0.0)};

        auto resolved = resolveBC(types, rho, vel, {"xlo", "yhi"});
        if (resolved.type != lbmBCtype::NoSlip) {
            throw std::runtime_error("NoSlip should beat MovingWall");
        }
        if (std::abs(resolved.U[0]) > 1e-12) {
            throw std::runtime_error("NoSlip velocity should be zero");
        }
    }

    // Two connected MovingWalls conflict (different velocities at same edge).
    {
        amrex::Vector<lbmBCtype> types{lbmBCtype::MovingWall, lbmBCtype::MovingWall};
        if (!hasConflictingBC(types)) {
            throw std::runtime_error("two MovingWalls at an edge must conflict");
        }
    }

    // Mutually exclusive inflow/outlet combinations conflict.
    {
        amrex::Vector<lbmBCtype> types{lbmBCtype::Inflow, lbmBCtype::Outflow};
        if (!hasConflictingBC(types)) {
            throw std::runtime_error("Inflow + Outflow should conflict");
        }
    }
    {
        amrex::Vector<lbmBCtype> types{lbmBCtype::Inflow, lbmBCtype::PressureOutlet};
        if (!hasConflictingBC(types)) {
            throw std::runtime_error("Inflow + PressureOutlet should conflict");
        }
    }
    {
        amrex::Vector<lbmBCtype> types{lbmBCtype::Outflow, lbmBCtype::PressureOutlet};
        if (!hasConflictingBC(types)) {
            throw std::runtime_error("Outflow + PressureOutlet should conflict");
        }
    }

    // NoSlip + PressureOutlet is allowed — NoSlip wins.
    {
        amrex::Vector<lbmBCtype> types{lbmBCtype::NoSlip, lbmBCtype::PressureOutlet};
        if (hasConflictingBC(types)) {
            throw std::runtime_error("NoSlip + PressureOutlet should not conflict");
        }

        amrex::Vector<amrex::Real> rho{1.0, 1.0};
        amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> vel{
            makeVelocity(0.0, 0.0), makeVelocity(0.0, 0.0)};

        auto resolved = resolveBC(types, rho, vel, {"ylo", "xhi"});
        if (resolved.type != lbmBCtype::NoSlip) {
            throw std::runtime_error("NoSlip should beat PressureOutlet");
        }
    }

    // MovingWall outranks PressureOutlet.
    {
        amrex::Vector<lbmBCtype> types{lbmBCtype::MovingWall, lbmBCtype::PressureOutlet};
        amrex::Vector<amrex::Real> rho{1.0, 1.0};
        amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> vel{
            makeVelocity(0.1, 0.0), makeVelocity(0.0, 0.0)};

        auto resolved = resolveBC(types, rho, vel, {"ylo", "xhi"});
        if (resolved.type != lbmBCtype::MovingWall) {
            throw std::runtime_error("MovingWall should beat PressureOutlet");
        }
    }

    // Three faces meeting at a corner — NoSlip still dominates.
    {
        amrex::Vector<lbmBCtype> types{
            lbmBCtype::NoSlip, lbmBCtype::MovingWall, lbmBCtype::Outflow};
        amrex::Vector<amrex::Real> rho{1.0, 1.0, 1.0};
        amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> vel{
            makeVelocity(0.0, 0.0), makeVelocity(0.1, 0.0), makeVelocity(0.0, 0.0)};

        auto resolved = resolveBC(types, rho, vel, {"xlo", "ylo", "zlo"});
        if (resolved.type != lbmBCtype::NoSlip) {
            throw std::runtime_error("NoSlip should dominate three-face corner");
        }
    }

    // Three connected MovingWalls also conflict.
    {
        amrex::Vector<lbmBCtype> types{
            lbmBCtype::MovingWall, lbmBCtype::MovingWall, lbmBCtype::MovingWall};
        if (!hasConflictingBC(types)) {
            throw std::runtime_error("three MovingWalls at a corner must conflict");
        }
    }

    amrex::Print() << "      [PASS]\n";
}

//==============================================================================
// Test cases — PressureOutlet kernel
//==============================================================================

// A uniform equilibrium at rho_out should produce an equilibrium ghost
// (anti-bounce-back reduces to feq). For rho != rho_out the even part of
// each opposite pair is pinned to the outlet equilibrium so the face
// density is rho_out.
template <typename LatticeTag>
static void testPressureOutletKernel()
{
    amrex::Print() << "    PressureOutlet kernel...\n";

    using Vec = amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>;

    auto equilibrium = [](amrex::Real rho, const Vec& vel, int n) -> amrex::Real {
#if AMREX_SPACEDIM == 2
        return computeEquilibriumCell2D<LatticeTag>(rho, vel, n);
#else
        return computeEquilibriumCell3D<LatticeTag>(rho, vel, n);
#endif
    };

    const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(AMREX_D_DECL(3, 3, 3)));
    const amrex::Box fab_box = amrex::grow(domain, 1);
    amrex::FArrayBox fab(fab_box, LatticeTag::Q);
    auto arr = fab.array();

    Vec velocity{AMREX_D_DECL(0.03, -0.02, 0.01)};
    const amrex::Real rho_out = 1.0;

    // Ghost on the x-hi face, middle of the wall.
    const amrex::IntVect ghost{AMREX_D_DECL(4, 1, 1)};

    for (const amrex::Real rho : {amrex::Real(1.0), amrex::Real(1.15)}) {
        // Initialise interior to equilibrium at (rho, velocity); ghost to NaN.
        for (int n = 0; n < LatticeTag::Q; ++n) {
            amrex::LoopOnCpu(domain, [&](int i, int j, int k) {
                arr(i, j, k, n) = equilibrium(rho, velocity, n);
            });
            arr(ghost, n) = std::numeric_limits<amrex::Real>::quiet_NaN();
        }

        PressureOutletKernel<LatticeTag>::apply(ghost, arr, domain, rho_out);

        // Verify every incoming population at the ghost.
        for (int n = 0; n < LatticeTag::Q; ++n) {
            amrex::IntVect c = latticeVelocity<LatticeTag>(n);
            amrex::IntVect src = ghost + c;
            if (!domain.contains(src)) {
                continue;
            }

            amrex::Real got = arr(ghost, n);

            if (rho == rho_out) {
                // Outlet density matches interior — ghost should be equilibrium.
                amrex::Real expected = equilibrium(rho_out, velocity, n);
                if (std::abs(got - expected) > 1e-14) {
                    throw std::runtime_error("PressureOutlet: ghost is not equilibrium");
                }
            } else {
                // Even part of the pair must equal the outlet equilibrium's even part.
                const int opposite = LatticeTag::opposite(n);
                amrex::Real even_pair = got + arr(src, opposite);
                amrex::Real even_ref =
                    equilibrium(rho_out, velocity, n) +
                    equilibrium(rho_out, velocity, opposite);

                if (std::abs(even_pair - even_ref) > 1e-13) {
                    throw std::runtime_error("PressureOutlet: even part not pinned to rho_out");
                }
            }
        }
    }

    amrex::Print() << "      [PASS]\n";
}

//==============================================================================
// Lattice driver
//==============================================================================

template <typename LatticeTag>
void runForLattice(const char* name)
{
    amrex::Print() << "  Lattice " << name
                   << " (D=" << LatticeTag::D << " Q=" << LatticeTag::Q << ")\n";

    testCoverageSingleBox<LatticeTag>();
    testCoveragePeriodicX<LatticeTag>();
    testCoverageTwoBoxes<LatticeTag>();
    testSourceCell<LatticeTag>();
    testPressureOutletKernel<LatticeTag>();
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== BC corner/edge tests (DIM=" << AMREX_SPACEDIM << ") ===\n";

        testPriorityTable();

#if AMREX_SPACEDIM == 2
        runForLattice<muphfasa::D2Q9>("D2Q9");
#else
        runForLattice<muphfasa::D3Q15>("D3Q15");
        runForLattice<muphfasa::D3Q19>("D3Q19");
        runForLattice<muphfasa::D3Q27>("D3Q27");
#endif

        amrex::Print() << "All BC corner tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
