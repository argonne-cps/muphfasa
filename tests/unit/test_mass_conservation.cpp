// test_mass_conservation.cpp — Closed-box mass conservation regression.
// All walls NoSlip (or bounce-back), random f, N collide+stream steps.
// Total mass (sum rho over valid cells) must be conserved to ~1e-12.
// Tests single-box and two-box cases; covers corners/edges.

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Print.H>

#include "LBM/BoundaryConditions.H"
#include "LBM/LatticeDescriptor.H"
#include "LBM/Streaming.H"
#include "LBM/Collision.H" // BGK

#include <cmath>
#include <limits>
#include <array>
#include <random>
#include <string>
#include <stdexcept>

using namespace muphfasa;

static amrex::Geometry makeGeom(const amrex::Box& domain)
{
    amrex::RealBox rb(AMREX_D_DECL(0.,0.,0.), AMREX_D_DECL(1.,1.,1.));
    amrex::Array<int,AMREX_SPACEDIM> per{}; // all non-periodic
    amrex::Geometry geom;
    geom.define(domain, rb, amrex::CoordSys::cartesian, per);
    return geom;
}

// Generic helpers templated on LatticeTag
template <typename LatticeTag>
static amrex::Real totalMass(const amrex::MultiFab& mf)
{
    amrex::Real sum = 0;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const arr = mf.const_array(mfi);
        const amrex::Box& vbx = mfi.validbox();
        const auto lo = vbx.smallEnd(), hi = vbx.bigEnd();
#if AMREX_SPACEDIM==2
        for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i)
            for (int n=0; n<LatticeTag::Q; ++n) sum += arr(i,j,0,n);
#else
        for (int k=lo[2]; k<=hi[2]; ++k) for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i)
            for (int n=0; n<LatticeTag::Q; ++n) sum += arr(i,j,k,n);
#endif
    }
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    return sum;
}

template <typename LatticeTag>
static void fillRandomF(amrex::MultiFab& mf, std::mt19937& rng)
{
    std::uniform_real_distribution<double> dist(0.1, 1.0);
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr = mf.array(mfi);
        const amrex::Box& vbx = mfi.validbox();
        const auto lo=vbx.smallEnd(), hi=vbx.bigEnd();
#if AMREX_SPACEDIM==2
        for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i)
            for (int n=0; n<LatticeTag::Q; ++n) arr(i,j,0,n)=static_cast<amrex::Real>(dist(rng));
#else
        for (int k=lo[2]; k<=hi[2]; ++k) for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i)
            for (int n=0; n<LatticeTag::Q; ++n) arr(i,j,k,n)=static_cast<amrex::Real>(dist(rng));
#endif
    }
}

template <typename LatticeTag>
static BCManager<LatticeTag> makeAllNoSlipBCM()
{
    BCManager<LatticeTag> bcm;
    for (int d=0; d<AMREX_SPACEDIM; ++d) for (int b:{-1,1}) {
        bcm.setFaceBC(d,b,lbmBCtype::NoSlip);
        bcm.setFaceDensity(d,b,1.0);
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> z{AMREX_D_DECL(0.,0.,0.)};
        bcm.setFaceVel(d,b,z);
    }
    bcm.finalize();
    return bcm;
}

template <typename LatticeTag>
static void testMassConservation(const char* tag, amrex::Box domain, int max_grid_size)
{
    amrex::Geometry geom = makeGeom(domain);
    amrex::BoxArray ba(domain); ba.maxSize(max_grid_size);
    amrex::DistributionMapping dm(ba);

    amrex::MultiFab f(ba, dm, LatticeTag::Q, 1);
    amrex::MultiFab f2(ba, dm, LatticeTag::Q, 1);
    // feq scratch for collide; collideFullMF can be used but we use collideMF with explicit feq
    amrex::MultiFab feq(ba, dm, LatticeTag::Q, 0);

    std::mt19937 rng(12345);
    fillRandomF<LatticeTag>(f, rng);

    // Drive the BC through the same PhysBCFunct/GpuBndryFuncFab path as the solver,
    // so per-box BCRec handling (amrex::setBC marks inter-box faces int_dir) is covered.
    std::array<std::string, 2 * AMREX_SPACEDIM> names;
    names.fill("NoSlip");
    amrex::GpuBndryFuncFab<BCManager<LatticeTag>> gpu_bndry_func(makeAllNoSlipBCM<LatticeTag>());
    amrex::PhysBCFunct<amrex::GpuBndryFuncFab<BCManager<LatticeTag>>> phys_bc(
        geom, makeBCRecVector(names, LatticeTag::Q), gpu_bndry_func);

    // omega = 1.0 (tau=1) — BGK conserves mass exactly per cell
    const amrex::Real omega = 1.0;

    amrex::Real m0 = totalMass<LatticeTag>(f);
    const int Nsteps = 50;

    for (int step=0; step<Nsteps; ++step) {
        // collide: f -> feq then relax
        // We need feq: compute via equilibrium from rho/u? For mass conservation test,
        // any mass-conserving collide is fine. Instead of computing feq from macroscopic,
        // we can use a simple BGK that conserves mass: set feq = uniform 1/Q * rho per cell.
        // Easiest: use collideFullMF which computes rho/u/feq internally and relaxes.
        // But collideFullMF requires only f and fstar; we can do in-place via f->f2.
        // Use collideFullMF<LatticeTag,BGK>(f2, f, omega) then copy back.
        // For now use manual collide that conserves sum: we compute rho per cell and set feq = rho/Q?
        // That conserves mass. Let's just compute rho per cell and use feq = rho * weights.

        // Compute feq as rho * w_q (zero velocity equilibrium) — mass conserving.
        for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
            auto const farr = f.const_array(mfi);
            auto feqarr = feq.array(mfi);
            const auto vbx = mfi.validbox();
            const auto lo=vbx.smallEnd(), hi=vbx.bigEnd();
#if AMREX_SPACEDIM==2
            for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i) {
                amrex::Real rho=0; for(int n=0;n<LatticeTag::Q;++n) rho+=farr(i,j,0,n);
                for(int n=0;n<LatticeTag::Q;++n) feqarr(i,j,0,n)=rho*LatticeTag::weights(n);
            }
#else
            for (int k=lo[2]; k<=hi[2]; ++k) for(int j=lo[1]; j<=hi[1]; ++j) for(int i=lo[0]; i<=hi[0]; ++i){
                amrex::Real rho=0; for(int n=0;n<LatticeTag::Q;++n) rho+=farr(i,j,k,n);
                for(int n=0;n<LatticeTag::Q;++n) feqarr(i,j,k,n)=rho*LatticeTag::weights(n);
            }
#endif
        }
        collideMF<LatticeTag,BGK>(f, feq, omega);

        // fill ghosts: FillBoundary for interior, then BC
        f.FillBoundary(geom.periodicity());
        phys_bc(f, 0, LatticeTag::Q, f.nGrowVect(), 0.0, 0);

        // stream: f -> f2 via pull, then copy back
        // Use default_stream per tile
        for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) {
            const auto farr = f.const_array(mfi);
            auto f2arr = f2.array(mfi);
            const amrex::Box vbx = mfi.validbox();
            amrex::ParallelFor(vbx, LatticeTag::Q, [=] AMREX_GPU_DEVICE(int i,int j,int k,int n) noexcept {
                int ci = LatticeTag::cx(n), cj = LatticeTag::cy(n);
#if AMREX_SPACEDIM==2
                f2arr(i,j,0,n) = farr(i-ci, j-cj, 0, n);
#else
                int ck = LatticeTag::cz(n);
                f2arr(i,j,k,n) = farr(i-ci, j-cj, k-ck, n);
#endif
            });
        }
        amrex::MultiFab::Copy(f, f2, 0,0, LatticeTag::Q, 0);
        // f now holds post-stream valid data; ghosts will be refilled next iter
    }

    amrex::Real m1 = totalMass<LatticeTag>(f);
    amrex::Real rel = std::abs(m1 - m0) / std::abs(m0);
    amrex::Print() << "  " << tag << " domain " << domain << " max_grid " << max_grid_size
                   << " m0=" << m0 << " m1=" << m1 << " rel drift=" << rel << "\n";
    const amrex::Real tol = 1e-12;
    if (rel > tol) {
        throw std::runtime_error(std::string(tag)+": mass drift "+std::to_string(rel)+" exceeds tol "+std::to_string(tol));
    }
    amrex::Print() << "    [PASS]\n";
}

int main(int argc,char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "\n=== Closed-box mass conservation (DIM=" << AMREX_SPACEDIM << ") ===\n";
#if AMREX_SPACEDIM==2
        // Single-box tests
        testMassConservation<D2Q9>("D2Q9 single-box",
            amrex::Box(amrex::IntVect{0,0}, amrex::IntVect{7,7}), 64);
        testMassConservation<D2Q9>("D2Q9 two-box",
            amrex::Box(amrex::IntVect{0,0}, amrex::IntVect{7,7}), 4);
        // Non-square to exercise edges
        testMassConservation<D2Q9>("D2Q9 rect",
            amrex::Box(amrex::IntVect{0,0}, amrex::IntVect{11,5}), 64);
#else
        testMassConservation<D3Q15>("D3Q15 single-box",
            amrex::Box(amrex::IntVect{0,0,0}, amrex::IntVect{5,5,5}), 64);
        testMassConservation<D3Q15>("D3Q15 two-box",
            amrex::Box(amrex::IntVect{0,0,0}, amrex::IntVect{7,5,5}), 4);
        testMassConservation<D3Q19>("D3Q19 single-box",
            amrex::Box(amrex::IntVect{0,0,0}, amrex::IntVect{5,5,5}), 64);
        testMassConservation<D3Q19>("D3Q19 two-box",
            amrex::Box(amrex::IntVect{0,0,0}, amrex::IntVect{7,5,5}), 4);
        testMassConservation<D3Q27>("D3Q27 single-box",
            amrex::Box(amrex::IntVect{0,0,0}, amrex::IntVect{5,5,5}), 64);
        testMassConservation<D3Q27>("D3Q27 two-box",
            amrex::Box(amrex::IntVect{0,0,0}, amrex::IntVect{7,5,5}), 4);
#endif
        amrex::Print() << "All mass conservation tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
