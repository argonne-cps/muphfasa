// test_bc_corners.cpp — Unit tests for BCManager corner/edge handling,
// diagonal-population source cell, and resolveBC priority table.
// Host-only, tiny domains, every lattice of the active DIM.

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include "LBM/BoundaryConditions.H"
#include "LBM/BCTypes/BCCommon.H"
#include "LBM/LatticeDescriptor.H"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

using namespace muphfasa;

static amrex::Geometry makeGeometry(const amrex::Box& domain,
                                    const amrex::IntVect& is_periodic_iv)
{
    amrex::RealBox rb(AMREX_D_DECL(0.,0.,0.), AMREX_D_DECL(1.,1.,1.));
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{};
    for (int d=0; d<AMREX_SPACEDIM; ++d) is_periodic[d]=is_periodic_iv[d];
    amrex::Geometry geom;
    geom.define(domain, rb, amrex::CoordSys::cartesian, is_periodic);
    return geom;
}

static amrex::BCRec makeBCRecForTest(const amrex::IntVect& is_periodic)
{
    amrex::BCRec bcr;
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        int v = is_periodic[d] ? amrex::BCType::int_dir : kCustomBCSentinel;
        bcr.setLo(d, v); bcr.setHi(d, v);
    }
    return bcr;
}

template <typename LatticeTag>
static void fillNaN(amrex::MultiFab& mf)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr = mf.array(mfi);
        const amrex::Box bx = mfi.fabbox();
        amrex::ParallelFor(bx, LatticeTag::Q, [=] AMREX_GPU_DEVICE(int i,int j,int k,int n) noexcept {
            arr(i,j,k,n) = std::numeric_limits<amrex::Real>::quiet_NaN();
        });
    }
}

template <typename LatticeTag>
static void fillRandomValid(amrex::MultiFab& mf, std::mt19937& rng)
{
    std::uniform_real_distribution<double> dist(0.5, 1.5);
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr = mf.array(mfi);
        const amrex::Box vbx = mfi.validbox();
        const auto lo = vbx.smallEnd(), hi = vbx.bigEnd();
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
static void applyBCManager(amrex::MultiFab& mf,
                           const amrex::Geometry& geom,
                           const BCManager<LatticeTag>& bcm,
                           const amrex::BCRec& bcr)
{
    const amrex::Box& domain = geom.Domain();
    const amrex::BCRec* bcr_ptr = &bcr;
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr = mf.array(mfi);
        const amrex::Box fbox = mfi.fabbox();
        const auto lo = fbox.smallEnd(), hi = fbox.bigEnd();
#if AMREX_SPACEDIM==2
        for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i) {
            amrex::IntVect iv{AMREX_D_DECL(i,j,0)};
            if (domain.contains(iv)) continue;
            bool wall=false;
            for (int d=0; d<AMREX_SPACEDIM; ++d) if (bcr.lo(d)!=amrex::BCType::int_dir)
                if (iv[d]<domain.smallEnd(d)||iv[d]>domain.bigEnd(d)) wall=true;
            if (!wall) continue;
            bcm(iv, arr, 0, LatticeTag::Q, geom.data(), 0.0, bcr_ptr, 0, 0);
        }
#else
        for (int k=lo[2]; k<=hi[2]; ++k) for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i) {
            amrex::IntVect iv{AMREX_D_DECL(i,j,k)};
            if (domain.contains(iv)) continue;
            bool wall=false;
            for (int d=0; d<AMREX_SPACEDIM; ++d) if (bcr.lo(d)!=amrex::BCType::int_dir)
                if (iv[d]<domain.smallEnd(d)||iv[d]>domain.bigEnd(d)) wall=true;
            if (!wall) continue;
            bcm(iv, arr, 0, LatticeTag::Q, geom.data(), 0.0, bcr_ptr, 0, 0);
        }
#endif
    }
}

template <typename LatticeTag>
static void checkCoverage(const amrex::MultiFab& mf, const amrex::Geometry& geom)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const arr = mf.const_array(mfi);
        const amrex::Box vbx = mfi.validbox();
        const auto lo = vbx.smallEnd(), hi = vbx.bigEnd();
#if AMREX_SPACEDIM==2
        for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i) {
            amrex::IntVect y{AMREX_D_DECL(i,j,0)};
            for (int n=0; n<LatticeTag::Q; ++n) {
                amrex::IntVect src = y - cvec<LatticeTag>(n);
                amrex::Real v = arr(src[0],src[1],0,n);
                if (!std::isfinite(v))
                    throw std::runtime_error("coverage: non-finite src for y=("+std::to_string(i)+","+std::to_string(j)+") n="+std::to_string(n));
            }
        }
#else
        for (int k=lo[2]; k<=hi[2]; ++k) for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i) {
            amrex::IntVect y{AMREX_D_DECL(i,j,k)};
            for (int n=0; n<LatticeTag::Q; ++n) {
                amrex::IntVect src = y - cvec<LatticeTag>(n);
                amrex::Real v = arr(src[0],src[1],src[2],n);
                if (!std::isfinite(v))
                    throw std::runtime_error("coverage: non-finite src");
            }
        }
#endif
    }
}

template <typename LatticeTag>
static void testCoverageSingleBox()
{
    amrex::Print() << "    coverage: single non-periodic box...\n";
#if AMREX_SPACEDIM==2
    amrex::Box domain(amrex::IntVect{0,0}, amrex::IntVect{3,3});
#else
    amrex::Box domain(amrex::IntVect{0,0,0}, amrex::IntVect{3,3,3});
#endif
    amrex::IntVect is_per{AMREX_D_DECL(0,0,0)};
    amrex::Geometry geom = makeGeometry(domain, is_per);
    amrex::BoxArray ba(domain); ba.maxSize(64);
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);
    fillNaN<LatticeTag>(mf);
    std::mt19937 rng(42); fillRandomValid<LatticeTag>(mf, rng);
    BCManager<LatticeTag> bcm;
    for (int d=0; d<AMREX_SPACEDIM; ++d) for (int b:{-1,1}) {
        bcm.setFaceBC(d,b,lbmBCtype::NoSlip); bcm.setFaceDensity(d,b,1.0);
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> z{AMREX_D_DECL(0.,0.,0.)}; bcm.setFaceVel(d,b,z);
    }
    bcm.finalize(); auto bcr=makeBCRecForTest(is_per);
    applyBCManager<LatticeTag>(mf,geom,bcm,bcr);
    checkCoverage<LatticeTag>(mf,geom);
    amrex::Print() << "      [PASS]\n";
}

template <typename LatticeTag>
static void testCoveragePeriodicX()
{
    amrex::Print() << "    coverage: x-periodic + FillBoundary...\n";
#if AMREX_SPACEDIM==2
    amrex::Box domain(amrex::IntVect{0,0}, amrex::IntVect{3,3});
#else
    amrex::Box domain(amrex::IntVect{0,0,0}, amrex::IntVect{3,3,3});
#endif
    amrex::IntVect is_per{AMREX_D_DECL(1,0,0)};
    amrex::Geometry geom = makeGeometry(domain, is_per);
    amrex::BoxArray ba(domain); ba.maxSize(64);
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);
    fillNaN<LatticeTag>(mf);
    std::mt19937 rng(43); fillRandomValid<LatticeTag>(mf, rng);
    mf.FillBoundary(geom.periodicity());
    BCManager<LatticeTag> bcm;
    for (int d=0; d<AMREX_SPACEDIM; ++d) for (int b:{-1,1}) {
        if (is_per[d]) continue;
        bcm.setFaceBC(d,b,lbmBCtype::NoSlip); bcm.setFaceDensity(d,b,1.0);
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> z{AMREX_D_DECL(0.,0.,0.)}; bcm.setFaceVel(d,b,z);
    }
    bcm.finalize(); auto bcr=makeBCRecForTest(is_per);
    applyBCManager<LatticeTag>(mf,geom,bcm,bcr);
    checkCoverage<LatticeTag>(mf,geom);
    amrex::Print() << "      [PASS]\n";
}

template <typename LatticeTag>
static void testCoverageTwoBoxes()
{
    amrex::Print() << "    coverage: two boxes + FillBoundary...\n";
#if AMREX_SPACEDIM==2
    amrex::Box domain(amrex::IntVect{0,0}, amrex::IntVect{7,3});
#else
    amrex::Box domain(amrex::IntVect{0,0,0}, amrex::IntVect{7,3,3});
#endif
    amrex::IntVect is_per{AMREX_D_DECL(0,0,0)};
    amrex::Geometry geom = makeGeometry(domain, is_per);
    amrex::BoxArray ba(domain); ba.maxSize(4);
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);
    fillNaN<LatticeTag>(mf);
    std::mt19937 rng(44); fillRandomValid<LatticeTag>(mf, rng);
    mf.FillBoundary(geom.periodicity());
    BCManager<LatticeTag> bcm;
    for (int d=0; d<AMREX_SPACEDIM; ++d) for (int b:{-1,1}) {
        bcm.setFaceBC(d,b,lbmBCtype::NoSlip); bcm.setFaceDensity(d,b,1.0);
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> z{AMREX_D_DECL(0.,0.,0.)}; bcm.setFaceVel(d,b,z);
    }
    bcm.finalize(); auto bcr=makeBCRecForTest(is_per);
    applyBCManager<LatticeTag>(mf,geom,bcm,bcr);
    mf.FillBoundary(geom.periodicity());
    checkCoverage<LatticeTag>(mf,geom);
    amrex::Print() << "      [PASS]\n";
}

template <typename LatticeTag>
static void testSourceCell()
{
    amrex::Print() << "    source cell (face diagonal + corner)...\n";
#if AMREX_SPACEDIM==2
    amrex::Box domain(amrex::IntVect{0,0}, amrex::IntVect{5,5});
    amrex::IntVect is_per{AMREX_D_DECL(0,0,0)};
    amrex::Geometry geom = makeGeometry(domain, is_per);
    amrex::BoxArray ba(domain); ba.maxSize(64);
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr=mf.array(mfi);
        const auto vbx=mfi.validbox(); const auto lo=vbx.smallEnd(), hi=vbx.bigEnd();
        for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i)
            for (int n=0; n<LatticeTag::Q; ++n) arr(i,j,0,n)=i*1000+j*10+n*0.1+0.5;
        const auto fbx=mfi.fabbox(); const auto flo=fbx.smallEnd(), fhi=fbx.bigEnd();
        for (int j=flo[1]; j<=fhi[1]; ++j) for (int i=flo[0]; i<=fhi[0]; ++i) {
            amrex::IntVect iv{AMREX_D_DECL(i,j,0)};
            if (vbx.contains(iv)) continue;
            for (int n=0; n<LatticeTag::Q; ++n) arr(i,j,0,n)=std::numeric_limits<amrex::Real>::quiet_NaN();
        }
    }
    BCManager<LatticeTag> bcm;
    for (int d=0; d<AMREX_SPACEDIM; ++d) for (int b:{-1,1}) {
        bcm.setFaceBC(d,b,lbmBCtype::NoSlip); bcm.setFaceDensity(d,b,1.0);
        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> z{AMREX_D_DECL(0.,0.,0.)}; bcm.setFaceVel(d,b,z);
    }
    bcm.finalize(); auto bcr=makeBCRecForTest(is_per);
    applyBCManager<LatticeTag>(mf,geom,bcm,bcr);
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto const arr=mf.const_array(mfi);
        const auto fbx=mfi.fabbox(); const auto lo=fbx.smallEnd(), hi=fbx.bigEnd();
        for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i) {
            amrex::IntVect g{AMREX_D_DECL(i,j,0)}; if (domain.contains(g)) continue;
            for (int n=0; n<LatticeTag::Q; ++n) {
                amrex::IntVect c=cvec<LatticeTag>(n); if (!domain.contains(g+c)) continue;
                int opp=LatticeTag::opposite(n); amrex::IntVect src=g+c;
                amrex::Real expv=arr(src[0],src[1],0,opp), got=arr(g[0],g[1],0,n);
                if (std::abs(got-expv)>1e-12) throw std::runtime_error("source-cell mismatch");
            }
        }
    }
    { bool found=false; for (int n=0; n<LatticeTag::Q; ++n){ amrex::IntVect c=cvec<LatticeTag>(n); amrex::IntVect g{AMREX_D_DECL(-1,-1,0)}; if(domain.contains(g+c)){ if(c[0]!=1||c[1]!=1) throw std::runtime_error("corner wrong"); found=true; } } if(!found) throw std::runtime_error("corner no pop"); }
    // MovingWall
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr=mf.array(mfi);
        const auto fbx=mfi.fabbox(); const auto lo=fbx.smallEnd(), hi=fbx.bigEnd();
        const auto vbx=mfi.validbox();
        for (int j=lo[1]; j<=hi[1]; ++j) for (int i=lo[0]; i<=hi[0]; ++i){
            amrex::IntVect iv{AMREX_D_DECL(i,j,0)};
            for(int n=0;n<LatticeTag::Q;++n) arr(i,j,0,n)= vbx.contains(iv)? i*1000+j*10+n*0.1+0.5 : std::numeric_limits<amrex::Real>::quiet_NaN();
        }
    }
    BCManager<LatticeTag> bcm2;
    for (int d=0; d<AMREX_SPACEDIM; ++d) for (int b:{-1,1}) {
        if (d==1&&b==1){ bcm2.setFaceBC(d,b,lbmBCtype::MovingWall); bcm2.setFaceDensity(d,b,1.0); amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> U{AMREX_D_DECL(0.1,0.,0.)}; bcm2.setFaceVel(d,b,U); }
        else { bcm2.setFaceBC(d,b,lbmBCtype::NoSlip); bcm2.setFaceDensity(d,b,1.0); amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> z{AMREX_D_DECL(0.,0.,0.)}; bcm2.setFaceVel(d,b,z); }
    }
    bcm2.finalize(); applyBCManager<LatticeTag>(mf,geom,bcm2,bcr);
    { amrex::IntVect g{AMREX_D_DECL(2,6,0)}; for(amrex::MFIter mfi(mf);mfi.isValid();++mfi){ auto const arr=mf.const_array(mfi); for(int n=0;n<LatticeTag::Q;++n){ amrex::IntVect c=cvec<LatticeTag>(n); if(!domain.contains(g+c)) continue; int opp=LatticeTag::opposite(n); amrex::IntVect src=g+c; amrex::Real fsrc=src[0]*1000+src[1]*10+opp*0.1+0.5; amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> U{AMREX_D_DECL(0.1,0.,0.)}; amrex::Real cdot=LatticeTag::cx(opp)*U[0]+LatticeTag::cy(opp)*U[1]; amrex::Real contrib=-2.*1.0*LatticeTag::weights(opp)*cdot*LatticeTag::cs2inv; if(std::abs(arr(g[0],g[1],0,n)-(fsrc+contrib))>1e-12) throw std::runtime_error("MW contrib mismatch"); } } }
    amrex::Print() << "      [PASS]\n";
#else
    amrex::Box domain(amrex::IntVect{0,0,0}, amrex::IntVect{3,3,3});
    amrex::IntVect is_per{AMREX_D_DECL(0,0,0)};
    amrex::Geometry geom = makeGeometry(domain, is_per);
    amrex::BoxArray ba(domain); ba.maxSize(64);
    amrex::DistributionMapping dm(ba);
    amrex::MultiFab mf(ba, dm, LatticeTag::Q, 1);
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto arr=mf.array(mfi);
        const auto vbx=mfi.validbox(); const auto lo=vbx.smallEnd(), hi=vbx.bigEnd();
        for(int k=lo[2];k<=hi[2];++k) for(int j=lo[1];j<=hi[1];++j) for(int i=lo[0];i<=hi[0];++i)
            for(int n=0;n<LatticeTag::Q;++n) arr(i,j,k,n)=i*1000+j*10+k*0.1+n*0.01+0.5;
        const auto fbx=mfi.fabbox(); const auto flo=fbx.smallEnd(), fhi=fbx.bigEnd();
        for(int k=flo[2];k<=fhi[2];++k) for(int j=flo[1];j<=fhi[1];++j) for(int i=flo[0];i<=fhi[0];++i){
            amrex::IntVect iv{AMREX_D_DECL(i,j,k)}; if(vbx.contains(iv)) continue;
            for(int n=0;n<LatticeTag::Q;++n) arr(i,j,k,n)=std::numeric_limits<amrex::Real>::quiet_NaN();
        }
    }
    BCManager<LatticeTag> bcm;
    for(int d=0;d<AMREX_SPACEDIM;++d) for(int b:{-1,1}){ bcm.setFaceBC(d,b,lbmBCtype::NoSlip); bcm.setFaceDensity(d,b,1.0); amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> z{AMREX_D_DECL(0.,0.,0.)}; bcm.setFaceVel(d,b,z); }
    bcm.finalize(); auto bcr=makeBCRecForTest(is_per); applyBCManager<LatticeTag>(mf,geom,bcm,bcr);
    for(amrex::MFIter mfi(mf);mfi.isValid();++mfi){ auto const arr=mf.const_array(mfi); const auto fbx=mfi.fabbox(); const auto lo=fbx.smallEnd(), hi=fbx.bigEnd(); for(int k=lo[2];k<=hi[2];++k) for(int j=lo[1];j<=hi[1];++j) for(int i=lo[0];i<=hi[0];++i){ amrex::IntVect g{AMREX_D_DECL(i,j,k)}; if(domain.contains(g)) continue; for(int n=0;n<LatticeTag::Q;++n){ amrex::IntVect c=cvec<LatticeTag>(n); if(!domain.contains(g+c)) continue; int opp=LatticeTag::opposite(n); amrex::IntVect src=g+c; if(std::abs(arr(g[0],g[1],g[2],n)-arr(src[0],src[1],src[2],opp))>1e-12) throw std::runtime_error("3D src mismatch"); } } }
    amrex::IntVect cg{AMREX_D_DECL(-1,-1,-1)}; int need=0; for(int n=0;n<LatticeTag::Q;++n) if(domain.contains(cg+cvec<LatticeTag>(n))) ++need;
    if constexpr (std::is_same_v<LatticeTag,muphfasa::D3Q19>) if(need!=0) throw std::runtime_error("D3Q19 corner 0 pops");
    amrex::Print() << "      [PASS] (corner needed=" << need << ")\n";
#endif
}

static void testPriorityTable()
{
    amrex::Print() << "    priority table (resolveBC)...\n";
    auto makeU=[&](double x,double y){ amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> u{AMREX_D_DECL(0.,0.,0.)}; u[0]=x; if constexpr(AMREX_SPACEDIM>=2) u[1]=y; return u; };
    { amrex::Vector<lbmBCtype> t{lbmBCtype::NoSlip,lbmBCtype::MovingWall}; amrex::Vector<amrex::Real> r{1,1}; amrex::Vector<amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>> U{makeU(0,0),makeU(0.1,0)}; auto rr=resolveBC(t,r,U,{"xlo","yhi"}); if(rr.type!=lbmBCtype::NoSlip) throw std::runtime_error("NoSlip beats MW"); if(std::abs(rr.U[0])>1e-12) throw std::runtime_error("U should be 0"); }
    { amrex::Vector<lbmBCtype> t{lbmBCtype::MovingWall,lbmBCtype::MovingWall}; amrex::Vector<amrex::Real> r{1,1}; amrex::Vector<amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>> U{makeU(0.1,0),makeU(0,0.2)}; auto rr=resolveBC(t,r,U,{"ylo","yhi"}); if(rr.type!=lbmBCtype::MovingWall) throw std::runtime_error("MW+MW"); if(std::abs(rr.U[0]-0.05)>1e-12||std::abs(rr.U[1]-0.1)>1e-12) throw std::runtime_error("avg fail"); }
    { amrex::Vector<lbmBCtype> t{lbmBCtype::Inflow,lbmBCtype::Outflow}; if(!hasConflictingBC(t)) throw std::runtime_error("conflict"); }
    { amrex::Vector<lbmBCtype> t{lbmBCtype::NoSlip,lbmBCtype::MovingWall,lbmBCtype::Outflow}; amrex::Vector<amrex::Real> r{1,1,1}; amrex::Vector<amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>> U{makeU(0,0),makeU(0.1,0),makeU(0,0)}; auto rr=resolveBC(t,r,U,{"xlo","ylo","zlo"}); if(rr.type!=lbmBCtype::NoSlip) throw std::runtime_error("3-face NoSlip"); }
    { amrex::Vector<lbmBCtype> t{lbmBCtype::MovingWall,lbmBCtype::MovingWall,lbmBCtype::MovingWall}; amrex::Vector<amrex::Real> r{1,1,1}; amrex::Vector<amrex::GpuArray<amrex::Real,AMREX_SPACEDIM>> U{makeU(0.1,0),makeU(0.2,0),makeU(0.3,0)}; auto rr=resolveBC(t,r,U,{"xlo","ylo","zlo"}); if(std::abs(rr.U[0]-0.2)>1e-12) throw std::runtime_error("3MW avg"); }
    amrex::Print() << "      [PASS]\n";
}

template <typename LatticeTag> void runForLattice(const char* name){
    amrex::Print() << "  Lattice " << name << " (D=" << LatticeTag::D << " Q=" << LatticeTag::Q << ")\n";
    testCoverageSingleBox<LatticeTag>();
    testCoveragePeriodicX<LatticeTag>();
    testCoverageTwoBoxes<LatticeTag>();
    testSourceCell<LatticeTag>();
}

int main(int argc,char* argv[]){
    amrex::Initialize(argc,argv);
    { amrex::Print() << "\n=== BC corner/edge tests (DIM=" << AMREX_SPACEDIM << ") ===\n";
      testPriorityTable();
#if AMREX_SPACEDIM==2
      runForLattice<muphfasa::D2Q9>("D2Q9");
#else
      runForLattice<muphfasa::D3Q15>("D3Q15");
      runForLattice<muphfasa::D3Q19>("D3Q19");
      runForLattice<muphfasa::D3Q27>("D3Q27");
#endif
      amrex::Print() << "All BC corner tests passed.\n";
    } amrex::Finalize(); return 0;
}
