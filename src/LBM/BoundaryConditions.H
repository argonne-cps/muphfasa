// BoundaryConditions.H — BCManager<LatticeTag>: the GPU-callable functor
// plugged into amrex::PhysBCFunct<amrex::GpuBndryFuncFab<...>> that fills
// ghost cells for the LBM distribution MultiFab.
//
// Ghost population rule: for ghost g, population n is needed iff g + c_n is
// inside geom.Domain(); source cell is g + c_n; BounceBack: f(g,n)=f(src,opp)+contrib,
// Outflow: f(g,n)=f(src,n). This covers faces, edges, and corners uniformly
// and correctly handles diagonal populations.
//
// Priority resolution at edges/corners: NoSlip(3) > MovingWall(2) > Inflow(1) > Outflow(0).
// Winner type among touching faces determines BC; rho/U are averaged over
// faces of the winning type only. Inflow+Outflow together aborts.

#ifndef MUPHFASA_BOUNDARY_CONDITIONS_H
#define MUPHFASA_BOUNDARY_CONDITIONS_H

#include "BCTypes/MovingWallKernel.H"
#include "BCTypes/NoSlipKernel.H"
#include "BCTypes/InflowKernel.H"
#include "BCTypes/OutflowKernel.H"

#include <AMReX_Array.H>
#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <array>
#include <string>

namespace muphfasa {

// Shared face-index convention: [xlo, xhi, ylo, yhi, zlo, zhi].
inline constexpr int faceIndex(int dir, int bound) noexcept {
    return 2 * dir + (bound < 0 ? 0 : 1);
}

enum class lbmBCtype { MovingWall, NoSlip, Inflow, Outflow, Periodic };

inline constexpr int kCustomBCSentinel = amrex::BCType::user_1;
static_assert(kCustomBCSentinel != amrex::BCType::int_dir &&
              kCustomBCSentinel != amrex::BCType::ext_dir &&
              kCustomBCSentinel != amrex::BCType::foextrap &&
              kCustomBCSentinel != amrex::BCType::hoextrap &&
              kCustomBCSentinel != amrex::BCType::hoextrapcc &&
              kCustomBCSentinel != amrex::BCType::reflect_even &&
              kCustomBCSentinel != amrex::BCType::reflect_odd,
              "custom BC sentinel must not collide with a semantically-defined amrex::BCType");

// ── helpers for 3^D resolved BC indexing ────────────────────────────────
namespace detail {
inline constexpr int pow3(int d) noexcept {
    int r = 1;
    for (int i = 0; i < d; ++i) r *= 3;
    return r;
}
inline constexpr int numCodes() noexcept {
#if AMREX_SPACEDIM == 2
    return 9;
#else
    return 27;
#endif
}
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int ghostClass(const amrex::IntVect& code) noexcept {
    int idx = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        idx += (code[d] + 1) * pow3(d);
    }
    return idx;
}
} // namespace detail

// ── Priority resolution ────────────────────────────────────────────────
inline int bcRank(lbmBCtype t) noexcept {
    switch (t) {
        case lbmBCtype::NoSlip:     return 3;
        case lbmBCtype::MovingWall: return 2;
        case lbmBCtype::Inflow:     return 1;
        case lbmBCtype::Outflow:    return 0;
        case lbmBCtype::Periodic:   return -1;
    }
    return -1;
}

// Result of resolving touching faces at an edge/corner ghost.
struct ResolvedFaceBC {
    lbmBCtype type{};
    amrex::Real rho{1.0};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> U{AMREX_D_DECL(0.,0.,0.)};
};

// Resolve touching faces to a single BC. Host-side, unit-testable.
// face_types/rho/U are parallel arrays of length nfaces (faces that meet at ghost).
// Aborts if both Inflow and Outflow present.
ResolvedFaceBC resolveBC(
    const amrex::Vector<lbmBCtype>& face_types,
    const amrex::Vector<amrex::Real>& face_rho,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& face_U,
    const amrex::Vector<std::string>& face_names = {});

bool hasConflictingBC(const amrex::Vector<lbmBCtype>& face_types) noexcept;

template <typename LatticeTag>
class BCManager
{
public:
    struct ResolvedBC {
        lbmBCtype type{lbmBCtype::Periodic};
        amrex::Real rho{1.0};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> U{AMREX_D_DECL(0.,0.,0.)};
    };

    BCManager() { m_face_bc.fill(lbmBCtype::Periodic); }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator()(const amrex::IntVect& iv,
                     amrex::Array4<amrex::Real> const& dest,
                     const int dcomp,
                     const int numcomp,
                     amrex::GeometryData const& geom,
                     const amrex::Real time,
                     const amrex::BCRec* bcr,
                     const int bcomp,
                     const int orig_comp) const
    {
        amrex::ignore_unused(dcomp, numcomp, time, orig_comp);

        const amrex::Box& domain = geom.Domain();

        // Compute code[d] in {-1,0,+1} per dimension
        amrex::IntVect code{AMREX_D_DECL(0,0,0)};
        const auto dom_lo = domain.smallEnd();
        const auto dom_hi = domain.bigEnd();
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (iv[d] < dom_lo[d]) code[d] = -1;
            else if (iv[d] > dom_hi[d]) code[d] = 1;
            else code[d] = 0;
            // Periodic dims are not walls: FillBoundary already filled them.
            if (bcr[bcomp].lo(d) == amrex::BCType::int_dir ||
                bcr[bcomp].hi(d) == amrex::BCType::int_dir) {
                code[d] = 0;
            }
        }

        // All zero => periodic-only ghost, already filled by FillBoundary
        bool all_zero = true;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) if (code[d] != 0) all_zero = false;
        if (all_zero) return;

        int idx = detail::ghostClass(code);
        const ResolvedBC& rbc = m_resolved[idx];

        // Periodic entries are never used (FillBoundary handled them) but
        // may appear in the resolved table for debugging — just return.
        if (rbc.type == lbmBCtype::Periodic) return;

        switch (rbc.type) {
            case lbmBCtype::MovingWall:
                MovingWallKernel<LatticeTag>::apply(iv, dest, domain, rbc.rho, rbc.U);
                break;
            case lbmBCtype::NoSlip:
                NoSlipKernel<LatticeTag>::apply(iv, dest, domain);
                break;
            case lbmBCtype::Inflow:
                InflowKernel<LatticeTag>::apply(iv, dest, domain, rbc.rho, rbc.U);
                break;
            case lbmBCtype::Outflow:
                OutflowKernel<LatticeTag>::apply(iv, dest, domain);
                break;
            case lbmBCtype::Periodic:
                return;
            default:
                amrex::Abort("BCManager: unrecognized lbmBCtype");
                break;
        }
    }

    void setFaceBC(int dir, int bound, lbmBCtype kind) {
        m_face_bc[faceIndex(dir, bound)] = kind;
    }
    void setFaceDensity(int dir, int bound, amrex::Real rho) {
        m_face_rho_wall[faceIndex(dir, bound)] = rho;
    }
    void setFaceVel(int dir, int bound, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Uwall) {
        m_face_vel[faceIndex(dir, bound)] = Uwall;
    }

    // Must be called after all setFace* and before use on device
    void finalize();

private:
    amrex::GpuArray<lbmBCtype, 2 * AMREX_SPACEDIM> m_face_bc{};
    amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM> m_face_rho_wall{};
    amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 2 * AMREX_SPACEDIM> m_face_vel{};
    amrex::GpuArray<ResolvedBC, detail::numCodes()> m_resolved{};
};

lbmBCtype parselbmBCtype(const std::string& name);

amrex::Vector<amrex::BCRec> makeBCRecVector(
    const std::array<std::string, 2 * AMREX_SPACEDIM>& names, int ncomp);

} // namespace muphfasa

#endif // MUPHFASA_BOUNDARY_CONDITIONS_H
