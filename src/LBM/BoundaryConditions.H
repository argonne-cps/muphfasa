// BoundaryConditions.H — BCManager<LatticeTag>: the GPU-callable functor
// plugged into amrex::PhysBCFunct<amrex::GpuBndryFuncFab<...>> that fills
// ghost cells for the LBM distribution MultiFab. See BC_implementation_plan.md
// (Option 2.1) for the full design rationale.
//
// Contract: this functor is only ever invoked by AMReX for ghost cells (`iv`
// outside the valid box). Its job is to fill that ghost cell with whatever
// value makes the next stream() produce the right boundary physics at the
// adjacent fluid node — it never corrects a valid interior cell directly.

#ifndef MUPHFASA_BOUNDARY_CONDITIONS_H
#define MUPHFASA_BOUNDARY_CONDITIONS_H

#include "BCTypes/MovingWall.H"
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

// Shared face-index convention: [xlo, xhi, ylo, yhi, zlo, zhi]. Used by
// both BCManager::setFaceBC/operator() and the registry/SimulationSetup
inline constexpr int faceIndex(int dir, int bound) noexcept {
    return 2 * dir + (bound < 0 ? 0 : 1);
}

// Our own dispatch tag — never amrex::BCType. Adding a BC means adding an
// enumerator here, nowhere else in amrex's namespace.
enum class lbmBCtype { MovingWall, NoSlip, Inflow, Outflow };

// The only amrex::BCType code our custom BCs ever use. Chosen from the
// `user_*` range AMReX reserves for exactly this purpose; never ext_dir/
// foextrap/hoextrap/hoextrapcc/reflect_even/reflect_odd — those drive
// AMReX's own generic fill/interpolation machinery, which we don't want
// touching this field.
inline constexpr int kCustomBCSentinel = amrex::BCType::user_1;
static_assert(kCustomBCSentinel != amrex::BCType::int_dir &&
              kCustomBCSentinel != amrex::BCType::ext_dir &&
              kCustomBCSentinel != amrex::BCType::foextrap &&
              kCustomBCSentinel != amrex::BCType::hoextrap &&
              kCustomBCSentinel != amrex::BCType::hoextrapcc &&
              kCustomBCSentinel != amrex::BCType::reflect_even &&
              kCustomBCSentinel != amrex::BCType::reflect_odd,
              "custom BC sentinel must not collide with a semantically-defined amrex::BCType");

template <typename LatticeTag>
class BCManager
{
public:
    // numcomp/dcomp are not looped here: one BCRec (at bcomp) represents the
    // whole LatticeTag::Q-component field, since a wall applies to all
    // populations together, not per-component.
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
        const auto dom_lo = domain.smallEnd();
        const auto dom_hi = domain.bigEnd();

        // loop over directions
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            int bound = 0;
            int amrex_code = amrex::BCType::int_dir;

            if (iv[dir] < dom_lo[dir]) {
                bound = -1;
                amrex_code = bcr[bcomp].lo(dir);
            } else if (iv[dir] > dom_hi[dir]) {
                bound = 1;
                amrex_code = bcr[bcomp].hi(dir);
            } else {
                continue; // not a ghost cell in this direction
            }

            if (amrex_code == amrex::BCType::int_dir) {
                continue; // periodic — FillBoundary already handled it
            }

            // amrex_code is just our sentinel here; the real kind comes
            // from our own table.
            switch (m_face_bc[faceIndex(dir, bound)]) {
                case lbmBCtype::MovingWall:
                    MovingWallKernel<LatticeTag>::apply(iv, dest, dir, bound,
                                                        m_face_rho_wall[faceIndex(dir, bound)],
                                                        m_face_Uwall[faceIndex(dir, bound)]);
                    break;

                case lbmBCtype::NoSlip:
                    NoSlipKernel<LatticeTag>::apply(iv, dest, dir, bound);
                    break;

                case lbmBCtype::Inflow:
                    InflowKernel<LatticeTag>::apply(iv, dest, dir, bound);
                    break;

                case lbmBCtype::Outflow:
                    OutflowKernel<LatticeTag>::apply(iv, dest, dir, bound);
                    break;

                default:
                    amrex::Abort("BCManager: unrecognized lbmBCtype encountered while filling boundary ghost cells");
                    break;
            }
        }
    }

    // Topology: which face has which BC kind. Set once at setup
    void setFaceBC(int dir, int bound, lbmBCtype kind) {
        m_face_bc[faceIndex(dir, bound)] = kind;
    }

    void setFaceDensity(int dir, int bound, amrex::Real rho) {
        m_face_rho_wall[faceIndex(dir, bound)] = rho;
    }

    void setFaceUwall(int dir, int bound, amrex::Real Uwall) {
        m_face_Uwall[faceIndex(dir, bound)] = Uwall;
    }

private:
    amrex::GpuArray<lbmBCtype, 2 * AMREX_SPACEDIM> m_face_bc{};
    amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM> m_face_rho_wall{};
    amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM> m_face_Uwall{};
};

// Face name -> our own dispatch enum. 
lbmBCtype parselbmBCtype(const std::string& name);

// Face names -> the amrex::BCRec actually handed to PhysBCFunct. Every
// entry in the returned vector is identical and only ever carries 
//int_dir ("Periodic") or kCustomBCSentinel (everything else) 
// `ncomp` should be LatticeTag::Q;
amrex::Vector<amrex::BCRec> makeBCRecVector(
    const std::array<std::string, 2 * AMREX_SPACEDIM>& names, int ncomp);

} // namespace muphfasa

#endif // MUPHFASA_BOUNDARY_CONDITIONS_H
