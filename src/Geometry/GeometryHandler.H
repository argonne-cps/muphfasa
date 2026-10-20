// GeometryHandler.H — Interface for setting up the AMReX Geometry and,
// optionally, the Embedded Boundary (EB) description.
// Wraps BoxArray / DistributionMapping creation and exposes query helpers
// that the rest of the solver uses without touching AMReX internals directly.

#ifndef MUPHFASA_GEOMETRY_HANDLER_H
#define MUPHFASA_GEOMETRY_HANDLER_H

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <memory>
#include <string>

namespace muphfasa {

class InputParser;  // forward declaration

class GeometryHandler
{
public:
    explicit GeometryHandler(const InputParser& params);

    // ── AMReX objects ────────────────────────────────────────────────────
    const amrex::Geometry&            geometry()  const noexcept { return m_geom; }
    const amrex::BoxArray&            boxArray()  const noexcept { return m_ba;   }
    const amrex::DistributionMapping& distMap()   const noexcept { return m_dm;   }

    // ── Embedded boundary ────────────────────────────────────────────────
    bool hasEB() const noexcept { return m_has_eb; }

    // Initialise the EB geometry from the geometry type string (e.g. "sphere",
    // "cylinder", "stl").  Must be called before buildBoxArray().
    void initEB(const std::string& geom_type);

    // ── Grid construction ────────────────────────────────────────────────
    // Build BoxArray and DistributionMapping from the domain and max_grid_size.
    void buildBoxArray(int max_grid_size);

    // ── Physical domain extents ──────────────────────────────────────────
    amrex::RealBox physDomain() const noexcept;

private:
    amrex::Geometry            m_geom;
    amrex::BoxArray            m_ba;
    amrex::DistributionMapping m_dm;
    bool                       m_has_eb = false;
};

} // namespace muphfasa

#endif // MUPHFASA_GEOMETRY_HANDLER_H
