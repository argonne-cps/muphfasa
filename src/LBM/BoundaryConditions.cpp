// BoundaryConditions.cpp — Explicit instantiations and resolveBC impl.

#include "BoundaryConditions.H"
#include "LatticeDescriptor.H"

namespace muphfasa {

lbmBCtype parselbmBCtype(const std::string& name)
{
    if (name == "MovingWall")  { return lbmBCtype::MovingWall;  }
    if (name == "NoSlip")      { return lbmBCtype::NoSlip;  }
    if (name == "Inflow")      { return lbmBCtype::Inflow;  }
    if (name == "Outflow")     { return lbmBCtype::Outflow; }
    if (name == "PressureOutlet") { return lbmBCtype::PressureOutlet; }
    amrex::Abort("parselbmBCtype: unrecognized BC name \"" + name +
                 "\" (expected MovingWall, NoSlip, Inflow, Outflow, or PressureOutlet; Periodic faces "
                 "must be filtered out before calling this)");
    return lbmBCtype::NoSlip;
}

amrex::Vector<amrex::BCRec> makeBCRecVector(
    const std::array<std::string, 2 * AMREX_SPACEDIM>& names, int ncomp)
{
    amrex::BCRec bcr;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const std::string& lo_name = names[faceIndex(dir, -1)];
        const std::string& hi_name = names[faceIndex(dir, 1)];

        bcr.setLo(dir, (lo_name == "Periodic") ? amrex::BCType::int_dir
                                               : kCustomBCSentinel);
        bcr.setHi(dir, (hi_name == "Periodic") ? amrex::BCType::int_dir
                                               : kCustomBCSentinel);
    }
    return amrex::Vector<amrex::BCRec>(ncomp, bcr);
}

bool hasConflictingBC(const amrex::Vector<lbmBCtype>& face_types) noexcept {
    bool conflict = false;

    // at most one of Inflow / Outflow / PressureOutlet per edge/corner
    bool has_inflow = false, has_outflow = false, has_pressure = false;
    for (auto t : face_types) {
        if (t == lbmBCtype::Inflow) has_inflow = true;
        if (t == lbmBCtype::Outflow) has_outflow = true;
        if (t == lbmBCtype::PressureOutlet) has_pressure = true;
    }
    conflict = (has_inflow + has_outflow + has_pressure) > 1;

    // check two moving walls (currently not supported)
    int nMoving = 0;
    for (auto t : face_types) {
        if (t == lbmBCtype::MovingWall) nMoving++;
    }
    if (nMoving > 1) conflict = true;

    return conflict;
}

ResolvedFaceBC resolveBC(
    const amrex::Vector<lbmBCtype>& face_types,
    const amrex::Vector<amrex::Real>& face_rho,
    const amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>>& face_U,
    const amrex::Vector<std::string>& face_names)
{
    if (hasConflictingBC(face_types)) {
        std::string msg = "resolveBC: conflicting bound conditions at same edge/corner (faces:";
        for (size_t i = 0; i < face_names.size(); ++i) msg += " " + face_names[i];
        msg += ")";
        amrex::Abort(msg);
    }

    // Find max rank
    int max_rank = -1;
    for (auto t : face_types) max_rank = std::max(max_rank, bcRank(t));

    lbmBCtype winner{};
    for (auto t : face_types) if (bcRank(t) == max_rank) { winner = t; break; }

    // Average rho and U over faces of winning type only
    amrex::Real sum_rho = 0;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> sum_U{AMREX_D_DECL(0.,0.,0.)};
    int count = 0;
    for (size_t i = 0; i < face_types.size(); ++i) {
        if (face_types[i] == winner) {
            sum_rho += face_rho[i];
            for (int d = 0; d < AMREX_SPACEDIM; ++d) sum_U[d] += face_U[i][d];
            ++count;
        }
    }
    ResolvedFaceBC out;
    out.type = winner;
    if (count > 0) {
        out.rho = sum_rho / count;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) out.U[d] = sum_U[d] / count;
    }
    return out;
}

template <typename LatticeTag>
void BCManager<LatticeTag>::finalize()
{
    // Initialize all resolved entries to Periodic — purely-periodic ghosts
    // are never looked up (operator() zeroes periodic dims) and purely-periodic
    // codes should remain Periodic, not NoSlip.
    for (int i = 0; i < detail::numCodes(); ++i) {
        m_resolved[i].type = lbmBCtype::Periodic;
        m_resolved[i].rho = 1.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) m_resolved[i].U[d] = 0;
    }

    // Enumerate codes
    // For each dim, code[d] in {-1,0,1}
    int total = detail::numCodes();
    for (int idx = 0; idx < total; ++idx) {
        // Decode idx to code
        amrex::IntVect code{AMREX_D_DECL(0,0,0)};
        int tmp = idx;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            code[d] = (tmp % 3) - 1;
            tmp /= 3;
        }
        bool all_zero = true;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) if (code[d] != 0) all_zero = false;
        if (all_zero) continue;

        // Gather faces for nonzero dims, skipping Periodic faces (mirrors
        // operator()'s zeroing of periodic dims; e.g. a corner with one
        // periodic side resolves to the remaining wall face).
        amrex::Vector<lbmBCtype> types;
        amrex::Vector<amrex::Real> rhos;
        amrex::Vector<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>> Us;
        amrex::Vector<std::string> names;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (code[d] != 0) {
                int fi = faceIndex(d, code[d]);
                if (m_face_bc[fi] == lbmBCtype::Periodic) continue;
                types.push_back(m_face_bc[fi]);
                rhos.push_back(m_face_rho_wall[fi]);
                Us.push_back(m_face_vel[fi]);
                // name for error message
                std::string nm = (code[d] < 0 ? "lo" : "hi");
                nm = std::string("d")+std::to_string(d)+nm;
                names.push_back(nm);
            }
        }
        // If all contributing faces were periodic, leave as Periodic
        // (purely periodic ghost — FillBoundary handled it).
        if (types.empty()) continue;

        ResolvedFaceBC r = resolveBC(types, rhos, Us, names);
        m_resolved[idx].type = r.type;
        m_resolved[idx].rho = r.rho;
        m_resolved[idx].U = r.U;
    }
}

#if (AMREX_SPACEDIM == 2)
template class BCManager<D2Q9>;
#else
template class BCManager<D3Q15>;
template class BCManager<D3Q19>;
template class BCManager<D3Q27>;
#endif

} // namespace muphfasa
