// test_lattice.cpp — Unit tests for LatticeDescriptor specialisations.
// tests:
//
//    Weights sum to 1.
//    Opposite indices properly negate
//    Each velocity is unique
//    Proper isotropy is present

#include <AMReX.H>
#include <AMReX_Print.H>

#include "LBM/LatticeDescriptor.H"

#include <cmath>
#include <set>
#include <stdexcept>
#include <tuple>

namespace {

template <typename LatticeTag>
void testWeightSum()
{
    double s = 0.0;
    for (int q = 0; q < LatticeTag::Q; ++q) s += LatticeTag::weights(q);
    if (std::abs(s - 1.0) > 1e-12)
        throw std::runtime_error(std::string("Weight sum != 1 for lattice Q=")
                                 + std::to_string(LatticeTag::Q));
    amrex::Print() << "  [PASS] weight sum = 1  (Q=" << LatticeTag::Q << ")\n";
}

template <typename LatticeTag>
void testOppositeInvolution()
{
    for (int q = 0; q < LatticeTag::Q; ++q) {
        int r = LatticeTag::opposite(q);
        if (r < 0 || r >= LatticeTag::Q)
            throw std::runtime_error("opposite[] out of range at q=" + std::to_string(q));
        if (LatticeTag::opposite(r) != q)
            throw std::runtime_error("opposite[] is not an involution at q="
                                     + std::to_string(q));
        if (std::abs(LatticeTag::weights(r) - LatticeTag::weights(q)) > 1e-14)
            throw std::runtime_error("opposite weights differ at q=" + std::to_string(q));
    }
    amrex::Print() << "  [PASS] opposite[] involution  (Q=" << LatticeTag::Q << ")\n";
}

template <typename LatticeTag>
void testOppositeNegation()
{
    for (int q = 0; q < LatticeTag::Q; ++q) {
        int r = LatticeTag::opposite(q);
        if (LatticeTag::cx(r) != -LatticeTag::cx(q) || LatticeTag::cy(r) != -LatticeTag::cy(q))
            throw std::runtime_error("opposite[] does not negate velocity at q=" + std::to_string(q));
        if constexpr (LatticeTag::D == 3) {
            if (LatticeTag::cz(r) != -LatticeTag::cz(q))
                throw std::runtime_error("opposite[] does not negate cz at q=" + std::to_string(q));
        }
    }
    amrex::Print() << "  [PASS] opposite[] negation  (Q=" << LatticeTag::Q << ")\n";
}

template <typename LatticeTag>
void testUniqueness()
{
    std::set<std::tuple<int,int,int>> seen;
    for (int q = 0; q < LatticeTag::Q; ++q) {
        int cz = 0;
        if constexpr (LatticeTag::D == 3) cz = LatticeTag::cz(q);
        auto key = std::make_tuple(LatticeTag::cx(q), LatticeTag::cy(q), cz);
        if (seen.count(key))
            throw std::runtime_error("duplicate velocity at q=" + std::to_string(q));
        seen.insert(key);
    }
    amrex::Print() << "  [PASS] unique velocities  (Q=" << LatticeTag::Q << ")\n";
}

template <typename LatticeTag>
void testIsotropy()
{
    double sxx=0, syy=0, szz=0, sxy=0, sxz=0, syz=0;
    double sx=0, sy=0, sz=0;
    for (int q = 0; q < LatticeTag::Q; ++q) {
        double w = LatticeTag::weights(q);
        int cz = 0;
        if constexpr (LatticeTag::D == 3) cz = LatticeTag::cz(q);
        sx += w * LatticeTag::cx(q); sy += w * LatticeTag::cy(q); sz += w * cz;
        sxx += w * LatticeTag::cx(q) * LatticeTag::cx(q);
        syy += w * LatticeTag::cy(q) * LatticeTag::cy(q);
        szz += w * cz * cz;
        sxy += w * LatticeTag::cx(q) * LatticeTag::cy(q);
        sxz += w * LatticeTag::cx(q) * cz;
        syz += w * LatticeTag::cy(q) * cz;
    }
    if (std::abs(sx) > 1e-12 || std::abs(sy) > 1e-12 || std::abs(sz) > 1e-12)
        throw std::runtime_error("first moment !=0 Q=" + std::to_string(LatticeTag::Q));
    if (std::abs(sxx - 1./3.) > 1e-12 || std::abs(syy - 1./3.) > 1e-12)
        throw std::runtime_error("second moment diag !=1/3 Q=" + std::to_string(LatticeTag::Q));
    if constexpr (LatticeTag::D == 3) if (std::abs(szz - 1./3.) > 1e-12)
        throw std::runtime_error("second moment szz !=1/3 Q=" + std::to_string(LatticeTag::Q));
    if (std::abs(sxy) > 1e-12 || std::abs(sxz) > 1e-12 || std::abs(syz) > 1e-12)
        throw std::runtime_error("second moment off-diag !=0 Q=" + std::to_string(LatticeTag::Q));
    amrex::Print() << "  [PASS] isotropy (1st/2nd moments)  (Q=" << LatticeTag::Q << ")\n";
}

template <typename LatticeTag>
void testSoundSpeed()
{
        if (std::abs(LatticeTag::cs2 - 1./3.) > 1e-12)
            throw std::runtime_error("cs2 !=1/3");
        amrex::Print() << "  [PASS] cs2==1/3\n";
}

template <typename LatticeTag>
void runAll(const char* name)
{
    amrex::Print() << " Testing " << name << " (D=" << LatticeTag::D << " Q=" << LatticeTag::Q << ")\n";
    testWeightSum<LatticeTag>();
    testOppositeInvolution<LatticeTag>();
    testOppositeNegation<LatticeTag>();
    testUniqueness<LatticeTag>();
    testIsotropy<LatticeTag>();
    testSoundSpeed<LatticeTag>();
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Lattice descriptor tests ===\n";

        runAll<muphfasa::D2Q9>("D2Q9");
        runAll<muphfasa::D3Q15>("D3Q15");
        runAll<muphfasa::D3Q19>("D3Q19");
        runAll<muphfasa::D3Q27>("D3Q27");

        amrex::Print() << "All lattice tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
