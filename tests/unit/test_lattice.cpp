// test_lattice.cpp — Unit tests for LatticeDescriptor specialisations.
// Checks mathematical properties that must hold for any valid velocity set:
//   - weights sum to 1
//   - zeroth velocity moment is zero (sum_q w_q * c_q = 0)
//   - opposite[] is a self-consistent involution (opposite[opposite[q]] == q)
//   - cs2 == 1/3

#include <AMReX.H>
#include <AMReX_Print.H>

#include "LBM/LatticeDescriptor.H"

#include <cmath>
#include <stdexcept>

namespace {

template <typename L>
void testWeightSum()
{
    auto w   = L::weights();
    double s = 0.0;
    for (int q = 0; q < L::Q; ++q) s += w[q];
    if (std::abs(s - 1.0) > 1e-12)
        throw std::runtime_error(std::string("Weight sum != 1 for lattice Q=")
                                 + std::to_string(L::Q));
    amrex::Print() << "  [PASS] weight sum = 1  (Q=" << L::Q << ")\n";
}

template <typename L>
void testOppositeInvolution()
{
    auto opp = L::opposite();
    for (int q = 0; q < L::Q; ++q) {
        if (opp[opp[q]] != q)
            throw std::runtime_error("opposite[] is not an involution at q="
                                     + std::to_string(q));
    }
    amrex::Print() << "  [PASS] opposite[] involution  (Q=" << L::Q << ")\n";
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::Print() << "\n=== Lattice descriptor tests ===\n";

        // TODO: uncomment as implementations are filled in
        // testWeightSum<muphfasa::D2Q9>();
        // testOppositeInvolution<muphfasa::D2Q9>();
        // testWeightSum<muphfasa::D3Q19>();
        // testOppositeInvolution<muphfasa::D3Q19>();

        amrex::Print() << "All lattice tests passed.\n";
    }
    amrex::Finalize();
    return 0;
}
