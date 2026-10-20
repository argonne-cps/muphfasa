// SimulationFactory.H — Runtime -> compile-time lattice dispatch.
// The single place where InputParser::latticeType()'s runtime string is
// turned into a concrete LatticeTag template argument.

#ifndef MUPHFASA_SIMULATION_FACTORY_H
#define MUPHFASA_SIMULATION_FACTORY_H

#include <memory>

namespace muphfasa {

class InputParser;
class ISimulation;

// Reads params.latticeType() ("D2Q9" | "D3Q15" | "D3Q19" | "D3Q27") and
// constructs the matching SimulationImpl<LatticeTag>, returned behind the
// non-templated ISimulation interface. Aborts if the requested lattice
// doesn't match this build's AMREX_SPACEDIM.
std::unique_ptr<ISimulation> makeSimulation(const InputParser& params);

} // namespace muphfasa

#endif // MUPHFASA_SIMULATION_FACTORY_H
