# Top-level CMakeLists for muphfasa: an AMReX-based Lattice Boltzmann solver.
# Supports CPU (MPI + optional OpenMP) and GPU (CUDA, HIP) backends.
# AMReX is vendored as a git submodule under external/amrex and built from
# source via add_subdirectory(); mirrors the options exposed by GNUmakefile.

cmake_minimum_required(VERSION 3.20)
project(muphfasa VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Backend options ────────────────────────────────────────────────────────────
set(muphfasa_DIM 3 CACHE STRING "Dimensionality of the solver (1, 2, or 3)")
set_property(CACHE muphfasa_DIM PROPERTY STRINGS 1 2 3)

option(muphfasa_ENABLE_MPI  "Enable MPI distributed parallelism"   ON)
option(muphfasa_ENABLE_OMP  "Enable OpenMP threading"              OFF)
option(muphfasa_ENABLE_CUDA "Enable NVIDIA CUDA GPU backend"       OFF)
option(muphfasa_ENABLE_HIP  "Enable AMD HIP GPU backend"           OFF)
option(muphfasa_ENABLE_EB   "Enable AMReX embedded-boundary support" ON)

if(muphfasa_ENABLE_CUDA AND muphfasa_ENABLE_HIP)
    message(FATAL_ERROR "CUDA and HIP backends are mutually exclusive.")
endif()

if(muphfasa_ENABLE_CUDA)
    set(AMReX_GPU_BACKEND "CUDA" CACHE STRING "" FORCE)
elseif(muphfasa_ENABLE_HIP)
    set(AMReX_GPU_BACKEND "HIP" CACHE STRING "" FORCE)
else()
    set(AMReX_GPU_BACKEND "NONE" CACHE STRING "" FORCE)
endif()

# ── AMReX ─────────────────────────────────────────────────────────────────────
# Built from the vendored submodule at external/amrex; forward our backend
# choices to AMReX's own CMake options before pulling it in.
set(AMReX_SPACEDIM  ${muphfasa_DIM}       CACHE STRING "" FORCE)
set(AMReX_MPI       ${muphfasa_ENABLE_MPI}  CACHE BOOL   "" FORCE)
set(AMReX_OMP       ${muphfasa_ENABLE_OMP}  CACHE BOOL   "" FORCE)
set(AMReX_EB        ${muphfasa_ENABLE_EB}   CACHE BOOL   "" FORCE)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/amrex/CMakeLists.txt)
    message(FATAL_ERROR
        "external/amrex is empty. Run 'git submodule update --init --recursive'.")
endif()

add_subdirectory(external/amrex)

# ── Sources ───────────────────────────────────────────────────────────────────
set(MUPHFASA_SOURCES
    src/main.cpp
    src/LBM/LatticeDescriptor.cpp
    src/LBM/Macroscopic.cpp
    src/LBM/Equilibrium.cpp
    src/LBM/Streaming.cpp
    src/LBM/Collision.cpp
    src/LBM/BoundaryConditions.cpp
    src/LBM/Units.cpp
    src/Geometry/GeometryHandler.cpp
    src/IO/InputParser.cpp
    src/IO/Checkpoint.cpp
    src/IO/Plotfile.cpp
    src/Simulation/SimulationSetup.cpp
)

add_executable(muphfasa ${MUPHFASA_SOURCES})

target_include_directories(muphfasa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(muphfasa PRIVATE AMReX::amrex_${muphfasa_DIM}d)

if(AMReX_CUDA)
    setup_target_for_cuda_compilation(muphfasa)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
option(muphfasa_BUILD_TESTS "Build the validation and unit test suite" OFF)
if(muphfasa_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
