# CMakeLists for the muphfasa test suite.
# Unit tests use the lightweight AMReX test infrastructure; regression tests
# run the full solver binary and compare output against reference data.

cmake_minimum_required(VERSION 3.20)

# ── Unit tests ────────────────────────────────────────────────────────────
# Each test_*.cpp defines its own main(), so each gets its own executable
# (the AMReX idiom) rather than being linked together. All of them link
# against the solver sources directly (excluding main.cpp) so the tests stay
# in sync with the main executable without needing a shared library.
set(SOLVER_SOURCES_NO_MAIN
    ${CMAKE_SOURCE_DIR}/src/LBM/LatticeDescriptor.cpp
    ${CMAKE_SOURCE_DIR}/src/LBM/Macroscopic.cpp
    ${CMAKE_SOURCE_DIR}/src/LBM/Equilibrium.cpp
    ${CMAKE_SOURCE_DIR}/src/LBM/Streaming.cpp
    ${CMAKE_SOURCE_DIR}/src/LBM/Collision.cpp
    ${CMAKE_SOURCE_DIR}/src/LBM/BoundaryConditions.cpp
    ${CMAKE_SOURCE_DIR}/src/LBM/Units.cpp
    ${CMAKE_SOURCE_DIR}/src/Geometry/GeometryHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/IO/InputParser.cpp
    ${CMAKE_SOURCE_DIR}/src/IO/Checkpoint.cpp
    ${CMAKE_SOURCE_DIR}/src/IO/Plotfile.cpp
    ${CMAKE_SOURCE_DIR}/src/Simulation/SimulationSetup.cpp
    ${CMAKE_SOURCE_DIR}/src/Simulation/SimulationFactory.cpp
)

set(UNIT_TEST_NAMES
    test_lattice
    test_macroscopic
    test_equilibrium
    test_units
)

foreach(test_name IN LISTS UNIT_TEST_NAMES)
    add_executable(${test_name} unit/${test_name}.cpp ${SOLVER_SOURCES_NO_MAIN})

    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)

    target_link_libraries(${test_name} PRIVATE AMReX::amrex_${muphfasa_DIM}d)

    if(AMReX_CUDA)
        setup_target_for_cuda_compilation(${test_name})
    endif()

    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ── Regression tests ──────────────────────────────────────────────────────
# Each regression test runs the solver binary with a reference input file and
# checks that key diagnostics (e.g. error norms) fall below a tolerance.

function(add_regression_test name)
    add_test(
        NAME regression_${name}
        COMMAND $<TARGET_FILE:muphfasa> inputs
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/regression/${name}
    )
endfunction()

add_regression_test(poiseuille)
add_regression_test(lid_driven_cavity)

# Regression tests that need their own analysis code (e.g. comparing against
# an analytic solution) instead of just checking the solver runs to
# completion: build a dedicated executable, same idiom as the unit tests
# above, and run it from its regression/<name>/ directory so it can find its
# own "inputs" file via a relative path.
function(add_regression_analysis_test name)
    add_executable(test_${name} regression/${name}/test_${name}.cpp ${SOLVER_SOURCES_NO_MAIN})

    target_include_directories(test_${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_${name} PRIVATE AMReX::amrex_${muphfasa_DIM}d)

    if(AMReX_CUDA)
        setup_target_for_cuda_compilation(test_${name})
    endif()

    add_test(
        NAME regression_${name}
        COMMAND $<TARGET_FILE:test_${name}> inputs
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/regression/${name}
    )
endfunction()

add_regression_analysis_test(couette)
