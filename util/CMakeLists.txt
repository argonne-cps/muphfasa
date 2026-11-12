# CMakeLists for muphfasa's standalone helper utilities (util/).
# Enabled with -Dmuphfasa_BUILD_UTIL=ON; needs only the unit converter,
# input parser and checkpoint header reader, not the full solver. GNUMake users: see
# util/input_info/GNUmakefile.

add_executable(muphfasa_inputinfo
    input_info/input_info.cpp
    ${CMAKE_SOURCE_DIR}/src/LBM/Units.cpp
    ${CMAKE_SOURCE_DIR}/src/IO/InputParser.cpp
    ${CMAKE_SOURCE_DIR}/src/IO/Checkpoint.cpp
)

target_include_directories(muphfasa_inputinfo PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(muphfasa_inputinfo PRIVATE AMReX::amrex_${muphfasa_DIM}d)

if(AMReX_CUDA)
    setup_target_for_cuda_compilation(muphfasa_inputinfo)
endif()
