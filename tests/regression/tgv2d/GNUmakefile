# GNUmakefile for the Taylor-Green vortex regression test
# (tests/regression/tgv2d/). Builds test_tgv2d directly, without going
# through CMake.
#
# Usage:
#   make                    # build test_tgv2d (2-D, no MPI, by default)
#   ./test_tgv2d2d.gnu.ex inputs
#
# The InitData.cpp listed below resolves to THIS directory's own copy (the
# Taylor-Green initial condition), not src/Simulation/InitData.cpp's default
# quiescent start: Make.rules searches the current directory before any
# VPATH_LOCATIONS entry, so a same-named local file always wins.
#
# Only DIM and USE_EB are overridden below -- everything else (DEBUG,
# USE_MPI, USE_OMP, USE_CUDA, USE_HIP) already matches AMReX's own default.

AMREX_HOME ?= $(CURDIR)/../../../external/amrex

COMP ?= gnu
DIM  ?= 2
USE_EB ?= TRUE

BL_NO_FORT = TRUE

EBASE = test_tgv2d

SRC_DIR := ../../../src

VPATH_LOCATIONS   += $(SRC_DIR)/LBM $(SRC_DIR)/Geometry $(SRC_DIR)/IO $(SRC_DIR)/Simulation
INCLUDE_LOCATIONS += $(SRC_DIR) $(SRC_DIR)/LBM $(SRC_DIR)/Geometry $(SRC_DIR)/IO $(SRC_DIR)/Simulation

CEXE_sources += test_tgv2d.cpp

# Solver sources, excluding main.cpp (this test defines its own main()).
CEXE_sources += LatticeDescriptor.cpp
CEXE_sources += Macroscopic.cpp
CEXE_sources += Equilibrium.cpp
CEXE_sources += Streaming.cpp
CEXE_sources += Collision.cpp
CEXE_sources += BoundaryConditions.cpp
CEXE_sources += Units.cpp
CEXE_sources += GeometryHandler.cpp
CEXE_sources += InputParser.cpp
CEXE_sources += Checkpoint.cpp
CEXE_sources += Plotfile.cpp
CEXE_sources += SimulationSetup.cpp
CEXE_sources += SimulationFactory.cpp
CEXE_sources += InitData.cpp

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

Pdirs := Base Boundary AmrCore EB
Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
