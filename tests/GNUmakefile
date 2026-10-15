# GNUmakefile for the muphfasa unit test suite.
# Mirrors the top-level GNUmakefile options; run from this directory or pass
# the target as `make -C tests`.

AMREX_HOME ?= $(CURDIR)/../external/amrex
COMP  ?= gnu
DIM   ?= 3
DEBUG ?= TRUE

USE_MPI  ?= TRUE
USE_OMP  ?= FALSE
USE_CUDA ?= FALSE
USE_HIP  ?= FALSE

EBASE = muphfasa_unit_tests

VPATH_LOCATIONS   += unit
VPATH_LOCATIONS   += ../src
VPATH_LOCATIONS   += ../src/LBM
VPATH_LOCATIONS   += ../src/Geometry
VPATH_LOCATIONS   += ../src/IO
VPATH_LOCATIONS   += ../src/Simulation

INCLUDE_LOCATIONS += ../src
INCLUDE_LOCATIONS += ../src/LBM
INCLUDE_LOCATIONS += ../src/Geometry
INCLUDE_LOCATIONS += ../src/IO
INCLUDE_LOCATIONS += ../src/Simulation

CEXE_sources += test_lattice.cpp
CEXE_sources += test_macroscopic.cpp
CEXE_sources += test_equilibrium.cpp
CEXE_sources += test_units.cpp

# Include solver sources (exclude main.cpp)
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

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
