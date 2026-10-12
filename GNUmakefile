# GNUmakefile for muphfasa (uses the AMReX GNUMake build system).
#
# Common usage:
#   make DIM=2                       2-D solver, MPI, no GPU
#   make DIM=3                       3-D solver, MPI, no GPU
#   make DIM=2 DEBUG=TRUE            debug build
#   make DIM=2 USE_MPI=FALSE         single-process build
#   make DIM=2 USE_OMP=TRUE          OpenMP-threaded build
#   make DIM=2 USE_CUDA=TRUE         NVIDIA GPU build
#   make DIM=2 USE_HIP=TRUE          AMD GPU build

AMREX_HOME ?= $(CURDIR)/external/amrex

# Compiler collection: gnu | intel | cray | pgi | llvm
COMP ?= gnu

# Dimensionality (2 or 3)
DIM ?= 3

# Build flags
DEBUG    ?= FALSE
USE_MPI  ?= TRUE
USE_OMP  ?= FALSE
USE_CUDA ?= FALSE
USE_HIP  ?= FALSE
USE_EB   ?= TRUE

# No Fortran sources in this project.
BL_NO_FORT = TRUE

EBASE = muphfasa

# ── Source locations ───────────────────────────────────────────────────────────
VPATH_LOCATIONS   += src
VPATH_LOCATIONS   += src/LBM
VPATH_LOCATIONS   += src/Geometry
VPATH_LOCATIONS   += src/IO
VPATH_LOCATIONS   += src/Simulation

INCLUDE_LOCATIONS += src
INCLUDE_LOCATIONS += src/LBM
INCLUDE_LOCATIONS += src/Geometry
INCLUDE_LOCATIONS += src/IO
INCLUDE_LOCATIONS += src/Simulation

# ── Translation units ──────────────────────────────────────────────────────────
CEXE_sources += main.cpp
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

# ── AMReX library modules ────────────────────────────────────────────────────────
# Base    : core data structures (MultiFab, BoxArray, Geometry, ParmParse, ...)
# Boundary: ghost-cell/boundary-fill infrastructure, used by the EB module
# AmrCore : Interpolater base class, used by the EB module
# EB      : embedded-boundary geometry (AMReX_EB2.H, AMReX_EBFabFactory.H)
Pdirs := Base Boundary AmrCore EB
Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
