# GNUmakefile for muphfasa_inputinfo (util/input_info/), the standalone
# input-file inspector. Builds only what it needs (AMReX Base, Units.cpp,
# InputParser.cpp, Checkpoint.cpp); it does not use Make.muphfasa and compiles no solver code.
#
# Run it from this directory:
#   make DIM=2                 2-D build (inputs with 2 values per vector key)
#   make DIM=3                 3-D build (default)
#   make DIM=2 DEBUG=TRUE      debug build
#   make DIM=2 USE_MPI=TRUE    MPI build (not needed: the tool is serial)
#   make DIM=2 realclean       remove build products for this configuration
#
# The result is muphfasa_inputinfo<dim>d.<comp>[.<opts>].ex, e.g.
# muphfasa_inputinfo2d.gnu.ex. The dimension must match the input file being
# inspected, so build the one you need. Usage: ./muphfasa_inputinfo2d.gnu.ex <inputs>
# AMREX_HOME is taken from the environment if set, else external/amrex.

TOP := ../..

AMREX_HOME ?= $(TOP)/external/amrex

COMP ?= gnu
DIM  ?= 3

DEBUG    ?= FALSE
USE_MPI  ?= FALSE
USE_OMP  ?= FALSE
USE_CUDA ?= FALSE
USE_HIP  ?= FALSE
USE_SYCL ?= FALSE

EBASE = input_info

# No Fortran sources in this project.
BL_NO_FORT = TRUE

Blocs += $(TOP)/src/LBM $(TOP)/src/IO

CEXE_sources += input_info.cpp Units.cpp InputParser.cpp Checkpoint.cpp

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

# Base only: ParmParse, MultiFab and the other core types Units/InputParser use.
Bpack += $(AMREX_HOME)/Src/Base/Make.package

include $(Bpack)

VPATH_LOCATIONS   += $(Blocs)
INCLUDE_LOCATIONS += $(Blocs) $(TOP)/src

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
