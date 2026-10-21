# GNUmakefile for the Taylor-Green vortex regression test
# (tests/regression/tgv2d/). Builds test_tgv2d directly, without going
# through CMake.
#
# Usage:
#   make                    # build test_tgv2d (2-D, no MPI, by default)
#   ./test_tgv2d2d.gnu.ex inputs
#
# The InitData.cpp in this directory resolves to THIS directory's own copy
# (the Taylor-Green initial condition), not src/Simulation/InitData.cpp's
# default quiescent start: Make.rules searches the current directory before
# any VPATH_LOCATIONS entry, so a same-named local file always wins.
#
# All the shared build logic (AMReX modules, solver sources, Make.rules)
# lives in ../../../Make.muphfasa; see that file for defaults.

TOP := ../../..

DIM ?= 2

EBASE = test_tgv2d

CEXE_sources += test_tgv2d.cpp

include $(TOP)/Make.muphfasa
