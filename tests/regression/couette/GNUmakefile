# GNUmakefile for the Couette regression test (tests/regression/couette/).
# Builds test_couette directly, without going through CMake.
#
# Usage:
#   make                      # build test_couette (2-D, no MPI, by default)
#   ./test_couette2d.gnu.ex inputs
#
#   make DIM=3                # builds test_couette3d.gnu.ex
#   ./test_couette3d.gnu.ex inputs3d
#
# All the shared build logic (AMReX modules, solver sources, Make.rules)
# lives in ../../../Make.muphfasa; see that file for defaults.

TOP := ../../..

DIM ?= 2

EBASE = test_couette

CEXE_sources += test_couette.cpp

include $(TOP)/Make.muphfasa
