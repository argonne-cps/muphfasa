# GNUmakefile for the uniform-force regression test (tests/regression/uniform_force/).
# Builds test_uniform_force directly, without going through CMake.
#
# Usage:
#   make                      # test_uniform_force2d.gnu.ex (2-D, no MPI)
#   ./test_uniform_force2d.gnu.ex inputs
#
#   make DIM=3                # test_uniform_force3d.gnu.ex
#   ./test_uniform_force3d.gnu.ex inputs3d
#
# All the shared build logic lives in ../../../Make.muphfasa.

TOP := ../../..

DIM ?= 2

EBASE = test_uniform_force

CEXE_sources += test_uniform_force.cpp

include $(TOP)/Make.muphfasa
