# GNUmakefile for the Poiseuille regression test (tests/regression/poiseuille/).
# Builds test_poiseuille directly, without going through CMake.
#
# Usage:
#   make                      # test_poiseuille2d.gnu.ex (2-D, no MPI)
#   ./test_poiseuille2d.gnu.ex inputs
#
#   make DIM=3                # test_poiseuille3d.gnu.ex
#   ./test_poiseuille3d.gnu.ex inputs3d
#
# All the shared build logic lives in ../../../Make.muphfasa.

TOP := ../../..

DIM ?= 2

EBASE = test_poiseuille

CEXE_sources += test_poiseuille.cpp

include $(TOP)/Make.muphfasa
