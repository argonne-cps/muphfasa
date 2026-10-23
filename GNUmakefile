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
#
# All the shared build logic (AMReX modules, solver sources, Make.rules)
# lives in Make.muphfasa; this file only supplies the executable's own
# name and its own translation unit (main.cpp).

TOP := .

COMP ?= gnu
DIM  ?= 3

DEBUG    ?= FALSE
USE_MPI  ?= TRUE
USE_OMP  ?= FALSE
USE_CUDA ?= FALSE
USE_HIP  ?= FALSE

EBASE = muphfasa

CEXE_sources += main.cpp

include $(TOP)/Make.muphfasa
