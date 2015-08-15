#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT(gpucon, 0.1, roland.boubela@meduniwien.ac.at)
AC_CONFIG_SRCDIR([src/gpu_matmul.h])
# AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE

# Checks for programs.
AC_PROG_CC
AC_PROG_RANLIB
AC_PROG_INSTALL

DESTDIR=/usr/local/lib/gpucon
AC_SUBST(DESTDIR)


# Checks for libraries.

# CUDA setup
CUDA_HOME=/usr/local/cuda
CUDA_CFLAGS="-Xcompiler -fpic -gencode arch=compute_35,code=sm_35 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_20,code=sm_20"
CUDA_CFLAGS="${CUDA_CFLAGS} -I{$CUDA_HOME}/include"
CUDA_LDFLAGS="-shared -L${CUDA_HOME}\lib64"
CUDA_LIBS="-lcuda -lcublas"

NVCC="nvcc"

AC_SUBST(NVCC)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(CUDA_CFLAGS)


# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.

AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT

