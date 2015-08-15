#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([sparkniftireader], [0.1], [roland.boubela@meduniwien.ac.at])
AC_CONFIG_SRCDIR([src/sparkniftireader.c])
# AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE

# Checks for programs.
AC_PROG_CC

DESTDIR=/usr/local/lib/sparkniftireader
AC_SUBST(DESTDIR)

# Checks for libraries.
NIFTI_INCLUDE="-I/usr/include/nifti"
NIFTI_CFLAGS="-fpic"
NIFTI_LDFLAGS="-shared "
NIFTI_LIBS="-lniftiio -lznz -lz -lm"

CC=gcc

AC_SUBST(CC)
AC_SUBST(NIFTI_LIBS)
AC_SUBST(NIFTI_LDFLAGS)
AC_SUBST(NIFTI_CFLAGS)
AC_SUBST(NIFTI_INCLUDE)

# Checks for header files.
AC_CHECK_FILE([/usr/include/nifti/nifti1_io.h], [HAS_NIFTI_HEADER="yes"])
if test -z "${HAS_NIFTI_HEADER}"; then
	AC_MSG_ERROR("cannot find nifti1_io.h; sudo apt-get install libnifti-dev")
else
	AC_MSG_RESULT("libnifti header found")
fi

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL

# Checks for library functions.

AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
