#!/bin/bash
# Script that packages libsparkniftilib as a deb.
#
# Author: Roland Boubela <roland.boubela@meduniwien.ac.at>
# License: MIT

if [ ! $(id -u) ]; then echo "no." && exit; fi
if [ ! $(command -v fpm ) ]; then echo "fpm is missing (try 'gem install fpm')" && exit; fi

VERSION=0.0.2
TMPDIR=$(mktemp -d)

cd sparkniftireader
./configure
make
make install DESTDIR=$TMPDIR
mkdir -p $TMPDIR/usr/local/lib/sparkniftireader
mv $TMPDIR/lib $TMPDIR/usr/local/lib/sparkniftireader
cd ..

sudo fpm -s dir -t deb -n sparkniftireader -v $VERSION -C $TMPDIR \
  -p sparkniftireader-$VERSION.deb -d "libnifti-dev" \
  --maintainer "Roland Boubela <roland.boubela@meduniwien.ac.at>" \
  usr/local/lib/sparkniftireader/lib

