#!/bin/bash
# Script that packages libgpucon as a deb.
#
# Author: Roland Boubela <roland.boubela@meduniwien.ac.at>
# License: GPL

if [ ! $(id -u) ]; then echo "no." && exit; fi
if [ ! $(command -v fpm ) ]; then echo "fpm is missing (try 'gem install fpm')" && exit; fi

VERSION=0.0.1
TMPDIR=$(mktemp -d)

cd gpucon
./configure
make
make install DESTDIR=$TMPDIR
mkdir -p $TMPDIR/usr/local/lib/gpucon
mv $TMPDIR/lib $TMPDIR/usr/local/lib/gpucon
mv $TMPDIR/include $TMPDIR/usr/local/lib/gpucon
cd ..

sudo fpm -s dir -t deb -n gpucon -v $VERSION -C $TMPDIR \
  -p ../../deb/gpucon-$VERSION.deb -d "cuda-7-5" \
  --maintainer "Roland Boubela <roland.boubela@meduniwien.ac.at>" \
  usr/local/lib/gpucon/lib usr/local/lib/gpucon/include 


