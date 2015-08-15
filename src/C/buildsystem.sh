#!/usr/bin/bash

cd sparkniftireader

# http://stackoverflow.com/questions/5387167/why-automake-fails-to-generate-makefile-in
autoscan
mv configure.scan configure.ac
(edit configure.ac)
autoconf # autroreconf?
(edit Makefile.am)
aclocal
automake --add-missing