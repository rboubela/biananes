\name{fastCor-package}
\alias{fastCor-package}
\alias{fastCor}
\docType{package}
\title{
fastCor: Pearson Correlation on GPU
}
\description{
Provides an R interface to a fast and efficient implementation of the Pearson correlation
}
\details{
\tabular{ll}{
Package: \tab fastCor\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2014-05-23\cr
License: \tab GPL (>= 3.0)\cr
}
fastCor(A, gpuID = 0)
fastCor(A, B, gpuID = 1)
}
\author{
Roland Boubela

Maintainer: Roland Boubela <roland.boubela@meduniwien.ac.at>
}
\references{
}
\keyword{ package }
\seealso{
}
\examples{
cor.mat = fastCor(A, B)
}
