
#include <Rcpp.h>
#include <gpu_matmul_div.h>

using namespace Rcpp;

RcppExport SEXP r_gpu_matmul_div(SEXP rA, SEXP rB, SEXP rC, SEXP rA_sd, SEXP rB_sd, SEXP rgpuID) {
   
  NumericMatrix A(rA);
  NumericMatrix B(rB);
  NumericMatrix C(rC);
  NumericVector A_sd(rA_sd);
  NumericVector B_sd(rB_sd);
  IntegerVector gpuID(rgpuID);

  gpu_matmul_div(A.begin(), B.begin(), C.begin(), A_sd.begin(), B_sd.begin(), A.nrow(), A.ncol(), B.ncol(), gpuID[0]);

  return R_NilValue;
}


