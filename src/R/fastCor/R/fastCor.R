fastCor <- function(A, B = NULL, gpuID = 0) {

  if (ncol(A) > 24000) {
    stop("for matrices with more than 24000 colums use bigFastCor")
  }

  if (!is.null(B) ) {
    if(ncol(B) > 24000) {
      stop("for matrices with more than 24000 colums use bigFastCor")
    }
  }
  
  A.sd <- apply(A, 2, sd)
  A.centered <- scale(A, scale = FALSE, center = TRUE)
 
  if (is.null(B)) {
    B <- A
    B.sd <- A.sd
    B.centered <- A.centered
  } else {
    B.sd <- apply(B, 2, sd)
    B.centered <- scale(B, scale = FALSE, center = TRUE)
  }

  C <- matrix(0, nrow = ncol(A), ncol = ncol(B))

  .Call('r_gpu_matmul_div', A.centered, B.centered, C, A.sd, B.sd, gpuID)
  
  return (C)
}

