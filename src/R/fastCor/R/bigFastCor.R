bigFastCor <- function(A, B = NULL, gpuID = 0) {

  tile.length <- 24000 
  tiles.row <- ceiling(ncol(A) / tile.length)
  tiles.col <- ceiling(ncol(B) / tile.length)

  cor.mat <- matrix(0, nrow = ncol(A), ncol = ncol(B))

  pb <- txtProgressBar(min = 0, max = tiles.row*tiles.col)

  # TILING
  tile.counter <- 0
  timestamp()
  for (i in 1:tiles.row) {
    for (j in 1:tiles.col) {
      # print(paste("tile no.>", tile.counter))
      tile.counter <- tile.counter + 1
      setTxtProgressBar(pb, tile.counter)

      tile.indices.A <- (1 + (tile.length * (i - 1))):
                        min(tile.length * (i - 1) + tile.length, nrow(cor.mat))

      tile.indices.B <- (1 + (tile.length * (j - 1))):
                        min(tile.length * (j - 1) + tile.length, ncol(cor.mat))

       
       cor.mat[tile.indices.A, tile.indices.B] <- fastCor(A = A[, tile.indices.A], B = B[, tile.indices.B], gpuID = gpuID)

    }
  }

  return (cor.mat)
}

