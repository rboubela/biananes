bigSpastCor <- function (A, B = NULL, gpuID = 0, threshold = 0.5)
{
    ## adjust to your GPU's memory size
    tile.length <- 24000
    if (is.null(B)) {
      B <- A
    }
    tiles.row <- ceiling(ncol(A)/tile.length)
    tiles.col <- ceiling(ncol(B)/tile.length)
    cor.mat.indices <- list(fromnode = NULL, tonode = NULL, weight = NULL)
    pb <- txtProgressBar(min = 0, max = tiles.row * tiles.col)
    tile.counter <- 0
    timestamp()
    for (i in 1:tiles.row) {
        for (j in 1:tiles.col) {
            tile.counter <- tile.counter + 1
            setTxtProgressBar(pb, tile.counter)
            tile.indices.A <- (1 + (tile.length * (i - 1))):min(tile.length *
                (i - 1) + tile.length, ncol(A))
            tile.indices.B <- (1 + (tile.length * (j - 1))):min(tile.length *
                (j - 1) + tile.length, ncol(B))
            cor.values <- fastCor(A = A[, tile.indices.A], B = B[, tile.indices.B], gpuID = gpuID)
            cor.tile <- which(abs(cor.values) > threshold, arr.ind = TRUE)
            cor.vector <- which(abs(cor.values) > threshold, arr.ind = FALSE)
            cor.mat.indices$fromnode <- c(cor.mat.indices$fromnode, tile.indices.A[cor.tile[, 1]])
            cor.mat.indices$tonode <- c(cor.mat.indices$tonode, tile.indices.B[cor.tile[, 2]])
            cor.mat.indices$weight <- c(cor.mat.indices$weight, cor.values[,][cor.vector])
        }
    }
    return(cor.mat.indices)
}

