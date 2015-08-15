require(Rniftilib)
require(fastCor)
require(Matrix)

basepath <- Sys.getenv("HCP_ROOT")
subjectspath <- file.path(basepath, ".")
subjects <- dir(subjectspath)
inputfile <- "rfMRI_REST1_RL.nii.gz"
brainmaskfile <- "brainmask_fs.2.nii.gz"

subject <- subjects[1] # for testing purposes
subjects.existing <- file.exists(file.path(subjectspath, subjects, "MNINonLinear", "Results", "rfMRI_REST1_RL", brainmaskfile)) & file.exists(file.path(subjectspath, subjects, "MNINonLinear", "Results", "rfMRI_REST1_RL", inputfile))

for (subject in subjects) {
  message(sprintf("Subject: %s", subject))
  setwd(file.path(subjectspath, subject, "MNINonLinear", "Results", "rfMRI_REST1_RL"))
  cat(sprintf("%s\n", getwd()))

  rest <- nifti.image.read(inputfile)
  brainmask <- nifti.image.read(brainmaskfile)

  brainmask.coords <- which(brainmask[, , ] == 1, arr.ind = TRUE)
  nodes <- cbind(1:nrow(brainmask.coords), brainmask.coords)
  cat(sprintf("\nNumber of Nodes: %s\n", nrow(nodes)))

  rest2d <- t(matrix(rest[, , , ], nrow = prod(dim(rest)[1:3]), ncol = dim(rest)[4]))
  rest.masked <- rest2d[, as.logical(brainmask[, , ])]

  cor.mat.indices <- list(fromnode = NULL, tonode = NULL, weight = NULL)
  threshold <- 1
  S <- Inf
  cor.mat.indices.all <- bigSpastCor(rest.masked, threshold = 0.6, gpuID = 3)
  save(cor.mat.indices.all, file = sprintf("cor.mat.indices.all_0.6.RData"))
  while (threshold > 0.6 & (S < 0 | S > 4)) {
    threshold <- threshold - 0.01
    n.edges <- sum(cor.mat.indices.all$weight > threshold)
    S <- log(nrow(nodes)) / log(2*n.edges/nrow(nodes))
  }
  cat(sprintf("\nThreshold: %.2f\n", threshold))

  cor.mat.indices <- list(fromnode = cor.mat.indices.all$fromnode[cor.mat.indices.all$weight > threshold], tonode = cor.mat.indices.all$tonode[cor.mat.indices.all$weight > threshold], weight = cor.mat.indices.all$weight[cor.mat.indices.all$weight > threshold])

  cor.mat.indices.df <- data.frame(as.character(cor.mat.indices$fromnode), as.character(cor.mat.indices$tonode), as.numeric(cor.mat.indices$weight), stringsAsFactors = FALSE)
	write.table(cor.mat.indices.df[, 1:2], file = sprintf("edgelist_%.2f_weighted.csv", threshold), col.names = F, row.names = F, quote = F)
  cat(sprintf("Number of Edges: %s\n\n", nrow(cor.mat.indices.df)))
}


