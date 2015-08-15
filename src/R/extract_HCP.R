
require(parallel)

setwd("<folder containing the zipfiles>")
destpath <- Sys.getenv("HCP_ROOT")
zipfiles <- dir(pattern = "*zip$")
length(zipfiles)

# extract the zip files
mclapply(zipfiles, function(zf) {
  system(command = sprintf("unzip %s -d %s", zf, destpath)) }, mc.cores = 12)



