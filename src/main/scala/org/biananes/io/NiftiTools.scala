package org.biananes.io

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.SparkContext
import sna.Library
import xerial.larray.LArray

object NiftiTools {

  def NiftiImageReadMasked(img_file: String, mask_file: String, sc: SparkContext) : RowMatrix = {
    val lib_so_path = "/usr/local/lib/sparkniftireader/lib/libsparkniftireader.so"
    return NiftiImageReader(img_file, mask_file, sc, lib_so_path)
  }

  def NiftiImageReadMasked(img_file: String, mask_file: String, sc: SparkContext, lib_so_path: String) : RowMatrix = {
    return NiftiImageReader(img_file, mask_file, sc, lib_so_path)
  }

  private def NiftiImageReader(img_file: String, mask_file: String, sc: SparkContext, lib_so_path: String) : RowMatrix = {

    val cLib = Library(lib_so_path)

    // TODO: check if input is 4D fmri nii-file
    val nvoxel = cLib.get_nvoxel(img_file)[Long]
    val nvolumes = cLib.get_nvolumes(img_file)[Long]
    println("nvolumes: " + nvolumes)

    val nvoxel_in_mask = cLib.get_nvoxel_in_mask(mask_file)[Long]

    // TODO: add parameter for number of partitions; different values make sense for different cluster setups
    val brick_index = sc.parallelize(0 to (nvolumes.toInt - 1), 32)

    val img_file_bc = sc.broadcast(img_file)
    val mask_file_bc = sc.broadcast(mask_file)
    val lib_so_path_bc = sc.broadcast(lib_so_path)
    val nvoxel_in_mask_bc = sc.broadcast(nvoxel_in_mask)

    def read_brick(iter: Iterator[Int]): Iterator[Vector] = {

      val cLib = Library(lib_so_path_bc.value)
      // create LArray for one volume of the nii-file
      val nii = LArray.of[Double](nvoxel_in_mask_bc.value)
      var res = Array[Vector]()

      while (iter.hasNext) {
        val cur = iter.next
        cLib.larray_nifti_read_masked_brick(nii.address, img_file_bc.value, mask_file_bc.value, 1, Array[Int](cur))[Int]
        res = res :+ Vectors.dense(nii.toArray)
      }

      // free the LArray
      nii.free
      res.iterator
    }

    val vecs = brick_index.mapPartitions(read_brick)

    return (new RowMatrix(vecs))
  }
}

