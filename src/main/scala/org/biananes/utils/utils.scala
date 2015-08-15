package org.biananes.utils

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 *
 */
object utils {

  def meanCentering(mat: RowMatrix, sc: SparkContext) : RowMatrix = {

    // should be replaced by custom implementation => this one computes max, min, L1-, L2-norm, etc as well
    val colStats = mat.computeColumnSummaryStatistics

    // column means
    val colMeans = colStats.mean

    val colMeans_bc = sc.broadcast(colMeans)

    // substract column means
    val matMCrows = mat.rows.map {row =>
      var i = 0
      var res = Array[Double](row.size)

      while (i < row.size) {
        res(i) = row(i) - colMeans.apply(i)
      }
      Vectors.dense(res)
    }

    new RowMatrix(matMCrows)
  }

   // create RDD[LabeledPoint] from RowMatrix
   def getRegressionModelInput(mat: RowMatrix, colid: Long): RDD[LabeledPoint] = {
     val model_input = mat.rows.map(row => {
       val features = Array(row.toArray.take(colid.toInt), row.toArray.takeRight(row.toArray.length - colid.toInt - 1))
       LabeledPoint(row.apply(colid.toInt), Vectors.dense(features.flatten))
     })
     return model_input
   }


}
