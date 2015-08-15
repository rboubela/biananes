package org.biananes.examples

import org.apache.spark.SparkContext
import org.apache.spark.graphx._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.biananes.io.NiftiTools.NiftiImageReadMasked


/**
 * Created by rboubela on 10.08.15.
 */
object Examples {

  // reading a single subject fMRI dataset to a RowMatrix
  def ex1(sc: SparkContext): Unit = {

    val hcp_root = sys.env("HCP_ROOT")

    val img_file = hcp_root + "167743" + "/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL.nii.gz"
    val mask_file = hcp_root + "167743" + "/MNINonLinear/Results/rfMRI_REST1_RL/brainmask_fs.2.nii.gz"

    val mat = NiftiImageReadMasked(img_file, mask_file, sc)

  }

  // reading multiple subject fMRI dataset to a combined group-RowMatrix
  def ex2(sc: SparkContext): Unit = {

    val hcp_root = sys.env("HCP_ROOT")
    val subjects = sc.textFile(hcp_root + "subjectIDs.txt")
    val input_files = subjects.map{ subject =>
      new Tuple2(new String(hcp_root + subject + "/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL.nii.gz"),
        new String("/usr/share/data/fsl-mni152-templates/MNI152lin_T1_2mm.nii.gz")) }.collect

    val group_matrix = input_files.map{
      f => NiftiImageReadMasked(f._1, f._2, sc) }.reduce((m1, m2) => new RowMatrix(m1.rows.union(m2.rows)))

    val svd_result = group_matrix.computeSVD(1000)

  }

  def ex3(sc: SparkContext): Unit = {

    val hcp_root = sys.env("HCP_ROOT")
    val edgeListFiles = sc.textFile(hcp_root + "hcp_edgelist_files.txt").collect
    val graphs = edgeListFiles.map { edgefile => new Tuple2(edgefile, GraphLoader.edgeListFile(sc, edgefile, false)) }

    // compute the connected components for all graphs
    val allConnectedComponents = graphs.map { g => new Tuple2(g._1, g._2.connectedComponents().vertices) }
    val resfiles = allConnectedComponents.map{ cc => {
      val file = cc._1.substring(0, 106) + "connected_components"
      cc._2.coalesce(1, true).saveAsTextFile(file)
      file
    }}

  }
 
  def ex4(sc: SparkContext): Unit = {

    val hcp_root = sys.env("HCP_ROOT")
    val edgeListFiles = sc.textFile(hcp_root + "hcp_edgelist_files.txt").collect
    val graphs = edgeListFiles.map { edgefile => new Tuple2(edgefile, GraphLoader.edgeListFile(sc, edgefile, false)) }

    // compute clustering coefficents
    val allClusteringCoeff = graphs.map { g => new Tuple2(g._1,
      g._2.degrees.join(g._2.triangleCount.vertices).map{ case (id, (d, tc)) =>
      (id, tc / (d * (d - 1) / 2.0))})
    }

  }

}

