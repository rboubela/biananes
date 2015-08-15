biananes
========

Scalable fMRI Data Analysis

biananes is a library for fMRI data analysis on large datasets running on the [Apache Spark] framework. 

Installation
------------

A) Apache Spark: For the installation and setup of a Spark environment see the [Apache Spark](https://spark.apache.org/docs/latest/) documentation.

B) Spark NiftiReader C library

On all nodes running Spark workers need to have the libsparkniftireader installed

        # dpkg -i libsparkniftireader-0.0.2.deb
        
Note: libsparkniftireader depends on libnifti-dev

        # apt-get update
        # apt-get install libnifti-dev
        
C) After building the biananes.jar with

        $ mvn package
      
Usage
-----
      
One could add the JAR to a [spark-shell](https://spark.apache.org/docs/latest/)

        ./bin/spark-shell --master local[4] --jars /path/to/jar/biananes.jar

and start using it:

        scala> import org.binanes.io.NiftiTools.NiftiImageReadMasked
        scala> val hcp_root = sys.env("HCP_ROOT")
        scala> val img_file = hcp_root + "167743" + "/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL.nii.gz"
        scala> val mask_file = hcp_root + "167743" + "/MNINonLinear/Results/rfMRI_REST1_RL/brainmask_fs.2.nii.gz"
        scala> val mat = NiftiImageReadMasked(img_file, mask_file, sc)
