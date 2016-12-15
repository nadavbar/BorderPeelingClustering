Border-Peeling Clustering
=========================

![Illustration of Border-Peeling clustering](/docs/border_peeling.png)

This repository contains a python implementation for the Border-Peeling clustering algorithm as well as the datasets that were used to evaluate the method.

##Installation instructions##

In our implementation We use numpy, scipy and scikit-learn packages.
We recommend using the Anaconda python distribution which installs these packages automatically.

Anaconda can be downloaded from here: https://www.continuum.io/downloads

If you wish to install the packages separately, please follow the instructions specified here:
https://docs.scipy.org/doc/numpy-1.10.0/user/install.html
https://www.scipy.org/install.html
http://scikit-learn.org/stable/install.html

For the UnionFind implementation We also use the python_algorithms package: https://pypi.python.org/pypi/python_algorithms

If you are using the pip package manager, you can install it by running the command:

```
pip install python_algorithms
```

##Usage instructions##

In order to run border peeling clustering, run the file run_bp.py using python.
The script uses the same parameters that are described in the experiments section of the paper.

The following command line arguments are available for run_bp.py:

```
usage: run_bp.py [-h] --input <file path> --output <file path> [--no-labels]
                 [--pca <dimension>] [--spectral <dimension>]

Border-Peeling Clustering

optional arguments:
  -h, --help            show this help message and exit
  --input <file path>   Path to comma separated input file
  --output <file path>  Path to output file
  --no-labels           Specify that input file has no ground truth labels
  --pca <dimension>     Perform dimensionality reduction using PCA to the
                        given dimension before running the clustering
  --spectral <dimension>
                        Perform spectral embedding to the given dimension
                        before running the clustering (If combined with PCA,
                        PCA is performed first)

```

##Input and output files format##

The expected input file format is a comma separated file, where each row represents a different multi-dimensional data point.
If ground truth is provided (and the --no-labels flag isn't specified),
the ground truth labels should be placed in the last column of each row.

For example, the following input file contains 3 data points in 2D,
where the first 2 points are part of one cluster and the second one is part of another cluster:

```
0.5, 0.2, 0
-0.5, 1.0, 0
10.1, 3.2, 1
```

The output file format is a multi-line file, where each line contains the label that was assigned to the data point.
(The index of the data points correspond to the index in the input file).

In case ground truth is provided, the Adjusted Rand Index (ARI) and Adjusted Mutual Information scores are printed.


##Data files and sampling##

The synthetic datasets <i>[1,2,3]</i> that were evaluated in the paper are available under the "synthetic_data" folder.
The datasets were downloaded from https://cs.joensuu.fi/sipu/datasets/ and then converted to a comma separated format.
The datasets were published in the following papers.


The data produced from convolutional neural networks embeddings is available under the "cnn_data"
folder in the following files:

 - mnist_data.txt - The MNIST <i>[4]</i> test set features that were produced using a convolutional neural network, as described in the paper. Each row contains a 500 dimensional feature vector, and the true label of the data point. The full MNIST data set is available in: http://yann.lecun.com/exdb/mnist/.
 - cifar_data.txt - The CIFAR-10 <i>[5]</i> test set features that were produced using a convolutional neural network,  as described in the paper. Each row contains a 64 dimensional feature vector, and the true label of the data point. The full CIFAR-10 data set is available in: https://www.cs.toronto.edu/~kriz/cifar.html.

To repeat the data sampling that was done in our experiments, the sample_data.py script can be used.
Please note that the sampling script works with datasets that have ground truth.
The following command line arguments are available for the script:

```
usage: sample_data.py [-h] --input <file path> --output <file path> --radius
                      <radius> --centers <centers #> --max-points <max points
                      #> [--cluster-min-size <cluster min size>]

Data points sampling

optional arguments:
  -h, --help            show this help message and exit
  --input <file path>   Path to comma separated input file
  --output <file path>  Path to output file
  --radius <radius>     The sampling radius
  --centers <centers #>
                        The number of centers to use for sampling)
  --max-points <max points #>
                        Maximum number of points to sample
  --cluster-min-size <cluster min size>
                        Minimum number of points for a cluster
```


<i>[1] A. Gionis, H. Mannila, and P. Tsaparas, Clustering aggregation. ACM Transactions on Knowledge Discovery from Data (TKDD), 2007. 1(1): p. 1-30.</i>

<i>[2] L. Fu and E. Medico, FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. BMC bioinformatics, 2007. 8(1): p. 3.</i>

<i>[3] C.J. Veenman, M.J.T. Reinders, and E. Backer, A maximum variance cluster algorithm. IEEE Trans. Pattern Analysis and Machine Intelligence, 2002. 24(9): p. 1273-1280.</i>

<i>[4] Y.LeCun and C.Cortes. MNIST hand written digit database. 2010.</i>

<i>[5] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. 2009.</i>
