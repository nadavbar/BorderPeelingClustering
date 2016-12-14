import argparse
import numpy as np
import clustering_tools as ct
import border_tools as bt
import BorderPeel

from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import SpectralEmbedding


parser = argparse.ArgumentParser(description='Border-Peeling Clustering')
parser.add_argument('--input', type=str, metavar='<file path>',
                   help='Path to comma separated input file', required=True)
parser.add_argument('--output', type=str, metavar='<file path>',
                    help='Path to output file', required=True)
parser.add_argument("--no-labels", help="Specify that input file has no ground truth labels",
                    action="store_true")
parser.add_argument('--pca', type=int, metavar='<dimension>',
                    help='Perform dimensionality reduction using PCA to the given dimension before running the clustering', required=False)
parser.add_argument('--spectral', type=int, metavar='<dimension>',
                    help='Perform sepctral embdding to the given dimension before running the clustering (If comibined with PCA, PCA is performed first)', required=False)

args = parser.parse_args()

output_file_path = args.output
input_file_path =  args.input
input_has_labels = not args.no_labels

pca_dim = args.pca
spectral_dim = args.spectral

debug_output_dir = None
k=20
C=3
border_precentile = 0.1
mean_border_eps = 0.15
max_iterations = 100
stopping_precentile = 0.01

data, labels = ct.read_data(input_file_path, has_labels=input_has_labels)

if len(data) < 1000:
    min_cluster_size = 10
else:
    min_cluster_size = 30

embeddings = data

if pca_dim is not None:
    if pca_dim >= len(embeddings[0]):
        print "PCA target dimension (%d) must be smaller than data dimension (%d)"%(pca_dim, len(embeddings[0]))
        exit(1)
    print "Performing PCA to %d dimensions"%pca_dim
    pca = PCA(n_components=pca_dim)
    embeddings = pca.fit_transform(data)

if spectral_dim is not None:
    if spectral_dim >= len(embeddings[0]):
        print "Spectral Embedding dimension (%d) must be smaller than data dimension (%d)"%(spectral_dim, len(embeddings[0]))
        exit(1)
    print "Performing Spectral Embedding to %d dimensions" % spectral_dim
    se = SpectralEmbedding(n_components=spectral_dim)
    embeddings = se.fit_transform(data)

print "Running Border-Peeling clustering on: %s"%input_file_path

lambda_estimate = bt.estimate_lambda(embeddings, k)
bp = BorderPeel.BorderPeel(mean_border_eps=mean_border_eps, max_iterations=max_iterations, k=k, plot_debug_output_dir = None,
                min_cluster_size = min_cluster_size, dist_threshold = lambda_estimate, convergence_constant = 0, link_dist_expansion_factor = C,
                verbose = True, border_precentile = border_precentile, stopping_precentile=stopping_precentile)

clusters = bp.fit_predict(embeddings)

clusters_count = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
print "Found %d clusters"%clusters_count

with open(output_file_path, "wb") as handle:
    for c in clusters:
        handle.write("%d\n"%c)

print "Saved cluster results to %s"%output_file_path

if input_has_labels:
    print "ARI: %0.3f"%adjusted_rand_score(clusters, labels)
    print "AMI: %0.3f"%adjusted_mutual_info_score(clusters,labels)