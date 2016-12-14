import argparse
import numpy as np
import clustering_tools as ct


parser = argparse.ArgumentParser(description='Data points sampling')
parser.add_argument('--input', type=str, metavar='<file path>',
                   help='Path to comma separated input file', required=True)
parser.add_argument('--output', type=str, metavar='<file path>',
                    help='Path to output file', required=True)
parser.add_argument('--radius', type=int, metavar='<radius>',
                    help='The sampling radius', required=True)
parser.add_argument('--centers', type=int, metavar='<centers #>',
                    help='The number of centers to use for sampling)', required=True)
parser.add_argument('--max-points', type=int, metavar='<max points #>',
                    help='Maximum number of points to sample', required=True)
parser.add_argument('--cluster-min-size', type=int, metavar='<cluster min size>',
                    help='Minimum number of points for a cluster', default=30)

args = parser.parse_args()

output_file_path = args.output
input_file_path =  args.input
radius = args.radius
centers = args.centers
max_points = args.max_points
cluster_min_points = args.cluster_min_size

data, labels = ct.read_data(input_file_path)

filter_mask = ct.sample_with_radius_and_filter_small_clusters(data, labels, number_of_centers=centers,
                                                              max_size=max_points, radius=radius, cluster_min_points=cluster_min_points)
sampled_data = data[filter_mask]
sampled_labels = labels[filter_mask]

print "Sampled %d data points, %d classes"%(len(sampled_data), len(np.unique(sampled_labels)))

ct.save_data(output_file_path, sampled_data, sampled_labels)

print "Sampled data saved in: %s"%output_file_path