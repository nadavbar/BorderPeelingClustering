import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from matplotlib import patches
from matplotlib.collections import PatchCollection
from os.path import join
from sklearn import cluster
from sklearn.cluster import AffinityPropagation, DBSCAN, MeanShift, estimate_bandwidth, SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from sklearn import preprocessing


# read arff file:
def read_arff(file_path):
    read_data = False
    data  = []
    labels = []
    with open(file_path) as handle:
        for l in handle:
            l = l.rstrip()
            if (read_data):
                splitted = l.split(",")
                row = [float(s) for s in splitted[:len(splitted)-1]]
                data.append(row)
                labels.append(splitted[len(splitted)-1])
            elif (l.lower() == "@data"):
                read_data = True
    
    le = preprocessing.LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    return np.ndarray(shape=(len(data), len(data[0])), buffer=np.matrix(data)), np.array(encoded_labels)

def save_data(file_path, data , labels):
    with open(file_path, "w") as handle:
        for p, l in zip(data, labels):
            line = ",".join([str(s) for s in p]) + "," + str(l)
            handle.write(line + "\n")


def load_from_file_or_data(obj, seperator=',', dim=2, hasLabels=False):
    if (type(obj) is str):
        return read_data(obj, seperator=seperator, dim=dim, hasLabels=hasLabels)
    else:
        return obj

def add_random_noise(data, labels=None, noise_points_count=100):
    # get bounding box of data:
    dim = data.shape[1]
    min_vals = np.zeros(dim)
    max_vals = np.zeros(dim)

    # initialize the boundaries with the value of the first row
    for v,i in zip(data[0].A1,xrange(dim)):
        min_vals[i] = v
        max_vals[i] = v

    for r in data:
        for v,i in zip(r.A1,xrange(dim)):
            if (v > max_vals[i]):
                max_vals[i] = v
            if (v < min_vals[i]):
                min_vals[i] = v
            
    # add random points:
    noise_points = []
    for i in xrange(dim):
        noise_points.append(np.random.uniform(min_vals[i], max_vals[i], (noise_points_count,1)))

    noise = np.concatenate(tuple(noise_points), axis=1)
    noised_data = np.concatenate((data, noise))
    noised_labels = np.concatenate((labels, -1*np.ones(noise_points_count)))

    return noised_data, noised_labels
    

def draw_clusters(X, labels, colors=None, show_plt=True, show_title=False, name=None, ax=None,
                  markersize=15, markeredgecolor='k', use_clustes_as_keys = False, linewidth=0,
                  noise_data_color='k'):
    import seaborn as sns
    if (ax == None):
        ax = plt
    #unique_labels = set(labels)
    unique_labels = np.unique(labels)
    label_map = sorted(unique_labels)
    if (colors == None):
        colors = sns.color_palette()
        if len(colors) < len(unique_labels):
            colors = plt.cm.Spectral(np.linspace(1, 0, len(unique_labels)))
    has_noise = False

    if not use_clustes_as_keys:
        if (label_map[0] == -1):
            if (isinstance(colors, list)):
                colors = [noise_data_color] + colors
            else:
                colors = [noise_data_color] + colors.tolist()

    #for k, col in zip(label_map, colors):
    for k, i in zip(label_map, xrange(len(label_map))):
        if k == -1:
            # Black used for noise.
            col = noise_data_color
            has_noise = True
        else:
            if use_clustes_as_keys:
                col = colors[int(k)]
            else:
                col = colors[i]
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], s=markersize, facecolor=col,
                 edgecolor=markeredgecolor, linewidth=linewidth)
        #ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
        #         markeredgecolor=markeredgecolor, markersize=markersize, lw=lw)

    if (show_title):
        labels_count = len(unique_labels)
        if (has_noise):
            labels_count = labels_count - 1
        title_prefix = ""
        if (name != None):
            title_prefix = "%s - "%name
        if hasattr(ax, 'set_title'):
            ax.set_title((title_prefix + 'Estimated number of clusters: %d') % len(unique_labels))
        else:
            ax.title((title_prefix + 'Estimated number of clusters: %d') % len(unique_labels))
    #if (show_plt):
    #    ax.show()
    return ax


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_clusters3d(X, labels, colors=None, show_plt=True, show_title=False, name=None, ax=None, markersize=15, markeredgecolor='k', linewidth=0):
    import seaborn as sns
    #if (ax == None):
    #    ax = plt
    #unique_labels = set(labels)
    fig = plt.figure(figsize=(float(1600) / float(72), float(1600) / float(72)))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    label_map = sorted(unique_labels)
    if (colors == None):
        colors = sns.color_palette()
        #colors = plt.cm.Spectral(np.linspace(1, 0, len(unique_labels)))
    has_noise = False

    if (label_map[0] == -1):
        colors = ['k'] + colors

    for k, col in zip(label_map, colors):
        if k == -1:
            # Black used for noise.
            #col = 'k'
            has_noise = True

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        print col
        ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=markersize, c=col)
#                 edgecolor=markeredgecolor, linewidth=linewidth)
        #ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
        #         markeredgecolor=markeredgecolor, markersize=markersize, lw=lw)

    if (show_title):
        labels_count = len(unique_labels)
        if (has_noise):
            labels_count = labels_count - 1
        title_prefix = ""
        if (name != None):
            title_prefix = "%s - "%name
        if hasattr(ax, 'set_title'):
            ax.set_title((title_prefix + 'Estimated number of clusters: %d') % len(unique_labels))
        else:
            ax.title((title_prefix + 'Estimated number of clusters: %d') % len(unique_labels))
    #if (show_plt):
    #    ax.show()
    #ax.set_zlim([-0.01, 0])
    return ax

def read_data(filePath, seperator=',', has_labels=True):
    with open(filePath) as handle:
        data = []
        labels = None
        if (has_labels):
            labels = []
        
        for line in handle:
            line = line.rstrip()
            if len(line) == 0:
                continue
            row = []
            line_parts = line.split(seperator)
            row = [float(i) for i in line_parts[:len(line_parts)-1]]
            data.append(row)
            if (has_labels):
                label = int(line_parts[-1])
                labels.append(label)
                
        return np.ndarray(shape=(len(data), len(data[0])), buffer=np.matrix(data)), np.array(labels)
    

def show_clusters(X, labels_true):
    X, labels_true = load_from_file_or_data(pathOrData, hasLabels=True)
    draw_clusters(X, labels_true)
    

def run_dbscan(data, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    return db.labels_

def run_hdbscan(data, min_cluster_size):
    import hdbscan
    hdb = clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(data)
    return hdb.labels_

def clusters_to_labels(clusters, labels, force_unique=False):
    # for each cluster - find the matching label according to the majority of the data points
    # get the number of clusters:

    offset = 0
    if (0 not in np.unique(labels)):
        offset = 1

    labels_count = max(np.unique(labels)) + (1 - offset)

    clusters_hist_map = {}
    clusters_labels_map = {}

    for c,l in zip(clusters, labels):
        if (c == -1):
            continue
        if (not clusters_hist_map.has_key(c)):
            clusters_hist_map[c] = np.zeros(labels_count)
        clusters_hist_map[c][l - offset] += 1

    unset_clusters = []
    for c in clusters_hist_map.keys():
        l = np.argmax(clusters_hist_map[c])
        if force_unique:
            label_already_used = False
            for k in clusters_labels_map:
                if clusters_labels_map[k] == (l + offset):
                    unset_clusters.append(c)
                    label_already_used = True
            if label_already_used:
                continue

        clusters_labels_map[c] = l + offset

    if force_unique:
        current_max_label = np.max(labels)
        for c in unset_clusters:
            label_to_use = -1
            for l in labels:
                is_label_used = False
                for k in clusters_labels_map:
                    if clusters_labels_map[k] == l:
                        is_label_used = True
                        break
                if not is_label_used:
                    label_to_use = l
                    break
            if label_to_use == -1:
                current_max_label += 1
                label_to_use =  current_max_label
            clusters_labels_map[c] = label_to_use

    new_clusters = np.zeros(len(clusters))

    for i,c in zip(xrange(len(new_clusters)), clusters):
        if c == -1:
            new_clusters[i] = -1
        else:
            new_clusters[i] = clusters_labels_map[c]

    return new_clusters
                               
def evaluate_clustering(X, labels_true, labels):
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Normalized Mutual Information: %0.3f"
          % metrics.normalized_mutual_info_score(labels_true, labels))
    try:
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels))
    except ValueError:
        print("Silhouette Coefficient: None")

def cluster_and_evaluate(path_or_data, method):
    X, labels_true = load_from_file_or_data(path_or_data, hasLabels=True)
    labels = method(X)
    draw_clusters(X,labels)
    return evaluate_clustering(X, labels_true, labels)
    
def run_mean_shift(data, bandwidth=None, qunatile=0.09, cluster_all=False):
    if (bandwidth == None):
        bandwidth = estimate_bandwidth(data, quantile=qunatile, n_samples=len(data))
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=cluster_all)
    return ms.fit(data).labels_

def run_spectral_clustering(data, k):
    spectral = SpectralClustering(n_clusters=k,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors", n_init=1000)
    return spectral.fit(data).labels_

def run_affinity_propogation(data, damping=0.5):
    af = AffinityPropagation(damping=damping).fit(data)
    return af.labels_

def run_ward_clustering(data, k):
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(data, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    return cluster.AgglomerativeClustering(n_clusters=k, linkage='ward',
                                           connectivity=connectivity).fit(connectivity.toarray()).labels_

def sample_from_radius(data, number_of_centers, max_length, radius):
    order = np.random.permutation(len(data))
    centers = []
    filter_mask = np.zeros(len(data))
    for i in xrange(number_of_centers):
        centers.append(data[order[i]])
        filter_mask[order[i]] = 1.0
    samples_count = number_of_centers
    for i in order[1:]:
        if samples_count > max_length:
            break
        current = data[order[i]]
        for c in centers:
            dist = np.linalg.norm(c - current)
            if dist <= radius:
                filter_mask[order[i]] = 1.0
                samples_count += 1
                break

    return filter_mask.astype(bool)

def get_count_by_labels(labels):
    count = {}
    for l in labels:
        if (not count.has_key(l)):
            count[l] = 1
        else:
            count[l] += 1
    return count

def filter_small_clusters(labels, filter_mask, threshold):
    filtered_labels = labels[filter_mask]
    count_by_labels = get_count_by_labels(filtered_labels)

    #filter_mask = np.zeros(len(labels))
    for i in xrange(len(filter_mask)):
        if (not  filter_mask[i]):
            continue
        filter_mask[i] = 1.0 if count_by_labels[labels[i]] > threshold else 0.0

    # filtered_data = data[filter_mask.astype(bool)]
    # filtered_labels = labels[filter_mask.astype(bool)]
    # filter out clusters with less than 50 items
    # return filtered_data, filtered_labels

    return filter_mask

def sample_with_radius_and_filter_small_clusters(data, labels, number_of_centers, max_size, radius, cluster_min_points):
    filter_mask = sample_from_radius(data, number_of_centers, max_size, radius)
    return filter_small_clusters(labels, filter_mask, cluster_min_points)

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist

def draw_knn_dist_hist(data, k=20):
    nbrs = NearestNeighbors(n_neighbors=k).fit(data, data)
    all_dists = []
    distances, indices = nbrs.kneighbors()
    for dists in distances:
        for d in dists:
            all_dists.append(d)
    print "knn dists stats:"
    print_dists_statistics(all_dists)
    plt.figure()
    plt.title("knn distance hist")
    plt.hist(all_dists, bins=50);

def draw_dist_hist(data):

    dists = pdist(data)

    print "all dists stats:"
    print_dists_statistics(dists)

    plt.figure()
    plt.title("all distances hist")
    plt.hist(dists, bins=50)

def print_dists_statistics(dists):
    print "mean: %.3f"%(np.mean(dists))
    print "median: %.3f"%(np.median(dists))
    print "variance: %.3f"%(np.var(dists))
    print "max: %.3f"%(np.max(dists))
    print "min: %.3f"%(np.min(dists))
    print "std+mean: %.3f"%(np.std(dists) + np.mean(dists))

def show_dists_stats(data,k=20):
    draw_dist_hist(data)
    draw_knn_dist_hist(data, k)

class DebugPlotSession:
    def __init__(self, output_dir, marker_size=120, line_width=1.5):
        self.current_index = 0
        self.output_dir = output_dir
        self.axes_ylim = None
        self.axes_xlim = None
        self.line_width = line_width
        self.marker_size = marker_size

    def get_or_set_axes_lim(self, plt, data):
        axes = plt.gca()

        # find min and max x, min, max y:
        if (self.axes_ylim == None or self.axes_xlim == None):
            min_x = np.min(data[:, 0])
            max_x = np.max(data[:, 0])
            min_y = np.min(data[:, 1])
            max_y = np.max(data[:, 1])

            self.axes_ylim = (min_y - 0.5, max_y + 0.5)
            self.axes_xlim = (min_x - 0.5, max_x + 0.5)

        axes.set_ylim(self.axes_ylim)
        axes.set_xlim(self.axes_xlim)

    def add_circles(self, ax, centers, radis):
        circles = []
        for c, r in zip(centers, radis):
            circles.append(patches.Circle(tuple(c), r))

        patch_collection = PatchCollection(circles)
        patch_collection.set_facecolor("none")
        patch_collection.set_edgecolor("blue")
        patch_collection.set_linewidth(1.5)
        patch_collection.set_linestyle("dashed")
        ax.add_collection(patch_collection)
        print "added %d circles"%len(circles)


    def plot_and_save(self, data, filter=[],
                      colors=[(192.0 / 255.0, 0, 0), '#6b8ba4', (1.0, 1.0, 1.0)],
                      links = None, circles_and_radis = None):
        if (self.output_dir == None):
            return
        
        self.current_index = self.current_index + 1

        if (len(filter) == 0):
            colors = ['blue']
            filter = np.array([True] * len(data))
        #else:
            # for first iteration..
        #    colors = [(192.0/255.0, 0, 0),(146.0/255, 208.0/255, 80.0/255)]
            # for second iteration..
            #colors = [(255.0/255.0, 192.0/255.0, 0),(146.0/255, 208.0/255, 80.0/255)]
            # for 3rd iteration
            #colors = [(255.0/255.0, 255.0/255.0, 0),(146.0/255, 208.0/255, 80.0/255)]
            #colors = ['red','blue']
        plt.figure(figsize=(800.0/72.0, 800.0/72.0))

        fig = draw_clusters(data, filter, colors=colors, show_plt=False,
                            show_title=False, markersize=self.marker_size,
                            markeredgecolor=(56.0/255.0,93.0/255.0,138.0/255.0), linewidth=self.line_width)
        self.get_or_set_axes_lim(fig, data)
        ax = fig.axes()
        if (links != None):
            for p in links:
                #point1 = data[p[0],:].A1
                #point2 = data[p[1],:].A1
                point1 = data[p[0],:]
                point2 = data[p[1],:]
                #ax.arrow(point1[0], point1[1], point2[0] - point1[0], point2[1] - point1[1], head_width=0.1, head_length=0.2, fc='k', ec='k', color='green')
                fig.plot([point1[0], point2[0]], [point1[1], point2[1]], color = 'green')

        if (circles_and_radis != None):
            self.add_circles(plt.gca(), circles_and_radis[0], circles_and_radis[1])
        
        #plt.axes().get_xaxis().set_visible(False)
        #plt.axes().get_yaxis().set_visible(False)
        #plt.axes().patch.set_visible(False)

        plt.axis('off')

        if (self.output_dir != None):
            file_path = join(self.output_dir,"%s.png"%self.current_index)
            fig.savefig(file_path, bbox_inches='tight')

        plt.close()

    def plot_clusters_and_save(self, data, clusters, name=None, show_plt=False, noise_data_color='k'):
        if ((self.output_dir == None) and (show_plt == False)):
            return

        import seaborn as sns
        self.current_index = self.current_index + 1

        #fig = draw_clusters(data, clusters, show_plt=show_plt, show_title=False, name=name)
        plt.figure(figsize=(800.0 / 72.0, 800.0 / 72.0))
        colors = [(1.0, 192.0/255.0, 0), (0, 176.0/255.0, 240.0/255.0)] + sns.color_palette()
        fig = draw_clusters(data, clusters, show_plt=show_plt, show_title=False, markersize=self.marker_size,
                            markeredgecolor=(56.0/255.0,93.0/255.0,138.0/255.0), linewidth=self.line_width,
                            colors = colors, noise_data_color=noise_data_color)

        plt.axis('off')
        self.get_or_set_axes_lim(fig, data)

        if (self.output_dir != None):
            if (name == None):
                file_path = join(self.output_dir,"%s.png"%self.current_index)
            else:
                file_path = join(self.output_dir,"%s.png"%name)

            fig.savefig(file_path, bbox_inches='tight')
        
        plt.close()

CLUSTERS_EVALUATION_COLUMNS= [
    "Method",
    "Params",
    "Clusters #",
    "NMI",
    "AMI",
    "ARI",
    "RI",
#    "Homogeneity",
#    "Completeness",
#    "V-measure",
#    "Silhouette Coefficient"
]

class EvaluationFields(Enum):
    method = 1
    params = 2
    clusters_num = 3
    normazlied_mutual_information = 4
    adjusted_mutual_information = 5
    adjusted_rand_index = 6
    rand_index = 7
#    homogeneity = 8
#    completeness = 9
#    v_measure = 10
#    silhouette_coefficient = 11

CLUSTERS_EVALUATION_CRITERIA= [
    "Method",
    "Params",
    "Clusters #",
    "NMI",
    "AMI",
    "ARI",
    "RI",
#    "Homogeneity",
#    "Completeness",
#    "V-measure",
#    "Silhouette Coefficient"
]


def silhouette_safe(X, labels):
    try:
        return metrics.silhouette_score(X, labels)
    except ValueError:
        return -1


def rand_index(labels, cluster_assignments):
    correct = 0
    total = 0
    sample_ids = range(len(labels))
    for index_combo in itertools.combinations(sample_ids, 2):
        index1 = index_combo[0]
        index2 = index_combo[1]

        same_class = (labels[index1] == labels[index2])
        same_cluster = (cluster_assignments[index1]
                        == cluster_assignments[index2])

        if same_class and same_cluster:
            correct += 1
        elif not same_class and not same_cluster:
            correct += 1

        total += 1

    return float(correct) / total


evaulations_dict = {
    EvaluationFields.method : lambda X, labels_true, labels, name, params : name,
    EvaluationFields.params : lambda X, labels_true, labels, name, params : params,
    EvaluationFields.clusters_num : lambda X, labels_true, labels, name, params : len(np.unique(labels)) - (1 if -1 in labels else 0),
    EvaluationFields.normazlied_mutual_information : lambda X, labels_true, labels, name, params : metrics.normalized_mutual_info_score(labels_true, labels),
    EvaluationFields.adjusted_mutual_information : lambda X, labels_true, labels, name, params :  metrics.adjusted_mutual_info_score(labels_true, labels),
    EvaluationFields.adjusted_rand_index : lambda X, labels_true, labels, name, params : metrics.adjusted_rand_score(labels_true, labels),
    EvaluationFields.rand_index: lambda X, labels_true, labels, name, params: rand_index(labels_true, labels),
#    EvaluationFields.homogeneity : lambda X, labels_true, labels, name, params : metrics.homogeneity_score(labels_true, labels),
#    EvaluationFields.completeness : lambda X, labels_true, labels, name, params : metrics.completeness_score(labels_true, labels),
#    EvaluationFields.v_measure : lambda X, labels_true, labels, name, params : metrics.v_measure_score(labels_true, labels),
#    EvaluationFields.silhouette_coefficient : lambda X, labels_true, labels, name, params : silhouette_safe(X, labels),
}

def format_param(param):
    if isinstance(param, float):
        return "%0.2f"%param
    else:
        return param

class MethodsEvaluation:
    def __init__(self, output_dir=None):
        # key are method names, values are list of scores
        self.scores_table = OrderedDict()
        self.dbg_plot_session = DebugPlotSession(output_dir)
        self.sorted_by = None

    def evaulate_method(self, X, labels_true, labels, name, params, show_plt=False):
        if (not self.scores_table.has_key(name)):
            self.scores_table[name] = []

        scores = [evaulations_dict[m](X, labels_true, labels, name, params) for m in EvaluationFields]
        self.scores_table[name].append(scores)
        self.dbg_plot_session.plot_clusters_and_save(X, labels, name=name, show_plt=show_plt)
        return scores

    def cluster_params_range_evaluation(self, data, true_labels, base_params, params_range, method, method_name):
        for p in itertools.product(*params_range.values()):
            func_params = deepcopy(base_params)
            for i in xrange(len(params_range)):
                func_params[params_range.keys()[i]] = p[i]
            clusters = method(**func_params)
            params_str = "_".join(["%s=%s"%(k,format_param(j)) for k,j in zip(params_range,p)])
            self.evaulate_method(data, true_labels, clusters, method_name, params_str ,show_plt = False)
   
    # for each method, leave only the top n scores
    # this will also sort the list
    def filter_top_n_by_field_for_method(self, evaluation_field,  n):
        self.sort_scores_by_field(evaluation_field)
        for k in self.scores_table:
            self.scores_table[k] = self.scores_table[k][:n]

    def sort_scores_by_field(self, evaluation_field):
        self.sorted_by = evaluation_field
        for k in self.scores_table:
            self.scores_table[k].sort(key=lambda x: x[evaluation_field.value-1], reverse=True)
        
    def print_evaluations(self, draw_in_notebook=False):
        print "\t".join(CLUSTERS_EVALUATION_COLUMNS)

        for k in self.scores_table:
            for score in self.scores_table[kneighbors_graph]:
                formatted_string_list = [score[:EvaluationFields.clusters_num.value-1]] + \
                    ["%d"%score[EvaluationFields.clusters_num.value-1]] + \
                    ["%0.3f"%s for s in score[EvaluationFields.clusters_num.value:]]
                print "\t".join(formatted_string_list)

    def get_evaluation_table(self):
        formatted_list = []

        keys = self.scores_table.keys()

        #if self.sorted_by != None:
        #    keys.sort(key=lambda x: self.scores_table[x][0][self.sorted_by.value - 1], reverse=True)

        for k in keys:
            for score in self.scores_table[k]:
                formatted_list.append(score[:EvaluationFields.clusters_num.value-1] +\
                        ["%d"%score[EvaluationFields.clusters_num.value-1]] +\
                        ["%0.3f"%s for s in score[EvaluationFields.clusters_num.value:]])
        
        return CLUSTERS_EVALUATION_COLUMNS[:], formatted_list

