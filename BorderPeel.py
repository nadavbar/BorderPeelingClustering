import border_tools as bt
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin


class BorderPeel(BaseEstimator, ClusterMixin):
    """Perform DBSCAN clustering from vector array or distance matrix.
    TODO: Fill out doc
    BorderPeel - Border peel based clustering
    Read more in the :ref:`User Guide <BorderPeel>`.
    Parameters
    ----------
    TODO: Fill out parameters..
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    Attributes
    ----------
    core_sample_indices_ : array, shape = [n_core_samples]
        Indices of core samples.
    Notes
    -----

    References
    ----------

    """

    def __init__(self, method="exp_local_scaling", max_iterations=150,
                    mean_border_eps=-1, k=20, plot_debug_output_dir = None, min_cluster_size = 3,
                    dist_threshold = 3, convergence_constant = 0, link_dist_expansion_factor = 3,
                    verbose = True, border_precentile = 0.1, stopping_precentile=0, merge_core_points=True,
                    debug_marker_size=70):
        self.method = method
        self.k = k
        self.plot_debug_output_dir = plot_debug_output_dir
        self.min_cluster_size = min_cluster_size
        self.dist_threshold = dist_threshold
        self.convergence_constant = convergence_constant
        self.link_dist_expansion_factor = link_dist_expansion_factor
        self.verbose = verbose
        self.border_precentile = border_precentile
        self.stopping_precentile = stopping_precentile
        self.merge_core_points = merge_core_points
        self.max_iterations = max_iterations
        self.mean_border_eps = mean_border_eps
        self.debug_marker_size = debug_marker_size

        # out fields:
        self.labels_ = None
        self.core_points = None
        self.core_points_indices = None
        self.non_merged_core_points = None
        self.data_sets_by_iterations = None
        self.associations = None
        self.link_thresholds = None
        self.border_values_per_iteration = None

    def fit(self, X, X_plot_projection = None):
        """Perform BorderPeel clustering from features
        Parameters
        ----------
        X : array of features (TODO: make it work with sparse arrays)
        X_projected : A projection of the data to 2D used for plotting the graph during the cluster process
        """

        if (self.method == "exp_local_scaling"):
            border_func = lambda data: bt.rknn_with_distance_transform(data, self.k, bt.exp_local_scaling_transform)
            #threshold_func = lambda value: value > threshold

        result = bt.border_peel(X, border_func, None, max_iterations=self.max_iterations,
                           mean_border_eps=self.mean_border_eps,
                           plot_debug_output_dir=self.plot_debug_output_dir, k=self.k, precentile=self.border_precentile,
                           dist_threshold=self.dist_threshold, link_dist_expansion_factor=self.link_dist_expansion_factor,
                           verbose=self.verbose, vis_data=X_plot_projection, min_cluster_size=self.min_cluster_size,
                           stopping_precentile=self.stopping_precentile, should_merge_core_points=self.merge_core_points,
                           debug_marker_size=self.debug_marker_size)

        self.labels_, self.core_points, self.non_merged_core_points, \
        self.data_sets_by_iterations, self.associations, self.link_thresholds, \
        self.border_values_per_iteration, self.core_points_indices = result

        return self

    def fit_predict(self, X, X_plot_projection = None):
        """Performs BorderPeel clustering clustering on X and returns cluster labels.
        Parameters
        ----------
        X : array of features (TODO: make it work with sparse arrays)
        X_projected : A projection of the data to 2D used for plotting the graph during the cluster process
        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """
        self.fit(X, X_plot_projection=X_plot_projection)
        return self.labels_
