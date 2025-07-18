# CBR: Cluster-Based Resampler
# This method was introduced in the following paper:
#
# L. Akritidis, P. Bozanis, "A Clustering-Based Resampling Technique with Cluster Structure Analysis for Software Defect
# Detection in Imbalanced Datasets", Information Sciences, vol. 674, pp. 120724, 2024.

import numpy as np

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances

from imblearn.over_sampling import SMOTE

from tqdm import tqdm


class CentroidSampler:
    """
    Centroid-based oversampling.

    Mitigate data imbalance in a cluster of multi-class samples. For each class, compute a centroid point by
    averaging the co-ordinates of the points that belong to that class. Then, generate artificial samples
    randomly, in a point over the line that connects each point and the corresponding centroid.

    Args:
        sampling_strategy (string or dictionary): How the algorithm generates samples:

          * 'auto': the model balances the dataset by oversampling the minority classes.
          * dict: a dictionary that indicates the number of samples to be generated from each class.
    """
    def __init__(self, sampling_strategy='auto', random_state=0):
        self._n_samples = 0
        self._n_classes = 0
        self._input_dim = 0
        self._sampling_strategy = sampling_strategy
        self._random_state = random_state

    def fit_resample(self, x_in, y_in):
        np.random.seed(self._random_state)

        x_out = x_in.copy()
        y_out = y_in.copy()

        self._n_samples = x_in.shape[0]
        self._input_dim = x_in.shape[1]
        self._n_classes = len(set(y_in))

        y_res = np.array(y_in).reshape(-1, 1)

        samples_per_class = np.array([len(y_res[y_res == c]) for c in range(self._n_classes)])
        max_samples = np.max(samples_per_class)
        # print("Samples per Class:", samples_per_class, samples_per_class.shape)

        # For each class:
        for cls in range(self._n_classes):

            # Class balancing mode - this does not touch the majority class:
            if self._sampling_strategy == 'auto':

                # If this is a minority class and has more than 1 data instances:
                if max_samples > samples_per_class[cls] > 1:
                    idx = [p for p in range(y_res.shape[0]) if y_res[p] == cls]
                    x_class = x_in[idx, :]
                    num_min_samples = x_class.shape[0]

                    samples_to_create = max_samples - num_min_samples
                    centroid = np.mean(x_class, axis=0)

                    # print("Minority samples of class", cls, ":\n", X_class)
                    # print("Number of samples to create:", samples_to_create)
                    # print("\tCluster", cls, " Centroid: ", centroid)

                    # Keep creating samples until the desired number (=generated_samples) is reached:
                    generated_samples = 0
                    m = 0
                    while generated_samples < samples_to_create:

                        scale = np.random.uniform(0, 1)
                        x_new = x_class[m] + scale * (x_class[m] - centroid)
                        # print(X_new)

                        x_out = np.vstack((x_out, x_new.reshape(1, -1)))
                        y_out = np.hstack((y_out, cls))

                        generated_samples += 1
                        m += 1
                        if m >= num_min_samples:
                            m = 0

            # Dictionary mode: self._sampling_strategy explicitly declares the number of samples to be created per class
            elif isinstance(self._sampling_strategy, dict):
                idx = [p for p in range(y_res.shape[0]) if y_res[p] == cls]
                x_class = x_in[idx, :]
                num_min_samples = x_class.shape[0]

                samples_to_create = len(idx)
                centroid = np.mean(x_class, axis=0)

                # print("Minority samples of class", cls, ":\n", X_class)
                # print("Number of samples to create:", samples_to_create)
                # print("\tCluster", cls, " Centroid: ", centroid)

                # Keep creating samples until the desired number (=generated_samples) is reached:
                generated_samples = 0
                m = 0
                while generated_samples < samples_to_create:

                    scale = np.random.uniform(0, 1)
                    x_new = x_class[m] + scale * (x_class[m] - centroid)
                    # print(X_new)

                    x_out = np.vstack((x_out, x_new.reshape(1, -1)))
                    y_out = np.hstack((y_out, cls))

                    generated_samples += 1
                    m += 1
                    if m >= num_min_samples:
                        m = 0

        return x_out, y_out


class CBR:
    """
    Cluster-Based Over-sampler.

    An over-sampling algorithm for improving classification performance of imbalanced datasets.
    """
    def __init__(self, cluster_estimator='hac', cluster_resampler='cs', verbose=True, k_neighbors=1,
                 min_distance_factor=3, sampling_strategy='auto', random_state=0):
        """
        Initialize a Cluster-Based Resampler (CBR).

        Args:
            cluster_estimator (string): The clustering algorithm to be applied at the dataset. Values:

                * `hac` (default): Hierarchical Agglomerative Clustering.
                * `dbscan`: DBSCAN.
            cluster_resampler (string): The over-sampling algorithm to be applied at each cluster. Values:

                * `cs` (default): Centroid Sampler (implemented above)
                * `smote`: Synthetic Minority Oversampling Technique (SMOTE) - requires the `imblearn` library.
            verbose (bool): Print messages during execution.
            sampling_strategy (string or dictionary): How the algorithm generates samples:

                * 'auto': the model balances the dataset by oversampling the minority classes.
                * dict: a dictionary that indicates the number of samples to be generated from each class.
            k_neighbors: Hyper-parameter for `cluster_resampler='smote'`.
            min_distance_factor: Regulate the minimum distance for cluster merging in HAC. Decrease that value to
                increase the minimum allowed distance for cluster merging (fewer, yet larger clusters will be created).
            random_state:
        """

        self._cluster_estimator = cluster_estimator
        self._cluster_resampler = cluster_resampler
        self._sampling_strategy = sampling_strategy
        self.gcd = None  # Global Class Distribution
        self._majority_class = None
        self._random_state = random_state

        self._n_samples = 0
        self._n_clusters = 0

        self._verbose = verbose
        self._k_neighbors = k_neighbors
        self._min_distance_factor = min_distance_factor

    def display_info(self):
        print("Num samples:", self._n_samples)
        print("Dimensions:", self._input_dim)
        print("Num classes:", self._n_classes)
        print("Global Class Distribution:\n", self.gcd)

    # Compute initial stats
    def _fit(self, x_in, y_in):
        self._n_samples = x_in.shape[0]
        self._input_dim = x_in.shape[1]
        self._n_classes = len(set(y_in))

        # Compute class distribution for the entire dataset (GCD - global class distribution)
        y_res = np.array(y_in).reshape(-1, 1)
        self.gcd = np.array([[c, len(y_res[y_res == c])] for c in range(self._n_classes)])

        # Sort the array in decreasing order of sample population
        self.gcd = self.gcd[(-self.gcd[:, 1]).argsort()]

        # The majority class
        self._majority_class = self.gcd[0, 0]

        if self._verbose:
            self.display_info()

    def _perform_clustering(self, x_in):
        e_dists = euclidean_distances(x_in, x_in)
        med = np.median(e_dists)
        # men = np.mean(e_dists)
        # print("Mean distance:", men, "- Median distance:", med)

        eps = med / self._min_distance_factor

        if self._cluster_estimator == 'hac':
            clustering_method = AgglomerativeClustering(distance_threshold=eps, n_clusters=None, linkage='ward')
            clustering_method.fit(x_in)

        elif self._cluster_estimator == 'dbscan':
            clustering_method = DBSCAN(eps=eps, min_samples=3, metric='precomputed')
            clustering_method.fit(e_dists)

        else:
            print("Unsupported clustering method: ", self._cluster_estimator)
            return None

        self._n_clusters = len(set(clustering_method.labels_))

        return clustering_method.labels_

    def fit_resample(self, x_in, y_in):
        """`fit_resample` alleviates the problem of class imbalance in imbalanced datasets. The function renders CBR
        compatible with the `imblearn`'s interface, allowing its usage in over-sampling/under-sampling pipelines.

        Args:
            x_in (2D NumPy array): The training data instances.
            y_in (1D NumPy array): The classes of the training data instances.

        Returns:

            * The training data instances + the generated data instances.
            * The classes of the training data instances + the classes of the generated data instances.
        """

        self._fit(x_in, y_in)
        cluster_labels = self._perform_clustering(x_in)

        x_ret = []
        y_ret = []
        for cluster in tqdm(range(-1, self._n_clusters), desc="CBR Sampling        "):
            x_cluster_all = x_in[cluster_labels == cluster, :]
            y_cluster_all = y_in[cluster_labels == cluster]

            # Local (i.e. in-cluster) class distribution
            lcd = np.array([[c, len(y_cluster_all[y_cluster_all == c])] for c in range(self._n_classes)])
            max_samples_in_cluster = lcd.max(axis=0, initial=0)[1]

            if self._verbose:
                print("\n\n=================================\n===== CLUSTER", cluster, "- CLASS LABELS:", y_cluster_all)
                print("===== SAMPLES:\n", x_cluster_all)
                print("===== LOCAL CLASS DISTRIBUTION:\n", lcd)

            # If this is a singleton cluster and contains only one sample from the majority class
            if lcd[0, 0] == self._majority_class and lcd[0, 1] == 1 and np.sum(lcd[1:, 1]) == 0:
                # print("==== SINGLETON MAJORITY CLUSTER - ABORTING...")
                continue

            x_cluster_inc, y_cluster_inc, x_cluster_exc, y_cluster_exc = [], [], [], []
            included_classes = 0
            for cls in range(self._n_classes):
                if cls == self._majority_class:
                    # Include in cluster over-sampling
                    if lcd[cls, 1] == max_samples_in_cluster:
                        included_classes += 1
                        x_cluster_inc.extend(x_cluster_all[y_cluster_all == cls, :])
                        y_cluster_inc.extend(y_cluster_all[y_cluster_all == cls])
                    else:
                        x_cluster_exc.extend(x_cluster_all[y_cluster_all == cls, :])
                        y_cluster_exc.extend(y_cluster_all[y_cluster_all == cls])
                else:
                    if lcd[cls, 1] > 1:
                        included_classes += 1
                        x_cluster_inc.extend(x_cluster_all[y_cluster_all == cls, :])
                        y_cluster_inc.extend(y_cluster_all[y_cluster_all == cls])
                    else:
                        x_cluster_exc.extend(x_cluster_all[y_cluster_all == cls, :])
                        y_cluster_exc.extend(y_cluster_all[y_cluster_all == cls])

            if self._verbose:
                print("===== INCLUDED SAMPLES FOR OVER-SAMPLING:", np.array(x_cluster_inc).shape)
                print(np.array(x_cluster_inc))
                print("===== EXCLUDED SAMPLES FOR OVER-SAMPLING:", np.array(x_cluster_exc).shape)
                print(np.array(x_cluster_exc))

            # The samples that have been excluded from cluster over-sampling are copied to the output dataset
            x_ret.extend(x_cluster_exc)
            y_ret.extend(y_cluster_exc)

            # The samples that have been included in the cluster over-sampling process will be used as
            # reference points for data generation.
            if included_classes > 1:
                # print("Balancing cluster", cluster)

                if self._cluster_resampler == 'cs':
                    resampler = CentroidSampler(sampling_strategy=self._sampling_strategy,
                                                random_state=self._random_state)
                else:
                    resampler = SMOTE(k_neighbors=self._k_neighbors, sampling_strategy=self._sampling_strategy,
                                      random_state=self._random_state)

                x_1, y_1 = resampler.fit_resample(np.array(x_cluster_inc), y_cluster_inc)
                x_ret.extend(x_1)
                y_ret.extend(y_1)
            else:
                x_ret.extend(x_cluster_inc)
                y_ret.extend(y_cluster_inc)

            if self._verbose:
                print("===== NEW DATASET:", np.array(x_ret).shape)
                print(np.array(x_ret))

        return np.array(x_ret), np.array(y_ret)
