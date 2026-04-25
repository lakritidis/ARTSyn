import numpy as np

from sklearn.ensemble import IsolationForest

from sklearn.cluster import KMeans, AgglomerativeClustering

from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import gower

from joblib import Parallel, delayed

from DeepCoreML.generators.ctd_cluster import ctdCluster
from sklearn.utils import resample


class ctdClusterer:
    """  ctdGAN data preprocessor.

    It partitions the real space into clusters; then, it transforms the data of each cluster.
    """
    def __init__(self, cluster_method='kmeans', max_clusters=10, scaler='mms11', samples_per_class=(), embedding_dim=32,
                 continuous_columns=(), categorical_columns=(), random_state=0):
        """
        Initializer

        Args
            cluster_method (str): The clustering algorithm to apply. Supported values:
              * kmeans: K-Means
              * hac: Hierarchical Agglomerative Clustering
              * gmm: Gaussian Mixture Model

            max_clusters (int): The maximum number of clusters to create
            scaler (string): A descriptor that defines a transformation on the cluster's data. Supported values:

              * '`None`'  : No transformation takes place; the data is considered immutable
              * '`stds`'  : Standard scaler
              * '`mms01`' : Min-Max scaler in the range (0,1)
              * '`mms11`' : Min-Max scaler in the range (-1,1) - so that data is suitable for tanh activations
            embedding_dim (int): The dimensionality of the latent space (for the probability distribution)
            continuous_columns (tuple): The continuous columns in the input data
            categorical_columns (tuple): The columns in the input data that contain categorical variables
            samples_per_class (List or tuple of integers): Contains the number of samples per class
            random_state: Seed the random number generators. Use the same value for reproducible results.
        """
        self._cluster_method = cluster_method
        if cluster_method not in ['None', 'kmeans', 'hac', 'gmm', 'kprot']:
            self._cluster_method = 'kmeans'

        self._max_clusters = max_clusters
        if self._max_clusters < 2:
            self._cluster_method = 'None'

        self._scaler = scaler
        if scaler not in ['None', 'stds', 'mms01', 'mms11', 'yeo']:
            self._scaler = 'mms11'

        self._samples_per_class = samples_per_class
        self._embedding_dim = embedding_dim
        self._continuous_columns = continuous_columns
        self._categorical_columns = categorical_columns
        self._random_state = random_state

        self.num_clusters_ = 0
        self.clusters_ = []
        self.cluster_labels_ = None
        self.probability_matrix_ = None
        self.imbalance_matrix_ = None

    def perform_clustering(self, x_train, y_train, num_classes, pac):
        """

        Args:
            x_train: Training data
            y_train: The classes of the training samples
            num_classes: The number of distinct classes of the training data
            pac (int): The number of samples to group together as input to the Critic.

        Returns:
            Transformed data
        """
        # STEP 1: Prepare the data for clustering:
        # For KMeans or Gaussian Mixture
        k_range = range(2, self._max_clusters)
        if self._cluster_method == 'kmeans' or self._cluster_method == 'gmm':
            # MinMax scale the continuous variables, and 2) OneHotEncode the discrete variables.
            column_transformer = ColumnTransformer([
                ("mms", MinMaxScaler(), self._continuous_columns),
                ("ohe", OneHotEncoder(), self._categorical_columns)
            ], sparse_threshold=0)
            x_scaled = column_transformer.fit_transform(x_train)

            # Find the optimal number of clusters (best_k).
            # Perform multiple executions and pick the one that produces the minimum scaled inertia.
            # jobs = 0.2 * multiprocessing.cpu_count()
            jobs = -1
            scores = Parallel(n_jobs=jobs)(delayed(self._scaled_inertia)(x_scaled, k) for k in k_range)
            best = min(scores, key=lambda score_tuple: score_tuple[1])
            self.num_clusters_ = best[0]
            best_cov_type = best[2]

            print("\t\tEstimated number of clusters:", self.num_clusters_, "- Categorical columns:",
                  np.array(self._categorical_columns).astype(int))

            # After the optimal number of clusters best_k has been determined, execute one last k-Means with best_k
            # clusters, or GMM with best_k clusters and best_cov_type covariance type.
            if self._cluster_method == 'gmm':
                cluster_method = GaussianMixture(n_components=self.num_clusters_, covariance_type=best_cov_type,
                                                 random_state=self._random_state)
            else:
                cluster_method = KMeans(n_clusters=self.num_clusters_, n_init='auto', init='k-means++', random_state=self._random_state)

            self.cluster_labels_ = cluster_method.fit_predict(x_scaled)

            # For Hierarchical Agglomerative clustering (HAC), we first compute the Gower distance matrix and then, we apply
        # Agglomerative clustering to that matrix. The ideal number of clusters is determined by the max Silhouette score.
        elif self._cluster_method == 'hac':
            categorical_mask = np.zeros(x_train.shape[1], dtype=bool)
            categorical_mask[list(self._categorical_columns)] = True

            #stability_scores = self.stability_curve_optimized(x_train, categorical_mask, k_range=k_range, b=10, sample_fraction=0.6)
            #self.num_clusters_ = k_range[np.argmax(stability_scores)]
            print("\t\tEstimated number of clusters:", self.num_clusters_, "- Categorical columns:", np.array(self._categorical_columns).astype(int))
            gower_distance_matrix = gower.gower_matrix(x_train.astype(float), cat_features=categorical_mask)

            cluster_method = AgglomerativeClustering(n_clusters=self.num_clusters_, metric="precomputed", linkage="average")
            self.cluster_labels_ = cluster_method.fit_predict(gower_distance_matrix)

        # For HDBSCAN, we do not have to estimate the ideal numer of clusters. We first compute the Gower distance
        # matrix and then, we apply HDBSCAN to that distance matrix.
        elif self._cluster_method == 'kprot':
            num_runs = 2

            # If the dataset is too large, take a sample of it.
            sample_fraction = 0.7
            if x_train.shape[0] >= 3000:
                sample_fraction = 0.3 + 0.7 * np.exp(-0.00025 * (x_train.shape[0] - 3000))

            # scaler for the numerical data (used for clustering)
            scaler = MinMaxScaler()

            if len(self._categorical_columns) > 0 and len(self._continuous_columns) > 0:
                print("\t\tRunning with k-Prototypes")

                # Scale the numerical features, leave the categorical ones intact.
                x_scaled = x_train.copy()
                numerical_data = x_scaled[:, self._continuous_columns].astype(float)

                # The Gamma parameter controls the importance of categorical vs numeric columns
                gamma = numerical_data.std(axis=0).mean()
                numerical_data = scaler.fit_transform(numerical_data)
                x_scaled[:, self._continuous_columns] = numerical_data

                # Estimate the best number of clusters
                self.num_clusters_ = self.stability_analysis_parallel(x_scaled, k_values=k_range, gamma=gamma,
                                                                      n_runs=num_runs, sample_frac=sample_fraction,
                                                                      random_state=self._random_state)

                cluster_method = KPrototypes(n_clusters=self.num_clusters_, init='Cao', n_init=10, gamma=None,
                                             verbose=0, random_state=self._random_state)
                self.cluster_labels_ = cluster_method.fit_predict(x_scaled, categorical=self._categorical_columns)

            elif len(self._categorical_columns) > 0 and len(self._continuous_columns) == 0:
                print("\t\tRunning with k-Modes")

                # Estimate the best number of clusters
                #if current_dataset_name == 'nursery':
                #    self.num_clusters_ = 10
                #else:
                self.num_clusters_ = self.stability_analysis_parallel(x_train, k_values=k_range, gamma=0,
                                                                      n_runs=num_runs, sample_frac=sample_fraction,
                                                                      random_state=self._random_state)

                cluster_method = KModes(n_clusters=self.num_clusters_, init='Cao', n_init=10, verbose=0,
                                        random_state=self._random_state)
                self.cluster_labels_ = cluster_method.fit_predict(x_train)

            else:
                print("\t\tRunning with k-Means")
                sample_fraction = 0.7
                # Scale the numerical features, leave the categorical ones intact.
                x_scaled = scaler.fit_transform(x_train)

                # Estimate the best number of clusters
                self.num_clusters_ = self.stability_analysis_parallel(x_scaled, k_values=k_range, gamma=0,
                                                                      n_runs=num_runs, sample_frac=sample_fraction,
                                                                      random_state=self._random_state)

                cluster_method = KMeans(n_clusters=self.num_clusters_, n_init='auto', init='k-means++',
                                        random_state=self._random_state)
                self.cluster_labels_ = cluster_method.fit_predict(x_scaled)

        # Ablation Study Only:
        elif self._cluster_method == 'None':
            print("\t\t\tRunning in Ablation Mode with 1 cluster")
            self.num_clusters_ = 1
            self.cluster_labels_ = np.zeros(x_train.shape[0])

        # STEP 2: POST-CLUSTERING PROCESSING
        # Partition the dataset and create the appropriate Cluster objects.
        transformed_data = None
        for u in range(self.num_clusters_):
            x_u = x_train[self.cluster_labels_ == u, :]
            y_u = y_train[self.cluster_labels_ == u]

            cluster = ctdCluster(label=u, scaler=self._scaler,
                                 clip=False, embedding_dim=self._embedding_dim,
                                 continuous_columns=self._continuous_columns, categorical_columns=self._categorical_columns,
                                 random_state=self._random_state)
            cluster.fit(x_u, y_u, len(self._samples_per_class))

            # Transform the data in a cluster-wise manner.
            x_transformed = cluster.transform(x_u)
            # print("transformed:", x_transformed[0:10, :])
            # exit()
            cluster_labels = (u * np.ones(y_u.shape[0])).reshape(-1, 1)
            class_labels = np.array(y_u).reshape(-1, 1)

            if u == 0:
                transformed_data = np.concatenate((x_transformed, cluster_labels, class_labels), axis=1)
            else:
                concat = np.concatenate((x_transformed, cluster_labels, class_labels), axis=1)
                transformed_data = np.concatenate((transformed_data, concat))

            self.clusters_.append(cluster)

        # Construct the probability matrix; Each element (i,j) stores the conditional probability
        # P(cluster==u | class=y) = P( (class==y) AND (cluster==u) ) / P(class==y)
        if num_classes > 1:
            self.probability_matrix_ = np.zeros((num_classes, self.num_clusters_))
            self.imbalance_matrix_ = np.zeros((num_classes, self.num_clusters_))
            for c in range(num_classes):
                for u in range(self.num_clusters_):
                    cluster = self.clusters_[u]
                    # print("\t Cluster:", comp, "- Samples:", cluster.get_num_samples(),
                    #      "(", cluster.get_num_samples(c), "from class", c, ")")
                    self.probability_matrix_[c][u] = cluster.get_num_samples(c) / self._samples_per_class[c]
                    self.imbalance_matrix_[c][u] = cluster.get_num_samples(c)

        # Pad the dataset to align with the pac parameter (Create integral number of groups of pac samples).
        dataset_rows = transformed_data.shape[0]
        if dataset_rows % pac != 0:
            required_samples = pac * (dataset_rows // pac + 1) - dataset_rows
            random_samples = transformed_data[np.random.randint(0, dataset_rows, (required_samples,))]
            transformed_data = np.vstack((transformed_data, random_samples))

        # Shuffle the dataset
        np.random.shuffle(transformed_data)

        return transformed_data

    def remove_majority_outliers(self, x_train, y_train):
        num_samples = x_train.shape[1]
        majority_class = np.argmax(self._samples_per_class)
        maj_samples = np.array([x_train[s, :] for s in range(num_samples) if y_train[s] == majority_class])

        # Use an Isolation Forest to detect the outliers. The predictions array marks them with -1
        outlier_detector = IsolationForest(random_state=self._random_state)
        outlier_detector.fit(maj_samples)
        predictions = outlier_detector.predict(maj_samples)

        # Copy all the minority samples to the cleaned dataset
        x_clean = np.array([x_train[s, :] for s in range(num_samples) if y_train[s] != majority_class])
        y_clean = np.array([y_train[s] for s in range(num_samples) if y_train[s] != majority_class])

        # Copy the majority samples that are not outliers to the cleaned dataset
        for s in range(maj_samples.shape[0]):
            if predictions[s] == 1:
                x_clean = np.append(x_clean, [maj_samples[s, :]], axis=0)
                y_clean = np.append(y_clean, [majority_class], axis=0)

        # (x_clean, y_clean) is the new dataset without the outliers
        # print("Clean Dataset Shape:", x_clean.shape)


    def _scaled_inertia(self, scaled_data, num_clusters, alpha_k=0.02):
        """
        Args:
        scaled_data: matrix
            scaled data. rows are samples and columns are features for clustering
        max_clusters: int
            current k for applying KMeans
        alpha_k: float
            manually tuned factor that gives penalty to the number of clusters

        Returns:
            scaled_inertia: float
                scaled inertia value for current k
        """

        ret_val = 0
        cov_type = 'None'
        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()

        if self._cluster_method == 'kmeans':
            kmeans = KMeans(n_clusters=num_clusters, random_state=self._random_state, n_init='auto')
            kmeans.fit(scaled_data)
            ret_val = kmeans.inertia_ / inertia_o + alpha_k * num_clusters

        elif self._cluster_method == 'gmm':
            min_bic = 10 ** 9
            for cov in ['spherical', 'tied', 'diag', 'full']:
                gmm = GaussianMixture(n_components=num_clusters, covariance_type=cov, random_state=self._random_state)
                gmm.fit(scaled_data)
                bic_score = gmm.bic(scaled_data)
                if bic_score < min_bic:
                    min_bic = bic_score
                    cov_type = cov
            ret_val = min_bic

        return num_clusters, ret_val, cov_type

    def _fit_single_run(self, scaled_data, num_clusters, gamma, indices, random_state):
        if len(self._categorical_columns) > 0 and len(self._continuous_columns) > 0:
            model = KPrototypes(n_clusters=num_clusters, init='Cao', n_init=2, gamma=None, verbose=0, random_state=random_state)
            labels = model.fit_predict(scaled_data[indices], categorical=self._categorical_columns)

        elif len(self._categorical_columns) > 0 and len(self._continuous_columns) == 0:
            model = KModes(n_clusters=num_clusters, init='Cao', n_init=2, verbose=0, random_state=random_state)
            labels = model.fit_predict(scaled_data[indices])

        else:
            model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=random_state)
            labels = model.fit_predict(scaled_data[indices])

        return indices, labels

    def stability_analysis_parallel(self, scaled_data, k_values=range(2, 11), gamma=0, n_runs=5, sample_frac=0.7, random_state=0):
        np.random.seed(random_state)
        results = {}
        n = scaled_data.shape[0]
        sample_size = int(sample_frac * n)

        for k in k_values:
            # Prepare subsamples
            seeds = [random_state + i for i in range(n_runs)]
            subsamples = [
                resample(np.arange(n), replace=False, n_samples=sample_size, random_state=seed)
                for seed in seeds
            ]

            # Parallel model fitting
            runs = Parallel(n_jobs=-1)(delayed(self._fit_single_run)(scaled_data, k, gamma, indices, seed)
                for indices, seed in zip(subsamples, seeds)
            )

            # Compute ARI (can also parallelize if needed)
            ari_scores = []
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    idx_i, labels_i = runs[i]
                    idx_j, labels_j = runs[j]

                    common, i_pos, j_pos = np.intersect1d(idx_i, idx_j, return_indices=True)

                    if len(common) > 0:
                        ari = adjusted_rand_score(labels_i[i_pos], labels_j[j_pos])
                        ari_scores.append(ari)

            results[k] = np.mean(ari_scores)

        window = 3
        keys = np.array(list(results.keys()))
        vals = np.array(list(results.values()))
        smooth_vals = []
        for i in range(len(vals)):
            start = max(0, i - window // 2)
            end = min(len(vals), i + window // 2 + 1)
            smooth_vals.append(sum(vals[start:end]) / (end - start))

        #smoothed_vals = np.convolve(vals, np.ones(window) / window, mode='same')
        #print(vals)
        #print("Smooth vals:", smooth_vals)
        #print("Smoothed (convolved) vals: ", smoothed_vals)

        for d in range(4, 6):
            for i in range(1, len(smooth_vals)):
                delta = abs(smooth_vals[i] - smooth_vals[i - 1])
                if delta < d / 100 and smooth_vals[i] >= 0.4:
                    print("Estimated Number of clusters:", keys[i - 1])
                    return keys[i]  # first plateau point

        print("Estimated Number of clusters:", keys[-1])
        return keys[-1]

    def display(self):
        print("Num Clusters: ", self.num_clusters_)
        for i in range(self.num_clusters_):
            self.clusters_[i].display()

    def get_cluster(self, i):
        return self.clusters_[i]

    def get_cluster_center(self, i):
        return self.clusters_[i].center_
