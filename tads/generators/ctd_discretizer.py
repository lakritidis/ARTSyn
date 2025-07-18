import sys
import numpy as np

from collections import Counter

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.base import BaseEstimator, TransformerMixin


class ctdDiscretizer:
    def __init__(self, strategy=None, bins='auto', bin_weights=None, random_state=None):
        """
        Continuous variables discretizer

        Args:
            strategy(str, None):
              *  None: No discretization takes place - fit, transform and fit_transform do not touch the input data
              * '`bins-uni`' :  Discretize the continuous variables uniformly (bins of equal widths)
              * '`bins-q`'   :  Discretize the continuous variables with equal quantile (bins of same populations)
              * '`bins-k`'   :  Discretize the continuous variables with k-means clustering (values in each bin have
                                the same nearest center of a 1D k-means cluster)
              * '`bins-bgm`' :  Discretize the continuous variables with clustering based on Bayesian Gaussian Mixtures
              * '`chi-merge`' : Discretize the continuous variables with the ChiMerge method
              * '`caim`'     : Discretize the continuous variables with the CAIM algorithm

            bins (int, 'auto-bgm'): The number of bins to use for discretization. If 'auto', apply heuristic methods.
            bin_weights (None, 'auto'):
              * None: all bins will get the same weight equal 1.0
              * 'auto': the weight of each bin will be determined by a Bayesian Gaussian Mixture.
            random_state:
        """
        self.original_continuous_idx = []
        self.transformed_continuous_idx = []
        self._class_column = None

        self._discretization_models = []
        self._strategy = strategy
        self._bins = bins
        self._bin_weights = bin_weights

        if strategy in ('bins-uni', 'bins-q', 'bins-k', 'bins-bgm', 'chi-merge', 'caim'):
            self._strategy = strategy
            self._bins = bins

        self._random_state = random_state

    def fit(self, train_data, class_data, continuous_columns):
        self.original_continuous_idx = continuous_columns
        self._class_column = class_data

        # For each continuous column, fit a discretization model
        col = 0
        for c in self.original_continuous_idx:
            self.transformed_continuous_idx.append(col)
            col += 1

            continuous_col = train_data[:, c].reshape(-1, 1)

            # Fit a Bayesian Gaussian Mixture to automatically determine the optimal number of bins and the weight of
            # each bin.
            bg_mix = None
            temp_data = None
            if self._bins == 'auto-bgm' or self._bin_weights == 'auto':
                bg_mix = BayesianGaussianMixture(n_components=10, weight_concentration_prior=None,
                                                 max_iter=100, n_init=1, random_state=self._random_state)
                temp_data = bg_mix.fit_predict(continuous_col)

            # If the bins parameter has been set to 'auto', determine its value from the unique clusters created
            # by a Bayesian Gaussian Mixture (BGM) model.
            num_bins = self._bins
            if self._bins == 'auto-bgm':
                num_bins = len(np.unique(temp_data))
                if num_bins == 1:
                    num_bins = 2
            elif self._bins is None:
                num_bins = 2

            if self._strategy == 'bins-uni':
                model = KBinsDiscretizer(strategy='uniform', encode="ordinal", n_bins=num_bins,
                                         random_state=self._random_state)
            elif self._strategy == 'bins-q':
                model = KBinsDiscretizer(strategy='quantile', encode="ordinal", n_bins=num_bins,
                                         random_state=self._random_state)
            elif self._strategy == 'bins-k':
                model = KBinsDiscretizer(strategy='kmeans', encode="ordinal", n_bins=num_bins,
                                         random_state=self._random_state)
            elif self._strategy == 'bins-bgm':
                model = BayesianGaussianMixture(n_components=num_bins, max_iter=100, n_init=1,
                                                random_state=self._random_state)
            elif self._strategy == 'chi-merge':
                model = ChiMerge(max_num_bins=num_bins, class_data=class_data, random_state=self._random_state)

            elif self._strategy == 'caim':
                model = CAIMD(class_data=class_data, random_state=self._random_state)

            else:
                model = None

            # print("\tContinuous Column =", c)
            # print("\t\tNum Bins =", num_bins)
            # print("\t\tModel =", model)

            # Determine the weights of the bins: If none, then each bin gets an equal weight equal to 1. If 'auto',
            # then the weights correspond to the weights of the components of the Bayesian Gaussian Mixture bg_mix
            if self._bin_weights == 'auto':
                bin_weights = bg_mix.weights_
            else:
                bin_weights = [1.0 for _ in range(num_bins)]

            # Fit the discretizer
            if model:
                model.fit(continuous_col)
                self._discretization_models.append((model, num_bins, bin_weights))

        return self

    def transform(self, data):
        col = 0
        transformed_data = np.copy(data)
        for c in self.original_continuous_idx:
            discretizer = self._discretization_models[col][0]

            transform_op = getattr(discretizer, "transform", None)
            predict_op = getattr(discretizer, "predict", None)

            # Transform for scikit-learn discretization methods (e.g. kBinsDiscretizer)
            if callable(transform_op):
                transformed_col = discretizer.transform(data[:, c].reshape(-1, 1))
                transformed_data[:, c] = transformed_col.reshape(-1)

            # Predict for clustering algorithms (like Bayesian Gaussian Mixture)
            elif callable(predict_op):
                transformed_col = discretizer.predict(data[:, c].reshape(-1, 1))
                transformed_data[:, c] = transformed_col.reshape(-1)
            else:
                print("The discretizer implements neither 'transform', nor 'predict'", file=sys.stderr)
                exit(1)

            col += 1
        # print(transformed_data)
        return transformed_data

    def fit_transform(self, data, class_data, continuous_columns=None):
        self.fit(data, class_data, continuous_columns)
        return self.transform(data)


class CAIMD(BaseEstimator, TransformerMixin):

    def __init__(self, class_data=None, random_state=0):
        """
        CAIM discretization class

        Args:
            class_data:
            random_state:
        """
        self._class_data = class_data
        self._split_scheme = None
        self._random_state = random_state

    def fit(self, x_train):
        """
        Fit CAIM

        Args:
            x_train:

        Returns:

        """
        self._split_scheme = dict()

        min_splits = np.unique(self._class_data).shape[0]

        xj = x_train[np.invert(np.isnan(x_train))]
        new_index = xj.argsort()
        xj = xj[new_index]
        yj = self._class_data[new_index]

        all_splits = np.unique(xj)[1:-1].tolist()  # potential split points

        global_caim = -1
        main_scheme = [xj[0], xj[-1]]
        best_caim = 0
        k = 1

        while (k <= min_splits) or ((global_caim < best_caim) and all_splits):
            split_points = np.random.permutation(all_splits).tolist()
            best_scheme = None
            best_point = None
            best_caim = 0
            k = k + 1
            while split_points:
                scheme = main_scheme[:]
                sp = split_points.pop()
                scheme.append(sp)
                scheme.sort()
                c = self.get_caim(scheme, xj, yj)
                if c > best_caim:
                    best_caim = c
                    best_scheme = scheme
                    best_point = sp
            if (k <= min_splits) or (best_caim > global_caim):
                main_scheme = best_scheme
                global_caim = best_caim
                try:
                    all_splits.remove(best_point)
                except ValueError:
                    print('The feature does not have enough unique values for discretization!' +
                          ' Add it to categorical list!')

        self._split_scheme = main_scheme
        # print('#', j, ' GLOBAL CAIM ', global_caim)
        return self

    def transform(self, data):
        """
        Discretize X using a split scheme obtained with CAIM.

        Args:
            data: array-like or pandas dataframe, shape [n_samples, n_features]. Input array can contain missing values.

        Returns:
            X_di : sparse matrix if sparse=True else a 2-d array, dtype=int Transformed input.

        """
        x_di = data.copy()

        sh = self._split_scheme

        sh[-1] = sh[-1] + 1
        xj = data
        # xi = xi[np.invert(np.isnan(xi))]
        for i in range(len(sh) - 1):
            ind = np.where((xj >= sh[i]) & (xj < sh[i + 1]))[0]
            x_di[ind] = i

        return x_di

    def get_caim(self, scheme, xi, y):
        sp = self.index_from_scheme(scheme[1:-1], xi)
        sp.insert(0, 0)
        sp.append(xi.shape[0])
        n = len(sp) - 1
        isum = 0
        for j in range(n):
            init = sp[j]
            fin = sp[j + 1]
            Mr = xi[init:fin].shape[0]
            val, counts = np.unique(y[init:fin], return_counts=True)
            maxr = counts.max()
            isum = isum + (maxr / Mr) * maxr
        return isum / n

    @staticmethod
    def index_from_scheme(scheme, x_sorted):
        split_points = []
        for p in scheme:
            split_points.append(np.where(x_sorted > p)[0][0])
        return split_points

    @staticmethod
    def check_categorical(X, y):
        categorical = []
        ny2 = 2 * np.unique(y).shape[0]
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj[np.invert(np.isnan(xj))]
            if np.unique(xj).shape[0] < ny2:
                categorical.append(j)
        return categorical


class ChiMerge:
    def __init__(self, max_num_bins, class_data=None, random_state=0):
        """

        Args:
            max_num_bins:
            class_data:
            random_state:
        """
        self._max_intervals = max_num_bins
        self._intervals = None
        self._class_data = class_data

        if class_data is None:
            print("Cannot apply ChiMerge discretization without a class column")

        self._random_state = random_state

    def fit(self, x_train):
        data = np.hstack((x_train.reshape(-1, 1), self._class_data.reshape(-1, 1)))

        # Sort the distinct values
        distinct_vals = np.sort(np.unique(x_train))

        # Get all possible labels
        labels = np.sort(np.unique(self._class_data))

        # A helper function for padding the Counter()
        empty_count = {label: 0 for label in labels}

        # Initialize the intervals for each attribute
        self._intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))]
        # print(self._intervals)
        while len(self._intervals) > self._max_intervals:
            chi = []

            for i in range(len(self._intervals) - 1):
                # Calculate the Chi2 value
                obs0 = pd.DataFrame([x for x in data if self._intervals[i][0] <= x[0] <= self._intervals[i][1]],
                                    columns=[str('feature'), str('class')])
                obs1 = pd.DataFrame([x for x in data if self._intervals[i + 1][0] <= x[0] <= self._intervals[i + 1][1]],
                                    columns=[str('feature'), str('class')])
                total = len(obs0) + len(obs1)

                count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0['class'])}.items()])
                count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1['class'])}.items()])
                count_total = count_0 + count_1

                expected_0 = count_total * sum(count_0) / total
                expected_1 = count_total * sum(count_1) / total
                chi_ = (count_0 - expected_0)**2 / expected_0 + (count_1 - expected_1)**2/expected_1

                # Deal with the zero counts
                chi_ = np.nan_to_num(chi_)

                # Do the summation for Chi2
                chi.append(sum(chi_))

            # Find the minimal Chi2 for current iteration
            min_chi = min(chi)
            min_chi_index = 0
            for i, v in enumerate(chi):
                if v == min_chi:
                    min_chi_index = i
                    break

            # Prepare for the merged new data array
            new_intervals = []
            skip = False
            done = False
            for i in range(len(self._intervals)):
                if skip:
                    skip = False
                    continue

                # Merge the intervals
                if i == min_chi_index and not done:
                    t = self._intervals[i] + self._intervals[i + 1]
                    new_intervals.append([min(t), max(t)])
                    skip = True
                    done = True
                else:
                    new_intervals.append(self._intervals[i])

            self._intervals = new_intervals

        # for i in self._intervals:
        #    print('[', i[0], ',', i[1], ']', sep='')

    def transform(self, data):
        transformed_data = []

        for x in data:
            i_idx = 0
            for i in self._intervals:
                if i[0] <= x <= i[1]:
                    transformed_data.append(i_idx)
                i_idx += 1

        return np.array(transformed_data)
