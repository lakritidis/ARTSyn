import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

class ctdCluster:
    """A typical cluster for ctdGAN."""
    def __init__(self, label=None, scaler=None, embedding_dim=32, clip=False,
                 continuous_columns=(), categorical_columns=(), random_state=0):
        """
        ctdCluster initializer. A typical cluster for ctdGAN.

        Args:
            label: The cluster's label
            scaler (string): A descriptor that defines a transformation on the cluster's data. Values:

              * '`None`'  : No transformation takes place; the data is considered immutable
              * '`stds`'  : Standard scaler
              * '`mms01`' : Min-Max scaler in the range (0,1)
              * '`mms11`' : Min-Max scaler in the range (-1,1) - so that data is suitable for tanh activations
              * 'yeo': Yeo-Johnson power transformer.
            embedding_dim (int): The dimensionality of the latent space (for the probability distribution)
            continuous_columns (tuple): The continuous columns in the input data
            categorical_columns (tuple): The columns in the input data that contain categorical variables
            clip (bool): If 'True' the reconstructed data will be clipped to their original minimum and maximum values.
            random_state (int): Seed the random number generators. Use the same value for reproducible results.
        """
        self._label = label
        self._embedding_dim = embedding_dim
        self._continuous_columns = continuous_columns
        self._categorical_columns = categorical_columns
        self._clip = clip
        self._random_state = random_state

        self._num_samples = 0

        self._min = None
        self._max = None

        self.class_distribution_ = None

        # Define the Data Transformation Model for the continuous columns
        if len(continuous_columns) > 0:
            if scaler == 'stds':
                self._scaler = StandardScaler()
            elif scaler == 'mms01':
                self._scaler = MinMaxScaler(feature_range=(0, 1))
            elif scaler == 'mms11':
                self._scaler = MinMaxScaler(feature_range=(-1, 1))
            elif scaler == 'yeo':
                self._scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            else:
                self._scaler = None
        else:
            self._scaler = None

    def fit(self, x, y=None, num_classes=0):
        """
            Compute cluster statistics and fit the selected Data Transformer.

        Args:
            x (NumPy array): The data to be transformed
            y (NumPy array): If the data has classes, pass them here. The ctdGAN will be trained for resampling.
            num_classes: The distinct number of classes in `y`.
        """
        self._num_samples = x.shape[0]

        # Min/Max column values are used for clipping.
        self._min = [np.min(x[:, i]) for i in self._continuous_columns]
        self._max = [np.max(x[:, i]) for i in self._continuous_columns]

        self.class_distribution_ = np.zeros(num_classes)
        if y is not None:
            unique_classes = np.unique(y, return_counts=True)
            n = 0
            for uv in unique_classes[0]:
                self.class_distribution_[uv] = unique_classes[1][n]
                n += 1

        if self._scaler is not None:
            x_cont = x[:, self._continuous_columns]
            self._scaler.fit(x_cont)

    def transform(self, x):
        if len(self._continuous_columns) == 0:
            return x

        if self._scaler is None:
            return x

        # print("Before Transformation:\n", x)
        x_cont = x[:, self._continuous_columns]
        transformed = self._scaler.transform(x_cont)
        # print("After Transformation of Continuous Cols:\n", transformed)

        for d_col in self._categorical_columns:
            transformed = np.insert(transformed, d_col, x[:, d_col], axis=1)
        # print("After Transformation & concat with Discrete Cols:\n", transformed)
        # exit()
        return transformed

    def fit_transform(self, x, y=None, num_classes=0):
        """Transform the sample vectors by applying the transformation function of `self._scaler`. In fact, this
        is a simple wrapper for the `fit_transform` function of `self._scaler`.

        `self._scaler` may implement a `Pipeline`.

        Returns:
            The transformed data.
        """
        self.fit(x, y, num_classes)
        return self._scaler.transform(x)

    def inverse_transform(self, x):
        """
        Inverse the transformation that has been applied by `self.fit_transform()`. In fact, this is a wrapper for the
        `inverse_transform` function of `self._scaler`, followed by a filter that clips the returned values.

        Args:
            x: The input data to be reconstructed (NumPy array).
            x: The input data to be reconstructed (NumPy array).

        Returns:
            The reconstructed data.
        """

        if self._scaler is not None:
            x_cont = x[:, self._continuous_columns]
            reconstructed_data = self._scaler.inverse_transform(x_cont)

            if self._clip:
                np.clip(reconstructed_data, self._min, self._max, out=reconstructed_data)

            for d_col in self._categorical_columns:
                reconstructed_data = np.insert(reconstructed_data, d_col, x[:, d_col], axis=1)
        else:
            # reconstructed_data = x.copy()
            reconstructed_data = x[:, 0:(x.shape[1] - 2)]

        return reconstructed_data

    def display(self):
        """
        Display useful cluster properties.
        """
        print("\t--- Cluster ", self._label, "-----------------------------------------")
        print("\t\t* Num Samples: ", self._num_samples)
        print("\t\t* Class Distribution: ", self.class_distribution_)
        print("\t\t* Min values per column:", self._min)
        print("\t\t* Max values per column:", self._max)
        print("\t---------------------------------------------------------\n")

    def get_label(self):
        return self._label

    def get_num_samples(self, c=None):
        if c is None:
            return self._num_samples
        else:
            if len(self.class_distribution_) > 0:
                return self.class_distribution_[c]
            else:
                return -1

    def set_label(self, v):
        self._label = v