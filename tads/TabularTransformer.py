# Tabular Transformer object: based on the Data Transformer of ctGAN
# Forked and modified from https://github.com/sdv-dev/CTGAN

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder

from collections import namedtuple
import multiprocessing

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'column_max', 'column_min', 'output_info', 'output_dimensions'
    ]
)


class TabularTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """
    def __init__(self, cont_normalizer='None', max_clusters=10, weight_threshold=0.005, with_mean=True, with_std=True,
                 clip=False):
        """Create a data transformer.

        Args:
            cont_normalizer: Normalizer for the continuous columns:

              * '`None`'     : No transformation on the continuous columns takes place; the data is considered immutable
              * '`vgm`'      : Variational Gaussian Mixture + One-hot-encoded component labels.
              * '`stds`'     : Standard scaler
              * '`mms01`'    : Min-Max scaler in the range (0,1)
              * '`mms11`'    : Min-Max scaler in the range (-1,1) - so that data is suitable for tanh activations
              * '`stds-pca`' : Standard scaler & Principal Component Analysis
              * '`yeo`'      : Yeo-Johnson power transformer.

            max_clusters: Max number of Gaussian distributions in Bayesian GMM; used when `cont_normalizer='vgm'`
            weight_threshold: Weight threshold for a Gaussian distribution to be kept; used when `cont_normalizer='vgm'`
            with_mean: If True, it centers the data before scaling. This does not work (and will raise an exception)
                when attempted on sparse matrices, because centering them entails building a dense matrix which in
                common use cases is likely to be too large to fit in memory; used when `cont_normalizer='stds'`.
            with_std: If `True`, scale the data to unit variance (or equivalently, unit standard deviation);
                used when `cont_normalizer='stds'`.
            clip: If 'True' the reconstructed data will be clipped to their original minimum and maximum values.
        """
        self._cont_normalizer = cont_normalizer
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold
        self._with_mean = with_mean
        self._with_std = with_std
        self._column_raw_dtypes = []
        self._column_transform_info_list = []
        self._clip = clip
        self.output_info_list = []
        self.output_dimensions = 0
        self.ohe_dimensions = 0
        self.dataframe = True

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame): A dataframe containing a column.

        Returns:
            namedtuple: A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        max_val = np.array(data.max(axis=0))[0]
        min_val = np.array(data.min(axis=0))[0]

        cti = None
        if self._cont_normalizer == 'vgm':
            tran = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), self._max_clusters))
            tran.fit(data, column_name)
            num_components = sum(tran.valid_component_indicator)

            cti = ColumnTransformInfo(column_name=column_name, column_type='continuous', transform=tran,
                                      column_max=max_val, column_min=min_val,
                                      output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
                                      output_dimensions=1 + num_components)

        elif self._cont_normalizer == 'stds':
            tran = StandardScaler(with_std=self._with_std, with_mean=self._with_mean)
            tran.fit(data)

            cti = ColumnTransformInfo(column_name=column_name, column_type='continuous', transform=tran,
                                      column_max=max_val, column_min=min_val,
                                      output_info=[SpanInfo(1, 'tanh')], output_dimensions=1)

        elif self._cont_normalizer == 'mms01':
            tran = MinMaxScaler(feature_range=(0, 1))
            tran.fit(data)

            cti = ColumnTransformInfo(column_name=column_name, column_type='continuous', transform=tran,
                                      column_max=max_val, column_min=min_val,
                                      output_info=[SpanInfo(1, 'tanh')], output_dimensions=1)

        elif self._cont_normalizer == 'mms11':
            tran = MinMaxScaler(feature_range=(-1, 1))
            tran.fit(data)

            cti = ColumnTransformInfo(column_name=column_name, column_type='continuous', transform=tran,
                                      column_max=max_val, column_min=min_val,
                                      output_info=[SpanInfo(1, 'tanh')], output_dimensions=1)

        elif self._cont_normalizer == 'stds-pca':
            tran = Pipeline(steps=[("std", StandardScaler(with_std=self._with_std, with_mean=self._with_mean)),
                                   ("pca", PCA())])
            tran.fit(data)

            cti = ColumnTransformInfo(column_name=column_name, column_type='continuous', transform=tran,
                                      column_max=max_val, column_min=min_val,
                                      output_info=[SpanInfo(1, 'tanh')], output_dimensions=1)

        elif self._cont_normalizer == 'yeo':
            tran = PowerTransformer(method='yeo-johnson', standardize=True)
            tran.fit(data)

            cti = ColumnTransformInfo(column_name=column_name, column_type='continuous', transform=tran,
                                      column_max=max_val, column_min=min_val,
                                      output_info=[SpanInfo(1, 'tanh')], output_dimensions=1)

        elif self._cont_normalizer == 'None':
            cti = ColumnTransformInfo(column_name=column_name, column_type='continuous', transform=None,
                                      column_max=max_val, column_min=min_val,
                                      output_info=[SpanInfo(1, 'tanh')], output_dimensions=1)

        return cti

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame): A dataframe containing a column.

        Returns:
            namedtuple: A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)
        self.ohe_dimensions += num_categories

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            column_max=-1, column_min=-1,
            output_info=[SpanInfo(num_categories, 'softmax')], output_dimensions=num_categories)

    def fit(self, raw_data, discrete_columns=()):
        """Fit the ``DataTransformer`` in a column-wise fashion. One transformer is fitted per column.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a ``OneHotEncoder`` for discrete columns. This
        step also counts the #columns in matrix data and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False

            # work around for RDT issue #328 Fitting with numerical column names fails
            discrete_columns = [str(column) for column in discrete_columns]
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        self._column_raw_dtypes = raw_data.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(raw_data[[column_name]])
            else:
                column_transform_info = self._fit_continuous(raw_data[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        output = None
        if self._cont_normalizer == 'vgm':
            column_name = data.columns[0]
            flattened_column = data[column_name].to_numpy().flatten()
            data = data.assign(**{column_name: flattened_column})
            gm = column_transform_info.transform
            transformed = gm.transform(data)

            # Converts the transformed data to the appropriate output format. The first column (ending in '.normalized')
            # stays the same, but the label encoded column (ending in '.component') is one hot encoded.
            output = np.zeros((len(transformed), column_transform_info.output_dimensions))
            output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
            index = transformed[f'{column_name}.component'].to_numpy().astype(int)
            output[np.arange(index.size), index + 1] = 1.0

        elif (self._cont_normalizer == 'stds' or self._cont_normalizer == 'mms01' or self._cont_normalizer == 'mms11'
              or self._cont_normalizer == 'stds-pca' or self._cont_normalizer == 'yeo'):
            output = column_transform_info.transform.transform(data)

        elif self._cont_normalizer == 'None':
            return data

        return output

    def transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def _synchronous_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns synchronous.

        Outputs a list with Numpy arrays.
        """
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]

            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self.transform_discrete(column_transform_info, data))

        return column_data_list

    def _parallel_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]

            if column_transform_info.column_type == 'continuous':
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self.transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=-1)(processes)
        # return Parallel(n_jobs=0.5 * multiprocessing.cpu_count())(processes)

    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes. Otherwise, the transformation will be slower.
        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(raw_data, self._column_transform_info_list)
        else:
            column_data_list = self._parallel_transform(raw_data, self._column_transform_info_list)

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        ret_data = None
        encoder = column_transform_info.transform

        if self._cont_normalizer == 'vgm':
            data = pd.DataFrame(column_data[:, :2], columns=list(encoder.get_output_sdtypes()))
            data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
            if sigmas is not None:
                selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
                data.iloc[:, 0] = selected_normalized_value

            ret_data = encoder.reverse_transform(data)

        elif self._cont_normalizer == 'stds' or self._cont_normalizer == 'mms11' or self._cont_normalizer == 'stds-pca'\
                or self._cont_normalizer == 'yeo':
            ret_data = encoder.inverse_transform(column_data)

        elif self._cont_normalizer == 'None':
            ret_data = column_data

        # Apply value clipping here: Given an interval, values outside the interval are clipped to the interval edges.
        if self._clip:
            ret_data = np.clip(ret_data, column_transform_info.column_min, column_transform_info.column_max)

        return ret_data

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_col_data = self._inverse_transform_continuous(column_transform_info, column_data, sigmas, st)
            else:
                recovered_col_data = self._inverse_transform_discrete(column_transform_info, column_data)

            recovered_column_data_list.append(recovered_col_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(self._column_raw_dtypes)
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }

    def get_column_transform_info_list(self):
        return self._column_transform_info_list
