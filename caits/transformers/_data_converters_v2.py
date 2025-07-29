from sklearn.base import BaseEstimator, TransformerMixin
from caits.dataset._dataset3 import CaitsArray

class DatasetToArray(BaseEstimator, TransformerMixin):
    def __init__(self, flatten=False, dtype=None):
        """Initializes the DatasetToArray transformer.

        Args:
            flatten (bool): If True, the output array will be flattened
                            (window_size * channels). Otherwise (default),
                            it will be a 3D array.
            dtype (str): The data type of the output array. If None, the
                         default NumPy data type will be used.
        """
        self.flatten = flatten
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit method (no-op since nothing is learned)."""
        return self

    def transform(self, X):
        """Transforms the Dataset into a numpy array.

        Args:
            X: The Dataset object to be transformed.

        Returns:
            numpy.ndarray: Either a 2D (flattened) or 3D array.
        """
        if self.flatten:
            # Reshape to a 2D array by merging window and channel dimensions
            return X.flatten().reshape(-1, 1)
        else:
            return X.to_numpy()

class ArrayToDataset(BaseEstimator, TransformerMixin):
    def __init__(self, shape, data_class_fun, dtype=None, axis_names=None):
        """Initializes the ArrayToDataset transformer.
        """
        self.shape = shape
        self.dtype = dtype
        self.data_class_fun = data_class_fun
        self.axis_names = axis_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transforms the Dataset into a numpy array."""
        tmp = X.reshape(-1, self.shape[1])
        _X = []
        for i in range(0, len(tmp), self.shape[0]):
            _X.append(tmp[i:i + self.shape[0], :])

        return self.data_class_fun(_X, axis_names=self.axis_names)
