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
        self.shapes = []
        self.datasetList = None

    def fit(self, X, y=None):
        """Fit method (no-op since nothing is learned)."""
        self.shapes = [x.shape for x in X.X]
        self.datasetList = X
        return self

    def transform(self, X):
        """Transforms the Dataset into a numpy array.

        Args:
            X: The Dataset object to be transformed.

        Returns:
            numpy.ndarray: Either a 2D (flattened) or 3D array.
        """
        tmp = X.to_numpy()

        if self.flatten:
            # Reshape to a 2D array by merging window and channel dimensions
            return X.flatten()
        else:
            return tmp[0]  # Keep the 3D shape

    # TODO: Maybe axis names should be handled internally
    def inverse_transform(self, X):
        """Transforms the numpy array into a Dataset."""
        if self.flatten:
            dim1 = sum([s[0] for s in self.shapes])
            dim2 = self.shapes[0][1]

            tmp = X.reshape((dim1, dim2))
            _X = []
            i = 0
            for s in self.shapes:
                _X.append(tmp[i:i + s[0], :])
                i += s[0]
        else:
            _X = X

        return self.datasetList.numpy_to_dataset(_X, axis_names={"axis_1": self.datasetList.X[0].axis_names["axis_1"]})




