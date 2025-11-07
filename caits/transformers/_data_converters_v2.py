from sklearn.base import BaseEstimator, TransformerMixin

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
        self.fitted_ = True
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
            # return X.flatten().reshape(-1, 1)
            return X.flatten()
        else:
            return X

    def get_params(self, deep=True):
        """Returns the parameters of the transformer."""
        params = super().get_params(deep=deep)
        params.update(
            {
                "flatten": self.flatten,
                "dtype": self.dtype
            }
        )
        return params

    def set_params(self, **params):
        self.flatten = params.get("flatten", False)
        self.dtype = params.get("dtype", None)
        return self


class ArrayToDataset(BaseEstimator, TransformerMixin):
    def __init__(self, shape, dtype=None, axis_names=None, flattened=True):
        """Initializes the ArrayToDataset transformer.
        """
        self.shape = shape
        self.dtype = dtype
        self.axis_names = axis_names
        self.flattened = flattened

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        """Transforms the Dataset into a numpy array."""
        if self.flattened:
            _X = [X.X.values[i, :].reshape(self.shape) for i in range(X.X.shape[0])]
        else:
            _X = X.X.values

        return X.numpy_to_dataset(_X, axis_names=self.axis_names, split=self.flattened)

    def get_params(self, deep=True):
        """Returns the parameters of the transformer."""
        params = super().get_params(deep=deep)
        params.update(
            {
                "shape": self.shape,
                "dtype": self.dtype,
                "axis_names": self.axis_names,
            }
        )
        return params

    def set_params(self, **params):
        self.shape = params.get("shape", None)
        self.dtype = params.get("dtype", None)
        self.axis_names = params.get("axis_names", None)
        return self
