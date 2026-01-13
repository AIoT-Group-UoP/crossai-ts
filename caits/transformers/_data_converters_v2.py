from sklearn.base import BaseEstimator, TransformerMixin

class DatasetToArray(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            flatten=False,
            to_X=True,
            to_y=False,
            dtype=None
    ):
        """Initializes the DatasetToArray transformer.

        Args:
            flatten (bool): If True, the output array will be flattened
                            (window_size * channels). Otherwise (default),
                            it will be a 3D array.
            dtype (str): The data type of the output array. If None, the
                         default NumPy data type will be used.
        """
        self.flatten = flatten
        self.to_X = to_X
        self.to_y = to_y
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
            return X.flatten(self.to_X, self.to_y)
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
    def __init__(
            self,
            shape_X,
            shape_y,
            to_X=True,
            to_y=False,
            dtype=None,
            axis_names=None,
            flattened=True
    ):
        """Initializes the ArrayToDataset transformer.
        """
        self.shape_X = shape_X
        self.shape_y = shape_y
        self.to_X = to_X
        self.to_y = to_y
        self.dtype = dtype
        self.axis_names = axis_names
        self.flattened = flattened

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        """Transforms the Dataset into a numpy array."""
        if self.flattened:
            if self.to_X:
                _X = X.X.values.reshape(self.shape_X)
            else:
                _X = X.X.values

            if self.to_y:
                _y = X.y.values.reshape(self.shape_y)
            else:
                _y = X.y.values

            return X.__class__.numpy_to_dataset(
                _X,
                _y,
                axis_names_X=self.axis_names["X"],
                axis_names_y=self.axis_names["y"],
            )

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
