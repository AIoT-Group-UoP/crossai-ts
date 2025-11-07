from datetime import datetime
from typing import Any, List, Optional
from numpy import ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from ..dataset import Dataset


class ArrayToDataset(BaseEstimator, TransformerMixin):
    """A transformer that converts numpy arrays (X, y) into a custom Dataset
    format. Optionally uses timestamps as IDs if none are provided.
    """

    def __init__(self, ids: Optional[List[Any]] = None):
        """Initialize the transformer.

        Args:
            ids: An optional list of identifiers corresponding to each sample.
                 If None, timestamps will be generated for each sample.
        """
        self.ids = ids

    def fit(self, X, y=None):
        """This transformer does not need to fit anything, so the fit method
        just returns itself.

        Args:
            X: Feature data.
            y: Target data.
        """
        self.y = y
        return self

    def transform(self, X, y=None):
        """Transform the input numpy arrays into a Dataset object.

        Args:
            X: Feature data.
            y: Target data.

        Returns:
            Dataset Object.
        """
        # Validate X
        if not isinstance(X, (ndarray, list)):
            raise ValueError("X must be an 2D or 3D numpy array or list.")

        # Check if `y` is provided and create a placeholder if not
        if self.y is None:
            y = [None] * len(X)
        else:
            y = self.y
        # Generate timestamps as ids if ids are not provided
        if self.ids is None:
            self.ids = [datetime.now().isoformat() for _ in range(len(X))]

        # Convert X to a list of DataFrames
        X_dfs = [DataFrame(x) for x in X]

        # Create and return the Dataset object
        return Dataset(X=X_dfs, y=list(y), id=self.ids)


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
        _X, _, _ = X.to_numpy(dtype=self.dtype)

        if self.flatten:
            # Reshape to a 2D array by merging window and channel dimensions
            return _X.reshape(_X.shape[0], -1)
        else:
            return _X  # Keep the 3D shape
