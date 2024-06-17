from typing import Dict, Callable, Any
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ..dataset import Dataset


class FeatureExtractor2D(BaseEstimator, TransformerMixin):
    def __init__(self, func: Callable, **kw_args: Dict[str, Any]):
        self.func = func
        self.kw_args = kw_args

    def fit(self, X, y=None):
        return self

    def transform(self, X: Dataset) -> Dataset:
        transformed_X = []

        for df in X.X:
            # Ensure the DataFrame has only one column
            if df.shape[1] != 1:
                raise ValueError(
                    "2D Extractor does not support multi-channel signals yet. \
                    Expected single-channel signals."
                )

            # Extract information
            feature = self.func(df.values.flatten(), **self.kw_args)

            # Ensure feature is a 2D array
            if feature.ndim != 2:
                raise ValueError("Expected a 2D array after the extraction.")

            # Store the 2D numpy array as a DataFrame
            transformed_X.append(DataFrame(feature))

        return Dataset(transformed_X, X.y, X._id)

    def get_params(self, deep=True):
        """Overrides get_params to include func and kw_args."""
        params = super().get_params(deep=deep)
        params["func"] = self.func
        params.update(self.kw_args)
        return params

    def set_params(self, **params):
        """Overrides set_params to correctly handle func and kw_args."""
        if "func" in params:
            self.func = params.pop("func")
        self.kw_args = params
        return self