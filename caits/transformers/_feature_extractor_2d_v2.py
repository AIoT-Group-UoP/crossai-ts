from typing import Dict, Callable, Any, Union, TypeVar
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ..dataset import Dataset2

T = TypeVar('T', bound="Dataset2")


class FeatureExtractor2D(BaseEstimator, TransformerMixin):
    def __init__(self, func: Callable, **kw_args: Dict[str, Any]):
        self.func = func
        self.kw_args = kw_args

    def fit(self, X, y=None):
        return self

    def transform(self, data: T) -> T:
        transformed_X = []

        for d in data.X:
            transformed_X.append({})
            for col_name, col_values in d.items():
                # Extract information
                feature = self.func(col_values, **self.kw_args)

                # Ensure feature is a 2D array
                if feature.ndim != 2:
                    raise ValueError("Expected a 2D array after the extraction.")

                # Store the 2D numpy array as a DataFrame
                transformed_X[-1][col_name] = feature

        tmp = data.to_dict()
        tmp["X"] = transformed_X

        return data.__class__(**tmp)


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
