from typing import Dict, Callable, Any, Union, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin

T = TypeVar('T', bound="Dataset")


class FeatureExtractorSpectrum(BaseEstimator, TransformerMixin):
    def __init__(self, func: Callable, **kw_args: Dict[str, Any]):
        self.func = func
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, data: T) -> T:
        features = data.apply(self.func, **self.kw_args)
        axis_names = data.get_axis_names_X()["axis_1"]
        res = data.numpy_to_dataset(features, axis_names={"axis_0": axis_names})
        return res


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
