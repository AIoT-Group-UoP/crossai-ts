from typing import Dict, Callable, Any, Union, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin

T = TypeVar('T', bound="Dataset")


class FeatureExtractorSpectrum(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            func: Callable,
            to_X=True,
            to_y=False,
            **kw_args: Dict[str, Any]
    ):
        self.func = func
        self.kw_args = kw_args
        self.to_X = to_X
        self.to_y = to_y

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, data: T) -> T:
        features = data.apply(
            func=self.func,
            to_X=self.to_X,
            to_y=self.to_y,
            **self.kw_args
        )

        axis_names_X = data.get_axis_names_X()["axis_1"]
        axis_names_y = data.get_axis_names_y()["axis_1"]

        res = data.__class__.numpy_to_dataset(
            *features,
            axis_names_X={
                ("axis_0" if self.to_X else "axis_1"): axis_names_X
            },
            axis_names_y={
                ("axis_0" if self.to_y else "axis_1"): axis_names_y
            }
        )
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
