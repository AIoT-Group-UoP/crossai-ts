from typing import Dict, Callable, Any, Union, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame

from caits.dataset._dataset3 import Dataset3

T = TypeVar("T", bound="Dataset3")


class FunctionTransformerSpectrum(BaseEstimator, TransformerMixin):
    def __init__(self, func: Callable, **kwargs: Dict[str, Any]):
        """Initializes the Transformer class.

        Args:
            func: A function that will be applied to each DataFrame.
            **kw_args: Keyword arguments to be passed to the function.
        """
        self.func = func
        self.kw_args = kwargs

    def fit(self, X, y=None):
        """Fits the transformer

        Args:
            X: The input data (ignored).
            y: The target values (ignored).

        Returns:
            self: Returns the instance itself.
        """
        self.fitted_ = True
        return self

    def transform(self, data: T) -> T:
        """Applies the transformation function to each DataFrame.

        Each DataFrame is treated as a 2D matrix, and the transformation is
        applied to it as a single entity. The transformation result is then
        converted back to a DataFrame.

        Args:
            X: The Dataset object containing the data to be transformed.

        Returns:
            Dataset: A new Dataset object with the transformed data.
        """
        res = data.apply(self.func, **self.kw_args)
        # axis_names = data.get_axis_names_X()["axis_0"]
        # dfX = data.numpy_to_dataset(res, axis_names={"axis_1": axis_names})
        dfX = data.numpy_to_dataset(res, axis_names={"axis_1": "axis_0"})
        return dfX

    def get_params(self, deep=True):
        """Overrides get_params to include func_kwargs.

        Args:
            deep (bool): If True, will return the parameters for this
                         estimator and contained subobjects that are
                         estimators.

        Returns:
            dict: Parameters of the estimator.
        """
        params = super().get_params(deep=deep)
        params.update(self.kw_args)
        return params

    def set_params(self, **params):
        """Overrides set_params to correctly handle func_kwargs.

        Args:
            **params: Parameter names mapped to their values.

        Returns:
            self: The instance with updated parameters.
        """
        if "func" in params:
            self.func = params.pop("func")
        self.kw_args = params
        return self
