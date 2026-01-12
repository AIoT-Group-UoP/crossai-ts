from typing import Union, TypeVar, List, Dict, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from ..dataset import DatasetBase
T = TypeVar('T', bound="DatasetBase")

class SklearnWrapper(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            transformer,
            to_X=True,
            to_y=False,
            transformer_kwargs: Optional[Dict]=None
    ):
        """Initializes the Transformer class.

        Args:
            func: A function that will be applied column-wise to each
                  DataFrame.
            **func_kwargs: Keyword arguments to be passed to the function.
        """
        self.transformer = transformer
        self.to_X = to_X
        self.to_y = to_y

        if transformer_kwargs is None:
            transformer_kwargs = {}
        self.transformer_kwargs = transformer_kwargs

    def fit(self, X, y=None):
        """Fits the transformer

        Args:
            X: The input data (ignored).
            y: The target values (ignored).

        Returns:
            self: Returns the instance itself.
        """
        self.fitted_transformer_X_ = self.transformer(**self.transformer_kwargs).fit(X.X.values)
        self.fitted_transformer_y_ = self.transformer(**self.transformer_kwargs).fit(X.y.values)
        return self

    def transform(self, data: T) -> T:
        """Applies the transformation function column-wise to the data.

        Args:
            X: The Dataset object containing the data to be transformed.

        Returns:
            DatasetBase: A new Dataset object with the transformed data.
        """
        if self.to_X:
            transformed_X = self.fitted_transformer_X_.transform(data.X.values)
        else:
            transformed_X = data.X.values

        if self.to_y:
            transformed_y = self.fitted_transformer_y_.transform(data.y.values)
        else:
            transformed_y = data.y.values

        return data.__class__.numpy_to_dataset(
            transformed_X,
            transformed_y,
            axis_names_X=data.X.keys(),
            axis_names_y=data.y.keys(),
            split=False
        )

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
        params.update(self.transformer_kwargs)
        return params

    def set_params(self, **params):
        """Overrides set_params to correctly handle func_kwargs.

        Args:
            **params: Parameter names mapped to their values.

        Returns:
            self: The instance with updated parameters.
        """
        if "transformer" in params:
            self.transformer = params.get("transformer", FunctionTransformer)
            self.transformer_kwargs = params.get("transformer_kwargs", {})
        return self
