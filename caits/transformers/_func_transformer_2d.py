from typing import Dict, Callable, Any
from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame

from ..dataset import Dataset


class FunctionTransformer2D(BaseEstimator, TransformerMixin):
    def __init__(self, func: Callable, **kw_args: Dict[str, Any]):
        """Initializes the Transformer class.

        Args:
            func: A function that will be applied to each DataFrame.
            **kw_args: Keyword arguments to be passed to the function.
        """
        self.func = func
        self.kw_args = kw_args

    def fit(self, X, y=None):
        """Fits the transformer

        Args:
            X: The input data (ignored).
            y: The target values (ignored).

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X: Dataset) -> Dataset:
        """Applies the transformation function to each DataFrame.

        Each DataFrame is treated as a 2D matrix, and the transformation is
        applied to it as a single entity. The transformation result is then
        converted back to a DataFrame.

        Args:
            X: The Dataset object containing the data to be transformed.

        Returns:
            Dataset: A new Dataset object with the transformed data.
        """
        transformed_X = []
        for df in X.X:
            # Apply the function directly to the entire 2D matrix
            transformed_array = self.func(df.values, **self.kw_args)
            # Convert the transformed array back to a DataFrame
            transformed_df = DataFrame(transformed_array, index=df.index, columns=df.columns)
            transformed_X.append(transformed_df)

        # Return a new Dataset object with the transformed data
        return Dataset(transformed_X, X.y, X._id)

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