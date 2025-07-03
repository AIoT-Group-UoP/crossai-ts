import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..dataset import RegressionDataset


class RegressionFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func, **func_kwargs):
        """Initializes the Transformer class.

        Args:
            func: A function that will be applied column-wise to each
                  DataFrame.
            **func_kwargs: Keyword arguments to be passed to the function.
        """
        self.func = func
        self.func_kwargs = func_kwargs

    def fit(self, X, y=None):
        """Fits the transformer

        Args:
            X: The input data (ignored).
            y: The target values (ignored).

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X: RegressionDataset) -> RegressionDataset:
        """Applies the transformation function column-wise to the data.

        Args:
            X: The Dataset object containing the data to be transformed.

        Returns:
            Dataset: A new Dataset object with the transformed data.
        """
        transformed_X = self.func(X.X.values, **self.func_kwargs)

        # Return a new CAI object with the transformed data
        return RegressionDataset(pd.DataFrame(transformed_X, columns=list(X.X.columns)), X.y)

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
        params.update(self.func_kwargs)
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
        self.func_kwargs = params
        return self
