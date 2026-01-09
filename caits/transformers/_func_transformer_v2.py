from typing import Union, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin
from caits.dataset import DatasetBase

T = TypeVar('T', bound="DatasetBase")

class FunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            func,
            to_X=True,
            to_y=False,
            **func_kwargs
    ):
        """Initializes the Transformer class.

        Args:
            func: A function that will be applied column-wise to each
                  DataFrame.
            **func_kwargs: Keyword arguments to be passed to the function.
        """
        self.func = func
        self.func_kwargs = func_kwargs
        self.to_X = to_X
        self.to_y = to_y

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
        """Applies the transformation function column-wise to the data.

        Args:
            X: The Dataset object containing the data to be transformed.

        Returns:
            DatasetBase: A new Dataset object with the transformed data.
        """
        transformed_data = data.apply(
            func=self.func,
            to_X=self.to_X,
            to_y=self.to_y,
            **self.func_kwargs
        )
        return data.__class__.numpy_to_dataset(
            *transformed_data,
            axis_names_X={"axis_1": data.get_axis_names_X()["axis_1"]},
            axis_names_y={"axis_1": data.get_axis_names_y()["axis_1"]}
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
