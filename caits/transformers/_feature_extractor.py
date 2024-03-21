from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from pandas import DataFrame
from typing import List, Dict
from caits.dataset import Dataset


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractors: List[Dict]):
        self.feature_extractors = feature_extractors

    def fit(self, X, y=None):
        return self

    def transform(self, X: Dataset) -> Dataset:
        transformed_X = []
        transformed_y = X.y
        transformed_id = X._id

        for df in X.X:
            features_dict = {}

            for col_name, col_data in df.items():
                for extractor in self.feature_extractors:
                    func = extractor["func"]
                    params = extractor.get("params", {})
                    feature = func(col_data.values.flatten(), **params)

                    # Populate features_dict with feature names as
                    # keys and column names with values as lists
                    if np.isscalar(feature) or feature.ndim == 0:
                        features_dict.setdefault(func.__name__, []).append(feature)
                    elif feature.ndim == 1:
                        for i, val in enumerate(feature):
                            features_dict.setdefault(f"{func.__name__}_{i}", []).append(val)
                    elif feature.ndim == 2 and feature.shape[1] == 1:
                        feature = feature.ravel()  # Flatten (n, 1) arrays
                        for i, val in enumerate(feature):
                            features_dict.setdefault(f"{func.__name__}_{i}", []).append(val)
                    else:
                        raise ValueError("Unexpected feature shape.")

            # Convert the features_dict to a DataFrame,
            # with channels as columns and features as rows
            features_df = DataFrame(
                features_dict,
                index=[col_name for col_name in df.keys()]
            ).T
            transformed_X.append(features_df)

        return Dataset(transformed_X, transformed_y, transformed_id)
