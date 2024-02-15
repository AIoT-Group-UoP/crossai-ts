from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict
from pandas import DataFrame, Series
import numpy as np
from ._data_object import CAI


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractors: List[Dict]):
        self.feature_extractors = feature_extractors

    def fit(self, X, y=None):
        return self

    def transform(self, X: CAI) -> CAI:
        transformed_X = []
        transformed_y = X.y
        transformed_id = X._id

        for df in X.X:
            features_dict = {}

            for col_name, col_data in df.items():
                col_features = {}
                for extractor in self.feature_extractors:
                    func = extractor["func"]
                    params = extractor.get("params", {})
                    feature = func(col_data.values, **params)

                    # Determine how to handle the extracted
                    # feature based on its type
                    if np.isscalar(feature) or feature.ndim == 0:
                        col_features[func.__name__] = feature
                    elif feature.ndim == 1:
                        for i, val in enumerate(feature):
                            col_features[f"{func.__name__}_{i}"] = val
                    elif feature.ndim == 2 and feature.shape[1] == 1:
                        feature = feature.ravel()  # Flatten (n, 1) arrays
                        for i, val in enumerate(feature):
                            col_features[f"{func.__name__}_{i}"] = val
                    else:
                        raise ValueError("Unexpected feature shape.")

                features_dict[col_name] = Series(col_features)

            # Convert dictionary of Series to DataFrame
            # Transpose to get correct layout
            features_df = DataFrame(features_dict).T
            transformed_X.append(features_df)

        return CAI(transformed_X, transformed_y, transformed_id)
