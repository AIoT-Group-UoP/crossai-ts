from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ..dataset import Dataset, RegressionDataset


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractors: List[Dict]):
        self.feature_extractors = feature_extractors

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[Dataset, RegressionDataset]) -> Union[Dataset, RegressionDataset]:
        transformed_X = []

        for df in X.X:
            features_dict = defaultdict(lambda: [])

            for col_name, col_data in df.items():
                for extractor in self.feature_extractors:
                    func = extractor["func"]
                    params = extractor.get("params", {})
                    feature = func(col_data.values.flatten(), **params)

                    # Convert scalar features to np.float64
                    if np.isscalar(feature):
                        feature = np.float64(feature)

                    # Flatten 2D arrays with a single column
                    column_vector_cond = feature.ndim == 2 and feature.shape[1] == 1
                    if column_vector_cond:
                        feature = feature.ravel()

                    # Handle all features, including scalars, 1D, and 2D arrays
                    if np.isscalar(feature) or feature.ndim == 0:
                        # For scalars or 0D arrays
                        features_dict[func.__name__].append(feature)
                    elif feature.ndim == 1 or column_vector_cond:
                        # For 1D arrays or flattened 2D arrays
                        for i, val in enumerate(feature):
                            features_dict[f"{func.__name__}_{i}"].append(val)
                    else:
                        raise ValueError("Unexpected feature shape.")

            # Convert the features_dict to a DataFrame,
            # with channels as columns and features as rows
            features_df = DataFrame(features_dict, index=[col_name for col_name in df.keys()]).T
            transformed_X.append(features_df)
            print(transformed_X)

        if isinstance(X, Dataset):
            return Dataset(transformed_X, X.y, X._id)
        elif isinstance(X, RegressionDataset):
            return RegressionDataset(transformed_X, X.y)
        else:
            raise NotImplementedError("Transformer not implemented.")
