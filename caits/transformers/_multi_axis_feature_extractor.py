from collections import defaultdict
from typing import Dict, List

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from ..dataset import RegressionDataset


class RegressionFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractors: List[Dict]):
        self.feature_extractors = feature_extractors

    def fit(self, X, y=None):
        return self

    def transform(self, X: RegressionDataset) -> RegressionDataset:
        features_dict = defaultdict(lambda: [])

        for extractor in self.feature_extractors:
            func = extractor["func"]
            params = extractor.get("params", {})
            feature = func(X.X.values, **params)

            # Convert scalar features to np.float64
            if np.isscalar(feature):
                feature = np.float64(feature)

            features_dict[func.__name__].append(feature)

        # Convert the features_dict to a DataFrame,
        # with channels as columns and features as rows
        X_transformed = DataFrame(features_dict, index=X.X.columns)

        return RegressionDataset(X_transformed, X.y)
