from collections import defaultdict
from typing import Dict, List, Union, TypeVar

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from caits.dataset._dataset3 import Dataset3

T = TypeVar('T', bound="Dataset3")


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractors: List[Dict]):
        self.feature_extractors = feature_extractors

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, data: T) -> T:
        features = {}

        for extractor in self.feature_extractors:
            func = extractor["func"]
            params = extractor.get("params", {})
            feature = data.apply(func, **params)
            features[f"{func.__name__}"] = feature

        return features
