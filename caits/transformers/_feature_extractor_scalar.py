from collections import defaultdict
from typing import Dict, List, Union, TypeVar

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from caits.dataset._datasetList import Dataset, CoreArray

T = TypeVar('T', bound="Dataset3")


class FeatureExtractorScalar(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractors: List[Dict], axis: int=0, to_dataset: bool = True):
        self.feature_extractors = feature_extractors
        self.axis = axis
        self.to_dataset = to_dataset

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, data: T) -> Union[T, Dict]:
        features = {}

        for extractor in self.feature_extractors:
            func = extractor["func"]
            params = extractor.get("params", {})
            feature = data.apply(func, **params)

            features[f"{func.__name__}"] = feature

        if self.to_dataset:
            axis_names = data.get_axis_names_X()
            del axis_names[f"axis_{self.axis}"]
            return data.features_dict_to_dataset(features, axis_names, self.axis)

        else:
            return features
