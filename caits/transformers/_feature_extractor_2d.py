from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame
from typing import Dict
from caits.dataset import Dataset


class FeatureExtractor2D(BaseEstimator, TransformerMixin):
    def __init__(self, feature_extractor: Dict):
        self.feature_extractor = feature_extractor

    def fit(self, X, y=None):
        return self

    def transform(self, X: Dataset) -> Dataset:
        transformed_X = []

        for df in X.X:
            # Ensure the DataFrame has only one column
            if df.shape[1] != 1:
                raise ValueError(f"2D Extractor does not support multi-channel signals yet. \
                                 Expected single-channel signals.")
            
            func = self.feature_extractor["func"]
            params = self.feature_extractor.get("params", {})

            # Extract information
            feature = func(df.values.flatten(), **params)

            # Ensure feature is a 2D array
            if feature.ndim != 2:
                raise ValueError("Expected a 2D array.")
            
            # Store the 2D numpy array as a DataFrame
            transformed_X.append(DataFrame(feature))

        return Dataset(transformed_X, X.y, X._id)
