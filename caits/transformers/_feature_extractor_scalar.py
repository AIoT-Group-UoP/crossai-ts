from typing import Dict, List, Union, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin

T = TypeVar('T', bound="Dataset")


class FeatureExtractorScalar(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            feature_extractors: List[Dict],
            to_X=True,
            to_y=False,
            to_dataset: bool = True
    ):
        self.feature_extractors = feature_extractors
        self.to_X = to_X
        self.to_y = to_y
        self.to_dataset = to_dataset

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, data: T) -> Union[T, Dict]:
        features = {}
        if self.to_X:
            features["X"] = {}
        if self.to_y:
            features["y"] = {}

        for extractor in self.feature_extractors:
            func = extractor["func"]
            params = extractor.get("params", {})
            feature = data.apply(
                func=func,
                to_X=self.to_X,
                to_y=self.to_y,
                export_to="dict",
                **params
            )

            if self.to_X:
                features["X"][f"{func.__name__}"] = feature["X"]

            if self.to_y:
                features["y"][f"{func.__name__}"] = feature["y"]

        if self.to_dataset:
            axis_names = {}

            if self.to_X:
                axis_names["X"] = data.X.keys()
                del axis_names["X"][f"axis_0"]
            else:
                axis_names["X"] = {}

            if self.to_y:
                axis_names["y"] = data.y.keys()
                del axis_names["y"][f"axis_0"]
            else:
                axis_names["y"] = {}

            return data.__class__.features_dict_to_dataset(
                features=features,
                axis_names=axis_names,
                axis=0
            )

        else:
            return features
