import copy
from typing import Dict, List, Union, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin
from caits.dataset import CoreDataset

T = TypeVar('T', bound="CoreDataset")


class FeatureExtractorSignal(BaseEstimator, TransformerMixin):
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
        features = data.to_dict()

        if self.to_X:
            features["X"] = {}

        if self.to_y:
            features["y"] = {}

        for extractor in self.feature_extractors:
            func = extractor["func"]
            params = extractor.get("params", {})
            params["axis"] = 1
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
            if self.to_X:
                axis_names_X = {
                    f"axis_1": {name: i for i, name in enumerate(features["X"].keys())}
                }
            else:
                axis_names_X = {}

            if self.to_y:
                axis_names_y = {
                    f"axis_1": {name: i for i, name in enumerate(features["y"].keys())}
                }
            else:
                axis_names_y = copy.deepcopy(data.y.axis_names)

            return data.__class__.features_dict_to_dataset(
                features,
                {
                    "X": axis_names_X,
                    "y": axis_names_y
                },
                axis=1,
                to_X=self.to_X,
                to_y=self.to_y
            )

        else:
            return features
