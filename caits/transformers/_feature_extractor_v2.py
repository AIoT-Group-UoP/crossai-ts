from typing import Dict, List, Union, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin

T = TypeVar('T', bound="Dataset")


class FeatureExtractorSignal(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            feature_extractors: List[Dict],
            to_X=True,
            to_y=False,
            axis: int=0,
            to_dataset: bool = True
    ):
        self.feature_extractors = feature_extractors
        self.to_X = to_X
        self.to_y = to_y
        self.axis = axis
        self.to_dataset = to_dataset

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, data: T) -> Union[T, Dict]:
        features = {
            "X": {},
            "y": {}
        }

        for extractor in self.feature_extractors:
            func = extractor["func"]
            params = extractor.get("params", {})
            params["axis"] = self.axis
            feature = data.apply(
                func=func,
                to_X=self.to_X,
                to_y=self.to_y,
                export_to="dict",
                **params
            )

            features["X"][f"{func.__name__}"] = feature["X"]
            features["y"][f"{func.__name__}"] = feature["y"]

        if self.to_dataset:
            axis_names_X = {
                f"axis_{self.axis}": {name: i for i, name in enumerate(features["X"].keys())}
            }
            axis_names_y = {
                f"axis_{self.axis}": {name: i for i, name in enumerate(features["y"].keys())}
            }
            return data.features_dict_to_dataset(
                features,
                axis_names_X,
                axis_names_y,
                self.axis
            )

        else:
            return features
