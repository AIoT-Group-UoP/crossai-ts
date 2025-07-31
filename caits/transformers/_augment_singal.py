from typing import Callable, Dict, List

from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import TypedDict

from caits.dataset._dataset3 import Dataset3


class Augmentation(TypedDict):
    func: Callable
    params: Dict


class AugmenterSignal(BaseEstimator, TransformerMixin):
    """Augmenter Transformer that applies a list of augmentation functions,
    each with its parameters, to each DataFrame within the Dataset.X list,
    while retaining original instances and repeating the augmentation process
    a specified number of times.

    Args:
        augmentations: A list where each element is a dictionary consisting
                       of an augmentation function under the key 'func' and
                       a dictionary of its parameters under the key 'params'.
        repeats: The number of times each augmentation should be applied.
    """

    def __init__(self, augmentations: List[Augmentation], repeats: int = 1):
        self.augmentations = augmentations
        self.repeats = repeats

    def fit(self, X, y=None):
        return self

    def transform(self, X: Dataset3) -> Dataset3:
        transformed_x = X

        for augmentation in self.augmentations:
            _callable = augmentation['func']
            _params = augmentation['params']
            _params["repeats"] = self.repeats
            transformed_x_vals = transformed_x.apply(_callable, **_params)
            transformed_x = transformed_x.numpy_to_dataset(
                transformed_x_vals,
                axis_names={"axis_1": transformed_x.X[0].axis_names["axis_1"]},
            )

        return X + transformed_x


