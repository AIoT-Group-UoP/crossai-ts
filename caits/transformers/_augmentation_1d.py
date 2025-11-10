from typing import Callable, Dict, List

from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import TypedDict

from ..dataset import DatasetBase

class Augmentation(TypedDict):
    func: Callable
    params: Dict


class Augmenter1D(BaseEstimator, TransformerMixin):
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

    def transform(self, X: DatasetBase) -> DatasetBase:
        transformed_X = []
        transformed_y = []
        transformed_id = []

        for df, label, id_ in zip(X.X, X.y, X._id):
            # Keep original instance
            transformed_X.append(df)
            transformed_y.append(label)
            transformed_id.append(id_)

            for _ in range(self.repeats):  # Repeat augmentation process
                # Create a copy of the original DataFrame for each repeat
                augmented_df = df.copy()
                # Apply each augmentation and append augmented instances
                for augmentation in self.augmentations:
                    _callable = augmentation["func"]
                    _params = augmentation["params"]
                    augmented_df = augmented_df.apply(lambda col: _callable(col.values, **_params))
                transformed_X.append(augmented_df)
                transformed_y.append(label)  # Duplicate label
                transformed_id.append(id_)  # Duplicate ID

        return DatasetBase(transformed_X, transformed_y, transformed_id)
