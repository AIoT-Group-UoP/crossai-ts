from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Callable, Dict
from ._data_object import CAI


class Augmenter(BaseEstimator, TransformerMixin):
    """Augmenter Transformer that applies a list of augmentation functions,
    each with its parameters, to each DataFrame within the CAI.X list,
    while retaining original instances.

    Args:
        augmentations (list): A list where each element is a tuple consisting 
                              of an augmentation function and a dictionary of
                              its parameters.
    """

    def __init__(self, augmentations: List[Tuple[Callable, Dict]]):
        self.augmentations = augmentations

    def fit(self, X, y=None):
        return self

    def transform(self, X: CAI) -> CAI:
        transformed_X = []
        transformed_y = []
        transformed_id = []

        for df, label, id_ in zip(X.X, X.y, X._id):
            # Keep original instance
            transformed_X.append(df)
            transformed_y.append(label)
            transformed_id.append(id_)

            # Apply each augmentation and append augmented instances
            for func, params in self.augmentations:
                augmented_df = df.apply(lambda col: func(col, **params))
                transformed_X.append(augmented_df)
                transformed_y.append(label)  # Duplicate label
                transformed_id.append(id_)  # Duplicate ID

        return CAI(transformed_X, transformed_y, transformed_id)
