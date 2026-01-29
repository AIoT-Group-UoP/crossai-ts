import copy
from typing import Dict, Union, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin

T = TypeVar('T', bound="Dataset")

class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformations):
        self.transformations = transformations


    def fit(self, X, y=None):
        self.transformations_ = []

        for transformation in self.transformations:
            name, transformer, columns_set = transformation

            columns_X = columns_set["X"][0] if "X" in columns_set else []
            columns_y = columns_set["y"][0] if "y" in columns_set else []

            _data = X[:, columns_X + columns_y]

            self.transformations_.append(
                (name, transformer.fit(_data), columns_set)
            )

        return self

    # TODO: Replaces to the original data we do not want that
    def transform(self, data: T) -> Union[T, Dict]:

        tr_data = copy.deepcopy(data)

        for transformation in self.transformations_:
            name, transformer, columns_set = transformation

            if "X" in columns_set:
                columns_X = columns_set["X"][0]
                unify_X = columns_set["X"][1] is not None
                new_data = transformer.transform(tr_data[:, columns_X])

                if unify_X:
                    old_keys = new_data.X.keys()["axis_1"]
                    new_keys = columns_set["X"][1]

                    renamings = {
                        "X": {
                            "axis_1": {
                                old_key: new_key
                                for old_key, new_key in zip(old_keys, new_keys)
                            }
                        }
                    }

                    new_data = new_data.rename(renamings)
                    tr_data = tr_data.unify([new_data], axis=1, to_X=True, to_y=False)
                else:
                    tr_data = tr_data.replace(new_data)

            if "y" in columns_set:
                columns_y = columns_set["y"][0]
                unify_y = columns_set["y"][1] is not None
                new_data = transformer.transform(tr_data[:, columns_y])

                if unify_y:
                    old_keys = new_data.y.keys()["axis_1"]
                    new_keys = columns_set["y"][1]

                    renamings = {
                        "y": {
                            "axis_1": {
                                old_key: new_key
                                for old_key, new_key in zip(old_keys, new_keys)
                            }
                        }
                    }

                    new_data = new_data.rename(renamings)
                    tr_data = tr_data.unify([new_data], axis=1, to_X=False, to_y=True)
                else:
                    tr_data = tr_data.replace(new_data)

        return tr_data

