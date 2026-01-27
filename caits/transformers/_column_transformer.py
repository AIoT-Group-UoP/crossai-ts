from typing import Dict, Union, TypeVar
from sklearn.base import BaseEstimator, TransformerMixin

T = TypeVar('T', bound="Dataset")

class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformations, unify: bool = False):
        self.transformations = transformations
        self.unify = unify

    def fit(self, X, y=None):
        self.transformations_ = []

        for transformation in self.transformations:
            name, transformer, columns_set = transformation
            if "X" in columns_set:
                columns_X = columns_set["X"][0]
                self.to_X_ = True
            else:
                columns_X = []
                self.to_X_ = False

            if "y" in columns_set:
                columns_y = columns_set["y"][0]
                self.to_y_ = True
            else:
                columns_y = []
                self.to_y_ = False

            _data = X[:, columns_X + columns_y]

            self.transformations_.append(
                (name, transformer.fit(_data), columns_set)
            )

        return self

    # TODO: Replaces to the original data we do not want that
    def transform(self, data: T) -> Union[T, Dict]:

        tr_data = []
        if self.unify:
            column_names_arr_X = sum([t[-1]["X"][1] if "X" in t[-1] else [] for t in self.transformations_], [])
            column_names_arr_y = sum([t[-1]["y"][1] if "y" in t[-1] else [] for t in self.transformations_], [])

            tr_data.append(data)
            column_names_arr_X = list(data.get_axis_names_X()["axis_1"].keys()) + column_names_arr_X
            column_names_arr_y = list(data.y.keys()["axis_1"]) + column_names_arr_y
        else:
            column_names_arr_X = sum([t[-1]["X"][0] if "X" in t[-1] else [] for t in self.transformations_], [])
            column_names_arr_y = sum([t[-1]["y"][0] if "y" in t[-1] else [] for t in self.transformations_], [])

        column_names_X = {col: i for i, col in enumerate(column_names_arr_X)}
        column_names_y = {col: i for i, col in enumerate(column_names_arr_y)}

        for transformation in self.transformations_:
            name, transformer, columns_set = transformation

            columns_X = columns_set["X"][0] if "X" in columns_set else []
            columns_y = columns_set["y"][0] if "y" in columns_set else []

            _data = data[:, columns_X + columns_y]
            tr_data.append(transformer.transform(_data))

        final_tr_data = tr_data[0].unify(
            tr_data[1:],
            axis_names={
                "X": {"axis_1": column_names_X},
                "y": {"axis_1": column_names_y}
            },
            axis=1,
            to_X=self.to_X_,
            to_y=self.to_y_,
        )

        if self.unify:
            return final_tr_data
        else:
            ret_data = data.replace(final_tr_data)
            return ret_data
