from typing import Dict, List, Union, TypeVar, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

from caits.dataset._dataset3 import Dataset3

T = TypeVar('T', bound="Dataset3")


class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformations, unify: bool = False):
        self.transformations = transformations
        self.unify = unify

    def fit(self, X, y=None):
        self.transformations_ = []

        for transformation in self.transformations:
            name, transformer, columns, new_columns = transformation
            _data = X[:, columns]
            self.transformations_.append(
                (name, transformer.fit(_data), columns, new_columns)
            )

        return self

    # TODO: Maybe not replace
    def transform(self, data: T) -> Union[T, Dict]:

        tr_data = []
        if self.unify:
            column_names_arr = sum([t[-1] for t in self.transformations_], [])
            tr_data.append(data)
            column_names_arr = list(data.get_axis_names_X()["axis_1"].keys()) + column_names_arr
        else:
            column_names_arr = sum([t[-2] for t in self.transformations_], [])

        column_names = {col: i for i, col in enumerate(column_names_arr)}

        for transformation in self.transformations_:
            name, transformer, columns, new_columns = transformation
            _data = data[:, columns]
            tr_data.append(transformer.transform(_data))

        final_tr_data = tr_data[0].unify(tr_data[1:], axis_names={"axis_1": column_names}, axis=1)

        if self.unify:
            return final_tr_data
        else:
            data.replace(final_tr_data)
            return data
