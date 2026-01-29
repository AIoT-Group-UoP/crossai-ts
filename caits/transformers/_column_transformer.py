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
                    new_data = new_data.rename(columns_set["X"])
                    tr_data = tr_data.unify(new_data, axis=1)
                else:
                    tr_data = tr_data.replace(new_data)

            if "y" in columns_set:
                columns_y = columns_set["y"][0]
                unify_y = columns_set["y"][1] is not None

            _data = tr_data[:, columns_X + columns_y]
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
