from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict
from . import CoreArray

class Dataset(ABC):
    def __init__(
            self,
            X: Union[CoreArray, List[CoreArray]],
            y: Union[CoreArray, List]
    ):
        self.X = X
        self.y = y

    @abstractmethod
    def __len__(self):
        return len(self.X)

    @abstractmethod
    def __getitem__(self, idx: Union[int, slice, tuple, List]):
        pass

    @abstractmethod
    def __iter__(self):
        self._current = 0
        return self

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def batch(self, batch_size: int):
        pass

    @abstractmethod
    def unify(self, others, axis_names: Optional = None, axis: int=0):
        pass

    @abstractmethod
    def replace(self, other):
        pass

    @abstractmethod
    def to_numpy(self, flatten=False):
        pass

    @abstractmethod
    def to_df(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def to_list(self):
        pass

    @abstractmethod
    def get_axis_names_X(self):
        pass

    @abstractmethod
    def numpy_to_dataset(
            self,
            X,
            axis_names: Optional[Dict[str, Dict[Union[str, int], int]]] = None,
            split: bool=True
    ):
        pass

    @abstractmethod
    def features_dict_to_dataset(self, features, axis_names, axis):
        pass

    @abstractmethod
    def dict_to_dataset(self, X):
        pass

    @abstractmethod
    def train_test_split(self):
        pass

    @abstractmethod
    def apply(self, func, *args, **kwargs):
        pass

    @abstractmethod
    def stack(self, data: List):
        pass

    @abstractmethod
    def flatten(self, axis_names_sep=","):
        pass

    @abstractmethod
    def shuffle(self, seed: int=42):
        pass
