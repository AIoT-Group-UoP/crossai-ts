# The functionality in this implementation are basically derived from
# librosa v0.10.1:
# https://github.com/librosa/librosa/blob/main/librosa/_typing.py
from typing import Any, Callable, Sequence, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Literal

_Real = Union[float, "np.integer[Any]", "np.floating[Any]"]
_Number = Union[complex, "np.number[Any]"]
_BoolLike_co = Union[bool, np.bool_]
_UIntLike_co = Union[_BoolLike_co, "np.unsignedinteger[Any]"]
_IntLike_co = Union[_BoolLike_co, int, "np.integer[Any]"]
_FloatLike_co = Union[_IntLike_co, float, "np.floating[Any]"]
_WindowSpec = Union[str, Tuple[Any, ...], float, Callable[[int], np.ndarray], ArrayLike]
_ComplexLike_co = Union[_FloatLike_co, complex, "np.complexfloating[Any, Any]"]

_T = TypeVar("_T")

_STFTPad = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "reflect",
    "symmetric",
    "empty",
]
_PadModeSTFT = Union[_STFTPad, Callable[..., Any]]
_SequenceLike = Union[Sequence[_T], np.ndarray]
_ScalarOrSequence = Union[_T, _SequenceLike[_T]]
