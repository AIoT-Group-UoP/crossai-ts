# Following imports are generally deprecated
from ._augmentation_1d import Augmenter1D
# from ._data_converters import ArrayToDataset, DatasetToArray
from ._encoder import LE
# from ._feature_extractor import FeatureExtractor
# from ._feature_extractor_2d import FeatureExtractor2D
# from ._func_transformer import FunctionTransformer
# from ._sliding_window import SlidingWindow
# from ._func_transformer_2d import FunctionTransformer2D

# Following import will be used
from ._augment_signal import AugmentSignal
from ._column_transformer import ColumnTransformer
from ._data_converters_v2 import ArrayToDataset, DatasetToArray
from ._feature_extractor_scalar import FeatureExtractorScalar
from ._feature_extractor_v2 import FeatureExtractorSignal
from ._feature_extractor_2d_v2 import FeatureExtractorSpectrum
from ._func_transformer_v2 import FunctionTransformer
from ._func_transformer_2d_v2 import FunctionTransformer2D
from ._sklearn_wrapper import SklearnWrapper
from ._sliding_window_v2 import SlidingWindow


__all__ = [
    # "Augmenter1D",
    "ArrayToDataset",
    "DatasetToArray",
    # "FeatureExtractor",
    # "FeatureExtractor2D",
    "FunctionTransformer",
    "LE",
    "SlidingWindow",
    "AugmentSignal",
    "FeatureExtractorScalar",
    "FeatureExtractorSignal",
    "FeatureExtractorSpectrum",
    "ColumnTransformer",
    "FunctionTransformer",
    "FunctionTransformer2D",
    "SklearnWrapper",
]