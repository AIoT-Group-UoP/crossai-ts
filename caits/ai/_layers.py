from typing import Callable, List, Optional, Union, Literal, Iterable

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer, SpatialDropout1D, SpatialDropout2D
from tensorflow.python.keras.utils.conv_utils import normalize_tuple


class AdaptiveAveragePooling1D(tf.keras.layers.Layer):
    def __init__(self, output_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.data_format = data_format

    def call(self, inputs):
        return _adaptive_pooling_1d(inputs, tf.reduce_mean, self.output_size, self.data_format)

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}
    

class AdaptiveMaxPooling1D(tf.keras.layers.Layer):
    def __init__(self, output_size, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.data_format = data_format

    def call(self, inputs):
        return _adaptive_pooling_1d(inputs, tf.reduce_max, self.output_size, self.data_format)

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}
    

def _adaptive_pooling_1d(
    inputs: tf.Tensor,
    reduce_function: Callable,
    output_size: Union[int, Iterable[int]],
    data_format: Literal["channels_last", "channels_first"] = "channels_last",
) -> tf.Tensor:
    """Internal function for adaptive pooling in 1D.

    Args:
        inputs: A 3D input tensor with shape (batch_size, steps, channels) or
            (batch_size, channels, steps) depending on `data_format`.
        reduce_function: The reduction function to apply to each bin.
        output_size: The desired number of output features.
        data_format: "channels_last" or "channels_first".

    Returns:
        A 3D tensor with shape (batch_size, output_size, channels) or
        (batch_size, channels, output_size).
    """

    output_size = normalize_tuple(output_size, 1, "output_size")
    bins = output_size[0]

    if data_format == "channels_last":
        split_axis, reduce_axis = 1, 2
    else:  # "channels_first"
        split_axis, reduce_axis = 2, 3

    splits = tf.split(inputs, bins, axis=split_axis)
    pooled_splits = tf.stack(splits, axis=split_axis)
    out_vect = reduce_function(pooled_splits, axis=reduce_axis)
        
    return out_vect


class MCDropout(Dropout):
    """Monte Carlo Dropout layer.

    MC dropout can boost the performance of any trained dropout model without
    having to retrain it or even modify it at all, by incorporating uncertainty
    estimates through dropout during both training and inference. Also,
    since it is just regular dropout during training, it also acts like a
    regularizer.

    Attributes:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        **kwargs: Additional keyword arguments passed to the Dropout layer.

    Returns:
        A Dropout layer with the specified rate and additional arguments.
    """

    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

    def call(self, inputs, training):
        return super().call(inputs, training=training)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "rate": self.rate}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MCSpatialDropout1D(SpatialDropout1D):
    """Monte Carlo Spatial Dropout 1D layer.

    Monte Carlo Spatial Dropout 1D improves the performance of a pre-trained
    dropout model without requiring retraining or modifications. It
    incorporates uncertainty estimates through dropout during both training and
    inference, serving as an effective regularizer.

    Attributes:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        **kwargs: Additional keyword arguments passed to the SpatialDropout1D
            layer.

    Returns:
        A SpatialDropout1D layer with the specified rate and additional
            arguments.
    """

    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

    def call(self, inputs, training):
        return super().call(inputs, training=training)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "rate": self.rate}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MCSpatialDropout2D(SpatialDropout2D):
    """Monte Carlo Spatial Dropout 2D layer.

    Monte Carlo Spatial Dropout 2D improves the performance of a pre-trained
    dropout model without requiring retraining or modifications. It
    incorporates uncertainty estimates through dropout during both training and
    inference, serving as an effective regularizer.

    Attributes:
        rate: Float between 0 and 1. Fraction of the input units to drop.
        **kwargs: Additional keyword arguments passed to the SpatialDropout2D
            layer.

    Returns:
        A SpatialDropout2D layer with the specified rate and additional
            arguments.
    """

    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

    def call(self, inputs, training):
        return super().call(inputs, training=training)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "rate": self.rate}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def dropout_layer_1d(
    inputs: tf.Tensor,
    drp_rate: float = 0.1,
    spatial: bool = False,
    mc_inference: Optional[bool] = None,
) -> Layer:
    """Creates a Dropout layer suitable for a 1D model, with options for
    standard or spatial dropout.

    This layer can be configured to use either the standard dropout approach,
    where individual elements are dropped, or spatial dropout, where entire 1D
    feature maps are dropped. It also supports Monte Carlo Dropout mode.

    Args:
        inputs (Tensor): The input tensor to which dropout will be applied.
        drp_rate (float, optional): Float between 0 and 1, representing the
            fraction of the input units to drop.
        spatial (bool, optional): If True, applies Spatial 1D Dropout, dropping
            entire 1D feature maps.
        mc_inference (bool, optional):
            - If True, enables Dropout during inference as well as training.
            - If False, disables Dropout during training and inference.
            - If None, enables Dropout during training but not during
                inference.

    Returns:
        Layer: The output layer after applying the dropout layer.
    """
    if spatial:
        drp = MCSpatialDropout1D(drp_rate)(inputs, training=mc_inference)
    else:
        drp = MCDropout(drp_rate)(inputs, training=mc_inference)
    return drp


def dropout_layer_2d(
    inputs: tf.Tensor,
    drp_rate: float = 0.1,
    spatial: bool = False,
    mc_inference: Optional[bool] = None,
) -> Layer:
    """Creates a Dropout layer suitable for a 2D model, with options for
    standard or spatial dropout.

    This layer can be configured to use either the standard dropout approach,
    where individual elements are dropped, or spatial dropout, where entire 2D
    feature maps are dropped. It also supports Monte Carlo Dropout mode for
    inference.

    Args:
        inputs: The input tensor to which dropout will be applied.
        drp_rate: Float between 0 and 1, representing the  fraction of the
        input units to drop.
        spatial: If True, applies Spatial 2D Dropout, dropping entire 2D
            feature maps.
        mc_inference:
            - If True, enables Dropout during inference as well as training.
            - If False, disables Dropout during training and inference.
            - If None, enables Dropout during training but not during
                inference.

    Returns:
        Layer: The output layer after applying the dropout layer.
    """
    if spatial:
        drp = MCSpatialDropout2D(drp_rate)(inputs, mc_inference)
    else:
        drp = MCDropout(drp_rate)(inputs, mc_inference)
    return drp


def dense_drop_block(
    inputs: tf.Tensor,
    n_layers: int,
    dense_units: List[int],
    dropout: bool,
    drop_rate: List[float],
    drop_first: bool = False,
    activation_dense: Union[str, Callable] = "relu",
    kernel_initialize: Optional[Union[Callable, str]] = None,
    kernel_regularize: Optional[Union[str, float]] = None,
    kernel_constraint: Optional[int] = None,
    spatial: bool = False,
    mc_inference: Optional[bool] = None,
) -> Layer:
    """Creates a block of dense and dropout layers for a neural network.

    This block can be configured to include a series of dense layers with
    optional dropout layers either before or after each dense layer. It allows
    customization of the number of layers, units per layer, dropout rates, and
    other hyperparameters.

    Args:
        inputs: Input tensor to which the dense and dropout layers will be
            applied.
        n_layers: Number of dense-dropout pairs or dense layers.
        dense_units: Number of units in each dense layer.
        dropout: Whether to use dropout layers.
        drop_rate: Dropout rate for each dropout layer.
        drop_first: If True, adds dropout before the dense layer. If False,
            adds after. Defaults to False.
        activation_dense: Activation function for the dense layers.
        kernel_initialize: Kernel initializer for the dense layers.
        kernel_regularize: Kernel regularizer for the dense layers.
        kernel_constraint: Kernel constraint for the dense layers.
        spatial: If True, applies Spatial Dropout; otherwise, standard Dropout.
        mc_inference:
            - If True, enables Dropout during inference as well as training.
            - If False, disables Dropout during training and inference.
            - If None, enables Dropout during training but not during
                inference.

    Returns:
        Layer: The output layer after applying the dense and dropout layers.
    """
    x = inputs
    for d in range(0, n_layers):
        if dropout and drop_first:
            x = dropout_layer_1d(inputs=x, drp_rate=drop_rate[d], spatial=spatial, mc_inference=mc_inference)

        x = Dense(
            units=dense_units[d],
            kernel_initializer=kernel_initialize,
            kernel_regularizer=kernel_regularize,
            kernel_constraint=kernel_constraint,
            activation=activation_dense,
        )(x)

        if dropout and not drop_first:
            x = dropout_layer_1d(x, drp_rate=drop_rate[d], spatial=spatial, mc_inference=mc_inference)

    return x
