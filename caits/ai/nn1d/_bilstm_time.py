from typing import Callable, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input
from tensorflow.keras.models import Model

from .._layers_dropout import dropout_layer_1d


# Implementation of NN model for 1D based on:
# - https://ieeexplore.ieee.org/document/8488627
def BiLSTM_Time(
    input_shape: tuple,
    include_top: bool = True,
    num_classes: int = 1,
    classifier_activation: Union[str, Callable] = "softmax",
    n_layers: int = 3,
    lstm_units: List[int] = [32, 32, 32],
    dense_units: List[int] = [128],
    drp_rate: float = 0.0,
    spatial: bool = False,
    mc_inference: Union[bool, None] = None,
) -> tf.keras.Model:
    """Constructs a deep neural network using bidirectional LSTM.

    This model converts low-level time series features into high-level
        expressions.

    Args:
        input_shape: Shape of the input data, excluding batch size.
        include_top: If True, includes a fully-connected layer at the top.
            Set to False for adding a custom layer.
        num_classes: Number of prediction classes.
        classifier_activation: Activation function for the classification task.
            Can be a string identifier or a function from tf.keras.activations.
        n_layers: Number of Bidirectional LSTM layers.
        lstm_units: LSTM units for each layer.
        dense_units: Units for each dense layer.
        drp_rate: Dropout rate.
        spatial: Type of Dropout. True for SpatialDropout1D, False for regular
            Dropout.
        mc_inference: Dropout setting during inference. True enables,
            False disables, and None means dropout is applied during training
                only.

    Returns:
        A Keras model instance.

    References:
        https://ieeexplore.ieee.org/document/8488627
    """

    input_layer = Input(shape=input_shape, name="input_layer")

    x = Bidirectional(LSTM(units=lstm_units[0], activation="tanh", return_sequences=True))(input_layer)
    x = dropout_layer_1d(x, drp_rate, spatial=spatial, mc_inference=mc_inference)

    x_block = bilstm_block(x, n_layers, lstm_units, drp_rate)
    x_dense = dense_block(x_block, n_layers, dense_units)

    if include_top is True:
        outputs = Dense(num_classes, activation=classifier_activation)(x_dense)
    else:
        outputs = x_dense

    model = Model(inputs=input_layer, outputs=outputs, name="BiLSTM_Time")

    return model


def bilstm_block(
    inputs: tf.Tensor,
    n_layers: int,
    lstm_units: list,
    drp_rate: float,
    mc_inference: Optional[bool] = None,
) -> tf.Tensor:
    """Constructs a bidirectional LSTM (BiLSTM) block.

    Args:
        inputs: Input tensor for the BiLSTM block.
        n_layers: Number of LSTM layers in the block.
        lstm_units: Number of units in each LSTM layer.
        drp_rate: Dropout rate to be applied after each LSTM layer.
        mc_inference: If True, enables Monte Carlo dropout
            during inference. Defaults to None.

    Returns:
        x: The output tensor from the last layer of the BiLSTM block.
    """

    x = inputs
    for i in range(1, n_layers - 1):
        x = Bidirectional(LSTM(units=lstm_units[i], return_sequences=True, activation="tanh"))(x)
        x = dropout_layer_1d(x, drp_rate, False, mc_inference)

    x = Bidirectional(LSTM(units=lstm_units[-1], return_sequences=False, activation="tanh"))(x)
    x = dropout_layer_1d(x, drp_rate, False, mc_inference)

    return x


def dense_block(inputs: tf.Tensor, n_layers: int, dense_units: List[int]) -> tf.Tensor:
    """Builds a block of dense layers.

    Args:
        inputs: Input tensor for the dense layers.
        n_layers: Number of dense layers to be created.
        dense_units: Number of units in each dense layer.

    Returns:
        x: The output tensor from the last dense layer.
    """

    x = inputs
    for d in range(0, min(n_layers, len(dense_units))):
        x = Dense(units=dense_units[d], activation="relu")(x)

    return x
