from typing import Union, Callable, List
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer, l2
from tensorflow.keras.constraints import Constraint, MaxNorm
from tensorflow.keras.initializers import Initializer
from .._layers_dropout import dense_drop_block


# Implementation of Xception NN model based on:
# - https://arxiv.org/abs/1409.1556
def VGG16(
    input_shape: tuple,
    include_top: bool = True,
    num_classes: int = 1,
    classifier_activation: Union[str, Callable] = "softmax",
    kernel_initialize: Union[str, Initializer] = "he_normal",
    kernel_regularize: Union[float, str] = 1e-5,
    kernel_constraint: int = 3,
    dense_layers: int = 0,
    dense_units: List[int] = [128, 128],
    dropout: bool = False,
    dropout_first: bool = False,
    dropout_rate: List[float] = [0.5, 0.5],
    spatial: bool = False,
    mc_inference: Union[bool, None] = None
) -> tf.keras.Model:
    """Constructs the VGG16 model, a convolutional neural network architecture
    for image classification.

    The model includes customizable top layers, dropout settings, and other
    hyperparameters. VGG16 is known for its simplicity and depth, utilizing
    repeated blocks of convolutional layers followed by max pooling.

    Args:
        input_shape: The shape of a single instance of the dataset.
        include_top: Whether to include a fully-connected layer at the top of
            the network.
        num_classes: Number of classes for the output layer.
        classifier_activation: Activation function for the classification task.
        kernel_initialize: The variance scaling initializer. Can be an str
            identifier or a tf.keras.initializer object.
        kernel_regularize: Regularizer for the kernel weights. Can be a float
            or a string in scientific notation (e.g., '1e-5').
        kernel_constraint: Constraint on the kernel weights, expressed as an
            integer.
        dense_layers: Number of dense layers to add to the top of the network.
        dense_units: Number of units per dense layer.
        dropout: Whether to include dropout layers.
        dropout_first: Whether to apply dropout before or after the dense
            layers.
        dropout_rate: Dropout rate for each dropout layer.
        spatial: If True, applies Spatial Dropout; else applies standard
            Dropout.
        mc_inference: Determines the behavior of dropout during inference.
            If True, enabled during inference;
            if False, disabled during training and inference;
            if None, enabled during training but not during inference.

    Returns:
        A tf.keras.Model instance representing the VGG16 architecture.

    References:
        https://arxiv.org/abs/1409.1556
    """

    # regularizer settings
    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    # -- Initiating Model Topology --
    # Create the input vector
    input_layer = Input(shape=input_shape, name="input_layer")

    # Blocks of VGG16
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(input_layer)
    x = ReLU()(x)

    # 1 conv layer, 64 channels
    x = vgg_block(x, 1, 64, kernel_initialize, kernel_regularize,
                  kernel_constraint)
    # 2 conv layers, 128 channels
    x = vgg_block(x, 2, 128, kernel_initialize, kernel_regularize,
                  kernel_constraint)
    # 3 conv layers, 256 channels
    x = vgg_block(x, 3, 256, kernel_initialize, kernel_regularize,
                  kernel_constraint)
    # 3 conv layers, 512 channels
    x = vgg_block(x, 3, 512, kernel_initialize, kernel_regularize,
                  kernel_constraint)
    # 3 conv layers, 512 channels
    x = vgg_block(x, 3, 512, kernel_initialize, kernel_regularize,
                  kernel_constraint)

    if include_top:
        x = Flatten()(x)

        # apply multiple sequential dense/dropout layers
        x = dense_drop_block(inputs=x, n_layers=dense_layers,
                             dense_units=dense_units,
                             dropout=dropout, drop_first=dropout_first,
                             drop_rate=dropout_rate,
                             activation_dense="relu",
                             kernel_initialize=kernel_initialize,
                             kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint,
                             spatial=spatial,
                             mc_inference=mc_inference
                             )

        outputs = Dense(units=num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    model = Model(input_layer, outputs, name="VGG_16")
    return model


def vgg_block(
    input_tensor: tf.Tensor,
    num_convs: int,
    num_channels: int,
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, str],
    kernel_constraint: Union[Constraint, int]
) -> tf.Tensor:
    """Adds a VGG block to the model.

    Each VGG block consists of a specified number of convolutional layers, each
    with ReLU activation, followed by a max pooling layer.

    Args:
        input_tensor: Input tensor for the block.
        num_convs: Number of convolutional layers in the block.
        num_channels: Number of filters/channels for the convolutional layer.
        kernel_initialize: The variance scaling initializer, can be a string
            identifier or an initializer object.
        kernel_regularize: Regularizer for the kernel weights, can be None, a
            float, or a Regularizer object.
            If a string is provided in scientific notation (e.g., '1e-5'), it
                will be converted to float.
        kernel_constraint: Constraint for the kernel weights, can be None, an
            integer, or a Constraint object.

    Returns:
        Output tensor after passing through the VGG block.
    """

    x = input_tensor
    for _ in range(num_convs):
        x = Conv2D(filters=num_channels, kernel_size=(3, 3), padding="same",
                   kernel_initializer=kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint)(x)
        x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x
