import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Add, ReLU, Dense, Flatten
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.regularizers import Regularizer, l2
from tensorflow.keras.constraints import Constraint, MaxNorm
from tensorflow.keras import Model
from .._layers_dropout import dense_drop_block
from typing import Union, Callable, List, Tuple


# This architecture is based on ResNet 50 (2015)
# Paper: https://arxiv.org/pdf/1512.03385.pdf
def ResNet50(
    input_shape: tuple,
    include_top: bool = True,
    num_classes: int = 1,
    classifier_activation: Union[str, Callable] = "softmax",
    kernel_initialize: Union[str, Initializer] = "he_normal",
    kernel_regularize: Union[float, None] = 1e-3,
    kernel_constraint: Union[int, None] = 3,
    dense_layers: int = 0,
    dense_units: List[int] = [128, 128],
    dropout: bool = False,
    dropout_first: bool = False,
    dropout_rate: List[float] = [0.5, 0.5],
    spatial: bool = False,
    mc_inference: Union[bool, None] = None
) -> tf.keras.Model:
    """ResNet34 Model

    Args:
        input_shape: The shape of a single instance of the dataset.
        include_top: Whether to include a fully-connected layer at the top of
            the network.
        num_classes: Number of classes to predict.
        classifier_activation: Activation function for the classification task.
        kernel_initialize: Initializer for the kernel weights, can be a string
            identifier or an tf.keras.Initializer object.
        kernel_regularize: Regularizer for the kernel weights, typically a
            float indicating the L2 regularization factor.
        kernel_constraint: Constraint for the kernel weights, typically an
            integer for the max norm constraint.
        dense_layers: Number of dense layers to include in the model.
        dense_units: List of units for each dense layer.
        dropout: Whether to use dropout.
        dropout_first: If True, adds dropout before the dense layer(s).
        dropout_rate: List specifying the dropout rate for each layer.
        spatial: If True, applies Spatial Dropout, else applies standard
            Dropout.
        mc_inference: If True, enables Monte Carlo dropout during inference;
                      If False, disables Dropout in training and inference;
                      If None, enables Dropout during training but not during
                        inference.

    Returns:
        A Keras Model instance representing the ResNet34 architecture.

    References:
        https://arxiv.org/pdf/1512.03385.pdf
    """

    # Regularizer settings
    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    # Begin model topology
    input_layer = Input(shape=input_shape, name="input_layer")

    # Initial convolution block
    x = conv_bn_relu(inputs=input_layer, n_filters=64, kernel_size=7,
                     strides=2, kernel_initialize=kernel_initialize,
                     kernel_regularize=kernel_regularize,
                     kernel_constraint=kernel_constraint)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # Add ResNet blocks
    for filters, reps, s in zip([64, 128, 256, 512],
                                [3, 4, 6, 3],
                                [1, 2, 2, 2]):
        x = resnet_block(x,
                         filters,
                         reps,
                         s,
                         kernel_initialize,
                         kernel_regularize,
                         kernel_constraint)

    # Add top layer if specified
    if include_top:
        x = GlobalAvgPool2D()(x)
        x = Flatten()(x)
        x = dense_drop_block(inputs=x, n_layers=dense_layers,
                             dense_units=dense_units,
                             dropout=dropout, drop_first=dropout_first,
                             drop_rate=dropout_rate,
                             activation_dense="relu",
                             kernel_initialize=kernel_initialize,
                             kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint,
                             spatial=spatial,
                             mc_inference=mc_inference)
        outputs = Dense(units=num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    # Construct and return the model
    model = Model(input_layer, outputs, name="Resnet_50")
    return model


def resnet_block(
    x: tf.Tensor,
    n_filters: int,
    reps: int,
    strides: Union[int, tuple],
    kernel_initialize: Union[Initializer, str, None],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None]
) -> tf.Tensor:
    """Constructs a ResNet block with a specified number of repetitions.

    The block comprises a projection block followed by multiple identity
    blocks. The projection block aligns the dimensions of input and output,
    while identity blocks follow the standard building patterns of ResNet.

    Args:
        x: Input tensor to the ResNet block, represented as a tf.Tensor.
        n_filters: Number of filters for the convolution layers.
        reps: Number of repetitions of the identity blocks.
        strides: Strides for the convolution layer in the projection block.
        kernel_initialize: Initializer for the kernel weights, either a string
            or a tf.keras.initializers.Initializer.
        kernel_regularize: Regularizer for the kernel weights, either a float
            or a tf.keras.regularizers.Regularizer.
        kernel_constraint: Constraint for the kernel weights, either an integer
            or a tf.keras.constraints.Constraint.

    Returns:
        The output tensor after processing through the ResNet block.
    """

    x = projection_block(x, n_filters, strides, kernel_initialize,
                         kernel_regularize, kernel_constraint)
    for _ in range(reps-1):
        x = identity_block(x, n_filters, kernel_initialize, kernel_regularize,
                           kernel_constraint)
    return x


def projection_block(
    x: tf.Tensor,
    n_filters: int,
    strides: Union[int, tuple],
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None]
) -> tf.Tensor:
    """Constructs a projection block for a ResNet architecture.

    This block is used in ResNet when the dimensions of the input tensor and
    the output tensor of the residual layers do not match. It applies a series
    of convolutions and adds a shortcut connection that projects the input
    tensor to align its dimensions with the output.

    Args:
        x: Input tensor to the projection block.
        n_filters: Number of filters for the convolution layers within the
            block.
        strides: Strides for the first convolution layer in the block.
            Can be an integer or a tuple.
        kernel_initialize: Initializer for the kernel weights, can be a string
            identifier or an initializer object.
        kernel_regularize: Regularizer for the kernel weights, typically a
            float indicating the L2 regularization factor.
        kernel_constraint: Constraint for the kernel weights, typically an
            integer for the max norm constraint.

    Returns:
        Output tensor after applying the projection block operations.
    """
    shortcut = _conv_bn(x, 4*n_filters, 1, strides, kernel_initialize,
                        kernel_regularize, kernel_constraint)
    x = conv_bn_relu(x, n_filters, 1, strides, kernel_initialize,
                     kernel_regularize, kernel_constraint)
    x = conv_bn_relu(x, n_filters, 3, 1, kernel_initialize, kernel_regularize,
                     kernel_constraint)
    x = _conv_bn(x, 4*n_filters, 1, 1, kernel_initialize, kernel_regularize,
                 kernel_constraint)
    x = Add()([shortcut, x])
    return ReLU()(x)


def identity_block(
        x: tf.Tensor,
        n_filters: int,
        kernel_initialize: Union[Initializer, str],
        kernel_regularize: Union[Regularizer, float, None],
        kernel_constraint: Union[Constraint, int, None]
) -> tf.Tensor:
    """Constructs an identity block for a ResNet architecture.

    This block applies several convolutional layers with batch normalization
    and ReLU activation, followed by a skip connection that adds the input
    tensor to the output of the convolutional layers.

    Args:
        x: Input tensor to the identity block.
        n_filters: Number of filters for the convolution layers within
            the block.
        kernel_initialize: Initializer for the kernel weights, can be a string
            identifier or an initializer object.
        kernel_regularize: Regularizer for the kernel weights, typically a
            float indicating the L2 regularization factor.
        kernel_constraint: Constraint for the kernel weights, typically an
            integer for the max norm constraint.

    Returns:
        Output tensor after applying the identity block operations.
    """
    shortcut = x
    x = conv_bn_relu(x, n_filters, 1, 1, kernel_initialize, kernel_regularize,
                     kernel_constraint)
    x = conv_bn_relu(x, n_filters, 3, 1, kernel_initialize, kernel_regularize,
                     kernel_constraint)
    x = _conv_bn(x, 4*n_filters, 1, 1, kernel_initialize, kernel_regularize,
                 kernel_constraint)
    x = Add()([shortcut, x])
    return ReLU()(x)


def conv_bn_relu(
    inputs: tf.Tensor,
    n_filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]],
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None]
) -> tf.Tensor:
    """Applies a convolution followed by batch normalization and a ReLU
        activation to the input tensor.

    This function creates a 2D convolution layer using specified filters,
        kernel size, and strides, applies batch normalization to its output,
         and then applies a ReLU activation function.

    Args:
        inputs: The input tensor.
        n_filters: Number of filters in the convolution layer.
        kernel_size: Size of the convolution kernel. Can be an integer or a
            tuple of 2 integers.
        strides: Strides of the convolution. Can be an integer or a tuple
            of 2 integers.
        kernel_initialize: Initializer for the kernel weights. Can be a string
            identifier or a tf.keras.initializers.Initializer.
        kernel_regularize: Regularizer function for the kernel weights. Can be
            None, a float, or a tf.keras.regularizers.Regularizer.
        kernel_constraint: Constraint for the kernel weights. Can be None, an
            integer, or a tf.keras.constraints.Constraint.

    Returns:
        The output tensor after applying convolution, batch normalization,
            and ReLU activation, represented as a tf.Tensor.
    """
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides,
               padding="same",
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(inputs)
    x = BatchNormalization()(x)
    return ReLU()(x)


def _conv_bn(
    inputs: tf.Tensor,
    n_filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    strides: Union[int, Tuple[int, int]],
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None]
) -> tf.Tensor:
    """Applies a convolution followed by batch normalization to the input
        tensor.

    This function creates a 2D convolution layer using specified filters,
    kernel size, and strides, and then applies batch normalization to
    its output.

    Args:
        inputs: The input tensor.
        n_filters: Number of filters in the convolution layer.
        kernel_size: Size of the convolution kernel. Can be an integer or a
            tuple of 2 integers.
        strides: Strides of the convolution. Can be an integer or a tuple
            of 2 integers.
        kernel_initialize: Initializer for the kernel weights. Can be a string
            identifier or a tf.keras.initializers.Initializer.
        kernel_regularize: Regularizer function for the kernel weights. Can be
            None, a float, or a tf.keras.regularizers.Regularizer.
        kernel_constraint: Constraint for the kernel weights. Can be None, an
            integer, or a tf.keras.constraints.Constraint.

    Returns:
        The output tensor after applying the convolution and batch
            normalization, represented as a tf.Tensor.
    """
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides,
               kernel_initializer=kernel_initialize,
               kernel_regularizer=kernel_regularize,
               kernel_constraint=kernel_constraint)(inputs)
    x = BatchNormalization()(x)

    return x
