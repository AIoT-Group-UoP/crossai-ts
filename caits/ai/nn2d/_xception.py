from typing import Callable, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.constraints import Constraint, MaxNorm
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
    ReLU,
    SeparableConv2D,
)
from tensorflow.keras.regularizers import Regularizer, l2

from caits.ai import dense_drop_block


# Implementation of Xception NN model based on:
# - https://arxiv.org/abs/1610.02357
def Xception(
    input_shape: Tuple[int, ...],
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
    mc_inference: Optional[bool] = None,
) -> tf.keras.Model:
    """Constructs the Xception model, a deep convolutional neural network known
    for its depth and efficiency.

    The model is structured into entry, middle, and exit flows, utilizing
    depth-wise separable convolutions. It allows for customization of the top
    layer, dropout, dense layers, and other hyperparameters.

    Args:
        input_shape: The shape of a single instance of the dataset.
        include_top: Whether to include a fully-connected layer at the top of
            the network.
        num_classes: Number of classes for the output layer.
        classifier_activation: Activation function for the classification task.
        kernel_initialize: The variance scaling initializer.
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

    Returns:
        A tf.keras.Model instance representing the Xception architecture.

    References:
        https://arxiv.org/abs/1610.02357
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

    # Create entry section
    x = entry_flow(
        inputs=input_layer,
        kernel_initialize=kernel_initialize,
        kernel_regularize=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )

    # Create the middle section
    x = middle_flow(
        x=x,
        kernel_initialize=kernel_initialize,
        kernel_regularize=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )

    # Create the exit section for 2 classes
    x = exit_flow(
        x=x,
        kernel_initialize=kernel_initialize,
        kernel_regularize=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )

    if include_top:
        # flatten
        x = Flatten()(x)

        # apply multiple sequential dense/dropout layers
        x = dense_drop_block(
            inputs=x,
            n_layers=dense_layers,
            dense_units=dense_units,
            dropout=dropout,
            drop_first=dropout_first,
            drop_rate=dropout_rate,
            activation_dense="relu",
            kernel_initialize=kernel_initialize,
            kernel_regularize=kernel_regularize,
            kernel_constraint=kernel_constraint,
            spatial=spatial,
            mc_inference=mc_inference,
        )

        # Fully connected output layer (classification)
        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    model = Model(input_layer, outputs, name="Xception")

    return model


def entry_flow(
    inputs: tf.Tensor,
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None],
) -> tf.Tensor:
    """Creates the entry flow section of a convolutional neural network.

    The entry flow consists of an initial stem function for dimensionality
    reduction and expansion, followed by a series of projection blocks. This
    section prepares the input tensor for deeper processing in the subsequent
    sections of the network.

    Args:
        inputs: Input tensor to the entry flow section.
        kernel_initialize: Initializer for the kernel weights, can be a string
            identifier or an initializer object.
        kernel_regularize: Regularizer for the kernel weights, can be None, a
            float, or a Regularizer object.
        kernel_constraint: Constraint for the kernel weights, can be None, an
            integer, or a Constraint object.

    Returns:
        Output tensor after passing through the entry flow section.

    Nested Functions:
        stem: Creates the initial part of the entry flow, performing
            dimensionality reduction and expansion.
    """

    def stem(inputs, kernel_initialize, kernel_regularize, kernel_constraint):
        """Creates the stem entry into the neural network, performing initial
        dimensionality reduction and expansion.

        Args:
            inputs: Input tensor to the stem section.
            kernel_initialize: Initializer for the kernel weights.
            kernel_regularize: Regularizer for the kernel weights.
            kernel_constraint: Constraint for the kernel weights.

        Returns:
            Output tensor after processing through the stem section.
        """

        # Strided convolution - dimensionality reduction
        # Reduce feature maps by 75%
        x = Conv2D(
            32,
            (3, 3),
            strides=(2, 2),
            kernel_initializer=kernel_initialize,
            kernel_regularizer=kernel_regularize,
            kernel_constraint=kernel_constraint,
        )(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Convolution - dimensionality expansion
        # Double the number of filters
        x = Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            kernel_initializer=kernel_initialize,
            kernel_regularizer=kernel_regularize,
            kernel_constraint=kernel_constraint,
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    # Create the stem to the neural network
    x = stem(inputs, kernel_initialize, kernel_regularize, kernel_constraint)

    # Create three residual blocks using linear projection
    for n_filters in [128, 256, 728]:
        x = projection_block(x, n_filters, kernel_initialize, kernel_regularize, kernel_constraint)

    return x


def middle_flow(
    x: tf.Tensor,
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None],
) -> tf.Tensor:
    """Creates the middle flow section of a convolutional neural network.

    This section consists of multiple (typically 8) residual blocks with
    depth-wise separable convolutions. It processes the feature maps at a
    consistent depth, applying a series of transformations without altering
    their spatial dimensions.

    Args:
        x: Input tensor to the middle flow section.
        kernel_initialize: Initializer for the depth-wise and point-wise kernel
            weights, can be a string identifier or an initializer object.
        kernel_regularize: Regularizer for the depth-wise and point-wise kernel
            weights, can be None, a float, or a Regularizer object.
        kernel_constraint: Constraint for the depth-wise and point-wise kernel
            weights, can be None, an integer, or a Constraint object.

    Returns:
        Output tensor after passing through multiple residual blocks in the
            middle flow section.
    """
    # Create 8 residual blocks
    for _ in range(8):
        x = residual_block(x, 728, kernel_initialize, kernel_regularize, kernel_constraint)
    return x


def exit_flow(
    x: tf.Tensor,
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None],
) -> tf.Tensor:
    """Creates the exit flow section of a convolutional neural network.

    This section of the network applies depth-wise separable convolutions and
    pooling to transform the feature maps into a form suitable for the final
    classification layer. The depth of the feature maps is increased, and their
    spatial dimensions are reduced, ending with global average pooling.

    Args:
        x: Input tensor to the exit flow section.
        kernel_initialize: Initializer for the depth-wise and point-wise kernel
            weights, can be a string identifier or an initializer object.
        kernel_regularize: Regularizer for the depth-wise and point-wise kernel
            weights, can be None, a float, or a Regularizer object.
        kernel_constraint: Constraint for the depth-wise and point-wise kernel
            weights, can be None, an integer, or a Constraint object.

    Returns:
        Output tensor after applying the exit flow section, typically after
            global average pooling.
    """

    # 1x1 strided convolution to increase number and reduce size of
    # feature maps in identity link to match output of residual block for
    # the add operation (projection shortcut)
    shortcut = Conv2D(
        1024,
        (1, 1),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initialize,
        kernel_regularizer=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )(x)
    shortcut = BatchNormalization()(shortcut)

    x = ReLU()(x)
    # First Depthwise Separable Convolution
    # Dimensionality reduction - reduce number of filters
    x = SeparableConv2D(
        728,
        (3, 3),
        padding="same",
        depthwise_initializer=kernel_initialize,
        pointwise_initializer=kernel_initialize,
        depthwise_regularizer=kernel_regularize,
        pointwise_regularizer=kernel_regularize,
        depthwise_constraint=kernel_constraint,
        pointwise_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)

    x = ReLU()(x)
    # Second Depthwise Separable Convolution
    # Dimensionality restoration
    x = SeparableConv2D(
        1024,
        (3, 3),
        padding="same",
        depthwise_initializer=kernel_initialize,
        pointwise_initializer=kernel_initialize,
        depthwise_regularizer=kernel_regularize,
        pointwise_regularizer=kernel_regularize,
        depthwise_constraint=kernel_constraint,
        pointwise_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Add the projection shortcut to the output of the pooling layer
    x = Add()([x, shortcut])

    # Third Depthwise Separable Convolution
    x = SeparableConv2D(
        1556,
        (3, 3),
        padding="same",
        depthwise_initializer=kernel_initialize,
        pointwise_initializer=kernel_initialize,
        depthwise_regularizer=kernel_regularize,
        pointwise_regularizer=kernel_regularize,
        depthwise_constraint=kernel_constraint,
        pointwise_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Fourth Depthwise Separable Convolution
    x = SeparableConv2D(
        2048,
        (3, 3),
        padding="same",
        depthwise_initializer=kernel_initialize,
        pointwise_initializer=kernel_initialize,
        depthwise_regularizer=kernel_regularize,
        pointwise_regularizer=kernel_regularize,
        depthwise_constraint=kernel_constraint,
        pointwise_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Global Average Pooling will flatten the 10x10 feature maps into 1D
    # feature maps
    output = GlobalAveragePooling2D()(x)

    return output


def projection_block(
    x: tf.Tensor,
    n_filters: int,
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None],
) -> tf.Tensor:
    """Creates a residual block with depth-wise separable convolutions and a
    projection shortcut.

    The projection shortcut is used when the dimensions of the input and output
    of the block differ. It includes a strided convolutional layer to match the
    number of filters and reduce the spatial dimensions of the input tensor to
    align with the output tensor of the block. Depth-wise separable
    convolutions are then applied.

    Args:
        x: Input tensor to the projection block.
        n_filters: Number of filters for the depth-wise separable convolution
            layers.
        kernel_initialize: Initializer for the depth-wise and point-wise kernel
            weights, can be a string identifier or an initializer object.
        kernel_regularize: Regularizer for the depth-wise and point-wise kernel
            weights, can be None, a float, or a Regularizer object.
        kernel_constraint: Constraint for the depth-wise and point-wise kernel
            weights, can be None, an integer, or a Constraint object.

    Returns:
        Output tensor after applying the depth-wise separable convolutions and
        the projection shortcut.
    """
    # Remember the input
    shortcut = x

    # Strided convolution to double number of filters in identity link to
    # match output of residual block for the add operation
    # (projection shortcut)
    shortcut = Conv2D(
        n_filters,
        (1, 1),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initialize,
        kernel_regularizer=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # ReLu activation is applied before SeparableConv2D
    # only in the last two projections
    if n_filters in [256, 728]:
        x = ReLU()(x)

    # First Depthwise Separable Convolution
    x = SeparableConv2D(
        n_filters,
        (3, 3),
        padding="same",
        depthwise_initializer=kernel_initialize,
        pointwise_initializer=kernel_initialize,
        depthwise_regularizer=kernel_regularize,
        pointwise_regularizer=kernel_regularize,
        depthwise_constraint=kernel_constraint,
        pointwise_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second depthwise Separable Convolution
    x = SeparableConv2D(
        n_filters,
        (3, 3),
        padding="same",
        depthwise_initializer=kernel_initialize,
        pointwise_initializer=kernel_initialize,
        depthwise_regularizer=kernel_regularize,
        pointwise_regularizer=kernel_regularize,
        depthwise_constraint=kernel_constraint,
        pointwise_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Add the projection shortcut to the output of the block
    x = Add()([x, shortcut])

    return x


def residual_block(
    x: tf.Tensor,
    n_filters: int,
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None],
) -> tf.Tensor:
    """Creates a residual block using depth-wise separable convolutions.

    This block applies several depth-wise separable convolutions with ReLU
    activations. A residual connection (shortcut) is then added from the input
    to the output of these layers. Depth-wise separable convolutions offer a
    more efficient alternative to standard convolutions by first performing a
    spatial convolution on each channel separately before combining them using
    point-wise convolutions.

    Args:
        x: Input tensor to the residual block.
        n_filters: Number of filters for the depth-wise separable convolution
            layers.
        kernel_initialize: Initializer for the depth-wise and point-wise
            kernel weights, can be a string identifier or an Initializer
                object.
        kernel_regularize: Regularizer for the depth-wise and point-wise kernel
            weights, can be None, a float, or a Regularizer object.
        kernel_constraint: Constraint for the depth-wise and point-wise kernel
            weights, can be None, an integer, or a Constraint object.

    Returns:
        Output tensor after applying the depth-wise separable convolutions and
            the residual connection.
    """
    # Remember the input
    shortcut = x

    for _ in range(3):
        x = ReLU()(x)
        # First Depthwise Separable Convolution
        x = SeparableConv2D(
            n_filters,
            (3, 3),
            padding="same",
            depthwise_initializer=kernel_initialize,
            pointwise_initializer=kernel_initialize,
            depthwise_regularizer=kernel_regularize,
            pointwise_regularizer=kernel_regularize,
            depthwise_constraint=kernel_constraint,
            pointwise_constraint=kernel_constraint,
        )(x)
        x = BatchNormalization()(x)

    # Add the identity link to the output of the block
    x = Add()([x, shortcut])
    return x
