from typing import Callable, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.constraints import Constraint, MaxNorm
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer, l2

from caits.ai import dense_drop_block


# This architecture is based on ResNet 34 (2015)
# Paper: https://arxiv.org/pdf/1512.03385.pdf
def ResNet34(
    input_shape: Tuple[int, ...],
    include_top: bool = True,
    num_classes: int = 1,
    classifier_activation: Union[str, Callable] = "softmax",
    kernel_initialize: str = "he_normal",
    kernel_regularize: Union[float, str] = 1e-3,
    kernel_constraint: int = 3,
    dense_layers: int = 0,
    dense_units: List[int] = [128, 128],
    dropout: bool = False,
    dropout_first: bool = False,
    dropout_rate: List[float] = [0.5, 0.5],
    spatial: bool = False,
    mc_inference: Optional[bool] = None,
) -> tf.keras.Model:
    """Constructs a ResNet34 model suitable for image classification tasks.

    This model includes the standard ResNet34 architecture with an optional
    fully-connected layer at the top. It can be customized with dropout, dense
    layers, and other hyperparameters.

    Args:
        input_shape: The shape of a single instance of the dataset.
        include_top: Whether to include a fully-connected layer at the top of
            the network.
        num_classes: Number of classes to predict.
        classifier_activation: Activation function for the classification task.
        kernel_initialize: The variance scaling initializer.
        kernel_regularize: Regularizer to apply a penalty on the layer's
            kernel. Can be float or str in '1e-5' format.
        kernel_constraint: The constraint of the value of the incoming weights.
        dense_layers: Number of dense layers.
        dense_units: Number of units per dense layer.
        dropout: Whether to use dropout or not.
        dropout_first: Add dropout before or after the dense layer.
        dropout_rate: Dropout rate for each dropout layer.
        spatial: Determines the type of Dropout. If True, applies
            SpatialDropout2D, else Monte Carlo Dropout.
        mc_inference: Dropout behavior during inference. If true, enabled
            during inference; if false, disabled during training and inference;
            if None, enabled during training but not during inference.

    Returns:
        A Keras Model instance representing the ResNet34 architecture.

    References:
        https://arxiv.org/pdf/1512.03385.pdf
    """

    # regularizer settings
    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    # -- Initiating Model Topology --
    # input layer
    input_layer = Input(shape=input_shape, name="input_layer")

    # The Stem Convolution Group
    x = stem(
        inputs=input_layer,
        kernel_initialize=kernel_initialize,
        kernel_regularize=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )

    # The learner
    x = learner(
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

    model = Model(input_layer, outputs, name="Resnet_34")
    return model


def stem(
    inputs: tf.Tensor,
    kernel_initialize: Union[Initializer, str] = "he_normal",
    kernel_regularize: Union[Regularizer, float, None] = 1e-3,
    kernel_constraint: Union[Constraint, int, None] = 3,
) -> tf.Tensor:
    """Constructs the stem group for a convolutional neural network.

    This stem group consists of a large kernel-sized convolutional layer
    followed by batch normalization and ReLU activation, and finally a max
    pooling layer. It serves to rapidly reduce the spatial dimensions of the
    input and to increase the depth of feature maps, preparing the input for
    subsequent residual blocks.

    Args:
        inputs: Input tensor or layer to the stem group.
        kernel_initialize: Initializer for the kernel weights, can be a string
            identifier or an initializer object.
        kernel_regularize: Regularizer for the kernel weights, can be None, a
            float, or a Regularizer object.
        kernel_constraint: Constraint for the kernel weights, can be None, an
            integer, or a Constraint object.

    Returns:
        Output tensor after applying the stem group operations.
    """
    # First Convolutional layer, where pooled
    # feature maps will be reduced by 75%
    x = Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initialize,
        kernel_regularizer=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    return x


def learner(
    x: tf.Tensor,
    kernel_initialize: Union[Initializer, str] = "he_normal",
    kernel_regularize: Union[Regularizer, float, None] = 1e-3,
    kernel_constraint: Union[Constraint, int, None] = 3,
) -> tf.Tensor:
    """Constructs the main learning structure (learner) of a ResNet-like
        architecture.

    This function sequentially applies several residual groups, each with an
    increasing number of filters. It progressively deepens the network and
    increases its capacity. The function is designed to form the core learning
    part of a deep convolutional neural network.

    Args:
        x: Input tensor to the learner.
        kernel_initialize: Initializer for the kernel weights, can be a string
            identifier or an initializer object.
        kernel_regularize: Regularizer for the kernel weights, can be None, a
            float, or a Regularizer object.
        kernel_constraint: Constraint for the kernel weights, can be None, an
            integer, or a Constraint object.

    Returns:
        Output tensor after applying all residual groups forming the learner.
    """
    # First Residual Block Group of 64 filters
    x = residual_group(x, 64, 3, True, kernel_initialize, kernel_regularize, kernel_constraint)

    # Second Residual Block Group of 128 filters
    x = residual_group(x, 128, 3, True, kernel_initialize, kernel_regularize, kernel_constraint)

    # Third Residual Block Group of 256 filters
    x = residual_group(x, 256, 5, True, kernel_initialize, kernel_regularize, kernel_constraint)

    # Fourth Residual Block Group of 512 filters
    x = residual_group(x, 512, 2, False, kernel_initialize, kernel_regularize, kernel_constraint)
    return x


def residual_group(
    x: tf.Tensor,
    n_filters: int,
    n_blocks: int,
    conv: bool = True,
    kernel_initialize: Union[Initializer, str] = "he_normal",
    kernel_regularize: Union[Regularizer, float, None] = 1e-3,
    kernel_constraint: Union[Constraint, int, None] = 3,
) -> tf.Tensor:
    """Constructs a group of residual blocks within a ResNet architecture.

    This function builds a series of residual blocks. Optionally, it can
    include a convolutional block at the end of the group to double the number
    of filters and reduce the spatial dimensions of the feature maps, preparing
    the tensor for the next residual group.

    Args:
        x: Input tensor to the residual group.
        n_filters: Number of filters for each residual block in the group.
        n_blocks: Number of residual blocks in the group.
        conv: If True, adds a convolutional block at the end of the group.
        kernel_initialize: Initializer for the kernel weights, can be a string
            identifier or an initializer object.
        kernel_regularize: Regularizer for the kernel weights, can be None, a
            float, or a Regularizer object.
        kernel_constraint: Constraint for the kernel weights, can be None, an
            integer, or a Constraint object.

    Returns:
        Output tensor after applying the residual blocks and optional
            convolutional block.
    """

    for _ in range(n_blocks):
        x = residual_block(x, n_filters, kernel_initialize, kernel_regularize, kernel_constraint)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2)
    # to fit the next Residual Group
    if conv:
        x = conv_block(x, n_filters * 2, kernel_initialize, kernel_regularize, kernel_constraint)
    return x


def residual_block(
    x: tf.Tensor,
    n_filters: int,
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None],
) -> tf.Tensor:
    """Constructs a residual block with two convolutional layers.

    In this block, a skip connection is added that bypasses the two convolution
    layers. The output of the second convolution layer is added to the input
    tensor (skip connection) before applying the activation function. This
    helps to mitigate the vanishing gradient problem and enables the training
    of deeper networks.

    Args:
        x: Input tensor to the residual block.
        n_filters: Number of filters for the convolution layers.
        kernel_initialize: Initializer for the kernel weights, can be a string
            identifier or an initializer object.
        kernel_regularize: Regularizer for the kernel weights, can be None, a
            float, or a Regularizer object.
        kernel_constraint: Constraint for the kernel weights, can be None, an
            integer, or a Constraint object.

    Returns:
        The output tensor after applying the residual block operations.
    """
    # skip connection
    shortcut = x

    # First Layer
    x = Conv2D(
        n_filters,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=kernel_initialize,
        kernel_regularizer=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)  # axis = -1 = 3
    x = ReLU()(x)

    # Second Layer
    x = Conv2D(
        n_filters,
        (3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=kernel_initialize,
        kernel_regularizer=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)

    # Add residue
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x


def conv_block(
    x: tf.Tensor,
    n_filters: int,
    kernel_initialize: Union[Initializer, str],
    kernel_regularize: Union[Regularizer, float, None],
    kernel_constraint: Union[Constraint, int, None],
) -> tf.Tensor:
    """Constructs a block of convolutions without pooling.

    This function creates a series of convolution layers, each followed by
    batch normalization and ReLU activation. The convolutions are applied with
    a stride of 2, reducing the spatial dimensions of the input at each step.

    Args:
        x: Input tensor to the convolution block.
        n_filters: Number of filters for the convolution layers.
        kernel_initialize: Initializer for the kernel weights, can be a string
            identifier or an initializer object.
        kernel_regularize: Regularizer for the kernel weights, can be None, a
            float, or a Regularizer object.
        kernel_constraint: Constraint for the kernel weights, can be None, an
            integer, or a Constraint object.

    Returns:
        Output tensor after applying a series of convolutions, batch
            normalization, and ReLU activation.
    """
    x = Conv2D(
        n_filters,
        (3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initialize,
        kernel_regularizer=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(
        n_filters,
        (3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initialize,
        kernel_regularizer=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x
