from typing import Callable, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Flatten,
    Input,
    MaxPooling1D,
    SeparableConv1D,
)
from tensorflow.keras.regularizers import l2

from .._layers import dropout_layer_1d, AdaptiveAveragePooling1D


# Implementation of XceptionTime NN model based on:
# - https://arxiv.org/pdf/1911.03803.pdf
# - https://ieeexplore.ieee.org/document/9881698/
def XceptionTime(
    input_shape: Tuple[int, ...],
    include_top: bool = True,
    num_classes: int = 1,
    classifier_activation: Union[str, Callable] = "softmax",
    xception_adaptive_size: int = 50,
    xception_adapt_ws_divide: int = 4,
    n_filters: int = 16,
    drp_low: float = 0.0,
    drp_mid: float = 0.0,
    drp_high: float = 0.0,
    spatial: bool = False,
    kernel_initialize: Union[str, Callable] = "he_uniform",
    kernel_regularize: Union[str, float] = 4e-5,
    kernel_constraint: int = 3,
    mc_inference: Union[bool, None] = None,
) -> tf.keras.Model:
    """Constructs a deep neural network based on Xception Architecture meant
        for Time-Series tasks.

    A novel deep learning model referred to as the XceptionTime
    architecture. The proposed innovative XceptionTime is designed by the
    integration of depthwise separable convolutions, adaptive average pooling,
    and a novel non-linear normalization technique. By utilizing the depthwise
    separable convolutions, the XceptionTime network has far fewer parameters
    resulting in a less complex network. The updated architecture in this
    CrossAI topology is extended in such a way as to achieve higher confidence
    in the model’s predictions, it can be adapted to any window size, and its
    upgraded functionalities can avoid overfitting and achieve better model
    generalization.

    Args:
        input_shape: Shape of the input data, excluding the batch size.
        include_top: If true, includes a fully-connected layer at the top of
            the model. Set to False for a custom layer.
        num_classes: Number of classes to predict.
        classifier_activation: Activation function for the classification task.
            Can be a string identifier or a callable function from
                tf.keras.activations.
        xception_adaptive_size: The adaptive size.
        xception_adapt_ws_divide: The number that will divide the adaptive
            size.
        n_filters: An integer that represents the number of filters in the
            convolutional neural network layer.
        drp_low: Fraction of the input units to drop in the input dropout
            layer.
        drp_mid: Fraction of the input units to drop in the mid-dropout layer.
        drp_high: Fraction of the input units to drop in the last dropout
            layer.
        kernel_initialize: The variance scaling initializer. Can be a string
            identifieror a callable function from tf.keras.initializers.
        kernel_regularize: L2 regularization factor; can be a string or float.
            If a string is provided, it's converted to a float.
        kernel_constraint: Max norm constraint for the kernel weights.
        spatial: If true, applies Spatial Dropout, else applies standard
            Dropout.
        mc_inference: If True, enables Monte Carlo dropout during inference.
            Defaults to None.

    Returns:
        model: A Keras Model instance.

    References:
        https://arxiv.org/pdf/1911.03803.pdf
        https://ieeexplore.ieee.org/document/9881698/
    """

    # check and adjust adaptive size based on input shape
    if input_shape[0] % xception_adapt_ws_divide == 0:
        xception_adaptive_size = int(input_shape[0] / xception_adapt_ws_divide)
    else:
        xception_adaptive_size = xception_adaptive_size
        print("Provide a dividable number for the window size.")
        raise Exception("Provide a dividable number for the window size.")
    print(
        f"Input size W of window transformed into a fixed length of \
        {xception_adaptive_size} sample "
        "for AAP mid layer."
    )

    # regularizer settings
    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    # -- Initiating Model Topology --
    # input layer of the network
    input_layer = Input(shape=input_shape, name="input_layer")

    # Dropout
    x = dropout_layer_1d(inputs=input_layer, drp_rate=drp_low, spatial=spatial, mc_inference=mc_inference)

    # COMPONENT 1 - Xception Block
    x = xception_block(
        inputs=x,
        n_filters=n_filters,
        kernel_initialize=kernel_initialize,
        kernel_regularize=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )

    # COMPONENT 2
    # Head of the sequential component
    head_nf = n_filters * 32

    # transform the input with window size W to a fixed length
    # of adaptive size
    x = AdaptiveAveragePooling1D(output_size=xception_adaptive_size)(x)

    # Dropout
    x = dropout_layer_1d(inputs=x, drp_rate=drp_mid, spatial=spatial, mc_inference=mc_inference)

    # stack 3 Conv1x1 Convolutions to reduce the time-series
    # to the number of the classes
    x = conv1d_block(
        x,
        nf=head_nf // 2,
        drp_on=False,
        drp_rate=0.5,
        spatial=True,
        kernel_initialize=kernel_initialize,
        kernel_regularize=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )

    x = conv1d_block(
        x,
        nf=head_nf // 4,
        drp_on=False,
        drp_rate=0.5,
        spatial=True,
        kernel_initialize=kernel_initialize,
        kernel_regularize=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )

    x = conv1d_block(
        x,
        nf=num_classes,
        drp_on=False,
        drp_rate=0.5,
        spatial=True,
        kernel_initialize=kernel_initialize,
        kernel_regularize=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )

    # convert the length of the input signal to 1 with the
    x = AdaptiveAveragePooling1D(1)(x)

    # # Dropout
    x = dropout_layer_1d(inputs=x, drp_rate=drp_high, spatial=spatial, mc_inference=mc_inference)

    if include_top is True:
        # flatten
        x = Flatten()(x)

        # output
        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    # model
    model = Model(inputs=input_layer, outputs=outputs, name="Xception_Time")

    return model


def conv1d_block(
    inputs: tf.Tensor,
    nf: int,
    ks: int = 1,
    strd: int = 1,
    pad: str = "same",
    bias: bool = False,
    bn: bool = True,
    act: bool = True,
    act_func: str = "relu",
    drp_on: bool = False,
    drp_rate: float = 0.5,
    spatial: bool = True,
    mc_inference: Optional[bool] = None,
    kernel_initialize: Optional[Union[str, Callable]] = None,
    kernel_regularize: Optional[Union[str, float]] = None,
    kernel_constraint: Optional[int] = None,
) -> tf.Tensor:
    """Creates a block of layers consisting of Conv1D, BatchNormalization,
        Activation and Dropout.

    Args:
        inputs: Input tensor to the block.
        nf: Number of filters for the Conv1D layer.
        ks: Kernel size for the Conv1D layer.
        strd: Strides for the Conv1D layer.
        pad: Padding for the Conv1D layer.
        bias: Whether to use bias in Conv1D.
        bn: Whether to use BatchNormalization.
        act: Whether to use an Activation layer.
        act_func: Activation function to use.
        drp_on: Whether to include a Dropout layer.
        drp_rate: Dropout rate.
        spatial: If true, applies Spatial Dropout, else applies standard
            Dropout.
        mc_inference: If True, enables Monte Carlo dropout
            during inference.
        kernel_initialize: The variance scaling initializer.
        kernel_regularize: Penalty to apply on the layer's kernel.
        kernel_constraint: The constraint of the value of the incoming weights.

    Returns:
        tensor: Output tensor after applying the block of layers.
    """

    x = Conv1D(
        filters=nf,
        kernel_size=ks,
        strides=strd,
        padding=pad,
        use_bias=bias,
        kernel_initializer=kernel_initialize,
        kernel_regularizer=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )(inputs)

    if bn:
        x = BatchNormalization()(x)

    if act:
        x = Activation(act_func)(x)

    if drp_on:
        x = dropout_layer_1d(x, drp_rate, spatial, mc_inference)

    return x


def xception_block(
    inputs: tf.Tensor,
    n_filters: int,
    depth: int = 4,
    use_residual: bool = True,
    kernel_initialize: Optional[Union[str, Callable]] = None,
    kernel_regularize: Optional[Union[str, float]] = None,
    kernel_constraint: Optional[int] = None,
) -> tf.Tensor:
    """Applies a series of Xception modules, potentially with residual
        connections.

    Args:
        inputs: Input tensor to the block.
        n_filters: Number of filters for the convolutional layers.
        depth: Depth of the Xception block.
        use_residual: Whether to use residual connections.
        kernel_initialize: The variance scaling initializer.
        kernel_regularize: Penalty to apply on the layer's kernel.
        kernel_constraint: The constraint of the value of the incoming weights.

    Returns:
        x: Output tensor after applying the Xception block.
    """

    x = inputs
    input_res = inputs

    for d in range(depth):
        xception_filters = n_filters * 2**d
        x = xception_module(
            x,
            xception_filters,
            kernel_initialize=kernel_initialize,
            kernel_regularize=kernel_regularize,
            kernel_constraint=kernel_constraint,
        )

        if use_residual and d % 2 == 1:
            residual_conv_filters = n_filters * 4 * (2**d)
            res_out = Conv1D(
                filters=residual_conv_filters,
                kernel_size=1,
                padding="same",
                use_bias=False,
                kernel_initializer=kernel_initialize,
                kernel_regularizer=kernel_regularize,
                kernel_constraint=kernel_constraint,
            )(input_res)

            shortcut_y = BatchNormalization()(res_out)
            res_out = Add()([shortcut_y, x])

            x = Activation("relu")(res_out)
            input_res = x

    return x


def xception_module(
    inputs: tf.Tensor,
    n_filters: int,
    use_bottleneck: bool = True,
    kernel_size: int = 41,
    stride: int = 1,
    kernel_initialize: Optional[Union[str, Callable]] = None,
    kernel_regularize: Optional[Union[str, float]] = None,
    kernel_constraint: Optional[int] = None,
) -> tf.Tensor:
    """Applies the Xception module which is a series of SeparableConv1D layers
        and a MaxPooling1D layer, followed by concatenation.

    Args:
        inputs: A Keras tensor serving as the starting point for this module.
        n_filters: Number of filters for the convolutional layers.
        use_bottleneck: Whether to use a bottleneck Conv1D layer at the
            beginning.
        kernel_size: Kernel size for the SeparableConv1D layers.
        stride: Stride for the SeparableConv1D and MaxPooling1D layers.
        kernel_initialize: Initializer for the kernel weights.
        kernel_regularize: Regularizer for the kernel weights.
        kernel_constraint: Constraint for the kernel weights.

    Returns:
        x_post: Output tensor after applying the Xception module.
    """

    if use_bottleneck and n_filters > 1:
        x = Conv1D(
            filters=n_filters,
            kernel_size=1,
            padding="valid",
            use_bias=False,
            kernel_initializer=kernel_initialize,
            kernel_regularizer=kernel_regularize,
            kernel_constraint=kernel_constraint,
        )(inputs)
    else:
        x = inputs

    # Assuming kernel_padding_size_lists function is defined
    kernel_sizes, _ = kernel_padding_size_lists(kernel_size)

    separable_conv_list = []
    for kernel in kernel_sizes:
        separable_conv = SeparableConv1D(
            filters=n_filters,
            kernel_size=kernel,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initialize,
            kernel_regularizer=kernel_regularize,
            kernel_constraint=kernel_constraint,
        )(x)
        separable_conv_list.append(separable_conv)

    x2 = MaxPooling1D(pool_size=3, strides=stride, padding="same")(inputs)
    x2 = Conv1D(
        filters=n_filters,
        kernel_size=1,
        padding="valid",
        use_bias=False,
        kernel_initializer=kernel_initialize,
        kernel_regularizer=kernel_regularize,
        kernel_constraint=kernel_constraint,
    )(x2)

    separable_conv_list.append(x2)

    x_post = Concatenate(axis=2)(separable_conv_list)

    return x_post


def kernel_padding_size_lists(max_kernel_size: int) -> Tuple[List[int], List[int]]:
    """Generates lists of kernel sizes and corresponding paddings based on a
        given max kernel size.

    This function is designed to create lists of kernel sizes and their
    corresponding paddings suitable for constructing convolutional neural
    networks. The kernel sizes are determined by dividing the `max_kernel_size`
    by powers of 2, and the paddings are calculated accordingly.

    Args:
        max_kernel_size: The maximum kernel size used as the reference for
        generating smaller kernel sizes and corresponding paddings.

    Returns:
        A tuple containing two lists - the first list contains generated kernel
        sizes, and the second list contains corresponding paddings.
    """
    i = 0
    kernel_size_list = []
    padding_list = []
    while i < 3:
        size = max_kernel_size // (2**i)
        if size == max_kernel_size:
            kernel_size_list.append(int(size))
            padding_list.append(int((size - 1) / 2))
        else:
            kernel_size_list.append(int(size + 1))
            padding_list.append(int(size / 2))
        i += 1

    return kernel_size_list, padding_list
