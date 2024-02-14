from typing import Union, Callable
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D
from tensorflow.keras.layers import GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Add, Activation, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from .._layers_dropout import dropout_layer_1d


# Implementation of InceptionTime NN model based on:
# - https://arxiv.org/pdf/1909.04939
def InceptionTime(
    input_shape: tuple,
    include_top: bool = True,
    num_classes: int = 1,
    classifier_activation: Union[str, Callable] = "softmax",
    nb_filters: int = 32,
    use_residual: bool = True,
    use_bottleneck: bool = True,
    depth: int = 6,
    kernel_size: int = 41,
    bottleneck_size: int = 32,
    drp_low: float = 0.,
    drp_high: float = 0.,
    kernel_initialize: Union[str, Callable] = "he_uniform",
    kernel_regularize: Union[str, float] = 4e-5,
    kernel_constraint: int = 3,
    spatial: bool = False,
    mc_inference: Union[bool, None] = None,
) -> tf.keras.Model:
    """Constructs a deep neural network based on Inception (AlexNet)
    Architecture meant for Time-Series tasks.

    An ensemble of deep Convolutional Neural Network (CNN) models, inspired
    by the Inception-v4 architecture, transformed mainly for Time Series
    Classification (TSC) tasks.

    Args:
        input_shape: Shape of the input data, excluding the batch size.
        include_top: If true, includes a fully-connected layer at the top of
            the model. Set to False for a custom layer.
        num_classes: Number of classes to predict.
        classifier_activation: Activation function for the classification task.
            Can be a string identifier or a callable function from
                tf.keras.activations.
        nb_filters: The number of filters in each convolutional layer.
        use_residual: Whether to use a residual connection or not.
        use_bottleneck: Whether to use a bottleneck layer or not.
        depth: The depth of the network.
        kernel_size: The kernel size for convolutional layers.
        bottleneck_size: The number of output filters in the bottleneck layer.
        drp_low: Dropout rate after the input layer of the model.
        drp_high: Dropout rate before the top layer of the model.
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
        A Keras model instance.

    References:
        https://arxiv.org/pdf/1909.04939
    """

    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    kernel_size = kernel_size - 1

    # -- Initiating Model Topology --
    # define the input of the network
    input_layer = Input(shape=input_shape, name="input_layer")

    x = dropout_layer_1d(inputs=input_layer, drp_rate=drp_low,
                         spatial=spatial, mc_inference=mc_inference)

    x_incept = inception_block(inputs=x,
                               use_bottleneck=use_bottleneck,
                               bottleneck_size=bottleneck_size,
                               use_residual=use_residual,
                               activation=classifier_activation,
                               depth=depth,
                               nb_filters=nb_filters,
                               kernel_size=kernel_size,
                               kernel_initialize=kernel_initialize,
                               kernel_regularize=kernel_regularize,
                               kernel_constraint=kernel_constraint
                               )

    x = GlobalAveragePooling1D()(x_incept)

    # Dropout
    x = dropout_layer_1d(inputs=x, drp_rate=drp_high, spatial=spatial,
                         mc_inference=mc_inference)

    if include_top is True:
        # flatten
        x = Flatten()(x)
        # output
        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x
    # model
    model = Model(inputs=input_layer, outputs=outputs, name="Inception_Time")

    return model


def inception_block(
    inputs: tf.Tensor,
    depth: int = 6,
    use_bottleneck: bool = True,
    bottleneck_size: int = 1,
    use_residual: bool = True,
    activation: str = "softmax",
    nb_filters: int = 32,
    kernel_size: int = 41,
    kernel_initialize: str = "he_uniform",
    kernel_regularize: float = None,
    kernel_constraint: int = None
) -> tf.Tensor:
    """Creates an Inception Block.

    Args:
        inputs: Input tensor to the block
        depth: The depth of the block.
        use_bottleneck: Whether to use a bottleneck layer or not.
        bottleneck_size: he number of output filters in the convolution.
        use_residual: The number of output filters in the convolution.
        activation: Type of activation function.
        nb_filters: The number of nb filters.
        kernel_size: The kernel size of the convolution.
        kernel_initialize: The variance scaling initializer.
        kernel_regularize: Penalty to apply on the layer's kernel.
        kernel_constraint: The constraint of the value of the incoming weights.

    Returns:
        x: Input tensor passed thourgh the inception block.
    """

    x = inputs
    inputs_res = inputs

    for d in range(depth):

        x = inception_module(inputs=x, use_bottleneck=use_bottleneck,
                             bottleneck_size=bottleneck_size,
                             activation=activation, nb_filters=nb_filters,
                             kernel_size=kernel_size,
                             kernel_initialize=kernel_initialize,
                             kernel_regularize=kernel_regularize,
                             kernel_constraint=kernel_constraint)

        if use_residual and d % 3 == 2:
            residual_conv = Conv1D(filters=128,
                                   kernel_size=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_initializer=kernel_initialize,
                                   kernel_regularizer=kernel_regularize,
                                   kernel_constraint=kernel_constraint
                                   )(inputs_res)

            shortcut_y = BatchNormalization()(residual_conv)
            res_out = Add()([shortcut_y, x])
            x = Activation("relu")(res_out)
            inputs_res = x

    return x


def inception_module(
    inputs: tf.Tensor,
    use_bottleneck: bool = True,
    bottleneck_size: int = 32,
    activation: str = "softmax",
    nb_filters: int = 64,
    kernel_size: int = 41,
    kernel_initialize: str = "he_uniform",
    kernel_regularize: float = None,
    kernel_constraint: int = None,
    stride: int = 1
) -> tf.Tensor:
    """Creates an Inception Module.

    Args:
        inputs: Input tensor to the module.
        use_bottleneck: Whether to use a bottleneck layer or not.
        bottleneck_size: he number of output filters in the convolution.
        activation: Type of activation function.
        nb_filters: The number of nb filters.
        kernel_size: The kernel size of the convolution.
        kernel_initialize: The variance scaling initializer.
        kernel_regularize: Penalty to apply on the layer's kernel.
        kernel_constraint: The constraint of the value of the incoming weights.
        stride: Stride of the convolution.

    Returns:
        x_post: Input tensor passed through the inception module.
    """
    if use_bottleneck and nb_filters > 1:
        x = Conv1D(filters=bottleneck_size, kernel_size=1,
                   padding="same", activation="linear", use_bias=False,
                   kernel_initializer=kernel_initialize,
                   kernel_regularizer=kernel_regularize,
                   kernel_constraint=kernel_constraint
                   )(inputs)
    else:
        x = inputs

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
    print("kernel size list: ", kernel_size_s)

    conv_list = []
    for i in range(len(kernel_size_s)):
        print(f"Inception filters: {nb_filters} - kernel: {kernel_size_s[i]}")
        conv = Conv1D(filters=nb_filters,
                      kernel_size=kernel_size_s[i],
                      strides=stride,
                      padding="same",
                      activation=activation,
                      use_bias=False,
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint
                      )(x)

        conv_list.append(conv)

    x2 = MaxPooling1D(pool_size=3, strides=stride, padding="same")(inputs)

    # pass via a Conv1D to match the shapes
    last_conv = Conv1D(filters=nb_filters, kernel_size=1,
                       padding="same", activation=activation, use_bias=False,
                       kernel_initializer=kernel_initialize,
                       kernel_regularizer=kernel_regularize,
                       kernel_constraint=kernel_constraint
                       )(x2)

    conv_list.append(last_conv)

    x_concat = Concatenate(axis=2)(conv_list)
    x_bn = BatchNormalization()(x_concat)
    x_post = Activation("relu")(x_bn)

    return x_post
