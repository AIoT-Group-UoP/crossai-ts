from typing import Union, Callable, List, Optional
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, \
    GlobalAveragePooling2D, MaxPool2D, Flatten, concatenate, Dropout
from tensorflow.keras.initializers import Initializer, GlorotUniform, Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from crossai.ai import dense_drop_block


# Implementation of InceptionV1 NN model based on:
# - https://arxiv.org/pdf/1409.4842v1.pdf
# InceptionV2-3: https://arxiv.org/pdf/1512.00567v3.pdf
# InceptionV4-ResNet: https://arxiv.org/pdf/1602.07261.pdf
def InceptionV1(
    input_shape: tuple,
    include_top: bool = True,
    num_classes: int = 1,
    classifier_activation: Union[str, Callable] = "softmax",
    dense_layers: int = 0,
    dense_units: List[int] = [128, 128],
    dropout: bool = False,
    dropout_first: bool = False,
    dropout_rate: List[float] = [0.5, 0.5],
    spatial: bool = False,
    mc_inference: Union[bool, None] = None,
    kernel_initialize: Union[Callable, str] = GlorotUniform(),
    kernel_regularize: Union[float, str] = 1e-5,
    kernel_constraint: int = 3,
) -> tf.keras.Model:
    """Builds the InceptionV1 or GoogLeNet Model.

    This model is suitable for image classification and other computer vision
        tasks.

    Args:
        input_shape: Shape of a single instance of the dataset.
        include_top: If True, includes the fully-connected layer at the top of
            the model.
        num_classes: Number of classes for prediction.
        classifier_activation: Activation function for the classification task.
        kernel_initialize: Kernel initializer, can be a
            tf.keras.initializers.Initializer object or string identifier.
        kernel_regularize: Regularizer for the kernel, can be a float or
            string. If a string is provided, it's converted to a float.
        kernel_constraint: Constraint on the kernel values, expressed as
            an integer.
        dense_layers: Number of dense layers to include in the model.
        dense_units: List of units for each dense layer.
        dropout: If True, applies dropout to the layers.
        dropout_first: If True, adds dropout before the dense layer(s).
        dropout_rate: List specifying the dropout rate for each layer.
        spatial: If True, applies Spatial Dropout, else applies standard
            Dropout.
        mc_inference: If True, enables Monte Carlo dropout during inference.

    Returns:
        A Keras Model instance representing the InceptionV1 architecture.

        Referenes:
            - https://arxiv.org/pdf/1409.4842v1.pdf
            - https://arxiv.org/pdf/1512.00567v3.pdf
            - https://arxiv.org/pdf/1602.07261.pdf
    """

    # Initializer - regularizer settings
    if isinstance(kernel_regularize, str):
        kernel_regularize = float(kernel_regularize.replace("−", "-"))
    elif kernel_regularize is not None:
        kernel_regularize = l2(kernel_regularize)

    if kernel_constraint is not None:
        kernel_constraint = MaxNorm(max_value=kernel_constraint, axis=[0, 1])

    bias_initialize = Constant(value=0.2)

    # -- Initiating Model Topology --
    # Create the input vector
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Conv2D(64, (7, 7), padding="same", strides=(2, 2), activation="relu",
               name="conv_1_7x7/2", kernel_initializer=kernel_initialize,
               bias_initializer=bias_initialize)(input_layer)#
    x = MaxPool2D((3, 3), padding="same", strides=(2, 2),
                  name="max_pool_1_3x3/2")(x)
    x = Conv2D(64, (1, 1), padding="same", strides=(1, 1), activation="relu",
               name="conv_2a_3x3/1")(x)
    x = Conv2D(192, (3, 3), padding="same", strides=(1, 1), activation="relu",
               name="conv_2b_3x3/1")(x)
    x = MaxPool2D((3, 3), padding="same", strides=(2, 2),
                  name="max_pool_2_3x3/2")(x)

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_3a")

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_3b")

    x = MaxPool2D((3, 3), padding="same", strides=(2, 2),
                  name="max_pool_3_3x3/2")(x)

    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4a")

    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding="same", activation="relu")(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation="relu")(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(10, activation="softmax", name="auxilliary_output_1")(x1)

    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4b")

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4c")

    x = inception_module(x,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4d")

    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding="same", activation="relu")(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation="relu")(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(10, activation="softmax", name="auxilliary_output_2")(x2)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_4e")

    x = MaxPool2D((3, 3), padding="same", strides=(2, 2),
                  name="max_pool_4_3x3/2")(x)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_5a")

    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         kernel_initialize=kernel_initialize,
                         kernel_regularize=kernel_regularize,
                         kernel_constraint=kernel_constraint,
                         bias_initialize=bias_initialize,
                         name="inception_5b")

    if include_top:

        x = GlobalAveragePooling2D(name="avg_pool_5_3x3/1")(x)

        # apply multiple sequential dense/dropout layers
        x = dense_drop_block(inputs=x,
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
                             mc_inference=mc_inference
                             )

        # Fully connected output layer (classification)
        x = Dense(num_classes, activation=classifier_activation,
                  name="output")(x)

    model = Model(input_layer, [x, x1, x2], name="Inception_v1")

    return model


def inception_module(
    x: tf.Tensor,
    filters_1x1: int,
    filters_3x3_reduce: int,
    filters_3x3: int,
    filters_5x5_reduce: int,
    filters_5x5: int,
    filters_pool_proj: int,
    kernel_initialize: Union[Initializer, str, None] = None,
    kernel_regularize: Union[float, str, None] = None,
    kernel_constraint: Union[int, None] = None,
    bias_initialize: Union[Initializer, None] = None,
    name: Optional[str] = None
) -> tf.Tensor:
    """Creates an inception module that combines 1x1, 3x3, 5x5 convolutions,
        and max pooling.

    Args:
        x: Input tensor.
        filters_1x1: Number of 1x1 convolution filters.
        filters_3x3_reduce: Number of 1x1 filters before the 3x3 convolution.
        filters_3x3: Number of 3x3 convolution filters.
        filters_5x5_reduce: Number of 1x1 filters before the 5x5 convolution.
        filters_5x5: Number of 5x5 convolution filters.
        filters_pool_proj: Number of filters for the 1x1 convolution after max
            pooling.
        kernel_initialize: Kernel initializer for convolution layers.
        kernel_regularize: Regularizer for convolution layers.
        kernel_constraint: Constraint on the kernel values.
        bias_initialize: Initializer for bias values.
        name: Name for the inception module.

    Returns:
        Output tensor representing the concatenation of the inception module's
            branches.
    """

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(x)

    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(x)

    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding="same",
                      activation="relu",
                      kernel_initializer=kernel_initialize,
                      kernel_regularizer=kernel_regularize,
                      kernel_constraint=kernel_constraint,
                      bias_initializer=bias_initialize)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding="same")(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding="same",
                       activation="relu",
                       kernel_initializer=kernel_initialize,
                       kernel_regularizer=kernel_regularize,
                       kernel_constraint=kernel_constraint,
                       bias_initializer=bias_initialize)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj],
                         axis=3,
                         name=name)

    return output
