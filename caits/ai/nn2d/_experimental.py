from typing import Callable, Optional, Tuple, Union

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D

from caits.ai import dropout_layer_1d, dropout_layer_2d


def CNN2D(
    input_shape: Tuple[int, ...],
    include_top: bool = True,
    num_classes: int = 1,
    classifier_activation: Union[str, Callable] = "softmax",
    drp_rate: float = 0.0,
    spatial: bool = False,
    mc_inference: Optional[bool] = None,
) -> Model:
    """Creates a simple 2D CNN model for experimental purposes.

    Args:
        input_shape: Shape of the input data, excluding the batch size.
        include_top: If True, includes a fully-connected layer at the top
            of the model.
        num_classes: Number of classes for prediction.
        classifier_activation: Activation function for the classification task.
            Can be a string identifier or a function from tf.keras.activations.
        drp_rate: Dropout rate.
        spatial: If true, applies Spatial Dropout, else applies standard
            Dropout.
        mc_inference: If True, enables Monte Carlo dropout
            during inference.

    Returns:
        model: An instance of a Keras Model.
    """

    # Define input tensor for the network, batch size is omitted
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Conv2D(32, (3, 3), activation="relu")(input_layer)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D(3)(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)

    if include_top is True:
        # retain tensor shape (keepdims) since Spatial Dropout expects 4D input
        x = GlobalAveragePooling2D(keepdims=True)(x)
        x = dropout_layer_2d(inputs=x, drp_rate=drp_rate, spatial=spatial, mc_inference=mc_inference)
        x = Flatten()(x)

        x = Dense(128, activation="relu")(x)
        x = dropout_layer_1d(inputs=x, drp_rate=drp_rate)

        outputs = Dense(num_classes, activation=classifier_activation)(x)
    else:
        outputs = x

    model = Model(inputs=input_layer, outputs=outputs, name="CNN2D")

    return model
