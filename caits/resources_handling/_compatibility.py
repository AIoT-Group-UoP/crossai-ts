def libs_compatibility() -> None:
    """Checks the system's Python and AI libraries' compatibility.

    Returns:

    """
    python_compatibility()
    sklearn_compatibility()
    tf_compatibility()


def python_compatibility() -> None:
    """Checks the Python version. Python ≥ 3.10 is required

    Returns:

    """
    import sys

    assert sys.version_info >= (3, 10)
    print(">> Python is compatible.")


def sklearn_compatibility() -> None:
    """Checks the scikit-learn version. scikit-learn ≥ 1.3.0 is required.

    Returns:

    """
    import sklearn

    assert sklearn.__version__ >= "1.4.0"
    print(">> scikit-learn is compatible.")


def tf_compatibility() -> None:
    """Checks the TensorFlow version. TensorFlow ≥ 2.13 is required.

    Returns:

    """
    import tensorflow as tf

    assert tf.__version__ == "2.14"
    print(">> TensorFlow is compatible.")

    if not tf.config.list_physical_devices("GPU"):
        print("No GPU is detected. LSTMs and CNNs can be very " "slow without a GPU. A GPU usage is recommended.")
    else:
        print(">> GPU usage is detected:")
        gpus = tf.config.list_physical_devices("GPU")
        gpu_count = 0
        for gpu in gpus:
            print(">> * GPU {}: {}".format(gpu_count, gpu))
