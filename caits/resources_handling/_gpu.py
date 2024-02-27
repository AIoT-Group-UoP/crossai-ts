import tensorflow as tf


def tf_exploit_gpu_physical_growth(print_setup: bool = True) -> None:
    """Sets memory growth for all PhysicalDevices available.

       This functionality prevents TensorFlow from allocating
       all memory on the device.
       Source:
       https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth

       Args:
           print_setup: If False, then nothing is printed
            to the console about the GPUs' list and memory
            growth setup.

       Returns:

    """

    gpus = tf.config.list_physical_devices('GPU')
    if print_setup:
        print(f"Set up growth dynamic limit in all "
              f"the available GPUs: {gpus}", "info")
    gpu_list = []
    if gpus:
        try:
            for i in range(len(gpus)):
                tf.config.experimental.set_memory_growth(gpus[i], True)
                print("Growth set for GPU {}".format(i), "debug")
                assert tf.config.experimental.get_memory_growth(gpus[i])
                gpu_list.append(gpus[i].name)
            if print_setup:
                print("Growth for all the GPUs was set up "
                      "successfully.", "success")
        except RuntimeError as e:
            print("No successful set up of GPUs' growth.")
            print(e)
    if print_setup:
        print("GPUs in dynamic allocation functionality:", gpu_list)
