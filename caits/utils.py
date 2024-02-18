import numpy as np


def df_list_to_array(df_list) -> np.ndarray:
    """Converts a list of DataFrames to a 3D numpy array, where the first
    dimension is the number of instances, the second dimension is the number
    of rows (windows) in each instance (DataFrame), and the third dimension is
    the number of columns (channels, axes) in each DataFrame.

    Args:
        df_list: The list of DataFrames.

    Returns:
        np.ndarray: The 3D numpy array.
    """
    array = np.array(list(map(lambda x: x.to_numpy(), df_list)))

    return array
