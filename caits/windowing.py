import pandas as pd
from typing import Union


def sliding_window_df(
        df: pd.DataFrame,
        ws: int = 500,
        overlap: int = 250,
        w_type: str = "hann",
        w_center: bool = True,
        print_stats: bool = False
) -> list:
    """Applies the sliding window algorithm to the DataFrame rows.

    Args:
        df: The DataFrame with all the values that will be inserted to the
            sliding window algorithm.
        ws: The window size in number of samples.
        overlap: The hop length in number of samples.
        w_type: The windowing function.
        w_center: If False, set the window labels as the right edge of the
            window index. If True, set the window labels as the center of the
            window index.
        print_stats: Print statistical inferences from the process.
            Defaults to False.

    Returns:
        A list of DataFrames each one corresponding to a produced window.
    """
    counter = 0
    windows_list = list()
    # min_periods: Minimum number of observations in window required to
    # have a value;
    # For a window that is specified by an integer, min_periods will default
    # to the size of the window.
    for window in df.rolling(window=ws, step=overlap, min_periods=ws,
                             win_type=w_type, center=w_center):
        if window[window.columns[0]].count() >= ws:
            if print_stats:
                print("Print Window:", counter)
                print("Number of samples:", window[window.columns[0]].count())
            windows_list.append(window)
        counter += 1
    if print_stats:
        print("List number of window instances:", len(windows_list))

    return windows_list


def windowing_df(
        df: pd.DataFrame,
        ws: int = 500,
        overlap: int = 250,
        w_type: str = "hann",
        w_center: bool = False,
        mode: str = "dict"
) -> Union[dict, pd.DataFrame, str]:
    """Applies the sliding window algorithm to the DataFrame rows and returns
    the windows as a dictionary or a DataFrame, containing the corresponding
    labels.

    Args:
        df: The DataFrame with all the values that will be inserted to the
            sliding window algorithm.
        ws: The window size in number of samples. Defaults to 500.
        overlap: The hop length in number of samples. Defaults to 250.
        w_type: The windowing function. Defaults to "hann".
        w_center: If False, set the window labels as the right edge of the
            window index. If True, set the window labels as the center of the
            window index. Defaults to False.
        mode: The output mode. Either "dict" or "df". Defaults to "dict".

    Returns:
        A dictionary or a DataFrame containing the windows and the labels. The
        "X" key or column contains the windows and the "y" key or column
        contains the labels.
    """
    windows_list = []
    y_list = []
    for index, row in df.iterrows():
        windows = sliding_window_df(row["X"], ws=ws, overlap=overlap,
                                    w_type=w_type, w_center=w_center,
                                    print_stats=False)
        windows_list.extend(windows)
        y_list.extend([row["y"]] * len(windows))

    if mode == "dict":
        return {"X": windows_list, "y": y_list}
    elif mode == "df":
        return pd.DataFrame({"X": windows_list, "y": y_list})
    else:
        return "Invalid mode. Use 'dict' or 'df' as mode."
