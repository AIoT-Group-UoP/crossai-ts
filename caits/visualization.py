import os
from typing import Optional, Union, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as Fig


def plot_prediction_probas(
        probabilities: np.ndarray,
        sampling_rate: int,
        Ws: float,
        overlap_percentage: float,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 6)
) -> Fig:
    """Plots prediction probabilities as small horizontal lines, adjusting
    for window overlap. Only non-overlapping parts of the window segments
    are plotting for visualization purposes and time-matching of the original
    time series length and the visualized one.

    Args:
        predictions: Prediction probabilities, shape (n_instances, n_classes).
        sampling_rate: Sampling rate of the time series.
        Ws: Window size in seconds.
        overlap_percentage: Overlap percentage between windows (0 to 1).
        class_names: A list of class names for labeling purposes.
                     If not provided, classes will be labeled numerically.
        figzise: Figure size in inches.

    Returns:
        The matplotlib Figure object.
    """
    # Convert window size from seconds to samples
    Ws_samples = int(Ws * sampling_rate)

    # Calculate the step size based on the overlap
    OP_step = int(Ws_samples * (1 - overlap_percentage))

    # Colors for each class
    colors = plt.cm.jet(np.linspace(0, 1, probabilities.shape[1]))

    fig, ax = plt.subplots(figsize=figsize)

    # Iterate through each class
    # and plot
    for i, class_probs in enumerate(probabilities.T):
        label = class_names[i] if class_names is not None else f"Class {i+1}"
        for j, prob in enumerate(class_probs):
            start_idx = j * OP_step
            end_idx = start_idx + OP_step

            ax.hlines(prob, start_idx, end_idx, colors=colors[i], lw=2,
                      label=label if j == 0 else "")

    # Setting labels and title
    ax.set_xlabel("Instances")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities Across Time Series Windows")

    # Handling legend for multiple classes
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys())

    return fig


def plot_interpolated_probas(
        interpolated_probs: np.ndarray,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """Plots the interpolated prediction probabilities for each class.

    Args:
        interpolated_probs: 2D array of interpolated probabilities,
                            where each column represents a class.
        class_names: A list of class names for labeling purposes.
                     If not provided, classes will be labeled numerically.
        figzise: Figure size in inches.

    Returns:
        The matplotlib Figure object.
    """
    n_points, n_classes = interpolated_probs.shape
    x_interpolated = np.linspace(0, n_points - 1, num=n_points)

    # Colors for each class
    colors = plt.cm.jet(np.linspace(0, 1, interpolated_probs.shape[1]))

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_classes):
        label = class_names[i] if class_names is not None else f"Class {i+1}"
        ax.plot(x_interpolated, interpolated_probs[:, i], color=colors[i], label=label)

    plt.title("Interpolated Prediction Probabilities")
    plt.xlabel("Interpolated Instances")
    plt.ylabel("Probability")
    plt.legend()

    return fig


def export_fig(
    fig_object: plt.Figure,
    fig_id: str,
    save_path: Optional[str] = None,
    export: str = "save",
    tight_layout: bool = True,
    fig_extension: str = "png",
    resolution: Union[float, str] = "figure"
) -> None:
    """
    Exports a matplotlib figure object by saving, showing, or doing both.

    Args:
        fig_object: The matplotlib figure object to export.
        fig_id: Unique identifier for the figure. Used in naming the saved
                file.
        save_path: The path of the local saving directory. Required if "export"
                   includes "save".
        export: Determines the action to perform - "save", "show", or "both".
                Defaults to "save".
        tight_layout: Whether to apply tight layout adjustment before
                      exporting. Defaults to True.
        fig_extension: Format of the figure file if saving. Defaults to "png".
        resolution: Resolution of the exported figure if saving. Can be a float
                    or "figure". Defaults to "figure".

    Returns:
        None. The figure is either saved, shown, or both, based on the "export"
              argument.
    """
    if "save" in export and not save_path:
        raise ValueError("Save path must be provided to save the figure.")

    if tight_layout:
        fig_object.tight_layout()

    if "save" in export:
        if not os.path.isdir(save_path):
            raise FileNotFoundError(f"Provided path '{save_path}' does not \
                                      exist or is not a directory.")

        file_path = os.path.join(save_path, f"{fig_id}.{fig_extension}")
        dpi = resolution if isinstance(resolution, float) else None
        fig_object.savefig(file_path, format=fig_extension,
                           bbox_inches="tight", dpi=dpi)
        print(f"Figure saved to {file_path}")

    if "show" in export:
        plt.show()

    if "save" not in export and "show" not in export:
        raise ValueError("Invalid export option. Use 'save', 'show', \
                          or 'both'.")


def plot_signal(
    sig: np.ndarray,
    sr: int = 44100,
    mode: str = "samples",
    name: str = "Signal",
    channels: Optional[Union[List[str], str]] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """Plots a signal and returns the matplotlib figure object.

    Args:
        sig: The input signal as a numpy array.
        sr: The sampling rate of the signal. Defaults to 44100.
        mode: Plot mode - "samples" or "time". Defaults to "samples".
        name: Name of the signal. Defaults to "Signal".
        channels: Channel names, applicable for multi-channel signals
                  or a single label.
        figsize: Figure size in inches. Defaults to (10, 4).

    Returns:
        plt.Figure: The figure object containing the plot.
    """
    fig = plt.figure(figsize=figsize)
    plt.title(name)

    if isinstance(channels, str):
        channels = [channels]  # Convert single string to list for consistency

    # Handling multi-channel signals
    if sig.ndim > 1 and channels and len(channels) == sig.shape[0]:
        for i, channel in enumerate(sig):
            if mode == "time":
                t = np.linspace(0, len(channel) / sr, num=len(channel))
                plt.plot(t, channel, label=channels[i])
            elif mode == "samples":
                plt.plot(channel, label=channels[i])
    else:
        if mode == "time":
            t = np.linspace(0, len(sig) / sr, num=len(sig))
            plt.plot(t, sig, label=channels[0] if channels else None)
        elif mode == "samples":
            plt.plot(sig, label=channels[0] if channels else None)

    plt.xlabel("Time (s)" if mode == "time" else "Samples")
    plt.ylabel("Amplitude")

    # Adding legend if channel labels are provided or if a
    # single label is provided for single-channel signal
    if channels:
        plt.legend()

    return fig


def plot_spectrogram(
        f: np.ndarray,
        t: np.ndarray,
        spec: np.ndarray,
        factor: int = 1,
        log: str = None,
        plot_title: str = "Spectrogram"
) -> None:
    """Plots the spectrogram.

    Args:
        f: The array of sample frequencies in np.ndarray.
        t: The array of segment times in np.ndarray.
        spec: The spectrogram to plot in 2D np.ndarray.
        factor: The factor to multiply the log scale with. Defaults to 10.
        log: The log scale to use. If None, defaults to 10 * np.log10(spec).
        plot_title: The title of the plot. Defaults to "Spectrogram".

    Returns:

    """

    if log == "log10":
        spec_plot = factor * np.log10(spec)  # Convert to dB: 10 * log10(spec)
    elif log == "log2":
        spec_plot = factor * np.log2(spec)
    elif log == "log":
        spec_plot = factor * np.log(spec)
    elif log is None:
        spec_plot = factor * spec
    else:
        raise ValueError("log must be 'log10', 'log2', 'log', or None")

    plt.pcolormesh(t, f, spec_plot, shading="gouraud")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.title(plot_title)
    plt.colorbar(label="Intensity [dB]")
    plt.show()


def plot_simple_spectrogram(
        spectrogram: np.ndarray, 
        title: str = "Spectrogram") -> None:
    """Simple function that plots a Spectrogram.

    Args:
        spectrogram: The array of the spectrogram in 2D np.ndarray.

    Returns:

    """
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()        
