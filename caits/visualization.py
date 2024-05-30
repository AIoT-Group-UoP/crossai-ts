import os
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure as Fig
import seaborn as sns


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_prediction_probas(
    probabilities: np.ndarray,
    sr: int,
    ws: float,
    overlap_percentage: float,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    mode: str = "samples",  
    events: Optional[List[Tuple[float, float, int]]] = None,
    title: Optional[str] = "Prediction Probabilities Across Windows",
) -> plt.Figure:
    """Plots prediction probabilities as horizontal lines with optional event highlighting.

    Args:
        probabilities: Prediction probabilities, shape (n_instances, n_classes).
        sr: Sampling rate of the time series.
        ws: Window size in seconds.
        overlap_percentage: Overlap percentage between windows (0 to 1).
        class_names: A list of class names for labeling purposes.
                     If not provided, classes will be labeled numerically.
        figsize: Figure size in inches.
        mode: Plot mode - "samples" or "time". Defaults to "samples".
        events: List of event tuples (start, end, class). Start and end are in
                samples. If mode="time" and sr is provided, they will be converted to time units.

    Returns:
        The matplotlib Figure object.
    """

    ws_samples = int(ws * sr)
    OP_step = int(ws_samples * (1 - overlap_percentage))

    # Colors for each class
    colors = plt.cm.jet(np.linspace(0, 1, probabilities.shape[1]))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot probabilities
    for i, class_probs in enumerate(probabilities.T):
        label = class_names[i] if class_names is not None else f"Class {i + 1}"
        for j, prob in enumerate(class_probs):
            start_idx = j * OP_step / sr if mode == "time" and sr is not None else j * OP_step
            end_idx = start_idx + OP_step / sr if mode == "time" and sr is not None else start_idx + OP_step
            ax.hlines(prob, start_idx, end_idx, colors=colors[i], lw=2, label=label if j == 0 else "")

    # Fill events (optional)
    if events is not None:
        unique_classes = set([event[2] for event in events])
        palette = sns.color_palette("pastel", len(unique_classes))
        class_colors = {cls: color for cls, color in zip(unique_classes, palette)}

        for start, end, cls in events:
            class_label = class_names[cls] if class_names else f"Class {cls}"
            if mode == "time" and sr is not None:
                start, end = start / sr, end / sr
            ax.axvspan(start, end, color=class_colors[cls], alpha=0.5, label=class_label)


    # Set labels and title
    ax.set_xlabel("Time (s)" if mode == "time" else "Samples")
    ax.set_ylabel("Probability")
    ax.set_title(title)

    ax.grid(True)
    
    # Create a combined legend for both channel and event classes
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    return fig


def plot_interpolated_probas(
    interpolated_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
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
        label = class_names[i] if class_names is not None else f"Class {i + 1}"
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
    resolution: Union[float, str] = "figure",
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

    if "save" in export and save_path:
        if not os.path.isdir(save_path):
            raise FileNotFoundError(
                f"Provided path '{save_path}' does not \
                                      exist or is not a directory."
            )

        file_path = os.path.join(save_path, f"{fig_id}.{fig_extension}")
        dpi = resolution if isinstance(resolution, float) else None
        fig_object.savefig(file_path, format=fig_extension, bbox_inches="tight", dpi=dpi)
        print(f"Figure saved to {file_path}")

    if "show" in export:
        plt.show()

    if "save" not in export and "show" not in export:
        raise ValueError(
            "Invalid export option. Use 'save', 'show', \
                          or 'both'."
        )

    return


def plot_signal(
    sig: np.ndarray,
    sr: int = 44100,
    mode: str = "samples",
    name: str = "Signal",
    channels: Optional[Union[List[str], str]] = None,
    figsize: Tuple[int, int] = (10, 4),
    events: Optional[List[Tuple[int, int, int]]] = None,
    class_names: Optional[List[str]] = None,
    return_mode: bool = True,
) -> Optional[plt.Figure]:
    """Plots a signal with optional event highlighting (with classes)
    and returns the figure object.

    Args:
        sig: The input signal as a 2D numpy array (timesteps, channels).
        sr: The sampling rate of the signal. Defaults to 44100.
        mode: Plot mode - "samples" or "time". Defaults to "samples".
        name: Name of the signal. Defaults to "Signal".
        channels: Channel names, applicable for multichannel signals
                    or a single label.
        figsize: Figure size in inches. Defaults to (10, 4).
        events: List of event tuples (start, end, class). Start and end are in
                samples. If mode="time" and sr is provided, they will be converted to time units.
        class_names: List of class names corresponding to the class indices in events.
        return_mode: Whether to return the plot in the function. Defaults to True.

    Returns:
        plt.Figure: The figure object containing the plot.
    """

    # Ensure sig is at least 1D
    if sig.ndim == 0:
        raise ValueError("Input signal 'sig' must be at least 1-dimensional")

    # Convert to 2D if necessary
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)  # Reshape to (n, 1)

    fig = plt.figure(figsize=figsize)
    plt.title(name)

    num_channels = sig.shape[1]  # Number of channels (m)
    
    # Handle channels and labels
    if channels is None:
        channels = [f"Channel {i + 1}" for i in range(num_channels)]
    elif isinstance(channels, str):
        channels = [channels]
    elif len(channels) != num_channels:
        raise ValueError("Number of channels in 'channels' must match signal shape")

    # Create x_axis based on mode and signal length
    if mode == "time" and sr is not None:
        x_axis = np.linspace(0, sig.shape[0] / sr, num=sig.shape[0])
    else:
        x_axis = np.arange(sig.shape[0])
        sr = 1  # If in sample mode or sr is not provided, treat sr as 1 for event conversion

    # Plot each channel
    for i, channel in enumerate(sig.T):  # Transpose to iterate over channels
        plt.plot(x_axis, channel, label=channels[i])

    # Fill event areas with class-based colors and labels
    if events:

        # Color palette generation
        unique_classes = set([event[2] for event in events]) if events else set()
        palette = sns.color_palette("pastel", len(unique_classes))
        class_colors = {cls: color for cls, color in zip(unique_classes, palette)}

        for start, end, cls in events:
            class_label = class_names[cls] if class_names else f"Class {cls}"
            # Convert to time if needed
            if mode == "time" and sr is not None:
                start, end = start / sr, end / sr
            plt.axvspan(start, end, color=class_colors[cls], alpha=0.5, label=class_label)

    plt.xlabel("Time (s)" if mode == "time" else "Samples")
    plt.ylabel("Amplitude")

    plt.grid(True)

    if channels:
        plt.legend()

    # Create a combined legend for both channel and event classes
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if return_mode:
        return fig
    
    return


def plot_spectrogram(
    f: np.ndarray,
    t: np.ndarray,
    spec: np.ndarray,
    factor: int = 1,
    log: Optional[str] = None,
    plot_title: str = "Spectrogram",
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


def plot_simple_spectrogram(spectrogram: np.ndarray, title: str = "Spectrogram") -> None:
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
