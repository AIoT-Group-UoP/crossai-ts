import matplotlib.pyplot as plt
from typing import Optional, Union
import os


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
        save_path: The path of the local saving directory. Required if 'export'
                   includes 'save'.
        export: Determines the action to perform - 'save', 'show', or 'both'.
                Defaults to 'save'.
        tight_layout: Whether to apply tight layout adjustment before
                      exporting. Defaults to True.
        fig_extension: Format of the figure file if saving. Defaults to 'png'.
        resolution: Resolution of the exported figure if saving. Can be a float
                    or 'figure'. Defaults to 'figure'.

    Returns:
        None. The figure is either saved, shown, or both, based on the 'export'
              argument.
    """
    if 'save' in export and not save_path:
        raise ValueError("Save path must be provided to save the figure.")

    if tight_layout:
        fig_object.tight_layout()

    if 'save' in export:
        if not os.path.isdir(save_path):
            raise FileNotFoundError(f"Provided path '{save_path}' does not \
                                      exist or is not a directory.")

        file_path = os.path.join(save_path, f"{fig_id}.{fig_extension}")
        dpi = resolution if isinstance(resolution, float) else None
        fig_object.savefig(file_path, format=fig_extension,
                           bbox_inches="tight", dpi=dpi)
        print(f"Figure saved to {file_path}")

    if 'show' in export:
        plt.show()

    if 'save' not in export and 'show' not in export:
        raise ValueError("Invalid export option. Use 'save', 'show', \
                          or 'both'.")


def plot_spectrogram(
        f: np.ndarray,
        t: np.ndarray,
        spec: np.ndarray,
        factor: int = 1,
        log: str = None,
        plot_title: str = "Spectrogram"
):
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

    plt.pcolormesh(t, f, spec_plot, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(plot_title)
    plt.colorbar(label='Intensity [dB]')
    plt.show()
