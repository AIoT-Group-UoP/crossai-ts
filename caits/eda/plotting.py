from typing import Optional, List, Union
import pandas as pd
from matplotlib.figure import Figure as Fig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools


def plot_dim_reduced_scatter(
    dataframes_dict: dict,
    dimensions: int = 2,
    fig_size: tuple = (10, 8),
    fig_title: str = ''
) -> Fig:
    """Plots dimensional reduced scatter plots for the given
        dataframes with either 2D or 3D dimensions.

    Args:
        dataframes_dict: A dictionary where keys are labels (e.g., classes) and
                        values are corresponding DataFrames with
                        dimension-reduced data.
        dimensions: The number of dimensions for the reduced scatter plot
                    (2 or 3). Defaults to 2.
        fig_size: The size of the figure.
        fig_title: The title of the plot. Defaults to an empty string.

    Returns:
        The Figure object containing the scatter plot.

    Notes:
        - The function automatically generates a color and marker for each
            key in the `dataframes_dict` dictionary.
        - For more than 14 distinct keys in `dataframes`, markers will repeat.
    """

    # Color and marker setup
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataframes_dict)))
    markers = ['o', 's', 'v', '^', '<', '>', 'p',
               '*', 'H', '+', 'x', 'D', 'd', '|', '_']
    color_map = {disease: colors[i]
                 for i, disease in enumerate(dataframes_dict)}
    marker_map = {disease: markers[i % len(markers)]
                  for i, disease in enumerate(dataframes_dict)}

    # Setup figure and axes
    fig, ax = plt.subplots(
        figsize=fig_size,
        subplot_kw={'projection': '3d'} if dimensions == 3 else {}
    )

    # Scatter plot
    for label, df in dataframes_dict.items():
        if dimensions == 2:
            ax.scatter(df[0], df[1], color=color_map[label], s=30,
                       label=label, marker=marker_map[label], alpha=0.7)
        elif dimensions == 3:
            ax.scatter(df[0], df[1], df[2], color=color_map[label], s=10,
                       label=label, marker=marker_map[label], alpha=0.7)

    # Set labels and title
    ax.set_xlabel('Principal Component 1', fontsize=10)
    ax.set_ylabel('Principal Component 2', fontsize=10)
    if dimensions == 3:
        ax.set_zlabel('Principal Component 3', fontsize=10)
    if fig_title:
        plt.title(fig_title, fontsize=20)

    # Set legend and grid
    ax.legend(fontsize=8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    return fig


def plot_distribution(
    dataframe: pd.DataFrame,
    target_column: str,
    plot_type: str = 'boxplot',
    n_rows: int = 9,
    n_cols: int = 7
) -> Fig:
    """Plots each feature in the dataframe, differentiated by the class labels
    specified in the target_column, either as a boxplot or a histogram with
    Kernel Density Estimate (KDE).

    This function creates a grid of subplots with a shared x-axis across
    columns. Each subplot represents a different feature in the dataframe, and
    the `target_column` is used to categorize the data points in these plots.
    The function plots up to n_rows * n_cols features; if there are more
    features than subplots, only the first n_rows * n_cols features are
    plotted. For boxplots, the x-axis labels of the last row are set to class
    names from the `target_column` and rotated for readability.

    Args:
        dataframe: A pandas DataFrame containing the features to be plotted.
        target_column: Column in the DataFrame that contains class labels for
                        each instance. Used to differentiate the plots.
        plot_type: Type of plot to create for each feature. Options are
                    'boxplot' or 'histogram'. Defaults to 'boxplot'.
        n_rows: Number of rows in the subplot grid. Defaults to 9.
        n_cols: Number of columns in the subplot grid. Defaults to 7.

    Returns:
        The matplotlib Figure object containing the grid of plots.
    """

    # Create a figure and a grid of subplots with shared x-axis
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(30, 30), sharex='col')
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Loop through each feature and plot
    for i, feature in enumerate(dataframe.columns.drop(target_column)):
        if i >= n_rows * n_cols:  # Break more features than subplots
            break

        ax = axes[i]
        if plot_type == 'boxplot':
            sns.boxplot(x=target_column, y=feature, data=dataframe, ax=ax)
        elif plot_type == 'histogram':
            sns.histplot(data=dataframe, x=feature, hue=target_column,
                         element="step", kde=True, ax=ax)

        ax.set_title(f'{feature}')
        ax.set_ylabel('')

        # Only modify x-axis labels for the last row
        if i >= (n_rows - 1) * n_cols:
            ax.tick_params(axis='x', labelrotation=45)
        else:
            ax.set_xticklabels([])

    # Set x-axis labels with actual class names for the last row
    if plot_type == 'boxplot':
        for j in range((n_rows - 1) * n_cols, n_rows * n_cols):
            if j < len(axes):  # Check if the subplot exists
                axes[j].set_xticklabels(dataframe[target_column].unique(),
                                        rotation=45)
    return fig


def plot_triu_corr_heatmap(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    target_value: Optional[Union[str, int]] = None,
    title: str = 'Triangle Correlation Heatmap',
    cmap: str = 'coolwarm',
    fmt: str = ".2f",
    vmin: float = -1,
    vmax: float = 1,
    annot_size: int = 6,
    label_size: int = 6
) -> Fig:
    """Plots a triangular (upper triangle) correlation heatmap for the given
    DataFrame.

    This function creates a heatmap showing the Pearson correlation
    coefficients between numeric features in a DataFrame. Optionally, if
    `target_column` and `target_value` are specified, the function filters the
    DataFrame to include only rows where the 'target_column' matches the
    'target_value' before calculating correlations. Non-numeric columns are
    dropped from the calculation. The heatmap is triangular, showing only the
    upper triangle to avoid redundancy.

    Args:
        df: A pandas DataFrame containing the data for which to compute the
            correlation heatmap.
        target_column: Optional; filters the DataFrame to include only rows
                        where this column matches the 'target_value'.
        target_value: The specific value for filtering rows in conjunction
                        with 'target_column'.
        title: The title of the heatmap.
                Defaults to 'Triangle Correlation Heatmap'.
        cmap: Colormap for the heatmap. Defaults to 'coolwarm'.
        fmt: String format for annotations. Defaults to ".2f".
        vmin: Minimum value for colormap scale. Defaults to -1.
        vmax: Maximum value for colormap scale. Defaults to 1.
        annot_size: Font size for annotations. Defaults to 6.
        label_size: Font size for feature labels. Defaults to 6.

    Returns:
        The Figure object containing the heatmap.

    Raises:
        An exception is raised if `target_column` is provided but
            `target_value` is None.
    """

    fig, ax = plt.subplots(figsize=(25, 10))

    # Filter rows to a specific target_vavlue
    if target_column is not None:
        if target_value is not None:
            df = df[df[target_column] == target_value]
        else:
            raise ValueError("`target_value` cannot be None.")

    # drop non-numeric columns
    # in case `target_column` is numeric, an extra row on the correlation
    # heatmap will be created
    df = df.select_dtypes(include=['number'])

    # calculate pearson correlation
    corr_matrix = df.corr()

    # Define the mask to set the values in the upper triangle to True
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        cmap=cmap,
        fmt=fmt,
        annot_kws={"size": annot_size}
    )
    # Adjusting font size for feature labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=label_size)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=label_size)
    heatmap.set_title(title, fontdict={'fontsize': 15}, pad=16)

    return fig


def plot_scatter_feature_pairs(
    df: pd.DataFrame,
    target_column: str,
    target_value: Optional[str] = None,
    feature_list: List[str] = [],
    n_rows: int = 10,
    n_cols: int = 5
) -> Fig:
    """Creates a grid of scatter plots for combinations of features
    from a given DataFrame.

    This function plots scatter plots for each pair of features specified in
    `feature_list`. If `target_column` and `target_value` are specified, the
    function filters the DataFrame to include only rows where the
    'target_column' matches the 'target_value' before plotting. The function
    creates a grid of scatter plots with a specified number of rows (`n_rows`)
    and columns (`n_cols`). Each plot shows the relationship between two
    features. If there are more feature combinations than subplots available in
    the grid, excess combinations are not plotted.

    Args:
        df: Pandas DataFrame containing the data to be plotted.
        target_column: Optional; if specified, the DataFrame is filtered to
                       include only rows where this column matches the
                       'target_value'.
        target_value: The specific value for filtering rows in conjunction
                      with 'target_column'.
        feature_list: List of features to be plotted. Combinations are made
                      from these features.
        n_rows: Number of rows in the subplot grid. Defaults to 10.
        n_cols: Number of columns in the subplot grid. Defaults to 5.

    Returns:
        A matplotlib Figure object (Fig) containing the grid of scatter plots.

    Raises:
        ValueError: If the number of feature combinations exceeds 100.
    """

    # Create combinations of the selected features
    feature_combinations = list(itertools.combinations(feature_list, 2))

    # Check if combinations exceed 100
    if len(feature_combinations) > 100:
        raise ValueError("Number of feature combinations exceeds 100.")

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(30, 30))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    if target_value is not None:
        dataframe = df[df[target_column] == target_value]
    else:
        dataframe = df

    # Loop through each combination and plot
    for i, (feature1, feature2) in enumerate(feature_combinations):
        if i >= n_rows * n_cols:  # Break if more combinations than subplots
            break

        sns.scatterplot(x=feature1, y=feature2,
                        hue=target_column, data=dataframe, ax=axes[i])
        axes[i].set_title(f'Scatter Plot for {feature1} vs {feature2}')

    # Hide any unused subplots
    for j in range(i+1, n_rows * n_cols):
        fig.delaxes(axes[j])

    return fig


def plot_explained_variance(
    exp_var_ratio: List[float],
    title: str = "Explained Variance"
) -> Fig:
    """Plots the explained variance and cumulative explained variance from the
    provided PCA explained variance ratios.

    This function creates a bar and step plot illustrating the distribution of
    explained variance across principal components. The bar plot shows the
    individual explained variance for each component, while the step plot
    displays the cumulative explained variance. This visualization helps in
    understanding how much variance each principal component contributes and
    how many components are needed to explain a desired proportion of the
    total variance in the dataset.

    Args:
        exp_var_ratio: Array of explained variance ratios for each principal
                       component. This should be obtained from the results of
                       PCA performed on a dataset.
        title: Title of the plot. Defaults to "Explained Variance".

    Returns:
        fig: A matplotlib figure object representing the plot.
    """

    # Cumulative sum of explained variances
    cum_sum_eigenvalues = np.cumsum(exp_var_ratio)

    # Create the visualization plot
    fig, ax = plt.subplots()
    ax.bar(range(0, len(exp_var_ratio)), exp_var_ratio, alpha=0.5,
           align='center', label='Individual explained variance')
    ax.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues,
            where='mid', label='Cumulative explained variance')
    ax.set_ylabel('Explained variance ratio')
    ax.set_xlabel('Principal component index')
    ax.set_title(title)
    ax.legend(loc='best')

    return fig
