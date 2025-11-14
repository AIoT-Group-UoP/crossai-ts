from typing import List

import numpy as np
import pandas as pd


def get_high_corr_features_df(df: pd.DataFrame, threshold: float = 0.75) -> List[str]:
    """Selects features from a DataFrame that have a high correlation with each
    other, based on a specified correlation threshold. It excludes perfect
    self-correlations (correlation of a feature with itself).

    Args:
        df: The DataFrame from which features are to be selected.
        threshold: The threshold for selecting highly correlated features.
            Features with a correlation higher than this threshold will be
            selected. Defaults to 0.75.

    Returns:
        A list of feature names that have a correlation higher than the
        specified threshold.
    """

    # Get the absolute value of the correlation matrix
    corr_matrix_abs = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))

    # Find features with correlation greater than `low`
    high_corr_features = [column for column in upper.columns if any(upper[column] >= threshold)]

    return high_corr_features
