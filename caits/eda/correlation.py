from typing import List
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau


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


def corr(arr: np.ndarray, axis=0, method="pearson", export="array"):
    """Calculates the correlation between features of an array."""

    _arr = arr if axis == 0 else arr.T

    if method == "pearson":
        fun = pearsonr
    elif method == "spearman":
        fun = spearmanr
    elif method == "kendall":
        fun = kendalltau
    else:
        raise ValueError("method must be either 'pearson' or 'spearman' or 'kendall'")

    if export == "array":
        res = np.ones((_arr.shape[1], _arr.shape[1]))
    elif export == "dict":
        res = {}
    else:
        raise ValueError("export must be either 'array' or 'dict'")

    for i in range(_arr.shape[1]):
        for j in range(i + 1, _arr.shape[1]):
            tmp, _ = fun(_arr[:, i], _arr[:, j])
            if export == "dict":
                res[(i, j)] = tmp
            else:
                res[i, j] = tmp
                res[j, i] = tmp

    return res


def get_high_corr_features(arr: np.ndarray, axis=0, method="pearson", threshold: float = 0.75):
    """Selects features from an array that have a high correlation with each"""

    corrs = corr(arr, axis=axis, method=method, export="dict")
    feats = set()

    for k, v in corrs.items():
        if abs(v) > threshold:
            feats = feats.union(set(k))

    feats = list(feats)
    feats.sort()
    return feats


def keep_high_corr_features(arr: np.ndarray, axis=0, method="pearson", threshold: float = 0.75):
    """Selects features from an array that have a high correlation with each"""

    feats = get_high_corr_features(arr, axis=axis, method=method, threshold=threshold)

    if axis == 0:
        return arr[:, feats]
    else:
        return arr[feats, :]