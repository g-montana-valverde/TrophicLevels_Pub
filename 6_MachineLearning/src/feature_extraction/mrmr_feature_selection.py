"""
Contains the main function for MRMR feature selection
"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression


def mrmr_feature_selection(X, y, k=15):
    """
    Minimum redundancy maximum relevance (mRMR) feature selection.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    k : int, optional (default=15)
        Number of features to select

    Returns:
    --------
    selected_features : list
        Indices of selected features
    """

    n_samples, n_features = X.shape

    # If we have fewer features than k, return all feature indices
    if n_features <= k:
        return list(range(n_features))

    # Calculate mutual information between features and target using mutual_info_regression
    relevance = mutual_info_regression(X, y)

    # Initialize variables
    selected = []
    not_selected = list(range(n_features))

    # Select first feature (maximum relevance)
    first = np.argmax(relevance)
    selected.append(first)
    not_selected.remove(first)

    # Select remaining features
    for _ in range(k - 1):
        if not not_selected:
            break

        # Calculate redundancy and relevance scores
        scores = []
        for j in not_selected:
            # Calculate redundancy (average mutual information with selected features)
            redundancy = 0
            if selected:
                X_candidate = X[:, j].reshape(-1, 1)
                redundancy = np.mean(
                    [
                        mutual_info_regression(X_candidate, X[:, sel])[0]
                        for sel in selected
                    ]
                )

            # MODIFIED: Use a proper mRMR score formula using ratio instead of subtraction
            if redundancy < 1e-10:  # Effectively zero
                score = relevance[j]
            else:
                score = relevance[j] / redundancy
            scores.append(score)

        # Select feature with maximum score
        next_feature = not_selected[np.argmax(scores)]
        selected.append(next_feature)
        not_selected.remove(next_feature)

    return selected
