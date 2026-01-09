"""
Module with functions to load data from excel files and process it for scikit-learn-based
training of models.
"""

import numpy as np
import pandas as pd


def get_x_arr_for_scikit_from_excel(
    file_path, dict_groups_labels, id_key="ID", group_key="Group"
):
    """
    Loads the excel file in file_path and transforms the data into typical X,
    y-like arrays for scikit-learning training.

    This script is adapted to excel files .xlsx that have the following columns:
    [id_key, group_key, feature_1, feature_2, ....]

    Args:
        file_path: path to the file to load
        dict_groups_labels: dictionary that maps the group string to the int label we
            assign to it. Example: {"HC": 0, "AD": 1}
        id_key: Column name for the subject IDs
        group_key: Column name for the subject group

    Returns:
        x_arr: input-array with shape (n_samples, n_features)
        y_arr: labels-array with shape (n_samples, )
        feature_columns: list of str with the names of the columns

    """
    if len(dict_groups_labels) != 2:
        raise ValueError(
            f"Skipping {file_path} as dict_group_labels passed does not contain only two groups"
        )

    # Read the Excel file
    df_data = pd.read_excel(file_path)

    # Check if the 'Group' column exists
    if group_key not in df_data.columns:
        raise ValueError(
            f"ERROR loading data from {file_path} as it does not contain a {group_key} column"
        )

    # Get the data in the Group key that matches the keys in the dict_groups_labels
    df_data_groups = df_data[df_data[group_key].isin(list(dict_groups_labels.keys()))]

    # Encode the 'Group' column
    df_data_groups["Group"] = df_data_groups["Group"].map(dict_groups_labels)

    # Prepare features and labels
    y_arr = np.array(df_data_groups["Group"])

    feature_columns = [
        col for col in df_data_groups.columns if col not in [group_key, id_key]
    ]
    x_arr = np.array(df_data_groups[feature_columns])

    return x_arr, y_arr, feature_columns
