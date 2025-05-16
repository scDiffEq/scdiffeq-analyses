# -- import packages: ---------------------------------------------------------
import numpy as np
import pandas as pd


# -- function: ----------------------------------------------------------------
def rowwise_negative_cross_entropy(df1: pd.DataFrame, df2: pd.DataFrame):
    # Ensure that the dataframes have the same shape
    assert df1.shape == df2.shape, "DataFrames must have the same shape"

    df1 = df1.sort_index(axis=1)
    df2 = df2.sort_index(axis=1)

    # Convert the dataframes to numpy arrays for easier manipulation
    arr1 = df1.values
    arr2 = df2.values

    # Clip values to avoid log(0), which is undefined
    epsilon = 1e-12
    arr1 = np.clip(arr1, epsilon, 1.0 - epsilon)
    arr2 = np.clip(arr2, epsilon, 1.0 - epsilon)

    # Calculate negative cross-entropy row by row
    return -np.sum(arr2 * np.log(arr1), axis=1)
