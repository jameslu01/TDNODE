import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import utils
import numpy as np

def split_datasets(root_data_path: str, random_seed: int, train_fraction: float) -> None:
    """Performs random split of dataset and exports cohorts to respective train and test .csv files.

    Parameters
    ----------
    root_data_path : str
        The directory that contains a .csv file of the entire dataset.
    random_seed : int
        An integer that makes the splitting process deterministic.
    train_fraction : float
        A float with range [0, 1] that specificies the proportion of subjects to allocate to the
        training set. (1 - train_fraction) is thus the proportion of subjects allocated to the test
        set.
    """
    utils.log_message(f"Splitting dataset. Root data path: {root_data_path}/tgi.csv")
    df = pd.read_csv(f"{root_data_path}/tgi.csv")
    print(df)
    df["TIME"] = df["TIME"].astype(float).apply(lambda df: round(df, 3))
    TGI_var_list = ["NMID", "TIME", "SLD"]
    df = df[TGI_var_list]
    (train, test) = split_train_test(df, "NMID", seed=random_seed, train_fraction=train_fraction)
    train.to_csv(f"{root_data_path}/train.csv", index=False)
    test.to_csv(f"{root_data_path}/test.csv", index=False)
    return


def split_train_test(
    df: DataFrame, on_col: str, seed: int, train_fraction: float
) -> Tuple[DataFrame, DataFrame, List, List,]:
    """_summary_

    Parameters
    ----------
    df : DataFrame
        A DataFrame that contains the dataset to be split.
    on_col : str
        A column name that refers to the data component to be used as the splitting criterion.
    seed : int
        An integer that specifies the seed to be used when randomly assigning data to their
        respective cohort.
    train_fraction : float
         A float with range [0, 1] that specificies the proportion of subjects to allocate to the
        training set. (1 - train_fraction) is thus the proportion of subjects allocated to the test
        set.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        A Tuple of DataFrames containing the training and test set data.
    """
    target = df[on_col].unique()
    (train, test) = train_test_split(
        target, random_state=seed, train_size=train_fraction, shuffle=True
    )
    utils.log_message(f'Training NMIDs: {np.sort(train)}')
    utils.log_message(f'Test_NMIDs: {np.sort(test)}')
    train_df = df[df[on_col].isin(train)]
    test_df = df[df[on_col].isin(test)]
    return (train_df, test_df)
