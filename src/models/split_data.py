import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def split_data(df, target_column_name, number_folds, index_col=None, test_size=0.2, random_seed=0, save_in=None):
    np.random.seed(random_seed)

    data_ready_to_model = {}

    # Read DataFrame
    if isinstance(df, str):
        df = pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        df = df.copy()
    else:
        raise ValueError("df should be a path file or a pandas DataFrame")

    # Set Index if not None
    if index_col is not None:
        df.set_index(index_col, inplace=True)

    # Split Features and Target
    y = df[target_column_name].copy()
    X = df.drop([target_column_name], axis=1).copy()

    # Split Train and Test
    if isinstance(number_folds, int):
        # 2 folds
        if number_folds <= 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
            data_ready_to_model[0] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

        # K-fold (more than 2)
        else:
            kf = KFold(n_splits=number_folds,
                       shuffle=True,
                       random_state=random_seed)

            for i, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                data_ready_to_model[i] = {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }

    if save_in is not None:
        directory = os.path.dirname(save_in)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(save_in, 'wb') as f:
            pickle.dump(data_ready_to_model, f)

    return data_ready_to_model
