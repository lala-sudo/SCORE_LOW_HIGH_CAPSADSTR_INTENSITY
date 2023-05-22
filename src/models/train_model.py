import os
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(real, predicted):
    return 100 / len(real) * np.sum(2 * np.abs(predicted - real) / (np.abs(real) + np.abs(predicted)))


def evaluate(split_data, best_grid):
    metrics = {}

    for i_fold in split_data.keys():
        y_true = split_data[i_fold]['y_test']
        y_pred = best_grid.predict(split_data[i_fold]['X_test'])

        y_true_train = split_data[i_fold]['y_train']
        y_pred_train = best_grid.predict(split_data[i_fold]['X_train'])

        mse_train = mean_squared_error(y_true_train, y_pred_train)
        smape_train = smape(y_true_train, y_pred_train)
        mape_train = mape(y_true_train, y_pred_train)

        mse_ = mean_squared_error(y_true, y_pred)
        smape_ = smape(y_true, y_pred)
        mape_ = mape(y_true, y_pred)

        metrics[i_fold] = {
            "MSE": mse_, "SMAPE": smape_, "MAPE": mape_,
            "MSE-train": mse_train, "SMAPE-train": smape_train, "MAPE-train": mape_train
        }

    return metrics


def get_best_model(data_split_path_file, param_grid, random_seed=0, save_in=None):
    np.random.seed(random_seed)

    # Read Split Data
    with open(data_split_path_file, 'rb') as f:
        split_data = pickle.load(f)

    # Get X and y
    X = pd.concat([split_data[i]['X_train'] for i in split_data.keys()])
    X = X[~X.index.duplicated(keep='first')]

    y = pd.concat([split_data[i]['y_train'] for i in split_data.keys()])
    y = y[~y.index.duplicated(keep='first')]

    map_x_index = {i: j for i, j in zip(X.index, list(X.reset_index().index))}

    cv = [(np.array([map_x_index[j] for j in split_data[i]['X_train'].index]),
           np.array([map_x_index[j] for j in split_data[i]['X_test'].index]))
          for i in split_data.keys()]

    # Create a based model
    rf = RandomForestRegressor(random_state=random_seed)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               cv=cv,
                               n_jobs=-1,
                               verbose=0)

    grid_search.fit(X, y)
    print("Best Parameters are:", grid_search.best_params_)

    if save_in is not None:
        directory = os.path.dirname(save_in)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(save_in, 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)

        with open(os.path.join(directory, "X.pkl"), 'wb') as f:
            pickle.dump(X, f)

        with open(os.path.join(directory, "y.pkl"), 'wb') as f:
            pickle.dump(y, f)

    return grid_search, split_data, X, y
