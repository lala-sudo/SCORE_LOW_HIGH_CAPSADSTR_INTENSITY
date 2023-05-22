import os
from os.path import splitext

import pandas as pd


def make_dataset(input_path_file, index_col=None, columns_to_not_consider=None,
                 drop_all_rows_with_na=False, filter_by=None,
                 save_in=None):
    # Read data
    file_name, extension = splitext(input_path_file)
    if ".xlsx" in extension:
        df = pd.read_excel(input_path_file, engine='openpyxl')
    elif ".csv" in extension:
        df = pd.read_csv(input_path_file)
    else:
        raise ValueError("Extension Not Supported.")

    # Set Index if not None
    if index_col is not None:
        df.set_index(index_col, inplace=True)

    if filter_by is not None:
        filter_str = ''
        for i, (key, value) in enumerate(filter_by.items()):
            if value != "*":
                if i < len(filter_by) - 1:
                    filter_str = filter_str + key + '==' + '"' + value + '"' + ' & '
                elif i == len(filter_by) - 1:
                    filter_str = filter_str + key + '==' + '"' + str(value) + '"'
            else:
                print(f"Select all values for {key}")
        if filter_str != '':
            df = df.query(filter_str)

    # Consider Only those columns
    if columns_to_not_consider is not None:
        if isinstance(columns_to_not_consider, list):
            df = df[list(set(df.columns) - set(columns_to_not_consider))]
        else:
            raise ValueError("columns_to_not_consider should be a List.")

    # Drop those rows
    if drop_all_rows_with_na is True:
        df.drop(df[df.isnull().any(1)].index,
                axis=0, inplace=True)

    # Save dataframe in:
    if save_in is not None:
        directory = os.path.dirname(save_in)
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.reset_index().to_csv(save_in, index=False)

    return df.copy()
