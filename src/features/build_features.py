import os

import pandas as pd


def build_features(df, categorical_columns, index_col=None, save_in=None):
    # Read DataFrame
    if isinstance(df, str):
        df_transformed = pd.read_csv(df)

    elif isinstance(df, pd.DataFrame):
        df_transformed = df.copy()

    else:
        raise ValueError("df should be a path file or a pandas DataFrame.")

    # Set Index if not None
    if index_col is not None:
        df_transformed.set_index(index_col, inplace=True)

    if isinstance(categorical_columns, list):
        for cat_column in categorical_columns:
            df_dummies = pd.get_dummies(pd.Categorical(df_transformed[cat_column]), prefix=cat_column)
            df_dummies.set_index(df_transformed.index, inplace=True)
            df_transformed = pd.concat([df_transformed, df_dummies], axis=1)

        try:
            df_transformed = df_transformed.drop(categorical_columns, axis=1)
        except KeyError as k_exp:
            print("Warning:", k_exp)

        # Save dataframe in:
        if save_in is not None:
            directory = os.path.dirname(save_in)
            if not os.path.exists(directory):
                os.makedirs(directory)
            df_transformed.reset_index().to_csv(save_in, index=False)

    else:
        raise ValueError("cat_columns should be a list")

    return df_transformed
