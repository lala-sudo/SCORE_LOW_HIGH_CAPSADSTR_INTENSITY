import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt


def calculate_p_values(df, columns=None):
    if columns and isinstance(columns, list):
        df = df[columns].copy()

    df_cols = pd.DataFrame(columns=df.columns)
    p_values = df_cols.transpose().join(df_cols, how='outer')
    df_columns = df.columns

    for i, r in enumerate(df_columns):
        for j, c in enumerate(df_columns):
            tmp = df[df[r].notnull() & df[c].notnull()]
            pearson_ = pearsonr(tmp[r], tmp[c])
            p_values[r][c] = round(pearson_[1], 5)

    return p_values


def get_correlation_matrix(df_corr, p_values, save_in):
    plt.figure(figsize=(68, 54))
    sns.set(font_scale=5)

    # define the mask to set the values in the upper triangle to True
    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
    heatmap = sns.heatmap(df_corr,
                          linewidth=0,
                          mask=mask,
                          vmin=-1,
                          vmax=1,
                          annot=p_values,
                          cmap="rocket",
                          annot_kws={"size": 30, "color": "white"}
                          )

    heatmap.set_title("Pearson's Correlation with P-value between numerical features",
                      fontdict={'fontsize': 28},
                      pad=24)

    for i in range(0, len(heatmap.get_xticklabels())):
        heatmap.get_xticklabels()[i]._text = heatmap.get_xticklabels()[i]._text.replace('e_T', 'e\nT')

    for i in range(0, len(heatmap.get_yticklabels())):
        heatmap.get_yticklabels()[i]._text = heatmap.get_yticklabels()[i]._text.replace('e_T', 'e\nT')

    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

    return heatmap


def analyze_data(df, categorical_columns, index_col=None, save_in=None):
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
        try:
            cat_cols_with_suffix = []
            for cat_col in categorical_columns:
                for col in df_transformed.columns:
                    if cat_col in col:
                        cat_cols_with_suffix.append(col)

            df_transformed = df_transformed.drop(columns=list(np.unique(cat_cols_with_suffix)))
        except KeyError as k_exp:
            print("Warning:", k_exp)

        # Get Matrix
        df_corr = df_transformed.corr()
        p_values = calculate_p_values(df_transformed)
        matrix = get_correlation_matrix(df_corr, p_values, save_in)

        # Save Correlation Matrix
        if save_in is not None:
            if not os.path.exists(save_in):
                os.makedirs(save_in)
            fig = matrix.get_figure()
            fig.savefig(os.path.join(save_in, "correlation_matrix.jpg"))
            plt.clf()

    else:
        raise ValueError("cat_columns should be a list")

    return True
