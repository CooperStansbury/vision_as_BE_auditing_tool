"""
author:
    cstanbsu

description:
    a set of functions used to clean the structure of the google 
    features and align them with census/geocoded datasets
"""


import pandas as pd
import numpy as np
import ast
import sys
from sklearn.preprocessing import MultiLabelBinarizer


# ------------------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------------------


def apply_unlist(row):
    """A lambda apply paradigm function to convert 
    str objects into python lists
    """
    return ast.literal_eval(row)


def apply_make_map(row, label_col='labels', score_col='label_scores'):
    """A lambda apply paradigm function create a dictionary
    for labels and label scores. 
    """
    return dict(zip(row[label_col], row[score_col]))


def explode_labels(df, label_col='labels', score_col='label_scores'):
    """A function to convert labels into new columns.
    Labels will be one-hot encoded and corresponding confidence scores
    will be added such that every entry also has a weighted one-hot
    confidence entry.

    Ex.
        [label_1, ..., label_n] & [label_1_score, ..., label_n_score]

    Args:
        - df (pd.dataframe): input dataframe
        - label_col (str): column name of labels
        - score_col (str): column name of label scores

    Returns:
        - df (pd.dataframe): input dataframe with new columns. no columns
            are dropped.
        - LABEL_COLS (list of str): list of new col names for labels
        - SCORE_COLS (list of str): list of new col names for scores
    """
    # convert strings to lists
    df[label_col] = df[label_col].apply(lambda row: apply_unlist(row))
    df[score_col] = df[score_col].apply(lambda row: apply_unlist(row))

    # make a lookup table for each entry
    df['SCORE_MAP'] = df.apply(lambda row: apply_make_map(row), axis=1)

    # one-hot encode labels
    mlb = MultiLabelBinarizer()
    label_df = pd.DataFrame(mlb.fit_transform(df.pop(label_col)),
                                columns=mlb.classes_,
                                index=df.index)
    
    LABELS = label_df.columns.tolist()

    LABEL_COLS = [f"{x.replace(' ', '_')}_label" for x in LABELS]
    SCORE_COLS = [f"{x.replace(' ', '_')}_score" for x in LABELS]

    label_df.columns = LABELS
    scores_df = label_df.copy()

    # iterate through each value and replace
    # positive binary values with the actual 
    # score given by google. Uses the `SCORE_MAP`
    for idx, row in scores_df.iterrows():
        for col in LABELS:
            if row[col] == 1:
                score = df.iloc[idx]['SCORE_MAP'][col]
                scores_df.loc[(scores_df.index == idx), col] = score

    # rename columns
    scores_df.columns = SCORE_COLS
    label_df.columns = LABEL_COLS
    
    # distinguish between no value and low score
    scores_df[SCORE_COLS].replace({0:np.nan})

    df = df.join(label_df)
    df = df.join(scores_df)

    return df, LABEL_COLS, SCORE_COLS


def explode_n_colors(df, n=10, color_col='colors', 
                     fraction_cols='color_pixel_fraction', 
                     color_scores='color_scores'):
    """A function to structure the first N colors in wide format.


    NOTE: the top `n` is determined by score, not by pixel coverage
    
    Args:
        - df (pd.dataframe): input dataframe
        - n (int): the number of colors varies tremendously, 
            we take only the top `n` most prevelent colors and 
            transform them into new columns
        - color_col (str): column name of colors (RGB)
        - fraction_cols (str): column name of color fractions (pixel percent)
        - color_scores (str): column name of color scores

    Returns:
        - df (pd.dataframe): input dataframe with new columns. no columns
            are dropped.
        - COLOR_COLS (list of str): list of new col names for colors, fractions
            and scores
    """
    # convert strings to lists
    df[color_col] = df[color_col].apply(lambda row: apply_unlist(row))
    df[fraction_cols] = df[fraction_cols].apply(lambda row: apply_unlist(row))
    df[color_scores] = df[color_scores].apply(lambda row: apply_unlist(row))

    COLOR_COLS = [f"COLOR_{x}" for x in range(1, n+1)]

    # build a copyable template for new dataframe
    # construction
    row_template = {}
    for color in COLOR_COLS:
        row_template[f'{color}_R'] = np.nan
        row_template[f'{color}_G'] = np.nan
        row_template[f'{color}_B'] = np.nan
        row_template[f'{color}_fraction'] = np.nan
        row_template[f'{color}_score'] = np.nan

    # list for slightly more efficient dataframe construction
    new_rows = []

    for idx, row in df.iterrows():
        new_row = row_template.copy()
        for i, val in enumerate(row[color_col]):
            new_row[f"COLOR_{i+1}_R"] = val[0]
            new_row[f"COLOR_{i+1}_G"] = val[1]
            new_row[f"COLOR_{i+1}_B"] = val[2]

        for i, val in enumerate(row[fraction_cols]):
            new_row[f"COLOR_{i+1}_fraction"] = val

        for i, val in enumerate(row[color_scores]):
            new_row[f"COLOR_{i+1}_fraction"] = val

        new_rows.append(new_row)

    # build new dataframe
    color_df = pd.DataFrame(new_rows)
    COLOR_COLS = color_df.columns.tolist()
    df = df.join(color_df)
    return df, COLOR_COLS