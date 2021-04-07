"""
author:
    cstanbsu

description:
    (1) filter the annotations (successes only) to just 500 cities data
    (2) divide Google API results into multiple datasets for further analysis 
"""


# %% 
# -----------------------------------------------------------------------------------------------
# imports 

import pandas as pd
import numpy as np
import os
import sys
from importlib import reload

# local import
import feature_exploder


# %% 
# -----------------------------------------------------------------------------------------------
# load successfully geocoded Google annotations merge with image
# metadata 

GEOCODED_IMGS = f"../data/image_census_info.csv"
ANNOTATIONS_PATH = f"../data/annotations.csv"

# load geocoded image ids
geo_df = pd.read_csv(GEOCODED_IMGS)
print(geo_df.shape) # note that not all images returned valid geocodes

# load labels and image metadata
annotations_df = pd.read_csv(ANNOTATIONS_PATH)
print(annotations_df.shape)

# merge the datasets on image id and county
df = pd.merge(annotations_df, geo_df, how='left', on=['county', 'image_id'])
print("1", df.shape)

# drop NA geocodes
df = df[df['TRACT'].notna()]
print("2", df.shape)

# add tract ids
df['TRACT'] = df['TRACT'].astype(int).astype(str)

outpath = f"../data/all_valid_geocoded_images.csv"
df.to_csv(outpath, index=False)
df.head()

# %% 
# -----------------------------------------------------------------------------------------------
# load 500 cities data (just the Ann Arbor and Detriot subset)


AA_DT_PATH = f"../data/AA_DT_Tracts.csv"
cities_df = pd.read_csv(AA_DT_PATH)
print(cities_df.shape)

[print(x) for x in cities_df.columns]

# # add a cleaned column with the shortenedd FIPS tract id
# cities_df['TRACT'] = cities_df['TractFIPS'].astype(str).str[-6:] 
# cities_df.head()

# %% 
# -----------------------------------------------------------------------------------------------
# merge the data sets on tract FIPS

# merge
df = pd.merge(df, cities_df, on='TRACT', how='left')
print(df.shape)

# drop ALL IMAGES OUTSIDE ANN ARBOR AND DETRIOT
df = df[df['PlaceName'].notna()]
print(df.shape)
df.reset_index(drop=True, inplace=True)
outpath = f"../data/AA_DT_valid_geocoded_images.csv"
df.to_csv(outpath, index=False)



# %%
# -----------------------------------------------------------------------------------------------
# print column names
# [x for x in df.columns]

# %% 
# -----------------------------------------------------------------------------------------------
# Explode features: this cell depends on the contents of the feature_exploder.py file
# The purpose is to translate nested lists of features from Google API into new columns
# For example, the features are currently in list form per image tile submitted:
# image_id == 0 --> ['house', 'tree']. We seek one-hot encoded columns for each unique
# label or color to facilitate easier analysis.

reload(feature_exploder)

# define the top 10 colors, should there be 10
N_COLORS = 10

test_rec = df.shape[0] # a 'unit test' of sorts

# structure labels
df, label_cols, score_cols = feature_exploder.explode_labels(df)
assert(len(label_cols) == len(score_cols))
print(df.shape)

# structure colors
df, color_cols = feature_exploder.explode_n_colors(df, n=N_COLORS)
assert(df.shape[0] == test_rec)

# %% 
# -----------------------------------------------------------------------------------------------
# divide the datasets into logical tables and save

output_dir = "../prepared_data/"

df['tile_id'] = df['county'].astype(str) + '_' + df['image_id'].astype(str)
df['tile_latitude'] = df['latitiude'] # fix a dumb typo
df['tile_longitude'] = df['longitude']
df['census_tract'] = df['TRACT']
df['city'] = df['PlaceName']

# columns that must be in all dataframes
ID_COLUMNS = [
    'tile_id',
    'image_id',
    'county',
    'city',
    'census_tract',
    'tile_latitude',
    'tile_longitude',
]

LABEL_COLUMNS = label_cols + score_cols
COLOR_COLUMNS = color_cols
GEOCODE_COLUMNS = geo_df.columns.tolist()
CITIES_COLUMNS = cities_df.columns.tolist() 

# save the labels as separate file
outpath = f"{output_dir}google_labels.csv"
col_list = list(set(ID_COLUMNS + LABEL_COLUMNS))
df[col_list].to_csv(outpath, index=False)
print(f"done saving:  {outpath}")

# save the colors as separate file
outpath = f"{output_dir}google_colors.csv"
col_list = list(set(ID_COLUMNS + COLOR_COLUMNS))
df[col_list].to_csv(outpath, index=False)
print(f"done saving:  {outpath}")

# save the geocode metadata as separate file
outpath = f"{output_dir}geocode_info.csv"
col_list = list(set(ID_COLUMNS + GEOCODE_COLUMNS))
df[col_list].to_csv(outpath, index=False)
print(f"done saving:  {outpath}")

# save the 500 cities data as separate file
outpath = f"{output_dir}500_cities.csv"
col_list = list(set(ID_COLUMNS + CITIES_COLUMNS))
df[col_list].to_csv(outpath, index=False)
print(f"done saving:  {outpath}")

# save master
outpath = f"{output_dir}master.csv"
df.to_csv(outpath, index=False)
print(f"done saving:  {outpath}")

# %%ss