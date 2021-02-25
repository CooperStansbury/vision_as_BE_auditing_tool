"""
author:
    cstanbsu

description:
    (1) restructure image features to single file
    (2) clean and reduce 500 citiies dataset 
"""


# %% 
# -----------------------------------------------------------------------------------------------
# imports 

import pandas as pd
import numpy as np
import os
import sys

# %%
# -----------------------------------------------------------------------------------------------
# filter Ann Arbor and Detriot into a smaller data_file, 
# takes as input the 500 cities (2016 release) dataset as a .csv file

FILE_PATH = '../data/500_Cities__City-level_Data__GIS_Friendly_Format___2016_release.csv'
OUTPUT_DIR = '../data/'
cities_df = pd.read_csv(FILE_PATH)

cities_to_analyze = ['Detroit', 'Ann Arbor']
cities_df = cities_df[cities_df['PlaceName'].isin(cities_to_analyze)]
print(cities_df.shape)

save_path = f"{OUTPUT_DIR}AA_DT_Tracts.csv"
cities_df.to_csv(save_path, index=False)


# %%
# -----------------------------------------------------------------------------------------------
# concatenate all image annotations to a single dataframe

COUNTY_TAGS = ['Washtenaw', 'Wayne']
OUTPUT_DIR = '../data/'
ROOT_DIR = '../data/'

df_list = []

for file in os.listdir(ROOT_DIR):

    # filter out files not related to annotations
    for tag in COUNTY_TAGS:
        if file.__contains__(tag):

            file_path = f"{ROOT_DIR}{file}"
            
            tmp_df = pd.read_csv(file_path)
            df_list.append(tmp_df)

annotations_df = pd.concat(df_list)
print(annotations_df.shape)

save_path = f"{OUTPUT_DIR}image_labels.csv"
annotations_df.to_csv(save_path, index=False)

# %%
