"""
author:
    cstanbsu

description:
    map annotations to census tracts 
"""


# %% 
# -----------------------------------------------------------------------------------------------
# imports 

from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np
import os
import sys
import re
import time
import censusgeocode
import requests
from importlib import reload

# %%
# -----------------------------------------------------------------------------------------------
# load the image files and map the centriod coordinates back to the 
# images. combine coordinate maps

OUTPUT_DIR = '../data/'

def apply_make_id(row):
    f_name = row['file'].split("/")[-1]
    f_name = f_name.replace(".png", "")
    f_name_splt = f_name.split("_")
    chunk = (int(f_name_splt[1]) - 1) * 1000
    img = int(f_name_splt[3])
    image_id = str(chunk + img)
    return image_id


# load annotations and build image ids
label_df = pd.read_csv('../data/image_labels.csv')
label_df['image_id'] = label_df.apply(lambda row: apply_make_id(row), axis=1)
print(label_df.shape)

# load coordinate maps
washtenaw_map_df = pd.read_csv('../data/Wash_coordinate_map.csv')
wayne_map_df = pd.read_csv('../data/Way_coordinate_map.csv')

# build image ids for coordinate maps
washtenaw_map_df['image_id'] = washtenaw_map_df['image_index'].astype(str)
wayne_map_df['image_id'] = wayne_map_df['image_index'].astype(str)

# combine coordinate maps for both counties
coords = pd.concat([washtenaw_map_df, wayne_map_df])
print(coords.shape)

# merge the coordinates with the labels (annotations)
df = pd.merge(label_df, coords, how='left', on=['county', 'image_id'])
print(df.shape)

# save the result
save_path = f"{OUTPUT_DIR}annotations.csv"
df.to_csv(save_path, index=False)


# %%
# -----------------------------------------------------------------------------------------------
# add census tracts to the annotated images using the Census API

OUTPUT_DIR = '../data/'
BENCHMARK = 'Public_AR_Current'
VINTAGE = 'ACS2017_Current'
PRINT_EVERY = 200 # logging for sanity 

# set up coordinate converter
cg = censusgeocode.CensusGeocode(benchmark=BENCHMARK, 
                                 vintage=VINTAGE)

new_rows = []
errors = []

for idx, row in df.iterrows():
    if idx % PRINT_EVERY == 0:
        print(f"idx {idx} {idx}-{idx + PRINT_EVERY}...")

    # catch unsuccessful API calls
    try:
        result = cg.coordinates(x=row['longitude'], 
                                y=row['latitiude'],
                                timeout=60)

        # handle successful calls, but empty results
        if len(result['Census Tracts']) < 1:
            raise ValueError(f"NO RESULTS FOR IMAGE_ID: {row['image_id']}")

        # store the succesful results 
        else:
            cenrow = result['Census Tracts'][0]
            cenrow['image_id'] = row['image_id']
            cenrow['county'] = row['county']
            new_rows.append(cenrow)

    except Exception as e:
        errors.append(row['image_id'])
        print(f"FAILED at {idx} img {row['image_id']}: {e} added to recall list")

# convert sucesses to tabular format and save
census_df = pd.DataFrame(new_rows)
print(census_df.shape)
census_df.head()
print(len(errors))

save_path = f"{OUTPUT_DIR}image_census_info.csv"
census_df.to_csv(save_path, index=False)


# %%
# -----------------------------------------------------------------------------------------------
# save the records that didn't have results or timed out


recalls = df[df['image_id'].isin(errors)]
print(recalls.shape)
save_path = f"{OUTPUT_DIR}image_census_info_RECALLS.csv"
recalls.to_csv(save_path, index=False)


# %%
# -----------------------------------------------------------------------------------------------
# sanity checks

recalls['county'].value_counts()


# %%
# -----------------------------------------------------------------------------------------------
# sanity checks

tmp = recalls.sample(5)

for idx, row in tmp.iterrows():
    print(idx, row['latitiude'], row['longitude'])


"""
NOTES: after several checks, the vast majority of the images with issues come 
border areas, i.e., Windsor or the edge of Wayne/Washtenaw counties. This means we
will NOT recall these image_id against the API.
"""


# %%
# -----------------------------------------------------------------------------------------------
# This is what the API results look like for reference

ERROR_DICT = {
    'GEOID': None, 
    'CENTLAT': None, 
    'AREAWATER': None, 
    'STATE': None, 
    'BASENAME': None, 
    'OID': None, 
    'LSADC': None, 
    'FUNCSTAT': None, 
    'INTPTLAT': None, 
    'NAME': None, 
    'OBJECTID': None, 
    'TRACT': None, 
    'CENTLON': None, 
    'AREALAND': None, 
    'INTPTLON': None, 
    'MTFCC': None, 
    'COUNTY': None, 
    'CENT': None, 
    'INTPT': None
}


