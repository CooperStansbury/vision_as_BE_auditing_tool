

# %%
# ------------------------------------------
# imports
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# %%
# ------------------------------------------
# load label scores for all images
LABEL_PATH = f"../prepared_data/google_labels.csv"
label_df = pd.read_csv(LABEL_PATH)
print(f"raw labels: {label_df.shape}")

score_cols = [x for x in label_df.columns if '_score' in x]
ids_cols = [
    'census_tract',
    'city',
    'county',
    'image_id',
    'tile_id',
    'tile_latitude',
    'tile_longitude'
]

label_df = label_df[score_cols + ids_cols]
print(f"subset labels: {label_df.shape}\n")


# %%
# ------------------------------------------
# load 500 cities data
CITIES_PATH = "../prepared_data/500_cities.csv"
city_df = pd.read_csv(CITIES_PATH)
print(f"raw city: {city_df.shape}")

cols_of_interest = [
 'BPHIGH_CrudePrev',
 'BPMED_CrudePrev',
 'CHD_CrudePrev',
 'CHOLSCREEN_CrudePrev',
 'DIABETES_CrudePrev',
 'OBESITY_CrudePrev',
 'PHLTH_CrudePrev',
 'Population2010',
 'STROKE_CrudePrev',
]

city_df = city_df[cols_of_interest + ['tile_id']]
print(f"subset labels: {city_df.shape}\n")


# %%
# ------------------------------------------
# merge the data on the image

df = pd.merge(label_df, 
              city_df, 
              how='left',
               on='tile_id')
print(df.shape)
df.head()

# %%
# ------------------------------------------
# train test split

train, test = train_test_split(df, test_size=0.33, random_state=42)
print(f"train: {train.shape}")
print(f"test: {test.shape}")


# %%
# ------------------------------------------
# PCA on the score columns (trainning only)

N_COMPONENTS = 10
pca = PCA(n_components=N_COMPONENTS, svd_solver='full')
X = pca.fit_transform(train[score_cols])
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))

# testing uses the trainned PCA to re-project the remainning images 
projection = pca.transform(test[score_cols])


print(X.shape)
print(projection.shape)

pca_columns = []

for i in range(N_COMPONENTS):
    new_label = f"PCA_{i+1}"
    pca_columns.append(new_label)
    train[new_label] = X[:, i]
    test[new_label] = projection[:, i]

# %%


def run_regression(Y, X, y_ts, x_ts):
    """A functition to 
    """
