

# %%
# ------------------------------------------
# imports
from os import replace
import numpy as np
import pandas as pd
from pandas.tseries.offsets import YearOffset
from seaborn.miscplot import palplot
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.graphics.api import abline_plot

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
matplotlib.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-deep')
sns.set()
from matplotlib import rcParams

import utils

%matplotlib inline

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

city_df.head()

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

# add binary dummy coding for ann arbor and dt
df['CITY_BINARY'] = np.where(df['city'] == 'Ann Arbor', 1, -1)

df.head()

# %%
# ------------------------------------------
# 

print(df['census_tract'].nunique())

aa = df[df['city'] == 'Ann Arbor']
dt = df[df['city'] == 'Detroit']

print(aa['census_tract'].nunique())
print(dt['census_tract'].nunique())


# %%
# ------------------------------------------
# Distributions of the columns of interest

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['figure.figsize'] = 8, 8

ANALYSIS_COLUMNS = {
 'BPHIGH_CrudePrev' : 'High BP',
 'BPMED_CrudePrev' : 'High BP Medicated',
 'CHD_CrudePrev' : 'Coronary Heart Disease',
 'CHOLSCREEN_CrudePrev' : 'High Cholesterol',
 'DIABETES_CrudePrev' : 'Diabetes',
 'OBESITY_CrudePrev' : 'Obesity',
 'PHLTH_CrudePrev' : 'Physical Health',
 'STROKE_CrudePrev' : 'Stroke',
}

for COLUMN, label in ANALYSIS_COLUMNS.items():
    plt.cla()
    OUTPUT_DIR = f"../figures/"
    print(f'Working: {label}')

    aa = df[df['city'] == 'Ann Arbor']

    # remove to reduce skew
    aa = aa.drop_duplicates(subset='census_tract')

    dt = df[df['city'] == 'Detroit']
    
    # remove to reduce skew
    dt = dt.drop_duplicates(subset='census_tract')

    TITLE = f'Distribution of {label} in Ann Arbor vs. Detriot'

    sns.kdeplot(aa[COLUMN], 
                shade=True, 
                color="C0",
                label='Ann Arbor')

    sns.kdeplot(dt[COLUMN], 
                shade=True, 
                color="C3",
                label='Detriot')

    plt.suptitle(TITLE,  fontsize=18)
    plt.legend()
    plt.xlabel(label)
    save_filepath = f"{OUTPUT_DIR}{TITLE}.png"
    plt.savefig(save_filepath,  bbox_inches = 'tight')


# %%
# ------------------------------------------
# exclude lables that only occur in one city

EXCLUSIONS = []

for col in score_cols:
    t = df.groupby(['city'], as_index=False)[col].max()
    if t[col].min() == 0:
        EXCLUSIONS.append(col)

INCLUSIONS = [x for x in score_cols if not x in EXCLUSIONS]
print(len(score_cols))
print(len(INCLUSIONS))

# %%
# ------------------------------------------
# PCA on the score columns - use .80 explained variance as threshold

N_COMPONENTS = 12 # note that 12 is where the explained variance ratio reaches ~.80
# N_COMPONENTS = 11 # 11 for exclusions
pca = PCA(n_components=N_COMPONENTS, svd_solver='full')

X = pca.fit_transform(df[INCLUSIONS])
# X_excl = pca.fit_transform(df[EXCLUSIONS])
print(np.cumsum(pca.explained_variance_ratio_))

pca_columns = []

for i in range(N_COMPONENTS):
    new_label = f"PCA_{i+1}"
    pca_columns.append(new_label)
    df[new_label] = X[:, i]

    # df[new_label] = X_excl[:, i]




# %%
# ------------------------------------------
# run binomial regression models

models = {}

for COLUMN, label in ANALYSIS_COLUMNS.items():
    print()

    # structure dependent to [0, 1] range
    Y = df[COLUMN] / 100

    """
    FEATURE SETS:
    (1) city categorical only
    (2) PCA variables (0.80 rule) and city categorical
    (3) all inclusion columns and city categorical  
    (3) all exlcuded columns and city categorical  
    """

    PCA_VARS = [x for x in df.columns if 'PCA' in x]

    # IND_VARS = ['CITY_BINARY'] # (1)
    IND_VARS = ['CITY_BINARY'] + PCA_VARS # (2)
    # IND_VARS = ['CITY_BINARY'] + INCLUSIONS  # (3)
    # IND_VARS = ['CITY_BINARY'] + EXCLUSIONS  # (4)
    X = df[IND_VARS]

    """
    GLM Binomial Link: we estimate the intercept first
    """
    X = sm.add_constant(X)
    glm_binom = sm.GLM(Y, X, family=sm.families.Binomial())
    results = glm_binom.fit()

    # print results, graph the fit, store the results
    print(results.summary())
    models[label] = results
    print()


    # plot the fit
    matplotlib.rcParams['figure.dpi'] = 300
    matplotlib.rcParams['figure.figsize'] = 5,5

    fig = utils.build_fig()

    y = Y/Y.sum()
    yhat = results.mu
    sns.scatterplot(yhat, 
                    Y, 
                    hue=df['city'], 
                    alpha=0.5, 
                    palette='Set1', 
                    edgecolor='black')

    line_fit = sm.OLS(Y, sm.add_constant(yhat)).fit()
    abline_plot(model_results=line_fit, ax=plt.gca(), c='red')
    plt.title(f'Model Fit Plot for {label}')
    plt.ylabel(f'Observed {label} values')
    plt.xlabel(f'Fitted {label} values')

    outpath = f"../figures/{label} fit.png"
    plt.savefig(outpath, bbox_inches='tight')



# %%
# ------------------------------------------
# simple viz

term = 'Green_score'

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['figure.figsize'] = 8, 6

sns.scatterplot(x=df['tile_longitude'],
                y=df['tile_latitude'], 
                hue=df[term],
                s=50,
                palette='coolwarm',
                alpha=0.5)


# %%

t = np.random.choice(INCLUSIONS, 3, replace=False)
t


# %%
