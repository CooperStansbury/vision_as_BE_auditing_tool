"""
author:
    cstansbu

description:
    A notebook to visualze the google labels
"""

# %%
# ---------------------------------------------
# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

# local import
import utils

plt.switch_backend('agg')
matplotlib.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-deep')
sns.set()
from matplotlib import rcParams

# %%
# ---------------------------------------------
%matplotlib inline

# %%
# ---------------------------------------------
# load data

LABEL_PATH = f"../prepared_data/google_labels.csv"
df = pd.read_csv(LABEL_PATH)
print(df.shape)
labels = [x for x in df.columns if '_label' in x]
print(f"n labels: {len(labels)}")
df.head()


# %%
# sorted(labels)


# %%
# ---------------------------------------------
# lat/lon plots
# matplotlib.rcParams['figure.figsize'] = (12, 10)

# label = 'Metropolitan_area_score'

# # build the figure
# TITLE = f'Label `{label}` by Lat/Lon'
# COLORMAP = 'seismic'

# FAC = 0.001
# x = utils.rand_jitter(df['tile_longitude'], factor=FAC)
# y = utils.rand_jitter(df['tile_latitude'], factor=FAC)

# scat = sns.scatterplot(x,
#                        y, 
#                        s=80, 
#                        hue=df[label],
#                        alpha=0.5,
#                        marker="s",
#                        palette=COLORMAP)


# norm = plt.Normalize(df[label].min(), df[label].max())
# sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)

# scat.figure.colorbar(sm)
# scat.get_legend().remove()
# plt.title(f"{TITLE}")

# save_filepath = f"../figures/{TITLE}.png"
# plt.savefig(save_filepath,  bbox_inches = 'tight')

# %%
# ---------------------------------------------
# PCA of all scores

label = 'Metropolitan_area_score'
matplotlib.rcParams['figure.figsize'] = (8, 8)
N_COMPONENTS = 2
COLORMAP = 'coolwarm'
TITLE = f"PCA by {label}"


score_cols = [x for x in df.columns if '_score' in x]
pca = PCA(n_components=N_COMPONENTS, svd_solver='full')
X = pca.fit_transform(df[score_cols])

scat = sns.scatterplot(x=X[:, 0],
                       y=X[:, 1], 
                       s=100, 
                       hue = df[label],
                       alpha=0.6,
                       palette=COLORMAP)

norm = plt.Normalize(df[label].min(), df[label].max())
sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)

scat.figure.colorbar(sm)
scat.get_legend().remove()
plt.title(f"{TITLE}")


# %%
