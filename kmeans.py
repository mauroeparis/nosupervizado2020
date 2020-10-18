# %%
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',1000)
import itertools
import warnings
import io

from plotly.offline import init_notebook_mode, plot,iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
import seaborn as sns

# %%
df = pd.read_csv("./players_20.csv")

# %%
df = df[
    [
        'short_name',
        'height_cm', 'weight_kg',
        'overall', 'potential',
        'pace', 'shooting', 'passing', 'dribbling',
        'defending', 'physic',
        'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed',
        'gk_positioning',
        'attacking_crossing', 'attacking_finishing',
        'attacking_heading_accuracy', 'attacking_short_passing',
        'attacking_volleys',
        'skill_moves', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy',
        'skill_long_passing', 'skill_ball_control',
        'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
        'movement_reactions', 'movement_balance',
        'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
        'power_long_shots',
        'mentality_aggression', 'mentality_interceptions',
        'mentality_positioning', 'mentality_vision', 'mentality_penalties',
        'mentality_composure',
        'defending_marking', 'defending_standing_tackle',
        'defending_sliding_tackle',
        'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
        'goalkeeping_positioning', 'goalkeeping_reflexes'
    ]
]

df.head()

# %%
df = df[df.overall > 85] # extracting players with overall above 86
df = df.fillna(df.mean()) # initialize null values with avg

# %%
names = df.short_name.tolist() # saving names for later
df = df.drop(['short_name'], axis = 1) # drop the short_name column

# %%
# normalize, investiate why
from sklearn import preprocessing

x = df.values # numpy array
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
X_norm = pd.DataFrame(x_scaled)

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components = 2) # 2D PCA for the plot
reduced = pd.DataFrame(pca.fit_transform(X_norm))

# %%
# perform k-means
from sklearn.cluster import KMeans

# specify the number of clusters
kmeans = KMeans(n_clusters=5)

# fit the input data
kmeans = kmeans.fit(reduced)

# get the cluster labels
labels = kmeans.predict(reduced)

# centroid values
centroid = kmeans.cluster_centers_

# cluster values
clusters = kmeans.labels_.tolist()

# %%
# Make a new data frame by adding players' names and their cluster
reduced['cluster'] = clusters
reduced['name'] = names
reduced.columns = ['x', 'y', 'cluster', 'name']
reduced.head()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set(style="white")

ax = sns.lmplot(x="x", y="y", hue='cluster', data = reduced, legend=False,
                   fit_reg=False, height=15, scatter_kws={"s": 250})

texts = []
for x, y, s in zip(reduced.x, reduced.y, reduced.name):
    texts.append(plt.text(x, y, s))

ax.set(ylim=(-2, 2))
plt.tick_params(labelsize=15)
plt.xlabel("PC 1", fontsize = 20)
plt.ylabel("PC 2", fontsize = 20)

plt.show()
