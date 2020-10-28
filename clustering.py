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
positions_df = df[["short_name", "player_positions"]]

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

# %%
# df = df[df.overall > 85] # extracting players with overall above 86
df = df.fillna(df.mean()) # initialize null values with avg

# %%
names = df.short_name.tolist() # saving names for later
df = df.drop(['short_name'], axis = 1) # drop the short_name column

# %%markdown
La normalización es necesaria dado a que no todas las variables utilizan las
mismas métricas y escalas.

# %%
from sklearn import preprocessing

x = df.values
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
X_norm = pd.DataFrame(x_scaled)

# %%markdown
Utilicemos el método del codo para obtener el número de clusters a utilizar

# %%
from sklearn.cluster import KMeans,MeanShift
from sklearn import decomposition
from matplotlib import pyplot as plt

scores_df_over = [KMeans(n_clusters=i+2).fit(df).inertia_ for i in range(10)]

plt.plot(np.arange(2, 12), scores_df_over)
plt.xlabel('Number of clusters')
plt.ylabel("Inertia")
plt.title("Inertia of k-Means versus number of clusters")

# %%markdown
Vemos que 4 clusters es la cantidad óptima de clusters

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2) # 2D PCA for the plot
reduced_kmeans = pd.DataFrame(pca.fit_transform(X_norm))

# %%markdwon
## k-means

# %%
# perform k-means
from sklearn.cluster import KMeans

# specify the number of clusters
kmeans = KMeans(n_clusters=4)

# fit the input data
kmeans = kmeans.fit(reduced_kmeans)

# get the cluster labels
labels_kmeans = kmeans.predict(reduced_kmeans)

# cluster values
clusters_kmeans = kmeans.labels_.tolist()

# %%
# Make a new data frame by adding players' names and their cluster
reduced_kmeans['cluster'] = clusters_kmeans
reduced_kmeans['name'] = names
reduced_kmeans.columns = ['x', 'y', 'cluster', 'name']
reduced_kmeans.head()

# %%
graf_kmeans = go.Scatter(
    x=reduced_kmeans['x'],
    y=reduced_kmeans['y'],
    mode='markers',
    text=reduced_kmeans["name"],
    marker=dict(size=5, color=reduced_kmeans['cluster'], colorscale='Viridis',)
)

layout_kmeans = go.Layout(
    title="Visualización de la base de a dos variables numéricas",
    titlefont=dict(size=20),
    xaxis=dict(title="PC 1"),
    yaxis=dict(title="PC 2"),
    autosize=False, width=1000, height=1000
)

data = [graf_kmeans]

fig = go.Figure(data=data, layout=layout_kmeans)
(fig)

# %%markdown
Veamos qué porcentaje de los diferentes jugadores hay en cada cluster:

# %%
def get_position_per_in_cluster(cluster_df, positions_df, clusters_set):
    position_types = {
        "attackers": {"ST", "LW", "RW", "CF"},
        "midfielders": {"CAM", "LM", "CM", "RM", "CDM"},
        "defenders": {"LWB", "RWB", "LB", "CB", "RB"},
        "goalkeepers": {"GK"},
    }
    cluster_pos_count = {clus: {} for clus in clusters_set}
    position_dict = {
        row["short_name"]: set(row["player_positions"].split(", "))
        for index, row in positions_df.iterrows()
    }
    for index, row in cluster_df.iterrows():
        cluster_pos_counter = cluster_pos_count[row["cluster"]]
        player_position = position_dict[row["name"]]
        get_position_type = lambda pos: next((
            pos_type for pos_type in position_types
            if pos.intersection(position_types[pos_type])),
        pos)

        position_type = get_position_type(player_position)
        cluster_pos_counter[position_type] = cluster_pos_counter.get(
            position_type, 0) + 1

    cluster_pos_percentajes = {}
    for cluster in cluster_pos_count:
        cluster_pos_percentajes[cluster] = {
            pos: round((cluster_pos_count[cluster][pos]/sum(cluster_pos_count[cluster].values())) * 100)
            for pos in cluster_pos_count[cluster]
        }

    return pd.DataFrame.from_records(list(cluster_pos_percentajes.values()))

# %%markdown
Veamos que porcentajes de cada tipo de posición tiene cada cluster:

# %%
get_position_per_in_cluster(reduced_kmeans, positions_df, set(clusters_kmeans))


# %%markdown
## Mean Shift

# %%
reduced_mean_shift = pd.DataFrame(pca.fit_transform(X_norm))

# %%
from sklearn.cluster import MeanShift

mean_shift = MeanShift(
    bandwidth=0.253,
    bin_seeding=True
)

# fit the input data
mean_shift = mean_shift.fit(reduced_mean_shift)

# get the cluster labels
labels_mean_shift = mean_shift.predict(reduced_mean_shift)

# cluster values
clusters_mean_shift = mean_shift.labels_.tolist()

# %%
# Make a new data frame by adding players' names and their cluster
reduced_mean_shift['cluster'] = clusters_mean_shift
reduced_mean_shift['name'] = names
reduced_mean_shift.columns = ['x', 'y', 'cluster', 'name']
reduced_mean_shift.head()

# %%
graf_mean_shift = go.Scatter(
    x=reduced_mean_shift['x'],
    y=reduced_mean_shift['y'],
    mode='markers',
    text=reduced_mean_shift["name"],
    marker=dict(size=5, color=reduced_mean_shift['cluster'], colorscale='Rainbow',)
)

layout_mean_shift = go.Layout(
    title="Visualización de la base de a dos variables numéricas",
    titlefont=dict(size=20),
    xaxis=dict(title="PC 1"),
    yaxis=dict(title="PC 2"),
    autosize=False, width=1000, height=1000
)

data_mean_shift = [graf_mean_shift]

fig_mean_shift = go.Figure(data=data_mean_shift, layout=layout_mean_shift)
(fig_mean_shift)

# %%markdown
Veamos que porcentajes de cada tipo de posición tiene cada cluster:

# %%
get_position_per_in_cluster(reduced_mean_shift, positions_df, set(clusters_mean_shift))

# %%markdown
## Jerárquico

# %%
reduced_hierarchical = pd.DataFrame(pca.fit_transform(X_norm))

# %%
from sklearn.cluster import AgglomerativeClustering

# specify the number of clusters
hierarchical = AgglomerativeClustering(
    n_clusters=4, affinity='l1', linkage='complete')

# fit and get the cluster labels
labels_hierarchical = hierarchical.fit_predict(reduced_hierarchical)

# cluster values
clusters_hierarchical = hierarchical.labels_.tolist()

# %%
# Make a new data frame by adding players' names and their cluster
reduced_hierarchical['cluster'] = clusters_hierarchical
reduced_hierarchical['name'] = names
reduced_hierarchical.columns = ['x', 'y', 'cluster', 'name']
reduced_hierarchical.head()


# %%
graf1_hierarchical = go.Scatter(
    x=reduced_hierarchical['x'],
    y=reduced_hierarchical['y'],
    mode='markers',
    text=reduced_hierarchical["name"],
    marker=dict(size=5, color=reduced_hierarchical['cluster'], colorscale='Rainbow',)
)

layout_hierarchical = go.Layout(
    title="Visualización de la base de a dos variables numéricas",
    titlefont=dict(size=20),
    xaxis=dict(title="PC 1"),
    yaxis=dict(title="PC 2"),
    autosize=False, width=1000, height=1000
)

data_hierarchical = [graf1_hierarchical]

fig_hierarchical = go.Figure(data=data_hierarchical, layout=layout_hierarchical)
(fig_hierarchical)

# %%markdown
Veamos que porcentajes de cada tipo de posición tiene cada cluster:

# %%
get_position_per_in_cluster(reduced_hierarchical, positions_df, set(clusters_hierarchical))
