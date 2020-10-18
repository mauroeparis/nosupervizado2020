# %%
import numpy as np
import pandas as pd
pd.set_option("display.max_columns",100)
pd.set_option("display.max_rows",1000)
import itertools
import warnings
import io

from plotly.offline import init_notebook_mode, plot,iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import plotly.tools as tls #visualization
import plotly.figure_factory as ff #visualization
import seaborn as sns

# %%
df = pd.read_csv("./players_20.csv")
list(df.columns)

# %%
# Dividamos los puntajes en categorias:
attacking_stats = [
    'attacking_finishing', 'attacking_crossing',
    'attacking_heading_accuracy', 'attacking_short_passing',
    'attacking_volleys'
]
skill_stats = [
    'skill_moves', 'skill_dribbling', 'skill_curve',
    'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
]
movement_stats = [
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
    'movement_reactions', 'movement_balance',
]
power_stats = [
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
    'power_long_shots',
]
mentality_stats = [
    'mentality_aggression', 'mentality_interceptions',
    'mentality_positioning', 'mentality_vision', 'mentality_penalties',
    'mentality_composure',
]
defending_stats = [
    'defending_marking', 'defending_standing_tackle',
    'defending_sliding_tackle',
]

# %%markdown
Veamos quienes son los mejores y peores en cada categoría

# %%
get_best_players = lambda stats: list(
    df[['short_name', *stats]].sort_values(
        stats, ascending=False
    )['short_name'].head(5)
)

best_data = {
    "best_attacking": get_best_players(attacking_stats),
    "best_skill": get_best_players(skill_stats),
    "best_movement": get_best_players(movement_stats),
    "best_power": get_best_players(power_stats),
    "best_mentality": get_best_players(mentality_stats),
    "best_defending": get_best_players(defending_stats),
}

best_df = pd.DataFrame(best_data, columns=list(best_data.keys()))
best_df

# %%
get_worst_players = lambda stats: list(
    df[['short_name', *stats]].sort_values(
        stats, ascending=True
    )['short_name'].head(5)
)

worst_data = {
    "worst_attacking": get_worst_players(attacking_stats),
    "worst_skill": get_worst_players(skill_stats),
    "worst_movement": get_worst_players(movement_stats),
    "worst_power": get_worst_players(power_stats),
    "worst_mentality": get_worst_players(mentality_stats),
    "worst_defending": get_worst_players(defending_stats),
}

worst_df = pd.DataFrame(worst_data, columns=list(worst_data.keys()))
worst_df

# %%markdown
Bajo los datos anteriores podemos suponer que los mejores atacando son los delanteros
y lo peores en movilidad son los arqueros dado a que los 5 mejores y peores de esas
categorías estan formados solo por esas posiciones respectivamente.

# %%
short_df = df[
    [
        'short_name',

        'overall', 'potential',

        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',

        'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed',
        'gk_positioning',

        *attacking_stats,

        *skill_stats,

        *movement_stats,

        *power_stats,

        *mentality_stats,

        *defending_stats,

        'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
        'goalkeeping_positioning', 'goalkeeping_reflexes'
    ]
]
short_df = short_df.fillna(short_df.mean())

# %%
skill_1 = "movement_acceleration"
skill_2 = "attacking_finishing"
bool_crack = short_df["overall"] > 85

graf1 = go.Scatter(
    x=short_df[skill_1],
    y=short_df[skill_2],
    mode='markers',
    text=short_df["short_name"],
    marker=dict(size=5)
)

crack = go.Scatter(
    x=short_df[bool_crack][skill_1],
    y=short_df[bool_crack][skill_2],
    name='Top players',
    text=short_df[bool_crack]['short_name'],
    textfont=dict(family='sans serif', size=14, color='black'),
    opacity=0.9, mode='text'
)

data = [graf1, crack]

layout = go.Layout(
    title="Visualización de la base de a dos variables numéricas",
    titlefont=dict(size=20),
    xaxis=dict(title=skill_1),
    yaxis=dict(title=skill_2),
    autosize=False, width=1000, height=1000
)

fig = go.Figure(data=data, layout=layout)
(fig)

# %% markdown
En el gráfico anterior vemos como con las variables `movement_acceleration` y
`attacking_finishing` los jugadores que tienen una posición de arqueros estan
muy abajo con poca aceleración y definición, los defenzores se encuentran más
arriba, seguimos con los mediocampistas y luego los delanteros son los que
tienen más aceleración y mejor puntaje al momento de patear al arco en ataque.
