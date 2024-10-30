import plotly.graph_objects as go
import numpy as np

# Parámetros del modelo
N0 = 100000  # Población inicial
r = 0.05  # Tasa de crecimiento inicial
t_max = 1000  # Tiempo total de simulación


t = np.linspace(0, t_max, 1000)

# funcion del mpdelo
def crecimiento_exponencial(r):
    return N0 * np.exp(r * t)


fig = go.Figure()

fig.add_trace(go.Scatter(x=t, y=crecimiento_exponencial(r),
                         mode='lines',
                         name=f'Crecimiento Exponencial'))


r_values = np.linspace(0.01, 0.1, 20)
if r not in r_values:
    r_values = np.append(r_values, r)
r_values.sort()

r_slider = dict(
    active=r_values.tolist().index(r),
    currentvalue={"prefix": "Tasa de crecimiento (r): ", "font": {"size": 15}},  
    pad={"t": 50},  
    steps=[dict(
        method='update',
        args=[{"y": [crecimiento_exponencial(r_value)]},
              {"title": f'Crecimiento Exponencial'}],
        label=f'{r_value:.2f}'
    ) for r_value in r_values]
)


fig.update_layout(
    sliders=[r_slider],  
    title='Modelo de Crecimiento Exponencial',
    xaxis_title='Tiempo',
    yaxis_title='Población',
    template="plotly_white",
    margin=dict(l=40, r=40, b=100, t=80)  
)


fig.show()
