import plotly.graph_objects as go
import numpy as np

# Parámetros del modelo
N0 = 100  # Población inicial
r = 0.1  # Tasa de crecimiento inicial
K = 1000  # Capacidad máxima de la población
t_max = 50  # Tiempo total de simulación

# Array tiempo
t = np.linspace(0, t_max, num=500)

# funcion del modelo
def crecimiento_logistico(r, K):
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


fig = go.Figure()


fig.add_trace(go.Scatter(x=t, y=crecimiento_logistico(r, K),
                         mode='lines',
                         name=f'Crecimiento Logístico'))

#slider para r 
r_values = np.linspace(0.05, 0.5, 10)
if r not in r_values:
    r_values = np.append(r_values, r)
r_values.sort()

r_slider = dict(
    active=r_values.tolist().index(r),
    currentvalue={"prefix": "Tasa de crecimiento (r): ", "font": {"size": 15}},  
    pad={"t": 50},  
    steps=[dict(
        method='update',
        args=[{"y": [crecimiento_logistico(r_value, K)]},
              {"title": f"Crecimiento Logístico"}],
        label=f'{r_value:.2f}'
    ) for r_value in r_values]
)

#slider para K
K_values = np.linspace(500, 2000, 10)
if K not in K_values:
    K_values = np.append(K_values, K)
K_values.sort()

K_slider = dict(
    active=K_values.tolist().index(K),
    currentvalue={"prefix": "Capacidad de carga (K): ", "font": {"size": 15}},  
    pad={"t": 150}, 
    steps=[dict(
        method='update',
        args=[{"y": [crecimiento_logistico(r, K_value)]},
              {"title": f"Crecimiento Logístico"}],
        label=f'{int(K_value)}'
    ) for K_value in K_values]
)


fig.update_layout(
    sliders=[r_slider, K_slider],  
    title='Modelo de Crecimiento Logístico',
    xaxis_title='Tiempo',
    yaxis_title='Población',
    template="plotly_white",
    margin=dict(l=40, r=40, b=200, t=80)  
)


fig.show()
