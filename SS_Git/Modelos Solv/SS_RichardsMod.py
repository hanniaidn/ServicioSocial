import plotly.graph_objects as go
import numpy as np

# Función del modelo de Richards
def richards_model(t, K, r, a, v):
    return K / ((1 + a * np.exp(-r * t)) ** (1 / v))

# Parámetros del modelo
K = 1500  # Capacidad máxima de la población
r = 0.05  # Tasa de crecimiento
a = 10    # Posición de la curva
v = 2     # Forma de la curva
t_max = 50  # Tiempo total de simulación

# Array tiempo
t = np.linspace(0, t_max, num=500)

# Función para actualizar el modelo
def actualizar_modelo(K, r, a, v):
    return richards_model(t, K, r, a, v)

# Inicializar el gráfico
fig = go.Figure()

# Añadir la primera traza del gráfico
fig.add_trace(go.Scatter(x=t, y=actualizar_modelo(K, r, a, v),
                         mode='lines',
                         name=f'Modelo Generalizado de Richards'))

# Crear slider para la tasa de crecimiento (r)
r_values = np.linspace(0.05, 0.5, 20)
if r not in r_values:
    r_values = np.append(r_values, r)
r_values.sort()
r_slider = dict(
    active=r_values.tolist().index(r),
    currentvalue={"prefix": "Tasa de crecimiento (r): ", "font": {"size": 15}},  # Texto más pequeño
    pad={"t": 50},  # Espacio entre sliders
    steps=[dict(
        method='update',
        args=[{"y": [actualizar_modelo(K, r_value, a, v)]},
              {"title": f"Modelo de Richards"}],
        label=f'{r_value:.2f}'
    ) for r_value in r_values]
)

# Crear slider para la capacidad de carga (K)
K_values = np.linspace(1000, 2000, 10)
if K not in K_values:
    K_values = np.append(K_values, K)
K_values.sort()
K_slider = dict(
    active=K_values.tolist().index(K),
    currentvalue={"prefix": "Capacidad de carga (K): ", "font": {"size": 15}},  # Texto más pequeño
    pad={"t": 150},  # Espacio adicional para separar sliders
    steps=[dict(
        method='update',
        args=[{"y": [actualizar_modelo(K_value, r, a, v)]},
              {"title": f"Modelo de Richards"}],
        label=f'{K_value:.0f}'
    ) for K_value in K_values]
)

# Crear slider para la posición de la curva (a)
a_values = np.linspace(5, 15, 10)
if a not in a_values:
    a_values = np.append(a_values, a)
a_values.sort()
a_slider = dict(
    active=a_values.tolist().index(a),
    currentvalue={"prefix": "Posición de la curva (a): ", "font": {"size": 15}},  # Texto más pequeño
    pad={"t": 250},  # Más espacio para que los sliders no choquen
    steps=[dict(
        method='update',
        args=[{"y": [actualizar_modelo(K, r, a_value, v)]},
              {"title": f"Modelo de Richards"}],
        label=f'{a_value:.0f}'
    ) for a_value in a_values]
)

# Crear slider para la forma de la curva (v)
v_values = np.linspace(1, 3, 5)
if v not in v_values:
    v_values = np.append(v_values, v)
v_values.sort()
v_slider = dict(
    active=v_values.tolist().index(v),
    currentvalue={"prefix": "Forma de la curva (v): ", "font": {"size": 15}},  # Texto más pequeño
    pad={"t": 350},  # Espacio suficiente entre sliders
    steps=[dict(
        method='update',
        args=[{"y": [actualizar_modelo(K, r, a, v_value)]},
              {"title": f"Modelo de Richards"}],
        label=f'{v_value:.2f}'
    ) for v_value in v_values]
)

# Actualizar el layout del gráfico
fig.update_layout(
    sliders=[r_slider, K_slider, a_slider, v_slider],  # Añadir los sliders
    title='Modelo Generalizado de Richards',
    xaxis_title='Tiempo',
    yaxis_title='Población',
    template="plotly_white",
    margin=dict(l=40, r=40, b=200, t=80),  # Ajustar márgenes para acomodar sliders
)

# Mostrar el gráfico
fig.show()
