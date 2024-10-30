import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint

# Parámetros iniciales
N = 1000  # Población total
I0 = 10    # Infectados iniciales
R0 = 0    # Recuperados iniciales
S0 = N - I0 - R0  # Susceptibles iniciales
beta = 0.3  # Tasa de infección inicial
gamma = 0.1 # Tasa de recuperación inicial
t_max = 160  # Tiempo total de simulación

#define eqs de SIR
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

#funcion SIR
def solve_sir(beta, gamma):
    y0 = S0, I0, R0
    t = np.linspace(0, t_max, t_max)
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return t, S, I, R

#resuelve SIR
t, S, I, R = solve_sir(beta, gamma)

#inicializa la grafica
fig = go.Figure()

#confg de las lineas
fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptibles', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infectados', line=dict(color='red')))
fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recuperados', line=dict(color='green')))

#slider para beta
beta_values = np.linspace(0.1, 1, 100)
beta_slider = dict(
    active=2,  
    currentvalue={"prefix": "Tasa de infección (beta): ", "font": {"size": 15}},  
    pad={"t": 70},  #muestra en un renglon dif
    steps=[dict(
        method='update',
        args=[{"y": [solve_sir(b, gamma)[1], solve_sir(b, gamma)[2], solve_sir(b, gamma)[3]]},
              {"title": f"Modelo SIR (beta={b:.2f}, gamma={gamma:.2f})"}],
        label=f'{b:.2f}'
    ) for b in beta_values]
)

#slider para gamma
gamma_values = np.linspace(0.05, 0.5, 10)
gamma_slider = dict(
    active=1,  
    currentvalue={"prefix": "Tasa de recuperación (gamma): ", "font": {"size": 15}},  #
    pad={"t": 150},  #muestra en un renglon dif
    steps=[dict(
        method='update',
        args=[{"y": [solve_sir(beta, g)[1], solve_sir(beta, g)[2], solve_sir(beta, g)[3]]},
              {"title": f"Modelo SIR "}],
        label=f'{g:.2f}'
    ) for g in gamma_values]
)

#actualiza el grafico
fig.update_layout(
    sliders=[beta_slider, gamma_slider],
    title='Modelo SIR',
    xaxis_title='Tiempo',
    yaxis_title='Número de personas',
    xaxis=dict(tickangle=45),  
    margin=dict(l=40, r=40, b=100, t=80),  #acomoda los sliders
    template="plotly_white"
)


fig.show()
