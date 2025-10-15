import panel as pn
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import odeint

pn.extension('plotly')

# Parámetros iniciales
N0_logistico = 100
r_logistico = 0.1
K_logistico = 1000

N0_exp = 100000
r_exp = 0.05

N_sir = 1000
I0_sir = 100
R0_sir = 0
beta_sir = 0.3
gamma_sir = 0.1
t_max_sir = 160

K_richards = 1500
r_richards = 0.25
a_richards = 10
v_richards = 2
t_max_richards = 50

t_max = 1000  # Tiempo total


# Funcion para cada modelo 

# modelo de crecimiento logístico
def crecimiento_logistico(r, K):
    t = np.linspace(0, t_max, num=500)
    return t, K / (1 + ((K - N0_logistico) / N0_logistico) * np.exp(-r * t))


# modelo de crecimiento exponencial
def crecimiento_exponencial(r):
    t = np.linspace(0, t_max, num=500)
    return t, N0_exp * np.exp(r * t)


# modelo SIR
def deriv_sir(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def solve_sir(beta, gamma):
    t = np.linspace(0, t_max_sir, t_max_sir)
    S0 = N_sir - I0_sir - R0_sir
    y0 = S0, I0_sir, R0_sir
    ret = odeint(deriv_sir, y0, t, args=(N_sir, beta, gamma))
    S, I, R = ret.T
    return t, S, I, R


# modelo de Richards
def richards_model(t, K, r, a, v):
    return K / ((1 + a * np.exp(-r * t)) ** (1 / v))

def solve_richards(K, r, a, v):
    t = np.linspace(0, t_max_richards, num=500)
    return t, richards_model(t, K, r, a, v)


# Función para actualizar las gráficas
def update_crecimiento_logistico(r, K):
    t, N_t = crecimiento_logistico(r, K)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, 
                             y=N_t, 
                             mode='lines', 
                             name=f'Crecimiento Logístico'))
    fig.update_layout(title='Modelo de Crecimiento Logístico', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

def update_crecimiento_exponencial(r):
    t, N_t = crecimiento_exponencial(r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, 
                             y=N_t, 
                             mode='lines', 
                             name=f'Crecimiento Exponencial'))
    fig.update_layout(title='Modelo de Crecimiento Exponencial', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

def update_sir(beta, gamma):
    t, S, I, R = solve_sir(beta, gamma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, 
                             y=S, 
                             mode='lines', 
                             name='Susceptibles', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, 
                             y=I, 
                             mode='lines', 
                             name='Infectados', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=t, 
                             y=R, 
                             mode='lines', 
                             name='Recuperados', line=dict(color='green')))
    fig.update_layout(title='Modelo SIR', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

def update_richards(K, r, a, v):
    t, N_t = solve_richards(K, r, a, v)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, 
                             y=N_t, 
                             mode='lines', 
                             name=f'Modelo Richards'))
    fig.update_layout(title='Modelo Generalizado de Richards', xaxis_title='Tiempo', yaxis_title='Población')
    return fig


# Sliders para los modelos
r_slider_logistico = pn.widgets.FloatSlider(name='Tasa de crecimiento', 
                                            start=0.05, end=0.5, 
                                            value=r_logistico,
                                            step=0.001)
K_slider_logistico = pn.widgets.IntSlider(name='Capacidad máxima', 
                                          start=500, 
                                          end=2000, 
                                          value=K_logistico, 
                                          step=100)

r_slider_exponencial = pn.widgets.FloatSlider(name='Tasa de crecimiento', 
                                              start=0.01, 
                                              end=0.1, 
                                              value=r_exp, 
                                              step=0.001)

beta_slider_sir = pn.widgets.FloatSlider(name='Tasa de infección', 
                                         start=0.1, 
                                         end=0.5, 
                                         value=beta_sir, 
                                         step=0.01)
gamma_slider_sir = pn.widgets.FloatSlider(name='Tasa de recuperación', 
                                          start=0.05, 
                                          end=0.2, 
                                          value=gamma_sir, 
                                          step=0.001)

K_slider_richards = pn.widgets.IntSlider(name='Capacidad máxima', 
                                         start=1000, 
                                         end=2000, 
                                         value=K_richards, 
                                         step=100)
r_slider_richards = pn.widgets.FloatSlider(name='Tasa de crecimiento', 
                                           start=0.1, 
                                           end=0.5, 
                                           value=r_richards, 
                                           step=0.001)
a_slider_richards = pn.widgets.IntSlider(name='Posición', 
                                         start=5, 
                                         end=15, 
                                         value=a_richards, 
                                         step=1)
v_slider_richards = pn.widgets.FloatSlider(name='Forma de la curva', 
                                           start=1, 
                                           end=3, 
                                           value=v_richards, 
                                           step=0.001)

# Panel Layouts
logistico_panel = pn.Column(r_slider_logistico, 
                            K_slider_logistico, 
                            pn.bind(update_crecimiento_logistico, r_slider_logistico,  K_slider_logistico))
exponencial_panel = pn.Column(r_slider_exponencial, 
                              pn.bind(update_crecimiento_exponencial, r_slider_exponencial))
sir_panel = pn.Column(beta_slider_sir, 
                      gamma_slider_sir, 
                      pn.bind(update_sir, beta_slider_sir, gamma_slider_sir))
richards_panel = pn.Column(K_slider_richards, 
                           r_slider_richards, 
                           a_slider_richards, 
                           v_slider_richards, 
                           pn.bind(update_richards, K_slider_richards, r_slider_richards, a_slider_richards, v_slider_richards))

# Layout principal
sidebar = pn.Column(
    pn.layout.HSpacer(),
    pn.pane.Markdown("# Análisis teórico de los modelos", align='center'),
    pn.Tabs(
        ("Crecimiento Logístico", logistico_panel),
        ("Crecimiento Exponencial", exponencial_panel),
        ("Modelo SIR", sir_panel),
        ("Modelo de Richards", richards_panel),
        align='center'
    )
)


pn.Row(sidebar).show()
