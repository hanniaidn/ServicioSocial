import panel as pn
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import odeint

pn.extension('plotly')

# Título del dashboard centrado y con estilo colorido
titulo = pn.pane.Markdown("""
# <center style='color:#4CAF50; font-size:40px;'>Análisis de Modelos Epidemiológicos</center>
""", width=800)

# Parámetros iniciales de todos los modelos
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

# Modelos

# Crecimiento logístico
def solve_crecimiento_logistico(r, K, N0):
    t = np.linspace(0, 1000, num=500)
    N_t = K / (1 + ((K - N0) / N0) * np.exp(-r * t))
    return t, N_t

# Crecimiento exponencial
def solve_crecimiento_exponencial(r, N0):
    t = np.linspace(0, 1000, num=500)
    N_t = N0 * np.exp(r * t)
    return t, N_t

# Modelo SIR
def deriv_sir(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def solve_sir(beta, gamma, N, I0, R0):
    t = np.linspace(0, t_max_sir, t_max_sir)
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    ret = odeint(deriv_sir, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return t, S, I, R

# Modelo de Richards
def solve_richards(K, r, a, v):
    t = np.linspace(0, t_max_richards, num=500)
    N_t = K / ((1 + a * np.exp(-r * t)) ** (1 / v))
    return t, N_t

# Actualizaciones interactivas

# Gráfica del modelo logístico
def update_crecimiento_logistico(r, K, N0):
    t, N_t = solve_crecimiento_logistico(r, K, N0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=N_t, mode='lines', name='Crecimiento Logístico'))
    fig.update_layout(title='Modelo de Crecimiento Logístico', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

# Gráfica del modelo exponencial
def update_crecimiento_exponencial(r, N0):
    t, N_t = solve_crecimiento_exponencial(r, N0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=N_t, mode='lines', name='Crecimiento Exponencial'))
    fig.update_layout(title='Modelo de Crecimiento Exponencial', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

# Gráfica del modelo SIR
def update_sir(beta, gamma):
    t, S, I, R = solve_sir(beta, gamma, N_sir, I0_sir, R0_sir)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptibles', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infectados', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recuperados', line=dict(color='green')))
    fig.update_layout(title='Modelo SIR', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

# Gráfica del modelo de Richards
def update_richards(K, r, a, v):
    t, N_t = solve_richards(K, r, a, v)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=N_t, mode='lines', name='Modelo Richards'))
    fig.update_layout(title='Modelo Generalizado de Richards', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

# Sliders para todos los modelos

# Sliders logístico
r_slider_logistico = pn.widgets.FloatSlider(name='Tasa de crecimiento', start=0.05, end=0.5, value=r_logistico, step=0.001)
K_slider_logistico = pn.widgets.IntSlider(name='Capacidad máxima', start=500, end=2000, value=K_logistico, step=100)
N0_slider_logistico = pn.widgets.IntSlider(name='Población inicial', start=50, end=200, value=N0_logistico, step=10)

# Sliders exponencial
r_slider_exponencial = pn.widgets.FloatSlider(name='Tasa de crecimiento', start=0.01, end=0.1, value=r_exp, step=0.001)
N0_slider_exponencial = pn.widgets.IntSlider(name='Población inicial', start=50000, end=150000, value=N0_exp, step=1000)

# Sliders SIR
beta_slider_sir = pn.widgets.FloatSlider(name='Tasa de infección', start=0.1, end=0.5, value=beta_sir, step=0.01)
gamma_slider_sir = pn.widgets.FloatSlider(name='Tasa de recuperación', start=0.05, end=0.2, value=gamma_sir, step=0.001)

# Sliders Richards
K_slider_richards = pn.widgets.IntSlider(name='Capacidad máxima', start=1000, end=2000, value=K_richards, step=100)
r_slider_richards = pn.widgets.FloatSlider(name='Tasa de crecimiento', start=0.1, end=0.5, value=r_richards, step=0.001)
a_slider_richards = pn.widgets.IntSlider(name='Posición', start=5, end=15, value=a_richards, step=1)
v_slider_richards = pn.widgets.FloatSlider(name='Forma de la curva', start=1, end=3, value=v_richards, step=0.001)

# Paneles para cada modelo

logistico_panel = pn.Column(
    r_slider_logistico, 
    K_slider_logistico, 
    N0_slider_logistico, 
    pn.bind(update_crecimiento_logistico, r_slider_logistico, K_slider_logistico, N0_slider_logistico)
)

exponencial_panel = pn.Column(
    r_slider_exponencial, 
    N0_slider_exponencial, 
    pn.bind(update_crecimiento_exponencial, r_slider_exponencial, N0_slider_exponencial)
)

sir_panel = pn.Column(
    beta_slider_sir, 
    gamma_slider_sir, 
    pn.bind(update_sir, beta_slider_sir, gamma_slider_sir)
)

richards_panel = pn.Column(
    K_slider_richards, 
    r_slider_richards, 
    a_slider_richards, 
    v_slider_richards, 
    pn.bind(update_richards, K_slider_richards, r_slider_richards, a_slider_richards, v_slider_richards)
)

# Diseño del menú centrado y atractivo
estudio_teorico = pn.pane.Markdown("<center style='color:#FFA500; font-size:30px;'>Estudio Teórico</center>")
estudio_estadistico = pn.pane.Markdown("<center style='color:#FFA500; font-size:30px;'>Estudio Estadístico</center>")

# Organizar los subtítulos en una fila
subtitulos = pn.Row(estudio_teorico, estudio_estadistico)

# Panel principal con las opciones
dashboard = pn.Column(
    titulo,
    subtitulos,
    pn.Tabs(
        ('Modelo Logístico', logistico_panel),
        ('Modelo Exponencial', exponencial_panel),
        ('Modelo SIR', sir_panel),
        ('Modelo Richards', richards_panel)
    )
)

# Servir el dashboard
pn.serve(dashboard)
