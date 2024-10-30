import panel as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy import stats
import io  # para leer los csv

pn.extension('plotly')

# titulo del dashboard 
titulo = pn.pane.Markdown("""
# <center style='color:#4CAF50; font-size:40px;'>Análisis de Modelos Epidemiológicos</center>
""", width=800)

#MODELO LOGISTICO 
r_log = pn.widgets.FloatSlider(name='Tasa de crecimiento', start=0.1, end=1.0, step=0.01, value=0.3)
K_log = pn.widgets.FloatSlider(name='Capacidad de carga', start=100, end=1000, step=10, value=500)
N0_log = pn.widgets.FloatSlider(name='Población inicial', start=10, end=100, step=1, value=50)
t_log = pn.widgets.IntSlider(name='Tiempo', start=0, end=100, step=1, value=50)

def modelo_logistico(N, t, r, K):
    dNdt = r * N * (1 - N/K)
    return dNdt

def sol_log(N0, r, K, t):
    t_values = np.linspace(0, t, 100)
    N_values = odeint(modelo_logistico, N0, t_values, args=(r, K))
    return t_values, N_values[:, 0]

@pn.depends(r_log, K_log, N0_log, t_log)
def grafica_logistico(r_log, K_log, N0_log, t_log):
    t_values, N_values = sol_log(N0_log, r_log, K_log, t_log)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo Logístico'))
    fig.update_layout(title='Modelo Logístico', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

#MODELO EXPONENCIAL 
r_exp = pn.widgets.FloatSlider(name='Tasa de crecimiento', start=0.1, end=1.0, step=0.01, value=0.3)
N0_exp = pn.widgets.FloatSlider(name='Población inicial ', start=10, end=100, step=1, value=50)
t_exp = pn.widgets.IntSlider(name='Tiempo', start=0, end=100, step=1, value=50)

def modelo_exponencial(N, t, r):
    dNdt = r * N
    return dNdt

def sol_exp(N0, r, t):
    t_values = np.linspace(0, t, 100)
    N_values = odeint(modelo_exponencial, N0, t_values, args=(r,))
    return t_values, N_values[:, 0]

@pn.depends(r_exp, N0_exp, t_exp)
def grafica_exponencial(r_exp, N0_exp, t_exp):
    t_values, N_values = sol_exp(N0_exp, r_exp, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo Exponencial'))
    fig.update_layout(title='Modelo Exponencial', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

#MODELO SIR 
beta_sir = pn.widgets.FloatSlider(name='Tasa de infección', start=0.1, end=1.0, step=0.01, value=0.3)
gamma_sir = pn.widgets.FloatSlider(name='Tasa de recuperación', start=0.05, end=0.5, step=0.01, value=0.1)
t_sir = pn.widgets.IntSlider(name='Tiempo', start=0, end=100, step=1, value=50)
N_sir = 10000  #
I0_sir = 150   
R0_sir = 0    

def modelo_sir(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def sol_sir(beta, gamma, N, I0, R0, t):
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t_values = np.linspace(0, t, 100)
    ret = odeint(modelo_sir, y0, t_values, args=(N, beta, gamma))
    S, I, R = ret.T
    return t_values, S, I, R

@pn.depends(beta_sir, gamma_sir, t_sir)
def grafica_sir(beta_sir, gamma_sir, t_sir):
    t_values, S, I, R = sol_sir(beta_sir, gamma_sir, N_sir, I0_sir, R0_sir, t_sir)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=S, mode='lines', name='Susceptibles'))
    fig.add_trace(go.Scatter(x=t_values, y=I, mode='lines', name='Infectados'))
    fig.add_trace(go.Scatter(x=t_values, y=R, mode='lines', name='Recuperados'))
    fig.update_layout(title='Modelo SIR', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

#MODELO RICHARDS
K_rich = pn.widgets.FloatSlider(name='Capacidad de carga', start=100, end=2000, step=50, value=1000)
r_rich = pn.widgets.FloatSlider(name='Tasa de crecimiento', start=0.1, end=1.0, step=0.01, value=0.2)
a_rich = pn.widgets.FloatSlider(name='Parámetro', start=0.1, end=5.0, step=0.1, value=1.0)
v_rich = pn.widgets.FloatSlider(name='Parámetro', start=0.1, end=3.0, step=0.1, value=1.0)
t_rich = pn.widgets.IntSlider(name='Tiempo', start=0, end=100, step=1, value=50)

def modelo_richards(N, t, r, K, a, v):
    return r * N * (1 - (N/K)**v) / (1 + a * N)

def sol_rich(N0, r, K, a, v, t):
    t_values = np.linspace(0, t, 100)
    N_values = odeint(modelo_richards, N0, t_values, args=(r, K, a, v))
    return t_values, N_values[:, 0]

@pn.depends(K_rich, r_rich, a_rich, v_rich, t_rich)
def grafica_richards(K_rich, r_rich, a_rich, v_rich, t_rich):
    t_values, N_values = sol_rich(10, r_rich, K_rich, a_rich, v_rich, t_rich)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo Richards'))
    fig.update_layout(title='Modelo de Richards', xaxis_title='Tiempo', yaxis_title='Población')
    return fig


#CARGA DE CSV Y GRAFICA 
fileInput = pn.widgets.FileInput(accept='.csv')
uploadButton = pn.widgets.Button(name='Upload', button_type = 'primary')

table = pn.widgets.Tabulator(pagination='remote', page_size=10)
#tab.getElementById('table').style.display='none'


def process_file(event):
   if fileInput.value is not None:
       table.value = pd.read_csv(io.BytesIO(fileInput.value))
        #tab.getElementById('table').style.display = 'block'

uploadButton.on_click(process_file)


teorico_tab = pn.Tabs(
    ('Modelo Logístico', pn.Column(r_log, K_log, N0_log, t_log, grafica_logistico)),
    ('Modelo Exponencial', pn.Column(r_exp, N0_exp, t_exp, grafica_exponencial)),
    ('Modelo SIR', pn.Column(beta_sir, gamma_sir, t_sir, grafica_sir)),
    ('Modelo de Richards', pn.Column(K_rich, r_rich, a_rich, v_rich, t_rich, grafica_richards))
)

estadistico_tab = pn.Column(fileInput, uploadButton, table)


secciones = pn.Tabs(
    ('Estudio Teórico', teorico_tab),
    ('Estudio Estadístico', estadistico_tab)
)


dashboard = pn.Column(titulo, secciones)

dashboard.show()
