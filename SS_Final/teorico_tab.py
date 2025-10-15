# Servicio Social: Desarrollo de herramientas para el análisis de datos epidemiológicos
# Clave de registro: 2024-12/211-6690
# Responsable: Dr. Mario Santana Cibrian 
# Persona prestadora del servicio: Hannia Isela Dominguez Nuñez 
# ENES Unidad Juriquilla, UNAM 
# Septiembre 2025

import panel as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize
import io
from datetime import datetime

pn.extension('plotly', 'katex')

# Configuración del diseño
pn.config.sizing_mode = "stretch_width"
pn.config.raw_css = ["""
    .custom-tabs .bk-tab {
        font-size: 18px; /* Tamaño de letra más grande */
        font-weight: bold;
        padding: 10px 20px;
        color: #333; /* Color de texto más oscuro */
    }
    .custom-tabs .bk-tab:hover {
        background-color: #eee; /* Fondo gris claro al pasar el mouse */
    }
    .custom-tabs .bk-tab.active { /* Estilo para la pestaña activa */
        background-color: #ddd;
        color: #007bff; /* Color azul para la pestaña activa */
    }
    .custom-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px;
        background-color: #f9f9f9;
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .custom-header {
        display: flex;
        align-items: center; /* Contenido centrado verticalmente */
        justify-content: center; /* Contenido centrado horizontalmente */
    }
    .title {
        text-align: center; /* Título centrado */
    }
"""]

def render_ecuacion(ecuacion):
    return pn.pane.LaTeX(f"${ecuacion}$", sizing_mode="stretch_width")

##### SECCIÓN TEÓRICA #####

# Soluciones de modelos epidemiológicos 
# Modelo exponencial
r_exp = pn.widgets.FloatSlider(name='Tasa de crecimiento inicial', start=0.1, end=1.0, step=0.01, value=0.3)
N0_exp = pn.widgets.FloatSlider(name='Población inicial', start=10, end=100, step=1, value=50)
t_exp = pn.widgets.IntSlider(name='Tiempo', start=0, end=50, step=1, value=25)

def modelo_exponencial(t, N, r):
    return r * N

def sol_exp(N0, r, t):
    t_values = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_exponencial(t, N, r), [0, t], [N0], method='RK45', t_eval=t_values)
    return sol.t, sol.y[0]

@pn.depends(r_exp, N0_exp, t_exp)
def grafica_exponencial(r_exp, N0_exp, t_exp):
    t_values, N_values = sol_exp(N0_exp, r_exp, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo Exponencial'))
    fig.update_layout(title='Modelo Exponencial', xaxis_title='Tiempo', yaxis_title='Casos')
    return fig

# Modelo logístico
K_log = pn.widgets.FloatSlider(name='Capacidad de carga', start=50, end=500, step=10, value=200)
r_log = pn.widgets.FloatSlider(name='Tasa de crecimiento logístico', start=0.1, end=1.0, step=0.01, value=0.3)

def modelo_logistico(t, N, r, K):
    return r * N * (1 - N / K)

def sol_log(N0, r, K, t):
    t_values = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_logistico(t, N, r, K), [0, t], [N0], method='RK45', t_eval=t_values)
    return sol.t, sol.y[0]

@pn.depends(N0_exp, r_log, K_log, t_exp)
def grafica_logistico(N0_exp, r_log, K_log, t_exp):
    t_values, N_values = sol_log(N0_exp, r_log, K_log, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo Logístico'))
    fig.update_layout(title='Modelo Logístico', xaxis_title='Tiempo', yaxis_title='Casos')
    return fig

# Modelo de Richards
v_richards = pn.widgets.FloatSlider(name='Parámetro de forma (v)', start=0.1, end=5.0, step=0.1, value=1.0)

def modelo_richards(t, N, r, K, v):
    return r * N * (1 - (N / K)**v)

def sol_richards(N0, r, K, v, t):
    t_values = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_richards(t, N, r, K, v), [0, t], [N0], method='RK45', t_eval=t_values)
    return sol.t, sol.y[0]

@pn.depends(N0_exp, r_log, K_log, v_richards, t_exp)
def grafica_richards(N0_exp, r_log, K_log, v_richards, t_exp):
    t_values, N_values = sol_richards(N0_exp, r_log, K_log, v_richards, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Curva de Richards'))
    fig.update_layout(
        title='Curva de Richards',
        xaxis_title='Tiempo',
        yaxis_title='Casos',
        yaxis=dict(range=[0, K_log * 1.1])  # Ajustar el límite superior
    )
    return fig

# Modelo SIR 
beta_sir = pn.widgets.FloatSlider(name='Tasa de infección', start=0.1, end=1.0, step=0.01, value=0.3)
gamma_sir = pn.widgets.FloatSlider(name='Tasa de recuperación', start=0.05, end=0.5, step=0.01, value=0.1)
t_sir = pn.widgets.IntSlider(name='Tiempo', start=0, end=100, step=1, value=50)
N_sir = 10000  
S0_sir = pn.widgets.FloatSlider(name='Población inicial Susceptibles', start=1000, end=10000, step=100, value=9850)
I0_sir = pn.widgets.FloatSlider(name='Población inicial Infectados', start=1, end=1000, step=1, value=150)
R0_sir = pn.widgets.FloatSlider(name='Población inicial Recuperados', start=0, end=1000, step=1, value=0)
  
def modelo_sir(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def sol_sir(beta, gamma, N, S0, I0, R0, t):
    y0 = S0, I0, R0
    t_values = np.linspace(0, t, 100)
    ret = odeint(modelo_sir, y0, t_values, args=(N, beta, gamma))
    S, I, R = ret.T
    return t_values, S, I, R

@pn.depends(beta_sir, gamma_sir, t_sir, S0_sir, I0_sir, R0_sir)
def grafica_sir(beta_sir, gamma_sir, t_sir, S0_sir, I0_sir, R0_sir):
    N = S0_sir + I0_sir + R0_sir
    t_values, S, I, R = sol_sir(beta_sir, gamma_sir, N, S0_sir, I0_sir, R0_sir, t_sir)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=S, mode='lines', name='Susceptibles'))
    fig.add_trace(go.Scatter(x=t_values, y=I, mode='lines', name='Infectados'))
    fig.add_trace(go.Scatter(x=t_values, y=R, mode='lines', name='Recuperados'))
    fig.update_layout(title='Modelo SIR', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

# Modelo de Gompertz
K_gompertz = pn.widgets.FloatSlider(name='Capacidad de carga (K)', start=50, end=500, step=10, value=200)
a_gompertz = pn.widgets.FloatSlider(name='Tasa de crecimiento (a)', start=0.01, end=1.0, step=0.01, value=0.1)

def modelo_gompertz(t, N, a, K):
    return a * N * np.log(K / N)

def sol_gompertz(N0, a, K, t):
    t_values = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_gompertz(t, N, a, K), [0, t], [N0], method='RK45', t_eval=t_values)
    return sol.t, sol.y[0]

@pn.depends(N0_exp, a_gompertz, K_gompertz, t_exp)
def grafica_gompertz(N0_exp, a_gompertz, K_gompertz, t_exp):
    t_values, N_values = sol_gompertz(N0_exp, a_gompertz, K_gompertz, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo de Gompertz'))
    fig.update_layout(
        title='Modelo de Gompertz (Curva Sigmoide)',
        xaxis_title='Tiempo',
        yaxis_title='Población',
        yaxis=dict(range=[0, K_gompertz + 10])  
    )
    return fig

# Modelo Von Bertalanffy
L_bi = pn.widgets.FloatSlider(name='Tamaño límite (L)', start=50, end=500, step=10, value=300)
K_bi = pn.widgets.FloatSlider(name='Coeficiente de crecimiento (K)', start=0.01, end=1.0, step=0.01, value=0.1)
N0_bi = pn.widgets.FloatSlider(name='Población inicial', start=10, end=100, step=1, value=50)

def modelo_bertalanffy(t, L, K, N0):
    return L * (1 - np.exp(-K * t)) + N0

def sol_bertalanffy(L, K, N0, t):
    t_values = np.linspace(0, t, 100)
    N_values = modelo_bertalanffy(t_values, L, K, N0)
    return t_values, N_values

@pn.depends(L_bi, K_bi, N0_bi, t_exp)
def grafica_bertalanffy(L_bi, K_bi, N0_bi, t_exp):
    t_values, N_values = sol_bertalanffy(L_bi, K_bi, N0_bi, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo Von Bertalanffy'))
    fig.update_layout(
        title='Modelo Von Bertalanffy',
        xaxis_title='Tiempo',
        yaxis_title='Población',
        yaxis=dict(range=[0, L_bi*1.1])  
    )
    return fig

# Modelo de Janoschek
beta_jan = pn.widgets.FloatSlider(name='Asintota inferior (β)', start=0, end=50, step=1, value=10)
L_jan = pn.widgets.FloatSlider(name='Asintota superior (L)', start=50, end=500, step=10, value=300)
k_jan = pn.widgets.FloatSlider(name='Tasa de crecimiento (k)', start=0.01, end=1.0, step=0.01, value=0.1)
delta_jan = pn.widgets.FloatSlider(name='Parámetro δ', start=0.5, end=5.0, step=0.1, value=1.0)
N0_jan = pn.widgets.FloatSlider(name='Población inicial', start=10, end=100, step=1, value=50)

def modelo_janoschek(t, beta, L, k, delta, N0):
    return beta + (L - beta) * (1 - np.exp(-k * t))**delta + N0

def sol_janoschek(beta, L, k, delta, N0, t):
    t_values = np.linspace(0, t, 100)
    N_values = modelo_janoschek(t_values, beta, L, k, delta, N0)
    return t_values, N_values

@pn.depends(beta_jan, L_jan, k_jan, delta_jan, N0_jan, t_exp)
def grafica_janoschek(beta_jan, L_jan, k_jan, delta_jan, N0_jan, t_exp):
    t_values, N_values = sol_janoschek(beta_jan, L_jan, k_jan, delta_jan, N0_jan, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo Janoschek'))
    fig.update_layout(
        title='Modelo Janoschek',
        xaxis_title='Tiempo',
        yaxis_title='Población',
        yaxis=dict(range=[0, L_jan*1.1])  
    )
    return fig

# Descripción de modelos 
descripcion_exponencial = pn.Card(
    pn.Column(
        pn.pane.Markdown(r"""
                         
        **Descripción:**  
        El modelo exponencial describe el crecimiento de una población en condiciones ideales, donde no hay limitaciones de recursos. 
        La tasa de crecimiento es constante y proporcional al tamaño de la población. 
        
        **Parámetros:**  
        - r: Tasa de crecimiento inicial (rango: 0.1 a 1.0).  
        - N: Población inicial (rango: 10 a 100).  
        
        **Ecuación:**"""),
        render_ecuacion(r"\frac{dN}{dt} = rN"), 
    ),
    title="Modelo Exponencial",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

descripcion_logistico = pn.Card(
    pn.Column(
    pn.pane.Markdown(r"""
                     
    **Descripción:**  
    El modelo logístico describe el crecimiento de una población en condiciones más realistas, donde hay una capacidad de carga máxima 
    (K) que limita el crecimiento. La tasa de crecimiento disminuye a medida que la población se acerca a la capacidad de carga.
    
    **Parámetros:**  
    - r: Tasa de crecimiento logístico (rango: 0.1 a 1.0).  
    - K: Capacidad de carga (rango: 50 a 500).  
    
     **Ecuación:**"""),
    render_ecuacion(r"\frac{dN}{dt} = rN \left(1 - \frac{N}{K}\right)"),
    ), 
    title="Modelo Logístico",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

descripcion_sir = pn.Card(
    pn.Column(
    pn.pane.Markdown(r"""  
    
    **Descripción:**  
    El modelo SIR es un modelo epidemiológico que divide la población en tres grupos: Susceptibles (S), Infectados (I) 
    y Recuperados (R). El modelo describe cómo una enfermedad se propaga a través de la población y cómo los individuos se recuperan.
   
    **Parámetros:**
    - 𝛽: Tasa de infección (rango: 0.1 a 1.0).  
    - 𝛾: Tasa de recuperación (rango: 0.05 a 0.5).  
    
    **Ecuaciones:**"""),
    render_ecuacion(r"\frac{dS}{dt} = -\beta \frac{SI}{N}"),
    render_ecuacion(r"\frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I"),
    render_ecuacion(r"\frac{dR}{dt} = \gamma I"), 
    ),
    title="Modelo SIR",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

descripcion_richards = pn.Card(
    pn.Column(
    pn.pane.Markdown(r""" 
    
    **Descripción:**  
    El modelo de Richards es una generalización del modelo logístico que permite una mayor flexibilidad en la forma de la curva de 
    crecimiento. Incluye un parámetro de forma (v) que controla la asimetría de la curva.
    
    **Parámetros:**  
    - r: Tasa de crecimiento (rango: 0.1 a 1.0).  
    - K: Capacidad de carga (rango: 50 a 500).  
    - v: Parámetro de forma (rango: 0.1 a 5.0). 
    
    **Ecuación:**  """),
    render_ecuacion(r"[ \frac{dN}{dt} = rN \left(1 - \left(\frac{N}{K}\right)^v\right)]"),
    ),
    title="Modelo de Richards",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

descripcion_gompertz = pn.Card(
    pn.Column(
    pn.pane.Markdown(r""" 
    
    **Descripción:**  
    El modelo de Gompertz describe el crecimiento de una población donde la tasa de crecimiento disminuye exponencialmente con el 
    tiempo. Es útil para modelar el crecimiento de tumores y otros fenómenos biológicos.
    
    **Parámetros:**  
    - a: Tasa de crecimiento (rango: 0.01 a 1.0).  
    - K: Capacidad de carga (rango: 50 a 500). 
    
    **Ecuación:**"""),
    render_ecuacion(r"\frac{dN}{dt} = aN \ln\left(\frac{K}{N}\right)"),
    ),
    title="Modelo de Gompertz",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

descripcion_bertalanffy = pn.Card(
    pn.Column(
    pn.pane.Markdown(r"""  
   
    **Descripción:**  
    El modelo Von Bertalanffy describe el crecimiento de una población donde la tasa de crecimiento depende de la diferencia entre el 
    tamaño actual y el tamaño límite (L). Es comúnmente utilizado en ecología para modelar el crecimiento de peces y otros organismos.
   
    **Parámetros:**  
    - L: Tamaño límite (rango: 50 a 500).  
    - K: Coeficiente de crecimiento (rango: 0.01 a 1.0).
    
    **Ecuación:** """),
    render_ecuacion(r"\frac{dN}{dt} = L \left(1 - e^{-Kt}\right)+N"),
    ),
    title="Modelo Von Bertalanffy",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

descripcion_janoschek = pn.Card(
    pn.Column(
    pn.pane.Markdown(r"""  
   
    **Descripción:**  
    El modelo Janoschek es un modelo de crecimiento que permite una asintota inferior (β) y una superior (L), con una tasa de 
    crecimiento (k) y un parámetro de forma (δ). Es útil para modelar fenómenos donde el crecimiento no comienza desde cero o 
    no alcanza un límite superior.
   
    **Parámetros:**  
    - 𝛽: Asintota inferior (rango: 0 a 50).  
    - L: Asintota superior (rango: 50 a 500).  
    - k: Tasa de crecimiento (rango: 0.01 a 1.0).  
    - δ: Parámetro de forma (rango: 0.5 a 5.0).
   
    **Ecuación:** """),
    render_ecuacion(r"[ N(t) = \beta + (L - \beta) \left(1 - e^{-kt}\right)^\delta]"),
    ),
    title="Modelo Janoschek",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

model_tabs_content = {
    'Modelo Exponencial': pn.Row(descripcion_exponencial, pn.Column(grafica_exponencial, r_exp, N0_exp, t_exp)),
    'Modelo Logístico': pn.Row(descripcion_logistico, pn.Column(grafica_logistico, K_log, r_log, N0_exp, t_exp)),
    'Modelo SIR': pn.Row(descripcion_sir, pn.Column(grafica_sir, beta_sir, gamma_sir, t_sir, S0_sir, I0_sir, R0_sir)),
    'Modelo de Richards': pn.Row(descripcion_richards, pn.Column(grafica_richards, v_richards, N0_exp, t_exp)),
    'Modelo de Gompertz': pn.Row(descripcion_gompertz, pn.Column(grafica_gompertz, a_gompertz, K_gompertz, N0_exp, t_exp)),
    'Modelo Von Bertalanffy': pn.Row(descripcion_bertalanffy, pn.Column(grafica_bertalanffy, L_bi, K_bi, N0_bi, t_exp)),
    'Modelo Janoschek': pn.Row(descripcion_janoschek, pn.Column(grafica_janoschek, beta_jan, L_jan, k_jan, delta_jan, N0_jan, t_exp))
}

selected_tab_content = model_tabs_content['Modelo Exponencial']

def create_teorico_tab():
    menu = pn.widgets.Select(options=['Modelo Exponencial', 'Modelo Logístico', 'Modelo SIR', 'Modelo de Richards', 'Modelo de Gompertz', 'Modelo Von Bertalanffy', 'Modelo Janoschek'])
    
    model_tabs_content = {
    'Modelo Exponencial': pn.Row(descripcion_exponencial, pn.Column(grafica_exponencial, r_exp, N0_exp, t_exp)),
    'Modelo Logístico': pn.Row(descripcion_logistico, pn.Column(grafica_logistico, K_log, r_log, N0_exp, t_exp)),
    'Modelo SIR': pn.Row(descripcion_sir, pn.Column(grafica_sir, beta_sir, gamma_sir, t_sir, S0_sir, I0_sir, R0_sir)),
    'Modelo de Richards': pn.Row(descripcion_richards, pn.Column(grafica_richards, v_richards, N0_exp, t_exp)),
    'Modelo de Gompertz': pn.Row(descripcion_gompertz, pn.Column(grafica_gompertz, a_gompertz, K_gompertz, N0_exp, t_exp)),
    'Modelo Von Bertalanffy': pn.Row(descripcion_bertalanffy, pn.Column(grafica_bertalanffy, L_bi, K_bi, N0_bi, t_exp)),
    'Modelo Janoschek': pn.Row(descripcion_janoschek, pn.Column(grafica_janoschek, beta_jan, L_jan, k_jan, delta_jan, N0_jan, t_exp))
}
    
    selected_tab_content = model_tabs_content['Modelo Exponencial']
    
    def update_tab(event):
        nonlocal selected_tab_content
        selected_tab_content = model_tabs_content[event.new]
        dynamic_area.objects = [selected_tab_content]
    
    menu.param.watch(lambda event: update_tab(event), 'value')
    
    dynamic_area = pn.Column(selected_tab_content)
    
    teorico_tab = pn.Column(
        pn.Row( 
            pn.layout.HSpacer(),
            menu,
            pn.layout.HSpacer()
        ),
        dynamic_area
    )
    
    return teorico_tab