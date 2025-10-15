import panel as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize
import io
from datetime import datetime

pn.extension('plotly', 'katex')

# configuración del diseño
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
"""]

def render_ecuacion(ecuacion):
    return pn.pane.LaTeX(f"${ecuacion}$", sizing_mode="stretch_width")

# Modelo Exponencial
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

# Modelo Logístico
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
    fig.update_layout(title='Curva de Richards', xaxis_title='Tiempo', yaxis_title='Casos')
    return fig

# Modelo SIR 
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

# Modelo Bertalanffy-Ivlev
L_bi = pn.widgets.FloatSlider(name='Tamaño límite (L)', start=50, end=500, step=10, value=300)
K_bi = pn.widgets.FloatSlider(name='Coeficiente de crecimiento (K)', start=0.01, end=1.0, step=0.01, value=0.1)

def modelo_bertalanffy(t, L, K):
    return L * (1 - np.exp(-K * t))

def sol_bertalanffy(L, K, t):
    t_values = np.linspace(0, t, 100)
    N_values = modelo_bertalanffy(t_values, L, K)
    return t_values, N_values

@pn.depends(L_bi, K_bi, t_exp)
def grafica_bertalanffy(L_bi, K_bi, t_exp):
    t_values, N_values = sol_bertalanffy(L_bi, K_bi, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo Bertalanffy-Ivlev'))
    fig.update_layout(
        title='Modelo Bertalanffy-Ivlev',
        xaxis_title='Tiempo',
        yaxis_title='Población',
        yaxis=dict(range=[0, L_bi + 10])  
    )
    return fig

# Modelo Janoschek
beta_jan = pn.widgets.FloatSlider(name='Asintota inferior (β)', start=0, end=50, step=1, value=10)
L_jan = pn.widgets.FloatSlider(name='Asintota superior (L)', start=50, end=500, step=10, value=300)
k_jan = pn.widgets.FloatSlider(name='Tasa de crecimiento (k)', start=0.01, end=1.0, step=0.01, value=0.1)
delta_jan = pn.widgets.FloatSlider(name='Parámetro δ', start=0.5, end=5.0, step=0.1, value=1.0)

def modelo_janoschek(t, beta, L, k, delta):
    return beta + (L - beta) * (1 - np.exp(-k * t))**delta

def sol_janoschek(beta, L, k, delta, t):
    t_values = np.linspace(0, t, 100)
    N_values = modelo_janoschek(t_values, beta, L, k, delta)
    return t_values, N_values

@pn.depends(beta_jan, L_jan, k_jan, delta_jan, t_exp)
def grafica_janoschek(beta_jan, L_jan, k_jan, delta_jan, t_exp):
    t_values, N_values = sol_janoschek(beta_jan, L_jan, k_jan, delta_jan, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_values, y=N_values, mode='lines', name='Modelo Janoschek'))
    fig.update_layout(
        title='Modelo Janoschek (Crecimiento Exponencial Modificado)',
        xaxis_title='Tiempo',
        yaxis_title='Población',
        yaxis=dict(range=[0, L_jan + 10])  
    )
    return fig

# Carga de CSV
file_input = pn.widgets.FileInput(accept='.csv')
upload_button = pn.widgets.Button(name='Cargar Archivo', button_type='primary')
estado_select = pn.widgets.Select(name='Seleccionar Estado', options=[])
modelo_select = pn.widgets.Select(name='Seleccionar Modelo', options=['Exponencial', 'Logístico', 'Richards', 'Gompertz', 'Bertalanffy-Ivlev', 'Janoschek'], value='Exponencial')

# Widget para seleccionar fecha
date_range_slider = pn.widgets.DateRangeSlider(name='Rango de Fechas', start=datetime(2020, 1, 1), end=datetime(2023, 12, 31), value=(datetime(2020, 1, 1), datetime(2023, 12, 31)))

# Almacenar datos del CSV
df_dengue = None

# Procesamiento de archivo 
def process_file(event):
    global df_dengue
    if file_input.value is not None:
        df_dengue = pd.read_csv(io.BytesIO(file_input.value))
        estados = df_dengue.columns[2:]  # Excluye columnas de año y semana
        estado_select.options = list(estados)
        estado_select.value = estados[0]  # Selecciona el primer estado por defecto

upload_button.on_click(process_file)

# Cálculo acumulado de datos
def acumulado(data):
    return np.cumsum(data)

# Ajuste exponencial
def ajuste_exponencial(estado):
    if df_dengue is None:
        return [], [], [], 0
    data_acumulado = acumulado(df_dengue[estado].dropna().values[:104])  # Usamos acumulado solo para ajuste
    semanas = np.arange(len(data_acumulado))
    
    def funcion_objetivo(params):
        r, N0 = params
        pred = N0 * np.exp(r * semanas)
        return np.sum((data_acumulado - pred) ** 2)

    resultado = minimize(funcion_objetivo, x0=[0.1, data_acumulado[0]], method='Nelder-Mead')
    r_opt, N0_opt = resultado.x
    predicciones = N0_opt * np.exp(r_opt * semanas)
    
    return semanas, data_acumulado, predicciones, r_opt

# Ajuste logístico
def ajuste_logistico(estado):
    if df_dengue is None:
        return [], [], [], 0
    data_acumulado = acumulado(df_dengue[estado].dropna().values[:104])  # Usamos acumulado solo para ajuste
    semanas = np.arange(len(data_acumulado))
    
    def funcion_objetivo(params):
        r, N0, K = params
        pred = K / (1 + ((K - N0) / N0) * np.exp(-r * semanas))
        return np.sum((data_acumulado - pred) ** 2)

    resultado = minimize(funcion_objetivo, x0=[0.1, data_acumulado[0], max(data_acumulado)], method='Nelder-Mead')
    r_opt, N0_opt, K_opt = resultado.x
    predicciones = K_opt / (1 + ((K_opt - N0_opt) / N0_opt) * np.exp(-r_opt * semanas))
    
    return semanas, data_acumulado, predicciones, r_opt

# Modelo de Richards
def ajuste_richards(estado):
    if df_dengue is None:
        return [], [], [], 0
    data_acumulado = acumulado(df_dengue[estado].dropna().values[:104])
    semanas = np.arange(len(data_acumulado))
    
    def funcion_objetivo(params):
        r, K, v = params
        pred = K / (1 + ((K - data_acumulado[0]) / data_acumulado[0]) * np.exp(-r * semanas))**v
        return np.sum((data_acumulado - pred) ** 2)
    
    resultado = minimize(funcion_objetivo, x0=[0.1, max(data_acumulado), 1.0], method='Nelder-Mead')
    r_opt, K_opt, v_opt = resultado.x
    predicciones = K_opt / (1 + ((K_opt - data_acumulado[0]) / data_acumulado[0]) * np.exp(-r_opt * semanas))**v_opt
    
    return semanas, data_acumulado, predicciones, r_opt

# Modelo de Gompertz
def ajuste_gompertz(estado):
    if df_dengue is None:
        return [], [], [], 0
    data_acumulado = acumulado(df_dengue[estado].dropna().values[:104])
    semanas = np.arange(len(data_acumulado))
    
    def funcion_objetivo(params):
        a, K = params
        pred = K * np.exp(-np.exp(-a * (semanas - 10)))  # Parámetro interno fijo para inflexión
        return np.sum((data_acumulado - pred) ** 2)
    
    resultado = minimize(funcion_objetivo, x0=[0.1, max(data_acumulado)], method='Nelder-Mead')
    a_opt, K_opt = resultado.x
    predicciones = K_opt * np.exp(-np.exp(-a_opt * (semanas - 10)))
    
    return semanas, data_acumulado, predicciones, a_opt

# Modelo Bertalanffy-Ivlev
def ajuste_bertalanffy(estado):
    if df_dengue is None:
        return [], [], [], 0
    data_acumulado = acumulado(df_dengue[estado].dropna().values[:104])
    semanas = np.arange(len(data_acumulado))
    
    def funcion_objetivo(params):
        L, K = params
        pred = L * (1 - np.exp(-K * semanas))
        return np.sum((data_acumulado - pred) ** 2)
    
    resultado = minimize(funcion_objetivo, x0=[max(data_acumulado), 0.1], method='Nelder-Mead')
    L_opt, K_opt = resultado.x
    predicciones = L_opt * (1 - np.exp(-K_opt * semanas))
    
    return semanas, data_acumulado, predicciones, K_opt

# Modelo Janoschek
def ajuste_janoschek(estado):
    if df_dengue is None:
        return [], [], [], 0
    data_acumulado = acumulado(df_dengue[estado].dropna().values[:104])
    semanas = np.arange(len(data_acumulado))
    
    def funcion_objetivo(params):
        beta, L, k, delta = params
        pred = beta + (L - beta) * (1 - np.exp(-k * semanas))**delta
        return np.sum((data_acumulado - pred) ** 2)
    
    resultado = minimize(funcion_objetivo, x0=[0, max(data_acumulado), 0.1, 1.0], method='Nelder-Mead')
    beta_opt, L_opt, k_opt, delta_opt = resultado.x
    predicciones = beta_opt + (L_opt - beta_opt) * (1 - np.exp(-k_opt * semanas))**delta_opt
    
    return semanas, data_acumulado, predicciones, k_opt

# Actualización del layout 
@pn.depends(estado=estado_select.param.value, modelo=modelo_select.param.value, date_range=date_range_slider.param.value)
def grafica_datos_y_ajuste(estado, modelo, date_range):
    if df_dengue is None or estado not in df_dengue.columns:
        return pn.pane.HTML("<b style='color:red;'>Por favor, cargue un archivo válido y seleccione un estado</b>")
    
    # Filtrar datos por rango de fechas
    start_date, end_date = date_range
    df_filtered = df_dengue[(df_dengue['Fecha'] >= start_date) & (df_dengue['Fecha'] <= end_date)]
    
    semanas, datos_acumulados, predicciones, parametro = [], [], [], 0
    titulo_ajuste = modelo

    if modelo == 'Exponencial':
        semanas, datos_acumulados, predicciones, parametro = ajuste_exponencial(estado)
    elif modelo == 'Logístico':
        semanas, datos_acumulados, predicciones, parametro = ajuste_logistico(estado)
    elif modelo == 'Richards':
        semanas, datos_acumulados, predicciones, parametro = ajuste_richards(estado)
    elif modelo == 'Gompertz':
        semanas, datos_acumulados, predicciones, parametro = ajuste_gompertz(estado)
    elif modelo == 'Bertalanffy-Ivlev':
        semanas, datos_acumulados, predicciones, parametro = ajuste_bertalanffy(estado)
    elif modelo == 'Janoschek':
        semanas, datos_acumulados, predicciones, parametro = ajuste_janoschek(estado)
    else:
        return pn.pane.HTML("<b style='color:red;'>Seleccione un modelo válido</b>")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(datos_acumulados)), y=datos_acumulados, mode='lines+markers', name='Datos Reales'))
    fig.add_trace(go.Scatter(x=semanas, y=predicciones, mode='lines', name=f'Ajuste {titulo_ajuste}'))
    fig.update_layout(
        title=f'Ajuste {titulo_ajuste} para {estado}',
        xaxis_title='Semana',
        yaxis_title='Casos'
    )
    
    return fig


descripcion_exponencial = pn.Card(
    pn.Column(
        pn.pane.Markdown(r"""
        **Descripción:**  
        El modelo exponencial describe el crecimiento de una población en condiciones ideales, donde no hay limitaciones de recursos. 
        La tasa de crecimiento es constante y proporcional al tamaño de la población. 
        **Parámetros:**  
        - \( r \): Tasa de crecimiento inicial (rango: 0.1 a 1.0).  
        - \( N \): Población inicial (rango: 10 a 100).  
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
    El modelo logístico describe el crecimiento de una población en condiciones más realistas, donde hay una capacidad de carga máxima (K) que limita el crecimiento.
    La tasa de crecimiento disminuye a medida que la población se acerca a la capacidad de carga.
    **Parámetros:**  
    - \( r \): Tasa de crecimiento logístico (rango: 0.1 a 1.0).  
    - \( K \): Capacidad de carga (rango: 50 a 500).  
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
    El modelo SIR es un modelo epidemiológico que divide la población en tres grupos: Susceptibles (S), Infectados (I) y Recuperados (R).
    El modelo describe cómo una enfermedad se propaga a través de la población y cómo los individuos se recuperan.
    **Parámetros:**
    - \( \beta \): Tasa de infección (rango: 0.1 a 1.0).  
    - \( \gamma \): Tasa de recuperación (rango: 0.05 a 0.5).  
    **Ecuaciones:**"""),
    render_ecuacion(r"[ \frac{dS}{dt} = -\beta \frac{SI}{N}]"),
    render_ecuacion(r"[ \frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I]"),
    render_ecuacion(r"[ \frac{dR}{dt} = \gamma I]"), 
    ),
    title="Modelo SIR",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

descripcion_richards = pn.Card(
    pn.Column(
    pn.pane.Markdown(r""" 
    **Descripción:**  
    El modelo de Richards es una generalización del modelo logístico que permite una mayor flexibilidad en la forma de la curva de crecimiento.
    Incluye un parámetro de forma (v) que controla la asimetría de la curva.
    **Parámetros:**  
    - \( r \): Tasa de crecimiento (rango: 0.1 a 1.0).  
    - \( K \): Capacidad de carga (rango: 50 a 500).  
    - \( v \): Parámetro de forma (rango: 0.1 a 5.0). 
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
    El modelo de Gompertz describe el crecimiento de una población donde la tasa de crecimiento disminuye exponencialmente con el tiempo.
    Es útil para modelar el crecimiento de tumores y otros fenómenos biológicos.
    **Parámetros:**  
    - \( a \): Tasa de crecimiento (rango: 0.01 a 1.0).  
    - \( K \): Capacidad de carga (rango: 50 a 500). 
    **Ecuación:**"""),
    render_ecuacion(r"[\frac{dN}{dt} = aN \ln\left(\frac{K}{N}\right)]"),
    ),
    title="Modelo de Gompertz",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

descripcion_bertalanffy = pn.Card(
    pn.Column(
    pn.pane.Markdown(r"""  
    **Descripción:**  
    El modelo Bertalanffy-Ivlev describe el crecimiento de una población donde la tasa de crecimiento depende de la diferencia entre el tamaño actual y el tamaño límite (L).
    Es comúnmente utilizado en ecología para modelar el crecimiento de peces y otros organismos.
    **Parámetros:**  
    - \( L \): Tamaño límite (rango: 50 a 500).  
    - \( K \): Coeficiente de crecimiento (rango: 0.01 a 1.0).
    **Ecuación:** """),
    render_ecuacion(r"[\frac{dN}{dt} = L \left(1 - e^{-Kt}\right)]"),
    ),
    title="Modelo Bertalanffy-Ivlev",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)

descripcion_janoschek = pn.Card(
    pn.Column(
    pn.pane.Markdown(r"""  
    **Descripción:**  
    El modelo Janoschek es un modelo de crecimiento que permite una asintota inferior (β) y una superior (L), con una tasa de crecimiento (k) y un parámetro de forma (δ).
    Es útil para modelar fenómenos donde el crecimiento no comienza desde cero o no alcanza un límite superior.
    **Parámetros:**  
    - \( \beta \): Asintota inferior (rango: 0 a 50).  
    - \( L \): Asintota superior (rango: 50 a 500).  
    - \( k \): Tasa de crecimiento (rango: 0.01 a 1.0).  
    - \( \delta \): Parámetro de forma (rango: 0.5 a 5.0).
    **Ecuación:** """),
    render_ecuacion(r"[ N(t) = \beta + (L - \beta) \left(1 - e^{-kt}\right)^\delta]"),
    ),
    title="Modelo Janoschek",
    styles={"background": "#f9f9f9", "border": "1px solid #ddd", "border-radius": "5px"},
    width=300
)


#tab de menu para el estudio teorico, tipo dropdown 
# No need to change the function definitions - they're already correct
# Just update the model_tabs_content dictionary:

model_tabs_content = {
    'Modelo Exponencial': pn.Row(
        descripcion_exponencial, 
        pn.Column(grafica_exponencial, r_exp, N0_exp, t_exp)
    ),
    'Modelo Logístico': pn.Row(
        descripcion_logistico, 
        pn.Column(grafica_logistico, K_log, r_log, N0_exp, t_exp)
    ),
    'Modelo SIR': pn.Row(
        descripcion_sir, 
        pn.Column(grafica_sir, beta_sir, gamma_sir, t_sir)
    ),
    'Modelo de Richards': pn.Row(
        descripcion_richards, 
        pn.Column(grafica_richards, v_richards, N0_exp, t_exp)
    ),
    'Modelo de Gompertz': pn.Row(
        descripcion_gompertz, 
        pn.Column(grafica_gompertz, a_gompertz, K_gompertz, N0_exp, t_exp)
    ),
    'Modelo Bertalanffy-Ivlev': pn.Row(
        descripcion_bertalanffy, 
        pn.Column(grafica_bertalanffy, L_bi, K_bi, t_exp)
    ),
    'Modelo Janoschek': pn.Row(
        descripcion_janoschek, 
        pn.Column(grafica_janoschek, beta_jan, L_jan, k_jan, delta_jan, t_exp)
    )
}

selected_tab_content = model_tabs_content['Modelo Exponencial']

#callback para el contenido 
def update_tab(event):
    global selected_tab_content
    selected_tab_content = model_tabs_content[event.new]
    dynamic_area.objects = [selected_tab_content]

menu = pn.widgets.Select(options=list(model_tabs_content.keys()))

menu.param.watch(lambda event: update_tab(event), 'value')

dynamic_area = pn.Column(selected_tab_content)

# Pestaña de Análisis Teórico
teorico_tab = pn.Column(
    pn.Row( 
        pn.layout.HSpacer(),
        menu,
        pn.layout.HSpacer()
    ),
    dynamic_area
)

# Pestaña de Análisis Estadístico
estadistico_tab = pn.Column(
    pn.layout.HSpacer(),
    pn.Row(
        pn.layout.VSpacer(),
        pn.Column(
            file_input,
            upload_button,
            estado_select,
            modelo_select,
            date_range_slider,
            grafica_datos_y_ajuste,
            css_classes=["center-content"]
        ),
        pn.layout.VSpacer()
    ),
    pn.layout.HSpacer()
)

secciones = pn.Tabs(
    ('Estudio Teórico', teorico_tab),
    ('Estudio Estadístico', estadistico_tab),
    css_classes=["custom-tabs"],  
    sizing_mode="stretch_width"
)

template = pn.template.MaterialTemplate(title='Análisis de Modelos Epidemiológicos')

template.main.append(secciones)


 
template.show()