import panel as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize
import io

pn.extension('plotly')

# titulo
titulo = pn.pane.Markdown("""
# <center style='color:#4CAF50; font-size:40px;'>Análisis de Modelos Epidemiológicos </center>
""", width=800)

# modelo exponencial
r_exp = pn.widgets.FloatSlider(name='Tasa de crecimiento inicial', start=0.1, end=1.0, step=0.01, value=0.3)
N0_exp = pn.widgets.FloatSlider(name='Población inicial', start=10, end=100, step=1, value=50)
t_exp = pn.widgets.IntSlider(name='Tiempo', start=0, end=50, step=1, value=25)

# Modelo Exponencial
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

# modelo de Gompertz
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

# modelo Janoschek
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


# --- Widgets ---
file_input    = pn.widgets.FileInput(accept='.csv')
upload_button = pn.widgets.Button(name='Cargar Archivo', button_type='primary')
estado_select = pn.widgets.Select(name='Seleccionar Estado', options=[])
modelo_select = pn.widgets.Select(
    name='Seleccionar Modelo',
    options=['Exponencial', 'Logístico', 'Richards', 'Gompertz',
             'Bertalanffy-Ivlev', 'Janoschek'],
    value='Exponencial'
)

# Variable global para datos
df_dengue = None

# --- Cargar archivo ---
def process_file(event):
    global df_dengue
    df_dengue = pd.read_csv(io.BytesIO(file_input.value))
    estados = df_dengue.columns[2:]  # ignora año y semana
    estado_select.options = list(estados)

upload_button.on_click(process_file)

# --- Funciones de acumulado y ajustes ---
def acumulado(data): return np.cumsum(data)

# ajuste exponencial
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

# ajuste logistico
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

# modelo de Richards
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

# modelo de Gompertz
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

# modelo Bertalanffy-Ivlev
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

# modelo Janoschek
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

# --- Callback corregido ---
@pn.depends(estado_select.param.value, modelo_select.param.value)
def grafica_datos_y_ajuste(estado, modelo):
    if df_dengue is None or estado not in df_dengue.columns:
        return pn.pane.HTML("<b style='color:red;'>Cargue un CSV y seleccione un estado</b>")

    if modelo == 'Exponencial':
        semanas, datos, pred, _ = ajuste_exponencial(estado)
    elif modelo == 'Logístico':
        semanas, datos, pred, _ = ajuste_logistico(estado)
    elif modelo == 'Richards':
        semanas, datos, pred, _ = ajuste_richards(estado)
    elif modelo == 'Gompertz':
        semanas, datos, pred, _ = ajuste_gompertz(estado)
    elif modelo == 'Bertalanffy-Ivlev':
        semanas, datos, pred, _ = ajuste_bertalanffy(estado)
    elif modelo == 'Janoschek':
        semanas, datos, pred, _ = ajuste_janoschek(estado)
    else:
        return pn.pane.HTML("<b style='color:red;'>Modelo no válido</b>")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=semanas, y=datos, mode='lines+markers', name='Datos Reales'))
    fig.add_trace(go.Scatter(x=semanas, y=pred, mode='lines', name=f'Ajuste {modelo}'))
    fig.update_layout(title=f'Ajuste {modelo} para {estado}',
                      xaxis_title='Semana', yaxis_title='Casos')
    return fig

# --- Layout Estudio Estadístico ---
estadistico_tab = pn.Column(
    file_input,
    upload_button,
    estado_select,
    modelo_select,              # <--- un único selector aquí
    grafica_datos_y_ajuste      # <--- depende de selector y estado
)

# --- Layout completo ---
teorico_tab = pn.Tabs(
    ('Logístico', pn.Column(K_log, r_log, grafica_logistico)),
    ('Exponencial', pn.Column(r_exp, N0_exp, t_exp, grafica_exponencial)),
    ('SIR', pn.Column(beta_sir, gamma_sir, t_sir, grafica_sir)),
    ('Richards', pn.Column(v_richards, grafica_richards)),
    ('Gompertz', pn.Column(a_gompertz, K_gompertz, grafica_gompertz)),
    ('Bertalanffy-Ivlev', pn.Column(L_bi, K_bi, grafica_bertalanffy)),
    ('Janoschek', pn.Column(beta_jan, L_jan, k_jan, delta_jan, grafica_janoschek))
)

dashboard = pn.Column(
    pn.pane.Markdown("<center style='color:#4CAF50; font-size:40px;'>Análisis de Modelos Epidemiológicos</center>", width=800),
    pn.Tabs(
        ('Estudio Teórico', teorico_tab),
        ('Estudio Estadístico', estadistico_tab)
    )
)

dashboard.show()
