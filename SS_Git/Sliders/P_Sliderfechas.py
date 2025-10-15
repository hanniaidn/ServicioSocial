import panel as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, odeint
import io

pn.extension('plotly')

# Título
titulo = pn.pane.Markdown("""
# <center style='color:#4CAF50; font-size:40px;'>Análisis de Modelos Epidemiológicos </center>
""", width=800)

# --------- MODELOS TEÓRICOS ---------

# Parámetros generales
r_exp = pn.widgets.FloatSlider(name='Tasa de crecimiento', start=0.1, end=1.0, step=0.01, value=0.3)
N0_exp = pn.widgets.FloatSlider(name='Población inicial', start=10, end=100, step=1, value=50)
t_exp = pn.widgets.IntSlider(name='Tiempo', start=0, end=50, step=1, value=25)

def modelo_exponencial(t, N, r): return r * N

def sol_exp(N0, r, t):
    t_vals = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_exponencial(t, N, r), [0, t], [N0], t_eval=t_vals)
    return sol.t, sol.y[0]

@pn.depends(r_exp, N0_exp, t_exp)
def grafica_exponencial(r_exp, N0_exp, t_exp):
    t, N = sol_exp(N0_exp, r_exp, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=N, mode='lines', name='Exponencial'))
    return fig

# Modelo logístico
r_log = pn.widgets.FloatSlider(name='Tasa crecimiento', start=0.1, end=1.0, step=0.01, value=0.3)
K_log = pn.widgets.FloatSlider(name='Capacidad de carga', start=50, end=500, step=10, value=200)

def modelo_logistico(t, N, r, K): return r * N * (1 - N / K)

def sol_log(N0, r, K, t):
    t_vals = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_logistico(t, N, r, K), [0, t], [N0], t_eval=t_vals)
    return sol.t, sol.y[0]

@pn.depends(N0_exp, r_log, K_log, t_exp)
def grafica_logistico(N0_exp, r_log, K_log, t_exp):
    t, N = sol_log(N0_exp, r_log, K_log, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=N, mode='lines', name='Logístico'))
    return fig

# Richards
v_richards = pn.widgets.FloatSlider(name='Parámetro v', start=0.1, end=5.0, step=0.1, value=1.0)

def modelo_richards(t, N, r, K, v): return r * N * (1 - (N / K)**v)

def sol_richards(N0, r, K, v, t):
    t_vals = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_richards(t, N, r, K, v), [0, t], [N0], t_eval=t_vals)
    return sol.t, sol.y[0]

@pn.depends(N0_exp, r_log, K_log, v_richards, t_exp)
def grafica_richards(N0_exp, r_log, K_log, v_richards, t_exp):
    t, N = sol_richards(N0_exp, r_log, K_log, v_richards, t_exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=N, mode='lines', name='Richards'))
    return fig

# SIR
beta_sir = pn.widgets.FloatSlider(name='β', start=0.1, end=1.0, step=0.01, value=0.3)
gamma_sir = pn.widgets.FloatSlider(name='γ', start=0.05, end=0.5, step=0.01, value=0.1)
t_sir = pn.widgets.IntSlider(name='Tiempo', start=0, end=100, step=1, value=50)
N_sir = 10000; I0_sir = 150; R0_sir = 0

def modelo_sir(y, t, N, beta, gamma):
    S, I, R = y
    return -beta*S*I/N, beta*S*I/N - gamma*I, gamma*I

def sol_sir(beta, gamma, N, I0, R0, t):
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t_vals = np.linspace(0, t, 100)
    res = odeint(modelo_sir, y0, t_vals, args=(N, beta, gamma))
    return t_vals, res.T

@pn.depends(beta_sir, gamma_sir, t_sir)
def grafica_sir(beta_sir, gamma_sir, t_sir):
    t, (S, I, R) = sol_sir(beta_sir, gamma_sir, N_sir, I0_sir, R0_sir, t_sir)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptibles'))
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infectados'))
    fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recuperados'))
    return fig

# Teórico tabs
teorico_tab = pn.Tabs(
    ('Exponencial', pn.Column(r_exp, N0_exp, t_exp, grafica_exponencial)),
    ('Logístico', pn.Column(K_log, r_log, t_exp, grafica_logistico)),
    ('Richards', pn.Column(v_richards, t_exp, grafica_richards)),
    ('SIR', pn.Column(beta_sir, gamma_sir, t_sir, grafica_sir))
)

# --------- ESTADÍSTICO ---------

file_input    = pn.widgets.FileInput(accept='.csv')
upload_button = pn.widgets.Button(name='Cargar Archivo', button_type='primary')
estado_select = pn.widgets.Select(name='Seleccionar Estado', options=[])
selector_comparativa = pn.widgets.MultiChoice(name='Modelos a comparar',
    options=['Exponencial', 'Logístico', 'Richards', 'Gompertz', 'Bertalanffy-Ivlev', 'Janoschek'],
    value=['Exponencial']
)
fecha_slider = pn.widgets.DiscreteSlider(name='Seleccionar fecha inicial', options=[])

leyenda = pn.pane.Markdown("""
### Instrucciones para cargar archivo CSV:
Por favor carga un archivo **.csv** que contenga el siguiente formato:
- La primera columna debe ser el **año**
- La segunda columna debe ser la **fecha** en formato **dd/mm/aaaa** o **aaaa-mm-dd**
- Las columnas restantes deben ser los nombres de los estados con los casos semanales
""", width=600)

df_dengue = None
fechas_validas = []

def process_file(event):
    global df_dengue, fechas_validas
    df_dengue = pd.read_csv(io.BytesIO(file_input.value))
    df_dengue.iloc[:, 0] = pd.to_datetime(df_dengue.iloc[:, 0], dayfirst=True)
    fechas_validas = df_dengue.iloc[:, 0].dt.strftime('%Y-%m-%d').tolist()
    fecha_slider.options = fechas_validas
    estado_select.options = list(df_dengue.columns[2:])
    
    #probar que cada instrucción este haciendo lo que debe
    

upload_button.on_click(process_file)

def acumulado(data): return np.cumsum(data)

def ajuste_modelo(modelo, estado, fecha_inicio):
    if df_dengue is None: return [], [], [], 0
    df_filtrado = df_dengue[df_dengue.iloc[:, 1] >= pd.to_datetime(fecha_inicio)]
    fechas = df_filtrado.iloc[:, 1].dt.strftime('%Y-%m-%d').values
    datos = acumulado(df_filtrado[estado].dropna().values)
    t = np.arange(len(datos))
    if len(t) == 0: return [], [], [], 0
    if modelo == 'Exponencial':
        f = lambda p: np.sum((datos - p[1]*np.exp(p[0]*t))**2)
        r, N0 = minimize(f, [0.1, datos[0]]).x
        pred = N0 * np.exp(r * t)
    elif modelo == 'Logístico':
        f = lambda p: np.sum((datos - p[2]/(1+((p[2]-p[1])/p[1])*np.exp(-p[0]*t)))**2)
        r, N0, K = minimize(f, [0.1, datos[0], max(datos)]).x
        pred = K / (1 + ((K-N0)/N0)*np.exp(-r*t))
    elif modelo == 'Richards':
        f = lambda p: np.sum((datos - p[1]/(1+((p[1]-datos[0])/datos[0])*np.exp(-p[0]*t))**p[2])**2)
        r, K, v = minimize(f, [0.1, max(datos), 1.0]).x
        pred = K / (1 + ((K-datos[0])/datos[0])*np.exp(-r*t))**v
    elif modelo == 'Gompertz':
        f = lambda p: np.sum((datos - p[1]*np.exp(-np.exp(-p[0]*(t-10))))**2)
        a, K = minimize(f, [0.1, max(datos)]).x
        pred = K * np.exp(-np.exp(-a*(t-10)))
    elif modelo == 'Bertalanffy-Ivlev':
        f = lambda p: np.sum((datos - p[0]*(1 - np.exp(-p[1]*t)))**2)
        L, K = minimize(f, [max(datos), 0.1]).x
        pred = L * (1 - np.exp(-K * t))
    elif modelo == 'Janoschek':
        f = lambda p: np.sum((datos - (p[0] + (p[1]-p[0])*(1-np.exp(-p[2]*t))**p[3]))**2)
        b, L, k, d = minimize(f, [0, max(datos), 0.1, 1.0]).x
        pred = b + (L - b) * (1 - np.exp(-k*t))**d
    else: return [], [], [], 0
    return fechas[:len(pred)], datos[:len(pred)], pred, 0

colores = {
    'Exponencial': 'blue',
    'Logístico': 'green',
    'Richards': 'orange',
    'Gompertz': 'purple',
    'Bertalanffy-Ivlev': 'brown',
    'Janoschek': 'red'
}

@pn.depends(estado_select, selector_comparativa, fecha_slider)
def grafica_comparativa(estado, modelos_sel, fecha_ini):
    if df_dengue is None or estado not in df_dengue.columns or not fecha_ini:
        return pn.pane.HTML("<b style='color:red;'>Cargue un archivo y seleccione un estado y una fecha</b>")
    fig = go.Figure()
    fechas, datos, _, _ = ajuste_modelo('Exponencial', estado, fecha_ini)
    fig.add_trace(go.Scatter(x=fechas, y=datos, mode='lines+markers', name='Datos Reales', line=dict(color='black')))
    for modelo in modelos_sel:
        fechas, _, pred, _ = ajuste_modelo(modelo, estado, fecha_ini)
        fig.add_trace(go.Scatter(x=fechas, y=pred, mode='lines', name=modelo, line=dict(color=colores[modelo])))
    fig.update_layout(title=f'Comparativa de modelos: {estado}', xaxis_title='Fecha', yaxis_title='Casos acumulados')
    return fig

estadistico_tab = pn.Column(
    leyenda,
    file_input,
    upload_button,
    estado_select,
    fecha_slider,
    selector_comparativa,
    grafica_comparativa
)

# --- DASHBOARD COMPLETO ---
dashboard = pn.Column(
    titulo,
    pn.Tabs(
        ('Estudio Teórico', teorico_tab),
        ('Estudio Estadístico', estadistico_tab)
    )
)

dashboard.show()
