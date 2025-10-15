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
        title='Modelo Janoschek (Crecimiento Exponencial Modificado)',
        xaxis_title='Tiempo',
        yaxis_title='Población',
        yaxis=dict(range=[0, L_jan + 10])  
    )
    return fig


# TAB ESTADISTICA !!!! 

file_input    = pn.widgets.FileInput(accept='.csv')
upload_button = pn.widgets.Button(name='Cargar Archivo', button_type='primary')
estado_select = pn.widgets.Select(name='Seleccionar Estado', options=[])
modelo_select = pn.widgets.Select(
    name='Seleccionar Modelo',
    options=['Exponencial', 'Logístico', 'Richards', 'Gompertz',
             'Bertalanffy-Ivlev', 'Janoschek'],
    value='Exponencial'
)

date_slider = pn.widgets.DatetimeRangeSlider(
    name='Selecciona el periodo a graficar',
    start=pd.Timestamp('2000-01-01'),  
    end=pd.Timestamp('2000-12-31'),   
    value=(pd.Timestamp('2000-01-01'), pd.Timestamp('2000-12-31')),
    step=86400000,  
    width=400
)

forecast_slider = pn.widgets.IntSlider(
    name='Días de pronóstico',
    start=0,
    end=180,
    step=1,
    value=30,
    width=280
)

model_colors = {
    'Exponencial': 'blue',
    'Logístico': 'green',
    'Richards': 'red',
    'Gompertz': 'purple',
    'Bertalanffy-Ivlev': 'orange',
    'Janoschek': 'brown'
}

# Variable global para datos
df_dengue = None

#  Cargar archivo 
def process_file(event):
    global df_dengue
    df_dengue = pd.read_csv(io.BytesIO(file_input.value), parse_dates=[0], dayfirst=True)
    estados = df_dengue.columns[1:]  
    estado_select.options = list(estados)
    
    # deja el slider de fecha en el rango del csv 
    date_slider.start = df_dengue.iloc[:, 0].min()
    date_slider.end = df_dengue.iloc[:, 0].max()
    date_slider.value = (date_slider.start, date_slider.end)

upload_button.on_click(process_file)

# Funciones de acumulado y ajustes 
def acumulado(data): return np.cumsum(data)

#  AJUSTES 
def _x_real(filtered_df):
    return (filtered_df.iloc[:, 0] - filtered_df.iloc[:, 0].iloc[0]).dt.days.values

def ajuste_exponencial(estado, date_range, forecast_days):
    if df_dengue is None:
        return [], [], [], 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    df = df_dengue.loc[mask]
    y = acumulado(df[estado].dropna().values)
    x = _x_real(df)[:len(y)]
    N0 = y[0]

    def obj(p):
        r = p[0]
        return np.sum((N0 * np.exp(r * x) - y) ** 2)

    r_opt = minimize(obj, [0.1], method='Nelder-Mead').x[0]
    x_future = np.arange(x[0], x[-1] + forecast_days + 1)
    pred = N0 * np.exp(r_opt * x_future)
    return x, y, pred, r_opt

def ajuste_logistico(estado, date_range, forecast_days):
    if df_dengue is None:
        return [], [], [], 0, 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    df = df_dengue.loc[mask]
    y = acumulado(df[estado].dropna().values)
    x = _x_real(df)[:len(y)]
    N0 = y[0]

    def obj(p):
        r, K = p
        return np.sum((K / (1 + ((K - N0) / N0) * np.exp(-r * x)) - y) ** 2)

    r_opt, K_opt = minimize(obj, [0.1, max(y)], method='Nelder-Mead').x
    x_future = np.arange(x[0], x[-1] + forecast_days + 1)
    pred = K_opt / (1 + ((K_opt - N0) / N0) * np.exp(-r_opt * x_future))
    return x, y, pred, r_opt, K_opt

def ajuste_richards(estado, date_range, forecast_days):
    if df_dengue is None:
        return [], [], [], 0, 0, 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    df = df_dengue.loc[mask]
    y = acumulado(df[estado].dropna().values)
    x = _x_real(df)[:len(y)]
    N0 = y[0]

    def obj(p):
        r, K, v = p
        return np.sum((K / (1 + ((K - N0) / N0) * np.exp(-r * x)) ** v - y) ** 2)

    r_opt, K_opt, v_opt = minimize(obj, [0.1, max(y), 1.0], method='Nelder-Mead').x
    x_future = np.arange(x[0], x[-1] + forecast_days + 1)
    pred = K_opt / (1 + ((K_opt - N0) / N0) * np.exp(-r_opt * x_future)) ** v_opt
    return x, y, pred, r_opt, K_opt, v_opt

def ajuste_gompertz(estado, date_range, forecast_days):
    if df_dengue is None:
        return [], [], [], 0, 0, 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    df = df_dengue.loc[mask]
    y = acumulado(df[estado].dropna().values)
    x = _x_real(df)[:len(y)]
    N0 = y[0]                               
    t0 = x[0]                                

    def obj(p):
        K, a = p
        pred = K * (N0 / K) ** np.exp(-a * (x - t0))
        return np.sum((pred - y) ** 2)

    K0 = max(y) * 1.2
    a0 = 0.05
    res = minimize(obj, [K0, a0], method='Nelder-Mead')
    K_opt, a_opt = res.x
    x_future = np.arange(x[0], x[-1] + forecast_days + 1)
    pred = K_opt * (N0 / K_opt) ** np.exp(-a_opt * (x_future - t0))
    return x, y, pred, a_opt, K_opt     

def ajuste_bertalanffy(estado, date_range, forecast_days):
    if df_dengue is None:
        return [], [], [], 0, 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    df = df_dengue.loc[mask]
    y = acumulado(df[estado].dropna().values)
    x = _x_real(df)[:len(y)]

    def obj(p):
        L, K = p
        return np.sum((L * (1 - np.exp(-K * x)) - y) ** 2)

    L_opt, K_opt = minimize(obj, [max(y), 0.1], method='Nelder-Mead').x
    x_future = np.arange(x[0], x[-1] + forecast_days + 1)
    pred = L_opt * (1 - np.exp(-K_opt * x_future))
    return x, y, pred, L_opt, K_opt

def ajuste_janoschek(estado, date_range, forecast_days):
    if df_dengue is None:
        return [], [], [], 0, 0, 0, 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    df = df_dengue.loc[mask]
    y = acumulado(df[estado].dropna().values)
    x = _x_real(df)[:len(y)]

    def obj(p):
        beta, L, k, delta = p
        return np.sum((beta + (L - beta) * (1 - np.exp(-k * x)) ** delta - y) ** 2)

    beta_opt, L_opt, k_opt, delta_opt = minimize(
        obj, [0, max(y), 0.1, 1.0], method='Nelder-Mead').x
    x_future = np.arange(x[0], x[-1] + forecast_days + 1)
    pred = beta_opt + (L_opt - beta_opt) * (1 - np.exp(-k_opt * x_future)) ** delta_opt
    return x, y, pred, beta_opt, L_opt, k_opt, delta_opt

modelos_disponibles = ['Exponencial', 'Logístico', 'Richards', 'Gompertz', 'Bertalanffy-Ivlev', 'Janoschek']
selector_comparativa = pn.widgets.MultiChoice(
    name='Modelos a comparar', 
    options=modelos_disponibles, 
    value=[],  
    width=300
)

selector_comparativa.param.watch(
    lambda e: (
        grafica_comparativa(
            estado_select.value,
            date_slider.value,
            e.new,
            forecast_slider.value
        ),
        actualiza_tabla(
            estado_select.value,
            date_slider.value,
            e.new,
            forecast_slider.value
        )
    ), 'value'
)

forecast_slider.param.watch(
    lambda e: (
        grafica_comparativa(
            estado_select.value,
            date_slider.value,
            selector_comparativa.value or [],
            e.new
        ),
        actualiza_tabla(
            estado_select.value,
            date_slider.value,
            selector_comparativa.value or [],
            e.new
        )
    ), 'value'
)

estado_select.param.watch(
    lambda e: (
        grafica_comparativa(
            e.new,
            date_slider.value,
            selector_comparativa.value or [],
            forecast_slider.value
        ),
        actualiza_tabla(
            e.new,
            date_slider.value,
            selector_comparativa.value or [],
            forecast_slider.value
        )
    ), 'value'
)

date_slider.param.watch(
    lambda e: (
        grafica_comparativa(
            estado_select.value,
            e.new,
            selector_comparativa.value or [],
            forecast_slider.value
        ),
        actualiza_tabla(
            estado_select.value,
            e.new,
            selector_comparativa.value or [],
            forecast_slider.value
        )
    ), 'value'
)

def calcula_ajuste(modelo, estado, date_range, forecast_days):
    if modelo == 'Exponencial':
        x, y, pred, r = ajuste_exponencial(estado, date_range, forecast_days)
        return x, y, pred, (r,)
    elif modelo == 'Logístico':
        x, y, pred, r, K = ajuste_logistico(estado, date_range, forecast_days)
        return x, y, pred, (r, K)
    elif modelo == 'Richards':
        x, y, pred, r, K, v = ajuste_richards(estado, date_range, forecast_days)
        return x, y, pred, (r, K, v)
    elif modelo == 'Gompertz':
        x, y, pred, a, K = ajuste_gompertz(estado, date_range, forecast_days)
        return x, y, pred, (a, K)
    elif modelo == 'Bertalanffy-Ivlev':
        x, y, pred, L, K = ajuste_bertalanffy(estado, date_range, forecast_days)
        return x, y, pred, (L, K)
    elif modelo == 'Janoschek':
        x, y, pred, beta, L, k, delta = ajuste_janoschek(estado, date_range, forecast_days)
        return x, y, pred, (beta, L, k, delta)
    else:
        return [], [], [], ()

@pn.depends(estado_select.param.value, modelo_select.param.value, date_slider.param.value, forecast_slider.param.value)
def grafica_datos_y_ajuste(estado, modelo, date_range, forecast_days):
    if df_dengue is None or estado not in df_dengue.columns:
        return pn.pane.HTML("<b style='color:red;'>Cargue un CSV y seleccione un estado</b>")

    x, y, pred, *_ = calcula_ajuste(modelo, estado, date_range, forecast_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Datos Reales'))
    fig.add_trace(go.Scatter(x=x, y=pred[:len(x)], mode='lines', name=f'Ajuste {modelo}'))
    fig.update_layout(title=f'Ajuste {modelo} para {estado}',
                      xaxis_title='Días desde inicio', yaxis_title='Casos')
    return fig

@pn.depends(estado_select.param.value, date_slider.param.value, selector_comparativa.param.value, forecast_slider.param.value)
def grafica_comparativa(estado, date_range, modelos_sel, forecast_days):
    if df_dengue is None or estado not in df_dengue.columns:
        return pn.pane.HTML("<b style='color:red;'>Cargue un CSV y seleccione un estado</b>")

    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    filtered_df = df_dengue.loc[mask]
    semanas = filtered_df.iloc[:, 0]
    datos_reales = acumulado(filtered_df[estado].dropna().values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=semanas, y=datos_reales,
                             mode='markers+lines', name='Datos Reales', line=dict(color='black')))

    for modelo in modelos_sel:
        x, y, pred, *_ = calcula_ajuste(modelo, estado, date_range, forecast_days)
        extended_semanas = pd.date_range(start=semanas.min(), periods=len(pred), freq='D')
        fig.add_trace(go.Scatter(x=extended_semanas, y=pred,
                                 mode='lines', name=f'Ajuste {modelo}',
                                 line=dict(color=model_colors[modelo])))

    fig.update_layout(title=f'Comparativa modelos para {estado}',
                      xaxis_title='Fecha', yaxis_title='Casos')
    return fig

#tabla de parametros estadisticos 
param_table = pn.widgets.DataFrame(
    pd.DataFrame(columns=['Modelo', 'Parámetros']),
    name='Parámetros estimados',
    width=600,
    height=220,
    sizing_mode='stretch_width',
    show_index=False
)

@pn.depends(estado_select.param.value,date_slider.param.value,selector_comparativa.param.value,forecast_slider.param.value)
def actualiza_tabla(estado, date_range, modelos_sel, forecast_days):
    if df_dengue is None or estado not in df_dengue.columns or not modelos_sel:
        param_table.value = pd.DataFrame(columns=['Modelo', 'Parámetros'])
        return

    filas = []
    for modelo in modelos_sel:
        try:
            if modelo == 'Exponencial':
                _, _, _, r = ajuste_exponencial(estado, date_range, forecast_days)
                filas.append({'Modelo': modelo, 'Parámetros': f"r = {r:.4f}"})
            elif modelo == 'Logístico':
                _, _, _, r, K = ajuste_logistico(estado, date_range, forecast_days)
                filas.append({'Modelo': modelo, 'Parámetros': f"r = {r:.4f}, K = {K:.4f}"})
            elif modelo == 'Richards':
                _, _, _, r, K, v = ajuste_richards(estado, date_range, forecast_days)
                filas.append({'Modelo': modelo, 'Parámetros': f"r = {r:.4f}, K = {K:.0f}, v = {v:.2f}"})
            elif modelo == 'Gompertz':
                _, _, _, a, K = ajuste_gompertz(estado, date_range, forecast_days)
                filas.append({'Modelo': modelo, 'Parámetros': f"a = {a:.4f}, K = {K:.4f}"})
            elif modelo == 'Bertalanffy-Ivlev':
                _, _, _, L, K = ajuste_bertalanffy(estado, date_range, forecast_days)
                filas.append({'Modelo': modelo, 'Parámetros': f"L = {L:.4f}, K = {K:.4f}"})
            elif modelo == 'Janoschek':
                _, _, _, beta, L, k, delta = ajuste_janoschek(estado, date_range, forecast_days)
                filas.append({'Modelo': modelo, 'Parámetros': f"β = {beta:.4f}, L = {L:.4f}, k = {k:.4f}, δ = {delta:.2f}"})
        except Exception as e:
            filas.append({'Modelo': modelo, 'Parámetros': 'Error en cálculo'})

    param_table.value = pd.DataFrame(filas)

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
    El modelo logístico describe el crecimiento de una población en condiciones más realistas, donde hay una capacidad de carga máxima (K) que limita el crecimiento.
    La tasa de crecimiento disminuye a medida que la población se acerca a la capacidad de carga.
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
    El modelo SIR es un modelo epidemiológico que divide la población en tres grupos: Susceptibles (S), Infectados (I) y Recuperados (R).
    El modelo describe cómo una enfermedad se propaga a través de la población y cómo los individuos se recuperan.
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
    El modelo de Richards es una generalización del modelo logístico que permite una mayor flexibilidad en la forma de la curva de crecimiento.
    Incluye un parámetro de forma (v) que controla la asimetría de la curva.
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
    El modelo de Gompertz describe el crecimiento de una población donde la tasa de crecimiento disminuye exponencialmente con el tiempo.
    Es útil para modelar el crecimiento de tumores y otros fenómenos biológicos.
    **Parámetros:**  
    - a: Tasa de crecimiento (rango: 0.01 a 1.0).  
    - K: Capacidad de carga (rango: 50 a 500). 
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
    - L: Tamaño límite (rango: 50 a 500).  
    - K: Coeficiente de crecimiento (rango: 0.01 a 1.0).
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
    'Modelo Bertalanffy-Ivlev': pn.Row(descripcion_bertalanffy, pn.Column(grafica_bertalanffy, L_bi, K_bi, N0_bi, t_exp)),
    'Modelo Janoschek': pn.Row(descripcion_janoschek, pn.Column(grafica_janoschek, beta_jan, L_jan, k_jan, delta_jan, N0_jan, t_exp))
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

# Layout Estadístico
control_panel = pn.Column(
    file_input,
    upload_button,
    estado_select,
    date_slider,
    forecast_slider,      
    selector_comparativa,
    sizing_mode='stretch_width',
    width=350,
    margin=(10, 10, 10, 10)
)

content_area = pn.Column(
    sizing_mode='stretch_width',
    margin=(0, 10, 0, 0)
)

# estadistico_tab
estadistico_tab = pn.Row(
    pn.layout.HSpacer(width=20),  
    pn.Column(
        pn.layout.VSpacer(height=10),  
        control_panel,
        content_area,
        sizing_mode='stretch_width'
    ),
    pn.layout.HSpacer(width=20),  
    sizing_mode='stretch_width'
)

# funcion de actualizacion
def update_layout_based_on_file():
    if df_dengue is None:
        # layout previo
        estadistico_tab[1][0] = pn.layout.HSpacer(width=25)
        estadistico_tab[1][1] = control_panel
        estadistico_tab[1][2] = pn.layout.HSpacer()
        
        control_panel.width = 400
        file_input.width = 380
        upload_button.width = 380
        estado_select.width = 380
        date_slider.width = 380
        selector_comparativa.width = 380
        
        content_area.clear()
        content_area.extend([
            pn.pane.HTML(
                "<div style='display:flex;justify-content:center;align-items:center;height:400px;font-size:18px;color:#666;'>"
                "Por favor, cargue un archivo CSV para comenzar el análisis"
                "</div>",
                sizing_mode='stretch_width'
            )
        ])
        
        param_table.value = pd.DataFrame(columns=['Modelo', 'Parámetros'])
    else:
        estadistico_tab[1][0] = control_panel
        estadistico_tab[1][1] = content_area
        estadistico_tab[1][2] = None
        
        control_panel.width = 300
        file_input.width = 280
        upload_button.width = 280
        estado_select.width = 280
        date_slider.width = 280
        selector_comparativa.width = 280
        forecast_slider.width=280
        
        content_area.clear()
        content_area.extend([
            grafica_comparativa,
            pn.layout.VSpacer(height=20),
            pn.pane.Markdown("### Parámetros Estimados", margin=(0, 0, 10, 0)),
            param_table
        ])
        
        actualiza_tabla(estado_select.value, date_slider.value, selector_comparativa.value)

original_process_file = process_file
def enhanced_process_file(event):
    original_process_file(event)
    update_layout_based_on_file()
    if estado_select.value and selector_comparativa.value:
        actualiza_tabla(estado_select.value, date_slider.value, selector_comparativa.value, forecast_slider.value)
    else:
        actualiza_tabla(estado_select.value, date_slider.value, [])

upload_button.on_click(enhanced_process_file)

# Layout Teorico
teorico_tab = pn.Column(
    pn.Row( 
        pn.layout.HSpacer(),
        menu,
        pn.layout.HSpacer()
    ),
    dynamic_area
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
