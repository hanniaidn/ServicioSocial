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

# -------------  AJUSTES CORREGIDOS  ------------------------------
def _x_real(filtered_df):
    return (filtered_df.iloc[:, 0] - filtered_df.iloc[:, 0].iloc[0]).dt.days.values

def ajuste_exponencial(estado, date_range, forecast_days=30):
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

def ajuste_logistico(estado, date_range, forecast_days=30):
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

def ajuste_richards(estado, date_range, forecast_days=30):
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

def ajuste_gompertz(estado, date_range, forecast_days=30):
    if df_dengue is None:
        return [], [], [], 0, 0, 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    df = df_dengue.loc[mask]
    y = acumulado(df[estado].dropna().values)
    x = _x_real(df)[:len(y)]
    N0 = y[0]                                # first data point
    t0 = x[0]                                # first day (should be 0)

    # Gompertz with fixed N0:  N(t)=K*(N0/K)^(exp(-a*(t-t0)))
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

def ajuste_bertalanffy(estado, date_range, forecast_days=30):
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

def ajuste_janoschek(estado, date_range, forecast_days=30):
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
selector_comparativa = pn.widgets.MultiChoice(name='Modelos a comparar', 
                                              options=modelos_disponibles, value=['Exponencial'], width=300)


def calcula_ajuste(modelo, estado, date_range):
    if modelo == 'Exponencial':
        x, y, pred, r = ajuste_exponencial(estado, date_range)
        return x, y, pred, (r,)
    elif modelo == 'Logístico':
        x, y, pred, r, K = ajuste_logistico(estado, date_range)
        return x, y, pred, (r, K)
    elif modelo == 'Richards':
        x, y, pred, r, K, v = ajuste_richards(estado, date_range)
        return x, y, pred, (r, K, v)
    elif modelo == 'Gompertz':
        x, y, pred, a, K = ajuste_gompertz(estado, date_range)
        return x, y, pred, (a, K)
    elif modelo == 'Bertalanffy-Ivlev':
        x, y, pred, L, K = ajuste_bertalanffy(estado, date_range)
        return x, y, pred, (L, K)
    elif modelo == 'Janoschek':
        x, y, pred, beta, L, k, delta = ajuste_janoschek(estado, date_range)
        return x, y, pred, (beta, L, k, delta)
    else:
        return [], [], [], ()

# Callback BUENO
@pn.depends(estado_select.param.value, modelo_select.param.value, date_slider.param.value)
def grafica_datos_y_ajuste(estado, modelo, date_range):
    if df_dengue is None or estado not in df_dengue.columns:
        return pn.pane.HTML("<b style='color:red;'>Cargue un CSV y seleccione un estado</b>")

    x, y, pred, *_ = calcula_ajuste(modelo, estado, date_range)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Datos Reales'))
    fig.add_trace(go.Scatter(x=x, y=pred[:len(x)], mode='lines', name=f'Ajuste {modelo}'))
    fig.update_layout(title=f'Ajuste {modelo} para {estado}',
                      xaxis_title='Días desde inicio', yaxis_title='Casos')
    return fig

@pn.depends(estado_select.param.value, date_slider.param.value, selector_comparativa.param.value)
def grafica_comparativa(estado, date_range, modelos_sel):
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
        x, y, pred, *_ = calcula_ajuste(modelo, estado, date_range)
        extended_semanas = pd.date_range(start=semanas.min(), periods=len(pred), freq='D')
        fig.add_trace(go.Scatter(x=extended_semanas, y=pred,
                                 mode='lines', name=f'Ajuste {modelo}',
                                 line=dict(color=model_colors[modelo])))

    fig.update_layout(title=f'Comparativa modelos para {estado}',
                      xaxis_title='Fecha', yaxis_title='Casos')
    return fig

@pn.depends(estado_select.param.value, date_slider.param.value, selector_comparativa.param.value)
def grafica_comparativa(estado, date_range, modelos_sel):
    if df_dengue is None or estado not in df_dengue.columns:
        return pn.pane.HTML("<b style='color:red;'>Cargue un CSV y seleccione un estado</b>")
    
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    filtered_df = df_dengue.loc[mask]
    semanas = filtered_df.iloc[:, 0]
    
    # acumulado de datos
    datos_reales = acumulado(filtered_df[estado].dropna().values)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=semanas, y=datos_reales, mode='markers+lines', name='Datos Reales', line=dict(color='black')))
    
    for modelo in modelos_sel:
        _, _, pred, _ = calcula_ajuste(modelo, estado, date_range)
        extended_semanas = pd.date_range(start=semanas.min(), periods=len(pred), freq='D')
        fig.add_trace(go.Scatter(x=extended_semanas, y=pred, mode='lines', name=f'Ajuste {modelo}', line=dict(color=model_colors[modelo])))
    
    fig.update_layout(title=f'Comparativa modelos para {estado}',
                      xaxis_title='Fecha', yaxis_title='Casos')
    return fig

#tabla de parametros estadisticos 
param_table = pn.widgets.DataFrame(pd.DataFrame(),
                                   name='Parámetros estimados',
                                   width=600, height=220)

@pn.depends(estado_select.param.value, date_slider.param.value, selector_comparativa.param.value)
def actualiza_tabla(estado, date_range, modelos_sel):
    if df_dengue is None or estado not in df_dengue.columns:
        param_table.value = pd.DataFrame()
        return

    filas = []
    for modelo in modelos_sel:
        if modelo == 'Exponencial':
            _, _, _, r = ajuste_exponencial(estado, date_range)
            filas.append({'Modelo': modelo, 'r': r})
        elif modelo == 'Logístico':
            _, _, _, r, K = ajuste_logistico(estado, date_range)
            filas.append({'Modelo': modelo, 'r': r, 'K': K})
        elif modelo == 'Richards':
            _, _, _, r, K, v = ajuste_richards(estado, date_range)
            filas.append({'Modelo': modelo, 'r': r, 'K': K, 'v': v})
        elif modelo == 'Gompertz':
             _, _, _, a, K = ajuste_gompertz(estado, date_range)
             filas.append({'Modelo': modelo, 'a': a, 'K': K})
        elif modelo == 'Bertalanffy-Ivlev':
            _, _, _, L, K = ajuste_bertalanffy(estado, date_range)
            filas.append({'Modelo': modelo, 'L': L, 'K': K})
        elif modelo == 'Janoschek':
            _, _, _, beta, L, k, delta = ajuste_janoschek(estado, date_range)
            filas.append({'Modelo': modelo, 'β': beta, 'L': L, 'k': k, 'δ': delta})

    param_table.value = pd.DataFrame(filas)


# Layout Estadístico
estadistico_tab = pn.Column(
    file_input,
    upload_button,
    estado_select,
    date_slider,              
    selector_comparativa, 
    grafica_comparativa
         
)

estadistico_tab.append(actualiza_tabla)   #actualiza la tabla cuando cambian widgets
estadistico_tab.append(param_table)       #muestra la tabla debajo del grafico


# Layout Teorico
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
