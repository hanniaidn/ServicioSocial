# ==========================================================
#  FULL DASHBOARD  –  ORIGINAL LOGIC  +  NICE STYLING
# ==========================================================
import panel as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize
import io
from datetime import datetime

pn.extension('plotly', 'katex')
pn.config.sizing_mode = "stretch_width"
pn.config.raw_css = [
    """
    .custom-tabs .bk-tab {
        font-size: 18px; font-weight: bold; padding: 10px 20px; color: #333;
    }
    .custom-tabs .bk-tab.active {
        background-color: #ddd; color: #007bff;
    }
    .center-content {
        display: flex; justify-content: center; align-items: center; text-align: center;
    }
    """
]

# ----------------------------------------------------------
#  1.  ORIGINAL  –  THEORETICAL  TAB  (no change)
# ----------------------------------------------------------
def render_ecuacion(eq):
    return pn.pane.LaTeX(f"${eq}$", sizing_mode="stretch_width")

# --- widgets (theoretical) ---
r_exp      = pn.widgets.FloatSlider(name='Tasa de crecimiento inicial', start=0.1, end=1.0, step=0.01, value=0.3)
N0_exp     = pn.widgets.FloatSlider(name='Población inicial', start=10, end=100, step=1, value=50)
t_exp      = pn.widgets.IntSlider(name='Tiempo', start=0, end=50, step=1, value=25)

K_log      = pn.widgets.FloatSlider(name='Capacidad de carga', start=50, end=500, step=10, value=200)
r_log      = pn.widgets.FloatSlider(name='Tasa de crecimiento logístico', start=0.1, end=1.0, step=0.01, value=0.3)

v_richards = pn.widgets.FloatSlider(name='Parámetro de forma (v)', start=0.1, end=5.0, step=0.1, value=1.0)

beta_sir   = pn.widgets.FloatSlider(name='Tasa de infección', start=0.1, end=1.0, step=0.01, value=0.3)
gamma_sir  = pn.widgets.FloatSlider(name='Tasa de recuperación', start=0.05, end=0.5, step=0.01, value=0.1)
t_sir      = pn.widgets.IntSlider(name='Tiempo', start=0, end=100, step=1, value=50)
N_sir, I0_sir, R0_sir = 10000, 150, 0

K_gompertz = pn.widgets.FloatSlider(name='Capacidad de carga (K)', start=50, end=500, step=10, value=200)
a_gompertz = pn.widgets.FloatSlider(name='Tasa de crecimiento (a)', start=0.01, end=1.0, step=0.01, value=0.1)

L_bi       = pn.widgets.FloatSlider(name='Tamaño límite (L)', start=50, end=500, step=10, value=300)
K_bi       = pn.widgets.FloatSlider(name='Coeficiente de crecimiento (K)', start=0.01, end=1.0, step=0.01, value=0.1)

beta_jan   = pn.widgets.FloatSlider(name='Asintota inferior (β)', start=0, end=50, step=1, value=10)
L_jan      = pn.widgets.FloatSlider(name='Asintota superior (L)', start=50, end=500, step=10, value=300)
k_jan      = pn.widgets.FloatSlider(name='Tasa de crecimiento (k)', start=0.01, end=1.0, step=0.01, value=0.1)
delta_jan  = pn.widgets.FloatSlider(name='Parámetro δ', start=0.5, end=5.0, step=0.1, value=1.0)

# --- model equations ---
def modelo_exponencial(t, N, r): return r * N
def modelo_logistico(t, N, r, K): return r * N * (1 - N / K)
def modelo_richards(t, N, r, K, v): return r * N * (1 - (N / K)**v)
def modelo_gompertz(t, N, a, K): return a * N * np.log(K / N)
def modelo_bertalanffy(t, L, K): return L * (1 - np.exp(-K * t))
def modelo_janoschek(t, beta, L, k, delta): return beta + (L - beta) * (1 - np.exp(-k * t))**delta
def modelo_sir(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# --- solvers ---
def sol_exp(N0, r, t):
    tv = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_exponencial(t, N, r), [0, t], [N0], t_eval=tv)
    return tv, sol.y[0]

def sol_log(N0, r, K, t):
    tv = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_logistico(t, N, r, K), [0, t], [N0], t_eval=tv)
    return tv, sol.y[0]

def sol_richards(N0, r, K, v, t):
    tv = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_richards(t, N, r, K, v), [0, t], [N0], t_eval=tv)
    return tv, sol.y[0]

def sol_gompertz(N0, a, K, t):
    tv = np.linspace(0, t, 100)
    sol = solve_ivp(lambda t, N: modelo_gompertz(t, N, a, K), [0, t], [N0], t_eval=tv)
    return tv, sol.y[0]

def sol_bertalanffy(L, K, t):
    tv = np.linspace(0, t, 100)
    return tv, modelo_bertalanffy(tv, L, K)

def sol_janoschek(beta, L, k, delta, t):
    tv = np.linspace(0, t, 100)
    return tv, modelo_janoschek(tv, beta, L, k, delta)

def sol_sir(beta, gamma, N, I0, R0, t):
    S0 = N - I0 - R0
    y0 = [S0, I0, R0]
    tv = np.linspace(0, t, 100)
    ret = odeint(modelo_sir, y0, tv, args=(N, beta, gamma))
    S, I, R = ret.T
    return tv, S, I, R

# --- plotting functions (theoretical) ---
@pn.depends(r_exp, N0_exp, t_exp)
def grafica_exponencial(r, N0, t):
    tv, y = sol_exp(N0, r, t)
    fig = go.Figure(go.Scatter(x=tv, y=y, mode='lines', name='Exponencial'))
    fig.update_layout(title='Modelo Exponencial', xaxis_title='Tiempo', yaxis_title='Casos')
    return fig

@pn.depends(N0_exp, r_log, K_log, t_exp)
def grafica_logistico(N0, r, K, t):
    tv, y = sol_log(N0, r, K, t)
    fig = go.Figure(go.Scatter(x=tv, y=y, mode='lines', name='Logístico'))
    fig.update_layout(title='Modelo Logístico', xaxis_title='Tiempo', yaxis_title='Casos')
    return fig

@pn.depends(N0_exp, r_log, K_log, v_richards, t_exp)
def grafica_richards(N0, r, K, v, t):
    tv, y = sol_richards(N0, r, K, v, t)
    fig = go.Figure(go.Scatter(x=tv, y=y, mode='lines', name='Richards'))
    fig.update_layout(title='Modelo Richards', xaxis_title='Tiempo', yaxis_title='Casos')
    return fig

@pn.depends(beta_sir, gamma_sir, t_sir)
def grafica_sir(beta, gamma, t):
    tv, S, I, R = sol_sir(beta, gamma, N_sir, I0_sir, R0_sir, t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tv, y=S, mode='lines', name='Susceptibles'))
    fig.add_trace(go.Scatter(x=tv, y=I, mode='lines', name='Infectados'))
    fig.add_trace(go.Scatter(x=tv, y=R, mode='lines', name='Recuperados'))
    fig.update_layout(title='Modelo SIR', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

@pn.depends(N0_exp, a_gompertz, K_gompertz, t_exp)
def grafica_gompertz(N0, a, K, t):
    tv, y = sol_gompertz(N0, a, K, t)
    fig = go.Figure(go.Scatter(x=tv, y=y, mode='lines', name='Gompertz'))
    fig.update_layout(title='Modelo Gompertz', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

@pn.depends(L_bi, K_bi, t_exp)
def grafica_bertalanffy(L, K, t):
    tv, y = sol_bertalanffy(L, K, t)
    fig = go.Figure(go.Scatter(x=tv, y=y, mode='lines', name='Bertalanffy-Ivlev'))
    fig.update_layout(title='Modelo Bertalanffy-Ivlev', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

@pn.depends(beta_jan, L_jan, k_jan, delta_jan, t_exp)
def grafica_janoschek(beta, L, k, delta, t):
    tv, y = sol_janoschek(beta, L, k, delta, t)
    fig = go.Figure(go.Scatter(x=tv, y=y, mode='lines', name='Janoschek'))
    fig.update_layout(title='Modelo Janoschek', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

# ----------  LaTeX description cards ----------
desc = {
    'Exponencial': pn.Card(pn.pane.Markdown('**Crecimiento sin límites**'), render_ecuacion(r"\frac{dN}{dt}=rN"), title='Exponencial', styles={'background':'#f9f9f9','border':'1px solid #ddd','border-radius':'5px'}, width=320),
    'Logístico':   pn.Card(pn.pane.Markdown('**Crecimiento limitado por capacidad K**'), render_ecuacion(r"\frac{dN}{dt}=rN(1-\frac{N}{K})"), title='Logístico', styles={'background':'#f9f9f9','border':'1px solid #ddd','border-radius':'5px'}, width=320),
    'Richards':    pn.Card(pn.pane.Markdown('**Generalización logística flexible**'), render_ecuacion(r"\frac{dN}{dt}=rN(1-(\frac{N}{K})^v)"), title='Richards', styles={'background':'#f9f9f9','border':'1px solid #ddd','border-radius':'5px'}, width=320),
    'Gompertz':    pn.Card(pn.pane.Markdown('**Tasa decreciente exponencialmente**'), render_ecuacion(r"\frac{dN}{dt}=aN\ln(\frac{K}{N})"), title='Gompertz', styles={'background':'#f9f9f9','border':'1px solid #ddd','border-radius':'5px'}, width=320),
    'Bertalanffy-Ivlev': pn.Card(pn.pane.Markdown('**Crecimiento limitado tipo Bertalanffy**'), render_ecuacion(r"N(t)=L(1-e^{-Kt})"), title='Bertalanffy-Ivlev', styles={'background':'#f9f9f9','border':'1px solid #ddd','border-radius':'5px'}, width=320),
    'Janoschek':   pn.Card(pn.pane.Markdown('**Asíntota inferior y superior**'), render_ecuacion(r"N(t)=\beta+(L-\beta)(1-e^{-kt})^\delta"), title='Janoschek', styles={'background':'#f9f9f9','border':'1px solid #ddd','border-radius':'5px'}, width=320),
}

# ----------  theoretical tab ----------
selector = pn.widgets.Select(options=list(desc.keys()), name='Seleccione un modelo', value='Logístico')
dyn_area = pn.Column(pn.Row(desc['Logístico'], pn.Column(grafica_logistico, r_log, K_log)))

def change_tab(ev):
    m = ev.new
    dyn_area[:] = [pn.Row(desc[m], pn.Column(
        {'Exponencial': grafica_exponencial, 'Logístico': grafica_logistico,
         'Richards': grafica_richards, 'Gompertz': grafica_gompertz,
         'Bertalanffy-Ivlev': grafica_bertalanffy, 'Janoschek': grafica_janoschek}[m],
        {'Exponencial': [r_exp, N0_exp, t_exp], 'Logístico': [r_log, K_log],
         'Richards': [v_richards], 'Gompertz': [a_gompertz, K_gompertz],
         'Bertalanffy-Ivlev': [L_bi, K_bi], 'Janoschek': [beta_jan, L_jan, k_jan, delta_jan]}[m]))]
selector.param.watch(change_tab, 'value')
teorico_tab = pn.Column(selector, dyn_area)

# ----------------------------------------------------------
#  2.  ORIGINAL  –  STATISTICAL  TAB  (kept verbatim)
# ----------------------------------------------------------
# widgets (statistical)
file_input    = pn.widgets.FileInput(accept='.csv')
upload_button = pn.widgets.Button(name='Cargar Archivo', button_type='primary')
estado_select = pn.widgets.Select(name='Seleccionar Estado', options=[])
modelo_select = pn.widgets.Select(name='Seleccionar Modelo', options=list(desc.keys()), value='Logístico')
date_slider   = pn.widgets.DateRangeSlider(name='Rango de fechas', start=datetime(2020, 1, 1), end=datetime(2023, 12, 31))
modelos_disp  = list(desc.keys())
selector_comp = pn.widgets.MultiChoice(name='Modelos a comparar', options=modelos_disp, value=['Logístico'])

# global dataframe
df_dengue = None
FORECAST_DAYS = 30
model_colors = {
    'Exponencial': 'blue', 'Logístico': 'green', 'Richards': 'red',
    'Gompertz': 'purple', 'Bertalanffy-Ivlev': 'orange', 'Janoschek': 'brown'
}

def process_file(event):
    global df_dengue
    df_dengue = pd.read_csv(io.BytesIO(file_input.value), parse_dates=[0], dayfirst=True)
    estados = df_dengue.columns[1:]
    estado_select.options = list(estados)
    date_slider.start = df_dengue.iloc[:, 0].min()
    date_slider.end   = df_dengue.iloc[:, 0].max()
    date_slider.value = (date_slider.start, date_slider.end)

upload_button.on_click(process_file)

def acumulado(data): return np.cumsum(data)

# ------------  ORIGINAL 6  FITTING  FUNCTIONS  ------------
def ajuste_exponencial(estado, date_range, forecast_days=FORECAST_DAYS):
    if df_dengue is None: return [], [], [], 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    filtered_df = df_dengue.loc[mask]
    data_acumulado = acumulado(filtered_df[estado].dropna().values)
    t = np.arange(len(data_acumulado))
    N0 = data_acumulado[0]
    def loss(p):
        r = p[0]
        pred = N0 * np.exp(r * t)
        return np.sum((data_acumulado - pred) ** 2)
    res = minimize(loss, x0=[0.1], bounds=[(1e-4, 5)], method='L-BFGS-B')
    r_opt = res.x[0]
    pred = N0 * np.exp(r_opt * np.arange(len(t) + forecast_days))
    return t, data_acumulado, pred, r_opt

def ajuste_logistico(estado, date_range, forecast_days=FORECAST_DAYS):
    if df_dengue is None: return [], [], [], 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    filtered_df = df_dengue.loc[mask]
    data_acumulado = acumulado(filtered_df[estado].dropna().values)
    t = np.arange(len(data_acumulado))
    N0 = data_acumulado[0]
    def loss(p):
        r, K = p
        pred = K / (1 + ((K - N0) / N0) * np.exp(-r * t))
        return np.sum((data_acumulado - pred) ** 2)
    res = minimize(loss, x0=[0.1, data_acumulado[-1] * 1.2],
                   bounds=[(1e-4, 5), (data_acumulado[-1], data_acumulado[-1] * 5)], method='L-BFGS-B')
    r_opt, K_opt = res.x
    pred = K_opt / (1 + ((K_opt - N0) / N0) * np.exp(-r_opt * np.arange(len(t) + forecast_days)))
    return t, data_acumulado, pred, r_opt

def ajuste_richards(estado, date_range, forecast_days=FORECAST_DAYS):
    if df_dengue is None: return [], [], [], 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    filtered_df = df_dengue.loc[mask]
    data_acumulado = acumulado(filtered_df[estado].dropna().values)
    t = np.arange(len(data_acumulado))
    N0 = data_acumulado[0]
    def loss(p):
        r, K, v = p
        pred = K / (1 + ((K - N0) / N0) * np.exp(-r * t)) ** v
        return np.sum((data_acumulado - pred) ** 2)
    res = minimize(loss, x0=[0.1, data_acumulado[-1] * 1.2, 1.0],
                   bounds=[(1e-4, 5), (data_acumulado[-1], data_acumulado[-1] * 5), (0.1, 10)], method='L-BFGS-B')
    r_opt, K_opt, v_opt = res.x
    pred = K_opt / (1 + ((K_opt - N0) / N0) * np.exp(-r_opt * np.arange(len(t) + forecast_days))) ** v_opt
    return t, data_acumulado, pred, r_opt

def ajuste_gompertz(estado, date_range, forecast_days=FORECAST_DAYS):
    if df_dengue is None: return [], [], [], 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    filtered_df = df_dengue.loc[mask]
    data_acumulado = acumulado(filtered_df[estado].dropna().values)
    t = np.arange(len(data_acumulado))
    def loss(p):
        a, K = p
        pred = K * np.exp(-np.exp(-a * (t - t[len(t)//2])))
        return np.sum((data_acumulado - pred) ** 2)
    res = minimize(loss, x0=[0.1, data_acumulado[-1] * 1.2],
                   bounds=[(1e-4, 5), (data_acumulado[-1], data_acumulado[-1] * 5)], method='L-BFGS-B')
    a_opt, K_opt = res.x
    pred = K_opt * np.exp(-np.exp(-a_opt * (np.arange(len(t) + forecast_days) - t[len(t)//2])))
    return t, data_acumulado, pred, a_opt

def ajuste_bertalanffy(estado, date_range, forecast_days=FORECAST_DAYS):
    if df_dengue is None: return [], [], [], 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    filtered_df = df_dengue.loc[mask]
    data_acumulado = acumulado(filtered_df[estado].dropna().values)
    t = np.arange(len(data_acumulado))
    def loss(p):
        L, K = p
        pred = L * (1 - np.exp(-K * t))
        return np.sum((data_acumulado - pred) ** 2)
    res = minimize(loss, x0=[data_acumulado[-1] * 1.2, 0.05],
                   bounds=[(data_acumulado[-1], data_acumulado[-1] * 5), (1e-4, 2)], method='L-BFGS-B')
    L_opt, K_opt = res.x
    pred = L_opt * (1 - np.exp(-K_opt * np.arange(len(t) + forecast_days)))
    return t, data_acumulado, pred, K_opt

def ajuste_janoschek(estado, date_range, forecast_days=FORECAST_DAYS):
    if df_dengue is None: return [], [], [], 0
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    filtered_df = df_dengue.loc[mask]
    data_acumulado = acumulado(filtered_df[estado].dropna().values)
    t = np.arange(len(data_acumulado))
    def loss(p):
        beta, L, k, delta = p
        pred = beta + (L - beta) * (1 - np.exp(-k * t))**delta
        return np.sum((data_acumulado - pred) ** 2)
    res = minimize(loss, x0=[0, data_acumulado[-1] * 1.2, 0.05, 1.0],
                   bounds=[(0, data_acumulado[-1]), (data_acumulado[-1], data_acumulado[-1] * 5),
                           (1e-4, 2), (0.1, 10)], method='L-BFGS-B')
    beta_opt, L_opt, k_opt, delta_opt = res.x
    pred = beta_opt + (L_opt - beta_opt) * (1 - np.exp(-k_opt * np.arange(len(t) + forecast_days)))**delta_opt
    return t, data_acumulado, pred, k_opt

def calcula_ajuste(modelo, estado, date_range):
    return {
        'Exponencial': ajuste_exponencial,
        'Logístico': ajuste_logistico,
        'Richards': ajuste_richards,
        'Gompertz': ajuste_gompertz,
        'Bertalanffy-Ivlev': ajuste_bertalanffy,
        'Janoschek': ajuste_janoschek,
    }[modelo](estado, date_range)

# --- statistical plotting ---
@pn.depends(estado_select.param.value, date_slider.param.value, selector_comp.param.value)
def grafica_comparativa(estado, date_range, modelos_sel):
    if df_dengue is None or estado not in df_dengue.columns:
        return pn.pane.HTML("<b style='color:red;'>Cargue un CSV y seleccione un estado</b>")
    mask = (df_dengue.iloc[:, 0] >= date_range[0]) & (df_dengue.iloc[:, 0] <= date_range[1])
    filtered_df = df_dengue.loc[mask]
    semanas = filtered_df.iloc[:, 0]
    datos_reales = acumulado(filtered_df[estado].dropna().values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=semanas, y=datos_reales, mode='markers+lines',
                             name='Datos Reales', line=dict(color='black')))
    for modelo in modelos_sel:
        _, _, pred, _ = calcula_ajuste(modelo, estado, date_range)
        extended_semanas = pd.date_range(start=semanas.min(), periods=len(pred), freq='D')
        fig.add_trace(go.Scatter(x=extended_semanas, y=pred, mode='lines',
                                 name=f'Ajuste {modelo}', line=dict(color=model_colors[modelo])))
    fig.update_layout(title=f'Comparativa modelos para {estado}',
                      xaxis_title='Fecha', yaxis_title='Casos')
    return fig

# ----------  statistical layout ----------
stat_layout = pn.Column(
    file_input,
    upload_button,
    estado_select,
    date_slider,
    selector_comp,
    grafica_comparativa
)

# ----------  theoretical tab ----------
teorico_tabs = pn.Tabs(
    ('Logístico', pn.Column(K_log, r_log, grafica_logistico)),
    ('Exponencial', pn.Column(r_exp, N0_exp, t_exp, grafica_exponencial)),
    ('SIR', pn.Column(beta_sir, gamma_sir, t_sir, grafica_sir)),
    ('Richards', pn.Column(v_richards, grafica_richards)),
    ('Gompertz', pn.Column(a_gompertz, K_gompertz, grafica_gompertz)),
    ('Bertalanffy-Ivlev', pn.Column(L_bi, K_bi, grafica_bertalanffy)),
    ('Janoschek', pn.Column(beta_jan, L_jan, k_jan, delta_jan, grafica_janoschek))
)

# ----------  final dashboard ----------
template = pn.template.MaterialTemplate(title='Análisis de Modelos Epidemiológicos')
template.main.append(
    pn.Tabs(
        ('Estudio Teórico', teorico_tabs),
        ('Estudio Estadístico', stat_layout),
        css_classes=['custom-tabs']
    )
)
template.show()