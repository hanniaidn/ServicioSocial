import panel as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy import stats
import io

pn.extension('plotly')

# Título del dashboard centrado y con estilo colorido
titulo = pn.pane.Markdown("""
# <center style='color:#4CAF50; font-size:40px;'>Análisis de Modelos Epidemiológicos</center>
""", width=800)

# ---------- MODELOS EPIDEMIOLÓGICOS ----------
# Definimos las ecuaciones diferenciales para cada modelo

# Modelo exponencial
def modelo_exponencial(N, t, r):
    dNdt = r * N
    return dNdt

# Modelo logístico
def modelo_logistico(N, t, r, K):
    dNdt = r * N * (1 - N / K)
    return dNdt

# Modelo SIR
def modelo_sir(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Modelo Richards
def modelo_richards(N, t, r, K, alpha):
    dNdt = r * N * (1 - (N / K)**alpha)
    return dNdt

# Parámetros y sliders para los modelos
r_slider = pn.widgets.FloatSlider(name='Tasa de Crecimiento (r)', start=0.1, end=1.0, step=0.01, value=0.2)
K_slider = pn.widgets.FloatSlider(name='Capacidad de Carga (K)', start=100, end=1000, step=10, value=500)
beta_slider = pn.widgets.FloatSlider(name='Tasa de Transmisión (beta)', start=0.1, end=1.0, step=0.01, value=0.3)
gamma_slider = pn.widgets.FloatSlider(name='Tasa de Recuperación (gamma)', start=0.05, end=0.5, step=0.01, value=0.1)
alpha_slider = pn.widgets.FloatSlider(name='Parámetro de forma (alpha)', start=0.1, end=2.0, step=0.1, value=1.0)

# Función para resolver y graficar el modelo seleccionado
def graficar_modelo(modelo):
    t = np.linspace(0, 50, 100)
    if modelo == 'Exponencial':
        N0 = 10
        N = odeint(modelo_exponencial, N0, t, args=(r_slider.value,))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=N[:, 0], mode='lines', name='Exponencial'))
    elif modelo == 'Logístico':
        N0 = 10
        N = odeint(modelo_logistico, N0, t, args=(r_slider.value, K_slider.value))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=N[:, 0], mode='lines', name='Logístico'))
    elif modelo == 'SIR':
        S0, I0, R0 = 0.99, 0.01, 0.0
        y0 = S0, I0, R0
        sol = odeint(modelo_sir, y0, t, args=(beta_slider.value, gamma_slider.value))
        S, I, R = sol.T
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='S'))
        fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='I'))
        fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='R'))
    elif modelo == 'Richards':
        N0 = 10
        N = odeint(modelo_richards, N0, t, args=(r_slider.value, K_slider.value, alpha_slider.value))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=N[:, 0], mode='lines', name='Richards'))
    fig.update_layout(title=f'Modelo {modelo}', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

# Selección de modelo
modelo_selector = pn.widgets.Select(name='Seleccionar Modelo', options=['Exponencial', 'Logístico', 'SIR', 'Richards'])

@pn.depends(modelo_selector, r_slider, K_slider, beta_slider, gamma_slider, alpha_slider)
def mostrar_grafica(modelo_selector):
    return graficar_modelo(modelo_selector)

# ---------- CARGA DE CSV Y GRÁFICA DE MÍNIMOS CUADRADOS ----------
file_input = pn.widgets.FileInput(accept='.csv')
alerta_csv = pn.pane.Alert("Asegúrate de cargar un archivo CSV con al menos dos columnas.", alert_type="info")

@pn.depends(file_input)
def grafica_csv(file_input):
    if file_input is None or file_input.value is None:
        return pn.pane.Markdown("**Por favor, selecciona un archivo CSV para continuar.**")
    
    try:
        # Leer archivo CSV desde el input binario
        data = pd.read_csv(io.BytesIO(file_input.value), encoding='utf-8')

        # Verificamos si el archivo tiene al menos dos columnas
        if data.shape[1] < 2:
            return pn.pane.Markdown("**Error: El archivo CSV debe contener al menos dos columnas.**")
        
        # Asegurarnos de que no hay filas vacías o con datos inválidos
        data = data.dropna()

        x = data.iloc[:, 0]
        y = data.iloc[:, 1]
        
        # Ajuste por mínimos cuadrados
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line = slope * x + intercept

        # Crear la gráfica
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Datos'))
        fig.add_trace(go.Scatter(x=x, y=line, mode='lines', name='Ajuste Lineal'))
        fig.update_layout(title='Gráfica de Mínimos Cuadrados', xaxis_title='X', yaxis_title='Y')
        return fig
    
    except Exception as e:
        return pn.pane.Markdown(f"**Error al procesar el archivo CSV: {str(e)}**")

# ---------- LAYOUT DEL DASHBOARD ----------
estadistico_tab = pn.Column(alerta_csv, file_input, grafica_csv)
teorico_tab = pn.Column(modelo_selector, r_slider, K_slider, beta_slider, gamma_slider, alpha_slider, mostrar_grafica)

# Tabs para "Estudio Teórico" y "Estudio Estadístico"
tabs = pn.Tabs(
    ('Estudio Teórico', teorico_tab),
    ('Estudio Estadístico', estadistico_tab)
)

# Layout principal
dashboard = pn.Column(titulo, tabs)

dashboard.show()
