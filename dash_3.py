
import panel as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import io

pn.extension('plotly')

# titulo dash
titulo = pn.pane.Markdown("""
# <center style='color:#4CAF50; font-size:40px;'>Análisis de Modelos Epidemiológicos </center>
""", width=800)

# widgets para el modelo
r_exp = pn.widgets.FloatSlider(name='Tasa de crecimiento inicial', start=0.1, end=1.0, step=0.01, value=0.3)
N0_exp = pn.widgets.FloatSlider(name='Población inicial', start=10, end=100, step=1, value=50)
t_exp = pn.widgets.IntSlider(name='Tiempo', start=0, end=50, step=1, value=25)  # Limitar a 50 semanas

# modelo exponencial
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

# carga de CSV y ajuste del modelo
file_input = pn.widgets.FileInput(accept='.csv')
upload_button = pn.widgets.Button(name='Cargar Archivo', button_type='primary')
estado_select = pn.widgets.Select(name='Seleccionar Estado', options=[])

df_dengue = None  # se define como none para que la puede leer 

def process_file(event):
    global df_dengue
    if file_input.value is not None:
        df_dengue = pd.read_csv(io.BytesIO(file_input.value))
        estados = df_dengue.columns[2:]  # solemnete toma el nombre del estado
        estado_select.options = list(estados)  # actualiza la seleccion de estado

upload_button.on_click(process_file)

# ajuste exponencial
def ajuste_exponencial(estado):
    if df_dengue is None:
        return [], [], [], 0  # no retorna nada si no se ha cargado el csv
    data = df_dengue[estado].dropna().values[:50]  # 
    semanas = np.arange(len(data))  
    
    def funcion_objetivo(params):
        r = params[0]
        N0 = params[1]
        pred = N0 * np.exp(r * semanas)
        return np.sum((data - pred) ** 2)

    resultado = minimize(funcion_objetivo, x0=[0.1, data[0]], method='Nelder-Mead')
    r_opt, N0_opt = resultado.x
    predicciones = N0_opt * np.exp(r_opt * semanas)
    
    return semanas, data, predicciones, r_opt

# grafica del ajuste exponencial
@pn.depends(estado_select)
def grafica_ajuste(estado):
    if df_dengue is None or estado not in df_dengue.columns:
        return pn.pane.HTML("<b style='color:red;'>Por favor, cargue un archivo válido y seleccione un estado</b>")
    semanas, datos_reales, predicciones, r_calculado = ajuste_exponencial(estado)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=semanas, y=datos_reales, mode='lines+markers', name='Datos Reales'))
    fig.add_trace(go.Scatter(x=semanas, y=predicciones, mode='lines', name='Ajuste Exponencial'))
    fig.update_layout(title=f'Ajuste Exponencial para {estado}', xaxis_title='Semana', yaxis_title='Casos')
    return fig


teorico_tab = pn.Column(r_exp, N0_exp, t_exp, grafica_exponencial)
estadistico_tab = pn.Column(file_input, upload_button, estado_select, grafica_ajuste)
secciones = pn.Tabs(('Estudio Teórico', teorico_tab), ('Estudio Estadístico', estadistico_tab))
dashboard = pn.Column(titulo, secciones)

dashboard.show()
