import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import odeint

# Inicializa la aplicación Dash
app = dash.Dash(__name__)

# Población inicial para el gráfico
N = 1000
I0 = 10
R0 = 0
S0 = N - I0 - R0
beta = 0.3
gamma = 0.1
t_max = 160

# Función para calcular las derivadas del modelo SIR
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Función SIR para resolver el sistema de ecuaciones diferenciales
def solve_sir(beta, gamma):
    y0 = S0, I0, R0
    t = np.linspace(0, t_max, t_max)
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return t, S, I, R

# Diseño del layout de Dash
app.layout = html.Div([
    html.H1("Modelo SIR con Dash"),
    dcc.Graph(id='sir-graph'),
    html.Label('Tasa de infección:'),
    dcc.Slider(
        id='beta-slider',
        min=0.1,
        max=1.0,
        step=0.001,
        value=0.3,
        marks={i: f'{i:.1f}' for i in np.linspace(0.1, 1, 10)}
    ),
    html.Label('Tasa de recuperación:'),
    dcc.Slider(
        id='gamma-slider',
        min=0.05,
        max=0.5,
        step=0.001,
        value=0.1,
        marks={i: f'{i:.2f}' for i in np.linspace(0.05, 0.5, 10)}
    ),
])

# Callbacks para actualizar el gráfico
@app.callback(
    Output('sir-graph', 'figure'),
    [Input('beta-slider', 'value'),
     Input('gamma-slider', 'value')]
)
def update_graph(beta, gamma):
    t, S, I, R = solve_sir(beta, gamma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptibles', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infectados', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recuperados', line=dict(color='green')))
    fig.update_layout(title=f'Modelo SIR', xaxis_title='Tiempo', yaxis_title='Población')
    return fig

# Ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)

