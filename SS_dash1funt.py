import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import odeint

# inicializa Dash
app = dash.Dash(__name__)

# Funciones para los modelos
# Modelo SIR
def deriv_sir(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def solve_sir(beta, gamma):
    N = 1000
    I0, R0 = 10, 0
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.linspace(0, 160, 160)
    ret = odeint(deriv_sir, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return t, S, I, R

# Modelo de Crecimiento Exponencial
def crecimiento_exponencial(N0, r):
    t = np.linspace(0, 100, 1000)
    N = N0 * np.exp(r * t)
    return t, N

# Modelo de Crecimiento Logístico
def crecimiento_logistico(N0, K, r):
    t = np.linspace(0, 100, 1000)
    N = (K * N0) / (N0 + (K - N0) * np.exp(-r * t))
    return t, N

# Modelo de Richards
def modelo_richards(N0, r, K, v):
    t = np.linspace(0, 100, 1000)
    N = K / ((1 + (K / N0 - 1) * np.exp(-r * t)) ** (1 / v))
    return t, N

# layout 
app.layout = html.Div([
    html.Div([
        html.H1("Análisis teórico de los modelos", style={'textAlign': 'center'})
    ], style={'backgroundColor': '#f7f7f7', 'padding': '20px'}),
    
    html.Div([
        html.Div([
            html.H2("Modelos disponibles"),
            html.Button('Modelo SIR', id='sir-btn', n_clicks=0, style={'margin-bottom': '10px'}),
            html.Button('Modelo de Richards', id='richards-btn', n_clicks=0, style={'margin-bottom': '10px'}),
            html.Button('Modelo de Crecimiento Logístico', id='logistic-btn', n_clicks=0, style={'margin-bottom': '10px'}),
            html.Button('Modelo de Crecimiento Exponencial', id='exp-btn', n_clicks=0),
        ], style={'width': '20%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top', 'backgroundColor': '#f0f0f0'}),
        
        html.Div([
            dcc.Graph(id='model-graph')
        ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
    ])
])

# callback para actualizar la gráfica con base en el modelo seleccionado
@app.callback(
    Output('model-graph', 'figure'),
    [Input('sir-btn', 'n_clicks'),
     Input('richards-btn', 'n_clicks'),
     Input('logistic-btn', 'n_clicks'),
     Input('exp-btn', 'n_clicks')]
)
def display_model(sir_clicks, richards_clicks, logistic_clicks, exp_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure()  
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # crea figura para cada modelo
    if button_id == 'sir-btn':
        t, S, I, R = solve_sir(beta=0.3, gamma=0.1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptibles'))
        fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infectados'))
        fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recuperados'))
        fig.update_layout(title='Modelo SIR', xaxis_title='Tiempo', yaxis_title='Población')
    
    elif button_id == 'exp-btn':
        t, N = crecimiento_exponencial(N0=1000, r=0.05)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=N, mode='lines', name='Crecimiento Exponencial'))
        fig.update_layout(title='Modelo de Crecimiento Exponencial', xaxis_title='Tiempo', yaxis_title='Población')
    
    elif button_id == 'logistic-btn':
        t, N = crecimiento_logistico(N0=1000, K=5000, r=0.05)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=N, mode='lines', name='Crecimiento Logístico'))
        fig.update_layout(title='Modelo de Crecimiento Logístico', xaxis_title='Tiempo', yaxis_title='Población')
    
    elif button_id == 'richards-btn':
        t, N = modelo_richards(N0=1000, r=0.05, K=5000, v=2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=N, mode='lines', name='Modelo de Richards'))
        fig.update_layout(title='Modelo de Richards', xaxis_title='Tiempo', yaxis_title='Población')
    
    return fig

# ejecuta la aplicacion
if __name__ == '__main__':
    app.run_server(debug=True, port=8070)
