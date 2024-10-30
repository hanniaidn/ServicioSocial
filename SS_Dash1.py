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
            html.Div([
                html.Button('Modelo SIR', id='sir-btn', n_clicks=0, style={'width': '100%', 'padding': '15px', 'fontSize': 18}),
            ], style={'margin-bottom': '20px'}),
            html.Div([
                html.Button('Modelo de Richards', id='richards-btn', n_clicks=0, style={'width': '100%', 'padding': '15px', 'fontSize': 18}),
            ], style={'margin-bottom': '20px'}),
            html.Div([
                html.Button('Modelo de Crecimiento Logístico', id='logistic-btn', n_clicks=0, style={'width': '100%', 'padding': '15px', 'fontSize': 18}),
            ], style={'margin-bottom': '20px'}),
            html.Div([
                html.Button('Modelo de Crecimiento Exponencial', id='exp-btn', n_clicks=0, style={'width': '100%', 'padding': '15px', 'fontSize': 18}),
            ]),
        ], style={'width': '25%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top', 'backgroundColor': '#f0f0f0', 'height': '100vh'}),
        
        
        html.Div(id='model-display', style={'width': '70%', 'display': 'inline-block', 'padding': '20px'})
    ])
])

# callback para actualizar la gráfica con base en el modelo seleccionado
@app.callback(
    Output('model-display', 'children'),
    [Input('sir-btn', 'n_clicks'),
     Input('richards-btn', 'n_clicks'),
     Input('logistic-btn', 'n_clicks'),
     Input('exp-btn', 'n_clicks'),
     Input('sir-slider-beta', 'value'),
     Input('sir-slider-gamma', 'value'),
     Input('richards-slider-r', 'value'),
     Input('richards-slider-K', 'value'),
     Input('logistic-slider-r', 'value'),
     Input('logistic-slider-K', 'value'),
     Input('exp-slider-r', 'value')]
)
def display_model(sir_clicks, richards_clicks, logistic_clicks, exp_clicks,
                  beta=0.3, gamma=0.1, r_richards=0.1, K_richards=10, r_logistic=0.1, K_logistic=5000, r_exponential=0.05):
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure()  
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'sir-btn' or sir_clicks > 0:
        t, S, I, R = solve_sir(beta=beta, gamma=gamma)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptibles'))
        fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infectados'))
        fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recuperados'))
        fig.update_layout(title='Modelo SIR', xaxis_title='Tiempo', yaxis_title='Población')
        return [
            dcc.Graph(figure=fig),
            dcc.Slider(id='sir-slider-beta', min=0.1, max=1.0, step=0.01, value=0.3, 
                       marks={i/10: str(i/10) for i in range(1, 11)}, 
                       tooltip={"placement": "bottom", "always_visible": True}),
            dcc.Slider(id='sir-slider-gamma', min=0.01, max=0.5, step=0.01, value=0.1, 
                       marks={i/100: str(i/100) for i in range(1, 51)}, 
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]
    
    elif button_id == 'exp-btn' or exp_clicks > 0:
        t, N = crecimiento_exponencial(N0=1000, r=r_exponential)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=N, mode='lines', name='Crecimiento Exponencial'))
        fig.update_layout(title='Modelo de Crecimiento Exponencial', xaxis_title='Tiempo', yaxis_title='Población')
        return [
            dcc.Graph(figure=fig),
            dcc.Slider(id='exp-slider-r', min=0.01, max=1.0, step=0.01, value=0.05, 
                       marks={i/100: str(i/100) for i in range(1, 101)}, 
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]
    
    elif button_id == 'logistic-btn' or logistic_clicks > 0:
        t, N = crecimiento_logistico(N0=1000, K=K_logistic, r=r_logistic)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=N, mode='lines', name='Crecimiento Logístico'))
        fig.update_layout(title='Modelo de Crecimiento Logístico', xaxis_title='Tiempo', yaxis_title='Población')
        return [
            dcc.Graph(figure=fig),
            dcc.Slider(id='logistic-slider-r', min=0.01, max=1.0, step=0.01, value=0.05, 
                       marks={i/100: str(i/100) for i in range(1, 101)}, 
                       tooltip={"placement": "bottom", "always_visible": True}),
            dcc.Slider(id='logistic-slider-K', min=1000, max=10000, step=500, value=5000, 
                       marks={i: str(i) for i in range(1000, 11000, 1000)}, 
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]
    
    elif button_id == 'richards-btn' or richards_clicks > 0:
        t, N = modelo_richards(N0=10, r=r_richards, K=K_richards, v=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=N, mode='lines', name='Modelo de Richards'))
        fig.update_layout(title='Modelo de Richards', xaxis_title='Tiempo', yaxis_title='Población')
        return [
            dcc.Graph(figure=fig),
            dcc.Slider(id='richards-slider-r', min=0.01, max=1.0, step=0.01, value=0.05, 
                       marks={i/100: str(i/100) for i in range(1, 101)}, 
                       tooltip={"placement": "bottom", "always_visible": True}),
            dcc.Slider(id='richards-slider-K', min=1, max=100, step=1, value=10, 
                       marks={i: str(i) for i in range(1, 101)}, 
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]
    
    return go.Figure()  

if __name__ == '__main__':
    app.run_server(debug=True, port=8060)



## observaciones 
# es posible que el slider sea un objeto propio de dash y no de plotly 
# dcc.Slider component de dash 
# checar dcc & dmc 
# checar video de gmail 
# comparar panel y dash 
# implementar sliders con dash & dcc 