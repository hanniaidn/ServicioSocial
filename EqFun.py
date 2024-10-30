import panel as pn
import plotly.graph_objects as go
import numpy as np
from scipy.integrate import odeint

pn.extension('plotly')

# Parámetros iniciales
N0_logistico = pn.widgets.TextInput(name='Población inicial (N0)', value='100', width=200)
r_logistico = pn.widgets.TextInput(name='Tasa de crecimiento (r)', value='0.1', width=200)
K_logistico = pn.widgets.TextInput(name='Capacidad máxima (K)', value='1000', width=200)

N0_exp = pn.widgets.TextInput(name='Población inicial (N0)', value='100000', width=200)
r_exp = pn.widgets.TextInput(name='Tasa de crecimiento (r)', value='0.05', width=200)

beta_sir = pn.widgets.TextInput(name='Tasa de infección (beta)', value='0.3', width=200)
gamma_sir = pn.widgets.TextInput(name='Tasa de recuperación (gamma)', value='0.1', width=200)

K_richards = pn.widgets.TextInput(name='Capacidad máxima (K)', value='1500', width=200)
r_richards = pn.widgets.TextInput(name='Tasa de crecimiento (r)', value='0.25', width=200)
a_richards = pn.widgets.TextInput(name='Posición (a)', value='10', width=200)
v_richards = pn.widgets.TextInput(name='Forma de la curva (v)', value='2', width=200)

# Funciones para resolver los modelos
def solve_crecimiento_logistico():
    t = np.linspace(0, 100, num=500)
    N0 = float(N0_logistico.value)
    r = float(r_logistico.value)
    K = float(K_logistico.value)
    return t, K / (1 + ((K - N0) / N0) * np.exp(-r * t))

def solve_crecimiento_exponencial():
    t = np.linspace(0, 100, num=500)
    N0 = float(N0_exp.value)
    r = float(r_exp.value)
    return t, N0 * np.exp(r * t)

def solve_sir():
    N = 1000
    I0 = 100
    R0 = 0
    S0 = N - I0 - R0
    beta = float(beta_sir.value)
    gamma = float(gamma_sir.value)
    t = np.linspace(0, 160, 160)
    y0 = S0, I0, R0

    def deriv_sir(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    ret = odeint(deriv_sir, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return t, S, I, R

def solve_richards():
    t = np.linspace(0, 50, num=500)
    K = float(K_richards.value)
    r = float(r_richards.value)
    a = float(a_richards.value)
    v = float(v_richards.value)
    return t, K / ((1 + a * np.exp(-r * t)) ** (1 / v))