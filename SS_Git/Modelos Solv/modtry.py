import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parámetros globales
global_par_S = 1000.0
global_par_alpha = 0.1
global_par_k_t = 0.03
global_par_k_tl = 1.4
global_par_k_b = 7200.0
global_par_k_f = 5000.0
global_par_beta = 0.6
global_par_gamma = 0.2
global_par_delta = 11.0

# Definición de las ecuaciones diferenciales
def model(t, x):
    # Variables dinámicas
    x1, x2, x3, x4 = x
    
    # Ecuaciones diferenciales
    dx1 = global_par_S - global_par_k_f * x1 * x3 - global_par_alpha * x1 + (global_par_k_b + global_par_gamma) * x4
    dx2 = global_par_k_t * x1**2 - global_par_beta * x2
    dx3 = global_par_k_tl * x2 - global_par_k_f * x1 * x3 + (global_par_k_b + global_par_delta) * x4 - global_par_gamma * x3
    dx4 = global_par_k_f * x1 * x3 - (global_par_k_b + global_par_delta) * x4 - global_par_gamma * x4
    
    # Rate rules
    
    return [dx1, dx2, dx3, dx4]

# Condiciones iniciales
x0 = [1.0] * 4

# Intervalo de tiempo
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000)

# Resolución del sistema de ecuaciones diferenciales
solution = solve_ivp(model, t_span, x0, method='RK45', t_eval=t_eval, atol=1e-3)

# Extraer los resultados
t = solution.t
x = solution.y

# Graficar los resultados
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(t, x[i], label=f'x{i+1}')
plt.xlabel('Tiempo')
plt.ylabel('Concentraciones')
plt.title('Dinámica del modelo p53')
plt.legend()
plt.grid()
plt.show()
