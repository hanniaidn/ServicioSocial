# Servicio Social: Desarrollo de herramientas para el análisis de datos epidemiológicos
# Clave de registro: 2024-12/211-6690
# Responsable: Dr. Mario Santana Cibrian 
# Persona prestadora del servicio: Hannia Isela Dominguez Nuñez 
# ENES Unidad Juriquilla, UNAM 
# Septiembre 2025

import panel as pn
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize
import io
from datetime import datetime

pn.extension('plotly', 'katex')

def create_estadistico_tab(): 
# Sliders necesarios 
    file_input    = pn.widgets.FileInput(accept='.csv')
    upload_button = pn.widgets.Button(name='Cargar Archivo', button_type='primary')
    estado_select = pn.widgets.Select(name='Seleccionar Estado', options=[])
    modelo_select = pn.widgets.Select(name='Seleccionar Modelo',
        options=['Exponencial', 'Logístico', 'Richards', 'Gompertz',
                'Von Bertalanffy', 'Janoschek'],
        value='Exponencial'
    )

    date_slider = pn.widgets.DatetimeRangeSlider(name='Selecciona el periodo a graficar',start=pd.Timestamp('2000-01-01'),  
        end=pd.Timestamp('2000-12-31'),   
        value=(pd.Timestamp('2000-01-01'), pd.Timestamp('2000-12-31')),
        step=86400000,  
        width=400
    )

    forecast_slider = pn.widgets.IntSlider(name='Días de pronóstico', start=0,end=180,step=1,value=30, width=280)

    T_input = pn.widgets.FloatSlider(name='Tiempo de Generación (T)', start=1, end=30, step=1, value=10)

    model_colors = {
        'Exponencial': 'blue',
        'Logístico': 'green',
        'Richards': 'red',
        'Gompertz': 'purple',
        'Von Bertalanffy': 'orange',
        'Janoschek': 'brown'
    }

    # Variable global para datos
    df = None

    def reset_controles():
        # Restablecer sliders
        forecast_slider.value = forecast_slider.start
        T_input.value = T_input.start
        
        # Restablecer selectores
        selector_comparativa.value = []
        
        # Actualizar la tabla de parámetros
        actualiza_tabla(estado_select.value, date_slider.value, selector_comparativa.value, forecast_slider.value)
        
    #  Cargar archivo 
    def process_file(event):
        try:
            nonlocal df
            df = pd.read_csv(io.BytesIO(file_input.value), parse_dates=[0], dayfirst=True)
            estados = df.columns[1:]  
            estado_select.options = list(estados)
            
            date_slider.start = df.iloc[:, 0].min()
            date_slider.end = df.iloc[:, 0].max()
            date_slider.value = (date_slider.start, date_slider.end)
            
            # Llamar a la función de reset
            reset_controles()
            
            update_layout_based_on_file()
        except Exception as e:
            pn.state.notifications.error(f"Error al cargar el archivo: {str(e)}", duration=5000)
            
    upload_button.on_click(process_file)

    # Funciones de acumulado y ajustes 
    def acumulado(data): return np.cumsum(data)

    #  Ajustes de curva para modelos 
    def _x_real(filtered_df):
        return (filtered_df.iloc[:, 0] - filtered_df.iloc[:, 0].iloc[0]).dt.days.values

    def ajuste_exponencial(estado, date_range, forecast_days):
        if df is None:
            return [], [], [], 0
        mask = (df.iloc[:, 0] >= date_range[0]) & (df.iloc[:, 0] <= date_range[1])
        db = df.loc[mask]
        y = acumulado(db[estado].dropna().values)
        x = _x_real(db)[:len(y)]
        N0 = y[0]

        def obj(p):
            r = p[0]
            return np.sum((N0 * np.exp(r * x) - y) ** 2)

        r_opt = minimize(obj, [0.1], method='Nelder-Mead').x[0]
        x_future = np.arange(x[0], x[-1] + forecast_days + 1)
        pred = N0 * np.exp(r_opt * x_future)
        return x, y, pred, r_opt

    def ajuste_logistico(estado, date_range, forecast_days):
        if df is None:
            return [], [], [], 0, 0
        mask = (df.iloc[:, 0] >= date_range[0]) & (df.iloc[:, 0] <= date_range[1])
        db = df.loc[mask]
        y = acumulado(db[estado].dropna().values)
        x = _x_real(db)[:len(y)]
        N0 = y[0]

        def obj(p):
            r, K = p
            return np.sum((K / (1 + ((K - N0) / N0) * np.exp(-r * x)) - y) ** 2)

        r_opt, K_opt = minimize(obj, [0.1, max(y)], method='Nelder-Mead').x
        x_future = np.arange(x[0], x[-1] + forecast_days + 1)
        pred = K_opt / (1 + ((K_opt - N0) / N0) * np.exp(-r_opt * x_future))
        return x, y, pred, r_opt, K_opt

    def ajuste_richards(estado, date_range, forecast_days):
        if df is None:
            return [], [], [], 0, 0, 0
        mask = (df.iloc[:, 0] >= date_range[0]) & (df.iloc[:, 0] <= date_range[1])
        db = df.loc[mask]
        y = acumulado(db[estado].dropna().values)
        x = _x_real(db)[:len(y)]
        N0 = y[0]  # Población inicial de los datos

        def modelo_richards(t, N, r, K, v):
            return r * N * (1 - (N / K)**v)

        def obj(p):
            r, K, v = p
            sol = solve_ivp(lambda t, N: modelo_richards(t, N, r, K, v), [x[0], x[-1]], [N0], method='RK45', t_eval=x)
            return np.sum((sol.y[0] - y) ** 2)

        # Inicializar los parámetros con valores razonables
        r_initial = 0.1
        K_initial = max(y) * 1.2  # Capacidad de carga un poco mayor que el máximo de los datos
        v_initial = 1.0  # Valor inicial para el parámetro de forma

        r_opt, K_opt, v_opt = minimize(obj, [r_initial, K_initial, v_initial], method='Nelder-Mead').x

        # Generar la predicción futura
        x_future = np.arange(x[0], x[-1] + forecast_days + 1)
        sol_future = solve_ivp(lambda t, N: modelo_richards(t, N, r_opt, K_opt, v_opt), [x[0], x_future[-1]], [N0], method='RK45', t_eval=x_future)
        pred = sol_future.y[0]

        return x, y, pred, r_opt, K_opt, v_opt

    def ajuste_gompertz(estado, date_range, forecast_days):
        if df is None:
            return [], [], [], 0, 0, 0
        mask = (df.iloc[:, 0] >= date_range[0]) & (df.iloc[:, 0] <= date_range[1])
        db = df.loc[mask]
        y = acumulado(db[estado].dropna().values)
        x = _x_real(db)[:len(y)]
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
        if df is None:
            return [], [], [], 0, 0
        mask = (df.iloc[:, 0] >= date_range[0]) & (df.iloc[:, 0] <= date_range[1])
        db = df.loc[mask]
        y = acumulado(db[estado].dropna().values)
        x = _x_real(db)[:len(y)]
        N0 = y[0]  # Población inicial de los datos

        def obj(p):
            L, K = p
            return np.sum((L * (1 - np.exp(-K * x)) - y) ** 2)

        L_opt, K_opt = minimize(obj, [max(y), 0.1], method='Nelder-Mead').x
        x_future = np.arange(x[0], x[-1] + forecast_days + 1)
        pred = L_opt * (1 - np.exp(-K_opt * x_future))
        return x, y, pred, L_opt, K_opt

    def ajuste_janoschek(estado, date_range, forecast_days):
        if df is None:
            return [], [], [], 0, 0, 0, 0
        mask = (df.iloc[:, 0] >= date_range[0]) & (df.iloc[:, 0] <= date_range[1])
        db = df.loc[mask]
        y = acumulado(db[estado].dropna().values)
        x = _x_real(db)[:len(y)]
        N0 = y[0]  # Población inicial de los datos

        def obj(p):
            beta, L, k, delta = p
            return np.sum((beta + (L - beta) * (1 - np.exp(-k * x)) ** delta - y) ** 2)

        beta_opt, L_opt, k_opt, delta_opt = minimize(
            obj, [0, max(y), 0.1, 1.0], method='Nelder-Mead').x
        x_future = np.arange(x[0], x[-1] + forecast_days + 1)
        pred = beta_opt + (L_opt - beta_opt) * (1 - np.exp(-k_opt * x_future)) ** delta_opt
        return x, y, pred, beta_opt, L_opt, k_opt, delta_opt

    # Selector de modelos 
    modelos_disponibles = ['Exponencial', 'Logístico', 'Richards', 'Gompertz', 'Von Bertalanffy', 'Janoschek']
    selector_comparativa = pn.widgets.MultiChoice(name='Modelos a comparar', options=modelos_disponibles, value=[],  width=300)

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

    # Ajuste de curvas
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
        elif modelo == 'Von Bertalanffy':
            x, y, pred, L, K = ajuste_bertalanffy(estado, date_range, forecast_days)
            return x, y, pred, (L, K)
        elif modelo == 'Janoschek':
            x, y, pred, beta, L, k, delta = ajuste_janoschek(estado, date_range, forecast_days)
            return x, y, pred, (beta, L, k, delta)
        else:
            return [], [], [], ()

    @pn.depends(estado_select.param.value, modelo_select.param.value, date_slider.param.value, forecast_slider.param.value)
    def grafica_datos_y_ajuste(estado, modelo, date_range, forecast_days):
        if df is None or estado not in df.columns:
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
        if df is None or estado not in df.columns:
            return pn.pane.HTML("<b style='color:red;'>Cargue un CSV y seleccione un estado</b>")

        mask = (df.iloc[:, 0] >= date_range[0]) & (df.iloc[:, 0] <= date_range[1])
        filtered_df = df.loc[mask]
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

    # Tabla de parámetros estadisticos 
    param_table = pn.widgets.DataFrame(
        pd.DataFrame(columns=['Modelo', 'Parámetros']),
        name='Parámetros estimados',
        width=850,  
        height=220,
        sizing_mode='stretch_width',
        show_index=False,
        styles={'text-align': 'left', 'white-space': 'normal', 'word-wrap': 'break-word'}
    )

    def actualizar_R0(event):
        actualiza_tabla(estado_select.value, date_slider.value, selector_comparativa.value, forecast_slider.value)
        
    @pn.depends(estado_select.param.value, date_slider.param.value, selector_comparativa.param.value, forecast_slider.param.value)
    def actualiza_tabla(estado, date_range, modelos_sel, forecast_days):
        if df is None or estado not in df.columns or not modelos_sel:
            param_table.value = pd.DataFrame(columns=['Modelo', 'Parámetros'])
            return

        filas = []
        for modelo in modelos_sel:
            try:
                if modelo == 'Exponencial':
                    _, _, _, r = ajuste_exponencial(estado, date_range, forecast_days)
                    R0 = 1 + r * T_input.value
                    filas.append({'Modelo': modelo, 'Parámetros': f"r = {r:.4f}, R0 = {R0:.4f}"})
                elif modelo == 'Logístico':
                    _, _, _, r, K = ajuste_logistico(estado, date_range, forecast_days)
                    R0 = 1 + r * T_input.value
                    filas.append({'Modelo': modelo, 'Parámetros': f"r = {r:.4f}, K = {K:.4f}, R0 = {R0:.4f}"})
                elif modelo == 'Richards':
                    _, _, _, r, K, v = ajuste_richards(estado, date_range, forecast_days)
                    R0 = 1 + r * T_input.value
                    filas.append({'Modelo': modelo, 'Parámetros': f"r = {r:.4f}, K = {K:.0f}, v = {v:.2f}, R0 = {R0:.4f}"})
                elif modelo == 'Gompertz':
                    _, _, _, a, K = ajuste_gompertz(estado, date_range, forecast_days)
                    R0 = 1 + a * T_input.value
                    filas.append({'Modelo': modelo, 'Parámetros': f"r = {a:.4f}, K = {K:.4f}, R0 = {R0:.4f}"})
                elif modelo == 'Von Bertalanffy':
                    _, _, _, L, K = ajuste_bertalanffy(estado, date_range, forecast_days)
                    R0 = 1 + K * T_input.value
                    filas.append({'Modelo': modelo, 'Parámetros': f"L = {L:.4f}, r = {K:.4f}, R0 = {R0:.4f}"})
                elif modelo == 'Janoschek':
                    _, _, _, beta, L, k, delta = ajuste_janoschek(estado, date_range, forecast_days)
                    R0 = 1 + k * T_input.value
                    filas.append({'Modelo': modelo, 'Parámetros': f"β = {beta:.4f}, L = {L:.4f}, r = {k:.4f}, δ = {delta:.2f}, R0 = {R0:.4f}"})
            except Exception as e:
                filas.append({'Modelo': modelo, 'Parámetros': 'Error en cálculo'})

        param_table.value = pd.DataFrame(filas)

    T_input.param.watch(actualizar_R0, 'value')

    # Layout Estadístico
    control_panel = pn.Column(
        file_input,
        upload_button,
        estado_select,
        date_slider,
        forecast_slider,      
        selector_comparativa,
        T_input,
        sizing_mode='stretch_height',
        width_policy='max',
        margin=(10, 10, 10, 10)
    )

    content_area = pn.Column(
        grafica_comparativa,
        pn.layout.VSpacer(height=20),
        pn.pane.Markdown("### Parámetros Estimados", margin=(0, 0, 10, 0)),
        param_table,
        sizing_mode='stretch_width',
        margin=(0, 10, 0, 0)
    )

    estadistico_tab = pn.Row(
        pn.layout.HSpacer(width=20),  
        control_panel,
        pn.layout.HSpacer(width=20),  
        content_area,
        pn.layout.HSpacer(width=20),  
        sizing_mode='stretch_width'
    )

    # Función de actualizacion
    def update_layout_based_on_file():
        nonlocal df
        if df is None:
            content_area.objects = [
                pn.pane.HTML(
                    "<div style='display:flex;justify-content:center;align-items:center;height:400px;font-size:18px;color:#666;'>"
                    "Por favor, cargue un archivo CSV para comenzar el análisis"
                    "</div>",
                    sizing_mode='stretch_width'
                )
            ]
            param_table.value = pd.DataFrame(columns=['Modelo', 'Parámetros'])
        else:
            content_area.objects = [
                grafica_comparativa,
                pn.layout.VSpacer(height=20),
                pn.pane.Markdown("### Parámetros Estimados", margin=(0, 0, 10, 0)),
                param_table
            ]
            actualiza_tabla(estado_select.value, date_slider.value, selector_comparativa.value or [], forecast_slider.value)

    original_process_file = process_file
    def enhanced_process_file(event):
        nonlocal df
        df = pd.read_csv(io.BytesIO(file_input.value), parse_dates=[0], dayfirst=True)
        estados = df.columns[1:]  
        estado_select.options = list(estados)
        
        date_slider.start = df.iloc[:, 0].min()
        date_slider.end = df.iloc[:, 0].max()
        date_slider.value = (date_slider.start, date_slider.end)

        update_layout_based_on_file()
        if estado_select.value and selector_comparativa.value:
            actualiza_tabla(estado_select.value, date_slider.value, selector_comparativa.value, forecast_slider.value)
        else:
            actualiza_tabla(estado_select.value, date_slider.value, [])

    upload_button.on_click(enhanced_process_file)
    
    return estadistico_tab