
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
    .bk-tab {font-size: 18px; font-weight: bold; padding: 10px 20px; color: #333;}
    .bk-tab.bk-active {background-color: #ddd; color: #007bff;}
    """
]


def render_eq(eq):
    return pn.pane.LaTeX(f"${eq}$", sizing_mode="stretch_width")


r_exp      = pn.widgets.FloatSlider(name='r', start=0.1, end=1.0, step=0.01, value=0.3)
N0_exp     = pn.widgets.FloatSlider(name='N₀', start=10, end=100, step=1, value=50)
t_exp      = pn.widgets.IntSlider(name='tiempo', start=0, end=50, step=1, value=25)

K_log      = pn.widgets.FloatSlider(name='K', start=50, end=500, step=10, value=200)
r_log      = pn.widgets.FloatSlider(name='r', start=0.1, end=1.0, step=0.01, value=0.3)

v_richards = pn.widgets.FloatSlider(name='v', start=0.1, end=5, step=0.1, value=1)

beta_sir   = pn.widgets.FloatSlider(name='β', start=0.1, end=1, step=0.01, value=0.3)
gamma_sir  = pn.widgets.FloatSlider(name='γ', start=0.05, end=0.5, step=0.01, value=0.1)
t_sir      = pn.widgets.IntSlider(name='tiempo', start=0, end=100, step=1, value=50)

K_gompertz = pn.widgets.FloatSlider(name='K', start=50, end=500, step=10, value=200)
a_gompertz = pn.widgets.FloatSlider(name='a', start=0.01, end=1, step=0.01, value=0.1)

L_bi       = pn.widgets.FloatSlider(name='L', start=50, end=500, step=10, value=300)
K_bi       = pn.widgets.FloatSlider(name='K', start=0.01, end=1, step=0.01, value=0.1)

beta_jan   = pn.widgets.FloatSlider(name='β', start=0, end=50, step=1, value=10)
L_jan      = pn.widgets.FloatSlider(name='L', start=50, end=500, step=10, value=300)
k_jan      = pn.widgets.FloatSlider(name='k', start=0.01, end=1, step=0.01, value=0.1)
delta_jan  = pn.widgets.FloatSlider(name='δ', start=0.5, end=5, step=0.1, value=1)


def modelo_exponencial(t, N, r): return r*N
def modelo_logistico(t, N, r, K): return r*N*(1-N/K)
def modelo_richards(t, N, r, K, v): return r*N*(1-(N/K)**v)
def modelo_gompertz(t, N, a, K): return a*N*np.log(K/N)
def modelo_bertalanffy(t, L, K): return L*(1-np.exp(-K*t))
def modelo_janoschek(t, b, L, k, d): return b+(L-b)*(1-np.exp(-k*t))**d
def modelo_sir(y, t, N, b, g):
    S, I, R = y; return [-b*S*I/N, b*S*I/N-g*I, g*I]

def sol_exp(N0, r, t): tv=np.linspace(0,t,100); sol=solve_ivp(lambda t,N: modelo_exponencial(t,N,r),[0,t],[N0],t_eval=tv); return tv, sol.y[0]
def sol_log(N0, r, K, t): tv=np.linspace(0,t,100); sol=solve_ivp(lambda t,N: modelo_logistico(t,N,r,K),[0,t],[N0],t_eval=tv); return tv, sol.y[0]
def sol_richards(N0, r, K, v, t): tv=np.linspace(0,t,100); sol=solve_ivp(lambda t,N: modelo_richards(t,N,r,K,v),[0,t],[N0],t_eval=tv); return tv, sol.y[0]
def sol_gompertz(N0, a, K, t): tv=np.linspace(0,t,100); sol=solve_ivp(lambda t,N: modelo_gompertz(t,N,a,K),[0,t],[N0],t_eval=tv); return tv, sol.y[0]
def sol_bertalanffy(L,K,t): tv=np.linspace(0,t,100); return tv, modelo_bertalanffy(tv,L,K)
def sol_janoschek(b,L,k,d,t): tv=np.linspace(0,t,100); return tv, modelo_janoschek(tv,b,L,k,d)
def sol_sir(b,g,N,I0,R0,t): S0=N-I0-R0; tv=np.linspace(0,t,100); ret=odeint(modelo_sir,[S0,I0,R0],tv,args=(N,b,g)); return tv,*ret.T


@pn.depends(r_exp, N0_exp, t_exp)
def g_exp(r, N0, t): tv,y=sol_exp(N0,r,t); return go.Figure(go.Scatter(x=tv,y=y,mode='lines',name='Exponencial')).update_layout(title='Exponencial')
@pn.depends(N0_exp, r_log, K_log, t_exp)
def g_log(N0, r, K, t): tv,y=sol_log(N0,r,K,t); return go.Figure(go.Scatter(x=tv,y=y,mode='lines',name='Logístico')).update_layout(title='Logístico')
@pn.depends(N0_exp, r_log, K_log, v_richards, t_exp)
def g_rich(N0, r, K, v, t): tv,y=sol_richards(N0,r,K,v,t); return go.Figure(go.Scatter(x=tv,y=y,mode='lines',name='Richards')).update_layout(title='Richards')
@pn.depends(beta_sir, gamma_sir, t_sir)
def g_sir(b,g,t): tv,S,I,R=sol_sir(b,g,10000,150,0,t); return go.Figure([go.Scatter(x=tv,y=S,name='S'),go.Scatter(x=tv,y=I,name='I'),go.Scatter(x=tv,y=R,name='R')]).update_layout(title='SIR')
@pn.depends(N0_exp, a_gompertz, K_gompertz, t_exp)
def g_gomp(N0,a,K,t): tv,y=sol_gompertz(N0,a,K,t); return go.Figure(go.Scatter(x=tv,y=y,mode='lines',name='Gompertz')).update_layout(title='Gompertz')
@pn.depends(L_bi, K_bi, t_exp)
def g_bert(L,K,t): tv,y=sol_bertalanffy(L,K,t); return go.Figure(go.Scatter(x=tv,y=y,mode='lines',name='Bertalanffy')).update_layout(title='Bertalanffy-Ivlev')
@pn.depends(beta_jan, L_jan, k_jan, delta_jan, t_exp)
def g_jan(b,L,k,d,t): tv,y=sol_janoschek(b,L,k,d,t); return go.Figure(go.Scatter(x=tv,y=y,mode='lines',name='Janoschek')).update_layout(title='Janoschek')


desc = {
    'Exponencial': pn.Column(pn.pane.Markdown('**Crecimiento sin límites**'), pn.pane.LaTeX(r"\frac{dN}{dt}=rN")),
    'Logístico':   pn.Column(pn.pane.Markdown('**Crecimiento limitado por K**'), pn.pane.LaTeX(r"\frac{dN}{dt}=rN\bigl(1-\frac{N}{K}\bigr)")),
    'Richards':    pn.Column(pn.pane.Markdown('**Generaliza el logístico**'), pn.pane.LaTeX(r"\frac{dN}{dt}=rN\Bigl[1-\Bigl(\frac{N}{K}\Bigr)^v\Bigr]")),
    'Gompertz':    pn.Column(pn.pane.Markdown('**Tasa decrece exponencialmente**'), pn.pane.LaTeX(r"\frac{dN}{dt}=aN\ln\!\Bigl(\frac{K}{N}\Bigr)")),
    'Bertalanffy-Ivlev': pn.Column(pn.pane.Markdown('**Crecimiento limitado tipo Bertalanffy**'), pn.pane.LaTeX(r"N(t)=L\bigl(1-e^{-Kt}\bigr)")),
    'Janoschek':   pn.Column(pn.pane.Markdown('**Asíntota inferior y superior**'), pn.pane.LaTeX(r"N(t)=\beta+(L-\beta)\bigl(1-e^{-kt}\bigr)^\delta")),
}


teorico_tabs = pn.Tabs(
    ('Logístico',   pn.Row(desc['Logístico'],   pn.Column(K_log, r_log, g_log))),
    ('Exponencial', pn.Row(desc['Exponencial'], pn.Column(r_exp, N0_exp, t_exp, g_exp))),
    ('SIR',         pn.Column(beta_sir, gamma_sir, t_sir, g_sir)),
    ('Richards',    pn.Row(desc['Richards'],    pn.Column(v_richards, g_rich))),
    ('Gompertz',    pn.Row(desc['Gompertz'],    pn.Column(a_gompertz, K_gompertz, g_gomp))),
    ('Bertalanffy-Ivlev', pn.Row(desc['Bertalanffy-Ivlev'], pn.Column(L_bi, K_bi, g_bert))),
    ('Janoschek',   pn.Row(desc['Janoschek'],   pn.Column(beta_jan, L_jan, k_jan, delta_jan, g_jan)))
)


file_input    = pn.widgets.FileInput(accept='.csv')
upload_button = pn.widgets.Button(name='Cargar Archivo', button_type='primary')
estado_select = pn.widgets.Select(name='Seleccionar Estado', options=[])
date_slider   = pn.widgets.DateRangeSlider(name='Rango de fechas',start=pd.Timestamp('2000-01-01'),
                                           end=pd.Timestamp('2025-12-31'),
                                           value=(pd.Timestamp('2020-01-01'), pd.Timestamp('2025-12-31')))
selector_comp = pn.widgets.MultiChoice(name='Modelos a comparar',
                                       options=['Exponencial','Logístico','Richards','Gompertz','Bertalanffy-Ivlev','Janoschek'],
                                       value=['Logístico'])
model_colors  = {'Exponencial':'blue','Logístico':'green','Richards':'red','Gompertz':'purple','Bertalanffy-Ivlev':'orange','Janoschek':'brown'}

df = None
def load_file(event):
    global df
    if file_input.value is None: return
    df = pd.read_csv(io.BytesIO(file_input.value), parse_dates=[0], dayfirst=True)
    estado_select.options = list(df.columns[1:])
    date_slider.start = df.iloc[:,0].min()
    date_slider.end   = df.iloc[:,0].max()
upload_button.on_click(load_file)
def cum(data): return np.cumsum(data.values)


def fit_exp(est, rng, f=30):
    if df is None: return [],[],[],0
    mask=(df.iloc[:,0]>=rng[0])&(df.iloc[:,0]<=rng[1])
    d=cum(df.loc[mask,est]); t=np.arange(len(d)); N0=d[0]
    res=minimize(lambda p: np.sum((N0*np.exp(p[0]*t)-d)**2), [0.1], bounds=[(1e-4,5)], method='L-BFGS-B')
    pred=N0*np.exp(res.x[0]*np.arange(len(t)+f))
    return t,d,pred,res.x[0]

def fit_log(est,rng,f=30):
    mask=(df.iloc[:,0]>=rng[0])&(df.iloc[:,0]<=rng[1])
    d=cum(df.loc[mask,est]); t=np.arange(len(d)); N0=d[0]
    res=minimize(lambda p: np.sum((p[1]/(1+((p[1]-N0)/N0)*np.exp(-p[0]*t)) - d)**2),
                 [0.1,d[-1]*1.2], bounds=[(1e-4,5),(d[-1],d[-1]*5)], method='L-BFGS-B')
    r,K=res.x; pred=K/(1+((K-N0)/N0)*np.exp(-r*np.arange(len(t)+f)))
    return t,d,pred,r

def fit_rich(est,rng,f=30):
    mask=(df.iloc[:,0]>=rng[0])&(df.iloc[:,0]<=rng[1])
    d=cum(df.loc[mask,est]); t=np.arange(len(d)); N0=d[0]
    res=minimize(lambda p: np.sum((p[1]/(1+((p[1]-N0)/N0)*np.exp(-p[0]*t))**p[2]-d)**2),
                 [0.1,d[-1]*1.2,1], bounds=[(1e-4,5),(d[-1],d[-1]*5),(0.1,10)], method='L-BFGS-B')
    r,K,v=res.x; pred=K/(1+((K-N0)/N0)*np.exp(-r*np.arange(len(t)+f)))**v
    return t,d,pred,r

def fit_gomp(est,rng,f=30):
    mask=(df.iloc[:,0]>=rng[0])&(df.iloc[:,0]<=rng[1])
    d=cum(df.loc[mask,est]); t=np.arange(len(d)); N0=d[0]
    res=minimize(lambda p: np.sum((p[1]*np.exp(-np.exp(-p[0]*(t-t[len(t)//2])))-d)**2),
                 [0.05,d[-1]*1.2], bounds=[(1e-4,5),(d[-1],d[-1]*5)], method='L-BFGS-B')
    a,K=res.x; pred=K*np.exp(-np.exp(-a*(np.arange(len(t)+f)-t[len(t)//2])))
    return t,d,pred,a

def fit_bert(est,rng,f=30):
    mask=(df.iloc[:,0]>=rng[0])&(df.iloc[:,0]<=rng[1])
    d=cum(df.loc[mask,est]); t=np.arange(len(d)); N0=d[0]
    res=minimize(lambda p: np.sum((p[0]*(1-np.exp(-p[1]*t)) - d)**2),
                 [d[-1]*1.2,0.05], bounds=[(d[-1],d[-1]*5),(1e-4,2)], method='L-BFGS-B')
    L,K=res.x; pred=L*(1-np.exp(-K*np.arange(len(t)+f)))
    return t,d,pred,K

def fit_jan(est,rng,f=30):
    mask=(df.iloc[:,0]>=rng[0])&(df.iloc[:,0]<=rng[1])
    d=cum(df.loc[mask,est]); t=np.arange(len(d)); N0=d[0]
    res=minimize(lambda p: np.sum((p[0]+(p[1]-p[0])*(1-np.exp(-p[2]*t))**p[3]-d)**2),
                 [0,d[-1]*1.2,0.05,1], bounds=[(0,d[-1]),(d[-1],d[-1]*5),(1e-4,2),(0.1,10)], method='L-BFGS-B')
    b,L,k,delta=res.x; pred=b+(L-b)*(1-np.exp(-k*np.arange(len(t)+f)))**delta
    return t,d,pred,k

@pn.depends(estado_select.param.value, date_slider.param.value, selector_comp.param.value)
def grafica_comparativa(estado, date_range, modelos_sel):
    if df is None or estado not in df.columns:
        return pn.pane.HTML("<b style='color:red;'>Cargue un CSV y seleccione un estado</b>")
    mask=(df.iloc[:,0]>=date_range[0])&(df.iloc[:,0]<=date_range[1])
    filt=df.loc[mask]; semanas=filt.iloc[:,0]; datos=cum(filt[estado])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=semanas,y=datos,mode='markers+lines',name='Datos reales',line=dict(color='black')))
    for m in modelos_sel:
        t,d,pred,param={'Exponencial':fit_exp,'Logístico':fit_log,'Richards':fit_rich,'Gompertz':fit_gomp,'Bertalanffy-Ivlev':fit_bert,'Janoschek':fit_jan}[m](estado,date_range)
        ext=pd.date_range(start=semanas.min(),periods=len(pred),freq='D')
        fig.add_trace(go.Scatter(x=ext,y=pred,mode='lines',name=m,line=dict(color=model_colors[m])))
    fig.update_layout(title=f'Comparativa modelos – {estado}',xaxis_title='Fecha',yaxis_title='Casos acumulados')
    return fig

stat_layout = pn.Column(file_input, upload_button, estado_select, date_slider, selector_comp, grafica_comparativa)


template = pn.template.MaterialTemplate(title='Análisis de Modelos Epidemiológicos')
template.main.append(pn.Tabs(
    ('Estudio Teórico', teorico_tabs),
    ('Estudio Estadístico', stat_layout)
))
template.show()