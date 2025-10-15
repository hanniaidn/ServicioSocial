# Servicio Social: Desarrollo de herramientas para el análisis de datos epidemiológicos
# Clave de registro: 2024-12/211-6690
# Responsable: Dr. Mario Santana Cibrian 
# Persona prestadora del servicio: Hannia Isela Dominguez Nuñez 
# ENES Unidad Juriquilla, UNAM 
# Septiembre 2025

import panel as pn
from teorico_tab import create_teorico_tab
from estadistico_tab import create_estadistico_tab
from aboutus_tab import create_about_us_tab

# Configuración del diseño
pn.config.sizing_mode = "stretch_width"
pn.config.raw_css = ["""
    .custom-tabs .bk-tab {
        font-size: 18px; /* Tamaño de letra más grande */
        font-weight: bold;
        padding: 10px 20px;
        color: #333; /* Color de texto más oscuro */
    }
    .custom-tabs .bk-tab:hover {
        background-color: #eee; /* Fondo gris claro al pasar el mouse */
    }
    .custom-tabs .bk-tab.active { /* Estilo para la pestaña activa */
        background-color: #ddd;
        color: #007bff; /* Color azul para la pestaña activa */
    }
    .custom-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin: 10px;
        background-color: #f9f9f9;
    }
    .center-content {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .custom-header {
        display: flex;
        align-items: center; /* Contenido centrado verticalmente */
        justify-content: center; /* Contenido centrado horizontalmente */
    }
    .title {
        text-align: center; /* Título centrado */
    }
"""]

teorico_tab = create_teorico_tab()
estadistico_tab = create_estadistico_tab()
about_us_tab = create_about_us_tab()
    
logo_left = pn.pane.PNG('unamlogo.png', width=100, height=100, align='start')
logo_right = pn.pane.PNG('eneslogo.png', width=100, height=100, align='end')
header = pn.Row(
    logo_left,
    pn.pane.Markdown("# Análisis de Modelos Epidemiológicos", align='center', css_classes=["title"]),
    logo_right,
    sizing_mode='stretch_width',
    margin=(10, 10, 10, 10),
    css_classes=["custom-header"]
    )
    
template = pn.template.MaterialTemplate()
template.title = ''
template.main.append(header)
    
secciones = pn.Tabs(
    ('Estudio Teórico', teorico_tab),
    ('Estudio Estadístico', estadistico_tab),
    ('About Us', about_us_tab),
    css_classes=["custom-tabs"],
    sizing_mode="stretch_width"
    )
    
template.main.append(secciones)

template.show()