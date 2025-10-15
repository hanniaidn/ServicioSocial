# Servicio Social: Desarrollo de herramientas para el análisis de datos epidemiológicos
# Clave de registro: 2024-12/211-6690
# Responsable: Dr. Mario Santana Cibrian 
# Persona prestadora del servicio: Hannia Isela Dominguez Nuñez 
# ENES Unidad Juriquilla, UNAM 
# Septiembre 2025

import panel as pn

def create_about_us_tab():
    about_us_tab = pn.Column(
        pn.pane.Markdown("""
        # About Us
        ## ¡Nuestro Equipo!
        """),
        sizing_mode='stretch_width',
        margin=(10, 10, 10, 10)
    )
    return about_us_tab