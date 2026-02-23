import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go

# 1. BASE DE DATOS ESTRUCTURADA (Simulada para velocidad inmediata)
# En una versión Pro, esto se descarga en segundo plano.
def obtener_fuerza_equipos():
    # Esta tabla simula el pre-procesado de los miles de partidos de la 25/26
    # Permite que la selección sea instantánea.
    datos = {
        'Equipo': ['Real Madrid', 'Man City', 'Bayern Munich', 'Arsenal', 'Inter', 'Barcelona', 'Liverpool', 'Leverkusen'],
        'Ataque': [2.1, 2.3, 1.9, 2.0, 1.7, 2.1, 2.2, 1.8],
        'Defensa': [0.8, 0.9, 1.0, 0.7, 0.6, 1.1, 0.9, 0.9],
        'Liga': ['La Liga', 'Premier', 'Bundesliga', 'Premier', 'Serie A', 'La Liga', 'Premier', 'Bundesliga']
    }
    return pd.DataFrame(datos)

# 2. INTERFAZ ULTRA-RÁPIDA
st.set_page_config(page_title="D-GOL TURBO 25/26", layout="centered")
st.title("⚡ D-GOL UEFA TURBO v6.0")

df = obtener_fuerza_equipos()

col1, col2 = st.columns(2)
with col1:
    h_team = st.selectbox("🏠 Local", df['Equipo'])
    h_data = df[df['Equipo'] == h_team].iloc[0]
with col2:
    a_team = st.selectbox("✈️ Visitante", df['Equipo'], index=1)
    a_data = df[df['Equipo'] == a_team].iloc[0]

# 3. CÁLCULO INSTANTÁNEO (Matemática Pura)
if st.button("ANALIZAR AHORA", use_container_width=True):
    # Fórmula simplificada de Goles Esperados (xG)
    exp_h = h_data['Ataque'] * a_data['Defensa'] * 1.1 # 1.1 es factor localía
    exp_a = a_data['Ataque'] * h_data['Defensa']
    
    # Matriz Poisson rápida
    prob_h = sum(poisson.pmf(i, exp_h) * sum(poisson.pmf(j, exp_a) for j in range(i)) for i in range(1, 10))
    prob_a = sum(poisson.pmf(i, exp_a) * sum(poisson.pmf(j, exp_h) for j in range(i)) for i in range(1, 10))
    prob_e = 1 - prob_h - prob_a

    # Visualización
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Local", f"{prob_h*100:.1f}%")
    c2.metric("Empate", f"{prob_e*100:.1f}%")
    c3.metric("Visitante", f"{prob_a*100:.1f}%")
    
    st.success(f"📌 Marcador sugerido: {int(round(exp_h))} - {int(round(exp_a))}")
