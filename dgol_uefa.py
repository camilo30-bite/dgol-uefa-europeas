import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime
import time

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
st.set_page_config(page_title="D-GOL UEFA 25/26", page_icon="⚽", layout="wide")

LEAGUE_POWER = {
    'Premier League (Inglaterra)': 1.35, 'La Liga (España)': 1.25,
    'Bundesliga (Alemania)': 1.22, 'Serie A (Italia)': 1.20,
    'Ligue 1 (Francia)': 1.05, 'Eredivisie (Países Bajos)': 0.95,
    'Liga Portugal': 0.92, 'Pro League (Bélgica)': 0.85, 'Otros (Europa)': 0.70
}

# ============================================================================
# CARGA DE DATOS CON "FALLBACK" (PLAN B)
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos_seguros():
    urls = {
        'Inglaterra': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
        'España': 'https://www.football-data.co.uk/mmz4281/2526/SP1.csv',
        'Alemania': 'https://www.football-data.co.uk/mmz4281/2526/D1.csv',
        'Italia': 'https://www.football-data.co.uk/mmz4281/2526/I1.csv',
        'Francia': 'https://www.football-data.co.uk/mmz4281/2526/F1.csv'
    }
    
    list_df = []
    exito = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (liga, url) in enumerate(urls.items()):
        try:
            status_text.text(f"Conectando con {liga}...")
            # Timeout añadido para evitar que la app se quede colgada
            df_temp = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
            df_temp = df_temp[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
            df_temp['Liga'] = liga
            list_df.append(df_temp)
            exito = True
        except Exception as e:
            continue
        progress_bar.progress((i + 1) / len(urls))
    
    progress_bar.empty()
    status_text.empty()

    if not exito:
        return None
    
    return pd.concat(list_df, ignore_index=True)

# ============================================================================
# MODELO Y LÓGICA
# ============================================================================
def entrenar_modelo_pro(df):
    equipos = sorted(list(set(df['HomeTeam']) | set(df['AwayTeam'])))
    n = len(equipos)
    eq_idx = {e: i for i, e in enumerate(equipos)}
    params_iniciales = np.concatenate([np.repeat(0.2, n), np.repeat(-0.2, n), [0.3]])

    def log_likelihood(params):
        ataque, defensa, home_adv = params[:n], params[n:2*n], params[-1]
        penalty = (np.mean(ataque))**2 
        ll = 0
        for _, row in df.iterrows():
            idx_h, idx_a = eq_idx[row['HomeTeam']], eq_idx[row['AwayTeam']]
            l_h = np.exp(ataque[idx_h] + defensa[idx_a] + home_adv)
            l_a = np.exp(ataque[idx_a] + defensa[idx_h])
            ll += poisson.logpmf(row['FTHG'], l_h) + poisson.logpmf(row['FTAG'], l_a)
        return -ll + penalty

    res = minimize(log_likelihood, params_iniciales, method='L-BFGS-B')
    return {'equipos': equipos, 'ataque': res.x[:n], 'defensa': res.x[n:2*n], 'home_adv': res.x[-1], 'idx': eq_idx}

# ============================================================================
# RENDERIZADO DE LA APP
# ============================================================================
st.title("🏆 D-GOL UEFA PRO - FEBRERO 2026")
st.caption(f"Fecha: {datetime.now().strftime('%d/%m/%Y')} | Estado: Fase Final")

data = cargar_datos_seguros()

if data is not None:
    # Si hay datos, mostramos el resto de la app
    modelo = entrenar_modelo_pro(data)
    
    with st.sidebar:
        st.header("Configuración")
        equipo_h = st.selectbox("🏠 Local:", modelo['equipos'])
        liga_h = st.selectbox("Liga Local:", list(LEAGUE_POWER.keys()), index=1)
        st.divider()
        equipo_a = st.selectbox("✈️ Visitante:", modelo['equipos'], index=1)
        liga_a = st.selectbox("Liga Visitante:", list(LEAGUE_POWER.keys()), index=0)
        btn = st.button("🔍 ANALIZAR", use_container_width=True, type="primary")

    if btn:
        # Lógica de cálculo (la misma del anterior)
        idx_h, idx_a = modelo['idx'][equipo_h], modelo['idx'][equipo_a]
        l_h = np.exp(modelo['ataque'][idx_h] + modelo['defensa'][idx_a] + modelo['home_adv']) * (LEAGUE_POWER[liga_h] / LEAGUE_POWER[liga_a])
        l_a = np.exp(modelo['ataque'][idx_a] + modelo['defensa'][idx_h]) * (LEAGUE_POWER[liga_a] / LEAGUE_POWER[liga_h])
        
        matriz = np.outer(poisson.pmf(range(8), l_h), poisson.pmf(range(8), l_a))
        prob_h, prob_e, prob_a = np.sum(np.tril(matriz, -1))*100, np.sum(np.diag(matriz))*100, np.sum(np.triu(matriz, 1))*100

        c1, c2, c3 = st.columns(3)
        c1.metric(equipo_h, f"{prob_h:.1f}%")
        c2.metric("Empate", f"{prob_e:.1f}%")
        c3.metric(equipo_a, f"{prob_a:.1f}%")
        
        fig = go.Figure(data=[
            go.Bar(name=equipo_h, x=list(range(6)), y=poisson.pmf(range(6), l_h)),
            go.Bar(name=equipo_a, x=list(range(6)), y=poisson.pmf(range(6), l_a))
        ])
        st.plotly_chart(fig, use_container_width=True)
else:
    # Si falla la carga, mostramos este mensaje en lugar de una pantalla blanca
    st.error("⚠️ Error de conexión con los datos de Febrero 2026.")
    st.info("El servidor de 'Football-Data' está saturado o los archivos están siendo actualizados. Por favor, refresca la página en 1 minuto.")
    if st.button("🔄 Reintentar conexión"):
        st.rerun()
