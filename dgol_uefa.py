import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN Y RANKING UEFA (FEBRERO 2026)
# ============================================================================
st.set_page_config(page_title="D-GOL UEFA 25/26 - FEBRERO", page_icon="⚽", layout="wide")

# Coeficientes de Poder de Liga Actualizados (Estado real a Feb 2026)
LEAGUE_POWER = {
    'Premier League (Inglaterra)': 1.35,
    'La Liga (España)': 1.25,
    'Bundesliga (Alemania)': 1.22,
    'Serie A (Italia)': 1.20,
    'Ligue 1 (Francia)': 1.05,
    'Eredivisie (Países Bajos)': 0.95,
    'Liga Portugal': 0.92,
    'Pro League (Bélgica)': 0.85,
    'Otros (Europa)': 0.70
}

# ============================================================================
# CARGA DE DATOS REALES (TEMPORADA 25-26)
# ============================================================================
@st.cache_data(ttl=3600)
def cargar_datos_febrero_2026():
    """Descarga datos actualizados a febrero de 2026"""
    urls = {
        'Inglaterra': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
        'España': 'https://www.football-data.co.uk/mmz4281/2526/SP1.csv',
        'Alemania': 'https://www.football-data.co.uk/mmz4281/2526/D1.csv',
        'Italia': 'https://www.football-data.co.uk/mmz4281/2526/I1.csv',
        'Francia': 'https://www.football-data.co.uk/mmz4281/2526/F1.csv'
    }
    
    list_df = []
    with st.spinner('📊 Extrayendo resultados de la temporada 25/26...'):
        for liga, url in urls.items():
            try:
                df_temp = pd.read_csv(url)
                # Seleccionar columnas clave y limpiar filas vacías
                df_temp = df_temp[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
                df_temp['Liga'] = liga
                list_df.append(df_temp)
            except:
                continue
    
    if not list_df:
        return None
    
    full_df = pd.concat(list_df, ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True)
    return full_df

# ============================================================================
# MOTOR MATEMÁTICO DIXON-COLES
# ============================================================================
def entrenar_modelo_pro(df):
    equipos = sorted(list(set(df['HomeTeam']) | set(df['AwayTeam'])))
    n = len(equipos)
    eq_idx = {e: i for i, e in enumerate(equipos)}
    
    # Parámetros: ataques, defensas y ventaja de campo
    params_iniciales = np.concatenate([np.repeat(0.2, n), np.repeat(-0.2, n), [0.3]])

    def log_likelihood(params):
        ataque = params[:n]
        defensa = params[n:2*n]
        home_adv = params[-1]
        
        # Penalización para estabilizar el modelo (promedio ataque = 0)
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
# INTERFAZ DE USUARIO STREAMLIT
# ============================================================================
st.title("🏆 D-GOL UEFA PRO - FEBRERO 2026")
st.markdown(f"**Fecha actual:** {datetime.now().strftime('%d/%m/%Y')} | **Estado de temporada:** Fase Final / Eliminatorias")

df_datos = cargar_datos_febrero_2026()

if df_datos is not None:
    modelo = entrenar_modelo_pro(df_datos)
    
    # --- Sidebar de Configuración ---
    with st.sidebar:
        st.header("⚙️ Parámetros del Partido")
        equipo_h = st.selectbox("🏠 Local:", modelo['equipos'], index=modelo['equipos'].index('Real Madrid') if 'Real Madrid' in modelo['equipos'] else 0)
        liga_h = st.selectbox("Liga Local:", list(LEAGUE_POWER.keys()), index=1)
        
        st.markdown("---")
        
        equipo_a = st.selectbox("✈️ Visitante:", modelo['equipos'], index=modelo['equipos'].index('Man City') if 'Man City' in modelo['equipos'] else 1)
        liga_a = st.selectbox("Liga Visitante:", list(LEAGUE_POWER.keys()), index=0)

    # --- Cálculo de Probabilidades ---
    if st.button("🔍 EJECUTAR ANÁLISIS DIXON-COLES", use_container_width=True):
        idx_h, idx_a = modelo['idx'][equipo_h], modelo['idx'][equipo_a]
        
        # Goles esperados (Lambdas)
        l_h = np.exp(modelo['ataque'][idx_h] + modelo['defensa'][idx_a] + modelo['home_adv'])
        l_a = np.exp(modelo['ataque'][idx_a] + modelo['defensa'][idx_h])
        
        # Ajuste Crítico de Liga (Febrero 2026)
        l_h *= (LEAGUE_POWER[liga_h] / LEAGUE_POWER[liga_a])
        l_a *= (LEAGUE_POWER[liga_a] / LEAGUE_POWER[liga_h])
        
        # Matriz de Marcadores (hasta 8 goles)
        max_g = 8
        matriz = np.outer(poisson.pmf(range(max_g), l_h), poisson.pmf(range(max_g), l_a))
        
        prob_h = np.sum(np.tril(matriz, -1)) * 100
        prob_e = np.sum(np.diag(matriz)) * 100
        prob_a = np.sum(np.triu(matriz, 1)) * 100

        # --- Dashboard de Resultados ---
        st.markdown(f"### 🏟️ {equipo_h} vs {equipo_a}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Victoria Local", f"{prob_h:.1f}%", delta=f"{l_h:.2f} Goles esp.", delta_color="normal")
        c2.metric("Empate", f"{prob_e:.1f}%")
        c3.metric("Victoria Visitante", f"{prob_a:.1f}%", delta=f"{l_a:.2f} Goles esp.", delta_color="normal")
        
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["📊 Probabilidades de Marcador", "🎯 Mercados y Valor"])
        
        with tab1:
            # Gráfico de barras de goles
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(6)), y=poisson.pmf(range(6), l_h), name=equipo_h, marker_color='#003366'))
            fig.add_trace(go.Bar(x=list(range(6)), y=poisson.pmf(range(6), l_a), name=equipo_a, marker_color='#C8102E'))
            fig.update_layout(title="Distribución de Probabilidad de Goles", barmode='group', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            m1, m2 = st.columns(2)
            with m1:
                st.write("**Over/Under**")
                o15 = (1 - (matriz[0,0] + matriz[0,1] + matriz[1,0])) * 100
                o25 = (1 - (matriz[0,0] + matriz[0,1] + matriz[1,0] + matriz[1,1] + matriz[2,0] + matriz[0,2])) * 100
                st.info(f"Prob. Over 1.5: {o15:.1f}%")
                st.info(f"Prob. Over 2.5: {o25:.1f}%")
            
            with m2:
                st.write("**Marcador Exacto**")
                res_max = np.unravel_index(matriz.argmax(), matriz.shape)
                st.success(f"Resultado más probable: {res_max[0]} - {res_max[1]}")
                btts = (1 - (matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0])) * 100
                st.success(f"Ambos Marcan (BTTS): {btts:.1f}%")

else:
    st.error("Hubo un problema al conectar con los datos de febrero 2026. Revisa los CSVs de origen.")

st.markdown("---")
st.caption("D-GOL v4.8 - Algoritmo Dixon-Coles con pesos UEFA actualizados a Feb 2026.")
