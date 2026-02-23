import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN GENERAL Y NIVELES DE LIGA (FEBRERO 2026)
# ============================================================================
st.set_page_config(page_title="D-GOL UEFA PRO v5.0", page_icon="🏆", layout="wide")

LEAGUE_POWER = {
    'Premier League (Inglaterra)': 1.35,
    'La Liga (España)': 1.25,
    'Bundesliga (Alemania)': 1.22,
    'Serie A (Italia)': 1.20,
    'Ligue 1 (Francia)': 1.05,
    'Eredivisie (Países Bajos)': 0.95,
    'Liga Portugal': 0.92,
    'Pro League (Bélgica)': 0.85,
    'Super Lig (Turquía)': 0.80,
    'Otros (Europa)': 0.70
}

# ============================================================================
# EXTRACCIÓN DE DATOS ULTRA-RÁPIDA (FOOTBALL-DATA.CO.UK)
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos_velocidad():
    # URLs de la temporada 25/26
    ligas_url = {
        'Premier League': 'https://www.football-data.co.uk/mmz4281/2526/E0.csv',
        'La Liga': 'https://www.football-data.co.uk/mmz4281/2526/SP1.csv',
        'Bundesliga': 'https://www.football-data.co.uk/mmz4281/2526/D1.csv',
        'Serie A': 'https://www.football-data.co.uk/mmz4281/2526/I1.csv',
        'Ligue 1': 'https://www.football-data.co.uk/mmz4281/2526/F1.csv'
    }
    
    datasets = []
    
    # Barra de progreso para que la pantalla no se quede en blanco
    progreso = st.progress(0)
    texto_estado = st.empty()
    
    for i, (nombre, url) in enumerate(ligas_url.items()):
        texto_estado.text(f"📥 Descargando datos de {nombre}...")
        try:
            # Leemos el CSV. Usamos un User-Agent básico por seguridad
            df = pd.read_csv(url, storage_options={'User-Agent': 'Mozilla/5.0'})
            
            # Filtramos solo las columnas que el modelo matemático necesita
            columnas_necesarias = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
            df = df[columnas_necesarias].dropna() # dropna() evita errores con partidos no jugados
            df['Liga_Origen'] = nombre
            datasets.append(df)
        except Exception as e:
            st.warning(f"⚠️ No se pudo obtener {nombre}. El servidor podría estar actualizando el archivo.")
            continue
            
        progreso.progress((i + 1) / len(ligas_url))
        
    progreso.empty()
    texto_estado.empty()
    
    if not datasets:
        return None
        
    return pd.concat(datasets, ignore_index=True)

# ============================================================================
# MOTOR MATEMÁTICO DIXON-COLES
# ============================================================================
def entrenar_dixon_coles(df):
    equipos = sorted(list(set(df['HomeTeam']) | set(df['AwayTeam'])))
    n = len(equipos)
    eq_idx = {e: i for i, e in enumerate(equipos)}
    
    # Parámetros: [Ataque_0...n, Defensa_0...n, Ventaja_Local]
    params_iniciales = np.concatenate([np.repeat(0.2, n), np.repeat(-0.2, n), [0.3]])

    def log_likelihood(params):
        ataque = params[:n]
        defensa = params[n:2*n]
        home_adv = params[-1]
        
        # Penalización para estabilizar las matemáticas (centrado en cero)
        penalty = (np.mean(ataque))**2 
        
        ll = 0
        for _, row in df.iterrows():
            idx_h, idx_a = eq_idx[row['HomeTeam']], eq_idx[row['AwayTeam']]
            l_h = np.exp(ataque[idx_h] + defensa[idx_a] + home_adv)
            l_a = np.exp(ataque[idx_a] + defensa[idx_h])
            ll += poisson.logpmf(row['FTHG'], l_h) + poisson.logpmf(row['FTAG'], l_a)
        return -ll + penalty

    # Optimización del modelo
    res = minimize(log_likelihood, params_iniciales, method='L-BFGS-B')
    
    return {
        'equipos': equipos,
        'ataque': res.x[:n],
        'defensa': res.x[n:2*n],
        'home_adv': res.x[-1],
        'idx': eq_idx
    }

# ============================================================================
# INTERFAZ DE USUARIO (STREAMLIT)
# ============================================================================
st.title("🏆 D-GOL UEFA PRO v5.0")
st.markdown("**Sistema Predictivo Avanzado | Temporada 25/26**")

# Cargar datos
df_datos = cargar_datos_velocidad()

if df_datos is not None:
    modelo = entrenar_dixon_coles(df_datos)
    
    # --- BARRA LATERAL: CONFIGURACIÓN ---
    with st.sidebar:
        st.header("⚙️ Configuración del Partido")
        
        st.subheader("🏠 Equipo Local")
        equipo_h = st.selectbox("Selecciona Local:", modelo['equipos'], index=modelo['equipos'].index('Real Madrid') if 'Real Madrid' in modelo['equipos'] else 0)
        liga_h = st.selectbox("Nivel de Liga Local:", list(LEAGUE_POWER.keys()), index=1)
        
        st.divider()
        
        st.subheader("✈️ Equipo Visitante")
        equipo_a = st.selectbox("Selecciona Visitante:", modelo['equipos'], index=modelo['equipos'].index('Man City') if 'Man City' in modelo['equipos'] else 1)
        liga_a = st.selectbox("Nivel de Liga Visitante:", list(LEAGUE_POWER.keys()), index=0)
        
        st.divider()
        
        st.subheader("📊 Cuotas de Apuesta (Opcional)")
        st.caption("Ingresa las cuotas de tu casa de apuestas para ver si tienen valor matemático.")
        cuota_1 = st.number_input("Cuota Local (1)", min_value=1.01, value=2.00, step=0.1)
        cuota_X = st.number_input("Cuota Empate (X)", min_value=1.01, value=3.50, step=0.1)
        cuota_2 = st.number_input("Cuota Visitante (2)", min_value=1.01, value=3.20, step=0.1)
        
        analizar = st.button("🚀 CALCULAR PREDICCIÓN", use_container_width=True, type="primary")

    # --- ÁREA PRINCIPAL DE RESULTADOS ---
    if analizar:
        # 1. Obtener índices y lambdas base
        idx_h, idx_a = modelo['idx'][equipo_h], modelo['idx'][equipo_a]
        l_h = np.exp(modelo['ataque'][idx_h] + modelo['defensa'][idx_a] + modelo['home_adv'])
        l_a = np.exp(modelo['ataque'][idx_a] + modelo['defensa'][idx_h])
        
        # 2. Ajuste por Ranking de Liga (Poder Relativo UEFA)
        ajuste_liga = LEAGUE_POWER[liga_h] / LEAGUE_POWER[liga_a]
        l_h *= ajuste_liga
        l_a *= (1 / ajuste_liga)
        
        # 3. Matriz de Poisson (Probabilidades de Marcador hasta 8 goles)
        max_g = 8
        matriz = np.outer(poisson.pmf(range(max_g), l_h), poisson.pmf(range(max_g), l_a))
        
        prob_h = np.sum(np.tril(matriz, -1)) * 100
        prob_e = np.sum(np.diag(matriz)) * 100
        prob_a = np.sum(np.triu(matriz, 1)) * 100

        # Mostrar métricas principales
        st.markdown(f"### 🏟️ Análisis: {equipo_h} vs {equipo_a}")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Victoria {equipo_h}", f"{prob_h:.1f}%", f"Cuota Justa: {(100/prob_h):.2f}")
        c2.metric("Empate", f"{prob_e:.1f}%", f"Cuota Justa: {(100/prob_e):.2f}", delta_color="off")
        c3.metric(f"Victoria {equipo_a}", f"{prob_a:.1f}%", f"Cuota Justa: {(100/prob_a):.2f}")
        
        st.markdown("---")
        
        # Pestañas de análisis detallado
        tab1, tab2, tab3 = st.tabs(["🎯 Mercados Principales", "📈 Gráfico de Goles", "💰 Análisis de Valor (Value Bet)"])
        
        with tab1:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.write("**Mercado Over/Under**")
                o15 = (1 - (matriz[0,0] + matriz[0,1] + matriz[1,0])) * 100
                o25 = (1 - (matriz[0,0] + matriz[0,1] + matriz[1,0] + matriz[1,1] + matriz[2,0] + matriz[0,2])) * 100
                st.info(f"**Más de 1.5 Goles:** {o15:.1f}%")
                st.info(f"**Más de 2.5 Goles:** {o25:.1f}%")
                st.write(f"*Goles totales esperados: {(l_h + l_a):.2f}*")
            
            with col_m2:
                st.write("**Marcadores y BTTS**")
                btts = (1 - (matriz[0,:].sum() + matriz[:,0].sum() - matriz[0,0])) * 100
                marcador = np.unravel_index(matriz.argmax(), matriz.shape)
                st.success(f"**Ambos Equipos Marcan:** {btts:.1f}%")
                st.success(f"**Marcador Exacto Más Probable:** {marcador[0]} - {marcador[1]}")
                st.write(f"*Probabilidad del marcador: {(matriz[marcador]*100):.1f}%*")

        with tab2:
            fig = go.Figure(data=[
                go.Bar(name=equipo_h, x=list(range(6)), y=poisson.pmf(range(6), l_h), marker_color='#1e3a8a'),
                go.Bar(name=equipo_a, x=list(range(6)), y=poisson.pmf(range(6), l_a), marker_color='#10b981')
            ])
            fig.update_layout(
                title="Distribución de Probabilidad de Goles por Equipo", 
                barmode='group',
                xaxis_title="Número de Goles",
                yaxis_title="Probabilidad"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("### 💰 Detección de Apuestas de Valor (EV+)")
            st.write("El modelo compara tus cuotas ingresadas con la probabilidad real matemática. Si el valor esperado (EV) es mayor a 0%, es una apuesta rentable a largo plazo.")
            
            def calcular_ev(prob, cuota):
                ev = ((prob / 100) * cuota) - 1
                return ev * 100

            ev_1 = calcular_ev(prob_h, cuota_1)
            ev_X = calcular_ev(prob_e, cuota_X)
            ev_2 = calcular_ev(prob_a, cuota_2)

            v1, v2, v3 = st.columns(3)
            
            # Formato condicional para el valor
            def mostrar_valor(columna, titulo, ev_val):
                if ev_val > 0:
                    columna.success(f"**{titulo}**\n\nEV: +{ev_val:.1f}% ✅ ¡HAY VALOR!")
                else:
                    columna.error(f"**{titulo}**\n\nEV: {ev_val:.1f}% ❌ Sin valor")

            mostrar_valor(v1, f"Local ({equipo_h}) a cuota {cuota_1}", ev_1)
            mostrar_valor(v2, f"Empate a cuota {cuota_X}", ev_X)
            mostrar_valor(v3, f"Visitante ({equipo_a}) a cuota {cuota_2}", ev_2)

else:
    st.error("⚠️ Error Crítico: No se pudieron descargar los datos.")
    st.info("Esto suele ocurrir si no hay internet o si el servidor origen bloquea temporalmente la descarga. Intenta actualizar la página en un par de minutos.")

st.markdown("---")
st.caption("Desarrollado con Python & Streamlit | Modelo Dixon-Coles | Datos: football-data.co.uk")
