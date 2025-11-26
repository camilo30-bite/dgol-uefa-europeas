"""
🏆 D-GOL UEFA EUROPEAS 2025 🏆
Análisis de Champions League, Europa League y Conference League
Modelo: Dixon-Coles + Factor Local/Visitante
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import requests
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="D-GOL UEFA Europeas",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_KEY = '15edaff34ce64f30bbad0f0b3685e600'
MIN_PARTIDOS = 10  # UEFA tiene menos partidos por equipo

# IDs de competiciones UEFA en football-data.org
COMPETICIONES_UEFA = {
    'Champions League': 2001,
    'Europa League': 2146,
    'Conference League': 2149
}

# ============================================================================
# FUNCIONES BACKEND
# ============================================================================

@st.cache_data(ttl=14400)  # Cache por 4 horas
def cargar_datos_uefa(competicion_id, nombre_comp):
    """Carga datos de UEFA desde football-data.org API"""
    headers = {'X-Auth-Token': API_KEY}
    url = f'https://api.football-data.org/v4/competitions/{competicion_id}/matches?status=FINISHED'
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            st.error("⚠️ Límite de API alcanzado. Espera 1 minuto.")
            return None
        
        if response.status_code != 200:
            st.error(f"❌ Error API: {response.status_code}")
            return None
        
        data = response.json()
        matches = data.get('matches', [])
        
        if not matches:
            st.warning(f"⚠️ No hay partidos finalizados en {nombre_comp}")
            return None
        
        # Convertir a DataFrame
        rows = []
        for match in matches:
            if match['status'] == 'FINISHED' and match['score']['fullTime']['home'] is not None:
                score = match['score']['fullTime']
                
                # Determinar resultado
                if score['home'] > score['away']:
                    ftr = 'H'
                elif score['home'] < score['away']:
                    ftr = 'A'
                else:
                    ftr = 'D'
                
                rows.append({
                    'Date': pd.to_datetime(match['utcDate']),
                    'HomeTeam': match['homeTeam']['name'],
                    'AwayTeam': match['awayTeam']['name'],
                    'FTHG': score['home'],
                    'FTAG': score['away'],
                    'FTR': ftr
                })
        
        df = pd.DataFrame(rows)
        
        if len(df) < MIN_PARTIDOS:
            st.warning(f"⚠️ Pocos partidos en {nombre_comp} ({len(df)}). Necesita al menos {MIN_PARTIDOS}.")
            return None
        
        # Calcular modelo
        modelo = calcular_modelo_mejorado(df)
        
        if modelo:
            return {
                'df': df,
                'modelo': modelo,
                'equipos': sorted(list(set(df['HomeTeam']) | set(df['AwayTeam']))),
                'fecha_actualizacion': datetime.now(),
                'total_partidos': len(df)
            }
        
        return None
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def calcular_modelo_mejorado(df):
    """Modelo Dixon-Coles + estadísticas LOCAL/VISITANTE"""
    equipos = sorted(set(df['HomeTeam']) | set(df['AwayTeam']))
    n = len(equipos)
    
    if n < 4 or len(df) < MIN_PARTIDOS:
        return None
    
    eq_idx = {e: i for i, e in enumerate(equipos)}
    params_iniciales = np.concatenate([[0.3], np.zeros(n), np.zeros(n)])
    
    def log_likelihood(params):
        ha, ataque, defensa = params[0], params[1:n+1], params[n+1:]
        ll = 0
        for _, row in df.iterrows():
            try:
                h_idx, a_idx = eq_idx[row['HomeTeam']], eq_idx[row['AwayTeam']]
                lambda_h = np.exp(ha + ataque[h_idx] + defensa[a_idx])
                lambda_a = np.exp(ataque[a_idx] + defensa[h_idx])
                prob = poisson.pmf(int(row['FTHG']), lambda_h) * poisson.pmf(int(row['FTAG']), lambda_a)
                ll += np.log(prob + 1e-10)
            except:
                continue
        return -ll
    
    try:
        res = minimize(log_likelihood, params_iniciales, method='L-BFGS-B', options={'maxiter': 50})
        ha, ataque, defensa = res.x[0], res.x[1:n+1], res.x[n+1:]
        params_df = pd.DataFrame({'Equipo': equipos, 'Ataque': ataque, 'Defensa': defensa})
        
        # ESTADÍSTICAS SEPARADAS LOCAL/VISITANTE
        stats = {}
        for equipo in equipos:
            p_local = df[df['HomeTeam'] == equipo]
            p_visitante = df[df['AwayTeam'] == equipo]
            total_local = len(p_local)
            total_visit = len(p_visitante)
            total = total_local + total_visit
            
            if total > 0:
                # RENDIMIENTO LOCAL
                goles_local_casa = p_local['FTHG'].sum()
                goles_contra_local_casa = p_local['FTAG'].sum()
                victorias_local = len(p_local[p_local['FTR'] == 'H'])
                empates_local = len(p_local[p_local['FTR'] == 'D'])
                derrotas_local = len(p_local[p_local['FTR'] == 'A'])
                
                # RENDIMIENTO VISITANTE
                goles_visit_fuera = p_visitante['FTAG'].sum()
                goles_contra_visit_fuera = p_visitante['FTHG'].sum()
                victorias_visit = len(p_visitante[p_visitante['FTR'] == 'A'])
                empates_visit = len(p_visitante[p_visitante['FTR'] == 'D'])
                derrotas_visit = len(p_visitante[p_visitante['FTR'] == 'H'])
                
                # CALCULAR ÍNDICES DE RENDIMIENTO
                ptos_local = (victorias_local * 3 + empates_local) / max(total_local * 3, 1) * 100
                ptos_visit = (victorias_visit * 3 + empates_visit) / max(total_visit * 3, 1) * 100
                
                stats[equipo] = {
                    'partidos_total': total,
                    'partidos_local': total_local,
                    'partidos_visit': total_visit,
                    'goles_favor_local': goles_local_casa / total_local if total_local > 0 else 0,
                    'goles_contra_local': goles_contra_local_casa / total_local if total_local > 0 else 0,
                    'victorias_local': victorias_local,
                    'empates_local': empates_local,
                    'derrotas_local': derrotas_local,
                    'ptos_local_pct': ptos_local,
                    'goles_favor_visit': goles_visit_fuera / total_visit if total_visit > 0 else 0,
                    'goles_contra_visit': goles_contra_visit_fuera / total_visit if total_visit > 0 else 0,
                    'victorias_visit': victorias_visit,
                    'empates_visit': empates_visit,
                    'derrotas_visit': derrotas_visit,
                    'ptos_visit_pct': ptos_visit,
                    'diferencia_goles_local_visit': (goles_local_casa / max(total_local, 1)) - (goles_visit_fuera / max(total_visit, 1)),
                    'diferencia_ptos_local_visit': ptos_local - ptos_visit,
                }
        
        return {'params': params_df, 'ha': ha, 'stats': stats}
    except:
        return None

def analizar_partido_uefa(datos, equipo_local, equipo_visitante):
    """Análisis completo de partido UEFA"""
    if not datos or not datos['modelo']:
        return None
    
    df, modelo = datos['df'], datos['modelo']
    params, ha, stats = modelo['params'], modelo['ha'], modelo['stats']
    
    if equipo_local not in params['Equipo'].values or equipo_visitante not in params['Equipo'].values:
        return None
    
    try:
        ath = params[params['Equipo'] == equipo_local]['Ataque'].values[0]
        deh = params[params['Equipo'] == equipo_local]['Defensa'].values[0]
        ata = params[params['Equipo'] == equipo_visitante]['Ataque'].values[0]
        dea = params[params['Equipo'] == equipo_visitante]['Defensa'].values[0]
    except:
        return None
    
    # OBTENER ESTADÍSTICAS LOCAL/VISITANTE
    stats_local = stats.get(equipo_local, {})
    stats_visit = stats.get(equipo_visitante, {})
    
    # AJUSTAR LAMBDA CON FACTOR LOCAL/VISITANTE
    lambda_h_base = np.exp(ha + ath + dea)
    lambda_a_base = np.exp(ata + deh)
    
    # Factor de ajuste basado en rendimiento local/visitante
    factor_local_goles = stats_local.get('goles_favor_local', 1.5) / max(stats_local.get('goles_favor_visit', 1.0), 0.5)
    factor_visit_goles = stats_visit.get('goles_favor_visit', 1.0) / max(stats_visit.get('goles_favor_local', 1.5), 0.5)
    
    # Aplicar ajuste moderado (15% máximo)
    ajuste_local = 1 + (factor_local_goles - 1) * 0.15
    ajuste_visit = 1 + (factor_visit_goles - 1) * 0.15
    
    lambda_h = lambda_h_base * ajuste_local
    lambda_a = lambda_a_base * ajuste_visit
    
    # Matrices de probabilidades
    max_goles = 10
    matriz = np.array([[poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a) 
                       for j in range(max_goles)] for i in range(max_goles)])
    
    lambda_h_ht, lambda_a_ht = lambda_h * 0.45, lambda_a * 0.45
    matriz_ht = np.array([[poisson.pmf(i, lambda_h_ht) * poisson.pmf(j, lambda_a_ht) 
                          for j in range(8)] for i in range(8)])
    
    # Probabilidades básicas
    prob_over_05_ht = (1 - matriz_ht[0, 0]) * 100
    prob_over_15_ht = sum(matriz_ht[i, j] for i in range(8) for j in range(8) if i + j > 1.5) * 100
    prob_btts_si = (1 - (poisson.pmf(0, lambda_h) + poisson.pmf(0, lambda_a) - 
                         poisson.pmf(0, lambda_h) * poisson.pmf(0, lambda_a))) * 100
    prob_over_15 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 1.5) * 100
    prob_over_25 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 2.5) * 100
    prob_over_35 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 3.5) * 100
    prob_local = np.sum(np.tril(matriz, -1)) * 100
    prob_empate = np.sum(np.diag(matriz)) * 100
    prob_visitante = np.sum(np.triu(matriz, 1)) * 100
    
    # GOLES POR EQUIPO
    prob_local_over_05 = (1 - poisson.pmf(0, lambda_h)) * 100
    prob_local_over_15 = (1 - poisson.cdf(1, lambda_h)) * 100
    prob_local_over_25 = (1 - poisson.cdf(2, lambda_h)) * 100
    
    prob_visit_over_05 = (1 - poisson.pmf(0, lambda_a)) * 100
    prob_visit_over_15 = (1 - poisson.cdf(1, lambda_a)) * 100
    prob_visit_over_25 = (1 - poisson.cdf(2, lambda_a)) * 100
    
    return {
        'goles_esperados_total': lambda_h + lambda_a,
        'lambda_local': lambda_h,
        'lambda_visitante': lambda_a,
        'prob_over_05_ht': prob_over_05_ht,
        'prob_over_15_ht': prob_over_15_ht,
        'prob_btts_si': prob_btts_si,
        'prob_over_15': prob_over_15,
        'prob_over_25': prob_over_25,
        'prob_over_35': prob_over_35,
        'prob_local': prob_local,
        'prob_empate': prob_empate,
        'prob_visitante': prob_visitante,
        'cuota_local': 100 / prob_local if prob_local > 0 else 999,
        'cuota_empate': 100 / prob_empate if prob_empate > 0 else 999,
        'cuota_visitante': 100 / prob_visitante if prob_visitante > 0 else 999,
        'local_over_05': prob_local_over_05,
        'local_over_15': prob_local_over_15,
        'local_over_25': prob_local_over_25,
        'visit_over_05': prob_visit_over_05,
        'visit_over_15': prob_visit_over_15,
        'visit_over_25': prob_visit_over_25,
        'matriz': matriz,
        'prob_1x': prob_local + prob_empate,
        'prob_x2': prob_empate + prob_visitante,
        'prob_12': prob_local + prob_visitante,
        'stats_local': stats_local,
        'stats_visit': stats_visit,
        'ajuste_aplicado': {
            'factor_local': ajuste_local,
            'factor_visit': ajuste_visit
        }
    }

# ============================================================================
# INTERFAZ
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {background-color: #0e1117;}
    
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px 24px;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #1e40af;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4);
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align: center; color: #1e3a8a; font-size: 42px;">🏆 D-GOL UEFA EUROPEAS 2025 🏆</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-weight: 500;'>Champions League • Europa League • Conference League</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Modelo Dixon-Coles + Factor Local/Visitante</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuración")
    
    if st.button("🔄 Actualizar Datos", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Caché limpiado")
        st.rerun()
    
    competicion = st.selectbox("🏆 Competición:", list(COMPETICIONES_UEFA.keys()))
    
    with st.spinner(f"Cargando {competicion}..."):
        datos = cargar_datos_uefa(COMPETICIONES_UEFA[competicion], competicion)
    
    if datos:
        st.success(f"✅ {len(datos['equipos'])} equipos")
        st.info(f"📊 {datos['total_partidos']} partidos analizados")
        st.caption(f"📅 Actualizado: {datos['fecha_actualizacion'].strftime('%d/%m/%Y %H:%M')}")
        st.caption("♻️ Se actualiza cada 4 horas")
        
        local = st.selectbox("🏠 Local:", datos['equipos'])
        visitante = st.selectbox("✈️ Visitante:", [e for e in datos['equipos'] if e != local])
        analizar_btn = st.button("🔍 ANALIZAR PARTIDO", type="primary", use_container_width=True)
    else:
        st.error("❌ No se pudieron cargar datos")
        analizar_btn = False

# Main
if analizar_btn and datos:
    with st.spinner("🔍 Analizando partido UEFA..."):
        resultado = analizar_partido_uefa(datos, local, visitante)
    
    if resultado:
        # Header partido
        st.markdown(f"## 🏟️ {local} vs {visitante}")
        st.markdown(f"**Competición:** {competicion}")
        st.markdown("---")
        
        # Goles esperados
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("⚽ GOLES ESPERADOS TOTAL", f"{resultado['goles_esperados_total']:.2f}",
                     delta=f"🏠 {resultado['lambda_local']:.2f} | ✈️ {resultado['lambda_visitante']:.2f}")
        
        st.markdown("---")
        
        # TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Probabilidades", 
            "⚽ Goles por Equipo",
            "🏠 Local vs ✈️ Visitante",
            "📈 Gráficos", 
            "💾 Exportar"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Resultados Descanso")
                st.metric("Over 0.5 HT", f"{resultado['prob_over_05_ht']:.1f}%")
                st.metric("Over 1.5 HT", f"{resultado['prob_over_15_ht']:.1f}%")
                
                st.markdown("---")
                st.subheader("⚽⚽ BTTS (Ambos Marcan)")
                st.metric("Sí", f"{resultado['prob_btts_si']:.1f}%")
                st.metric("No", f"{100 - resultado['prob_btts_si']:.1f}%")
                
                st.markdown("---")
                st.subheader("📈 Over/Under Total")
                st.metric("Over 1.5", f"{resultado['prob_over_15']:.1f}%")
                st.metric("Over 2.5", f"{resultado['prob_over_25']:.1f}%")
                st.metric("Over 3.5", f"{resultado['prob_over_35']:.1f}%")
            
            with col2:
                st.subheader("📊 Resultado Final 1X2")
                col_1, col_x, col_2 = st.columns(3)
                with col_1:
                    st.metric("🏠 Local", f"{resultado['prob_local']:.1f}%")
                    st.info(f"Cuota: {resultado['cuota_local']:.2f}")
                with col_x:
                    st.metric("⚖️ Empate", f"{resultado['prob_empate']:.1f}%")
                    st.info(f"Cuota: {resultado['cuota_empate']:.2f}")
                with col_2:
                    st.metric("✈️ Visitante", f"{resultado['prob_visitante']:.1f}%")
                    st.info(f"Cuota: {resultado['cuota_visitante']:.2f}")
                
                st.markdown("---")
                st.subheader("🤝 Doble Oportunidad")
                st.metric("1X (Local o Empate)", f"{resultado['prob_1x']:.1f}%")
                st.metric("X2 (Empate o Visitante)", f"{resultado['prob_x2']:.1f}%")
                st.metric("12 (Local o Visitante)", f"{resultado['prob_12']:.1f}%")
        
        with tab2:
            st.subheader("⚽ GOLES POR EQUIPO")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏠 {local}")
                st.metric(f"Goles esperados", f"{resultado['lambda_local']:.2f}")
                st.markdown("---")
                st.metric("Over 0.5 goles", f"{resultado['local_over_05']:.1f}%")
                st.metric("Over 1.5 goles", f"{resultado['local_over_15']:.1f}%")
                st.metric("Over 2.5 goles", f"{resultado['local_over_25']:.1f}%")
            
            with col2:
                st.markdown(f"### ✈️ {visitante}")
                st.metric(f"Goles esperados", f"{resultado['lambda_visitante']:.2f}")
                st.markdown("---")
                st.metric("Over 0.5 goles", f"{resultado['visit_over_05']:.1f}%")
                st.metric("Over 1.5 goles", f"{resultado['visit_over_15']:.1f}%")
                st.metric("Over 2.5 goles", f"{resultado['visit_over_25']:.1f}%")
        
        with tab3:
            st.subheader("🏠 Análisis Local vs ✈️ Visitante")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏠 {local} (LOCAL)")
                stats_l = resultado['stats_local']
                
                st.metric("Partidos en casa", stats_l.get('partidos_local', 0))
                st.metric("Goles a favor (casa)", f"{stats_l.get('goles_favor_local', 0):.2f} por partido")
                st.metric("Goles en contra (casa)", f"{stats_l.get('goles_contra_local', 0):.2f} por partido")
                
                st.markdown("---")
                st.markdown("**Resultados en casa:**")
                st.metric("Victorias", stats_l.get('victorias_local', 0))
                st.metric("Empates", stats_l.get('empates_local', 0))
                st.metric("Derrotas", stats_l.get('derrotas_local', 0))
                st.metric("% Puntos obtenidos", f"{stats_l.get('ptos_local_pct', 0):.1f}%")
            
            with col2:
                st.markdown(f"### ✈️ {visitante} (VISITANTE)")
                stats_v = resultado['stats_visit']
                
                st.metric("Partidos fuera", stats_v.get('partidos_visit', 0))
                st.metric("Goles a favor (fuera)", f"{stats_v.get('goles_favor_visit', 0):.2f} por partido")
                st.metric("Goles en contra (fuera)", f"{stats_v.get('goles_contra_visit', 0):.2f} por partido")
                
                st.markdown("---")
                st.markdown("**Resultados fuera:**")
                st.metric("Victorias", stats_v.get('victorias_visit', 0))
                st.metric("Empates", stats_v.get('empates_visit', 0))
                st.metric("Derrotas", stats_v.get('derrotas_visit', 0))
                st.metric("% Puntos obtenidos", f"{stats_v.get('ptos_visit_pct', 0):.1f}%")
        
        with tab4:
            st.subheader("📊 Visualizaciones")
            
            # Gráfico 1X2
            fig1 = go.Figure(data=[
                go.Bar(name='Probabilidad', x=['Local', 'Empate', 'Visitante'],
                      y=[resultado['prob_local'], resultado['prob_empate'], resultado['prob_visitante']],
                      marker_color=['#1e3a8a', '#f59e0b', '#10b981'],
                      text=[f"{resultado['prob_local']:.1f}%", f"{resultado['prob_empate']:.1f}%", f"{resultado['prob_visitante']:.1f}%"],
                      textposition='outside')
            ])
            fig1.update_layout(title='Probabilidades 1X2', yaxis_title='Probabilidad (%)', height=400, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Matriz resultados exactos
            st.subheader("🎯 Matriz de Resultados Exactos (Top 6x6)")
            matriz_display = resultado['matriz'][:6, :6] * 100
            fig3 = px.imshow(matriz_display,
                            labels=dict(x="Goles Visitante", y="Goles Local", color="Prob. (%)"),
                            x=[str(i) for i in range(6)],
                            y=[str(i) for i in range(6)],
                            color_continuous_scale='Blues',
                            text_auto='.1f')
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab5:
            st.subheader("💾 Exportar Análisis")
            
            reporte = f"""
D-GOL UEFA EUROPEAS 2025 - ANÁLISIS COMPLETO
{"="*75}

Partido: {local} vs {visitante}
Competición: {competicion}
Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}

{"="*75}

GOLES ESPERADOS:
- Total: {resultado['goles_esperados_total']:.2f}
- Local: {resultado['lambda_local']:.2f}
- Visitante: {resultado['lambda_visitante']:.2f}

1X2:
- Local: {resultado['prob_local']:.1f}% (Cuota: {resultado['cuota_local']:.2f})
- Empate: {resultado['prob_empate']:.1f}% (Cuota: {resultado['cuota_empate']:.2f})
- Visitante: {resultado['prob_visitante']:.1f}% (Cuota: {resultado['cuota_visitante']:.2f})

GOLES POR EQUIPO - {local}:
- Over 0.5: {resultado['local_over_05']:.1f}%
- Over 1.5: {resultado['local_over_15']:.1f}%
- Over 2.5: {resultado['local_over_25']:.1f}%

GOLES POR EQUIPO - {visitante}:
- Over 0.5: {resultado['visit_over_05']:.1f}%
- Over 1.5: {resultado['visit_over_15']:.1f}%
- Over 2.5: {resultado['visit_over_25']:.1f}%

BTTS:
- Sí: {resultado['prob_btts_si']:.1f}%
- No: {100 - resultado['prob_btts_si']:.1f}%

{"="*75}
Generado por D-GOL UEFA Europeas 2025
Modelo: Dixon-Coles + Factor Local/Visitante
            """
            
            st.download_button(
                label="💾 Descargar Análisis Completo (TXT)",
                data=reporte,
                file_name=f"dgol_uefa_{local}_vs_{visitante}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Recomendaciones
        st.markdown("---")
        st.subheader("💡 Recomendaciones Inteligentes UEFA")
        
        recomendaciones = []
        
        if resultado['prob_over_25'] > 65:
            recomendaciones.append(("✅ Over 2.5 Total", "Alta probabilidad", "success"))
        
        if resultado['prob_btts_si'] > 65:
            recomendaciones.append(("✅ BTTS Sí", "Muy probable", "success"))
        
        if resultado['local_over_15'] > 65:
            recomendaciones.append((f"✅ {local} Over 1.5", "Alta probabilidad", "success"))
        
        if resultado['visit_over_15'] > 65:
            recomendaciones.append((f"✅ {visitante} Over 1.5", "Alta probabilidad", "success"))
        
        if resultado['prob_local'] > 55 and resultado['stats_local'].get('ptos_local_pct', 0) > 60:
            recomendaciones.append((f"✅ Victoria {local}", "Fuerte en casa", "success"))
        
        if recomendaciones:
            for rec, desc, tipo in recomendaciones:
                if tipo == "success":
                    st.success(f"{rec} - {desc}")
        else:
            st.info("ℹ️ No hay recomendaciones con alta confianza para este partido.")
    
    else:
        st.error("❌ No se pudo analizar el partido")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>🏆 D-GOL UEFA Europeas 2025 | Modelo Dixon-Coles + ML</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'>Datos: football-data.org API | Actualización automática cada 4 horas</p>", unsafe_allow_html=True)
