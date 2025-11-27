"""
🏆 D-GOL UEFA EUROPEAS v2.0 - MODELO AVANZADO ⚡
Análisis de Champions League, Europa League y Conference League
Modelo: Dixon-Coles + Time Decay + Forma + H2H
100% GRATIS con Web Scraping optimizado
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
import time

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="D-GOL UEFA Europeas v2.0",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

MIN_PARTIDOS = 10

COMPETICIONES_UEFA = {
    'Champions League': 'https://fbref.com/en/comps/8/schedule/Champions-League-Scores-and-Fixtures',
    'Europa League': 'https://fbref.com/en/comps/19/schedule/Europa-League-Scores-and-Fixtures',
    'Conference League': 'https://fbref.com/en/comps/882/schedule/Europa-Conference-League-Scores-and-Fixtures'
}

# ============================================================================
# FUNCIONES DE WEB SCRAPING OPTIMIZADO
# ============================================================================

@st.cache_data(ttl=21600)  # Caché de 6 horas
def scrape_fbref_matches(url, nombre_comp):
    """Extrae partidos desde FBref (gratis y rápido con lxml)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        with st.spinner(f'🔄 Cargando {nombre_comp}...'):
            response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            st.error(f"❌ Error al cargar {nombre_comp}: {response.status_code}")
            return None
        
        # Usar lxml para parsing más rápido
        soup = BeautifulSoup(response.content, 'lxml')
        tabla = soup.find('table', {'id': 'sched_all'})
        
        if not tabla:
            st.warning(f"⚠️ No se encontró tabla en {nombre_comp}")
            return None
        
        rows = []
        tbody = tabla.find('tbody')
        
        if not tbody:
            return None
        
        for tr in tbody.find_all('tr'):
            if tr.find('th', {'data-stat': 'date'}):
                continue
            
            try:
                date_td = tr.find('td', {'data-stat': 'date'})
                home_td = tr.find('td', {'data-stat': 'home_team'})
                away_td = tr.find('td', {'data-stat': 'away_team'})
                score_td = tr.find('td', {'data-stat': 'score'})
                
                if not all([date_td, home_td, away_td, score_td]):
                    continue
                
                score_text = score_td.get_text(strip=True)
                
                # Solo partidos finalizados
                if '–' in score_text or not score_text:
                    continue
                
                # Limpiar score
                if '(' in score_text:
                    score_text = score_text.split('(')[0].strip()
                
                parts = score_text.split('–')
                if len(parts) != 2:
                    continue
                
                home_goals = int(parts[0].strip())
                away_goals = int(parts[1].strip())
                
                # Determinar resultado
                if home_goals > away_goals:
                    ftr = 'H'
                elif home_goals < away_goals:
                    ftr = 'A'
                else:
                    ftr = 'D'
                
                fecha = pd.to_datetime(date_td.get_text(strip=True), format='%Y-%m-%d', errors='coerce')
                
                if pd.isna(fecha):
                    continue
                
                rows.append({
                    'Date': fecha,
                    'HomeTeam': home_td.get_text(strip=True),
                    'AwayTeam': away_td.get_text(strip=True),
                    'FTHG': home_goals,
                    'FTAG': away_goals,
                    'FTR': ftr
                })
                
            except Exception:
                continue
        
        if not rows:
            st.warning(f"⚠️ No se encontraron partidos finalizados en {nombre_comp}")
            return None
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Date').reset_index(drop=True)
        
        if len(df) < MIN_PARTIDOS:
            st.warning(f"⚠️ Solo {len(df)} partidos en {nombre_comp}. Mínimo: {MIN_PARTIDOS}")
            return None
        
        # Calcular modelo
        modelo = calcular_modelo_avanzado(df)
        
        if modelo:
            st.success(f"✅ {len(df)} partidos cargados correctamente")
            return {
                'df': df,
                'modelo': modelo,
                'equipos': sorted(list(set(df['HomeTeam']) | set(df['AwayTeam']))),
                'fecha_actualizacion': datetime.now(),
                'total_partidos': len(df)
            }
        
        return None
        
    except Exception as e:
        st.error(f"❌ Error scraping {nombre_comp}: {e}")
        return None

# ============================================================================
# FUNCIONES DEL MODELO AVANZADO
# ============================================================================

def calcular_peso_temporal(fecha, fecha_max, xi=0.003):
    """Time decay exponencial - partidos recientes pesan más"""
    dias_diff = (fecha_max - fecha).days
    peso = np.exp(-xi * dias_diff)
    return peso

def correccion_dixon_coles(home_goals, away_goals, lambda_h, lambda_a, rho=0.1):
    """Corrección Dixon-Coles para resultados de bajo puntaje (0-0, 1-0, 0-1, 1-1)"""
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_h * lambda_a * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_h * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_a * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0

def calcular_forma_reciente(df, equipo, local=True, ultimos=5):
    """Calcula forma reciente (últimos 5 partidos)"""
    if local:
        partidos = df[df['HomeTeam'] == equipo].tail(ultimos)
        goles_favor = partidos['FTHG'].mean()
        goles_contra = partidos['FTAG'].mean()
        victorias = len(partidos[partidos['FTR'] == 'H'])
        empates = len(partidos[partidos['FTR'] == 'D'])
    else:
        partidos = df[df['AwayTeam'] == equipo].tail(ultimos)
        goles_favor = partidos['FTAG'].mean()
        goles_contra = partidos['FTHG'].mean()
        victorias = len(partidos[partidos['FTR'] == 'A'])
        empates = len(partidos[partidos['FTR'] == 'D'])
    
    if len(partidos) == 0:
        return 0, 0, 0
    
    puntos_promedio = (victorias * 3 + empates) / len(partidos)
    return puntos_promedio, goles_favor if not np.isnan(goles_favor) else 0, goles_contra if not np.isnan(goles_contra) else 0

def calcular_head_to_head(df, equipo1, equipo2):
    """Historial directo entre dos equipos"""
    h2h = df[((df['HomeTeam'] == equipo1) & (df['AwayTeam'] == equipo2)) | 
             ((df['HomeTeam'] == equipo2) & (df['AwayTeam'] == equipo1))]
    
    if len(h2h) == 0:
        return {'partidos': 0, 'goles_eq1': 0, 'goles_eq2': 0, 'victorias_eq1': 0}
    
    goles_eq1 = 0
    goles_eq2 = 0
    victorias_eq1 = 0
    
    for _, row in h2h.iterrows():
        if row['HomeTeam'] == equipo1:
            goles_eq1 += row['FTHG']
            goles_eq2 += row['FTAG']
            if row['FTR'] == 'H':
                victorias_eq1 += 1
        else:
            goles_eq1 += row['FTAG']
            goles_eq2 += row['FTHG']
            if row['FTR'] == 'A':
                victorias_eq1 += 1
    
    return {
        'partidos': len(h2h),
        'goles_eq1': goles_eq1 / len(h2h),
        'goles_eq2': goles_eq2 / len(h2h),
        'victorias_eq1': victorias_eq1
    }

def calcular_modelo_avanzado(df):
    """
    Modelo Dixon-Coles MEJORADO con:
    - Time decay (peso temporal)
    - Corrección Dixon-Coles
    - Ajuste por calidad de rival
    """
    equipos = sorted(set(df['HomeTeam']) | set(df['AwayTeam']))
    n = len(equipos)
    
    if n < 4 or len(df) < MIN_PARTIDOS:
        return None
    
    eq_idx = {e: i for i, e in enumerate(equipos)}
    
    # Calcular pesos temporales
    fecha_max = df['Date'].max()
    df['peso'] = df['Date'].apply(lambda x: calcular_peso_temporal(x, fecha_max))
    
    # Parámetros iniciales: [ha, rho, ataque1...ataqueN, defensa1...defensaN]
    params_iniciales = np.concatenate([[0.3, 0.1], np.zeros(n), np.zeros(n)])
    
    def log_likelihood_mejorado(params):
        ha = params[0]
        rho = params[1]
        ataque = params[2:n+2]
        defensa = params[n+2:]
        
        ll = 0
        for _, row in df.iterrows():
            try:
                h_idx = eq_idx[row['HomeTeam']]
                a_idx = eq_idx[row['AwayTeam']]
                
                lambda_h = np.exp(ha + ataque[h_idx] + defensa[a_idx])
                lambda_a = np.exp(ataque[a_idx] + defensa[h_idx])
                
                # Probabilidad base de Poisson
                prob = poisson.pmf(int(row['FTHG']), lambda_h) * poisson.pmf(int(row['FTAG']), lambda_a)
                
                # Aplicar corrección Dixon-Coles
                tau = correccion_dixon_coles(int(row['FTHG']), int(row['FTAG']), lambda_h, lambda_a, rho)
                prob_corregida = prob * tau
                
                # Aplicar peso temporal
                peso = row['peso']
                
                ll += peso * np.log(prob_corregida + 1e-10)
            except:
                continue
        
        return -ll
    
    try:
        # Optimización con límites para rho (-0.5 a 0.5)
        bounds = [(None, None), (-0.5, 0.5)] + [(None, None)] * (2 * n)
        res = minimize(log_likelihood_mejorado, params_iniciales, method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': 100})
        
        ha = res.x[0]
        rho = res.x[1]
        ataque = res.x[2:n+2]
        defensa = res.x[n+2:]
        
        params_df = pd.DataFrame({
            'Equipo': equipos, 
            'Ataque': ataque, 
            'Defensa': defensa,
            'Fuerza_Total': ataque - defensa
        })
        
        # ESTADÍSTICAS COMPLETAS
        stats = {}
        for equipo in equipos:
            p_local = df[df['HomeTeam'] == equipo]
            p_visitante = df[df['AwayTeam'] == equipo]
            total_local = len(p_local)
            total_visit = len(p_visitante)
            total = total_local + total_visit
            
            if total > 0:
                # Rendimiento local
                goles_local_casa = p_local['FTHG'].sum()
                goles_contra_local_casa = p_local['FTAG'].sum()
                victorias_local = len(p_local[p_local['FTR'] == 'H'])
                empates_local = len(p_local[p_local['FTR'] == 'D'])
                derrotas_local = len(p_local[p_local['FTR'] == 'A'])
                
                # Rendimiento visitante
                goles_visit_fuera = p_visitante['FTAG'].sum()
                goles_contra_visit_fuera = p_visitante['FTHG'].sum()
                victorias_visit = len(p_visitante[p_visitante['FTR'] == 'A'])
                empates_visit = len(p_visitante[p_visitante['FTR'] == 'D'])
                derrotas_visit = len(p_visitante[p_visitante['FTR'] == 'H'])
                
                # Forma reciente
                forma_local = calcular_forma_reciente(df, equipo, local=True)
                forma_visit = calcular_forma_reciente(df, equipo, local=False)
                
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
                    'forma_local_ptos': forma_local[0],
                    'forma_local_gf': forma_local[1],
                    'forma_local_gc': forma_local[2],
                    'forma_visit_ptos': forma_visit[0],
                    'forma_visit_gf': forma_visit[1],
                    'forma_visit_gc': forma_visit[2],
                }
        
        return {
            'params': params_df, 
            'ha': ha, 
            'rho': rho,
            'stats': stats,
            'df': df
        }
    except Exception as e:
        st.error(f"Error en optimización: {e}")
        return None

def analizar_partido_uefa_avanzado(datos, equipo_local, equipo_visitante):
    """Análisis completo con todas las mejoras del modelo"""
    if not datos or not datos['modelo']:
        return None
    
    df, modelo = datos['df'], datos['modelo']
    params, ha, rho, stats = modelo['params'], modelo['ha'], modelo['rho'], modelo['stats']
    
    if equipo_local not in params['Equipo'].values or equipo_visitante not in params['Equipo'].values:
        return None
    
    try:
        ath = params[params['Equipo'] == equipo_local]['Ataque'].values[0]
        deh = params[params['Equipo'] == equipo_local]['Defensa'].values[0]
        ata = params[params['Equipo'] == equipo_visitante]['Ataque'].values[0]
        dea = params[params['Equipo'] == equipo_visitante]['Defensa'].values[0]
        fuerza_local = params[params['Equipo'] == equipo_local]['Fuerza_Total'].values[0]
        fuerza_visit = params[params['Equipo'] == equipo_visitante]['Fuerza_Total'].values[0]
    except:
        return None
    
    # Estadísticas
    stats_local = stats.get(equipo_local, {})
    stats_visit = stats.get(equipo_visitante, {})
    
    # Head to head
    h2h = calcular_head_to_head(df, equipo_local, equipo_visitante)
    
    # Lambda base
    lambda_h_base = np.exp(ha + ath + dea)
    lambda_a_base = np.exp(ata + deh)
    
    # AJUSTE POR FORMA RECIENTE (20% de influencia)
    forma_local_factor = 1 + (stats_local.get('forma_local_ptos', 1.5) - 1.5) * 0.2
    forma_visit_factor = 1 + (stats_visit.get('forma_visit_ptos', 1.5) - 1.5) * 0.2
    
    # AJUSTE POR CALIDAD DE RIVAL (10% de influencia)
    if fuerza_local > fuerza_visit:
        ajuste_calidad_local = 1.05
        ajuste_calidad_visit = 0.95
    elif fuerza_visit > fuerza_local:
        ajuste_calidad_local = 0.95
        ajuste_calidad_visit = 1.05
    else:
        ajuste_calidad_local = 1.0
        ajuste_calidad_visit = 1.0
    
    # AJUSTE POR H2H (5% si hay historial)
    if h2h['partidos'] >= 3:
        if h2h['victorias_eq1'] > h2h['partidos'] / 2:
            ajuste_h2h_local = 1.03
            ajuste_h2h_visit = 0.97
        else:
            ajuste_h2h_local = 0.97
            ajuste_h2h_visit = 1.03
    else:
        ajuste_h2h_local = 1.0
        ajuste_h2h_visit = 1.0
    
    # Aplicar todos los ajustes
    lambda_h = lambda_h_base * forma_local_factor * ajuste_calidad_local * ajuste_h2h_local
    lambda_a = lambda_a_base * forma_visit_factor * ajuste_calidad_visit * ajuste_h2h_visit
    
    # Matrices con corrección Dixon-Coles
    max_goles = 10
    matriz = np.zeros((max_goles, max_goles))
    
    for i in range(max_goles):
        for j in range(max_goles):
            prob_base = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
            tau = correccion_dixon_coles(i, j, lambda_h, lambda_a, rho)
            matriz[i, j] = prob_base * tau
    
    # Normalizar matriz
    matriz = matriz / matriz.sum()
    
    # Calcular probabilidades
    prob_over_15 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 1.5) * 100
    prob_over_25 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 2.5) * 100
    prob_over_35 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 3.5) * 100
    prob_local = np.sum(np.tril(matriz, -1)) * 100
    prob_empate = np.sum(np.diag(matriz)) * 100
    prob_visitante = np.sum(np.triu(matriz, 1)) * 100
    
    prob_btts_si = (1 - (poisson.pmf(0, lambda_h) + poisson.pmf(0, lambda_a) - 
                         poisson.pmf(0, lambda_h) * poisson.pmf(0, lambda_a))) * 100
    
    # Goles por equipo
    prob_local_over_05 = (1 - poisson.pmf(0, lambda_h)) * 100
    prob_local_over_15 = (1 - poisson.cdf(1, lambda_h)) * 100
    prob_local_over_25 = (1 - poisson.cdf(2, lambda_h)) * 100
    
    prob_visit_over_05 = (1 - poisson.pmf(0, lambda_a)) * 100
    prob_visit_over_15 = (1 - poisson.cdf(1, lambda_a)) * 100
    prob_visit_over_25 = (1 - poisson.cdf(2, lambda_a)) * 100
    
    # Resultado más probable
    resultado_mas_probable = np.unravel_index(matriz.argmax(), matriz.shape)
    prob_resultado_mp = matriz[resultado_mas_probable] * 100
    
    return {
        'goles_esperados_total': lambda_h + lambda_a,
        'lambda_local': lambda_h,
        'lambda_visitante': lambda_a,
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
        'h2h': h2h,
        'fuerza_local': fuerza_local,
        'fuerza_visit': fuerza_visit,
        'resultado_mas_probable': resultado_mas_probable,
        'prob_resultado_mp': prob_resultado_mp,
        'rho': rho,
        'ajustes': {
            'forma_local': forma_local_factor,
            'forma_visit': forma_visit_factor,
            'calidad_local': ajuste_calidad_local,
            'calidad_visit': ajuste_calidad_visit,
            'h2h_local': ajuste_h2h_local,
            'h2h_visit': ajuste_h2h_visit
        }
    }

# ============================================================================
# INTERFAZ
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    .main {background-color: #0e1117;}
    .stButton>button {
        background-color: #1e3a8a; color: white; font-weight: 600;
        border-radius: 10px; padding: 12px 24px; transition: all 0.3s; border: none;
    }
    .stButton>button:hover {
        background-color: #1e40af; transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4);
    }
    h1, h2, h3 {font-family: 'Inter', sans-serif !important; font-weight: 700 !important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center; color: #1e3a8a; font-size: 42px;">🏆 D-GOL UEFA EUROPEAS v2.0 🏆</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-weight: 500;'>Dixon-Coles + Time Decay + Forma + H2H</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>100% Gratis con Web Scraping Optimizado</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuración")
    
    if st.button("🔄 Actualizar Datos", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Caché limpiado")
        st.rerun()
    
    competicion = st.selectbox("🏆 Competición:", list(COMPETICIONES_UEFA.keys()))
    
    datos = scrape_fbref_matches(COMPETICIONES_UEFA[competicion], competicion)
    
    if datos:
        st.success(f"✅ {len(datos['equipos'])} equipos")
        st.info(f"📊 {datos['total_partidos']} partidos analizados")
        st.caption(f"📅 Actualizado: {datos['fecha_actualizacion'].strftime('%d/%m/%Y %H:%M')}")
        st.caption("♻️ Caché: 6 horas")
        
        local = st.selectbox("🏠 Local:", datos['equipos'])
        visitante = st.selectbox("✈️ Visitante:", [e for e in datos['equipos'] if e != local])
        analizar_btn = st.button("🔍 ANALIZAR PARTIDO", type="primary", use_container_width=True)
    else:
        st.error("❌ No se pudieron cargar datos")
        analizar_btn = False

# Main
if analizar_btn and datos:
    with st.spinner("🔍 Analizando con modelo avanzado..."):
        resultado = analizar_partido_uefa_avanzado(datos, local, visitante)
    
    if resultado:
        st.markdown(f"## 🏟️ {local} vs {visitante}")
        st.markdown(f"**{competicion}** • Modelo Dixon-Coles v2.0")
        
        # Mostrar ajustes
        ajustes = resultado['ajustes']
        st.caption(f"⚙️ Ajustes: Forma L:{ajustes['forma_local']:.2f} V:{ajustes['forma_visit']:.2f} | Calidad L:{ajustes['calidad_local']:.2f} V:{ajustes['calidad_visit']:.2f}")
        
        st.markdown("---")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⚽ Goles Total", f"{resultado['goles_esperados_total']:.2f}")
        with col2:
            st.metric("🏠 Local", f"{resultado['lambda_local']:.2f}")
        with col3:
            st.metric("✈️ Visitante", f"{resultado['lambda_visitante']:.2f}")
        with col4:
            st.metric("🎯 Resultado Probable", f"{resultado['resultado_mas_probable'][0]}-{resultado['resultado_mas_probable'][1]} ({resultado['prob_resultado_mp']:.1f}%)")
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Probabilidades", "⚽ Goles", "📈 Análisis Avanzado", "📉 Gráficos", "💾 Exportar"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Resultado 1X2")
                st.metric("🏠 Local", f"{resultado['prob_local']:.1f}%")
                st.info(f"Cuota: {resultado['cuota_local']:.2f}")
                st.metric("⚖️ Empate", f"{resultado['prob_empate']:.1f}%")
                st.info(f"Cuota: {resultado['cuota_empate']:.2f}")
                st.metric("✈️ Visitante", f"{resultado['prob_visitante']:.1f}%")
                st.info(f"Cuota: {resultado['cuota_visitante']:.2f}")
                
                st.markdown("---")
                st.subheader("⚽⚽ BTTS")
                st.metric("Sí", f"{resultado['prob_btts_si']:.1f}%")
                st.metric("No", f"{100 - resultado['prob_btts_si']:.1f}%")
            
            with col2:
                st.subheader("📈 Over/Under")
                st.metric("Over 1.5", f"{resultado['prob_over_15']:.1f}%")
                st.metric("Over 2.5", f"{resultado['prob_over_25']:.1f}%")
                st.metric("Over 3.5", f"{resultado['prob_over_35']:.1f}%")
                
                st.markdown("---")
                st.subheader("🤝 Doble Oportunidad")
                st.metric("1X", f"{resultado['prob_1x']:.1f}%")
                st.metric("X2", f"{resultado['prob_x2']:.1f}%")
                st.metric("12", f"{resultado['prob_12']:.1f}%")
        
        with tab2:
            st.subheader("⚽ GOLES POR EQUIPO")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏠 {local}")
                st.metric("Over 0.5", f"{resultado['local_over_05']:.1f}%")
                st.metric("Over 1.5", f"{resultado['local_over_15']:.1f}%")
                st.metric("Over 2.5", f"{resultado['local_over_25']:.1f}%")
            
            with col2:
                st.markdown(f"### ✈️ {visitante}")
                st.metric("Over 0.5", f"{resultado['visit_over_05']:.1f}%")
                st.metric("Over 1.5", f"{resultado['visit_over_15']:.1f}%")
                st.metric("Over 2.5", f"{resultado['visit_over_25']:.1f}%")
        
        with tab3:
            st.subheader("🔬 Análisis Avanzado")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏠 {local}")
                st.metric("Fuerza Total (modelo)", f"{resultado['fuerza_local']:.3f}")
                st.metric("Forma reciente (puntos)", f"{resultado['stats_local'].get('forma_local_ptos', 0):.2f}/3")
                st.metric("Forma goles favor", f"{resultado['stats_local'].get('forma_local_gf', 0):.2f}")
                st.metric("Forma goles contra", f"{resultado['stats_local'].get('forma_local_gc', 0):.2f}")
            
            with col2:
                st.markdown(f"### ✈️ {visitante}")
                st.metric("Fuerza Total (modelo)", f"{resultado['fuerza_visit']:.3f}")
                st.metric("Forma reciente (puntos)", f"{resultado['stats_visit'].get('forma_visit_ptos', 0):.2f}/3")
                st.metric("Forma goles favor", f"{resultado['stats_visit'].get('forma_visit_gf', 0):.2f}")
                st.metric("Forma goles contra", f"{resultado['stats_visit'].get('forma_visit_gc', 0):.2f}")
            
            st.markdown("---")
            st.subheader("🤼 Head to Head")
            h2h = resultado['h2h']
            if h2h['partidos'] > 0:
                st.info(f"📊 {h2h['partidos']} partidos | {local}: {h2h['goles_eq1']:.1f} goles/partido | {visitante}: {h2h['goles_eq2']:.1f} goles/partido | Victorias {local}: {h2h['victorias_eq1']}")
            else:
                st.warning("⚠️ No hay historial directo")
        
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
            
            # Matriz
            st.subheader("🎯 Matriz de Resultados Exactos")
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
            reporte = f"""
D-GOL UEFA v2.0 - ANÁLISIS COMPLETO
{"="*75}

Partido: {local} vs {visitante}
Competición: {competicion}
Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Modelo: Dixon-Coles v2.0 (Time Decay + Forma + H2H)

{"="*75}

PREDICCIÓN PRINCIPAL:
- Resultado más probable: {resultado['resultado_mas_probable'][0]}-{resultado['resultado_mas_probable'][1]} ({resultado['prob_resultado_mp']:.1f}%)
- Goles esperados: {resultado['goles_esperados_total']:.2f} (Local: {resultado['lambda_local']:.2f}, Visit: {resultado['lambda_visitante']:.2f})

1X2:
- Local: {resultado['prob_local']:.1f}% (Cuota: {resultado['cuota_local']:.2f})
- Empate: {resultado['prob_empate']:.1f}% (Cuota: {resultado['cuota_empate']:.2f})
- Visitante: {resultado['prob_visitante']:.1f}% (Cuota: {resultado['cuota_visitante']:.2f})

OVER/UNDER:
- Over 1.5: {resultado['prob_over_15']:.1f}%
- Over 2.5: {resultado['prob_over_25']:.1f}%
- Over 3.5: {resultado['prob_over_35']:.1f}%

BTTS: {resultado['prob_btts_si']:.1f}%

ANÁLISIS AVANZADO:
- Fuerza {local}: {resultado['fuerza_local']:.3f}
- Fuerza {visitante}: {resultado['fuerza_visit']:.3f}
- Parámetro rho: {resultado['rho']:.3f}

{"="*75}
Generado por D-GOL UEFA v2.0 - Modelo Avanzado
            """
            
            st.download_button(
                "💾 Descargar Análisis",
                reporte,
                f"dgol_uefa_v2_{local}_vs_{visitante}.txt",
                "text/plain",
                use_container_width=True
            )
        
        # Recomendaciones
        st.markdown("---")
        st.subheader("💡 Recomendaciones Inteligentes")
        
        if resultado['prob_over_25'] > 70:
            st.success(f"✅ **Over 2.5 Total** - {resultado['prob_over_25']:.1f}% probabilidad")
        
        if resultado['prob_btts_si'] > 70:
            st.success(f"✅ **BTTS Sí** - {resultado['prob_btts_si']:.1f}% probabilidad")
        
        if resultado['prob_resultado_mp'] > 15:
            st.success(f"✅ **Resultado {resultado['resultado_mas_probable'][0]}-{resultado['resultado_mas_probable'][1]}** - {resultado['prob_resultado_mp']:.1f}%")
        
        if resultado['fuerza_local'] > resultado['fuerza_visit'] + 0.5 and resultado['stats_local'].get('forma_local_ptos', 0) > 2.0:
            st.success(f"✅ **Victoria {local}** - Más fuerte y en buena forma")
    
    else:
        st.error("❌ No se pudo analizar el partido")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>🏆 D-GOL UEFA v2.0 | Dixon-Coles + ML</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'>Datos: FBref.com (Web Scraping) | Actualización cada 6 horas</p>", unsafe_allow_html=True)
